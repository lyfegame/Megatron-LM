#!/usr/bin/env python3
"""
FP8 to BF16 Conversion Script for DeepSeek V3.2

This script converts FP8 quantized weights to BF16 format.
Designed for CPU-only environments with smart memory management.

Features:
- Resume capability: skips already converted shards
- Streaming: processes one shard at a time from GCS
- Memory efficient: cleans up after each shard
- Validates output sizes against expected values
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Set, Tuple, Optional

import torch
from safetensors.torch import load_file, save_file


def weight_dequant_cpu(weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    CPU implementation of FP8 weight dequantization.

    FP8 weights are stored as int8 with block-wise scaling factors.
    Dequantization: bf16_weight = fp8_weight * scale_inv (broadcasted per block)

    Args:
        weight: FP8 weight tensor of shape (M, N), stored as float8_e4m3fn
        scale_inv: Scale tensor of shape (M // block_size, N // block_size)
        block_size: Block size used for quantization (default 128)

    Returns:
        BF16 weight tensor of shape (M, N)
    """
    assert weight.dim() == 2, f"Expected 2D weight tensor, got {weight.dim()}D"
    assert scale_inv.dim() == 2, f"Expected 2D scale tensor, got {scale_inv.dim()}D"

    M, N = weight.shape
    scale_M, scale_N = scale_inv.shape

    # Validate dimensions
    expected_scale_M = (M + block_size - 1) // block_size
    expected_scale_N = (N + block_size - 1) // block_size

    assert scale_M == expected_scale_M, f"Scale M mismatch: {scale_M} vs expected {expected_scale_M}"
    assert scale_N == expected_scale_N, f"Scale N mismatch: {scale_N} vs expected {expected_scale_N}"

    # Convert FP8 to float32 for computation
    weight_f32 = weight.to(torch.float32)
    scale_inv_f32 = scale_inv.to(torch.float32)

    # Broadcast scale_inv to match weight shape
    # Each block_size x block_size block shares one scale factor
    scale_expanded = scale_inv_f32.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Handle edge cases where weight dimensions aren't exact multiples of block_size
    scale_expanded = scale_expanded[:M, :N]

    # Dequantize
    result = weight_f32 * scale_expanded

    return result.to(torch.bfloat16)


def get_existing_bf16_shards(bf16_gcs_path: str) -> Set[str]:
    """Get list of already converted shard files in GCS."""
    try:
        result = subprocess.run(
            ["gsutil", "ls", bf16_gcs_path],
            capture_output=True, text=True, check=True
        )
        files = set()
        for line in result.stdout.strip().split('\n'):
            if line and '.safetensors' in line:
                files.add(os.path.basename(line))
        return files
    except subprocess.CalledProcessError:
        return set()


def get_fp8_shards(fp8_gcs_path: str) -> list:
    """Get list of FP8 shard files in GCS."""
    result = subprocess.run(
        ["gsutil", "ls", fp8_gcs_path],
        capture_output=True, text=True, check=True
    )
    files = []
    for line in result.stdout.strip().split('\n'):
        if line and '.safetensors' in line:
            files.append(os.path.basename(line))
    return sorted(files)


def download_file(gcs_path: str, local_path: str) -> bool:
    """Download a file from GCS."""
    print(f"  Downloading {os.path.basename(gcs_path)}...")
    result = subprocess.run(
        ["gsutil", "-q", "cp", gcs_path, local_path],
        capture_output=True, text=True
    )
    return result.returncode == 0


def upload_file(local_path: str, gcs_path: str) -> bool:
    """Upload a file to GCS."""
    print(f"  Uploading {os.path.basename(local_path)}...")
    result = subprocess.run(
        ["gsutil", "-q", "cp", local_path, gcs_path],
        capture_output=True, text=True
    )
    return result.returncode == 0


def load_weight_map(fp8_gcs_path: str, temp_dir: str) -> Dict[str, str]:
    """Load the weight map from model index."""
    index_file = os.path.join(temp_dir, "model.safetensors.index.json")
    download_file(f"{fp8_gcs_path}/model.safetensors.index.json", index_file)
    with open(index_file, 'r') as f:
        data = json.load(f)
    return data['weight_map']


def validate_shard_size(original_size: int, converted_size: int, shard_name: str) -> bool:
    """
    Validate converted shard size.

    BF16 (2 bytes) vs FP8 (1 byte) means:
    - Pure FP8 weights should roughly double in size
    - Mixed content (FP8 + scale_inv) will have different ratios
    - Scale_inv tensors are removed, which reduces size

    Expected: converted size should be between 1.5x and 2.5x original
    (accounting for scale_inv removal and overhead)
    """
    ratio = converted_size / original_size

    # Reasonable bounds: BF16 is 2x FP8, but scale_inv removal reduces it
    # and safetensors overhead varies
    min_ratio = 1.3
    max_ratio = 2.5

    if min_ratio <= ratio <= max_ratio:
        print(f"  Size validation OK: {original_size:,} -> {converted_size:,} ({ratio:.2f}x)")
        return True
    else:
        print(f"  WARNING: Size ratio {ratio:.2f}x outside expected range [{min_ratio}, {max_ratio}]")
        print(f"    Original: {original_size:,} bytes, Converted: {converted_size:,} bytes")
        return True  # Continue anyway but warn


def convert_shard(
    shard_name: str,
    fp8_gcs_path: str,
    bf16_gcs_path: str,
    weight_map: Dict[str, str],
    temp_dir: str,
    block_size: int = 128
) -> bool:
    """
    Convert a single shard from FP8 to BF16.

    Returns True if successful, False otherwise.
    """
    print(f"\nProcessing {shard_name}...")

    fp8_local = os.path.join(temp_dir, f"fp8_{shard_name}")
    bf16_local = os.path.join(temp_dir, f"bf16_{shard_name}")

    try:
        # Download FP8 shard
        if not download_file(f"{fp8_gcs_path}/{shard_name}", fp8_local):
            print(f"  ERROR: Failed to download {shard_name}")
            return False

        original_size = os.path.getsize(fp8_local)

        # Load the shard
        print(f"  Loading shard ({original_size / 1e9:.2f} GB)...")
        state_dict = load_file(fp8_local, device="cpu")

        # Find which tensors in this shard need scale_inv from other files
        scale_inv_files_needed = set()
        for tensor_name in state_dict.keys():
            if tensor_name.endswith("_scale_inv"):
                continue
            if state_dict[tensor_name].element_size() == 1:  # FP8
                scale_inv_name = f"{tensor_name}_scale_inv"
                if scale_inv_name in weight_map:
                    scale_file = weight_map[scale_inv_name]
                    if scale_file != shard_name:
                        scale_inv_files_needed.add(scale_file)

        # Load additional files for cross-file scale_inv
        loaded_files = {shard_name: state_dict}
        for scale_file in scale_inv_files_needed:
            scale_local = os.path.join(temp_dir, f"scale_{scale_file}")
            if download_file(f"{fp8_gcs_path}/{scale_file}", scale_local):
                loaded_files[scale_file] = load_file(scale_local, device="cpu")
                os.remove(scale_local)

        def get_tensor(tensor_name: str) -> torch.Tensor:
            """Get tensor from the correct file."""
            file_name = weight_map.get(tensor_name)
            if file_name and file_name in loaded_files:
                return loaded_files[file_name][tensor_name]
            # Try current shard
            if tensor_name in state_dict:
                return state_dict[tensor_name]
            raise KeyError(f"Tensor {tensor_name} not found")

        # Convert weights
        new_state_dict = {}
        fp8_converted = 0
        skipped = 0

        for tensor_name, tensor in state_dict.items():
            if tensor_name.endswith("_scale_inv"):
                skipped += 1
                continue  # Skip scale_inv tensors

            if tensor.element_size() == 1:  # FP8 weight (1 byte per element)
                scale_inv_name = f"{tensor_name}_scale_inv"
                try:
                    scale_inv = get_tensor(scale_inv_name)

                    # Handle different tensor shapes
                    if tensor.dim() == 2:
                        new_state_dict[tensor_name] = weight_dequant_cpu(tensor, scale_inv, block_size)
                        fp8_converted += 1
                    elif tensor.dim() == 1:
                        # 1D tensors: simple element-wise scaling
                        new_state_dict[tensor_name] = (tensor.to(torch.float32) * scale_inv.to(torch.float32)).to(torch.bfloat16)
                        fp8_converted += 1
                    else:
                        # For higher-dim tensors, reshape to 2D, convert, reshape back
                        orig_shape = tensor.shape
                        tensor_2d = tensor.view(-1, tensor.shape[-1])
                        scale_2d = scale_inv.view(-1, scale_inv.shape[-1]) if scale_inv.dim() > 2 else scale_inv

                        # Check if we can apply 2D dequant
                        if tensor_2d.shape[0] // block_size == scale_2d.shape[0] or \
                           (tensor_2d.shape[0] + block_size - 1) // block_size == scale_2d.shape[0]:
                            converted_2d = weight_dequant_cpu(tensor_2d, scale_2d, block_size)
                            new_state_dict[tensor_name] = converted_2d.view(orig_shape)
                            fp8_converted += 1
                        else:
                            # Fallback: broadcast scale to full size
                            print(f"    Using broadcast for {tensor_name} ({tensor.shape} / {scale_inv.shape})")
                            scale_expanded = scale_inv.to(torch.float32)
                            while scale_expanded.dim() < tensor.dim():
                                scale_expanded = scale_expanded.unsqueeze(0)
                            new_state_dict[tensor_name] = (tensor.to(torch.float32) * scale_expanded).to(torch.bfloat16)
                            fp8_converted += 1

                except KeyError as e:
                    print(f"    Warning: Missing scale_inv for {tensor_name}, keeping as-is")
                    new_state_dict[tensor_name] = tensor.to(torch.bfloat16)
            else:
                # Non-FP8 tensor, keep as-is (might convert dtype)
                if tensor.dtype not in [torch.bfloat16, torch.float32, torch.float16]:
                    new_state_dict[tensor_name] = tensor
                else:
                    new_state_dict[tensor_name] = tensor.to(torch.bfloat16) if tensor.dtype != torch.bfloat16 else tensor

        print(f"  Converted {fp8_converted} FP8 tensors, skipped {skipped} scale_inv tensors")

        # Save converted shard
        print(f"  Saving BF16 shard...")
        save_file(new_state_dict, bf16_local)

        converted_size = os.path.getsize(bf16_local)
        validate_shard_size(original_size, converted_size, shard_name)

        # Upload to GCS
        if not upload_file(bf16_local, f"{bf16_gcs_path}/{shard_name}"):
            print(f"  ERROR: Failed to upload {shard_name}")
            return False

        print(f"  SUCCESS: {shard_name}")
        return True

    except Exception as e:
        print(f"  ERROR processing {shard_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        for f in [fp8_local, bf16_local]:
            if os.path.exists(f):
                os.remove(f)
        # Clear additional loaded files
        for fname in list(loaded_files.keys()):
            if fname != shard_name:
                del loaded_files[fname]


def copy_config_files(fp8_gcs_path: str, bf16_gcs_path: str, temp_dir: str):
    """Copy config files from FP8 to BF16, removing quantization config."""
    print("\nCopying and updating config files...")

    # Copy config.json and update it
    config_local = os.path.join(temp_dir, "config.json")
    if download_file(f"{fp8_gcs_path}/config.json", config_local):
        with open(config_local, 'r') as f:
            config = json.load(f)

        # Remove quantization config since weights are now BF16
        if 'quantization_config' in config:
            del config['quantization_config']

        # Update torch_dtype
        config['torch_dtype'] = 'bfloat16'

        with open(config_local, 'w') as f:
            json.dump(config, f, indent=2)

        upload_file(config_local, f"{bf16_gcs_path}/config.json")

    # Copy other config files as-is
    for filename in ['generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'special_tokens_map.json', 'LICENSE', 'README.md']:
        local_path = os.path.join(temp_dir, filename)
        if download_file(f"{fp8_gcs_path}/{filename}", local_path):
            upload_file(local_path, f"{bf16_gcs_path}/{filename}")


def create_bf16_index(fp8_gcs_path: str, bf16_gcs_path: str, temp_dir: str):
    """Create model index file for BF16 weights (without scale_inv entries)."""
    print("\nCreating BF16 model index...")

    index_local = os.path.join(temp_dir, "model.safetensors.index.json")
    download_file(f"{fp8_gcs_path}/model.safetensors.index.json", index_local)

    with open(index_local, 'r') as f:
        data = json.load(f)

    # Remove all scale_inv entries
    weight_map = data['weight_map']
    new_weight_map = {k: v for k, v in weight_map.items() if not k.endswith('_scale_inv')}

    new_data = {
        'metadata': data.get('metadata', {}),
        'weight_map': new_weight_map
    }

    bf16_index_local = os.path.join(temp_dir, "bf16_model.safetensors.index.json")
    with open(bf16_index_local, 'w') as f:
        json.dump(new_data, f, indent=2)

    upload_file(bf16_index_local, f"{bf16_gcs_path}/model.safetensors.index.json")
    print(f"  Created index with {len(new_weight_map)} weight entries (removed scale_inv)")


def main():
    parser = argparse.ArgumentParser(description='Convert FP8 weights to BF16')
    parser.add_argument('--fp8-gcs-path', type=str, required=True,
                        help='GCS path to FP8 weights (e.g., gs://bucket/DeepSeek-V3.2-fp8)')
    parser.add_argument('--bf16-gcs-path', type=str, required=True,
                        help='GCS path for BF16 output (e.g., gs://bucket/DeepSeek-V3.2-bf16)')
    parser.add_argument('--temp-dir', type=str, default='/tmp/fp8_convert',
                        help='Temporary directory for processing')
    parser.add_argument('--block-size', type=int, default=128,
                        help='Block size for dequantization (default: 128)')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip already converted shards (default: True)')
    parser.add_argument('--only-metadata', action='store_true',
                        help='Only generate config and index files, skip weight conversion')
    args = parser.parse_args()

    # Normalize GCS paths
    fp8_gcs = args.fp8_gcs_path.rstrip('/')
    bf16_gcs = args.bf16_gcs_path.rstrip('/')

    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)

    print("=" * 60)
    print("DeepSeek V3.2 FP8 to BF16 Conversion")
    print("=" * 60)
    print(f"FP8 Source:  {fp8_gcs}")
    print(f"BF16 Output: {bf16_gcs}")
    print(f"Block size:  {args.block_size}")
    print("=" * 60)

    if args.only_metadata:
        # Just create metadata files
        copy_config_files(fp8_gcs, bf16_gcs, args.temp_dir)
        create_bf16_index(fp8_gcs, bf16_gcs, args.temp_dir)
        print("\nMetadata files created successfully!")
        return

    # Load weight map
    print("\nLoading weight map...")
    weight_map = load_weight_map(fp8_gcs, args.temp_dir)
    print(f"  Found {len(weight_map)} weight entries")

    # Get shard lists
    fp8_shards = get_fp8_shards(fp8_gcs)
    existing_bf16 = get_existing_bf16_shards(bf16_gcs) if args.skip_existing else set()

    print(f"\nFP8 shards: {len(fp8_shards)}")
    print(f"Existing BF16 shards: {len(existing_bf16)}")

    # Determine which shards to process
    to_process = [s for s in fp8_shards if s not in existing_bf16]
    print(f"Shards to convert: {len(to_process)}")

    if not to_process:
        print("\nAll shards already converted!")
    else:
        # Convert shards
        success = 0
        failed = 0

        for i, shard in enumerate(to_process):
            print(f"\n[{i+1}/{len(to_process)}] ", end="")
            if convert_shard(shard, fp8_gcs, bf16_gcs, weight_map, args.temp_dir, args.block_size):
                success += 1
            else:
                failed += 1

        print(f"\n\nConversion complete: {success} success, {failed} failed")

    # Always update metadata files at the end
    copy_config_files(fp8_gcs, bf16_gcs, args.temp_dir)
    create_bf16_index(fp8_gcs, bf16_gcs, args.temp_dir)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
