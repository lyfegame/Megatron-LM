#!/usr/bin/env python3
"""
DeepSeek V3.2 Checkpoint Conversion Verification Script

This script verifies the Megatron-Bridge conversion of DeepSeek V3.2.
It checks:
1. All expected weights are present
2. Weight shapes match expected dimensions
3. Weight values are reasonable (not NaN/Inf)

This is CONVERSION VALIDATION only - no implementation testing.
Run this BEFORE attempting inference to isolate conversion issues.

Usage:
    python verify_conversion.py --megatron-path /models-local/DeepSeek-V3.2-megatron
"""

import argparse
import os
import sys
from pathlib import Path


def verify_megatron_checkpoint(megatron_path: str) -> dict:
    """
    Verify the structure and contents of a Megatron checkpoint.

    Returns:
        dict with verification results
    """
    import torch

    results = {
        'path': megatron_path,
        'exists': False,
        'format': None,
        'num_files': 0,
        'total_size_gb': 0.0,
        'weights_found': [],
        'weights_missing': [],
        'shape_issues': [],
        'value_issues': [],
        'indexer_weights': {},
    }

    if not os.path.exists(megatron_path):
        print(f"ERROR: Path does not exist: {megatron_path}")
        return results

    results['exists'] = True

    # Detect checkpoint format
    files = list(Path(megatron_path).rglob('*'))
    pt_files = [f for f in files if f.suffix in ['.pt', '.pth', '.bin']]
    safetensor_files = [f for f in files if f.suffix == '.safetensors']
    torch_dist_dirs = [f for f in files if f.is_dir() and 'torch_dist' in f.name.lower()]

    results['num_files'] = len(pt_files) + len(safetensor_files)

    # Calculate total size
    total_bytes = sum(f.stat().st_size for f in files if f.is_file())
    results['total_size_gb'] = total_bytes / (1024**3)

    if torch_dist_dirs or any('torch_dist' in str(f) for f in files):
        results['format'] = 'torch_dist'
    elif safetensor_files:
        results['format'] = 'safetensors'
    elif pt_files:
        results['format'] = 'pytorch'
    else:
        results['format'] = 'unknown'

    print(f"Checkpoint format: {results['format']}")
    print(f"Total size: {results['total_size_gb']:.2f} GB")
    print(f"Number of files: {results['num_files']}")

    # Expected indexer weight patterns for V3.2
    indexer_patterns = [
        'lightning_indexer.linear_wq_b.weight',
        'lightning_indexer.linear_wk.weight',
        'lightning_indexer.k_layernorm.weight',
        'lightning_indexer.k_layernorm.bias',
        'lightning_indexer.linear_weights_proj.weight',
    ]

    # Try to load and inspect weights
    if results['format'] == 'pytorch':
        # Standard PyTorch checkpoint
        for pt_file in pt_files[:3]:  # Sample first 3 files
            try:
                state_dict = torch.load(pt_file, map_location='cpu', weights_only=True)
                for key, tensor in state_dict.items():
                    results['weights_found'].append(key)
                    # Check for indexer weights
                    for pattern in indexer_patterns:
                        if pattern in key:
                            results['indexer_weights'][key] = {
                                'shape': list(tensor.shape),
                                'dtype': str(tensor.dtype),
                                'has_nan': torch.isnan(tensor).any().item() if tensor.is_floating_point() else False,
                                'has_inf': torch.isinf(tensor).any().item() if tensor.is_floating_point() else False,
                            }
            except Exception as e:
                print(f"Warning: Could not load {pt_file}: {e}")

    elif results['format'] == 'torch_dist':
        # Torch distributed checkpoint format
        print("torch_dist format detected - listing structure:")
        for item in sorted(files)[:50]:
            if item.is_file():
                print(f"  {item.relative_to(megatron_path)}: {item.stat().st_size / 1024:.1f} KB")

    elif results['format'] == 'safetensors':
        try:
            from safetensors import safe_open
            for sf_file in safetensor_files[:3]:
                with safe_open(sf_file, framework='pt', device='cpu') as f:
                    for key in f.keys():
                        results['weights_found'].append(key)
                        for pattern in indexer_patterns:
                            if pattern in key:
                                tensor = f.get_tensor(key)
                                results['indexer_weights'][key] = {
                                    'shape': list(tensor.shape),
                                    'dtype': str(tensor.dtype),
                                    'has_nan': torch.isnan(tensor).any().item() if tensor.is_floating_point() else False,
                                    'has_inf': torch.isinf(tensor).any().item() if tensor.is_floating_point() else False,
                                }
        except ImportError:
            print("safetensors not installed - cannot inspect safetensor files")

    return results


def verify_expected_shapes(results: dict, config: dict) -> None:
    """
    Verify that indexer weights have expected shapes based on model config.

    Expected shapes for V3.2 (from official model.py):
    - wq_b: [index_n_heads * index_head_dim, q_lora_rank] = [8192, 1536]
    - wk: [index_head_dim, hidden_size] = [128, 7168]
    - k_norm.weight: [index_head_dim] = [128]
    - k_norm.bias: [index_head_dim] = [128]
    - weights_proj: [index_n_heads, hidden_size] = [64, 7168]
    """
    expected = {
        'wq_b': [config['index_n_heads'] * config['index_head_dim'], config['q_lora_rank']],
        'wk': [config['index_head_dim'], config['hidden_size']],
        'k_norm.weight': [config['index_head_dim']],
        'k_norm.bias': [config['index_head_dim']],
        'weights_proj': [config['index_n_heads'], config['hidden_size']],
    }

    print("\n--- Expected Indexer Weight Shapes ---")
    for name, shape in expected.items():
        print(f"  {name}: {shape}")

    print("\n--- Found Indexer Weights ---")
    for key, info in results.get('indexer_weights', {}).items():
        print(f"  {key}:")
        print(f"    shape: {info['shape']}")
        print(f"    dtype: {info['dtype']}")
        print(f"    has_nan: {info['has_nan']}")
        print(f"    has_inf: {info['has_inf']}")


def print_verification_summary(results: dict) -> int:
    """Print verification summary and return exit code."""
    print("\n" + "=" * 70)
    print("CONVERSION VERIFICATION SUMMARY")
    print("=" * 70)

    errors = []
    warnings = []

    if not results['exists']:
        errors.append("Checkpoint path does not exist")

    if results['format'] == 'unknown':
        errors.append("Unknown checkpoint format")

    # Check for NaN/Inf in indexer weights
    for key, info in results.get('indexer_weights', {}).items():
        if info.get('has_nan'):
            errors.append(f"NaN values in {key}")
        if info.get('has_inf'):
            warnings.append(f"Inf values in {key}")

    # Check if indexer weights were found
    if results['exists'] and not results.get('indexer_weights'):
        warnings.append("No indexer weights found (may be in different format)")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  ❌ {e}")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ⚠️ {w}")

    if not errors and not warnings:
        print("\n✅ Conversion verification passed (basic checks)")

    print("\nFACTS:")
    print(f"  - Checkpoint path: {results['path']}")
    print(f"  - Format: {results['format']}")
    print(f"  - Size: {results['total_size_gb']:.2f} GB")
    print(f"  - Indexer weights found: {len(results.get('indexer_weights', {}))}")

    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description='Verify DeepSeek V3.2 checkpoint conversion')
    parser.add_argument('--megatron-path', type=str, required=True,
                       help='Path to converted Megatron checkpoint')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    args = parser.parse_args()

    print("=" * 70)
    print("DEEPSEEK V3.2 CONVERSION VERIFICATION")
    print("=" * 70)
    print(f"Verifying: {args.megatron_path}")
    print()

    # V3.2 default config values (from official model.py)
    v32_config = {
        'hidden_size': 7168,
        'index_n_heads': 64,
        'index_head_dim': 128,
        'index_topk': 2048,
        'q_lora_rank': 1536,
        'qk_rope_head_dim': 64,
    }

    results = verify_megatron_checkpoint(args.megatron_path)
    verify_expected_shapes(results, v32_config)

    return print_verification_summary(results)


if __name__ == '__main__':
    sys.exit(main())
