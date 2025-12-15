#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DeepSeek V3.2 Checkpoint Conversion Example

This script demonstrates how to convert DeepSeek V3.2 checkpoints from HuggingFace
format to Megatron format for training/fine-tuning.

DeepSeek V3.2 Features:
- 685B parameter MoE model
- Lightning Indexer for sparse attention (O(L*K) instead of O(L^2), K=2048)
- Multi-Latent Attention (MLA)
- FP8 quantized checkpoint (auto-dequantized to BF16 for training)

Usage:
    # Convert from local HuggingFace checkpoint (FP8)
    python examples/conversion/convert_deepseek_v32.py \\
        --hf-path /path/to/DeepSeek-V3.2-fp8 \\
        --megatron-path ./checkpoints/deepseek_v32

    # Convert with explicit output dtype
    python examples/conversion/convert_deepseek_v32.py \\
        --hf-path /path/to/DeepSeek-V3.2-fp8 \\
        --megatron-path ./checkpoints/deepseek_v32 \\
        --output-dtype bfloat16

Note:
    The FP8 checkpoint will be automatically dequantized to BF16 during conversion
    since FP8 is only supported for inference, not training.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def convert_deepseek_v32(
    hf_path: str,
    megatron_path: str,
    output_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> None:
    """
    Convert DeepSeek V3.2 checkpoint from HuggingFace to Megatron format.

    Args:
        hf_path: Path to the HuggingFace checkpoint (local or HF hub ID)
        megatron_path: Output directory for Megatron checkpoint
        output_dtype: Target dtype for weights (bfloat16 recommended for training)
        trust_remote_code: Whether to trust remote code from HuggingFace
    """
    from megatron.bridge import AutoBridge

    logger.info(f"Converting DeepSeek V3.2: {hf_path} -> {megatron_path}")
    logger.info(f"Output dtype: {output_dtype}")

    # Validate input path exists
    hf_path_obj = Path(hf_path)
    if not hf_path_obj.exists() and not hf_path.startswith("deepseek-ai/"):
        logger.warning(f"Local path does not exist: {hf_path}")
        logger.info("Attempting to load from HuggingFace Hub...")

    # Check for config.json to verify it's a valid checkpoint
    config_path = hf_path_obj / "config.json"
    if hf_path_obj.exists() and config_path.exists():
        import json

        with open(config_path) as f:
            config = json.load(f)

        # Verify this is a V3.2 checkpoint
        model_type = config.get("model_type", "")
        if "deepseek" not in model_type.lower():
            logger.warning(f"Unexpected model_type: {model_type}. Expected deepseek.")

        # Log V3.2 specific params
        if config.get("index_topk"):
            logger.info(f"Lightning Indexer config:")
            logger.info(f"  - index_n_heads: {config.get('index_n_heads', 'N/A')}")
            logger.info(f"  - index_head_dim: {config.get('index_head_dim', 'N/A')}")
            logger.info(f"  - index_topk: {config.get('index_topk', 'N/A')}")
        else:
            logger.warning("index_topk not found in config - may not be V3.2 checkpoint")

        # Check torch_dtype
        torch_dtype = config.get("torch_dtype", "")
        if "float8" in torch_dtype.lower():
            logger.info("Detected FP8 checkpoint - will auto-dequantize to BF16")

    # Prepare torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(output_dtype, torch.bfloat16)

    # Perform conversion using AutoBridge
    logger.info("Starting checkpoint conversion...")
    logger.info("This may take a while for large models like DeepSeek V3.2 (685B)")

    try:
        AutoBridge.import_ckpt(
            hf_model_id=hf_path,
            megatron_path=megatron_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        logger.info(f"Successfully converted checkpoint to: {megatron_path}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

    # Verify output
    output_path = Path(megatron_path)
    if output_path.exists():
        logger.info("Output checkpoint structure:")
        for item in sorted(output_path.iterdir()):
            if item.is_dir():
                logger.info(f"  [DIR] {item.name}/")
            else:
                size_mb = item.stat().st_size / (1024 * 1024)
                logger.info(f"  [FILE] {item.name} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek V3.2 checkpoint from HuggingFace to Megatron format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hf-path",
        required=True,
        help="Path to HuggingFace checkpoint (local path or HF hub ID)",
    )
    parser.add_argument(
        "--megatron-path",
        required=True,
        help="Output directory for Megatron checkpoint",
    )
    parser.add_argument(
        "--output-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Target dtype for weights (default: bfloat16)",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Don't trust remote code from HuggingFace (not recommended for DeepSeek)",
    )

    args = parser.parse_args()

    convert_deepseek_v32(
        hf_path=args.hf_path,
        megatron_path=args.megatron_path,
        output_dtype=args.output_dtype,
        trust_remote_code=not args.no_trust_remote_code,
    )

    # Clean up distributed if initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main() or 0)
