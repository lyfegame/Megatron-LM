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
DeepSeek V3.2 Bridge for Megatron-LM.

Extends the V3 bridge to support the Lightning Indexer for sparse attention.

V3.2 Architecture Changes from V3:
- Adds Lightning Indexer in each attention layer for sparse attention
- New weights: indexer.wq_b, indexer.wk, indexer.k_norm, indexer.weights_proj
- Enables O(L*K) attention instead of O(L^2) with K=2048

Usage:
    >>> from megatron.bridge import AutoBridge
    >>> bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3.2", trust_remote_code=True)
    >>> provider = bridge.to_megatron_provider()
"""

import logging
from dataclasses import dataclass
from typing import Mapping, Optional

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.deepseek.common import get_common_configs, get_common_mapping_list
from megatron.bridge.models.deepseek.deepseek_provider import DeepSeekV3ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


logger = logging.getLogger(__name__)

# FP8 block size used in DeepSeek V3.2 quantization
FP8_BLOCK_SIZE = 128


def _dequantize_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = FP8_BLOCK_SIZE,
    target_dtype: torch.dtype = torch.bfloat16,
    is_scale_inv: bool = False,
) -> torch.Tensor:
    """
    Dequantize FP8 (float8_e4m3fn) weight tensor using block-wise scales.

    This implements the dequantization formula from DeepSeek V3.2:
    - Weights are quantized with block_size x block_size blocks
    - Each block has an associated scale factor
    - Dequantization:
      - If scale: weight_bf16 = weight_fp8 * scale
      - If scale_inv: weight_bf16 = weight_fp8 / scale_inv

    Args:
        weight: FP8 weight tensor of shape (out_features, in_features)
        scale: Scale tensor of shape (out_features // block_size, in_features // block_size)
        block_size: Block size used in quantization (default 128)
        target_dtype: Target dtype for dequantized weights (default bfloat16)
        is_scale_inv: If True, scale is the inverse scale (divide instead of multiply)

    Returns:
        Dequantized weight tensor of shape (out_features, in_features) in target_dtype
    """
    if weight.dtype != torch.float8_e4m3fn:
        # Already dequantized or not FP8
        return weight.to(target_dtype)

    shape = weight.shape
    assert weight.dim() == 2, f"Expected 2D weight tensor, got {weight.dim()}D"

    # Reshape weight into blocks: [out_blocks, block_size, in_blocks, block_size]
    out_blocks = shape[0] // block_size
    in_blocks = shape[1] // block_size

    # Handle case where dimensions aren't perfectly divisible
    if shape[0] % block_size != 0 or shape[1] % block_size != 0:
        # Pad weight to be divisible by block_size
        pad_out = (block_size - shape[0] % block_size) % block_size
        pad_in = (block_size - shape[1] % block_size) % block_size
        if pad_out > 0 or pad_in > 0:
            weight = torch.nn.functional.pad(weight.float(), (0, pad_in, 0, pad_out))
            out_blocks = weight.shape[0] // block_size
            in_blocks = weight.shape[1] // block_size

    # Reshape: [out, in] -> [out_blocks, block_size, in_blocks, block_size]
    weight_blocked = weight.view(out_blocks, block_size, in_blocks, block_size)
    # Transpose to: [out_blocks, in_blocks, block_size, block_size]
    weight_blocked = weight_blocked.transpose(1, 2).contiguous()
    # Flatten blocks: [out_blocks * in_blocks, block_size * block_size]
    weight_flat = weight_blocked.view(-1, block_size * block_size)

    # Apply scale (scale is [out_blocks, in_blocks], flatten to [out_blocks * in_blocks])
    # For scale_inv, we divide; for regular scale, we multiply
    scale_flat = scale.view(-1, 1).float()
    if is_scale_inv:
        weight_dequant = (weight_flat.float() / scale_flat).to(target_dtype)
    else:
        weight_dequant = (weight_flat.float() * scale_flat).to(target_dtype)

    # Reshape back: [out_blocks, in_blocks, block_size, block_size]
    weight_dequant = weight_dequant.view(out_blocks, in_blocks, block_size, block_size)
    # Transpose back: [out_blocks, block_size, in_blocks, block_size]
    weight_dequant = weight_dequant.transpose(1, 2).contiguous()
    # Final reshape: [out, in]
    weight_dequant = weight_dequant.view(out_blocks * block_size, in_blocks * block_size)

    # Remove padding if added
    if weight_dequant.shape != shape:
        weight_dequant = weight_dequant[: shape[0], : shape[1]]

    return weight_dequant


@dataclass
class DeepSeekV32ModelProvider(DeepSeekV3ModelProvider):
    """
    DeepSeek-V3.2 Model Provider extending V3 with Lightning Indexer support.

    V3.2 adds sparse attention via Lightning Indexer which selects top-K tokens
    (default K=2048) for each query position.

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
    """

    # V3.2 Lightning Indexer parameters for sparse attention
    index_n_heads: Optional[int] = 64
    """Number of attention heads in the Lightning Indexer."""

    index_head_dim: Optional[int] = 128
    """Head dimension for the Lightning Indexer."""

    index_topk: Optional[int] = 2048
    """Number of top-K tokens to select for sparse attention."""


def get_v32_indexer_mapping_list() -> list:
    """
    Returns parameter mappings specific to V3.2 Lightning Indexer.

    The indexer has 5 weight tensors per layer:
    - wq_b: Projects compressed Q to indexer heads (replicated - indexer is not tensor-parallel)
    - wk: Projects hidden states to single key (replicated)
    - k_norm.weight: LayerNorm weight for K (replicated)
    - k_norm.bias: LayerNorm bias for K (replicated)
    - weights_proj: Per-head aggregation weights (replicated)

    Note: All indexer weights use ReplicatedMapping because the Lightning Indexer
    uses standard torch.nn.Linear (not Megatron parallel layers) and operates
    independently on each TP rank with the full sequence.
    """
    # All indexer mappings use ReplicatedMapping since they use torch.nn.Linear
    indexer_mappings = [
        # wq_b: q_lora_rank -> n_heads * head_dim
        ReplicatedMapping(
            megatron_param="decoder.layers.*.self_attention.lightning_indexer.linear_wq_b.weight",
            hf_param="model.layers.*.self_attn.indexer.wq_b.weight",
        ),
        # wk: hidden_size -> head_dim (replicated)
        ReplicatedMapping(
            megatron_param="decoder.layers.*.self_attention.lightning_indexer.linear_wk.weight",
            hf_param="model.layers.*.self_attn.indexer.wk.weight",
        ),
        # k_norm: LayerNorm for K (replicated)
        ReplicatedMapping(
            megatron_param="decoder.layers.*.self_attention.lightning_indexer.k_layernorm.weight",
            hf_param="model.layers.*.self_attn.indexer.k_norm.weight",
        ),
        ReplicatedMapping(
            megatron_param="decoder.layers.*.self_attention.lightning_indexer.k_layernorm.bias",
            hf_param="model.layers.*.self_attn.indexer.k_norm.bias",
        ),
        # weights_proj: per-head aggregation weights
        ReplicatedMapping(
            megatron_param="decoder.layers.*.self_attention.lightning_indexer.linear_weights_proj.weight",
            hf_param="model.layers.*.self_attn.indexer.weights_proj.weight",
        ),
    ]

    return indexer_mappings


@MegatronModelBridge.register_bridge(source="DeepseekV32ForCausalLM", target=GPTModel)
class DeepSeekV32Bridge(MegatronModelBridge):
    """
    Megatron Bridge for DeepSeek-V3.2.

    Extends V3 bridge with Lightning Indexer weight mappings for sparse attention.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("deepseek-ai/DeepSeek-V3.2", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()

    Note on FP8 Checkpoints:
        The official V3.2 checkpoint uses FP8 quantization for inference efficiency.
        For training/fine-tuning, weights will be dequantized to BF16 during loading.
        Set `dequantize_fp8=True` when loading to handle this automatically.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> DeepSeekV32ModelProvider:
        hf_config = hf_pretrained.config
        configs = get_common_configs(hf_pretrained)

        # Dtype handling
        configs["fp16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16
        configs["bf16"] = self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16
        configs["params_dtype"] = self.dtype_from_hf(hf_config, default=torch.float32)

        # V3 specific configs
        configs["make_vocab_size_divisible_by"] = 1280
        configs["moe_router_score_function"] = "sigmoid"
        configs["moe_router_enable_expert_bias"] = True
        if hasattr(hf_config, "aux_loss_alpha"):
            configs["moe_aux_loss_coeff"] = hf_config.aux_loss_alpha

        # V3.2 Lightning Indexer configs
        if hasattr(hf_config, "index_n_heads"):
            configs["index_n_heads"] = hf_config.index_n_heads
        if hasattr(hf_config, "index_head_dim"):
            configs["index_head_dim"] = hf_config.index_head_dim
        if hasattr(hf_config, "index_topk"):
            configs["index_topk"] = hf_config.index_topk

        provider = DeepSeekV32ModelProvider(**configs)
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Start with common V3 mappings
        mapping_list = get_common_mapping_list()

        # Add V3 expert bias mapping
        v3_mappings = {
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
        }
        for megatron_param, hf_param in v3_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add V3.2 Lightning Indexer mappings
        mapping_list.extend(get_v32_indexer_mapping_list())

        return MegatronMappingRegistry(*mapping_list)

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Load weights from HuggingFace state dict and dequantize FP8 weights if necessary.

        The official DeepSeek V3.2 checkpoint uses FP8 (float8_e4m3fn) quantization with
        block-wise scales. For training/fine-tuning, we need to dequantize these weights
        to BF16.

        FP8 checkpoint format:
        - Weight tensor: `model.layers.*.self_attn.*.weight` (dtype=float8_e4m3fn)
        - Scale tensor: `model.layers.*.self_attn.*.weight_scale` (dtype=float32)

        Args:
            hf_param: The parameter name or dictionary of parameter names to load.
            hf_state_dict: The HuggingFace state dictionary.

        Returns:
            The loaded (and potentially dequantized) weights.
        """
        if isinstance(hf_param, str):
            # Check if this is an FP8 weight with a corresponding scale
            hf_weights = hf_state_dict[hf_param]

            # If weight is FP8, look for scale tensor and dequantize
            if hf_weights.dtype == torch.float8_e4m3fn:
                # Try common scale naming conventions
                # Note: DeepSeek V3.2 uses weight_scale_inv (inverse scale)
                scale_keys = [
                    (hf_param + "_scale_inv", True),   # weight_scale_inv (DeepSeek V3.2)
                    (hf_param + "_scale", False),      # weight_scale
                    (hf_param.replace(".weight", ".scale_inv"), True),  # .scale_inv
                    (hf_param.replace(".weight", ".scale"), False),     # .scale
                    (hf_param.replace(".weight", "_scale"), False),     # _scale
                ]

                scale_tensor = None
                is_scale_inv = False
                for scale_key, is_inv in scale_keys:
                    if scale_key in hf_state_dict:
                        scale_tensor = hf_state_dict[scale_key]
                        is_scale_inv = is_inv
                        break

                if scale_tensor is not None:
                    logger.info(f"Dequantizing FP8 weight: {hf_param} (scale_inv={is_scale_inv})")
                    hf_weights = _dequantize_fp8_weight(
                        hf_weights,
                        scale_tensor,
                        block_size=FP8_BLOCK_SIZE,
                        target_dtype=torch.bfloat16,
                        is_scale_inv=is_scale_inv,
                    )
                else:
                    logger.warning(
                        f"FP8 weight {hf_param} found but no scale tensor. "
                        f"Tried: {[k for k, _ in scale_keys]}. Converting directly to bfloat16."
                    )
                    hf_weights = hf_weights.to(torch.bfloat16)

            return hf_weights
        else:
            # Dictionary case - handle each parameter
            result = {}
            for k, v in hf_param.items():
                # Recursively handle each parameter
                result[k] = self.maybe_modify_loaded_hf_weight(v, hf_state_dict)
            return result
