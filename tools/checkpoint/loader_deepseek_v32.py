# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
DeepSeek V3.2 Checkpoint Loader for Megatron-LM.

This script converts HuggingFace DeepSeek V3.2 checkpoints to Megatron format.

Key features:
- Handles MLA (Multi-Latent Attention) weight mapping
- Handles Lightning Indexer weight mapping for V3.2
- Supports tensor parallel sharding for indexer weights
- Handles MoE expert weights

Usage:
    python tools/checkpoint/convert.py \
        --model-type deepseek_v32 \
        --loader deepseek_v32 \
        --load-dir /path/to/hf/checkpoint \
        --save-dir /path/to/megatron/checkpoint \
        --target-tensor-parallel-size 8 \
        --target-pipeline-parallel-size 1
"""

import json
import os
import sys
from typing import Dict, Optional

import torch
from tqdm import tqdm

try:
    import safetensors.torch as safetensors
    HAVE_SAFETENSORS = True
except ImportError:
    HAVE_SAFETENSORS = False


def add_arguments(parser):
    """Add DeepSeek V3.2 specific arguments."""
    group = parser.add_argument_group(title='DeepSeek V3.2 loader')
    group.add_argument(
        '--bf16',
        action='store_true',
        help='Whether to load weights in bf16.'
    )
    group.add_argument(
        '--fp16',
        action='store_true',
        help='Whether to load weights in fp16.'
    )
    group.add_argument(
        '--megatron-path',
        type=str,
        default=None,
        help='Base directory of Megatron repository'
    )
    group.add_argument(
        '--loader-transformer-impl',
        default='transformer_engine',
        choices=['local', 'transformer_engine'],
        help='Which Transformer implementation to use.'
    )
    group.add_argument(
        '--with-indexer',
        action='store_true',
        default=True,
        help='Load Lightning Indexer weights (V3.2 only).'
    )


def load_hf_config(model_path: str) -> dict:
    """Load HuggingFace config.json."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path, "r") as f:
        return json.load(f)


def load_hf_weights(model_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load HuggingFace weights from safetensors or pytorch files.
    """
    weights = {}

    # Try safetensors first
    safetensor_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    if safetensor_files and HAVE_SAFETENSORS:
        print(f"Loading from safetensors ({len(safetensor_files)} files)...")
        for filename in tqdm(safetensor_files):
            filepath = os.path.join(model_path, filename)
            file_weights = safetensors.load_file(filepath, device=device)
            weights.update(file_weights)
    else:
        # Fall back to pytorch bin files
        bin_files = [f for f in os.listdir(model_path) if f.endswith('.bin')]
        print(f"Loading from pytorch bin ({len(bin_files)} files)...")
        for filename in tqdm(bin_files):
            filepath = os.path.join(model_path, filename)
            file_weights = torch.load(filepath, map_location=device)
            weights.update(file_weights)

    return weights


# Weight name mappings from HuggingFace to Megatron

# MLA weight mappings (shared with V3)
MLA_WEIGHT_MAP = {
    # Q path
    "self_attn.q_a_proj.weight": "self_attention.linear_q_down_proj.weight",
    "self_attn.q_a_layernorm.weight": "self_attention.q_layernorm.weight",
    "self_attn.q_b_proj.weight": "self_attention.linear_q_up_proj.weight",
    # KV path
    "self_attn.kv_a_proj_with_mqa.weight": "self_attention.linear_kv_down_proj.weight",
    "self_attn.kv_a_layernorm.weight": "self_attention.kv_layernorm.weight",
    "self_attn.kv_b_proj.weight": "self_attention.linear_kv_up_proj.weight",
    # Output
    "self_attn.o_proj.weight": "self_attention.linear_proj.weight",
}

# V3.2 Lightning Indexer weight mappings
INDEXER_WEIGHT_MAP = {
    "self_attn.indexer.wq_b.weight": "self_attention.lightning_indexer.linear_wq_b.weight",
    "self_attn.indexer.wk.weight": "self_attention.lightning_indexer.linear_wk.weight",
    "self_attn.indexer.k_norm.weight": "self_attention.lightning_indexer.k_layernorm.weight",
    "self_attn.indexer.k_norm.bias": "self_attention.lightning_indexer.k_layernorm.bias",
    "self_attn.indexer.weights_proj.weight": "self_attention.lightning_indexer.linear_weights_proj.weight",
}

# MoE weight mappings
MOE_WEIGHT_MAP = {
    "mlp.gate.weight": "mlp.router.weight",
    "mlp.shared_experts.gate_proj.weight": "mlp.shared_experts.linear_fc1.weight",  # First half
    "mlp.shared_experts.up_proj.weight": "mlp.shared_experts.linear_fc1.weight",    # Second half
    "mlp.shared_experts.down_proj.weight": "mlp.shared_experts.linear_fc2.weight",
}


def shard_weight_for_tp(
    weight: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    dim: int = 0,
    replicate: bool = False,
) -> torch.Tensor:
    """
    Shard a weight tensor for tensor parallelism.

    Args:
        weight: The weight tensor to shard
        tp_size: Tensor parallel world size
        tp_rank: Current tensor parallel rank
        dim: Dimension to split along (0 for column-parallel, 1 for row-parallel)
        replicate: If True, return the full tensor (for replicated weights)

    Returns:
        Sharded weight tensor for the given rank
    """
    if replicate or tp_size == 1:
        return weight

    # Split along the specified dimension
    chunks = weight.chunk(tp_size, dim=dim)
    return chunks[tp_rank].clone()


def convert_mla_weights(
    hf_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    tp_size: int,
    tp_rank: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert MLA weights for a single layer.

    Args:
        hf_weights: HuggingFace weight dict
        layer_idx: Layer index
        tp_size: Tensor parallel size
        tp_rank: Current TP rank

    Returns:
        Dict of Megatron weight names to tensors
    """
    megatron_weights = {}
    prefix = f"model.layers.{layer_idx}."

    for hf_name, meg_name in MLA_WEIGHT_MAP.items():
        full_hf_name = prefix + hf_name
        if full_hf_name not in hf_weights:
            continue

        weight = hf_weights[full_hf_name]

        # Apply TP sharding based on layer type
        if "q_down_proj" in meg_name or "kv_down_proj" in meg_name:
            # These are replicated (gather output in forward)
            sharded_weight = weight
        elif "q_up_proj" in meg_name or "kv_up_proj" in meg_name:
            # Column parallel: split output dim
            sharded_weight = shard_weight_for_tp(weight, tp_size, tp_rank, dim=0)
        elif "linear_proj" in meg_name:
            # Row parallel: split input dim
            sharded_weight = shard_weight_for_tp(weight, tp_size, tp_rank, dim=1)
        else:
            # Layernorms are replicated
            sharded_weight = weight

        megatron_weights[meg_name] = sharded_weight

    return megatron_weights


def convert_indexer_weights(
    hf_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    tp_size: int,
    tp_rank: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert Lightning Indexer weights for a single layer (V3.2 only).

    Args:
        hf_weights: HuggingFace weight dict
        layer_idx: Layer index
        tp_size: Tensor parallel size
        tp_rank: Current TP rank

    Returns:
        Dict of Megatron weight names to tensors
    """
    megatron_weights = {}
    prefix = f"model.layers.{layer_idx}."

    for hf_name, meg_name in INDEXER_WEIGHT_MAP.items():
        full_hf_name = prefix + hf_name
        if full_hf_name not in hf_weights:
            continue

        weight = hf_weights[full_hf_name]

        # Apply TP sharding based on layer type
        if "wq_b" in meg_name:
            # ColumnParallel: split output (heads) dimension
            sharded_weight = shard_weight_for_tp(weight, tp_size, tp_rank, dim=0)
        elif "wk" in meg_name or "k_layernorm" in meg_name:
            # REPLICATED: wk and k_norm are shared by all heads
            sharded_weight = weight
        elif "weights_proj" in meg_name:
            # ColumnParallel: split output (heads) dimension
            sharded_weight = shard_weight_for_tp(weight, tp_size, tp_rank, dim=0)
        else:
            sharded_weight = weight

        megatron_weights[meg_name] = sharded_weight

    return megatron_weights


def convert_moe_weights(
    hf_weights: Dict[str, torch.Tensor],
    layer_idx: int,
    num_experts: int,
    tp_size: int,
    tp_rank: int,
    ep_size: int = 1,
    ep_rank: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Convert MoE weights for a single layer.

    Args:
        hf_weights: HuggingFace weight dict
        layer_idx: Layer index
        num_experts: Total number of experts
        tp_size: Tensor parallel size
        tp_rank: Current TP rank
        ep_size: Expert parallel size
        ep_rank: Current EP rank

    Returns:
        Dict of Megatron weight names to tensors
    """
    megatron_weights = {}
    prefix = f"model.layers.{layer_idx}."

    # Router weight
    router_hf = prefix + "mlp.gate.weight"
    if router_hf in hf_weights:
        megatron_weights["mlp.router.weight"] = hf_weights[router_hf]

    # Shared experts (if present)
    for hf_suffix, meg_suffix in [
        ("mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.linear_fc1.weight"),
        ("mlp.shared_experts.up_proj.weight", "mlp.shared_experts.linear_fc1.weight"),
        ("mlp.shared_experts.down_proj.weight", "mlp.shared_experts.linear_fc2.weight"),
    ]:
        hf_name = prefix + hf_suffix
        if hf_name in hf_weights:
            weight = hf_weights[hf_name]
            # For gate+up, we need to handle them specially
            if "gate_proj" in hf_suffix:
                gate_weight = weight
            elif "up_proj" in hf_suffix:
                # Concatenate with gate for GLU
                if "gate_weight" in dir():
                    weight = torch.cat([gate_weight, weight], dim=0)
                    sharded_weight = shard_weight_for_tp(weight, tp_size, tp_rank, dim=0)
                    megatron_weights[meg_suffix] = sharded_weight
            else:
                # down_proj: row parallel
                sharded_weight = shard_weight_for_tp(weight, tp_size, tp_rank, dim=1)
                megatron_weights[meg_suffix] = sharded_weight

    # Per-expert weights
    experts_per_rank = num_experts // ep_size
    start_expert = ep_rank * experts_per_rank
    end_expert = start_expert + experts_per_rank

    for expert_idx in range(start_expert, end_expert):
        local_expert_idx = expert_idx - start_expert

        # gate_proj and up_proj -> linear_fc1 (concatenated for GLU)
        gate_hf = prefix + f"mlp.experts.{expert_idx}.gate_proj.weight"
        up_hf = prefix + f"mlp.experts.{expert_idx}.up_proj.weight"
        down_hf = prefix + f"mlp.experts.{expert_idx}.down_proj.weight"

        if gate_hf in hf_weights and up_hf in hf_weights:
            gate_weight = hf_weights[gate_hf]
            up_weight = hf_weights[up_hf]
            fc1_weight = torch.cat([gate_weight, up_weight], dim=0)
            fc1_sharded = shard_weight_for_tp(fc1_weight, tp_size, tp_rank, dim=0)
            megatron_weights[f"mlp.experts.local_experts.{local_expert_idx}.linear_fc1.weight"] = fc1_sharded

        if down_hf in hf_weights:
            down_weight = hf_weights[down_hf]
            fc2_sharded = shard_weight_for_tp(down_weight, tp_size, tp_rank, dim=1)
            megatron_weights[f"mlp.experts.local_experts.{local_expert_idx}.linear_fc2.weight"] = fc2_sharded

    return megatron_weights


def convert_embedding_weights(
    hf_weights: Dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> Dict[str, torch.Tensor]:
    """Convert embedding layer weights."""
    megatron_weights = {}

    if "model.embed_tokens.weight" in hf_weights:
        embed_weight = hf_weights["model.embed_tokens.weight"]
        # Vocab parallel: split vocab dimension
        sharded_weight = shard_weight_for_tp(embed_weight, tp_size, tp_rank, dim=0)
        megatron_weights["embedding.word_embeddings.weight"] = sharded_weight

    return megatron_weights


def convert_output_weights(
    hf_weights: Dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
) -> Dict[str, torch.Tensor]:
    """Convert output layer weights."""
    megatron_weights = {}

    if "lm_head.weight" in hf_weights:
        output_weight = hf_weights["lm_head.weight"]
        # Column parallel: split output dimension
        sharded_weight = shard_weight_for_tp(output_weight, tp_size, tp_rank, dim=0)
        megatron_weights["output_layer.weight"] = sharded_weight

    return megatron_weights


def convert_norm_weights(
    hf_weights: Dict[str, torch.Tensor],
    layer_idx: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Convert normalization layer weights (replicated, no sharding)."""
    megatron_weights = {}

    if layer_idx is not None:
        prefix = f"model.layers.{layer_idx}."
        # Input layernorm
        if prefix + "input_layernorm.weight" in hf_weights:
            megatron_weights["input_layernorm.weight"] = hf_weights[prefix + "input_layernorm.weight"]
        # Post-attention layernorm
        if prefix + "post_attention_layernorm.weight" in hf_weights:
            megatron_weights["pre_mlp_layernorm.weight"] = hf_weights[prefix + "post_attention_layernorm.weight"]
    else:
        # Final norm
        if "model.norm.weight" in hf_weights:
            megatron_weights["decoder.final_layernorm.weight"] = hf_weights["model.norm.weight"]

    return megatron_weights


def load_checkpoint(args, queue, tp_rank, pp_rank):
    """
    Main checkpoint loading function called by the converter.

    This follows the Megatron checkpoint loader interface.
    """
    # Setup path
    if args.megatron_path:
        sys.path.insert(0, args.megatron_path)

    # Load HF config and weights
    hf_config = load_hf_config(args.load_dir)
    hf_weights = load_hf_weights(args.load_dir)

    # Extract model configuration
    num_layers = hf_config.get("num_hidden_layers", 61)  # V3.2 default
    hidden_size = hf_config.get("hidden_size", 7168)
    num_attention_heads = hf_config.get("num_attention_heads", 128)
    num_experts = hf_config.get("n_routed_experts", 256)
    vocab_size = hf_config.get("vocab_size", 129280)

    # V3.2 specific config
    has_indexer = "index_n_heads" in hf_config or args.with_indexer

    tp_size = args.target_tensor_parallel_size
    pp_size = args.target_pipeline_parallel_size

    # Calculate layer distribution for PP
    layers_per_pp = num_layers // pp_size
    start_layer = pp_rank * layers_per_pp
    end_layer = start_layer + layers_per_pp
    if pp_rank == pp_size - 1:
        end_layer = num_layers  # Last rank gets remaining layers

    # Put results in queue
    def send_message(message):
        queue.put(message)

    # Send embedding (first PP rank only)
    if pp_rank == 0:
        embed_weights = convert_embedding_weights(hf_weights, tp_size, tp_rank)
        send_message(("embedding", embed_weights))

    # Send transformer layers
    for layer_idx in range(start_layer, end_layer):
        layer_weights = {}

        # MLA weights
        layer_weights.update(convert_mla_weights(hf_weights, layer_idx, tp_size, tp_rank))

        # Lightning Indexer weights (V3.2)
        if has_indexer:
            layer_weights.update(convert_indexer_weights(hf_weights, layer_idx, tp_size, tp_rank))

        # MoE weights
        layer_weights.update(convert_moe_weights(
            hf_weights, layer_idx, num_experts, tp_size, tp_rank
        ))

        # Norm weights
        layer_weights.update(convert_norm_weights(hf_weights, layer_idx))

        send_message((f"layer_{layer_idx}", layer_weights))

    # Send output (last PP rank only)
    if pp_rank == pp_size - 1:
        # Final norm
        final_norm_weights = convert_norm_weights(hf_weights)
        send_message(("final_norm", final_norm_weights))

        # Output layer
        output_weights = convert_output_weights(hf_weights, tp_size, tp_rank)
        send_message(("output", output_weights))

    # Signal completion
    send_message(("done", None))


if __name__ == "__main__":
    print("DeepSeek V3.2 Checkpoint Loader")
    print("Use with tools/checkpoint/convert.py --model-type deepseek_v32")
