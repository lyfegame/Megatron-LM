# DeepSeek V3.2 Parallelism Choices for Megatron Conversion

This document explains the parallelism decisions made for converting DeepSeek V3.2 weights from HuggingFace format to Megatron format.

## Overview

DeepSeek V3.2 uses multiple parallelism strategies:
- **TP (Tensor Parallelism)**: Splits weight matrices across GPUs
- **EP (Expert Parallelism)**: Distributes MoE experts across GPUs
- **PP (Pipeline Parallelism)**: Splits layers across GPUs (not used in conversion)
- **SP (Sequence Parallelism)**: Runtime only, doesn't affect weights

## Component-by-Component Parallelism Decisions

### 1. Embeddings

| Component | Mapping Type | Reason |
|-----------|--------------|--------|
| `embed_tokens.weight` | `VocabParallelMapping` | Vocabulary split across TP ranks |

```python
# HF: [vocab_size, hidden_dim] = [129280, 7168]
# Megatron TP=8: [vocab_size/8, hidden_dim] = [16160, 7168] per rank
VocabParallelMapping(
    megatron_param="embedding.word_embeddings.weight",
    hf_param="model.embed_tokens.weight",
)
```

### 2. Multi-Latent Attention (MLA)

| Component | Mapping Type | Reason |
|-----------|--------------|--------|
| `q_a_proj` | `ColumnParallelMapping` | Projects hidden→q_lora_rank, split output |
| `q_b_proj` | `ColumnParallelMapping` | Projects q_lora_rank→heads, split heads |
| `kv_a_proj_with_mqa` | `ColumnParallelMapping` | KV compression, split output |
| `kv_b_proj` | `ColumnParallelMapping` | KV expansion, split heads |
| `o_proj` | `RowParallelMapping` | Output projection, split input |

```python
# Q projection chain: hidden → q_lora_rank → n_heads * head_dim
ColumnParallelMapping(
    megatron_param="decoder.layers.*.self_attention.linear_q_a.weight",
    hf_param="model.layers.*.self_attn.q_a_proj.weight",
)
ColumnParallelMapping(
    megatron_param="decoder.layers.*.self_attention.linear_q_b.weight",
    hf_param="model.layers.*.self_attn.q_b_proj.weight",
)

# Output projection (RowParallel because input is split across TP)
RowParallelMapping(
    megatron_param="decoder.layers.*.self_attention.linear_proj.weight",
    hf_param="model.layers.*.self_attn.o_proj.weight",
)
```

### 3. Lightning Indexer (Sparse Attention)

| Component | Mapping Type | Reason |
|-----------|--------------|--------|
| `indexer.wq_b` | **`ReplicatedMapping`** | Must produce identical top-K on all ranks |
| `indexer.wk` | **`ReplicatedMapping`** | Single key projection, replicated |
| `indexer.k_norm` | **`ReplicatedMapping`** | LayerNorm, replicated |
| `indexer.weights_proj` | **`ReplicatedMapping`** | Per-head weights, replicated |

**Why Replicated (NOT ColumnParallel)?**

From official `model.py`:
```python
class Indexer(torch.nn.Module):
    def __init__(self, args):
        # Uses regular Linear, NOT ColumnParallelLinear
        self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wk = Linear(self.dim, self.head_dim)
        self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)

    def forward(self, ...):
        # ... compute top-K indices ...

        # CRITICAL: All ranks must have same indices
        topk_indices_ = topk_indices.clone()
        dist.broadcast(topk_indices_, src=0)  # Verify consistency
        assert torch.all(topk_indices == topk_indices_)
        return topk_indices
```

**Reasoning:**
1. **Consistency requirement**: All TP ranks must select the **same** top-K tokens
2. **No communication overhead**: Each rank computes independently with full weights
3. **Broadcast is sanity check**: Not data gathering, just verification
4. **Small overhead**: Indexer is ~100M params vs 671B total (<0.02%)

```python
# All indexer weights use ReplicatedMapping
ReplicatedMapping(
    megatron_param="decoder.layers.*.self_attention.lightning_indexer.linear_wq_b.weight",
    hf_param="model.layers.*.self_attn.indexer.wq_b.weight",
)
ReplicatedMapping(
    megatron_param="decoder.layers.*.self_attention.lightning_indexer.linear_wk.weight",
    hf_param="model.layers.*.self_attn.indexer.wk.weight",
)
ReplicatedMapping(
    megatron_param="decoder.layers.*.self_attention.lightning_indexer.k_layernorm.weight",
    hf_param="model.layers.*.self_attn.indexer.k_norm.weight",
)
ReplicatedMapping(
    megatron_param="decoder.layers.*.self_attention.lightning_indexer.linear_weights_proj.weight",
    hf_param="model.layers.*.self_attn.indexer.weights_proj.weight",
)
```

**Additional Note: Non-interleaved RoPE**
```python
# Official code comment:
# "rope in indexer is not interleaved"
q_pe = apply_rotary_emb(q_pe, freqs_cis, False)  # False = non-interleaved
```
The indexer uses non-interleaved RoPE layout, while MLA uses interleaved.

### 4. MLP / Dense Layers (Layers 0-2)

| Component | Mapping Type | Reason |
|-----------|--------------|--------|
| `gate_proj` | `ColumnParallelMapping` | Split output dimension |
| `up_proj` | `ColumnParallelMapping` | Split output dimension |
| `down_proj` | `RowParallelMapping` | Split input dimension |

```python
# Gate and Up projections (ColumnParallel)
ColumnParallelMapping(
    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
    hf_param={"gate": "...gate_proj.weight", "up": "...up_proj.weight"},
)

# Down projection (RowParallel)
RowParallelMapping(
    megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
    hf_param="model.layers.*.mlp.down_proj.weight",
)
```

### 5. MoE Layers (Layers 3-60)

| Component | Mapping Type | Reason |
|-----------|--------------|--------|
| `gate` (router) | `ReplicatedMapping` | All ranks need same routing decisions |
| `shared_experts.*_proj` | `ColumnParallel/RowParallel` | Same as dense MLP |
| `experts.*.gate_proj` | `ExpertMapping` | Distributed by EP |
| `experts.*.up_proj` | `ExpertMapping` | Distributed by EP |
| `experts.*.down_proj` | `ExpertMapping` | Distributed by EP |

```python
# Router weights (replicated for consistent routing)
ReplicatedMapping(
    megatron_param="decoder.layers.*.mlp.router.weight",
    hf_param="model.layers.*.mlp.gate.weight",
)

# Expert weights (distributed by Expert Parallelism)
# With EP=1 (conversion default): all experts in checkpoint
# With EP=8: 32 experts per rank (256/8)
```

### 6. Layer Norms

| Component | Mapping Type | Reason |
|-----------|--------------|--------|
| `input_layernorm` | `ReplicatedMapping` | Small, replicate everywhere |
| `post_attention_layernorm` | `ReplicatedMapping` | Small, replicate everywhere |
| `norm` (final) | `ReplicatedMapping` | Small, replicate everywhere |

```python
ReplicatedMapping(
    megatron_param="decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
    hf_param="model.layers.*.input_layernorm.weight",
)
```

### 7. Output Head

| Component | Mapping Type | Reason |
|-----------|--------------|--------|
| `lm_head` | `ColumnParallelMapping` | Split vocabulary across TP |

```python
ColumnParallelMapping(
    megatron_param="output_layer.weight",
    hf_param="lm_head.weight",
)
```

## Summary Table

| Component | Params | Mapping | Parallelism |
|-----------|--------|---------|-------------|
| Embeddings | 927M | VocabParallel | TP |
| Q/K/V projections | ~2B/layer | ColumnParallel | TP |
| O projection | ~500M/layer | RowParallel | TP |
| **Lightning Indexer** | ~100M total | **Replicated** | **None** |
| Dense MLP | ~1B/layer | Column/RowParallel | TP |
| MoE Router | ~2M/layer | Replicated | None |
| MoE Experts | ~10B/layer | ExpertMapping | EP |
| Layer Norms | ~14K/layer | Replicated | None |
| LM Head | 927M | ColumnParallel | TP |

## Conversion Configuration

The conversion uses:
- **TP=1**: No tensor parallelism (most flexible, can be re-sharded at load time)
- **PP=1**: No pipeline parallelism (all layers in one checkpoint)
- **EP=1**: No expert parallelism (all 256 experts in checkpoint)

This configuration produces a checkpoint that can be loaded with any TP/PP/EP configuration at runtime, as Megatron handles re-sharding during load.

## References

- Official DeepSeek V3.2 inference demo: `docs/reference/deepseek_v32_official/model.py`
- DeepSeek V3.2 tech report: https://arxiv.org/abs/2512.02556
- Megatron-Bridge: `Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v32_bridge.py`
