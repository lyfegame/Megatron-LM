# DeepSeek V3/V3.2 Support in Megatron-LM

This document describes how to run DeepSeek V3 and V3.2 models in Megatron-LM.

## References

- **DeepSeek-V3 Technical Report**: https://arxiv.org/abs/2412.19437
- **DeepSeek-V3.2 Technical Report**: https://arxiv.org/abs/2512.02556
- **Model Weights**: https://huggingface.co/deepseek-ai/DeepSeek-V3

## Quick Start

### DeepSeek V3 (MLA)

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --multi-latent-attention \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
    --rope-type yarn \
    --rotary-scaling-factor 40 \
    # ... other args
```

### DeepSeek V3.2 (MLA + Sparse Attention)

```bash
torchrun --nproc_per_node=8 pretrain_gpt.py \
    --multi-latent-attention \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
    --rope-type yarn \
    --rotary-scaling-factor 40 \
    --use-sparse-attention \
    --index-n-heads 64 \
    --index-head-dim 128 \
    --index-topk 2048 \
    # ... other args
```

## Configuration

### MLA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--multi-latent-attention` | - | Enable Multi-Latent Attention |
| `--q-lora-rank` | None | Query compression rank (1536 for V3) |
| `--kv-lora-rank` | 32 | KV compression rank (512 for V3) |
| `--qk-head-dim` | 128 | QK head dimension |
| `--qk-pos-emb-head-dim` | 64 | Position embedding dimension |
| `--v-head-dim` | 128 | Value head dimension |
| `--rope-type` | rope | RoPE type (`yarn` for V3) |
| `--rotary-scaling-factor` | 1.0 | YaRN scaling factor (40 for V3) |

### Sparse Attention Parameters (V3.2)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use-sparse-attention` | False | Enable Lightning Indexer |
| `--index-n-heads` | 64 | Number of indexer heads |
| `--index-head-dim` | 128 | Indexer head dimension |
| `--index-topk` | 2048 | Top-k tokens per position |

## Programmatic API

### DeepSeek V3

```python
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

config = MLATransformerConfig(
    hidden_size=7168,
    num_attention_heads=128,
    num_layers=61,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_head_dim=128,
    qk_pos_emb_head_dim=64,
    v_head_dim=128,
    rope_type="yarn",
    rotary_scaling_factor=40,
)

layer_spec = get_gpt_layer_with_transformer_engine_spec(
    multi_latent_attention=True,
)
```

### DeepSeek V3.2

```python
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_mla_with_dsa_spec
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

config = MLATransformerConfig(
    hidden_size=7168,
    num_attention_heads=128,
    num_layers=61,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_head_dim=128,
    qk_pos_emb_head_dim=64,
    v_head_dim=128,
    rope_type="yarn",
    rotary_scaling_factor=40,
    # Sparse attention
    use_sparse_attention=True,
    index_n_heads=64,
    index_head_dim=128,
    index_topk=2048,
)

layer_spec = get_mla_with_dsa_spec(TESpecProvider())
```

## Model Configuration (671B)

| Parameter | Value |
|-----------|-------|
| Hidden Size | 7168 |
| Num Layers | 61 |
| Num Attention Heads | 128 |
| FFN Hidden Size | 18432 |
| Vocab Size | 129280 |
| Max Position Embeddings | 163840 |
| Q LoRA Rank | 1536 |
| KV LoRA Rank | 512 |
| Num MoE Experts | 256 |
| Experts Per Token | 8 |

## Sparse Attention (V3.2)

DeepSeek V3.2 adds the **Lightning Indexer** for sparse attention, reducing complexity from O(n²) to O(n·k) for long sequences.

The indexer computes:
```
I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)
```

Only the top-k positions participate in full attention.

### When to Use

- **Short sequences (<4K)**: Sparse attention adds overhead; use V3 (MLA only)
- **Long sequences (>8K)**: Sparse attention provides significant speedup

### Tuning `index_topk`

Recommended: `k = min(2048, seq_len // 4)`

| Sequence Length | Recommended `index_topk` |
|-----------------|--------------------------|
| 4K | 1024 |
| 8K | 2048 |
| 32K | 2048 |
| 128K | 2048 |

## Checkpoint Conversion from HuggingFace

Convert DeepSeek V3 checkpoints from HuggingFace format to Megatron format:

```bash
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader deepseek_hf \
    --saver core \
    --load-dir /path/to/deepseek-ai/DeepSeek-V3 \
    --save-dir /path/to/megatron-checkpoint \
    --tokenizer-model /path/to/deepseek-ai/DeepSeek-V3 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 4 \
    --bf16
```

### Conversion Options

| Option | Description |
|--------|-------------|
| `--loader deepseek_hf` | Use DeepSeek HuggingFace loader |
| `--saver core` | Save in Megatron Core format |
| `--target-tensor-parallel-size` | Target TP size for saved checkpoint |
| `--target-pipeline-parallel-size` | Target PP size for saved checkpoint |
| `--bf16` | Convert weights to bfloat16 |
| `--fp16` | Convert weights to float16 |
| `--tokenizer-model` | Path to HuggingFace tokenizer |

### Supported Features

The checkpoint converter handles:
- **MLA weights**: Q/KV LoRA projections, layernorms
- **MoE weights**: Router, expert MLPs, shared experts
- **YaRN RoPE**: Extended context configuration
- **SwiGLU**: Fused gate/up projections

### Memory Requirements

DeepSeek V3 (671B parameters) requires significant memory for conversion:
- Minimum ~1.5TB CPU RAM for full precision
- Use `--bf16` to reduce memory by 50%
- Consider using a high-memory instance (e.g., p4d.24xlarge)

## Known Limitations

1. Sparse attention not tested with sequence packing (THD format)
2. Context parallelism (CP > 1) may require modifications
3. Lightning Indexer weights (V3.2) not yet included in HuggingFace checkpoints
