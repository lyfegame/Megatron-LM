# DeepSeek V3.2 Bridge - Handoff Notes

## Overview

This document describes the Megatron Bridge implementation for DeepSeek V3.2 checkpoint conversion. The bridge enables converting HuggingFace V3.2 checkpoints (including FP8 quantized) to Megatron format for training/fine-tuning.

## What's Implemented

### Files Created

1. **`src/megatron/bridge/models/deepseek/deepseek_v32_bridge.py`**
   - `DeepSeekV32Bridge` - Main bridge class registered for `DeepseekV32ForCausalLM`
   - `DeepSeekV32ModelProvider` - Model provider with V3.2 indexer configs
   - `_dequantize_fp8_weight()` - FP8 → BF16 dequantization function
   - `get_v32_indexer_mapping_list()` - Weight mappings for Lightning Indexer

2. **`examples/conversion/convert_deepseek_v32.py`**
   - Standalone conversion script with CLI interface

3. **`src/megatron/bridge/models/deepseek/__init__.py`** (modified)
   - Added exports for V3.2 classes

## V3.2 Architecture (vs V3)

DeepSeek V3.2 adds the **Lightning Indexer** for sparse attention:

```
V3.2 Attention:
  hidden_states ─┬─► MLA (Q/KV projections) ─► Full Q, K, V
                 │
                 └─► Lightning Indexer ─► top-K indices (K=2048)
                                              │
                                              ▼
                                       Sparse Attention O(L×K)
```

### New Weight Tensors per Layer (5 total)

| HF Name | Megatron Name | Shape | TP Handling |
|---------|---------------|-------|-------------|
| `indexer.wq_b.weight` | `lightning_indexer.linear_wq_b.weight` | [n_heads×head_dim, q_lora_rank] | Column-parallel |
| `indexer.wk.weight` | `lightning_indexer.linear_wk.weight` | [head_dim, hidden_size] | **Replicated** |
| `indexer.k_norm.weight` | `lightning_indexer.k_layernorm.weight` | [head_dim] | **Replicated** |
| `indexer.k_norm.bias` | `lightning_indexer.k_layernorm.bias` | [head_dim] | **Replicated** |
| `indexer.weights_proj.weight` | `lightning_indexer.linear_weights_proj.weight` | [n_heads, hidden_size] | Column-parallel |

## Usage

### Option 1: Python API

```python
from megatron.bridge import AutoBridge

# Load V3.2 checkpoint (auto-detects architecture and uses DeepSeekV32Bridge)
bridge = AutoBridge.from_hf_pretrained(
    "/path/to/DeepSeek-V3.2-fp8",
    trust_remote_code=True,
    # device_map="auto"  # Optional: use GPUs if available
)

# Get Megatron provider with V3.2 configs
provider = bridge.to_megatron_provider()

# Or directly convert and save
AutoBridge.import_ckpt(
    hf_model_id="/path/to/DeepSeek-V3.2-fp8",
    megatron_path="./checkpoints/deepseek_v32",
    trust_remote_code=True,
)
```

### Option 2: CLI Script

```bash
# Using the generic conversion script
python examples/conversion/convert_checkpoints.py import \
    --hf-model /path/to/DeepSeek-V3.2-fp8 \
    --megatron-path ./checkpoints/deepseek_v32 \
    --trust-remote-code

# Using the V3.2-specific script
python examples/conversion/convert_deepseek_v32.py \
    --hf-path /path/to/DeepSeek-V3.2-fp8 \
    --megatron-path ./checkpoints/deepseek_v32
```

## FP8 Dequantization

The official V3.2 checkpoint uses FP8 quantization (`torch.float8_e4m3fn`) with block-wise scales:

- **Block size**: 128×128
- **Scale format**: `weight_scale` tensor alongside each FP8 weight
- **Dequantization**: Automatic during `maybe_modify_loaded_hf_weight()`

The bridge automatically:
1. Detects FP8 weights by dtype
2. Finds corresponding scale tensors (tries `*_scale`, `*.scale`, etc.)
3. Applies block-wise dequantization to BF16

## Memory Requirements

For DeepSeek V3.2 (685B parameters):

| Format | Approximate Size |
|--------|------------------|
| FP8 checkpoint | ~700 GB |
| After BF16 dequantization | ~1.4 TB peak |

The bridge runs on **CPU by default** (`use_cpu_initialization=True`). For limited RAM, use `device_map="auto"` to offload to GPUs.

## Testing

To verify the bridge works:

```python
# Quick verification with config only (no weights)
from megatron.bridge import AutoBridge
from transformers import AutoConfig

config = AutoConfig.from_pretrained(
    "/path/to/DeepSeek-V3.2-fp8",
    trust_remote_code=True
)

# Check V3.2 params are present
print(f"index_n_heads: {config.index_n_heads}")     # Expected: 64
print(f"index_head_dim: {config.index_head_dim}")   # Expected: 128
print(f"index_topk: {config.index_topk}")           # Expected: 2048

# Verify bridge can handle the architecture
assert AutoBridge.can_handle("/path/to/DeepSeek-V3.2-fp8", trust_remote_code=True)
```

## Known Limitations / TODOs

1. **MTP (Multi-Token Prediction)**: Not yet mapped - V3.2 has MTP layers for speculative decoding
2. **Hadamard Transform**: Skipped - vLLM confirmed no accuracy benefit for training
3. **LoRA for Indexer**: Indexer weights should be frozen during LoRA fine-tuning (not yet enforced)

## Checkpoint Location

Official FP8 checkpoint: `gs://fundamental_ml_shared_storage/models/DeepSeek-V3.2-fp8/`

## Related Files in Megatron-Core

The V3.2 support also requires changes in megatron-core (separate from this bridge):

- `megatron/core/transformer/lightning_indexer.py` - Indexer module
- `megatron/core/transformer/sparse_attention.py` - Sparse attention
- `megatron/core/transformer/transformer_config.py` - V3.2 config params
- `megatron/core/transformer/multi_latent_attention.py` - Integration

These are in the main repo, not Megatron-Bridge.

## Contact

For questions about this implementation, refer to:
- Tech report: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main/assets
- Official inference code: `docs/reference/deepseek_v32_official/`
