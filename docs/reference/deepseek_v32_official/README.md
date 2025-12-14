# DeepSeek V3.2 Official Inference Code Reference

This directory contains the official DeepSeek V3.2 inference implementation from HuggingFace,
downloaded for reference when implementing training support in Megatron-LM.

## Source

- Repository: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
- Path: `inference/`
- Downloaded: 2025-12-13

## Files

| File | Lines | Description |
|------|-------|-------------|
| `model.py` | 922 | Main model implementation (Transformer, MLA, Indexer, MoE) |
| `kernel.py` | 274 | Custom TileLang kernels (fp8_index, act_quant, etc.) |
| `generate.py` | 186 | Text generation script |

## Key Components for V3.2 Training Implementation

### From `model.py`:

1. **`Indexer` class (lines ~600-700)** - Lightning Indexer implementation
   - `wq_b`, `wk`, `k_norm`, `weights_proj` weights
   - Non-interleaved RoPE: `apply_rotary_emb(q_pe, freqs_cis, False)`
   - FP8 scoring with `fp8_index` kernel

2. **`MLA` class** - Multi-Latent Attention with sparse attention integration
   - Uses indexer output for token selection
   - KV cache management

3. **`apply_rotary_emb()` function** - RoPE with interleaved flag
   - `interleaved=True` for MLA attention
   - `interleaved=False` for Indexer

### From `kernel.py`:

1. **`fp8_index` kernel** - FP8 scoring for indexer
   - Formula: `logits = relu(q @ k.T) * softmax_scale`
   - `index_score = sum_h(weights[h] * logits[h]) * k_scale`

2. **`act_quant` function** - FP8 quantization (inference only)

3. **`rotate_activation` / Hadamard transform** - vLLM confirmed not needed for accuracy

## Training Adaptations Required

| Official (Inference) | Training Version |
|---------------------|------------------|
| FP8 with `act_quant` | BF16 throughout |
| `fp8_index` kernel | `torch.einsum` (autograd) |
| Hadamard transform | Skip (not needed) |
| KV cache | Full recompute |
| `dist.broadcast` assertion | All-reduce for TP |

## Usage

These files are for **reference only**. Do not import or run them directly.

When implementing V3.2 training support:
1. Compare weight names and shapes
2. Verify RoPE implementation matches
3. Ensure scoring formula is identical
4. Use chunked BF16 instead of FP8 for memory efficiency

## License

Original code is from DeepSeek-AI. See their repository for license terms.
