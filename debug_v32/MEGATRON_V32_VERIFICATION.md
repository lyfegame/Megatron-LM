# DeepSeek V3.2 Megatron-Core Verification

## Overview

This document records the verification of Megatron-Core's DeepSeek V3.2 implementation against:
1. Official DeepSeek V3.2 inference code
2. Validated HuggingFace transformers fork

## Methodology

Following the methodology from `debug_v32/VERIFICATION_METHODOLOGY.md`:
- Semantic equivalence testing (primary)
- Numerical divergence measurement (secondary)
- 6 reference prompts covering dense and sparse attention paths
- Sparse attention trigger verification (>2048 tokens)

## Environment

- **Cluster**: h200-mig-cluster-rn1h (8x H200 GPUs)
- **Checkpoints**:
  - FP8: `gs://fundamental_ml_shared_storage/models/DeepSeek-V3.2-fp8/`
  - BF16: `/models-local/DeepSeek-V3.2-bf16`
- **Branch**: `shuyingl/deepseek-v32-bridge`

---

## Test Results

### Unit Tests

| Test | Status | Notes |
|------|--------|-------|
| `test_lightning_indexer.py` | PENDING | |
| `test_multi_latent_attention.py` | PENDING | |

### Reference Prompt Tests

| Prompt | Input Tokens | Expected Path | Status | Output Summary |
|--------|--------------|---------------|--------|----------------|
| 0: simple_math | ~10 | Dense | PENDING | |
| 1: greeting | ~10 | Dense | PENDING | |
| 2: code_generation | ~15 | Dense | PENDING | |
| 3: explanation | ~15 | Dense | PENDING | |
| 4: long_context | ~500 | Dense | PENDING | |
| 5: sparse_trigger | ~2251 | **Sparse** | PENDING | |

### Numerical Metrics

| Metric | Target | Measured | Notes |
|--------|--------|----------|-------|
| Semantic equivalence | Pass | PENDING | |
| Logits cosine similarity | >0.99 | PENDING | |
| Sparse attention shape | [1, 2251, 2048] | PENDING | |

---

## Technical Comparisons

### 1. Hadamard Transform

**Question**: Is Hadamard transform necessary for accuracy?

**Official Implementation** (`docs/reference/deepseek_v32_official/model.py:472-473`):
```python
q = rotate_activation(q)
k = rotate_activation(k)
```

**Megatron Implementation** (`megatron/core/transformer/lightning_indexer.py`):
- Hadamard transform is NOT applied
- Comment states: "No Hadamard transform (vLLM confirmed unnecessary for accuracy)"

**Verification Results**:
- [ ] Test with Hadamard: PENDING
- [ ] Test without Hadamard: PENDING
- [ ] Accuracy difference: PENDING

### 2. RoPE Layout

**Official Implementation** (`docs/reference/deepseek_v32_official/model.py:463-464`):
```python
# rope in indexer is not interleaved
q_pe = apply_rotary_emb(q_pe, freqs_cis, False)
```

**Megatron Implementation** (`megatron/core/transformer/lightning_indexer.py:48-90`):
```python
def apply_rotary_emb_non_interleaved(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings with NON-interleaved layout."""
```

**Finding**: Both use non-interleaved RoPE for indexer. ✓

### 3. FP8 vs BF16

**Official Implementation**: Uses FP8 for Q/K caching and `fp8_index` kernel
**Megatron Implementation**: Uses BF16 (for gradient flow in training)

**Rationale**: Training requires gradient flow, FP8 is inference optimization

---

## Issues Found

| Issue | Severity | Status | Fix |
|-------|----------|--------|-----|
| | | | |

---

## Timeline

| Date | Action | Result |
|------|--------|--------|
| 2025-12-14 | Started verification | In progress |

