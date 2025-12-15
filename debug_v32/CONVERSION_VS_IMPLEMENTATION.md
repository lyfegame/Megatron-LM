# Distinguishing Conversion vs Implementation Problems

## Reference: Official DeepSeek V3.2 Inference Code

**Location**: `docs/reference/deepseek_v32_official/model.py`

**NOT HuggingFace** - There is no official HF transformers implementation for V3.2.
The reference must always be the official inference demo.

---

## Conversion Problems

Issues related to converting HF checkpoint → Megatron checkpoint format.

### 1. Weight Name Mapping

**HF Format** (from safetensor inspection):
```
model.layers.{i}.self_attn.indexer.wq_b.weight
model.layers.{i}.self_attn.indexer.wk.weight
model.layers.{i}.self_attn.indexer.k_norm.weight
model.layers.{i}.self_attn.indexer.k_norm.bias
model.layers.{i}.self_attn.indexer.weights_proj.weight
```

**Megatron Format** (from bridge mapping):
```
decoder.layers.{i}.self_attention.lightning_indexer.linear_wq_b.weight
decoder.layers.{i}.self_attention.lightning_indexer.linear_wk.weight
decoder.layers.{i}.self_attention.lightning_indexer.k_layernorm.weight
decoder.layers.{i}.self_attention.lightning_indexer.k_layernorm.bias
decoder.layers.{i}.self_attention.lightning_indexer.linear_weights_proj.weight
```

**Status**: Fixed - ReplicatedMapping for all indexer weights

**FACT: Megatron Lightning Indexer Module Types** (`lightning_indexer.py:146-192`):
```python
self.linear_wq_b = build_module(...)       # → ColumnParallelLinear
self.linear_wk = nn.Linear(...)            # → Standard PyTorch Linear
self.k_layernorm = nn.LayerNorm(...)       # → Standard PyTorch LayerNorm
self.linear_weights_proj = build_module(...) # → ColumnParallelLinear
```

**FACT: Official Indexer Module Types** (`model.py:445-449`):
```python
self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)
self.wk = Linear(self.dim, self.head_dim)
self.k_norm = LayerNorm(self.head_dim)
self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)
```

**Issue**: AutoMapping handles ColumnParallelLinear but not plain `nn.Linear`.
**Error seen**: `Cannot determine parallelism type for module 'Linear' at weight 'linear_wk.weight'`
**Fix Applied**: All set to ReplicatedMapping (works for TP=1 conversion)
**Location**: `Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v32_bridge.py:159-184`
**Note**: For TP>1, `linear_wq_b` and `linear_weights_proj` should use AutoMapping/ColumnParallelMapping

### 2. FP8 Dequantization Scale Naming

**FACT**: HF checkpoint uses `weight_scale_inv` (inverse scale)
**FACT**: Bridge looks for `weight_scale`, `scale`, `_scale`
**FACT**: Bridge location: `deepseek_v32_bridge.py:280-283`

```python
# Current bridge code (deepseek_v32_bridge.py:280-283):
scale_keys = [
    hf_param + "_scale",  # weight_scale
    hf_param.replace(".weight", ".scale"),  # .scale
    hf_param.replace(".weight", "_scale"),  # _scale
]
```

**Potential Fix**:
```python
scale_keys = [
    hf_param + "_scale",  # weight_scale
    hf_param + "_scale_inv",  # weight_scale_inv (V3.2 uses this)
    hf_param.replace(".weight", ".scale"),
    hf_param.replace(".weight", ".scale_inv"),  # Add inverse scale pattern
    hf_param.replace(".weight", "_scale"),
]
```

**Impact**: FP8 weights converted directly to BF16 without proper dequantization
**Severity**: Medium - may affect numerical precision
**Note**: The inverse scale needs to be inverted during dequantization: `weight * (1/scale_inv)`

### 3. Weight Shapes

**FACT: V3.2 Config Values** (from `model.py:17-91`):
- `hidden_size (dim)`: 7168
- `index_n_heads`: 64
- `index_head_dim`: 128
- `q_lora_rank`: 1536
- `qk_rope_head_dim`: 64

**Expected Shapes** (from official `model.py:445-449`):
| Weight | Shape Computation | Expected Shape | Status |
|--------|-------------------|----------------|--------|
| wq_b | [n_heads * head_dim, q_lora_rank] | [8192, 1536] | Verify |
| wk | [head_dim, hidden_size] | [128, 7168] | Verify |
| k_norm.weight | [head_dim] | [128] | Verify |
| k_norm.bias | [head_dim] | [128] | Verify |
| weights_proj | [n_heads, hidden_size] | [64, 7168] | Verify |

**Note**: Shape computation verified against official code:
```python
# model.py:445-449
self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)  # [q_lora_rank=1536] -> [64*128=8192]
self.wk = Linear(self.dim, self.head_dim)  # [dim=7168] -> [head_dim=128]
self.k_norm = LayerNorm(self.head_dim)  # [head_dim=128]
self.weights_proj = Linear(self.dim, self.n_heads, dtype=torch.float32)  # [dim=7168] -> [n_heads=64]
```

---

## Implementation Problems

Issues related to algorithmic differences between Official vs Megatron.

### 1. Hadamard Transform

**Official Implementation** (`model.py:472-473`):
```python
q = rotate_activation(q)  # hadamard_transform(x, scale=hidden_size ** -0.5)
k = rotate_activation(k)
```

**Megatron Implementation** (`lightning_indexer.py`):
- Does NOT apply Hadamard transform
- Comment: "No Hadamard transform (vLLM confirmed unnecessary for accuracy)"

**FACT**: Official DOES use Hadamard; Megatron DOES NOT
**Impact**: May affect index selection distribution
**Status**: Claimed unnecessary per vLLM validation

### 2. Precision

**Official Implementation**:
- FP8 (float8_e4m3fn) for Q/K in indexer
- Uses `fp8_index` kernel for scoring

**Megatron Implementation**:
- BF16 for training gradient flow
- Uses einsum for scoring

**FACT**: This is an intentional design choice for training vs inference

### 3. RoPE Layout

**Official Implementation** (`model.py:464,470`):
```python
q_pe = apply_rotary_emb(q_pe, freqs_cis, False)  # interleaved=False
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False).squeeze(2)
```

**Megatron Implementation** (`lightning_indexer.py:48-90`):
```python
def apply_rotary_emb_non_interleaved(x, freqs_cis):
    # Non-interleaved layout
```

**FACT**: Both use non-interleaved RoPE ✓

### 4. Softmax Scale

**Official Implementation** (`model.py:450`):
```python
self.softmax_scale = self.head_dim ** -0.5
```

**Megatron Implementation** (`lightning_indexer.py:195-196`):
```python
self.softmax_scale = self.index_head_dim**-0.5
self.n_heads_scale = self.index_n_heads**-0.5
```

**FACT**: Both use `head_dim ** -0.5` for softmax scale ✓
**FACT**: Both use `n_heads ** -0.5` for weights projection scale ✓

---

## Verification Strategy

### Phase A: Conversion Validation (Isolated from Implementation)

1. **Weight Inventory**: Check all expected weights present in converted checkpoint
2. **Shape Comparison**: Compare tensor shapes between HF and Megatron
3. **Value Comparison**: For sample weights, compare values (accounting for FP8 dequant)

### Phase B: Implementation Validation (Isolated from Conversion)

1. **Unit Tests**: Already done (RoPE, Config pass)
2. **Forward Pass**: Run with known inputs, compare intermediate tensors
3. **Output Comparison**: Compare against official inference outputs

### Phase C: End-to-End Validation

1. Run Megatron inference on 6 reference prompts
2. Compare outputs against official inference demo (NOT HF)
3. Document semantic equivalence and numerical metrics

---

## Critical Path

To determine if a problem is **conversion** or **implementation**:

1. If weight loading fails → **Conversion problem**
2. If shapes mismatch → **Conversion problem**
3. If values differ significantly → Check **FP8 dequant** (conversion) vs **algorithm** (implementation)
4. If model produces garbage → Check both
5. If model produces different but reasonable output → **Implementation problem** (likely Hadamard)

---

## Next Steps

1. Wait for conversion to complete
2. Verify checkpoint structure (weight names, shapes)
3. Run Megatron inference (no HF dependencies post-conversion)
4. Compare against official inference demo outputs
