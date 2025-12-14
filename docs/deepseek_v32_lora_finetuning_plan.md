# DeepSeek V3.2 LoRA Fine-tuning Implementation Plan

## Executive Summary

**Goal:** Enable LoRA fine-tuning of DeepSeek V3.2 (685B) in Megatron-LM with sparse attention support.

**Constraints:**
- 40 × H100 80GB GPUs
- Up to 32K context length
- LoRA fine-tuning (not full fine-tuning)
- Training support required (not just inference)

**Timeline:** 8-10 weeks

### Key Findings from Official Implementation Analysis

| Finding | Source | Impact |
|---------|--------|--------|
| Hadamard transform NOT needed | [vLLM blog](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html) | Simplifies training implementation |
| Indexer uses non-interleaved RoPE | [HF model.py](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py) | Critical correctness requirement |
| FP8 kernels are inference-only | Official kernel.py | Need BF16 path for training |
| MoE for-loop is reference code only | Megatron has GroupedMLP | Not a performance concern |
| Indexer weights directly reusable | Checkpoint analysis | Easy weight loading |

### Implementation Approach

1. **Adapt official HF inference code** for training (BF16, autograd-compatible)
2. **Reuse existing Megatron infrastructure** (MoE, MLA, RoPE, parallelism)
3. **Add only what's missing**: Lightning Indexer + Sparse Attention + Checkpoint conversion

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation Phases](#3-implementation-phases)
4. [Detailed Component Design](#4-detailed-component-design)
5. [Parallelism Strategy](#5-parallelism-strategy)
6. [Checkpoint Conversion](#6-checkpoint-conversion)
7. [LoRA Integration](#7-lora-integration)
8. [Testing Strategy](#8-testing-strategy)
9. [Risk Assessment](#9-risk-assessment)
10. [Timeline](#10-timeline)

---

## 1. Current State Analysis

### What Megatron-LM Already Has (V3 Support)

| Component | Status | File Location |
|-----------|--------|---------------|
| MLA (Multi-Latent Attention) | ✅ Complete | `megatron/core/transformer/multi_latent_attention.py` |
| MoE with EP | ✅ Complete | `megatron/core/transformer/moe/` |
| YARN RoPE | ✅ Complete | `megatron/core/models/common/embeddings/rope_utils.py` |
| DeepEP Dispatcher | ✅ Complete | `megatron/core/transformer/moe/fused_a2a.py` |
| Fused MLA RoPE | ✅ Complete | `megatron/core/fusions/fused_mla_yarn_rope_apply.py` |
| FSDP + EP | ✅ Supported | `megatron/core/distributed/fsdp/` |
| Node-Limited Routing | ✅ Complete | `megatron/core/transformer/moe/router.py` |

### What's Missing for V3.2

| Component | Status | Complexity | Priority |
|-----------|--------|------------|----------|
| Lightning Indexer | ❌ Not implemented | Medium | P0 |
| Sparse Attention | ❌ Not implemented | Medium | P0 |
| Indexer KV Cache | ❌ Not implemented | Low | P0 |
| V3.2 Config Support | ❌ Not implemented | Low | P0 |
| Checkpoint Conversion | ❌ Not implemented | Medium | P0 |
| LoRA for Indexer | ❌ Not implemented | Low | P1 |

### MoE Dispatch Efficiency (NOT a Blocker)

The official HuggingFace V3.2 inference demo uses an inefficient for-loop for expert dispatch:

```python
# HuggingFace demo - INEFFICIENT (O(num_experts) kernel launches)
for i in range(self.experts_start_idx, self.experts_end_idx):
    if counts[i] == 0:
        continue
    expert = self.experts[i]
    idx, top = torch.where(indices == i)
    y[idx] += expert(x[idx]) * weights[idx, top, None]
```

**Megatron already has efficient implementations:**

| Implementation | Kernel Launches | File |
|---------------|-----------------|------|
| `GroupedMLP` | O(1) - batched GEMM | `megatron/core/transformer/moe/experts.py` |
| `SequentialMLP` | O(num_experts) - same as HF | `megatron/core/transformer/moe/experts.py` |
| DeepEP Dispatcher | Fused dispatch+AlltoAll | `megatron/core/transformer/moe/fused_a2a.py` |
| HybridEP Dispatcher | Fused dispatch+AlltoAll | `megatron/core/transformer/moe/token_dispatcher.py` |

```python
# Megatron GroupedMLP - EFFICIENT (O(1) kernel launches)
w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

# Single grouped GEMM handles all experts
fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False)
intermediate = self.activation_func(fc1_output) * probs
fc2_output = gg.ops.gmm(intermediate, w2, tokens_per_expert, trans_b=False)
```

**Key insight:** The for-loop in HuggingFace is for **simplicity of reference code**, not performance. Megatron's production `GroupedMLP` batches all 256 experts into a single optimized kernel call.

**Conclusion:** MoE dispatch is NOT a blocker for V3.2 - existing Megatron infrastructure handles it efficiently.

---

## 2. Architecture Overview

### DeepSeek V3.2 vs V3 Comparison

```
DeepSeek V3 (Current Support)
├── Embedding
├── 61 × TransformerLayer
│   ├── RMSNorm
│   ├── MLA (Multi-Latent Attention)
│   │   ├── Q: hidden → q_lora(1536) → q_up(128×192)
│   │   ├── KV: hidden → kv_lora(512) + k_pe(64) → kv_up
│   │   ├── RoPE (interleaved, YARN)
│   │   └── Full Attention O(L²)
│   ├── RMSNorm
│   └── MoE (256 experts, top-8, shared expert)
├── RMSNorm
└── LM Head

DeepSeek V3.2 (To Implement)
├── Embedding
├── 61 × TransformerLayer
│   ├── RMSNorm
│   ├── MLA + Sparse Attention
│   │   ├── Q: hidden → q_lora(1536) → q_up(128×192)
│   │   ├── KV: hidden → kv_lora(512) + k_pe(64) → kv_up
│   │   ├── RoPE (interleaved, YARN)
│   │   ├── ┌─────────────────────────────────────┐
│   │   ├── │ NEW: Lightning Indexer              │
│   │   ├── │   ├── wq_b: q_lora → index_heads×d  │
│   │   ├── │   ├── wk: hidden → index_head_dim   │
│   │   ├── │   ├── k_norm: LayerNorm             │
│   │   ├── │   ├── weights_proj: hidden → heads  │
│   │   ├── │   ├── RoPE (NON-interleaved!)       │
│   │   ├── │   └── Top-K Selection (k=2048)      │
│   │   ├── └─────────────────────────────────────┘
│   │   └── Sparse Attention O(L×k)
│   ├── RMSNorm
│   └── MoE (256 experts, top-8, shared expert)
├── RMSNorm
└── LM Head
```

### V3.2 Config Parameters (New)

```python
# From HuggingFace config.json
{
    # Existing V3 params...

    # NEW V3.2 Indexer params
    "index_n_heads": 64,        # Number of indexer attention heads
    "index_head_dim": 128,      # Dimension per indexer head
    "index_topk": 2048,         # Top-K tokens to select

    # Model type
    "model_type": "deepseek_v32",
    "architectures": ["DeepseekV32ForCausalLM"]
}
```

---

## 3. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Add V3.2 config support to `MLATransformerConfig`
- [ ] Implement `LightningIndexer` module (BF16, autograd-compatible)
- [ ] Implement sparse attention gather/scatter ops
- [ ] Unit tests for indexer forward/backward
- [ ] Unit tests for sparse attention

### Phase 2: Integration (Weeks 3-4)
- [ ] Integrate indexer into `MLASelfAttention`
- [ ] Handle non-interleaved RoPE for indexer
- [ ] Checkpoint conversion script (HF → Megatron)
- [ ] Validate forward pass matches reference

### Phase 3: LoRA Support (Weeks 5-6)
- [ ] Identify LoRA target modules for V3.2
- [ ] Implement LoRA adapters for indexer
- [ ] Test LoRA with existing Megatron LoRA infrastructure
- [ ] Memory profiling and optimization

### Phase 4: Training Pipeline (Weeks 7-8)
- [ ] Configure parallelism for 40 GPUs
- [ ] Gradient checkpointing for indexer
- [ ] End-to-end training test
- [ ] Fine-tuning experiment on validation set

### Phase 5: Optimization (Weeks 9-10)
- [ ] Integrate TileLang sparse attention backward kernel
- [ ] Performance benchmarking
- [ ] Scale to 32K context
- [ ] Documentation

---

## 4. Detailed Component Design

### 4.1 Lightning Indexer Module

**File:** `megatron/core/transformer/lightning_indexer.py`

**Reference:** Adapted from [HuggingFace DeepSeek-V3.2 inference code](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py)

#### Key Findings from Official Implementation

| Aspect | Official (Inference) | Training Version |
|--------|---------------------|------------------|
| Precision | FP8 with `act_quant` | BF16 throughout |
| Scoring kernel | `fp8_index` (TileLang) | `torch.einsum` (autograd) |
| Hadamard transform | Used for FP8 stability | **Removed** (vLLM confirmed not needed) |
| KV cache | `k_cache`, `k_scale_cache` | None (full recompute) |
| Top-K gradient | N/A (inference) | `.detach()` for stability |

#### Config

```python
@dataclass
class LightningIndexerConfig:
    """Configuration for DeepSeek V3.2 Lightning Indexer"""
    hidden_size: int = 7168
    q_lora_rank: int = 1536
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    qk_rope_head_dim: int = 64      # Must match MLA
    rotary_base: float = 10000.0
    rotary_scaling_factor: float = 40.0
    # YARN params
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    # Parallelism
    tensor_model_parallel_size: int = 1


class LightningIndexer(nn.Module):
    """
    DeepSeek V3.2 Lightning Indexer for sparse attention (Training-compatible).

    Adapted from official HF inference code with these changes:
    - BF16 instead of FP8 (for gradient flow)
    - PyTorch ops instead of TileLang kernels (for autograd)
    - No Hadamard transform (confirmed unnecessary by vLLM)
    - No KV cache (training recomputes each forward)

    Key differences from MLA attention:
    1. Uses NON-interleaved RoPE (MLA uses interleaved)
    2. Uses ReLU activation on attention logits
    3. Aggregates scores across heads with learned weights
    4. Returns indices, not attention output

    Scoring formula (from official fp8_index kernel):
        logits = relu(q @ k.T) * softmax_scale
        index_score = sum_h(weights[h] * logits[h]) * k_scale
    """

    def __init__(self, config: LightningIndexerConfig, layer_number: int):
        super().__init__()
        self.config = config
        self.layer_number = layer_number

        # Dimensions
        self.hidden_size = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank

        # TP setup
        self.tp_size = config.tensor_model_parallel_size
        self.n_local_heads = self.n_heads // self.tp_size

        # Weights (EXACT match to official HF checkpoint)
        # wq_b: projects compressed Q to indexer heads
        # Shape: [n_local_heads * head_dim, q_lora_rank] (column-parallel)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            gather_output=False,  # Keep split across TP
        )

        # wk: projects hidden to single key (REPLICATED, not split)
        # Shape: [head_dim, hidden_size]
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)

        # k_norm: LayerNorm on key (REPLICATED)
        self.k_norm = nn.LayerNorm(self.head_dim)

        # weights_proj: per-head aggregation weights (column-parallel)
        # Shape: [n_local_heads, hidden_size]
        self.weights_proj = ColumnParallelLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            gather_output=False,  # Keep split across TP
        )

        # Scaling factors
        self.softmax_scale = self.head_dim ** -0.5
        self.n_heads_scale = self.n_heads ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,       # [b, s, hidden]
        q_compressed: torch.Tensor,        # [b, s, q_lora_rank]
        freqs_cis: torch.Tensor,           # RoPE frequencies
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute top-k token indices for sparse attention.

        Args:
            hidden_states: Input hidden states [batch, seq, hidden]
            q_compressed: Compressed Q from MLA's q_down_proj [batch, seq, q_lora_rank]
            freqs_cis: Precomputed RoPE frequencies
            attention_mask: Optional causal mask

        Returns:
            topk_indices: Selected token indices [batch, seq, topk]
        """
        bsz, seqlen, _ = hidden_states.size()

        # === Q Projection (from shared compressed representation) ===
        # Same as official: q = self.wq_b(qr)
        q = self.wq_b(q_compressed)  # [b, s, n_local_heads * head_dim]
        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # === Non-interleaved RoPE for Q ===
        # CRITICAL: Official uses interleaved=False
        q_pe, q_nope = q.split([self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb_non_interleaved(q_pe, freqs_cis)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # === K Projection + Norm ===
        # Same as official: k = self.k_norm(self.wk(x))
        k = self.wk(hidden_states)  # [b, s, head_dim]
        k = self.k_norm(k)

        # === Non-interleaved RoPE for K ===
        k_pe, k_nope = k.split([self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb_non_interleaved(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # === NO Hadamard Transform ===
        # Official uses rotate_activation() for FP8 stability
        # vLLM confirmed: "removed Hadamard transforms (no measurable accuracy benefit)"

        # === BF16 Scoring (replaces fp8_index kernel) ===
        # logits[b, s, h, t] = q[b, s, h, :] @ k[b, t, :].T
        logits = torch.einsum('bshd,btd->bsht', q, k) * self.softmax_scale

        # ReLU activation (same as official)
        logits = F.relu(logits)

        # === Weighted sum across heads ===
        # Same as official: weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        weights = self.weights_proj(hidden_states.float()) * self.n_heads_scale  # [b, s, n_local_heads]

        # index_score = sum_h(weights[h] * logits[h])
        index_score_local = torch.einsum('bsh,bsht->bst', weights, logits)  # [b, s, t]

        # === TP All-Reduce ===
        # Sum partial scores across TP ranks to get full index_score
        if self.tp_size > 1:
            index_score = tensor_model_parallel_all_reduce(index_score_local)
        else:
            index_score = index_score_local

        # === Apply Mask ===
        if attention_mask is not None:
            index_score = index_score + attention_mask

        # === Top-K Selection ===
        actual_topk = min(self.topk, seqlen)
        topk_indices = index_score.topk(actual_topk, dim=-1)[1]

        # Detach for LoRA fine-tuning (indices don't need gradients)
        # Gradients flow through the indexer weights, not through the selection
        return topk_indices.detach()


def apply_rotary_emb_non_interleaved(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings with NON-interleaved layout.

    Non-interleaved: rotate halves (x0,x4), (x1,x5), (x2,x6), (x3,x7)
    Interleaved (MLA): rotate pairs (x0,x1), (x2,x3), (x4,x5), (x6,x7)

    Adapted from official HF code: apply_rotary_emb(x, freqs_cis, interleaved=False)
    """
    dtype = x.dtype
    shape = x.shape

    # Non-interleaved: reshape to put halves together
    # [b, s, h, d] -> [b, s, h, 2, d//2] -> [b, s, h, d//2, 2]
    x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()

    # Apply rotation
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)

    # Undo non-interleaved layout
    y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)

    return y.to(dtype)
```

**Weight Shapes:**

| Weight | Shape | Parameters | Notes |
|--------|-------|------------|-------|
| `wq_b` | `[8192, 1536]` | 12.6M | Projects compressed Q to indexer heads |
| `wk` | `[128, 7168]` | 0.9M | Projects hidden to indexer key |
| `k_norm.weight` | `[128]` | 128 | LayerNorm |
| `k_norm.bias` | `[128]` | 128 | LayerNorm |
| `weights_proj` | `[64, 7168]` | 0.46M | Per-head aggregation weights |
| **Per-layer total** | | **~14M** | |
| **All 61 layers** | | **~854M** | |

#### Adoption Analysis from Official `Indexer` Class

**What Can Be Adopted Directly (100% Compatible):**

| Component | Official Code | Training Adaptation |
|-----------|---------------|---------------------|
| `wq_b` | `Linear(q_lora_rank, n_heads * head_dim)` | ✅ Same, use `ColumnParallelLinear` |
| `wk` | `Linear(dim, head_dim)` | ✅ Same, replicated |
| `k_norm` | `LayerNorm(head_dim)` | ✅ Same |
| `weights_proj` | `Linear(dim, n_heads, dtype=fp32)` | ✅ Same, use `ColumnParallelLinear` |
| Q/K projections | `q = wq_b(qr)`, `k = wk(x)` | ✅ Same |
| Non-interleaved RoPE | `apply_rotary_emb(q_pe, freqs_cis, False)` | ✅ Same |
| Softmax scale | `head_dim ** -0.5` | ✅ Same |
| n_heads scale | `n_heads ** -0.5` | ✅ Same |

**What CANNOT Be Adopted (Inference-Only):**

| Component | Official Code | Training Replacement |
|-----------|---------------|---------------------|
| `act_quant` | FP8 quantization | BF16 (no quantization) |
| `fp8_index` | TileLang kernel | `torch.einsum` |
| `rotate_activation` | Hadamard transform | **Skip** (vLLM confirmed unnecessary) |
| `k_cache` | FP8 KV cache | No cache (full recompute) |
| `dist.broadcast` assertion | Consistency check | All-reduce (handles TP) |

#### Efficiency Analysis

**Computational Complexity:**

```
┌─────────────────────────────────────────────────────────────────────┐
│ Operation              │ FLOPs                    │ Memory BW       │
├────────────────────────┼──────────────────────────┼─────────────────┤
│ Q proj (wq_b)          │ 2 × b×s×1536×8192        │ Read 12.6M      │
│                        │ = 25.2M per token        │ params          │
├────────────────────────┼──────────────────────────┼─────────────────┤
│ K proj (wk)            │ 2 × b×s×7168×128         │ Read 0.9M       │
│                        │ = 1.8M per token         │ params          │
├────────────────────────┼──────────────────────────┼─────────────────┤
│ RoPE (Q)               │ O(b×s×64×64)             │ Negligible      │
│ RoPE (K)               │ O(b×s×64)                │                 │
├────────────────────────┼──────────────────────────┼─────────────────┤
│ Scoring: q @ k.T       │ 2 × b×s×t×64×128         │ ⚠️ DOMINANT     │
│ [b,s,64,128]@[b,t,128] │ = 16.4M × s×t            │                 │
├────────────────────────┼──────────────────────────┼─────────────────┤
│ Weighted sum           │ O(b×s×t×64)              │                 │
├────────────────────────┼──────────────────────────┼─────────────────┤
│ Top-K                  │ O(b×s×t×log(k))          │                 │
└────────────────────────┴──────────────────────────┴─────────────────┘
```

**The O(L²) Problem:**

The scoring operation is O(L²):
```
For 32K context (s = t = 32768):
  Scoring FLOPs = 2 × 1 × 32K × 32K × 64 × 128
                = 17.6 TFLOPs per layer × 61 layers
                = 1.07 PFLOPs total for indexer
```

**Why it's still efficient:**

| Factor | Indexer | Full MLA Attention |
|--------|---------|-------------------|
| Head dim | 128 | 192 (effective) |
| Num heads | 64 | 128 |
| Softmax | **No** (just ReLU) | Yes (expensive) |
| Backward | Detached (no grad through top-k) | Full gradient |
| Output | Indices only | Full attention output |

**Memory Efficiency:**

```python
# Indexer activation memory at 32K
b, s, t = 1, 32768, 32768
n_heads, head_dim = 64, 128

# Q: [b, s, n_heads, head_dim] in BF16
q_mem = b * s * n_heads * head_dim * 2  # 512 MB

# K: [b, t, head_dim] in BF16
k_mem = b * t * head_dim * 2  # 8 MB

# Logits: [b, s, n_heads, t] in BF16  ← DOMINANT
logits_mem = b * s * n_heads * t * 2  # 256 GB ❌ TOO BIG!

# With chunked computation (chunk_size=1024):
logits_mem_chunked = b * 1024 * n_heads * t * 2  # 8 GB per chunk ✓
```

#### Implementation Options

**Option 1: Simple BF16 (for correctness testing)**
- Memory: O(b × s × t × n_heads) - ~256GB for 32K ❌
- Use for: Unit tests, short sequences

**Option 2: Chunked BF16 (recommended for training)**
- Memory: O(b × chunk_size × t × n_heads) - ~8GB per chunk ✓
- Use for: Production training at 32K

```python
def forward_chunked(self, x, qr, freqs_cis, mask, chunk_size=1024):
    """Memory-efficient chunked implementation"""
    # Precompute K once (small: 8MB for 32K)
    k = self.k_norm(self.wk(x))
    k = apply_rope_non_interleaved(k, freqs_cis)

    # Precompute weights once
    weights = self.weights_proj(x.float()) * (self.n_heads ** -0.5)

    # Process Q in chunks
    index_scores = []
    for i in range(0, seqlen, chunk_size):
        q_chunk = self.wq_b(qr[:, i:i+chunk_size])
        q_chunk = apply_rope_non_interleaved(q_chunk, freqs_cis[i:i+chunk_size])

        # Score chunk against ALL keys
        logits_chunk = einsum('bshd,btd->bsht', q_chunk, k) * self.softmax_scale
        score_chunk = einsum('bsh,bsht->bst', weights[:, i:i+chunk_size], F.relu(logits_chunk))
        index_scores.append(score_chunk)

    return torch.cat(index_scores, dim=1).topk(self.topk, dim=-1)[1].detach()
```

**Option 3: Fused Triton Kernel (optional optimization)**
- Memory: O(L × topk) - minimal
- Use for: If chunked is too slow

#### Efficiency Summary

| Implementation | Memory | Speed | Effort |
|----------------|--------|-------|--------|
| Official (FP8) | O(1) with KV cache | Fast | N/A (inference only) |
| Simple BF16 | O(L² × heads) | Slow | Low |
| **Chunked BF16** | **O(chunk × L × heads)** | **Medium** | **Medium** |
| Fused Triton | O(L × topk) | Fast | High |

**Recommendation:** Start with Chunked BF16 - it works for 32K and is memory-efficient.

**Key insight:** The indexer is O(L²) in compute but enables 94% sparsity (2048/32768) in the attention that follows, making the overall complexity O(L × k) where k=2048.

### 4.2 Sparse Attention Module

**File:** `megatron/core/transformer/sparse_attention.py`

```python
class SparseAttention(nn.Module):
    """
    Efficient sparse attention using gathered K/V.

    Instead of computing full [s, s] attention matrix,
    gathers only top-k K/V and computes [s, k] attention.
    """

    def forward(
        self,
        query: torch.Tensor,           # [b, s, h, d]
        key: torch.Tensor,             # [b, s, h, d]
        value: torch.Tensor,           # [b, s, h, d]
        topk_indices: torch.Tensor,    # [b, s, k]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Efficient implementation:
        1. Gather K[topk_indices] and V[topk_indices]
        2. Compute attention on [s, k] instead of [s, s]
        3. Apply causal mask based on indices
        """
        batch_size, seq_len, n_heads, head_dim = query.shape
        topk = topk_indices.shape[-1]

        # Expand indices for gathering
        # [b, s, k] -> [b, s, k, h, d]
        indices_expanded = topk_indices.unsqueeze(-1).unsqueeze(-1)
        indices_expanded = indices_expanded.expand(-1, -1, -1, n_heads, head_dim)

        # Gather selected K and V
        # key_gathered[b, s, k, h, d] = key[b, topk_indices[b,s,k], h, d]
        key_expanded = key.unsqueeze(2).expand(-1, -1, topk, -1, -1)
        key_gathered = torch.gather(key_expanded, dim=1, index=indices_expanded)

        value_expanded = value.unsqueeze(2).expand(-1, -1, topk, -1, -1)
        value_gathered = torch.gather(value_expanded, dim=1, index=indices_expanded)

        # Compute attention scores [b, s, h, k]
        scale = head_dim ** -0.5
        scores = torch.einsum('bshd,bskhd->bshk', query, key_gathered) * scale

        # Causal mask: mask positions where index > current position
        positions = torch.arange(seq_len, device=query.device).view(1, -1, 1, 1)
        indices_for_mask = topk_indices.unsqueeze(2)  # [b, s, 1, k]
        causal_mask = torch.where(
            indices_for_mask > positions,
            torch.tensor(float('-inf'), device=query.device, dtype=query.dtype),
            torch.tensor(0.0, device=query.device, dtype=query.dtype),
        )
        scores = scores + causal_mask

        # Softmax and output
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bshk,bskhd->bshd', attn_weights, value_gathered)

        return output
```

### 4.3 Modified MLA with Sparse Attention

**File:** `megatron/core/transformer/multi_latent_attention.py` (modifications)

```python
class MLASelfAttention(MultiLatentAttention):
    def __init__(self, config, submodules, layer_number, ...):
        super().__init__(...)

        # Detect V3.2 mode
        self.use_sparse_attention = (
            hasattr(config, 'index_topk') and
            config.index_topk is not None and
            config.index_topk > 0
        )

        if self.use_sparse_attention:
            # Initialize Lightning Indexer
            self.indexer = LightningIndexer(
                LightningIndexerConfig(
                    hidden_size=config.hidden_size,
                    q_lora_rank=config.q_lora_rank,
                    index_n_heads=config.index_n_heads,
                    index_head_dim=config.index_head_dim,
                    index_topk=config.index_topk,
                    qk_rope_head_dim=config.qk_pos_emb_head_dim,
                    rotary_base=config.rotary_base,
                    rotary_scaling_factor=config.rotary_scaling_factor,
                    beta_fast=config.beta_fast,
                    beta_slow=config.beta_slow,
                    mscale=config.mscale,
                    mscale_all_dim=config.mscale_all_dim,
                ),
                layer_number=layer_number,
            )

            # Sparse attention module
            self.sparse_attention = SparseAttention()
        else:
            self.indexer = None
            self.sparse_attention = None

    def forward(self, hidden_states, attention_mask, rotary_pos_emb, ...):
        # Get Q, K, V (existing code)
        query, key, value, q_compressed = self._compute_qkv(hidden_states, rotary_pos_emb)

        if self.use_sparse_attention:
            # Get top-k indices from indexer
            topk_indices = self.indexer(
                hidden_states=hidden_states,
                q_compressed=q_compressed,  # Share compressed Q
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
            )

            # Sparse attention
            context = self.sparse_attention(
                query=query,
                key=key,
                value=value,
                topk_indices=topk_indices,
                attention_mask=attention_mask,
            )
        else:
            # Full attention (existing code)
            context = self.core_attention(query, key, value, attention_mask)

        # Output projection (existing code)
        output = self.linear_proj(context)
        return output
```

---

## 5. Parallelism Strategy

### 5.1 Configuration for 40 H100 GPUs

**Recommended Configuration:**

```python
# For LoRA fine-tuning with 32K context
parallelism_config = {
    "tensor_model_parallel_size": 8,
    "pipeline_model_parallel_size": 5,
    "data_parallel_size": 1,
    "expert_model_parallel_size": 1,  # All experts on each rank for LoRA
    "context_parallel_size": 1,
    "sequence_parallel": True,

    # Memory optimization
    "use_distributed_optimizer": True,
    "overlap_grad_reduce": True,
    "overlap_param_gather": True,
}
```

**Memory Breakdown (per GPU):**

| Component | Size | Notes |
|-----------|------|-------|
| Base model weights (frozen, FP8) | ~17 GB | 685B / 40 GPUs |
| LoRA adapters (BF16) | ~0.5 GB | ~1% of model |
| Optimizer states (LoRA only) | ~2 GB | Adam for LoRA params |
| Activations (32K, checkpointed) | ~20 GB | With gradient checkpointing |
| KV Cache (inference) | ~10 GB | Not needed during training |
| Indexer cache | ~0.5 GB | FP8 K cache for indexer |
| **Total** | **~50 GB** | Fits in 80GB with headroom |

### 5.2 Layer Distribution (PP=5)

```
Stage 0 (GPUs 0-7):   Embedding + Layers 0-11  (12 layers)
Stage 1 (GPUs 8-15):  Layers 12-23             (12 layers)
Stage 2 (GPUs 16-23): Layers 24-35             (12 layers)
Stage 3 (GPUs 24-31): Layers 36-47             (12 layers)
Stage 4 (GPUs 32-39): Layers 48-60 + LM Head   (13 layers)
```

### 5.3 Tensor Parallelism (TP=8)

Within each pipeline stage:
- Attention heads: 128 / 8 = 16 heads per GPU
- MoE experts: 256 / 8 = 32 experts per GPU (but frozen for LoRA)
- Indexer heads: 64 / 8 = 8 heads per GPU

### 5.4 Lightning Indexer Parallelism (TP + EP)

#### Tensor Parallelism for Indexer

The indexer requires careful TP handling because it aggregates scores across heads:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Lightning Indexer TP Layout                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: hidden_states [b, s, hidden]  (replicated across TP)    │
│         q_compressed  [b, s, q_lora_rank] (from sequence-parallel)│
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ wq_b: ColumnParallelLinear                                │   │
│  │   Full:  [q_lora_rank, n_heads * head_dim]               │   │
│  │   Local: [q_lora_rank, n_local_heads * head_dim]         │   │
│  │   → q: [b, s, n_local_heads, head_dim]                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ wk: REPLICATED (not split!)                               │   │
│  │   Shape: [hidden_size, head_dim]                          │   │
│  │   Reason: Single key shared across ALL heads              │   │
│  │   → k: [b, s, head_dim]                                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ k_norm: REPLICATED                                        │   │
│  │   Shape: [head_dim]                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ weights_proj: ColumnParallelLinear                        │   │
│  │   Full:  [hidden_size, n_heads]                          │   │
│  │   Local: [hidden_size, n_local_heads]                    │   │
│  │   → weights: [b, s, n_local_heads]                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Scoring: Local computation                                │   │
│  │   logits = einsum('bshd,btd->bsht', q, k)                │   │
│  │   logits = relu(logits) * softmax_scale                  │   │
│  │   score_local = einsum('bsh,bsht->bst', weights, logits) │   │
│  │   → score_local: [b, s, t] (partial sum over local heads)│   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ALL-REDUCE across TP ranks                                │   │
│  │   index_score = all_reduce(score_local, op=SUM)          │   │
│  │   → index_score: [b, s, t] (full sum over all heads)     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Output: topk_indices [b, s, topk]  (identical on all TP ranks) │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### TP Sharding Summary

| Weight | Parallelism | Reason |
|--------|-------------|--------|
| `wq_b` | Column-parallel | Splits heads across TP ranks |
| `wk` | **Replicated** | Single key shared by all heads |
| `k_norm` | **Replicated** | Applied to replicated k |
| `weights_proj` | Column-parallel | Splits per-head weights |

#### Communication Pattern

```python
# Forward pass communication
def indexer_forward_tp(hidden_states, q_compressed, ...):
    # 1. Local Q projection (no comm)
    q_local = wq_b(q_compressed)  # [b, s, n_local_heads * head_dim]

    # 2. Replicated K projection (no comm, same on all ranks)
    k = k_norm(wk(hidden_states))  # [b, s, head_dim]

    # 3. Local scoring (no comm)
    logits = einsum('bshd,btd->bsht', q_local, k)
    score_local = einsum('bsh,bsht->bst', weights_local, relu(logits))

    # 4. ALL-REDUCE to sum across heads (COMMUNICATION)
    index_score = tensor_model_parallel_all_reduce(score_local)

    # 5. Top-K selection (same on all ranks after all-reduce)
    topk_indices = index_score.topk(k, dim=-1)[1]

    return topk_indices  # Identical across all TP ranks
```

#### Expert Parallelism (EP) - Not Applicable to Indexer

The Lightning Indexer is part of the **attention mechanism**, not the MoE layer:

```
TransformerLayer
├── Attention (contains Indexer)  ← Indexer here, affected by TP only
│   ├── MLA
│   └── LightningIndexer
└── MoE                           ← EP applies here, not to indexer
    ├── Router
    └── Experts (256)
```

**Key Points:**
1. **Indexer is NOT affected by EP** - it runs on all ranks regardless of EP configuration
2. **EP only affects MoE experts** - expert weights are sharded across EP ranks
3. **Indexer output (topk_indices) is used by all ranks** - must be identical

#### Index Consistency Across Ranks

The official code includes an assertion to ensure indices are identical:

```python
# From official HF inference code
topk_indices_ = topk_indices.clone()
dist.broadcast(topk_indices_, src=0)
assert torch.all(topk_indices == topk_indices_), f"{topk_indices=} {topk_indices_=}"
```

For training, we ensure consistency by:
1. Using all-reduce after local head summation
2. Using deterministic top-k (same seed, same result)
3. Running identical computation on replicated weights (wk, k_norm)

#### Sequence Parallelism Interaction

When sequence parallelism is enabled:

```
hidden_states: [b, s/tp, hidden]  (split along sequence)
q_compressed:  [b, s/tp, q_lora_rank]  (from sequence-parallel MLA)
```

The indexer computes scores only for local sequence positions, but needs full sequence K:

```python
# With sequence parallelism
def indexer_forward_sp(hidden_states_local, q_compressed_local, ...):
    # Gather full hidden states for K computation
    hidden_states_full = gather_from_sequence_parallel_region(hidden_states_local)

    # K is computed on full sequence
    k = k_norm(wk(hidden_states_full))  # [b, s, head_dim]

    # Q is local to sequence partition
    q_local = wq_b(q_compressed_local)  # [b, s/tp, n_local_heads, head_dim]

    # Scoring: local Q against full K
    logits = einsum('bshd,btd->bsht', q_local, k)  # [b, s/tp, n_local_heads, s]

    # ... rest same as non-SP
```

---

## 6. Checkpoint Conversion

### 6.1 Weight Mapping: HuggingFace → Megatron

```python
# Checkpoint conversion mapping
HF_TO_MEGATRON = {
    # Embeddings
    "model.embed_tokens.weight": "embedding.word_embeddings.weight",

    # Per-layer mappings (for layer i)
    "model.layers.{i}.input_layernorm.weight":
        "decoder.layers.{i}.input_layernorm.weight",

    # MLA weights
    "model.layers.{i}.self_attn.q_a_proj.weight":
        "decoder.layers.{i}.self_attention.linear_q_down_proj.weight",
    "model.layers.{i}.self_attn.q_a_layernorm.weight":
        "decoder.layers.{i}.self_attention.q_layernorm.weight",
    "model.layers.{i}.self_attn.q_b_proj.weight":
        "decoder.layers.{i}.self_attention.linear_q_up_proj.weight",
    "model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight":
        "decoder.layers.{i}.self_attention.linear_kv_down_proj.weight",
    "model.layers.{i}.self_attn.kv_a_layernorm.weight":
        "decoder.layers.{i}.self_attention.kv_layernorm.weight",
    "model.layers.{i}.self_attn.kv_b_proj.weight":
        "decoder.layers.{i}.self_attention.linear_kv_up_proj.weight",
    "model.layers.{i}.self_attn.o_proj.weight":
        "decoder.layers.{i}.self_attention.linear_proj.weight",

    # NEW: Indexer weights (V3.2 only)
    "model.layers.{i}.self_attn.indexer.q_proj.weight":
        "decoder.layers.{i}.self_attention.indexer.wq_b.weight",
    "model.layers.{i}.self_attn.indexer.k_proj.weight":
        "decoder.layers.{i}.self_attention.indexer.wk.weight",
    "model.layers.{i}.self_attn.indexer.k_layernorm.weight":
        "decoder.layers.{i}.self_attention.indexer.k_norm.weight",
    "model.layers.{i}.self_attn.indexer.k_layernorm.bias":
        "decoder.layers.{i}.self_attention.indexer.k_norm.bias",
    "model.layers.{i}.self_attn.indexer.head_weights.weight":
        "decoder.layers.{i}.self_attention.indexer.weights_proj.weight",

    # MoE weights
    "model.layers.{i}.mlp.gate.weight":
        "decoder.layers.{i}.mlp.router.weight",
    "model.layers.{i}.mlp.experts.{e}.gate_proj.weight":
        "decoder.layers.{i}.mlp.experts.local_experts.{e}.linear_fc1.weight",
    # ... etc

    # LM Head
    "lm_head.weight": "output_layer.weight",
}
```

### 6.2 Conversion Script Outline

```python
# tools/checkpoint_conversion/deepseek_v32_to_megatron.py

def convert_checkpoint(
    hf_checkpoint_path: str,
    megatron_checkpoint_path: str,
    tp_size: int = 8,
    pp_size: int = 5,
):
    """
    Convert DeepSeek V3.2 HuggingFace checkpoint to Megatron format.

    Steps:
    1. Load HF checkpoint shards
    2. Map weight names
    3. Reshape for TP/PP sharding
    4. Save in Megatron format
    """
    # Load HF config
    config = load_hf_config(hf_checkpoint_path)

    # Verify V3.2
    assert config.get("index_topk") is not None, "Not a V3.2 checkpoint"

    # Process each shard
    for shard_path in get_checkpoint_shards(hf_checkpoint_path):
        state_dict = load_safetensors(shard_path)

        for hf_name, tensor in state_dict.items():
            megatron_name = map_weight_name(hf_name)

            # Handle TP sharding
            if needs_tp_shard(megatron_name):
                tensor = shard_for_tp(tensor, tp_size, megatron_name)

            # Handle PP placement
            pp_rank = get_pp_rank(megatron_name, pp_size)

            save_weight(megatron_checkpoint_path, pp_rank, megatron_name, tensor)
```

---

## 7. LoRA Integration

### 7.1 Target Modules for V3.2 LoRA

```python
# LoRA configuration for DeepSeek V3.2
lora_config = {
    "r": 64,                    # LoRA rank
    "alpha": 128,               # Scaling factor
    "dropout": 0.05,            # Dropout rate
    "target_modules": [
        # MLA Q path
        "linear_q_down_proj",   # q_a_proj
        "linear_q_up_proj",     # q_b_proj

        # MLA KV path
        "linear_kv_down_proj",  # kv_a_proj
        "linear_kv_up_proj",    # kv_b_proj

        # MLA output
        "linear_proj",          # o_proj

        # Indexer (optional - can freeze for stability)
        "indexer.wq_b",
        "indexer.wk",
        "indexer.weights_proj",

        # MoE (optional - typically frozen)
        # "router.weight",
        # "experts.*.linear_fc1",
        # "experts.*.linear_fc2",
    ],

    # Modules to freeze
    "freeze_modules": [
        "embedding",
        "experts",              # Freeze MoE experts
        "router",               # Freeze router
        "layernorm",            # Freeze norms
    ],
}
```

### 7.2 Memory Comparison

| Fine-tuning Method | Trainable Params | Memory (40 GPUs) | Feasibility |
|-------------------|------------------|------------------|-------------|
| Full fine-tuning | 685B | ~8.2 TB | ❌ Won't fit |
| LoRA r=64 (all modules) | ~6.8B (~1%) | ~200 GB | ✅ Comfortable |
| LoRA r=64 (MLA only) | ~2.5B (~0.4%) | ~100 GB | ✅ Very comfortable |
| LoRA r=64 (MLA + Indexer) | ~3.4B (~0.5%) | ~120 GB | ✅ Comfortable |

### 7.3 LoRA Module Implementation

```python
# megatron/core/transformer/lora.py

class LoRALinear(nn.Module):
    """
    LoRA adapter for linear layers.

    output = W @ x + (B @ A) @ x * (alpha / r)

    Where:
    - W: frozen base weights
    - A: down-projection [in, r]
    - B: up-projection [r, out]
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 64,
        alpha: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Freeze base weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # LoRA matrices
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

| Test | Description | File |
|------|-------------|------|
| `test_lightning_indexer_forward` | Verify indexer output shapes and values | `tests/unit_tests/transformer/test_lightning_indexer.py` |
| `test_lightning_indexer_backward` | Verify gradients flow correctly | `tests/unit_tests/transformer/test_lightning_indexer.py` |
| `test_sparse_attention_forward` | Verify sparse attention computation | `tests/unit_tests/transformer/test_sparse_attention.py` |
| `test_sparse_attention_backward` | Verify sparse attention gradients | `tests/unit_tests/transformer/test_sparse_attention.py` |
| `test_rope_interleaved_vs_non` | Verify RoPE layout handling | `tests/unit_tests/transformer/test_rope.py` |
| `test_mla_with_indexer` | End-to-end MLA with sparse attention | `tests/unit_tests/transformer/test_multi_latent_attention.py` |

### 8.1.1 TP/EP Specific Tests

| Test | Description | Config |
|------|-------------|--------|
| `test_indexer_tp_consistency` | Verify identical indices across TP ranks | TP=2, compare rank outputs |
| `test_indexer_tp1_vs_tp2` | Verify TP=1 and TP=2 produce same indices | Compare outputs numerically |
| `test_indexer_wk_gradient_allreduce` | Verify replicated wk gets correct gradients | TP=2, check gradient sync |
| `test_indexer_sp_interaction` | Verify sequence parallel gathers correctly | TP=2, SP=True |
| `test_indexer_ep_independence` | Verify indexer unaffected by EP config | EP=1 vs EP=2, same indices |

```python
# Example: test_indexer_tp_consistency.py
def test_indexer_tp_consistency():
    """Ensure all TP ranks compute identical top-k indices."""
    # Setup TP=2
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)

    indexer = LightningIndexer(config, layer_number=0)
    hidden = torch.randn(batch, seq, hidden_size, device='cuda')
    q_compressed = torch.randn(batch, seq, q_lora_rank, device='cuda')

    # Forward
    topk_indices = indexer(hidden, q_compressed, freqs_cis)

    # Broadcast from rank 0 and compare (same as official code)
    topk_indices_ref = topk_indices.clone()
    torch.distributed.broadcast(topk_indices_ref, src=0)

    assert torch.all(topk_indices == topk_indices_ref), \
        f"TP rank {get_tensor_model_parallel_rank()} has different indices!"


def test_indexer_tp1_vs_tp2():
    """Verify TP=2 produces same output as TP=1."""
    # Run with TP=1
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=1)
    indexer_tp1 = LightningIndexer(config_tp1, layer_number=0)
    indices_tp1 = indexer_tp1(hidden, q_compressed, freqs_cis)

    # Run with TP=2
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    indexer_tp2 = LightningIndexer(config_tp2, layer_number=0)
    # Load sharded weights from TP=1 model
    load_tp_sharded_weights(indexer_tp2, indexer_tp1)
    indices_tp2 = indexer_tp2(hidden, q_compressed, freqs_cis)

    # Compare on rank 0
    if get_tensor_model_parallel_rank() == 0:
        assert torch.all(indices_tp1 == indices_tp2), "TP=1 vs TP=2 mismatch!"
```

### 8.2 Functional Tests

| Test | Description | Config |
|------|-------------|--------|
| `test_v32_forward_match` | Compare forward with HF reference | TP=1, PP=1, batch=1, seq=2048 |
| `test_v32_checkpoint_load` | Verify checkpoint conversion | Full model load |
| `test_v32_lora_training` | One training step with LoRA | TP=2, PP=1, batch=1, seq=1024 |
| `test_v32_distributed` | Multi-GPU training | TP=8, PP=5, batch=1, seq=4096 |

### 8.3 Validation Approach

```python
def validate_forward_pass():
    """
    Compare Megatron V3.2 output with reference implementation.

    Steps:
    1. Load same weights in both implementations
    2. Run forward pass with same input
    3. Compare outputs (allow small numerical diff)
    """
    # Load Megatron model
    megatron_model = load_megatron_v32(checkpoint_path)

    # Load reference (HF inference code)
    reference_model = load_hf_v32_inference(checkpoint_path)

    # Same input
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    # Forward pass
    megatron_out = megatron_model(input_ids)
    reference_out = reference_model(input_ids)

    # Compare
    diff = (megatron_out - reference_out).abs().max()
    assert diff < 1e-4, f"Output mismatch: {diff}"
```

---

## 9. Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| RoPE layout mismatch | Silent accuracy loss | Explicit unit tests comparing interleaved vs non-interleaved |
| Memory OOM at 32K | Training fails | Start with 16K, gradually increase with profiling |
| Checkpoint weight name mismatch | Loading fails | Verify against HF inference code naming |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Sparse attention gradient issues | Training diverges | Compare gradients numerically with finite differences |
| TP sharding bugs for indexer | Incorrect output | Unit tests with TP=1 vs TP=2 comparison |
| Pipeline imbalance | Slow training | Profile and adjust layer distribution |
| Index inconsistency across TP ranks | Silent correctness bug | Add assertion like official code: broadcast + compare |
| Sequence parallelism + indexer interaction | Wrong indices for SP tokens | Test SP=True vs SP=False output match |
| Replicated wk gradient accumulation | Incorrect gradients | Ensure all-reduce on wk gradients across TP |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| LoRA performance regression | Slower than expected | Benchmark and tune LoRA rank |
| Numerical precision | Minor quality loss | Use BF16 throughout, FP32 for critical ops |

---

## 10. Timeline

### Week 1-2: Foundation

- **Week 1**
  - [ ] Day 1-2: Set up development environment, study V3.2 reference code
  - [ ] Day 3-4: Implement `LightningIndexerConfig` and add to `MLATransformerConfig`
  - [ ] Day 5: Implement `LightningIndexer` module (forward pass)

- **Week 2**
  - [ ] Day 1-2: Implement non-interleaved RoPE for indexer
  - [ ] Day 3-4: Implement `SparseAttention` module
  - [ ] Day 5: Unit tests for indexer and sparse attention

### Week 3-4: Integration

- **Week 3**
  - [ ] Day 1-2: Integrate indexer into `MLASelfAttention`
  - [ ] Day 3-4: Handle shared q_compressed between MLA and indexer
  - [ ] Day 5: End-to-end forward pass test

- **Week 4**
  - [ ] Day 1-3: Checkpoint conversion script
  - [ ] Day 4-5: Validate forward pass matches HF reference

### Week 5-6: LoRA Support

- **Week 5**
  - [ ] Day 1-2: Implement LoRA wrapper for V3.2 modules
  - [ ] Day 3-4: Integrate with Megatron LoRA infrastructure
  - [ ] Day 5: Test LoRA adapter creation and forward pass

- **Week 6**
  - [ ] Day 1-2: LoRA training loop integration
  - [ ] Day 3-4: Gradient checkpointing for indexer
  - [ ] Day 5: Memory profiling

### Week 7-8: Training Pipeline

- **Week 7**
  - [ ] Day 1-2: Configure 40-GPU parallelism
  - [ ] Day 3-4: End-to-end training test (small scale)
  - [ ] Day 5: Debug and fix distributed issues

- **Week 8**
  - [ ] Day 1-3: Scale to 32K context
  - [ ] Day 4-5: Fine-tuning experiment on validation dataset

### Week 9-10: Optimization & Documentation

- **Week 9**
  - [ ] Day 1-3: Integrate TileLang sparse attention backward (optional)
  - [ ] Day 4-5: Performance benchmarking

- **Week 10**
  - [ ] Day 1-2: Optimization based on profiling
  - [ ] Day 3-4: Documentation
  - [ ] Day 5: Code review and cleanup

---

## Appendix A: File Structure

```
megatron/core/
├── transformer/
│   ├── lightning_indexer.py          # NEW
│   ├── sparse_attention.py           # NEW
│   ├── multi_latent_attention.py     # MODIFIED
│   └── transformer_config.py         # MODIFIED (add V3.2 config)
├── models/
│   └── gpt/
│       └── gpt_layer_specs.py        # MODIFIED (add indexer to spec)
└── fusions/
    └── fused_sparse_attn.py          # NEW (optional optimization)

tools/
└── checkpoint_conversion/
    └── deepseek_v32_to_megatron.py   # NEW

tests/
└── unit_tests/
    └── transformer/
        ├── test_lightning_indexer.py  # NEW
        └── test_sparse_attention.py   # NEW

examples/
└── deepseek_v32/
    ├── lora_finetune_32k.sh          # NEW
    └── README.md                      # NEW
```

---

## Appendix B: Training Command Example

```bash
#!/bin/bash
# examples/deepseek_v32/lora_finetune_32k.sh

CHECKPOINT_PATH=/path/to/megatron/checkpoint
DATA_PATH=/path/to/training/data
OUTPUT_PATH=/path/to/output

torchrun --nproc_per_node=8 --nnodes=5 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=6000 \
    pretrain_gpt.py \
    \
    # Model config
    --num-layers 61 \
    --hidden-size 7168 \
    --num-attention-heads 128 \
    --seq-length 32768 \
    --max-position-embeddings 163840 \
    \
    # MLA config
    --multi-latent-attention \
    --q-lora-rank 1536 \
    --kv-lora-rank 512 \
    --qk-head-dim 128 \
    --qk-pos-emb-head-dim 64 \
    --v-head-dim 128 \
    \
    # V3.2 Indexer config (NEW)
    --index-n-heads 64 \
    --index-head-dim 128 \
    --index-topk 2048 \
    \
    # MoE config
    --num-experts 256 \
    --moe-router-topk 8 \
    --moe-shared-expert-intermediate-size 2048 \
    \
    # RoPE config
    --position-embedding-type rope \
    --rotary-base 10000 \
    --rotary-scaling-factor 40 \
    --mscale 1.0 \
    \
    # Parallelism
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 5 \
    --sequence-parallel \
    --use-distributed-optimizer \
    \
    # LoRA config
    --lora-r 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --lora-target-modules "linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,indexer.wq_b,indexer.wk" \
    \
    # Training config
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --train-iters 10000 \
    --lr 1e-4 \
    --lr-decay-style cosine \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    \
    # Memory optimization
    --recompute-granularity selective \
    --recompute-modules "mla_up_proj,mlp" \
    --bf16 \
    \
    # Paths
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --save $OUTPUT_PATH \
    --save-interval 1000
```

---

## Appendix C: Quick Reference

### Config Detection

```python
# Check if model is V3.2
def is_deepseek_v32(config):
    return (
        hasattr(config, 'index_topk') and
        config.index_topk is not None and
        config.index_topk > 0
    )
```

### RoPE Layout

```python
# MLA: interleaved RoPE
# x = [x0, x1, x2, x3, x4, x5, x6, x7]
# rotate pairs: (x0,x1), (x2,x3), (x4,x5), (x6,x7)

# Indexer: non-interleaved RoPE
# x = [x0, x1, x2, x3, x4, x5, x6, x7]
# rotate halves: (x0,x4), (x1,x5), (x2,x6), (x3,x7)
```

### Memory Formula

```python
# Per-GPU memory estimate for LoRA fine-tuning
def estimate_memory(seq_len, batch_size, tp_size, pp_size):
    model_params = 685e9
    lora_params = model_params * 0.01  # 1% for LoRA

    # Weights (frozen, FP8)
    weights_gb = (model_params / (tp_size * pp_size)) * 1 / 1e9

    # LoRA + optimizer
    lora_gb = (lora_params / (tp_size * pp_size)) * 12 / 1e9  # BF16 + Adam

    # Activations (with checkpointing)
    act_gb = (seq_len * batch_size * 7168 * 4) / 1e9  # Rough estimate

    return weights_gb + lora_gb + act_gb
```

---

## References and Sources

### Local Reference Files (Downloaded)

The official DeepSeek V3.2 inference code has been downloaded for reference:

```
docs/reference/deepseek_v32_official/
├── README.md       # Documentation and usage notes
├── model.py        # 922 lines - Main model (Transformer, MLA, Indexer, MoE)
├── kernel.py       # 274 lines - TileLang kernels (fp8_index, act_quant)
└── generate.py     # 186 lines - Text generation script
```

**Key sections in `model.py`:**
- `class Indexer` (lines ~600-700): Lightning Indexer implementation
- `class MLA` (lines ~400-600): Multi-Latent Attention with sparse attention
- `apply_rotary_emb()`: RoPE with interleaved flag

**Key sections in `kernel.py`:**
- `fp8_index` kernel: Scoring formula and top-k selection
- `act_quant`: FP8 quantization (inference only)

### Official DeepSeek V3.2 Resources

| Resource | URL | Used For |
|----------|-----|----------|
| HuggingFace Checkpoint | https://huggingface.co/deepseek-ai/DeepSeek-V3.2 | Config, weights, reference code |
| Official Inference Code | https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py | Indexer implementation reference |
| Kernel Implementations | https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/kernel.py | FP8 kernels, scoring formula |
| V3.2 Technical Report | https://api-docs.deepseek.com/news/news251201 | Architecture details |

### Third-Party V3.2 Implementations

| Framework | URL | Key Insight |
|-----------|-----|-------------|
| vLLM | https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html | Hadamard transform removal confirmed |
| SGLang | https://github.com/sgl-project/sglang/pull/11061 | Day-0 support patterns |
| vLLM PR | https://github.com/vllm-project/vllm/pull/25869 | V3.2 detection via `index_topk` |

### Megatron-LM Files Analyzed

| File | Contains |
|------|----------|
| `megatron/core/transformer/multi_latent_attention.py` | Existing MLA implementation |
| `megatron/core/transformer/moe/experts.py` | GroupedMLP, SequentialMLP |
| `megatron/core/transformer/moe/token_dispatcher.py` | DeepEP, HybridEP, Flex dispatchers |
| `megatron/core/transformer/moe/fused_a2a.py` | Fused all-to-all operations |
| `megatron/core/transformer/transformer_config.py` | MLATransformerConfig |

### Key Technical References

1. **Non-interleaved vs Interleaved RoPE**: The indexer uses non-interleaved RoPE (`apply_rotary_emb(x, freqs_cis, interleaved=False)`) while MLA uses interleaved. This is a critical correctness requirement.

2. **Indexer Scoring Formula** (from official `fp8_index` kernel):
   ```
   logits = relu(q @ k.T) * softmax_scale
   index_score = Σ_h(weights[h] * logits[h]) * k_scale
   ```

3. **Grouped GEMM**: Megatron uses `grouped_gemm` library that batches all expert computations into O(1) kernel launches, unlike the O(num_experts) for-loop in reference implementations.
