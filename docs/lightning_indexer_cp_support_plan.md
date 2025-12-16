# Lightning Indexer Context Parallelism (CP) Support Plan

## Overview

This document outlines the implementation plan for adding Context Parallelism (CP) support to the Lightning Indexer in DeepSeek V3.2's Megatron implementation.

### Problem Statement

The Lightning Indexer computes top-K token indices for sparse attention. With CP enabled:
- Each CP rank only has a portion of the sequence (e.g., 32K of 128K tokens)
- The indexer currently computes top-K only within the local chunk
- This results in **local** sparse selection instead of **global** optimal selection

### Proposed Solution

All-gather hidden states before the indexer so each CP rank computes identical global top-K indices. The MLA attention continues to use ring attention with CP.

---

## Design Decision: All-Gather + Redundant Compute

### Two Approaches Considered

| Approach | Description |
|----------|-------------|
| **Option A: All-Gather + Redundant Compute** | All CP ranks all-gather hidden states, all ranks compute indices (redundant but identical) |
| **Option B: Compute on Rank 0 + Broadcast** | Only rank 0 all-gathers and computes, then broadcasts indices to other ranks |

### Why We Chose Option A

**1. Same code path on all ranks = fewer bugs**

```python
# Option A: Simple and uniform
hidden_full = all_gather(hidden_local)   # All ranks
indices = compute_indices(hidden_full)   # All ranks (same input → same output)

# Option B: Branching logic
if cp_rank == 0:
    hidden_full = all_gather(hidden_local)
    indices = compute_indices(hidden_full)
else:
    indices = empty_tensor()  # Edge cases: shape, dtype, device
broadcast(indices, src=0)
```

**2. Backward pass works naturally**

The indexer weights (`linear_wq_b`, `linear_wk`, `linear_weights_proj`) need gradients for training.

| Approach | Gradient Flow |
|----------|---------------|
| **Option A** | All CP ranks compute gradients through indexer → automatic, correct |
| **Option B** | Only rank 0 computes → need manual gradient sync across CP ranks |

With Option B, you'd need additional gradient synchronization:
```python
# Extra complexity required for Option B
if cp_size > 1:
    for param in indexer.parameters():
        torch.distributed.all_reduce(param.grad, group=cp_group)
```

Option A avoids this entirely - gradients naturally work because all ranks compute the same forward pass on identical data.

**3. Autograd is transparent**

- Option A: autograd traces through `all_gather → indexer → indices` seamlessly
- Option B: `broadcast` is non-differentiable, requires manual gradient handling

**4. Easier to verify and debug**

- Option A: Verify that all-gather gives full sequence, existing indexer code handles the rest
- Option B: Verify all-gather, broadcast, empty tensor shapes, gradient handling, async timing...

### Trade-off Accepted

| Factor | Option A | Option B |
|--------|----------|----------|
| Compute | Redundant (CP× indexer ops) | Single (only rank 0) |
| Communication | 1× all-gather | 1× all-gather + 1× broadcast |
| **Correctness risk** | **Low** | **Medium** |
| **Implementation complexity** | **Low** | **Medium-High** |

**Decision:** Accept redundant compute for correctness and simplicity. The indexer compute (~1B ops per layer at 128K) is small compared to MLA attention. Can optimize later if profiling shows it's a bottleneck.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Current Flow (CP > 1, BROKEN)                                          │
│                                                                          │
│  Rank 0: tokens [0:32K]     Rank 1: tokens [32K:64K]    ...             │
│       │                           │                                      │
│       ▼                           ▼                                      │
│  Indexer (local)             Indexer (local)                            │
│  top-K from [0:32K]          top-K from [32K:64K]                       │
│       │                           │                                      │
│       ▼                           ▼                                      │
│  Different indices!          Different indices!   ← PROBLEM             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Proposed Flow (CP > 1, CORRECT)                                        │
│                                                                          │
│  Rank 0: tokens [0:32K]     Rank 1: tokens [32K:64K]    ...             │
│       │                           │                                      │
│       └───────────┬───────────────┘                                      │
│                   ▼                                                      │
│           ALL-GATHER (hidden_states, q_compressed)                       │
│                   │                                                      │
│       ┌───────────┴───────────┐                                          │
│       ▼                       ▼                                          │
│  Indexer (full seq)      Indexer (full seq)                             │
│  top-K from [0:128K]     top-K from [0:128K]                            │
│       │                       │                                          │
│       ▼                       ▼                                          │
│  IDENTICAL indices       IDENTICAL indices   ← CORRECT                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Core CP Support in Lightning Indexer

**File:** `megatron/core/transformer/lightning_indexer.py`

#### Step 1.1: Add CP Communication Imports

```python
# Add to imports section (around line 10-20)
from megatron.core.parallel_state import (
    get_context_parallel_world_size,
    get_context_parallel_rank,
    get_context_parallel_group,
)
from megatron.core.tensor_parallel.mappings import (
    _gather_along_first_dim,  # For all-gather
)
```

#### Step 1.2: Create Helper Functions

```python
# Add after imports, before class definition

def _all_gather_from_cp_region(tensor: torch.Tensor, cp_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    """All-gather tensor from all CP ranks along sequence dimension (dim 0).

    Args:
        tensor: Input tensor of shape [local_seq, batch, ...]
        cp_group: Context parallel process group

    Returns:
        Gathered tensor of shape [full_seq, batch, ...]
    """
    cp_size = get_context_parallel_world_size()
    if cp_size == 1:
        return tensor

    # Gather along first dimension (sequence)
    gathered = _gather_along_first_dim(tensor, cp_group)
    return gathered


def _get_full_rotary_pos_emb(rotary_pos_emb: torch.Tensor, full_seq_len: int) -> torch.Tensor:
    """Get full rotary position embeddings for the entire sequence.

    Args:
        rotary_pos_emb: Rotary embeddings (may be pre-computed for full seq or longer)
        full_seq_len: Full sequence length across all CP ranks

    Returns:
        Rotary embeddings for full sequence [full_seq_len, ...]
    """
    return rotary_pos_emb[:full_seq_len]
```

#### Step 1.3: Modify `forward()` Method

**Location:** `LightningIndexer.forward()` (around line 200-230)

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    q_compressed: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute top-K token indices for sparse attention.

    With CP enabled, all-gathers hidden states to compute global top-K.
    """
    local_seq_len, batch_size, _ = hidden_states.size()
    cp_size = get_context_parallel_world_size()

    # === CP Handling: All-gather for global indexing ===
    if cp_size > 1:
        cp_group = get_context_parallel_group()

        # All-gather hidden states from all CP ranks
        # [local_seq, batch, hidden] -> [full_seq, batch, hidden]
        hidden_states_full = _all_gather_from_cp_region(hidden_states, cp_group)
        q_compressed_full = _all_gather_from_cp_region(q_compressed, cp_group)

        full_seq_len = hidden_states_full.size(0)

        # Get full rotary embeddings
        rotary_pos_emb_full = _get_full_rotary_pos_emb(rotary_pos_emb, full_seq_len)

        # Expand attention mask if provided
        if attention_mask is not None:
            # attention_mask should already be for full sequence in most cases
            # but verify dimensions match
            attention_mask_full = attention_mask
        else:
            attention_mask_full = None
    else:
        hidden_states_full = hidden_states
        q_compressed_full = q_compressed
        rotary_pos_emb_full = rotary_pos_emb
        full_seq_len = local_seq_len
        attention_mask_full = attention_mask

    # === Compute indices on full sequence ===
    # All CP ranks compute identical indices (redundant but correct)
    if full_seq_len > self.chunk_threshold:
        topk_indices = self._forward_chunked(
            hidden_states_full,
            q_compressed_full,
            rotary_pos_emb_full,
            attention_mask_full,
        )
    else:
        topk_indices = self._forward_simple(
            hidden_states_full,
            q_compressed_full,
            rotary_pos_emb_full,
            attention_mask_full,
        )

    # topk_indices: [batch, full_seq, topk]
    # Each CP rank has identical global indices
    return topk_indices
```

#### Step 1.4: Update `_forward_simple()` for Full Sequence

**Location:** Around line 240-320

Remove the CP-specific RoPE handling since we're now working on full sequence:

```python
def _forward_simple(
    self,
    hidden_states: torch.Tensor,
    q_compressed: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simple forward pass for sequences that fit in memory.

    Note: With CP, this receives the FULL sequence (already all-gathered).
    """
    seq_len, batch_size, _ = hidden_states.size()

    # === Project Q and K ===
    q, _ = self.linear_wq_b(q_compressed)
    q = q.view(seq_len, batch_size, self.n_local_heads, self.index_head_dim)

    k = self.linear_wk(hidden_states)
    k = self.k_layernorm(k)

    # Split into rope and non-rope portions
    q_pe = q[..., : self.qk_rope_head_dim]
    q_nope = q[..., self.qk_rope_head_dim :]
    k_pe = k[..., : self.qk_rope_head_dim]
    k_nope = k[..., self.qk_rope_head_dim :]

    # Get RoPE frequencies for FULL sequence
    # (No CP rank adjustment needed - we have full sequence)
    freqs_cis = rotary_pos_emb[:seq_len]

    # Apply non-interleaved RoPE
    q_pe = apply_rotary_emb_non_interleaved(q_pe, freqs_cis)
    k_pe = apply_rotary_emb_non_interleaved(k_pe.unsqueeze(2), freqs_cis).squeeze(2)

    # ... rest of _forward_simple unchanged ...
```

#### Step 1.5: Update `_forward_chunked()` Similarly

Apply the same changes to `_forward_chunked()` - remove CP-specific RoPE handling since input is already full sequence.

---

### Phase 2: Integration with MLA Attention

**File:** `megatron/core/transformer/multi_latent_attention.py`

The MLA attention needs to use the global indices correctly with CP.

#### Step 2.1: Verify Index Usage

The top-K indices from the indexer are global (full sequence indices). When MLA uses these for sparse gather:

```python
# In MLA attention, when using indices:
# indices are [batch, full_seq, topk] - global positions

# Need to adjust for local CP chunk when gathering KV
if cp_size > 1:
    cp_rank = get_context_parallel_rank()
    local_start = cp_rank * local_seq_len
    local_end = local_start + local_seq_len

    # Filter indices to those in local range, or use ring attention
    # to gather KV from other ranks
```

**Note:** This may already be handled by the existing ring attention implementation. Verify during testing.

---

### Phase 3: Memory Optimization (Optional)

#### Step 3.1: Gradient Checkpointing for Indexer

If memory is tight, the indexer all-gather can be checkpointed:

```python
from torch.utils.checkpoint import checkpoint

def forward(self, hidden_states, q_compressed, rotary_pos_emb, attention_mask):
    if self.config.recompute_granularity == "full":
        return checkpoint(
            self._forward_with_cp,
            hidden_states, q_compressed, rotary_pos_emb, attention_mask,
            use_reentrant=False
        )
    else:
        return self._forward_with_cp(
            hidden_states, q_compressed, rotary_pos_emb, attention_mask
        )
```

#### Step 3.2: Release Gathered Tensors Early

```python
# After computing indices, explicitly free gathered tensors
del hidden_states_full, q_compressed_full
torch.cuda.empty_cache()  # Optional, may hurt performance
```

---

### Phase 4: Testing

#### Step 4.1: Unit Test for CP Correctness

**File:** `tests/unit_tests/transformer/test_lightning_indexer_cp.py`

```python
import pytest
import torch
from megatron.core import parallel_state
from megatron.core.transformer.lightning_indexer import LightningIndexer

@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_indexer_cp_produces_identical_indices(cp_size):
    """Verify all CP ranks produce identical top-K indices."""
    # Initialize parallel state with CP
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        context_parallel_size=cp_size,
    )

    # Create indexer
    config = get_test_config()
    indexer = LightningIndexer(config)

    # Generate test inputs (same across ranks for verification)
    torch.manual_seed(42)
    full_seq_len = 8192
    local_seq_len = full_seq_len // cp_size
    batch_size = 2

    # Each rank gets its portion
    cp_rank = parallel_state.get_context_parallel_rank()
    start = cp_rank * local_seq_len
    end = start + local_seq_len

    hidden_states_full = torch.randn(full_seq_len, batch_size, config.hidden_size)
    hidden_states_local = hidden_states_full[start:end].cuda()

    # Run indexer
    indices = indexer(hidden_states_local, q_compressed_local, rotary_emb, mask)

    # All-gather indices from all ranks and verify they're identical
    all_indices = [torch.empty_like(indices) for _ in range(cp_size)]
    torch.distributed.all_gather(all_indices, indices)

    for i in range(1, cp_size):
        assert torch.equal(all_indices[0], all_indices[i]), \
            f"Rank 0 and rank {i} produced different indices!"

    parallel_state.destroy_model_parallel()
```

#### Step 4.2: Integration Test with Full Model

```bash
# Test V3.2 with CP=2 on 8 GPUs
torchrun --nproc_per_node=8 tests/integration/test_v32_forward.py \
    --tensor-model-parallel-size 4 \
    --context-parallel-size 2 \
    --seq-length 8192 \
    --verify-indexer-consistency
```

#### Step 4.3: Memory Profiling

```python
# Profile memory with and without CP
def profile_indexer_memory(cp_size, seq_len):
    torch.cuda.reset_peak_memory_stats()

    # Run forward pass
    indices = indexer(hidden_states, q_compressed, rotary_emb, mask)

    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"CP={cp_size}, seq_len={seq_len}: {peak_memory:.2f} GB peak")
```

---

## Memory Analysis

### Per-Layer Memory Overhead from All-Gather

| Sequence Length | Hidden Size | CP Size | All-Gather Size | Notes |
|-----------------|-------------|---------|-----------------|-------|
| 32K | 5120 | 2 | 0.33 GB | hidden_states only |
| 64K | 5120 | 2 | 0.66 GB | |
| 128K | 5120 | 2 | 1.31 GB | |
| 128K | 5120 | 4 | 1.31 GB | Same - full seq on each rank |

**Formula:** `all_gather_size = seq_len × hidden_size × 2 bytes (bf16)`

### Trade-off Analysis

| Approach | Memory | Compute | Correctness |
|----------|--------|---------|-------------|
| No CP | Baseline | Baseline | ✅ Global top-K |
| CP without fix | -50% seq memory | Same | ❌ Local top-K |
| CP with all-gather (proposed) | -40% seq memory | +5% (redundant indexer) | ✅ Global top-K |

---

## Rollout Plan

### Week 1: Implementation
- [ ] Implement `_all_gather_from_cp_region()` helper
- [ ] Modify `forward()` with CP handling
- [ ] Update `_forward_simple()` and `_forward_chunked()`
- [ ] Add config flag `indexer_ignore_cp: bool = False` for gradual rollout

### Week 2: Testing
- [ ] Unit tests for CP correctness
- [ ] Integration tests with V3.2 model
- [ ] Memory profiling at various sequence lengths
- [ ] Verify numerical equivalence: CP=1 vs CP>1 should produce identical indices

### Week 3: Optimization & Documentation
- [ ] Profile and optimize all-gather communication
- [ ] Add gradient checkpointing option if needed
- [ ] Update documentation
- [ ] Code review and merge

---

## Open Questions

1. **Attention mask handling:** Does the attention mask need adjustment for CP? Likely already handled at the model level.

2. **Backward pass:** The all-gather in forward needs corresponding reduce-scatter in backward. Verify autograd handles this correctly.

3. **Ring attention integration:** How do the global indices interact with ring attention's KV communication? May need coordination.

4. **Performance:** Is the redundant indexer compute on each CP rank acceptable? Alternative: compute on rank 0, broadcast indices.

---

## Alternative: Broadcast Indices Instead of Redundant Compute

Instead of all CP ranks computing identical indices:

```python
if cp_size > 1:
    cp_rank = get_context_parallel_rank()

    if cp_rank == 0:
        # Only rank 0 computes indices
        hidden_states_full = all_gather(hidden_states)
        indices = self._compute_indices(hidden_states_full)
    else:
        indices = torch.empty(batch, full_seq, topk, device=device, dtype=torch.long)

    # Broadcast from rank 0 to all CP ranks
    torch.distributed.broadcast(indices, src=0, group=cp_group)
```

**Trade-off:** Less compute, more communication (broadcast vs all-gather). May be better for very large indexer.

---

## References

- [Megatron Context Parallelism Docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html)
- [Ring Attention Paper](https://arxiv.org/abs/2310.01889)
- `megatron/core/tensor_parallel/mappings.py` - All-gather implementations
- `megatron/core/models/common/embeddings/rope_utils.py` - CP RoPE handling
