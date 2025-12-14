# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lightning Indexer for DeepSeek V3.2 sparse attention.

This module implements the Lightning Indexer which selects top-K tokens for sparse
attention, reducing attention complexity from O(L^2) to O(L*K).

Reference: DeepSeek V3.2 official inference code
https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from megatron.core.tensor_parallel import (
    ColumnParallelLinear,
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.mappings import _reduce
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class LightningIndexerSubmodules:
    """Submodule specifications for Lightning Indexer."""

    linear_wq_b: Union[ModuleSpec, type] = ColumnParallelLinear
    """Q projection from compressed representation."""

    linear_wk: Union[ModuleSpec, type] = None
    """K projection - uses regular Linear (replicated across TP)."""

    k_layernorm: Union[ModuleSpec, type] = None
    """LayerNorm for K - replicated across TP."""

    linear_weights_proj: Union[ModuleSpec, type] = ColumnParallelLinear
    """Per-head aggregation weights."""


def apply_rotary_emb_non_interleaved(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor:
    """
    Apply rotary embeddings with NON-interleaved layout.

    Non-interleaved: rotate halves (x0,x4), (x1,x5), (x2,x6), (x3,x7)
    Interleaved (standard): rotate pairs (x0,x1), (x2,x3), (x4,x5), (x6,x7)

    This matches the official DeepSeek V3.2 indexer RoPE implementation:
    apply_rotary_emb(x, freqs_cis, interleaved=False)

    Args:
        x: Input tensor of shape [..., head_dim] where head_dim has RoPE dimensions
        freqs_cis: Complex frequencies of shape [seq_len, rope_dim // 2]

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    dtype = x.dtype
    shape = x.shape

    # Non-interleaved: reshape to put halves together
    # [*, d] -> [*, 2, d//2] -> [*, d//2, 2]
    x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()

    # Convert to complex for rotation
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))

    # Expand freqs_cis to match x dimensions
    # freqs_cis: [seq, rope_dim//2] -> [1, seq, 1, rope_dim//2] for BSHD format
    if x.dim() == 4:  # [batch, seq, heads, rope_dim//2]
        freqs_cis = freqs_cis.view(1, freqs_cis.size(0), 1, freqs_cis.size(-1))
    elif x.dim() == 3:  # [batch, seq, rope_dim//2]
        freqs_cis = freqs_cis.view(1, freqs_cis.size(0), freqs_cis.size(-1))

    # Apply rotation
    y = torch.view_as_real(x * freqs_cis).flatten(-2)

    # Undo non-interleaved layout: [*, d//2, 2] -> [*, d]
    y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)

    return y.to(dtype)


class LightningIndexer(MegatronModule):
    """
    DeepSeek V3.2 Lightning Indexer for sparse attention token selection.

    The indexer computes attention-like scores to select the top-K most relevant
    tokens for each query position, enabling sparse attention that reduces
    complexity from O(L^2) to O(L*K).

    Key design decisions for training:
    - Uses BF16 instead of FP8 (for gradient flow)
    - Uses PyTorch ops instead of TileLang kernels (for autograd)
    - No Hadamard transform (vLLM confirmed unnecessary for accuracy)
    - No KV cache (training recomputes each forward)

    Tensor Parallel handling:
    - wq_b: ColumnParallel (splits heads across TP)
    - wk: REPLICATED (single key shared by all heads)
    - k_norm: REPLICATED
    - weights_proj: ColumnParallel (splits per-head weights)
    - After scoring: ALL-REDUCE to sum partial head scores
    - Top-K produces identical indices on all ranks

    Args:
        config: Transformer configuration with V3.2 indexer parameters
        submodules: Submodule specifications
        layer_number: Layer index (0-indexed)
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: LightningIndexerSubmodules,
        layer_number: int,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number

        # Dimensions from config
        self.hidden_size = config.hidden_size
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_pos_emb_head_dim  # RoPE dimension (64)

        # TP dimensions
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.n_local_heads = self.index_n_heads // self.tp_size

        # wq_b: ColumnParallelLinear - splits heads across TP
        # Projects compressed Q (from MLA's q_down_proj) to indexer heads
        # Shape: [q_lora_rank, n_local_heads * head_dim] after TP split
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            gather_output=False,  # Keep sharded across TP
            bias=False,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
        )

        # wk: Regular Linear - REPLICATED across all TP ranks
        # Projects hidden states to single key (shared by all heads)
        # Shape: [hidden_size, head_dim]
        self.linear_wk = nn.Linear(
            self.hidden_size,
            self.index_head_dim,
            bias=False,
            dtype=config.params_dtype,
        )
        # Initialize with config's init method
        config.init_method(self.linear_wk.weight)

        # k_norm: LayerNorm - REPLICATED
        # Using standard LayerNorm as in official code (not RMSNorm)
        self.k_layernorm = nn.LayerNorm(
            self.index_head_dim,
            eps=config.layernorm_epsilon,
            dtype=config.params_dtype,
        )

        # weights_proj: ColumnParallelLinear - splits per-head weights
        # Projects hidden states to per-head aggregation weights
        # Shape: [hidden_size, n_local_heads] after TP split
        # Note: Official code uses fp32 for this projection
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            gather_output=False,  # Keep sharded across TP
            bias=False,
            skip_bias_add=True,
            skip_weight_param_allocation=False,
        )

        # Scaling factors (match official implementation)
        self.softmax_scale = self.index_head_dim**-0.5
        self.n_heads_scale = self.index_n_heads**-0.5

        # Chunking for memory efficiency at long sequences
        self.chunk_size = 1024  # Process Q in chunks to avoid memory blowup

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute top-k token indices for sparse attention.

        Args:
            hidden_states: Input hidden states [seq, batch, hidden] (SBH format)
            q_compressed: Compressed Q from MLA's q_down_proj [seq, batch, q_lora_rank]
            rotary_pos_emb: Precomputed RoPE frequencies (complex tensor)
            attention_mask: Optional causal mask [batch, 1, seq, seq] or similar

        Returns:
            topk_indices: Selected token indices [batch, seq, topk]
                          Identical on all TP ranks after all-reduce.
        """
        seq_len, batch_size, _ = hidden_states.size()

        # Use chunked implementation for long sequences to save memory
        if seq_len > self.chunk_size:
            return self._forward_chunked(
                hidden_states, q_compressed, rotary_pos_emb, attention_mask
            )
        else:
            return self._forward_simple(
                hidden_states, q_compressed, rotary_pos_emb, attention_mask
            )

    def _forward_simple(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simple non-chunked forward pass for short sequences."""
        seq_len, batch_size, _ = hidden_states.size()

        # === Q Projection (from shared compressed representation) ===
        # Same as official: q = self.wq_b(qr)
        q, _ = self.linear_wq_b(q_compressed)  # [seq, batch, n_local_heads * head_dim]
        q = q.view(seq_len, batch_size, self.n_local_heads, self.index_head_dim)

        # === K Projection + Norm (REPLICATED) ===
        # Same as official: k = self.k_norm(self.wk(x))
        k = self.linear_wk(hidden_states)  # [seq, batch, head_dim]
        k = self.k_layernorm(k)

        # === Apply Non-interleaved RoPE ===
        # CRITICAL: Official uses interleaved=False for indexer
        # Split into rope and non-rope portions
        q_pe = q[..., : self.qk_rope_head_dim]
        q_nope = q[..., self.qk_rope_head_dim :]
        k_pe = k[..., : self.qk_rope_head_dim]
        k_nope = k[..., self.qk_rope_head_dim :]

        # Get RoPE frequencies for this CP rank if using context parallelism
        if parallel_state.get_context_parallel_world_size() > 1:
            freqs_cis = get_pos_emb_on_this_cp_rank(rotary_pos_emb, seq_len)
        else:
            freqs_cis = rotary_pos_emb[:seq_len]

        # Apply non-interleaved RoPE
        q_pe = apply_rotary_emb_non_interleaved(q_pe, freqs_cis)
        k_pe = apply_rotary_emb_non_interleaved(k_pe.unsqueeze(2), freqs_cis).squeeze(2)

        # Concatenate back
        q = torch.cat([q_pe, q_nope], dim=-1)  # [seq, batch, n_local_heads, head_dim]
        k = torch.cat([k_pe, k_nope], dim=-1)  # [seq, batch, head_dim]

        # === Scoring with ReLU ===
        # logits[s, b, h, t] = q[s, b, h, :] @ k[t, b, :].T
        # Transpose to [batch, seq, heads, head_dim] for einsum
        q = q.permute(1, 0, 2, 3)  # [batch, seq, n_local_heads, head_dim]
        k = k.permute(1, 0, 2)  # [batch, seq, head_dim]

        logits = torch.einsum("bshd,btd->bsht", q, k) * self.softmax_scale
        logits = F.relu(logits)  # ReLU activation (same as official)

        # === Weighted sum across heads ===
        # Same as official: weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
        hidden_states_bsh = hidden_states.permute(1, 0, 2)  # [batch, seq, hidden]
        weights, _ = self.linear_weights_proj(
            hidden_states_bsh.float()
        )  # [batch, seq, n_local_heads]
        weights = weights * self.n_heads_scale

        # index_score = sum_h(weights[h] * logits[h])
        # [batch, seq, n_local_heads] x [batch, seq, n_local_heads, seq] -> [batch, seq, seq]
        index_score_local = torch.einsum("bsh,bsht->bst", weights, logits)

        # === TP All-Reduce ===
        # Sum partial scores across TP ranks to get full index_score
        if self.tp_size > 1:
            index_score = _reduce(index_score_local)
        else:
            index_score = index_score_local

        # === Apply Causal Mask ===
        if attention_mask is not None:
            # attention_mask is typically [batch, 1, seq, seq] or [1, 1, seq, seq]
            # We need [batch, seq, seq]
            if attention_mask.dim() == 4:
                mask = attention_mask.squeeze(1)  # [batch, seq, seq]
            else:
                mask = attention_mask
            index_score = index_score + mask

        # === Top-K Selection ===
        actual_topk = min(self.index_topk, seq_len)
        topk_indices = index_score.topk(actual_topk, dim=-1)[1]  # [batch, seq, topk]

        # Detach indices - gradients flow through indexer weights, not through selection
        return topk_indices.detach()

    def _forward_chunked(
        self,
        hidden_states: torch.Tensor,
        q_compressed: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Memory-efficient chunked forward pass for long sequences.

        Instead of materializing full [seq, seq, heads] logits tensor,
        process Q in chunks and accumulate scores.
        """
        seq_len, batch_size, _ = hidden_states.size()

        # === Pre-compute K (small: ~8MB for 32K at head_dim=128) ===
        k = self.linear_wk(hidden_states)  # [seq, batch, head_dim]
        k = self.k_layernorm(k)

        # Apply RoPE to K
        k_pe = k[..., : self.qk_rope_head_dim]
        k_nope = k[..., self.qk_rope_head_dim :]

        if parallel_state.get_context_parallel_world_size() > 1:
            freqs_cis = get_pos_emb_on_this_cp_rank(rotary_pos_emb, seq_len)
        else:
            freqs_cis = rotary_pos_emb[:seq_len]

        k_pe = apply_rotary_emb_non_interleaved(k_pe.unsqueeze(2), freqs_cis).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)
        k = k.permute(1, 0, 2)  # [batch, seq, head_dim]

        # === Pre-compute weights (small) ===
        hidden_states_bsh = hidden_states.permute(1, 0, 2)  # [batch, seq, hidden]
        weights, _ = self.linear_weights_proj(hidden_states_bsh.float())
        weights = weights * self.n_heads_scale  # [batch, seq, n_local_heads]

        # === Process Q in chunks ===
        all_scores = []
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_len = end - start

            # Project Q chunk
            q_chunk, _ = self.linear_wq_b(
                q_compressed[start:end]
            )  # [chunk, batch, n_local_heads * head_dim]
            q_chunk = q_chunk.view(
                chunk_len, batch_size, self.n_local_heads, self.index_head_dim
            )

            # Apply RoPE to Q chunk
            q_pe_chunk = q_chunk[..., : self.qk_rope_head_dim]
            q_nope_chunk = q_chunk[..., self.qk_rope_head_dim :]
            freqs_chunk = freqs_cis[start:end]
            q_pe_chunk = apply_rotary_emb_non_interleaved(q_pe_chunk, freqs_chunk)
            q_chunk = torch.cat([q_pe_chunk, q_nope_chunk], dim=-1)
            q_chunk = q_chunk.permute(1, 0, 2, 3)  # [batch, chunk, n_local_heads, head_dim]

            # Score chunk against ALL keys
            logits_chunk = (
                torch.einsum("bshd,btd->bsht", q_chunk, k) * self.softmax_scale
            )
            logits_chunk = F.relu(logits_chunk)

            # Weighted sum for this chunk
            weights_chunk = weights[:, start:end, :]  # [batch, chunk, n_local_heads]
            score_chunk = torch.einsum(
                "bsh,bsht->bst", weights_chunk, logits_chunk
            )  # [batch, chunk, seq]
            all_scores.append(score_chunk)

        # Combine all chunks
        index_score_local = torch.cat(all_scores, dim=1)  # [batch, seq, seq]

        # === TP All-Reduce ===
        if self.tp_size > 1:
            index_score = _reduce(index_score_local)
        else:
            index_score = index_score_local

        # === Apply Causal Mask ===
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask = attention_mask.squeeze(1)
            else:
                mask = attention_mask
            index_score = index_score + mask

        # === Top-K Selection ===
        actual_topk = min(self.index_topk, seq_len)
        topk_indices = index_score.topk(actual_topk, dim=-1)[1]

        return topk_indices.detach()

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """
        Generate sharded state dict for distributed checkpointing.

        wq_b and weights_proj are column-parallel sharded.
        wk and k_layernorm are replicated (not sharded).
        """
        sharded_state_dict = {}

        # ColumnParallel layers have their own sharded_state_dict
        sharded_state_dict.update(
            self.linear_wq_b.sharded_state_dict(
                prefix=f"{prefix}linear_wq_b.",
                sharded_offsets=sharded_offsets,
                metadata=metadata,
            )
        )
        sharded_state_dict.update(
            self.linear_weights_proj.sharded_state_dict(
                prefix=f"{prefix}linear_weights_proj.",
                sharded_offsets=sharded_offsets,
                metadata=metadata,
            )
        )

        # Replicated layers - store full tensors
        sharded_state_dict[f"{prefix}linear_wk.weight"] = self.linear_wk.weight
        sharded_state_dict[f"{prefix}k_layernorm.weight"] = self.k_layernorm.weight
        sharded_state_dict[f"{prefix}k_layernorm.bias"] = self.k_layernorm.bias

        return sharded_state_dict
