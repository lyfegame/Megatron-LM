# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Lightning Indexer for DeepSeek Sparse Attention (DSA).

This module implements DeepSeek V3.2's sparse attention mechanism, which reduces
computational complexity while preserving model performance in long-context scenarios.

References:
    - DeepSeek-V3.2 Tech Report: https://arxiv.org/abs/2512.02556
    - Section 2.1: DeepSeek Sparse Attention (DSA)

Mathematical Formulation:
    The Lightning Indexer computes sparse attention indices using the formula:

        I_{t,s} = Σ_{j=1}^{H^I} w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)

    Where:
        - H^I = index_n_heads (64 in V3.2)
        - q_{t,j}^I ∈ ℝ^{d^I} = query vector for head j at position t
        - k_s^I ∈ ℝ^{d^I} = key vector at position s (single-head, MQA-style)
        - w_{t,j}^I ∈ ℝ = learned weight for head j at position t

Key Implementation Details:
    - Uses INTERLEAVED RoPE with complex multiplication (matches DeepSeek exactly)
    - Single-head keys (MQA) broadcast to multi-head queries
    - ReLU activation for throughput efficiency (zeros negative contributions)
    - Designed to integrate with MLA's existing q_compressed tensor

Example:
    >>> from megatron.core.transformer.transformer_config import MLATransformerConfig
    >>> from megatron.core.models.gpt.gpt_layer_specs import get_mla_with_dsa_spec
    >>> from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    >>>
    >>> # Create config with DSA enabled
    >>> config = MLATransformerConfig(
    ...     hidden_size=7168,
    ...     num_attention_heads=128,
    ...     use_sparse_attention=True,
    ...     index_n_heads=64,
    ...     index_head_dim=128,
    ...     index_topk=2048,
    ... )
    >>>
    >>> # Get layer spec with Lightning Indexer
    >>> layer_spec = get_mla_with_dsa_spec(TESpecProvider())

See Also:
    - :mod:`megatron.core.transformer.dsa_utils`: RoPE utilities for DSA
    - :class:`megatron.core.transformer.multi_latent_attention.MLASelfAttention`: MLA integration
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from megatron.core.transformer.dsa_utils import apply_rotary_emb, precompute_freqs_cis
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import MLATransformerConfig

__all__ = ['LightningIndexer', 'LightningIndexerSubmodules']


@dataclass
class LightningIndexerSubmodules:
    """Submodules for the Lightning Indexer."""

    linear_q_proj: Union[ModuleSpec, type] = None
    linear_k_proj: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class LightningIndexer(MegatronModule):
    """
    Lightning Indexer for DeepSeek Sparse Attention (DSA).

    Computes sparse attention indices using the formula from the V3.2 tech report:

        I_{t,s} = Σ_{j=1}^{H^I} w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)

    Where:
        - H^I = index_n_heads (64 in V3.2)
        - q_{t,j}^I ∈ ℝ^{d^I} = query vector for head j at position t
        - k_s^I ∈ ℝ^{d^I} = key vector at position s (single-head, MQA-style)
        - w_{t,j}^I ∈ ℝ = learned weight for head j at position t

    Key implementation details:
        - Uses INTERLEAVED RoPE with complex multiplication (matches DeepSeek exactly)
        - Single-head keys (MQA) broadcast to multi-head queries
        - ReLU activation for throughput efficiency (zeros negative contributions)
        - Designed to integrate with MLA's existing q_compressed tensor

    Reference: DeepSeek V3.2 Tech Report (arXiv:2512.02556), Section 2.1
    """

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: LightningIndexerSubmodules,
        layer_number: int,
    ):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number

        # Dimensions from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.index_n_heads  # H^I = 64
        self.head_dim = config.index_head_dim  # d^I = 128
        self.rope_dim = config.qk_pos_emb_head_dim  # 64 (shared with MLA)
        self.index_topk = config.index_topk  # k = 2048
        self.q_lora_rank = config.q_lora_rank  # 1536

        self.softmax_scale = self.head_dim**-0.5

        # Query projection: q_compressed → [S, B, num_heads * head_dim]
        self.linear_q_proj = build_module(
            submodules.linear_q_proj,
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=True,  # Gather to get full output
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        # Key projection: hidden_states → [S, B, head_dim] (single-head MQA)
        self.linear_k_proj = build_module(
            submodules.linear_k_proj,
            self.hidden_size,
            self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=True,  # Gather for single-head output
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        # Key LayerNorm
        self.k_layernorm = build_module(
            submodules.k_layernorm,
            hidden_size=self.head_dim,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        # Head weights projection: hidden_states → [S, B, num_heads]
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj,
            self.hidden_size,
            self.num_heads,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=True,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        # Precompute RoPE frequencies (will be moved to correct device on first forward)
        self._freqs_cis = None
        self._freqs_cis_seq_len = 0

    def _get_freqs_cis(self, seq_len: int, device: torch.device) -> Tensor:
        """Get or compute RoPE frequencies with caching."""
        if self._freqs_cis is None or seq_len > self._freqs_cis_seq_len:
            self._freqs_cis = precompute_freqs_cis(
                dim=self.rope_dim,
                seq_len=max(seq_len, 8192),  # Precompute extra for efficiency
                theta=self.config.rotary_base,
                scaling_factor=self.config.rotary_scaling_factor,
                original_max_position_embeddings=self.config.original_max_position_embeddings,
                beta_fast=self.config.beta_fast,
                beta_slow=self.config.beta_slow,
                device=device,
            )
            self._freqs_cis_seq_len = self._freqs_cis.size(0)
        return self._freqs_cis[:seq_len]

    def forward(
        self,
        hidden_states: Tensor,
        q_compressed: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_offset: int = 0,
    ) -> Tensor:
        """
        Compute top-k token indices for sparse attention.

        Args:
            hidden_states: [S, B, hidden_size] - input to attention layer
            q_compressed: [S, B, q_lora_rank] - compressed Q from MLA (DETACHED for sparse training)
            attention_mask: Optional causal mask [B, 1, S, T]
            position_offset: Starting position for RoPE (for inference with KV cache)

        Returns:
            topk_indices: [S, B, index_topk] - indices of top-k tokens per position
        """
        seq_len, batch_size, _ = hidden_states.shape
        device = hidden_states.device

        # Get RoPE frequencies
        freqs_cis = self._get_freqs_cis(seq_len + position_offset, device)
        freqs_cis = freqs_cis[position_offset : position_offset + seq_len]

        # ============================================
        # Query path: q_compressed → Q with RoPE
        # ============================================
        q_states, _ = self.linear_q_proj(q_compressed)
        q_states = q_states.view(seq_len, batch_size, self.num_heads, self.head_dim)

        # Split into RoPE and non-RoPE parts, apply RoPE, recombine
        q_rope = q_states[..., : self.rope_dim]
        q_nope = q_states[..., self.rope_dim :]

        # Transpose for RoPE: [S, B, H, D] -> [B, S, H, D]
        q_rope = q_rope.transpose(0, 1)
        q_rope = apply_rotary_emb(q_rope, freqs_cis)
        q_rope = q_rope.transpose(0, 1)

        q_states = torch.cat([q_rope, q_nope], dim=-1)

        # ============================================
        # Key path: hidden_states → K with RoPE (single-head MQA)
        # ============================================
        k_states, _ = self.linear_k_proj(hidden_states)
        k_states = self.k_layernorm(k_states)  # [S, B, head_dim]

        # Split, apply RoPE, recombine
        k_rope = k_states[..., : self.rope_dim]
        k_nope = k_states[..., self.rope_dim :]

        # Add head dim for RoPE, then remove: [S, B, D] -> [B, S, 1, D]
        k_rope = k_rope.transpose(0, 1).unsqueeze(2)
        k_rope = apply_rotary_emb(k_rope, freqs_cis)
        k_rope = k_rope.squeeze(2).transpose(0, 1)

        k_states = torch.cat([k_rope, k_nope], dim=-1)

        # ============================================
        # Compute index scores
        # I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)
        # ============================================

        # Head weights: [S, B, num_heads] (cast to float32 per reference)
        head_weights, _ = self.linear_weights_proj(hidden_states)
        head_weights = head_weights.float() * (self.num_heads**-0.5)

        # Transpose to [B, ...] for batched operations
        q_t = q_states.transpose(0, 1)  # [B, S, H, D]
        k_t = k_states.transpose(0, 1)  # [B, T, D]
        w_t = head_weights.transpose(0, 1)  # [B, S, H]

        # Attention scores: [B, S, H, D] @ [B, D, T] -> [B, S, H, T]
        scores = torch.einsum("bshd,btd->bsht", q_t.float(), k_t.float())

        # Apply ReLU, weight by head weights, sum over heads
        scores = torch.relu(scores)
        scores = scores * w_t.unsqueeze(-1)  # [B, S, H, T] * [B, S, H, 1]
        index_scores = scores.sum(dim=2)  # [B, S, T]

        # Apply softmax scale
        index_scores = index_scores * self.softmax_scale

        # ============================================
        # Apply mask and select top-k
        # ============================================
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                attention_mask = attention_mask.squeeze(1)  # [B, 1, S, T] -> [B, S, T]
            index_scores = index_scores + attention_mask

        T = index_scores.shape[-1]
        topk = min(self.index_topk, T)
        topk_indices = index_scores.topk(topk, dim=-1).indices

        # Transpose back to Megatron format: [B, S, topk] -> [S, B, topk]
        return topk_indices.transpose(0, 1)
