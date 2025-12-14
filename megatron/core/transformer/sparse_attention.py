# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sparse Attention for DeepSeek V3.2.

This module implements sparse attention that operates only on top-K selected tokens,
reducing attention complexity from O(L^2) to O(L*K) where K is typically 2048.

The sparse attention gathers K/V at selected indices and computes attention
over the subset, applying appropriate causal masking based on token positions.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class SparseAttention(MegatronModule):
    """
    Sparse attention that operates only on top-K selected tokens.

    Instead of computing full [seq, seq] attention matrix, this module:
    1. Gathers K and V tensors at selected indices
    2. Computes attention on [seq, topk] instead of [seq, seq]
    3. Applies causal mask based on index positions

    This is used in DeepSeek V3.2 after the Lightning Indexer selects
    the most relevant tokens for each query position.

    Memory Complexity: O(seq * topk * head_dim) instead of O(seq^2 * head_dim)

    Args:
        config: Transformer configuration
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config = config

        # Get attention scaling
        if config.softmax_scale is not None:
            self.softmax_scale = config.softmax_scale
        else:
            # For MLA, effective head dim is qk_head_dim + qk_pos_emb_head_dim
            q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
            self.softmax_scale = q_head_dim**-0.5

        # Whether to use attention softmax in fp32
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sparse attention using gathered K/V.

        Args:
            query: Query tensor [seq, batch, n_heads, head_dim]
            key: Key tensor [seq, batch, n_heads, head_dim]
            value: Value tensor [seq, batch, n_heads, v_head_dim]
            topk_indices: Selected token indices [batch, seq, topk]
            attention_mask: Optional attention mask (unused, we compute causal from indices)

        Returns:
            context: Attention output [seq, batch, n_heads, v_head_dim]
        """
        seq_len, batch_size, n_heads, head_dim = query.shape
        _, _, _, v_head_dim = value.shape
        topk = topk_indices.shape[-1]

        # Transpose to [batch, seq, heads, dim] for easier indexing
        query = query.permute(1, 0, 2, 3)  # [batch, seq, n_heads, head_dim]
        key = key.permute(1, 0, 2, 3)  # [batch, seq, n_heads, head_dim]
        value = value.permute(1, 0, 2, 3)  # [batch, seq, n_heads, v_head_dim]

        # === Gather selected K and V ===
        # topk_indices: [batch, seq, topk]
        # We need to gather along the seq dimension

        # Expand indices for gathering
        # [batch, seq, topk] -> [batch, seq, topk, n_heads, head_dim]
        indices_k = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, n_heads, head_dim
        )
        indices_v = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, -1, n_heads, v_head_dim
        )

        # Expand key/value for gathering
        # key: [batch, seq, n_heads, head_dim] -> [batch, 1, seq, n_heads, head_dim]
        # Then broadcast to [batch, seq, seq, n_heads, head_dim]
        # But we only need to gather topk positions

        # Reshape for efficient gather
        # key[batch, gather_idx, heads, dim] where gather_idx comes from topk_indices
        key_gathered = self._gather_along_seq(key, topk_indices, n_heads, head_dim)
        value_gathered = self._gather_along_seq(value, topk_indices, n_heads, v_head_dim)

        # key_gathered: [batch, seq, topk, n_heads, head_dim]
        # value_gathered: [batch, seq, topk, n_heads, v_head_dim]

        # === Compute attention scores ===
        # query: [batch, seq, n_heads, head_dim]
        # key_gathered: [batch, seq, topk, n_heads, head_dim]
        # scores: [batch, seq, n_heads, topk]
        scores = torch.einsum(
            "bshd,bskhd->bshk", query, key_gathered
        ) * self.softmax_scale

        # === Apply causal mask based on token positions ===
        # For each query position q, mask out indices where index > q
        # This ensures we only attend to past positions
        positions = torch.arange(seq_len, device=query.device).view(
            1, -1, 1, 1
        )  # [1, seq, 1, 1]
        indices_for_mask = topk_indices.unsqueeze(2)  # [batch, seq, 1, topk]

        # Mask where gathered position > current position (future tokens)
        causal_mask = torch.where(
            indices_for_mask > positions,
            torch.tensor(float("-inf"), device=query.device, dtype=scores.dtype),
            torch.tensor(0.0, device=query.device, dtype=scores.dtype),
        )
        scores = scores + causal_mask

        # === Softmax ===
        if self.attention_softmax_in_fp32:
            scores = scores.float()
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.type_as(query)

        # === Compute output ===
        # attn_weights: [batch, seq, n_heads, topk]
        # value_gathered: [batch, seq, topk, n_heads, v_head_dim]
        # output: [batch, seq, n_heads, v_head_dim]
        output = torch.einsum("bshk,bskhd->bshd", attn_weights, value_gathered)

        # Transpose back to [seq, batch, n_heads, v_head_dim]
        output = output.permute(1, 0, 2, 3)

        return output

    def _gather_along_seq(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor,
        n_heads: int,
        dim: int,
    ) -> torch.Tensor:
        """
        Gather tensor values at specified sequence indices.

        Args:
            tensor: Input tensor [batch, seq, n_heads, dim]
            indices: Gather indices [batch, seq, topk]
            n_heads: Number of attention heads
            dim: Head dimension

        Returns:
            gathered: [batch, seq, topk, n_heads, dim]
        """
        batch_size, seq_len, _, _ = tensor.shape
        topk = indices.shape[-1]

        # Method: Use advanced indexing
        # Create batch indices
        batch_idx = torch.arange(batch_size, device=tensor.device).view(-1, 1, 1)
        batch_idx = batch_idx.expand(-1, seq_len, topk)  # [batch, seq, topk]

        # Gather: tensor[batch_idx, indices] -> [batch, seq, topk, n_heads, dim]
        gathered = tensor[batch_idx, indices]  # [batch, seq, topk, n_heads, dim]

        return gathered


class SparseAttentionForPrefill(MegatronModule):
    """
    Optimized sparse attention for prefill phase.

    During prefill, we have the full sequence and can use more efficient
    batched operations. This implementation matches the official DeepSeek V3.2
    prefill attention pattern.

    The key difference from decode is that we apply the sparse mask to the
    full attention scores rather than gathering K/V first.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config=config)
        self.config = config

        if config.softmax_scale is not None:
            self.softmax_scale = config.softmax_scale
        else:
            q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
            self.softmax_scale = q_head_dim**-0.5

        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        topk_indices: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sparse attention for prefill using mask-based approach.

        This matches the official V3.2 prefill implementation:
        1. Compute full attention scores Q @ K.T
        2. Create sparse mask from topk_indices
        3. Apply causal + sparse mask
        4. Softmax and compute output

        Args:
            query: [seq, batch, n_heads, head_dim]
            key: [seq, batch, n_heads, head_dim]
            value: [seq, batch, n_heads, v_head_dim]
            topk_indices: [batch, seq, topk]
            attention_mask: Optional causal mask [batch, 1, seq, seq]

        Returns:
            context: [seq, batch, n_heads, v_head_dim]
        """
        seq_len, batch_size, n_heads, head_dim = query.shape
        _, _, _, v_head_dim = value.shape

        # Transpose to [batch, n_heads, seq, dim]
        query = query.permute(1, 2, 0, 3)  # [batch, n_heads, seq, head_dim]
        key = key.permute(1, 2, 0, 3)  # [batch, n_heads, seq, head_dim]
        value = value.permute(1, 2, 0, 3)  # [batch, n_heads, seq, v_head_dim]

        # Compute full attention scores
        # [batch, n_heads, seq, seq]
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.softmax_scale

        # Create sparse mask from topk_indices
        # index_mask[b, s, t] = -inf if t not in topk_indices[b, s, :] else 0
        index_mask = torch.full(
            (batch_size, seq_len, seq_len),
            float("-inf"),
            device=query.device,
            dtype=scores.dtype,
        )
        # Scatter 0 at selected positions
        index_mask.scatter_(-1, topk_indices, 0.0)

        # Expand mask for heads: [batch, 1, seq, seq]
        index_mask = index_mask.unsqueeze(1)

        # Combine with causal mask if provided
        if attention_mask is not None:
            index_mask = index_mask + attention_mask

        scores = scores + index_mask

        # Softmax
        if self.attention_softmax_in_fp32:
            scores = scores.float()
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_weights.type_as(query)

        # Compute output
        output = torch.matmul(attn_weights, value)  # [batch, n_heads, seq, v_head_dim]

        # Transpose back to [seq, batch, n_heads, v_head_dim]
        output = output.permute(2, 0, 1, 3)

        return output
