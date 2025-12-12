# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
DeepSeek Sparse Attention (DSA) utilities for Lightning Indexer.

This module provides RoPE (Rotary Position Embedding) utilities specifically
for the Lightning Indexer in DeepSeek V3.2's sparse attention mechanism.

References:
    - DeepSeek-V3.2 Tech Report: https://arxiv.org/abs/2512.02556
    - Section 2.1: DeepSeek Sparse Attention (DSA)

Key Differences from Megatron's Standard MLA RoPE:
    - Uses INTERLEAVED format with complex multiplication (matches DeepSeek exactly)
    - Does NOT convert from interleaved to non-interleaved before applying RoPE
    - Outputs complex polar form for frequency computation

Functions:
    apply_rotary_emb: Apply rotary embeddings using complex multiplication
    precompute_freqs_cis: Precompute RoPE frequencies with YaRN scaling

Example:
    >>> import torch
    >>> from megatron.core.transformer.dsa_utils import apply_rotary_emb, precompute_freqs_cis
    >>>
    >>> # Precompute frequencies for sequence length 4096
    >>> freqs_cis = precompute_freqs_cis(dim=64, seq_len=4096, device=torch.device('cuda'))
    >>>
    >>> # Apply RoPE to query tensor [batch, seq, heads, head_dim]
    >>> q = torch.randn(2, 1024, 8, 64, device='cuda')
    >>> q_rotated = apply_rotary_emb(q, freqs_cis[:1024])

See Also:
    - :class:`megatron.core.transformer.lightning_indexer.LightningIndexer`: Main indexer class
    - :mod:`megatron.core.models.common.embeddings`: Megatron's standard RoPE implementations
"""

import math

import torch
from torch import Tensor

__all__ = ['apply_rotary_emb', 'precompute_freqs_cis']


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """
    Apply rotary positional embeddings using complex multiplication.
    Matches DeepSeek V3/V3.2 implementation exactly.

    Mathematical operation (line by line):

    1. Input x has shape [..., head_dim] in INTERLEAVED format:
       x = [r0, i0, r1, i1, r2, i2, ...] where (r_k, i_k) form complex pairs

    2. x.view(*x.shape[:-1], -1, 2) reshapes to [..., head_dim//2, 2]:
       [[r0, i0], [r1, i1], [r2, i2], ...]

    3. torch.view_as_complex() creates complex tensor [..., head_dim//2]:
       [r0 + i*i0, r1 + i*i1, r2 + i*i2, ...]

    4. Complex multiplication x * freqs_cis applies rotation:
       freqs_cis[k] = e^(i*θ_k) = cos(θ_k) + i*sin(θ_k)

       (a + bi) * (cos(θ) + i*sin(θ)) =
           (a*cos(θ) - b*sin(θ)) + i*(a*sin(θ) + b*cos(θ))

       This rotates each complex pair by angle θ_k

    5. torch.view_as_real() converts back to [..., head_dim//2, 2]

    6. .flatten(-2) restores to [..., head_dim] in interleaved format

    Args:
        x: Input tensor [B, S, H, head_dim] in interleaved format
        freqs_cis: Complex frequency tensor [seq_len, head_dim//2]

    Returns:
        Rotated tensor [B, S, H, head_dim] in interleaved format
    """
    dtype = x.dtype

    # Reshape to pairs and view as complex
    # [B, S, H, D] -> [B, S, H, D//2, 2] -> [B, S, H, D//2] (complex)
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))

    # Broadcast freqs_cis: [S, D//2] -> [1, S, 1, D//2]
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))

    # Complex multiplication applies rotation
    # [B, S, H, D//2] * [1, S, 1, D//2] -> [B, S, H, D//2]
    y = torch.view_as_real(x * freqs_cis).flatten(-2)

    return y.to(dtype)


def precompute_freqs_cis(
    dim: int,
    seq_len: int,
    theta: float = 10000.0,
    scaling_factor: float = 1.0,
    original_max_position_embeddings: int = 4096,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: torch.device = None,
) -> Tensor:
    """
    Precompute complex rotary frequencies with full YaRN scaling.

    Matches megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py
    but outputs complex polar form for DeepSeek's RoPE implementation.

    YaRN (Yet Another RoPE Extension) applies frequency interpolation:
    - Low frequencies (slow-changing): Keep original (capture long-range patterns)
    - High frequencies (fast-changing): Scale down (avoid aliasing at long contexts)
    - Middle frequencies: Smooth linear interpolation between the two

    Args:
        dim: RoPE dimension (typically qk_pos_emb_head_dim = 64)
        seq_len: Maximum sequence length to precompute
        theta: Base frequency (rotary_base, default 10000)
        scaling_factor: YaRN scaling factor (e.g., 40 for 128K context)
        original_max_position_embeddings: Original context length (e.g., 4096)
        beta_fast: Upper frequency bound for interpolation (default 32)
        beta_slow: Lower frequency bound for interpolation (default 1)
        device: Target device for tensor creation

    Returns:
        freqs_cis: Complex tensor [seq_len, dim//2] where
                   freqs_cis[pos, k] = e^(i * pos * inv_freq[k])
    """
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    # Base inverse frequencies: inv_freq[k] = 1 / (theta^(2k/dim))
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    )

    # Apply YaRN scaling if factor > 1
    if scaling_factor > 1.0:

        # Find correction dimension range based on rotations
        def find_correction_dim(num_rotations, dim, base, max_pos):
            return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (
                2 * math.log(base)
            )

        low = math.floor(
            find_correction_dim(beta_fast, dim, theta, original_max_position_embeddings)
        )
        high = math.ceil(
            find_correction_dim(beta_slow, dim, theta, original_max_position_embeddings)
        )
        low = max(low, 0)
        high = min(high, dim - 1)

        # Linear ramp mask: 0 at low, 1 at high
        if low == high:
            high += 0.001  # Prevent division by zero
        linear_func = (
            torch.arange(dim // 2, dtype=torch.float32, device=device) - low
        ) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)
        inv_freq_mask = 1.0 - ramp_func

        # Scaled frequencies for extended context
        inv_freq_scaled = inv_freq / scaling_factor

        # Interpolate: high freq dims use scaled, low freq dims use original
        inv_freq = inv_freq_scaled * (1 - inv_freq_mask) + inv_freq * inv_freq_mask

    # Compute outer product: freqs[pos, k] = pos * inv_freq[k]
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)

    # Convert to complex polar form: e^(i*theta) = cos(theta) + i*sin(theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis
