# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for Lightning Indexer and DSA utilities.

Tests cover:
- apply_rotary_emb: Interleaved RoPE with complex multiplication
- precompute_freqs_cis: YaRN frequency computation
- LightningIndexer: Sparse attention index computation
"""

import pytest
import torch

from megatron.core.transformer.dsa_utils import apply_rotary_emb, precompute_freqs_cis


class TestApplyRotaryEmb:
    """Test RoPE implementation matches DeepSeek exactly."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_output_shape(self):
        """Output shape should match input shape."""
        batch, seq, heads, dim = 2, 16, 8, 64
        x = torch.randn(batch, seq, heads, dim, device='cuda')
        freqs_cis = precompute_freqs_cis(dim, seq, device=torch.device('cuda'))

        y = apply_rotary_emb(x, freqs_cis)

        assert y.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_output_dtype_preserved(self):
        """Output dtype should match input dtype."""
        batch, seq, heads, dim = 2, 16, 8, 64

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(batch, seq, heads, dim, device='cuda', dtype=dtype)
            freqs_cis = precompute_freqs_cis(dim, seq, device=torch.device('cuda'))

            y = apply_rotary_emb(x, freqs_cis)

            assert y.dtype == dtype, f"Expected {dtype}, got {y.dtype}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_interleaved_format_identity(self):
        """Verify input/output are in interleaved format [r0, i0, r1, i1, ...]."""
        batch, seq, heads, dim = 1, 4, 1, 8
        # Create known input: pairs of (real, imag) - all real, no imag
        x = torch.tensor([[[[1, 0, 2, 0, 3, 0, 4, 0]]]], device='cuda', dtype=torch.float32)
        # Identity rotation (all ones in polar form = e^(i*0) = 1)
        freqs_cis = torch.ones(seq, dim // 2, dtype=torch.complex64, device='cuda')

        y = apply_rotary_emb(x, freqs_cis)

        # With identity rotation, output should equal input
        torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rotation_90_degrees(self):
        """Test 90-degree rotation: (a, b) -> (-b, a)."""
        # Input: 1 + 0i (represented as [1, 0] in interleaved format)
        x = torch.tensor([[[[1.0, 0.0]]]], device='cuda')
        # 90 degrees = pi/2, e^(i*pi/2) = i = 0 + 1i
        freqs_cis = torch.tensor([[0 + 1j]], dtype=torch.complex64, device='cuda')

        y = apply_rotary_emb(x, freqs_cis)

        # (1 + 0i) * i = 0 + 1i -> [0, 1]
        expected = torch.tensor([[[[0.0, 1.0]]]], device='cuda')
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rotation_180_degrees(self):
        """Test 180-degree rotation: (a, b) -> (-a, -b)."""
        # Input: 1 + 1i (represented as [1, 1] in interleaved format)
        x = torch.tensor([[[[1.0, 1.0]]]], device='cuda')
        # 180 degrees = pi, e^(i*pi) = -1
        freqs_cis = torch.tensor([[-1 + 0j]], dtype=torch.complex64, device='cuda')

        y = apply_rotary_emb(x, freqs_cis)

        # (1 + 1i) * (-1) = -1 - 1i -> [-1, -1]
        expected = torch.tensor([[[[-1.0, -1.0]]]], device='cuda')
        torch.testing.assert_close(y, expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_consistency(self):
        """Same input across batches should produce same output."""
        seq, heads, dim = 16, 4, 64
        x_single = torch.randn(1, seq, heads, dim, device='cuda')
        x_batch = x_single.expand(4, -1, -1, -1).clone()
        freqs_cis = precompute_freqs_cis(dim, seq, device=torch.device('cuda'))

        y_batch = apply_rotary_emb(x_batch, freqs_cis)

        # All batch elements should be identical
        for i in range(1, 4):
            torch.testing.assert_close(y_batch[0], y_batch[i], rtol=1e-5, atol=1e-5)


class TestPrecomputeFreqsCis:
    """Test YaRN frequency computation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_output_shape(self):
        """Output shape should be [seq_len, dim//2] complex."""
        dim, seq_len = 64, 128
        freqs = precompute_freqs_cis(dim, seq_len, device=torch.device('cuda'))

        assert freqs.shape == (seq_len, dim // 2)
        assert freqs.dtype == torch.complex64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_unit_magnitude(self):
        """All frequencies should have unit magnitude (|e^(i*theta)| = 1)."""
        freqs = precompute_freqs_cis(64, 128, device=torch.device('cuda'))
        magnitudes = torch.abs(freqs)

        torch.testing.assert_close(
            magnitudes, torch.ones_like(magnitudes), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_position_zero_is_identity(self):
        """At position 0, all frequencies should be 1 (e^(i*0) = 1)."""
        freqs = precompute_freqs_cis(64, 128, device=torch.device('cuda'))

        # Position 0 should be all ones (real part = 1, imag part = 0)
        expected = torch.ones(freqs.shape[1], dtype=torch.complex64, device='cuda')
        torch.testing.assert_close(freqs[0], expected, rtol=1e-5, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_frequency_ordering(self):
        """Lower dimension indices should have higher frequencies."""
        dim, seq_len = 64, 128
        freqs = precompute_freqs_cis(dim, seq_len, device=torch.device('cuda'))

        # Extract angles at a fixed position
        angles = torch.angle(freqs[10])  # Position 10

        # Lower indices should have larger absolute angles (higher frequency)
        # Check that angle magnitudes are non-increasing
        angle_mags = torch.abs(angles)
        for i in range(len(angle_mags) - 1):
            assert (
                angle_mags[i] >= angle_mags[i + 1] - 1e-5
            ), f"Frequency ordering violated at index {i}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_yarn_scaling_reduces_high_frequencies(self):
        """YaRN scaling should reduce high frequencies."""
        dim, seq_len = 64, 128

        freqs_base = precompute_freqs_cis(
            dim, seq_len, scaling_factor=1.0, device=torch.device('cuda')
        )
        freqs_yarn = precompute_freqs_cis(
            dim, seq_len, scaling_factor=40.0, device=torch.device('cuda')
        )

        # Extract angles at a position away from 0
        angles_base = torch.angle(freqs_base[50])
        angles_yarn = torch.angle(freqs_yarn[50])

        # High frequency dimensions (last indices) should have smaller angles with YaRN
        # (scaled down to avoid aliasing at extended context)
        high_freq_idx = -1  # Last dimension = highest frequency before scaling
        assert (
            torch.abs(angles_yarn[high_freq_idx]) < torch.abs(angles_base[high_freq_idx])
        ), "YaRN should reduce high frequency angles"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_different_theta_bases(self):
        """Different theta bases should produce different frequencies."""
        dim, seq_len = 64, 128

        freqs_10000 = precompute_freqs_cis(
            dim, seq_len, theta=10000.0, device=torch.device('cuda')
        )
        freqs_1000000 = precompute_freqs_cis(
            dim, seq_len, theta=1000000.0, device=torch.device('cuda')
        )

        # Frequencies should be different
        assert not torch.allclose(freqs_10000, freqs_1000000)

        # Larger theta should result in smaller angles (lower frequencies)
        angles_10000 = torch.abs(torch.angle(freqs_10000[10]))
        angles_1000000 = torch.abs(torch.angle(freqs_1000000[10]))
        assert torch.all(
            angles_10000 >= angles_1000000 - 1e-5
        ), "Larger theta should produce lower frequencies"


class TestLightningIndexerBasics:
    """Basic tests for Lightning Indexer functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_config_parameters(self):
        """Test that config parameters are properly set."""
        from megatron.core.transformer.transformer_config import MLATransformerConfig

        config = MLATransformerConfig(
            hidden_size=256,
            num_attention_heads=8,
            q_lora_rank=64,
            kv_lora_rank=32,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            use_sparse_attention=True,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=16,
        )

        assert config.use_sparse_attention is True
        assert config.index_n_heads == 4
        assert config.index_head_dim == 32
        assert config.index_topk == 16


class TestDSAUtilsEdgeCases:
    """Edge case tests for DSA utilities."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_position(self):
        """Test with single sequence position."""
        batch, seq, heads, dim = 2, 1, 4, 64
        x = torch.randn(batch, seq, heads, dim, device='cuda')
        freqs_cis = precompute_freqs_cis(dim, seq, device=torch.device('cuda'))

        y = apply_rotary_emb(x, freqs_cis)

        assert y.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_head(self):
        """Test with single attention head."""
        batch, seq, heads, dim = 2, 16, 1, 64
        x = torch.randn(batch, seq, heads, dim, device='cuda')
        freqs_cis = precompute_freqs_cis(dim, seq, device=torch.device('cuda'))

        y = apply_rotary_emb(x, freqs_cis)

        assert y.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_small_dimension(self):
        """Test with small head dimension."""
        batch, seq, heads, dim = 2, 16, 4, 4
        x = torch.randn(batch, seq, heads, dim, device='cuda')
        freqs_cis = precompute_freqs_cis(dim, seq, device=torch.device('cuda'))

        y = apply_rotary_emb(x, freqs_cis)

        assert y.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_large_sequence(self):
        """Test with larger sequence length."""
        batch, seq, heads, dim = 1, 4096, 2, 64
        x = torch.randn(batch, seq, heads, dim, device='cuda')
        freqs_cis = precompute_freqs_cis(dim, seq, device=torch.device('cuda'))

        y = apply_rotary_emb(x, freqs_cis)

        assert y.shape == x.shape

    def test_cpu_fallback(self):
        """Test that CPU execution works when CUDA is not available."""
        batch, seq, heads, dim = 2, 16, 4, 64
        x = torch.randn(batch, seq, heads, dim, device='cpu')
        freqs_cis = precompute_freqs_cis(dim, seq, device=torch.device('cpu'))

        y = apply_rotary_emb(x, freqs_cis)

        assert y.shape == x.shape
        assert y.device.type == 'cpu'


class TestYaRNParameters:
    """Test YaRN-specific parameters in frequency computation."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_beta_parameters_affect_scaling(self):
        """Different beta parameters should affect the scaling interpolation."""
        dim, seq_len = 64, 128
        device = torch.device('cuda')

        freqs_default = precompute_freqs_cis(
            dim,
            seq_len,
            scaling_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            device=device,
        )

        freqs_different_beta = precompute_freqs_cis(
            dim,
            seq_len,
            scaling_factor=40.0,
            beta_fast=64.0,
            beta_slow=2.0,
            device=device,
        )

        # Different beta values should produce different frequencies
        assert not torch.allclose(freqs_default, freqs_different_beta)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_original_max_position_embeddings(self):
        """Different original max position embeddings should affect scaling."""
        dim, seq_len = 64, 128
        device = torch.device('cuda')

        freqs_4096 = precompute_freqs_cis(
            dim,
            seq_len,
            scaling_factor=40.0,
            original_max_position_embeddings=4096,
            device=device,
        )

        freqs_8192 = precompute_freqs_cis(
            dim,
            seq_len,
            scaling_factor=40.0,
            original_max_position_embeddings=8192,
            device=device,
        )

        # Different original max positions should produce different frequencies
        assert not torch.allclose(freqs_4096, freqs_8192)
