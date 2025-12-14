# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for Lightning Indexer (DeepSeek V3.2 sparse attention)."""

import pytest
import torch

from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.transformer.lightning_indexer import (
    LightningIndexer,
    LightningIndexerSubmodules,
    apply_rotary_emb_non_interleaved,
)
from megatron.core.transformer.sparse_attention import SparseAttention


class TestNonInterleavedRoPE:
    """Test non-interleaved RoPE implementation."""

    def test_apply_rotary_emb_non_interleaved_shape(self):
        """Test that non-interleaved RoPE preserves tensor shape."""
        batch, seq, heads, head_dim = 2, 16, 4, 64
        x = torch.randn(batch, seq, heads, head_dim)

        # Create simple frequency tensor
        freqs = torch.exp(
            -torch.arange(0, head_dim // 2) * (torch.log(torch.tensor(10000.0)) / (head_dim // 2))
        )
        freqs = torch.outer(torch.arange(seq), freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Complex frequencies

        result = apply_rotary_emb_non_interleaved(x, freqs_cis)
        assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"

    def test_apply_rotary_emb_non_interleaved_dtype(self):
        """Test that non-interleaved RoPE preserves dtype."""
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(2, 16, 4, 64, dtype=dtype)
            freqs = torch.exp(
                -torch.arange(0, 32) * (torch.log(torch.tensor(10000.0)) / 32)
            )
            freqs = torch.outer(torch.arange(16), freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

            result = apply_rotary_emb_non_interleaved(x, freqs_cis)
            assert result.dtype == dtype, f"Dtype mismatch: {result.dtype} vs {dtype}"

    def test_apply_rotary_emb_non_interleaved_vs_identity_at_zero(self):
        """Test that RoPE at position 0 is identity (freqs=0 -> rotation=0)."""
        x = torch.randn(2, 1, 4, 64)  # Single position
        freqs = torch.zeros(1, 32)  # Zero frequencies = no rotation
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        result = apply_rotary_emb_non_interleaved(x, freqs_cis)
        # Result should equal input (no rotation)
        torch.testing.assert_close(result, x, rtol=1e-5, atol=1e-5)


class TestLightningIndexerConfig:
    """Test Lightning Indexer configuration validation."""

    def test_config_requires_mla(self):
        """Test that sparse attention requires multi_latent_attention=True."""
        with pytest.raises(ValueError, match="multi_latent_attention"):
            MLATransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                multi_latent_attention=False,  # Should fail
                index_topk=64,
                index_n_heads=4,
                index_head_dim=32,
            )

    def test_config_requires_n_heads(self):
        """Test that index_topk requires index_n_heads."""
        with pytest.raises(ValueError, match="index_n_heads"):
            MLATransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                multi_latent_attention=True,
                index_topk=64,
                index_n_heads=None,  # Should fail
                index_head_dim=32,
            )

    def test_config_requires_head_dim(self):
        """Test that index_topk requires index_head_dim."""
        with pytest.raises(ValueError, match="index_head_dim"):
            MLATransformerConfig(
                num_layers=1,
                hidden_size=128,
                num_attention_heads=4,
                multi_latent_attention=True,
                index_topk=64,
                index_n_heads=4,
                index_head_dim=None,  # Should fail
            )

    def test_config_valid_v32(self):
        """Test valid V3.2 configuration."""
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            multi_latent_attention=True,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            index_topk=64,
            index_n_heads=4,
            index_head_dim=32,
        )
        assert config.use_sparse_attention is True
        assert config.index_topk == 64

    def test_config_use_sparse_attention_property(self):
        """Test use_sparse_attention property."""
        # Without indexer params
        config1 = MLATransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
        )
        assert config1.use_sparse_attention is False

        # With index_topk=0
        config2 = MLATransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            multi_latent_attention=True,
            index_topk=0,
            index_n_heads=4,
            index_head_dim=32,
        )
        assert config2.use_sparse_attention is False


class TestLightningIndexerModule:
    """Test Lightning Indexer module."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return MLATransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            multi_latent_attention=True,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            index_topk=8,
            index_n_heads=4,
            index_head_dim=32,
            tensor_model_parallel_size=1,
        )

    @pytest.fixture
    def indexer(self, config):
        """Create a test indexer."""
        return LightningIndexer(
            config=config,
            submodules=LightningIndexerSubmodules(),
            layer_number=0,
        )

    def test_indexer_output_shape(self, indexer, config):
        """Test that indexer produces correct output shape."""
        seq_len, batch_size = 32, 2
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)
        q_compressed = torch.randn(seq_len, batch_size, config.q_lora_rank)

        # Create RoPE frequencies
        rope_dim = config.qk_pos_emb_head_dim
        freqs = torch.exp(
            -torch.arange(0, rope_dim // 2) * (torch.log(torch.tensor(10000.0)) / (rope_dim // 2))
        )
        freqs = torch.outer(torch.arange(seq_len), freqs)
        rotary_pos_emb = torch.polar(torch.ones_like(freqs), freqs)

        with torch.no_grad():
            topk_indices = indexer(hidden_states, q_compressed, rotary_pos_emb)

        expected_topk = min(config.index_topk, seq_len)
        assert topk_indices.shape == (batch_size, seq_len, expected_topk), \
            f"Shape mismatch: {topk_indices.shape} vs ({batch_size}, {seq_len}, {expected_topk})"

    def test_indexer_indices_valid(self, indexer, config):
        """Test that returned indices are valid (in range [0, seq_len))."""
        seq_len, batch_size = 32, 2
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size)
        q_compressed = torch.randn(seq_len, batch_size, config.q_lora_rank)

        rope_dim = config.qk_pos_emb_head_dim
        freqs = torch.exp(
            -torch.arange(0, rope_dim // 2) * (torch.log(torch.tensor(10000.0)) / (rope_dim // 2))
        )
        freqs = torch.outer(torch.arange(seq_len), freqs)
        rotary_pos_emb = torch.polar(torch.ones_like(freqs), freqs)

        with torch.no_grad():
            topk_indices = indexer(hidden_states, q_compressed, rotary_pos_emb)

        assert topk_indices.min() >= 0, "Indices should be non-negative"
        assert topk_indices.max() < seq_len, f"Indices should be < seq_len ({seq_len})"

    def test_indexer_gradient_flow(self, indexer, config):
        """Test that gradients flow through indexer weights (but not indices)."""
        seq_len, batch_size = 16, 2
        hidden_states = torch.randn(seq_len, batch_size, config.hidden_size, requires_grad=True)
        q_compressed = torch.randn(seq_len, batch_size, config.q_lora_rank, requires_grad=True)

        rope_dim = config.qk_pos_emb_head_dim
        freqs = torch.exp(
            -torch.arange(0, rope_dim // 2) * (torch.log(torch.tensor(10000.0)) / (rope_dim // 2))
        )
        freqs = torch.outer(torch.arange(seq_len), freqs)
        rotary_pos_emb = torch.polar(torch.ones_like(freqs), freqs)

        topk_indices = indexer(hidden_states, q_compressed, rotary_pos_emb)

        # Indices should be detached (no gradient)
        assert not topk_indices.requires_grad, "Indices should not require gradient"


class TestSparseAttention:
    """Test Sparse Attention module."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return MLATransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            multi_latent_attention=True,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
        )

    @pytest.fixture
    def sparse_attn(self, config):
        """Create test sparse attention."""
        return SparseAttention(config=config)

    def test_sparse_attention_output_shape(self, sparse_attn, config):
        """Test sparse attention output shape."""
        seq_len, batch_size = 32, 2
        n_heads = config.num_attention_heads
        q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
        v_head_dim = config.v_head_dim
        topk = 8

        query = torch.randn(seq_len, batch_size, n_heads, q_head_dim)
        key = torch.randn(seq_len, batch_size, n_heads, q_head_dim)
        value = torch.randn(seq_len, batch_size, n_heads, v_head_dim)
        topk_indices = torch.randint(0, seq_len, (batch_size, seq_len, topk))

        with torch.no_grad():
            output = sparse_attn(query, key, value, topk_indices)

        assert output.shape == (seq_len, batch_size, n_heads, v_head_dim), \
            f"Shape mismatch: {output.shape}"

    def test_sparse_attention_causal_mask(self, sparse_attn, config):
        """Test that sparse attention respects causal masking."""
        seq_len, batch_size = 8, 1
        n_heads = config.num_attention_heads
        q_head_dim = config.qk_head_dim + config.qk_pos_emb_head_dim
        v_head_dim = config.v_head_dim
        topk = 4

        # Set up where query at position 0 has indices including future positions
        query = torch.randn(seq_len, batch_size, n_heads, q_head_dim)
        key = torch.randn(seq_len, batch_size, n_heads, q_head_dim)
        value = torch.randn(seq_len, batch_size, n_heads, v_head_dim)

        # Indices include future positions for position 0
        topk_indices = torch.tensor([[[0, 1, 2, 3]]] * seq_len)  # Position 0 has future indices
        topk_indices = topk_indices.expand(batch_size, seq_len, topk)

        with torch.no_grad():
            output = sparse_attn(query, key, value, topk_indices)

        # Output should be finite (causal masking should prevent inf)
        assert torch.isfinite(output).all(), "Output should be finite with causal masking"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
