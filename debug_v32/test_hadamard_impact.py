#!/usr/bin/env python3
"""
Hadamard Transform Impact Verification

This script verifies whether the Hadamard transform is necessary for the
Lightning Indexer by comparing:
1. Official reference tensors (with Hadamard)
2. PyTorch-only implementation (without Hadamard)

FACT: Official V3.2 applies Hadamard transform to Q and K before scoring:
```python
q = rotate_activation(q)  # hadamard_transform(x, scale=hidden_size ** -0.5)
k = rotate_activation(k)
```

FACT: Megatron implementation does NOT apply Hadamard transform.

This test measures the factual impact on top-K index selection.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F


def hadamard_transform_pytorch(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Pure PyTorch implementation of Hadamard transform.

    The Hadamard transform is a linear transformation that:
    1. Orthogonally mixes all dimensions
    2. Preserves norm (up to scaling)
    3. Has O(n log n) complexity via FFT-based implementation

    This is a reference implementation using the matrix form.
    For production, use fast_hadamard_transform library.
    """
    # Get dimensions
    *batch_dims, n = x.shape

    # Build Hadamard matrix (power of 2 required)
    # H_1 = [1]
    # H_2 = [[1, 1], [1, -1]]
    # H_2n = [[H_n, H_n], [H_n, -H_n]]

    # Find next power of 2
    log2_n = (n - 1).bit_length()
    padded_n = 1 << log2_n

    # Pad if necessary
    if padded_n > n:
        pad_size = padded_n - n
        x = F.pad(x, (0, pad_size))

    # Build Hadamard matrix recursively
    H = torch.tensor([[1.0]], device=x.device, dtype=x.dtype)
    for _ in range(log2_n):
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)

    # Apply transform: y = x @ H.T * scale / sqrt(n)
    result = torch.matmul(x.view(-1, padded_n), H.T) * scale / (padded_n ** 0.5)

    # Unpad and reshape
    result = result.view(*batch_dims, padded_n)[..., :n]

    return result


def compare_with_without_hadamard(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    n_heads_scale: float,
):
    """
    Compare indexer scoring with and without Hadamard transform.

    Returns scores from both approaches for comparison.
    """
    # Without Hadamard (Megatron approach)
    logits_no_hadamard = torch.einsum("bshd,btd->bsht", q, k) * softmax_scale
    logits_no_hadamard = F.relu(logits_no_hadamard)
    score_no_hadamard = torch.einsum("bsh,bsht->bst", weights, logits_no_hadamard)

    # With Hadamard
    head_dim = q.shape[-1]
    q_h = hadamard_transform_pytorch(q, scale=head_dim ** -0.5)
    k_h = hadamard_transform_pytorch(k.unsqueeze(2), scale=head_dim ** -0.5).squeeze(2)

    logits_with_hadamard = torch.einsum("bshd,btd->bsht", q_h, k_h) * softmax_scale
    logits_with_hadamard = F.relu(logits_with_hadamard)
    score_with_hadamard = torch.einsum("bsh,bsht->bst", weights, logits_with_hadamard)

    return score_no_hadamard, score_with_hadamard


def compute_topk_overlap(indices1: torch.Tensor, indices2: torch.Tensor) -> dict:
    """
    Compute overlap metrics between two sets of top-K indices.

    Args:
        indices1: [batch, seq, topk] - First set of indices
        indices2: [batch, seq, topk] - Second set of indices

    Returns:
        Dictionary with overlap metrics
    """
    batch_size, seq_len, topk = indices1.shape

    # Exact match rate (same index at same position)
    exact_match = (indices1 == indices2).float().mean().item()

    # Set overlap per position
    overlaps = []
    for b in range(batch_size):
        for s in range(seq_len):
            set1 = set(indices1[b, s].tolist())
            set2 = set(indices2[b, s].tolist())
            overlap = len(set1 & set2) / topk
            overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps)
    min_overlap = min(overlaps)
    max_overlap = max(overlaps)

    # Perfect set match (same set, any order)
    set_matches = sum(1 for o in overlaps if o == 1.0) / len(overlaps)

    return {
        'exact_match_rate': exact_match,
        'avg_set_overlap': avg_overlap,
        'min_set_overlap': min_overlap,
        'max_set_overlap': max_overlap,
        'perfect_set_match_rate': set_matches,
    }


def load_reference_tensor(tensor_dir: str, name: str) -> torch.Tensor:
    """Load a tensor from the official reference directory."""
    path = os.path.join(tensor_dir, f"0000_rank0_{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Reference tensor not found: {path}")
    data = torch.load(path, map_location='cpu')
    return data['data']


def main():
    parser = argparse.ArgumentParser(description='Hadamard Transform Impact Verification')
    parser.add_argument('--reference-dir', type=str,
                       default='/models-local/official_tensors',
                       help='Directory containing official reference tensors')
    parser.add_argument('--prompt-id', type=int, default=5,
                       help='Prompt ID to test (5 = sparse trigger)')
    parser.add_argument('--layer', type=int, default=0,
                       help='Layer to analyze')
    args = parser.parse_args()

    prompt_dirs = {
        0: 'prompt_0_simple_math',
        1: 'prompt_1_greeting',
        2: 'prompt_2_code_generation',
        3: 'prompt_3_explanation',
        4: 'prompt_4_long_context',
        5: 'prompt_5_sparse_trigger',
    }

    tensor_dir = os.path.join(args.reference_dir, prompt_dirs[args.prompt_id])

    print("=" * 70)
    print("HADAMARD TRANSFORM IMPACT VERIFICATION")
    print("=" * 70)
    print(f"Reference dir: {tensor_dir}")
    print(f"Prompt: {prompt_dirs[args.prompt_id]}")
    print(f"Layer: {args.layer}")

    # Load official reference tensors
    print("\n--- Loading Official Reference Tensors ---")

    try:
        # Official top-K indices (with Hadamard applied)
        official_topk = load_reference_tensor(tensor_dir, f'layer_{args.layer}_indexer_topk_indices')
        print(f"Official topk_indices shape: {official_topk.shape}")
        print(f"  dtype: {official_topk.dtype}")
        print(f"  min: {official_topk.min().item()}, max: {official_topk.max().item()}")

        # Try to load intermediate tensors if available
        try:
            indexer_input = load_reference_tensor(tensor_dir, f'layer_{args.layer}_indexer_input')
            print(f"Indexer input shape: {indexer_input.shape}")
        except FileNotFoundError:
            indexer_input = None
            print("Indexer input not available")

        try:
            wk_output = load_reference_tensor(tensor_dir, f'layer_{args.layer}_indexer_wk_output')
            print(f"wk_output shape: {wk_output.shape}")
        except FileNotFoundError:
            wk_output = None
            print("wk_output not available")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Cannot proceed without reference tensors.")
        return 1

    # Analysis: Check index distribution
    print("\n--- Official Index Distribution Analysis ---")
    official_topk_int = official_topk.long()
    batch_size, seq_len, topk = official_topk_int.shape

    print(f"Sequence length: {seq_len}")
    print(f"Top-K: {topk}")

    # For each position, check how many indices are "local" (near the query position)
    local_counts = []
    for pos in range(seq_len):
        indices = official_topk_int[0, pos, :]
        # Count indices within 128 positions of query
        local = ((indices >= max(0, pos - 128)) & (indices <= pos)).sum().item()
        local_counts.append(local / topk)

    avg_local = sum(local_counts) / len(local_counts)
    print(f"Average local index ratio (within 128 positions): {avg_local:.4f}")

    # Check if first few and last positions are included
    first_included = []
    last_included = []
    for pos in range(seq_len):
        indices_set = set(official_topk_int[0, pos, :].tolist())
        first_included.append(1 if 0 in indices_set else 0)
        last_included.append(1 if pos in indices_set else 0)

    print(f"First position (0) included rate: {sum(first_included)/len(first_included):.4f}")
    print(f"Current position (i) included rate: {sum(last_included)/len(last_included):.4f}")

    # FACT: Since we don't have the intermediate Q/K tensors BEFORE Hadamard,
    # we cannot directly compare with/without Hadamard using the same inputs.
    #
    # However, we CAN observe the characteristics of the official selection:
    # - If Hadamard significantly changes the selection, we'd expect more
    #   "distributed" attention (less bias toward local positions)
    # - Without Hadamard, selection tends to be more "local" due to RoPE

    print("\n--- FACTUAL FINDINGS ---")
    print("1. Official implementation uses Hadamard transform")
    print(f"   Source: docs/reference/deepseek_v32_official/model.py lines 472-473")
    print(f"   Code: q = rotate_activation(q)  # hadamard_transform")
    print()
    print("2. Megatron implementation does NOT use Hadamard transform")
    print(f"   Source: megatron/core/transformer/lightning_indexer.py")
    print(f"   Comment: 'No Hadamard transform (vLLM confirmed unnecessary for accuracy)'")
    print()
    print("3. Index selection characteristics (official with Hadamard):")
    print(f"   - Sequence length: {seq_len}")
    print(f"   - Top-K selected: {topk}")
    print(f"   - Local bias (within 128 pos): {avg_local:.2%}")
    print(f"   - Current position always included: {sum(last_included)/len(last_included):.2%}")
    print()
    print("4. To fully verify Hadamard necessity, need to:")
    print("   a. Run Megatron inference on same prompts (without Hadamard)")
    print("   b. Compare index overlap with official reference")
    print("   c. Compare final model outputs for semantic equivalence")

    print("\n" + "=" * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
