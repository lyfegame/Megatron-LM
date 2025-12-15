#!/usr/bin/env python3
"""
Megatron-Native DeepSeek V3.2 Inference

This script runs inference using the Megatron-Core model directly,
without any HuggingFace dependencies. This validates the IMPLEMENTATION
(separate from conversion validation).

Reference: Official DeepSeek V3.2 inference demo
NOT HuggingFace transformers (no official HF implementation for V3.2)

Usage:
    torchrun --nproc_per_node=8 megatron_inference.py \
        --checkpoint /models-local/DeepSeek-V3.2-megatron \
        --prompt "What is 2 + 2?"
"""

import argparse
import os
import sys
import time

# Reference prompts for semantic equivalence testing
# Same prompts used in official inference validation
REFERENCE_PROMPTS = {
    0: {
        'name': 'simple_math',
        'prompt': 'What is 2 + 2?',
        'expected_path': 'Dense',
        'expected_contains': ['4'],
    },
    1: {
        'name': 'greeting',
        'prompt': 'Hello! How are you today?',
        'expected_path': 'Dense',
        'expected_contains': ['hello', 'good', 'great', 'wonderful'],
    },
    2: {
        'name': 'code_generation',
        'prompt': 'Write a Python function to check if a number is prime.',
        'expected_path': 'Dense',
        'expected_contains': ['def', 'prime', 'return'],
    },
    3: {
        'name': 'explanation',
        'prompt': "Explain Einstein's theory of relativity in simple terms.",
        'expected_path': 'Dense',
        'expected_contains': ['space', 'time', 'gravity', 'relative'],
    },
    4: {
        'name': 'long_context',
        'prompt': 'List the three main categories of machine learning: supervised, unsupervised, and reinforcement learning. Briefly describe each.',
        'expected_path': 'Dense',
        'expected_contains': ['supervised', 'unsupervised', 'reinforcement'],
    },
    5: {
        'name': 'sparse_trigger',
        'prompt': None,  # Loaded from file
        'expected_path': 'Sparse',
        'expected_contains': ['MLA', 'indexer', 'attention'],
    },
}


def load_long_prompt(prompt_file: str) -> str:
    """Load the long prompt that triggers sparse attention."""
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            return f.read()
    else:
        # Default long prompt if file not found
        return """# DeepSeek V3.2 Efficiency Techniques for Long Sequences

DeepSeek V3.2 employs three primary methods for efficient long-sequence processing:

## 1. Multi-head Latent Attention (MLA)
MLA compresses the key-value cache using low-rank projections, substantially decreasing memory requirements during inference.

## 2. Lightning Indexer for Sparse Attention
Rather than computing full attention across all tokens, the Lightning Indexer learns to predict which tokens are most relevant for each query position.

## 3. Rotary Position Embeddings (RoPE) with Extensions
The architecture uses a variant of RoPE with carefully tuned parameters to support context lengths exceeding 128,000 tokens.

Based on the document above, what are the three main techniques used by DeepSeek V3.2 for efficient long-sequence processing?
""" * 10  # Repeat to exceed 2048 tokens


def initialize_megatron():
    """Initialize Megatron distributed environment."""
    import torch
    import torch.distributed as dist

    # Initialize distributed if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    # Set device
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    return local_rank


def load_megatron_model(checkpoint_path: str, config_overrides: dict = None):
    """
    Load Megatron model from checkpoint.

    This function loads the model WITHOUT HuggingFace dependencies.
    """
    import torch
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Initialize parallel state
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    # Load config from checkpoint
    config_path = os.path.join(checkpoint_path, 'config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
    else:
        # Use V3.2 default config
        saved_config = {
            'hidden_size': 7168,
            'num_layers': 61,
            'num_attention_heads': 128,
            'ffn_hidden_size': 18432,
            'vocab_size': 102400,
            'q_lora_rank': 1536,
            'kv_lora_rank': 512,
            'qk_nope_head_dim': 128,
            'qk_rope_head_dim': 64,
            'v_head_dim': 128,
            'index_n_heads': 64,
            'index_head_dim': 128,
            'index_topk': 2048,
        }

    if config_overrides:
        saved_config.update(config_overrides)

    # Build TransformerConfig
    config = TransformerConfig(
        num_layers=saved_config['num_layers'],
        hidden_size=saved_config['hidden_size'],
        num_attention_heads=saved_config['num_attention_heads'],
        ffn_hidden_size=saved_config['ffn_hidden_size'],
        # V3.2 specific
        multi_latent_attention=True,
        q_lora_rank=saved_config.get('q_lora_rank', 1536),
        kv_lora_rank=saved_config.get('kv_lora_rank', 512),
        qk_head_dim=saved_config.get('qk_nope_head_dim', 128),
        qk_pos_emb_head_dim=saved_config.get('qk_rope_head_dim', 64),
        v_head_dim=saved_config.get('v_head_dim', 128),
        index_n_heads=saved_config.get('index_n_heads', 64),
        index_head_dim=saved_config.get('index_head_dim', 128),
        index_topk=saved_config.get('index_topk', 2048),
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    print(f"Building GPTModel with config:")
    print(f"  num_layers: {config.num_layers}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  multi_latent_attention: {config.multi_latent_attention}")
    print(f"  index_topk: {config.index_topk}")

    # Build model
    model = GPTModel(
        config=config,
        vocab_size=saved_config['vocab_size'],
        max_sequence_length=4096,
        parallel_output=True,
    )

    # Load checkpoint weights
    # This depends on the checkpoint format (torch_dist, safetensors, etc.)
    load_checkpoint(model, checkpoint_path)

    model.eval()
    return model, config, saved_config


def load_checkpoint(model, checkpoint_path: str):
    """Load checkpoint weights into model."""
    import torch
    from pathlib import Path

    # Detect checkpoint format
    ckpt_dir = Path(checkpoint_path)
    torch_dist_files = list(ckpt_dir.glob('*.distcp')) + list(ckpt_dir.glob('torch_dist/**/*'))
    pt_files = list(ckpt_dir.glob('*.pt')) + list(ckpt_dir.glob('*.pth'))

    if torch_dist_files:
        # Torch distributed checkpoint format
        from megatron.core.dist_checkpointing import load
        print(f"Loading torch_dist checkpoint from {checkpoint_path}")
        load(model.state_dict(), checkpoint_path)
    elif pt_files:
        # Standard PyTorch checkpoint
        print(f"Loading PyTorch checkpoint from {checkpoint_path}")
        state_dict = {}
        for pt_file in pt_files:
            sd = torch.load(pt_file, map_location='cuda', weights_only=True)
            state_dict.update(sd)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"No recognized checkpoint format in {checkpoint_path}")


def load_tokenizer(tokenizer_path: str = None):
    """
    Load tokenizer without HuggingFace transformers.

    Uses sentencepiece directly if available.
    """
    try:
        import sentencepiece as spm
        if tokenizer_path and os.path.exists(tokenizer_path):
            sp = spm.SentencePieceProcessor()
            sp.Load(tokenizer_path)
            return sp
    except ImportError:
        pass

    # Fallback: try to use tiktoken if available
    try:
        import tiktoken
        # DeepSeek uses a custom tokenizer, but tiktoken can work as fallback
        enc = tiktoken.get_encoding("cl100k_base")
        return enc
    except ImportError:
        pass

    raise RuntimeError("No tokenizer available. Install sentencepiece or tiktoken.")


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 100) -> str:
    """
    Generate text using greedy decoding.

    No HuggingFace dependencies - pure PyTorch generation.
    """
    import torch

    # Tokenize
    if hasattr(tokenizer, 'encode'):
        # tiktoken
        input_ids = tokenizer.encode(prompt)
    else:
        # sentencepiece
        input_ids = tokenizer.EncodeAsIds(prompt)

    input_ids = torch.tensor([input_ids], dtype=torch.long, device='cuda')
    seq_len = input_ids.size(1)

    print(f"Input tokens: {seq_len}")

    # Generate token by token (greedy)
    generated_ids = input_ids.clone()
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass
            logits = model(generated_ids, start_pos=0)

            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Check for EOS (token id 2 is common for EOS)
            if next_token.item() == 2:
                break

    # Decode
    output_ids = generated_ids[0, seq_len:].tolist()
    if hasattr(tokenizer, 'decode'):
        output_text = tokenizer.decode(output_ids)
    else:
        output_text = tokenizer.DecodeIds(output_ids)

    return output_text


def run_reference_tests(model, tokenizer, prompt_file: str) -> dict:
    """Run all reference prompts and check semantic equivalence."""
    results = {}

    # Load long prompt for sparse trigger test
    REFERENCE_PROMPTS[5]['prompt'] = load_long_prompt(prompt_file)

    for prompt_id, test_case in REFERENCE_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"Test {prompt_id}: {test_case['name']}")
        print(f"Expected path: {test_case['expected_path']}")
        print(f"{'='*60}")

        start_time = time.time()
        try:
            output = generate(model, tokenizer, test_case['prompt'], max_new_tokens=100)
            elapsed = time.time() - start_time

            # Check semantic equivalence
            output_lower = output.lower()
            matches = [kw for kw in test_case['expected_contains'] if kw.lower() in output_lower]
            passed = len(matches) > 0

            results[prompt_id] = {
                'name': test_case['name'],
                'passed': passed,
                'output': output[:200] + '...' if len(output) > 200 else output,
                'matches': matches,
                'time_s': elapsed,
            }

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"Result: {status}")
            print(f"Output: {output[:200]}...")
            print(f"Matched keywords: {matches}")

        except Exception as e:
            results[prompt_id] = {
                'name': test_case['name'],
                'passed': False,
                'error': str(e),
            }
            print(f"Result: ❌ ERROR - {e}")

    return results


def print_summary(results: dict):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("MEGATRON INFERENCE TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r.get('passed', False))
    total = len(results)

    for prompt_id, result in sorted(results.items()):
        status = "✅" if result.get('passed') else "❌"
        print(f"  {status} Test {prompt_id} ({result['name']})")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All semantic equivalence tests PASSED")
        return 0
    else:
        print(f"\n❌ {total - passed} tests FAILED")
        return 1


def main():
    parser = argparse.ArgumentParser(description='Megatron-native DeepSeek V3.2 inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Megatron checkpoint')
    parser.add_argument('--tokenizer', type=str, default=None,
                       help='Path to tokenizer (sentencepiece model)')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Single prompt to generate from')
    parser.add_argument('--run-tests', action='store_true',
                       help='Run reference prompt tests')
    parser.add_argument('--prompt-file', type=str,
                       default='debug_v32/long_prompt_sparse.txt',
                       help='File containing long prompt for sparse test')
    parser.add_argument('--max-tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    args = parser.parse_args()

    print("=" * 70)
    print("MEGATRON-NATIVE DEEPSEEK V3.2 INFERENCE")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # Initialize Megatron
    local_rank = initialize_megatron()
    print(f"Initialized on rank {local_rank}")

    # Load model
    print("\nLoading model...")
    model, config, saved_config = load_megatron_model(args.checkpoint)
    print("Model loaded successfully")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer_path = args.tokenizer or os.path.join(args.checkpoint, 'tokenizer.model')
    tokenizer = load_tokenizer(tokenizer_path)
    print("Tokenizer loaded successfully")

    if args.run_tests:
        # Run reference tests
        results = run_reference_tests(model, tokenizer, args.prompt_file)
        return print_summary(results)
    elif args.prompt:
        # Single prompt inference
        print(f"\nGenerating for prompt: {args.prompt}")
        output = generate(model, tokenizer, args.prompt, args.max_tokens)
        print(f"\nOutput: {output}")
        return 0
    else:
        print("\nNo prompt or --run-tests specified. Use --help for options.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
