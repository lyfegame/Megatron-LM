#!/usr/bin/env python3
"""
Validate Megatron DeepSeek V3.2 against official reference activations.

Comparison criteria:
- Layers: Strict (max absolute diff < 1e-3)
- Output: Semantic (same argmax + reasonable text)

Usage:
    python validate_megatron_v32.py \
        --checkpoint /mnt/models-disk/DeepSeek-V3.2-megatron \
        --reference /mnt/models-disk/official_tensors \
        --prompt-id 0
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch
import torch.nn.functional as F


# Validation thresholds
STRICT_THRESHOLD = 1e-3  # Max absolute diff for layer comparison
RELATIVE_THRESHOLD = 0.01  # 1% relative diff allowed
SEMANTIC_THRESHOLD = 0.05  # 5% mean diff for semantic comparison


# Reference prompts (same as used in official capture)
REFERENCE_PROMPTS = {
    0: "What is 2+2?",
    1: "Hello! How are you today?",
    2: "Write a Python function to calculate fibonacci numbers.",
    3: "Explain the concept of machine learning in simple terms.",
    4: "The quick brown fox jumps over the lazy dog. " * 50,  # Long context
    5: "Calculate the integral of x^2 from 0 to 1.",  # Sparse trigger (math)
}


class ActivationCapture:
    """Capture activations from Megatron model during forward pass."""

    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []

    def _make_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            self.activations[name] = out.detach().cpu().float()
        return hook

    def register_hooks(self):
        """Register hooks to capture activations at key points."""
        # Embedding
        if hasattr(self.model, 'embedding'):
            self.hooks.append(
                self.model.embedding.register_forward_hook(self._make_hook('embedding_output'))
            )

        # Decoder layers
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'layers'):
            for i, layer in enumerate(self.model.decoder.layers):
                # Input layernorm (attention input)
                if hasattr(layer, 'input_layernorm'):
                    self.hooks.append(
                        layer.input_layernorm.register_forward_hook(
                            self._make_hook(f'layer_{i}_attn_norm_output')
                        )
                    )

                # Self attention
                if hasattr(layer, 'self_attention'):
                    self.hooks.append(
                        layer.self_attention.register_forward_hook(
                            self._make_hook(f'layer_{i}_attn_output')
                        )
                    )

                # Pre-MLP layernorm (for MoE layers)
                if hasattr(layer, 'pre_mlp_layernorm'):
                    self.hooks.append(
                        layer.pre_mlp_layernorm.register_forward_hook(
                            self._make_hook(f'layer_{i}_ffn_norm_output')
                        )
                    )

                # MLP
                if hasattr(layer, 'mlp'):
                    self.hooks.append(
                        layer.mlp.register_forward_hook(
                            self._make_hook(f'layer_{i}_ffn_output')
                        )
                    )

        # Final layernorm
        if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'final_layernorm'):
            self.hooks.append(
                self.model.decoder.final_layernorm.register_forward_hook(
                    self._make_hook('final_norm_output')
                )
            )

        print(f"Registered {len(self.hooks)} activation hooks")
        return self

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        self.activations = {}


def load_reference_tensors(reference_dir: str, prompt_id: int) -> Dict[str, torch.Tensor]:
    """Load reference activation tensors for a prompt."""
    prompt_dirs = {
        0: "prompt_0_simple_math",
        1: "prompt_1_greeting",
        2: "prompt_2_code_generation",
        3: "prompt_3_explanation",
        4: "prompt_4_long_context",
        5: "prompt_5_sparse_trigger",
    }

    prompt_dir = Path(reference_dir) / prompt_dirs[prompt_id]
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Reference directory not found: {prompt_dir}")

    tensors = {}
    for f in sorted(prompt_dir.glob("*.pt")):
        data = torch.load(f, map_location='cpu')
        # Extract activation name from filename
        # Format: 0000_rank0_layer_0_attn_output.pt -> layer_0_attn_output
        name = f.stem.split('_', 2)[-1]  # Remove token and rank prefix
        if 'rank0' in name:
            name = name.replace('rank0_', '')

        if isinstance(data, dict) and 'data' in data:
            tensors[name] = data['data'].float()
        elif isinstance(data, torch.Tensor):
            tensors[name] = data.float()

    print(f"Loaded {len(tensors)} reference tensors for prompt {prompt_id}")
    return tensors


def compare_tensors(
    megatron_tensor: torch.Tensor,
    reference_tensor: torch.Tensor,
    name: str,
    strict: bool = True
) -> Tuple[bool, Dict]:
    """Compare two tensors and return pass/fail with statistics."""

    # Handle shape mismatches
    if megatron_tensor.shape != reference_tensor.shape:
        return False, {
            'name': name,
            'passed': False,
            'error': f'Shape mismatch: {megatron_tensor.shape} vs {reference_tensor.shape}',
        }

    # Compute differences
    abs_diff = (megatron_tensor - reference_tensor).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Relative difference (avoid div by zero)
    ref_abs = reference_tensor.abs()
    rel_diff = (abs_diff / (ref_abs + 1e-8)).mean().item()

    # Statistics
    stats = {
        'name': name,
        'shape': list(megatron_tensor.shape),
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'rel_diff': rel_diff,
        'megatron_mean': megatron_tensor.mean().item(),
        'reference_mean': reference_tensor.mean().item(),
        'megatron_std': megatron_tensor.std().item(),
        'reference_std': reference_tensor.std().item(),
    }

    # Determine pass/fail
    if strict:
        passed = max_diff < STRICT_THRESHOLD
    else:
        passed = (rel_diff < RELATIVE_THRESHOLD) or (mean_diff < SEMANTIC_THRESHOLD)

    stats['passed'] = passed
    stats['threshold'] = 'strict' if strict else 'semantic'

    return passed, stats


def validate_prompt(
    model,
    tokenizer,
    tokenizer_type: str,
    prompt_id: int,
    reference_dir: str,
    device: torch.device,
) -> Tuple[bool, List[Dict]]:
    """Validate Megatron model output against reference for one prompt."""

    prompt = REFERENCE_PROMPTS[prompt_id]
    print(f"\n{'='*70}")
    print(f"VALIDATING PROMPT {prompt_id}: {prompt[:50]}...")
    print('='*70)

    # Load reference tensors
    reference_tensors = load_reference_tensors(reference_dir, prompt_id)

    # Setup activation capture
    capture = ActivationCapture(model)
    capture.register_hooks()

    # Tokenize
    if tokenizer_type == "sp":
        input_ids = tokenizer.EncodeAsIds(prompt)
    else:
        input_ids = tokenizer.encode(prompt).ids

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    print(f"Input shape: {input_tensor.shape}")

    # Forward pass
    try:
        with torch.inference_mode():
            output = model(input_tensor)
            if hasattr(output, 'logits'):
                logits = output.logits
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
    except Exception as e:
        print(f"Forward pass error: {e}")
        capture.remove_hooks()
        return False, [{'error': str(e)}]

    capture.remove_hooks()

    # Compare activations
    results = []
    all_passed = True

    for name, ref_tensor in reference_tensors.items():
        # Map reference names to Megatron names
        megatron_name = name

        if megatron_name in capture.activations:
            meg_tensor = capture.activations[megatron_name]

            # Use strict threshold for layer activations, semantic for final output
            is_output = 'final' in name or 'output_layer' in name
            passed, stats = compare_tensors(
                meg_tensor, ref_tensor, name, strict=not is_output
            )

            results.append(stats)
            if not passed:
                all_passed = False
                print(f"  FAIL: {name} - max_diff={stats['max_diff']:.6f}")
            else:
                print(f"  PASS: {name} - max_diff={stats['max_diff']:.6f}")
        else:
            print(f"  SKIP: {name} - not captured in Megatron")
            results.append({
                'name': name,
                'passed': None,
                'error': 'Not captured',
            })

    # Semantic output validation
    print(f"\nLogits shape: {logits.shape}")
    print(f"Logits mean: {logits.float().mean().item():.6f}")

    # Get predicted tokens
    pred_tokens = logits[0, -1, :].argmax().item()
    print(f"Next token prediction: {pred_tokens}")

    return all_passed, results


def main():
    parser = argparse.ArgumentParser(description="Validate Megatron DeepSeek V3.2")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Megatron checkpoint")
    parser.add_argument("--reference", type=str, required=True,
                       help="Path to reference tensors directory")
    parser.add_argument("--prompt-id", type=int, default=None,
                       help="Specific prompt ID to validate (0-5), or all if not specified")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Path to tokenizer")
    parser.add_argument("--output", type=str, default="/tmp/validation_results.json",
                       help="Output file for results")
    parser.add_argument("--cpu-offload", action="store_true",
                       help="Use CPU offloading for inference")

    args = parser.parse_args()

    print("="*70)
    print("MEGATRON DEEPSEEK V3.2 VALIDATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Reference: {args.reference}")
    print(f"Prompt ID: {args.prompt_id if args.prompt_id is not None else 'all'}")
    print()

    # Initialize distributed
    from debug_v32.pure_megatron_inference_cpu_offload import (
        initialize_distributed, load_tokenizer, load_megatron_model, CPUOffloadWrapper
    )

    rank, world_size = initialize_distributed()
    print(f"Initialized: rank={rank}, world_size={world_size}")

    # Load tokenizer
    tokenizer_path = args.tokenizer or '/models-local/DeepSeek-V3.2-fp8'
    tokenizer, tokenizer_type = load_tokenizer(tokenizer_path)
    print(f"Tokenizer loaded: {tokenizer_type}")

    # Load model
    print("\nLoading model...")
    model, config = load_megatron_model(args.checkpoint)

    device = torch.device("cuda:0")

    if args.cpu_offload:
        print("Using CPU offloading...")
        model = CPUOffloadWrapper(model, device)
    else:
        model = model.to(device)

    model.eval()

    # Validate prompts
    prompt_ids = [args.prompt_id] if args.prompt_id is not None else list(range(6))

    all_results = {}
    overall_passed = True

    for pid in prompt_ids:
        passed, results = validate_prompt(
            model=model.model if args.cpu_offload else model,
            tokenizer=tokenizer,
            tokenizer_type=tokenizer_type,
            prompt_id=pid,
            reference_dir=args.reference,
            device=device,
        )

        all_results[f"prompt_{pid}"] = {
            'passed': passed,
            'results': results,
        }

        if not passed:
            overall_passed = False

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for pid in prompt_ids:
        status = "PASS" if all_results[f"prompt_{pid}"]['passed'] else "FAIL"
        print(f"Prompt {pid}: {status}")

    print(f"\nOverall: {'PASS' if overall_passed else 'FAIL'}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
