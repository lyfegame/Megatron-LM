#!/usr/bin/env python3
"""
Run HuggingFace V3.2 Fork Verification Test

This script runs all 6 reference prompts on the HF V3.2 fork to verify:
1. Semantic equivalence with official outputs
2. Sparse attention triggering (prompt 5)
3. No crashes or OOM errors

Based on methodology from debug_v32/VERIFICATION_METHODOLOGY.md
"""

import argparse
import json
import os
import sys
import time
import torch
from pathlib import Path


def load_reference_prompts(prompts_file: str) -> dict:
    """Load reference prompts from JSON file."""
    with open(prompts_file, 'r') as f:
        return json.load(f)


def build_device_map(config, num_gpus: int = 8):
    """Build device map for multi-GPU inference."""
    device_map = {}
    device_map['model.embed_tokens'] = 0
    device_map['model.norm'] = num_gpus - 1
    device_map['lm_head'] = num_gpus - 1

    layers_per_gpu = config.num_hidden_layers // num_gpus
    for i in range(config.num_hidden_layers):
        gpu_id = min(i // layers_per_gpu, num_gpus - 1)
        device_map[f'model.layers.{i}'] = gpu_id

    return device_map


def run_generation(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    """Run generation on a single prompt."""
    messages = [{"role": "user", "content": prompt}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract assistant response
    if '<｜Assistant｜>' in response:
        response = response.split('<｜Assistant｜>')[-1].strip()
    elif 'Assistant:' in response:
        response = response.split('Assistant:')[-1].strip()

    return response, input_ids.shape[1]


def check_semantic_equivalence(output: str, expected: str) -> tuple:
    """
    Check if output is semantically equivalent to expected.

    Returns: (is_equivalent: bool, reason: str)
    """
    output_lower = output.lower().strip()
    expected_lower = expected.lower().strip()

    # Direct containment check
    if expected_lower in output_lower:
        return True, "Expected text found in output"

    # For math problems
    if "4" in output and "2+2" in expected_lower or "2 + 2" in expected_lower:
        return True, "Correct answer (4) found"

    # For code generation
    if "def is_prime" in output and "is_prime" in expected:
        return True, "is_prime function generated"

    # For ML categories
    if all(cat.lower() in output_lower for cat in ["supervised", "unsupervised", "reinforcement"]):
        return True, "All three ML categories mentioned"

    # For relativity explanation
    if "relativity" in expected_lower:
        if "space" in output_lower and "time" in output_lower:
            return True, "Relativity concepts (space, time) discussed"

    # For greeting
    if "hello" in expected_lower or "how are you" in expected_lower:
        if any(word in output_lower for word in ["hello", "hi", "well", "doing"]):
            return True, "Appropriate greeting response"

    # For sparse trigger (MLA/techniques)
    if "mla" in expected_lower or "lightning" in expected_lower:
        if any(word in output_lower for word in ["mla", "attention", "sparse", "indexer", "lightning"]):
            return True, "Efficient attention techniques discussed"

    return False, "Semantic check failed"


def main():
    parser = argparse.ArgumentParser(description='HF V3.2 Verification Test')
    parser.add_argument('--checkpoint', type=str,
                       default='/models-local/DeepSeek-V3.2-bf16',
                       help='Path to BF16 checkpoint')
    parser.add_argument('--prompts-file', type=str,
                       default='debug_v32/reference_prompts.json',
                       help='Path to reference prompts JSON')
    parser.add_argument('--output-file', type=str,
                       default='debug_v32/hf_v32_test_results.json',
                       help='Output file for results')
    parser.add_argument('--prompt-ids', type=str, default='0,1,2,3,4,5',
                       help='Comma-separated prompt IDs to run')
    parser.add_argument('--num-gpus', type=int, default=8,
                       help='Number of GPUs to use')
    args = parser.parse_args()

    print("=" * 70)
    print("HF V3.2 FORK VERIFICATION TEST")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompts file: {args.prompts_file}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Load reference prompts
    ref_data = load_reference_prompts(args.prompts_file)
    prompt_ids = [int(x) for x in args.prompt_ids.split(',')]

    print(f"Running prompts: {prompt_ids}")

    # Load config
    print("\nLoading config...")
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    print(f"  model_type: {config.model_type}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  index_topk: {config.index_topk}")

    # Build device map
    device_map = build_device_map(config, args.num_gpus)
    print(f"  Distributing {config.num_hidden_layers} layers across {args.num_gpus} GPUs")

    # Load model
    print("\nLoading model (this may take a few minutes)...")
    start_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    load_time = time.time() - start_time
    print(f"  Model loaded in {load_time:.1f}s")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Run tests
    results = {
        'checkpoint': args.checkpoint,
        'prompts': [],
        'summary': {
            'total': 0,
            'passed': 0,
            'failed': 0,
        }
    }

    for prompt_data in ref_data['prompts']:
        prompt_id = prompt_data['id']
        if prompt_id not in prompt_ids:
            continue

        print(f"\n{'='*60}")
        print(f"PROMPT {prompt_id}: {prompt_data['name']}")
        print(f"{'='*60}")

        # Handle prompt_file for prompt 5
        if 'prompt_file' in prompt_data:
            prompt_file = os.path.join(os.path.dirname(args.prompts_file), prompt_data['prompt_file'])
            with open(prompt_file, 'r') as f:
                prompt = f.read()
        else:
            prompt = prompt_data['prompt']

        print(f"Prompt: {prompt[:100]}...")
        print(f"Expected: {prompt_data['expected_output'][:100]}...")

        try:
            # Run generation
            start_time = time.time()
            output, input_tokens = run_generation(
                model, tokenizer, prompt,
                max_new_tokens=ref_data['settings']['max_new_tokens']
            )
            gen_time = time.time() - start_time

            print(f"\nInput tokens: {input_tokens}")
            print(f"Sparse attention active: {input_tokens > config.index_topk}")
            print(f"Generation time: {gen_time:.2f}s")
            print(f"\nOutput: {output[:300]}...")

            # Check semantic equivalence
            is_equiv, reason = check_semantic_equivalence(output, prompt_data['expected_output'])

            result = {
                'id': prompt_id,
                'name': prompt_data['name'],
                'input_tokens': input_tokens,
                'sparse_active': input_tokens > config.index_topk,
                'output': output,
                'expected': prompt_data['expected_output'],
                'semantic_equivalent': is_equiv,
                'reason': reason,
                'generation_time': gen_time,
                'status': 'PASS' if is_equiv else 'FAIL'
            }

            if is_equiv:
                print(f"\n✓ PASS: {reason}")
                results['summary']['passed'] += 1
            else:
                print(f"\n✗ FAIL: {reason}")
                results['summary']['failed'] += 1

        except Exception as e:
            print(f"\n✗ ERROR: {str(e)}")
            result = {
                'id': prompt_id,
                'name': prompt_data['name'],
                'error': str(e),
                'status': 'ERROR'
            }
            results['summary']['failed'] += 1

        results['prompts'].append(result)
        results['summary']['total'] += 1

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {results['summary']['total']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")

    # Return exit code based on results
    return 0 if results['summary']['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
