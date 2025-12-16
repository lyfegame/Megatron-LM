#!/usr/bin/env python3
"""
AIME 2025 Benchmark Runner for DeepSeek V3.2 Official Inference

This script runs the AIME 2025 benchmark (30 problems) using the official
DeepSeek V3.2 inference code and captures logits for baseline validation.

Expected DeepSeek V3.2 performance: 96.0% (Pass@1) = ~29/30 correct
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import re

# Add inference directory to path
INFERENCE_DIR = Path("/home/shuyingluo/deepseek-v3.2-inference")
sys.path.insert(0, str(INFERENCE_DIR))

# CRITICAL: Import tilelang BEFORE torch (required for H200)
import tilelang
from tilelang import tvm
_tvm_target = tvm.target.Target("cuda")

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model


# AIME 2025 Benchmark Data
AIME_2025_PROBLEMS = [
    # AIME I 2025
    {"id": "AIME_I_1", "problem": "Find the sum of all integer bases b > 9 for which 17_b is a divisor of 97_b.", "answer": 70},
    {"id": "AIME_I_2", "problem": "In triangle ABC, points D and E lie on sides AB and AC respectively. The reflection of D over the perpendicular bisector of AB is F and the reflection of E over the perpendicular bisector of AC is N. Calculate the area of heptagon AFNBCEM given specific measurements.", "answer": 588},
    {"id": "AIME_I_3", "problem": "The 9 members of a baseball team went to an ice-cream parlor after their game. Each player had a single scoop cone of chocolate, vanilla, or strawberry ice cream. At least one player chose each flavor, and the number of players who chose chocolate was greater than the number of players who chose vanilla, which was greater than the number of players who chose strawberry. Let N be the number of different assignments of flavors to players that meet these conditions. Find the remainder when N is divided by 1000.", "answer": 16},
    {"id": "AIME_I_4", "problem": "Find the number of ordered pairs (x,y), where both x and y are integers between -100 and 100, inclusive, such that 12x^2-xy-6y^2=0.", "answer": 117},
    {"id": "AIME_I_5", "problem": "Find the number of eight-digit positive integers that use each of the digits 1, 2, 3, 4, 5, 6, 7, and 8 exactly once and are divisible by 22. Then find the difference between this count and 2025.", "answer": 279},
    {"id": "AIME_I_6", "problem": "An isosceles trapezoid has an inscribed circle tangent to each of its four sides. The radius of the circle is 3, and the area of the trapezoid is 72. Let the parallel sides of the trapezoid have lengths r and s, with r != s. Find r^2 + s^2.", "answer": 504},
    {"id": "AIME_I_7", "problem": "The twelve letters A, B, C, D, E, F, G, H, I, J, K, and L are randomly grouped into six pairs of letters. The two letters in each pair are placed next to each other in alphabetical order to form six two-letter words, and then those six words are listed alphabetically. The probability that the last word listed contains G is m/n, where m and n are relatively prime positive integers. Find m+n.", "answer": 821},
    {"id": "AIME_I_8", "problem": "Find the sum of all values of k for which the system of complex number equations has exactly one solution, with the sum expressed as m/n in lowest terms.", "answer": 77},
    {"id": "AIME_I_9", "problem": "The parabola with equation y = x^2 - 4 is rotated 60 degrees counterclockwise around the origin. The unique point in the fourth quadrant where the original parabola and its image intersect has y-coordinate (a - sqrt(b))/c, where a, b, and c are positive integers, and a and c are relatively prime. Find a + b + c.", "answer": 62},
    {"id": "AIME_I_10", "problem": "Find the number of ways to fill a 3x9 grid with numbers 1-9 following Sudoku-like constraints. Answer is in form p^a * q^b * r^c * s^d.", "answer": 81},
    {"id": "AIME_I_11", "problem": "The parabola x = 34y^2 intersects a piecewise linear sawtooth function f(x) at finitely many points. The sum of the y-coordinates of all these intersection points can be expressed in the form (a + b*sqrt(c))/d. Find a + b + c + d.", "answer": 259},
    {"id": "AIME_I_12", "problem": "Three-dimensional geometry problem involving a plane x+y+z=75 with inequality constraints creating convex regions. Find the area of the finite region expressed as a*sqrt(b).", "answer": 510},
    {"id": "AIME_I_13", "problem": "Alex divides a disk into four quadrants with two perpendicular diameters intersecting at the center of the disk. He draws 25 more line segments through the disk, drawing each segment by selecting two points at random on the perimeter of the disk in different quadrants and connecting these two points. Find the expected number of regions into which these 27 line segments divide the disk.", "answer": 204},
    {"id": "AIME_I_14", "problem": "Pentagon ABCDE with specific side lengths and angles (angle B = angle E = 60 degrees). Find the minimum value of f(X)=AX+BX+CX+DX+EX expressed as m+n*sqrt(p).", "answer": 60},
    {"id": "AIME_I_15", "problem": "Let N denote the number of ordered triples of positive integers (a, b, c) such that a, b, c <= 3^6 and a^3 + b^3 + c^3 is a multiple of 3^7. Find the remainder when N is divided by 1000.", "answer": 735},
    # AIME II 2025
    {"id": "AIME_II_1", "problem": "Six collinear points A, B, C, D, E, F with point G not on the line. Given: AC=26, BD=22, CE=31, DF=33, AF=73, CG=40, DG=30. Find the area of triangle BGE.", "answer": 468},
    {"id": "AIME_II_2", "problem": "Find the sum of all positive integers n such that n+2 divides the product 3(n+3)(n^2+9).", "answer": 49},
    {"id": "AIME_II_3", "problem": "Four unit squares in a 2x2 grid with 12 line segments colored red or blue such that each unit square has 2 red sides and 2 blue sides. Find the number of such colorings.", "answer": 82},
    {"id": "AIME_II_4", "problem": "Evaluate the product of logarithmic expressions from k=4 to 63, expressed as m/n in lowest terms. Find m + n.", "answer": 106},
    {"id": "AIME_II_5", "problem": "Triangle ABC with angles 84, 60, 36 degrees. D, E, F are midpoints of sides. Circumcircle of triangle DEF intersects specific segments at points G, H, J. Find DE + 2*HJ + 3*FG.", "answer": 336},
    {"id": "AIME_II_6", "problem": "Two circles with internal tangency. Rectangle EFGH inscribed in smaller circle with specific geometric constraints. Find the area of rectangle EFGH expressed as m/n. Find m+n.", "answer": 293},
    {"id": "AIME_II_7", "problem": "Set A contains divisors of 2025. Subset B of A is randomly selected. Find the probability that B is nonempty with lcm of elements equaling 2025, expressed as m/n. Find m+n.", "answer": 237},
    {"id": "AIME_II_8", "problem": "Greedy algorithm for collecting 1 cent, 10 cent, 25 cent coins totaling N cents. Find the number of values of N between 1 and 1000 inclusive for which the greedy algorithm succeeds.", "answer": 610},
    {"id": "AIME_II_9", "problem": "Find n values of x in (0,2*pi) where f(x)=sin(7*pi*sin(5x))=0. For t of these values, the graph is tangent to the x-axis. Find n+t.", "answer": 149},
    {"id": "AIME_II_10", "problem": "Sixteen chairs in a row; eight people sit so no person sits next to two other people. Find N (number of valid chair subsets) mod 1000.", "answer": 907},
    {"id": "AIME_II_11", "problem": "Regular 24-gon with vertices S. Find the number of ways to draw 12 segments of equal lengths so each vertex is an endpoint of exactly one segment.", "answer": 113},
    {"id": "AIME_II_12", "problem": "11-sided non-convex polygon with specific area and angle constraints, perimeter equals 20. Find A1A2+A1A11 expressed as (m*sqrt(n)-p)/q. Find m+n+p+q.", "answer": 19},
    {"id": "AIME_II_13", "problem": "Sequence defined by x_1=25/11 and x_{k+1}=(1/3)(x_k+1/x_k-1). Find x_2025=m/n. Find remainder when m+n is divided by 1000.", "answer": 248},
    {"id": "AIME_II_14", "problem": "Right triangle ABC with angle A=90 degrees, BC=38. Points K, L inside satisfy AK=AL=BK=CL=KL=14. Find area of quadrilateral BKLC as n*sqrt(3).", "answer": 104},
    {"id": "AIME_II_15", "problem": "Exactly three positive k make f(x)=(x-18)(x-72)(x-98)(x-k)/x achieve minimum at exactly two positive x. Find the sum of these three k values.", "answer": 240},
]


def create_math_prompt(problem: str) -> str:
    """Create the prompt for a math problem following DeepSeek's format."""
    return f"""Solve the following mathematics problem step by step. Show your work clearly.

Problem: {problem}

Think through this carefully, then provide your final answer as a single integer between 0 and 999.
At the end, write your final answer in the format: ANSWER: <number>"""


def extract_answer(response: str) -> Optional[int]:
    """Extract the numerical answer from the model's response."""
    # Look for ANSWER: pattern
    match = re.search(r'ANSWER:\s*(\d+)', response, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try to find the last number in the response
    numbers = re.findall(r'\b(\d{1,3})\b', response)
    if numbers:
        return int(numbers[-1])

    return None


def sample(logits, temperature: float = 1.0):
    """Sample a token from logits."""
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate_with_logits(
    model,
    prompt_tokens: List[int],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    capture_logits: bool = False
) -> Dict:
    """Generate tokens and optionally capture logits."""
    from model import Transformer

    prompt_len = len(prompt_tokens)
    total_len = min(model.max_seq_len, max_new_tokens + prompt_len)
    tokens = torch.full((1, total_len), -1, dtype=torch.long, device="cuda")
    tokens[0, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda")

    all_logits = [] if capture_logits else None
    prev_pos = 0

    for cur_pos in range(prompt_len, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

        if capture_logits:
            all_logits.append(logits.cpu().clone())

        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)

        tokens[0, cur_pos] = next_token
        prev_pos = cur_pos

        if next_token.item() == eos_id:
            break

    completion_tokens = tokens[0, prompt_len:cur_pos+1].tolist()
    if eos_id in completion_tokens:
        completion_tokens = completion_tokens[:completion_tokens.index(eos_id)]

    result = {
        "tokens": completion_tokens,
    }
    if capture_logits:
        result["logits"] = torch.stack(all_logits, dim=0) if all_logits else None

    return result


def run_aime_benchmark(
    ckpt_path: str,
    config_path: str,
    output_dir: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,  # Greedy for reproducibility
    capture_logits: bool = True,
    problems: Optional[List[int]] = None,  # Subset of problem indices to run
):
    """Run the AIME 2025 benchmark."""
    from model import Transformer, ModelArgs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize distributed if needed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(0)

    print(f"[Rank {rank}] Loading model from {ckpt_path}...")

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # Create model
    model_args = ModelArgs(**config)
    model = Transformer(model_args)

    # Load weights
    load_model(model, os.path.join(ckpt_path, "model.safetensors"))
    model.cuda().eval()

    print(f"[Rank {rank}] Model loaded successfully")

    # Select problems to run
    if problems is not None:
        selected_problems = [AIME_2025_PROBLEMS[i] for i in problems]
    else:
        selected_problems = AIME_2025_PROBLEMS

    results = []
    correct = 0
    total = len(selected_problems)

    print(f"\n{'='*60}")
    print(f"AIME 2025 Benchmark - {total} problems")
    print(f"{'='*60}\n")

    for i, prob in enumerate(selected_problems):
        print(f"[{i+1}/{total}] Problem {prob['id']}...")

        # Create prompt
        prompt = create_math_prompt(prob["problem"])
        prompt_tokens = tokenizer.encode(prompt)

        # Generate
        start_time = time.time()
        result = generate_with_logits(
            model,
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_token_id,
            temperature=temperature,
            capture_logits=capture_logits,
        )
        gen_time = time.time() - start_time

        # Decode response
        response = tokenizer.decode(result["tokens"])

        # Extract answer
        predicted = extract_answer(response)
        expected = prob["answer"]
        is_correct = (predicted == expected)

        if is_correct:
            correct += 1

        # Store result
        prob_result = {
            "id": prob["id"],
            "problem": prob["problem"],
            "expected_answer": expected,
            "predicted_answer": predicted,
            "correct": is_correct,
            "response": response,
            "generation_time": gen_time,
            "num_tokens": len(result["tokens"]),
        }
        results.append(prob_result)

        # Save logits if captured
        if capture_logits and result.get("logits") is not None:
            logits_path = output_dir / f"{prob['id']}_logits.pt"
            torch.save(result["logits"], logits_path)
            prob_result["logits_path"] = str(logits_path)

        # Print status
        status = "✓" if is_correct else "✗"
        print(f"  {status} Expected: {expected}, Predicted: {predicted} ({gen_time:.1f}s)")

    # Calculate final score
    accuracy = correct / total * 100

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Expected (from tech report): 96.0%")
    print(f"{'='*60}\n")

    # Save results
    results_path = output_dir / "aime_2025_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "benchmark": "AIME 2025",
            "total_problems": total,
            "correct": correct,
            "accuracy": accuracy,
            "expected_accuracy": 96.0,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "results": results,
        }, f, indent=2)

    print(f"Results saved to {results_path}")

    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="AIME 2025 Benchmark for DeepSeek V3.2")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to DeepSeek V3.2 checkpoint")
    parser.add_argument("--config", type=str,
                        default="/home/shuyingluo/deepseek-v3.2-inference/config_671B_v3.2.json",
                        help="Path to model config")
    parser.add_argument("--output-dir", type=str, default="./aime_2025_outputs",
                        help="Output directory for results and logits")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--no-logits", action="store_true",
                        help="Disable logits capture")
    parser.add_argument("--problems", type=str, default=None,
                        help="Comma-separated list of problem indices to run (0-29)")

    args = parser.parse_args()

    # Parse problem indices if provided
    problems = None
    if args.problems:
        problems = [int(x.strip()) for x in args.problems.split(",")]

    accuracy, results = run_aime_benchmark(
        ckpt_path=args.ckpt_path,
        config_path=args.config,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        capture_logits=not args.no_logits,
        problems=problems,
    )


if __name__ == "__main__":
    main()
