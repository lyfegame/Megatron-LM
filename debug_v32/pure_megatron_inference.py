#!/usr/bin/env python3
"""
Pure Megatron-Core DeepSeek V3.2 Inference

NO HuggingFace dependencies after model loading.
Reference: Official DeepSeek V3.2 inference demo (model.py)

Key differences from official:
- Uses Megatron-Core GPTModel instead of custom Transformer
- Loads from converted Megatron checkpoint
- Uses SentencePiece tokenizer directly

Usage:
    # Single GPU
    python pure_megatron_inference.py --checkpoint /models-local/DeepSeek-V3.2-megatron --prompt "Hello"

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=8 pure_megatron_inference.py --checkpoint /models-local/DeepSeek-V3.2-megatron --prompt "Hello"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F


def initialize_distributed():
    """Initialize distributed environment for Megatron."""
    import torch.distributed as dist

    if not dist.is_initialized():
        # Check if running under torchrun
        if 'RANK' in os.environ:
            dist.init_process_group(backend='nccl')
        else:
            # Single process mode
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'
            dist.init_process_group(backend='gloo', world_size=1, rank=0)

    # Initialize Megatron parallel state
    from megatron.core import parallel_state
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=dist.get_world_size()
        )

    return dist.get_rank(), dist.get_world_size()


def load_tokenizer(tokenizer_path: str):
    """Load SentencePiece tokenizer directly (no HF dependency)."""
    import sentencepiece as spm

    tokenizer_file = Path(tokenizer_path) / "tokenizer.model"
    if not tokenizer_file.exists():
        # Try alternative locations
        for alt in ["tokenizer.model", "spiece.model"]:
            alt_path = Path(tokenizer_path) / alt
            if alt_path.exists():
                tokenizer_file = alt_path
                break

    if not tokenizer_file.exists():
        # Fall back to HF tokenizer json
        print("SentencePiece model not found, using tokenizers library")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))
        return tokenizer, "hf"

    sp = spm.SentencePieceProcessor()
    sp.Load(str(tokenizer_file))
    return sp, "sp"


def encode(tokenizer, tokenizer_type: str, text: str) -> List[int]:
    """Encode text to token IDs."""
    if tokenizer_type == "sp":
        return tokenizer.EncodeAsIds(text)
    else:  # hf tokenizers
        return tokenizer.encode(text).ids


def decode(tokenizer, tokenizer_type: str, ids: List[int]) -> str:
    """Decode token IDs to text."""
    if tokenizer_type == "sp":
        return tokenizer.DecodeIds(ids)
    else:
        return tokenizer.decode(ids)


def create_causal_mask(seq_len: int, device: torch.device) -> Optional[torch.Tensor]:
    """Create causal attention mask (upper triangular with -inf)."""
    if seq_len <= 1:
        return None
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


def create_position_ids(seq_len: int, start_pos: int, device: torch.device) -> torch.Tensor:
    """Create position IDs for RoPE."""
    return torch.arange(start_pos, start_pos + seq_len, device=device)


def load_megatron_model(checkpoint_path: str):
    """
    Load Megatron model from checkpoint.

    This loads the model architecture and weights without HF dependencies.
    """
    from megatron.bridge import AutoBridge
    from transformers import AutoConfig

    # Load config to get model architecture
    # Note: This is the only HF dependency - just for config parsing
    # Could be replaced with direct JSON parsing if needed
    run_config_path = Path(checkpoint_path) / "iter_0000000" / "run_config.yaml"

    if run_config_path.exists():
        import yaml
        with open(run_config_path) as f:
            run_config = yaml.safe_load(f)

        # Get HF model ID from checkpoint
        hf_model_id = AutoBridge.get_hf_model_id_from_checkpoint(checkpoint_path)
        if hf_model_id:
            config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
        else:
            # Fall back to original HF checkpoint
            config = AutoConfig.from_pretrained(
                '/models-local/DeepSeek-V3.2-fp8',
                trust_remote_code=True
            )
    else:
        config = AutoConfig.from_pretrained(
            '/models-local/DeepSeek-V3.2-fp8',
            trust_remote_code=True
        )

    print(f"Model config: {config.architectures}")

    # Create bridge and load model
    bridge = AutoBridge.from_hf_config(config)
    model = bridge.load_megatron_model(
        checkpoint_path,
        wrap_with_ddp=False,
        use_cpu_initialization=False,
    )

    if isinstance(model, list):
        model = model[0]

    model.eval()
    return model, config


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    tokenizer_type: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: torch.device = torch.device("cuda"),
) -> str:
    """
    Generate text using pure Megatron model.

    Mimics official DeepSeek V3.2 inference pattern:
    1. Encode prompt
    2. Create causal mask and position IDs
    3. Forward through model
    4. Sample next token
    5. Repeat until max_new_tokens or EOS
    """
    # Encode prompt
    input_ids = encode(tokenizer, tokenizer_type, prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    generated_ids = input_ids.tolist()[0]

    # EOS token ID (typically 1 for DeepSeek)
    eos_token_id = 1

    for _ in range(max_new_tokens):
        seq_len = len(generated_ids)
        tokens = torch.tensor([generated_ids], dtype=torch.long, device=device)

        # Create attention mask (causal)
        # For Megatron, we typically pass this as part of the forward
        attention_mask = create_causal_mask(seq_len, device)

        # Create position IDs
        position_ids = create_position_ids(seq_len, 0, device).unsqueeze(0)

        # Forward pass through Megatron model
        # Note: Megatron GPTModel expects different signature than official demo
        # This may need adjustment based on exact Megatron version
        try:
            output = model(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            # Get logits from output
            if hasattr(output, 'logits'):
                logits = output.logits[:, -1, :]
            elif isinstance(output, tuple):
                logits = output[0][:, -1, :]
            else:
                logits = output[:, -1, :]
        except Exception as e:
            print(f"Forward error: {e}")
            # Try simpler forward
            output = model(tokens)
            if hasattr(output, 'logits'):
                logits = output.logits[:, -1, :]
            else:
                logits = output[:, -1, :]

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_token)

        # Check for EOS
        if next_token == eos_token_id:
            break

    # Decode output
    output_text = decode(tokenizer, tokenizer_type, generated_ids)
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Pure Megatron DeepSeek V3.2 Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Megatron checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Path to tokenizer (default: use HF checkpoint)")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95,
                       help="Top-p (nucleus) sampling")

    args = parser.parse_args()

    print("=" * 70)
    print("PURE MEGATRON DEEPSEEK V3.2 INFERENCE")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print()

    # Initialize distributed
    rank, world_size = initialize_distributed()
    print(f"Initialized: rank={rank}, world_size={world_size}")

    # Load tokenizer
    tokenizer_path = args.tokenizer or '/models-local/DeepSeek-V3.2-fp8'
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer, tokenizer_type = load_tokenizer(tokenizer_path)
    print(f"Tokenizer type: {tokenizer_type}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_megatron_model(args.checkpoint)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Generate
    print("\nGenerating...")
    output = generate(
        model=model,
        tokenizer=tokenizer,
        tokenizer_type=tokenizer_type,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n" + "=" * 70)
    print("OUTPUT:")
    print("=" * 70)
    print(output)


if __name__ == "__main__":
    main()
