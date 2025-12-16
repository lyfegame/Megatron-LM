#!/usr/bin/env python3
"""
Pure Megatron-Core DeepSeek V3.2 Inference with CPU Offloading

NO HuggingFace dependencies after model loading.
Uses CPU offloading to handle 671B model on limited GPU memory.

Strategy:
- Load model weights on CPU
- Move layers to GPU one at a time during forward pass
- Move back to CPU after each layer computation

Usage:
    python pure_megatron_inference_cpu_offload.py \
        --checkpoint /models-local/DeepSeek-V3.2-megatron \
        --prompt "What is 2 + 2?"
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F


def initialize_distributed():
    """Initialize distributed environment for Megatron."""
    import torch.distributed as dist

    if not dist.is_initialized():
        # Single process mode with gloo (CPU-friendly)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend='gloo', world_size=1, rank=0)

    # Initialize Megatron parallel state
    from megatron.core import parallel_state
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1
        )

    return dist.get_rank(), dist.get_world_size()


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer using tokenizers library (no HF transformers)."""
    from tokenizers import Tokenizer

    tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
    if tokenizer_file.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        return tokenizer, "hf"

    # Try SentencePiece
    import sentencepiece as spm
    for name in ["tokenizer.model", "spiece.model"]:
        sp_file = Path(tokenizer_path) / name
        if sp_file.exists():
            sp = spm.SentencePieceProcessor()
            sp.Load(str(sp_file))
            return sp, "sp"

    raise FileNotFoundError(f"No tokenizer found in {tokenizer_path}")


def encode(tokenizer, tokenizer_type: str, text: str) -> List[int]:
    """Encode text to token IDs."""
    if tokenizer_type == "sp":
        return tokenizer.EncodeAsIds(text)
    else:
        return tokenizer.encode(text).ids


def decode(tokenizer, tokenizer_type: str, ids: List[int]) -> str:
    """Decode token IDs to text."""
    if tokenizer_type == "sp":
        return tokenizer.DecodeIds(ids)
    else:
        return tokenizer.decode(ids)


class CPUOffloadWrapper:
    """
    Wrapper for Megatron model that implements CPU offloading.

    Keeps model weights on CPU and moves layers to GPU one at a time
    during forward pass.
    """

    def __init__(self, model, device: torch.device = torch.device("cuda:0")):
        self.model = model
        self.device = device
        self.cpu_device = torch.device("cpu")

        # Ensure model is on CPU
        self.model.to(self.cpu_device)

        # Get model structure
        self._analyze_model_structure()

    def _analyze_model_structure(self):
        """Analyze Megatron model structure to find layers."""
        # Megatron GPTModel structure:
        # - embedding (word embeddings)
        # - decoder.layers (transformer layers)
        # - output_layer (lm_head)

        print(f"Analyzing model structure: {type(self.model).__name__}")
        print(f"  Model attributes: {[a for a in dir(self.model) if not a.startswith('_')][:20]}")

        # Find embedding
        self.embedding = None
        if hasattr(self.model, 'embedding'):
            self.embedding = self.model.embedding
        elif hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'embedding'):
            self.embedding = self.model.language_model.embedding
        else:
            # Try to find embedding
            for name, module in self.model.named_modules():
                if 'embedding' in name.lower() and hasattr(module, 'weight'):
                    self.embedding = module
                    print(f"  Found embedding via search: {name}")
                    break

        # Find decoder/transformer block and layers
        self.decoder = None
        self.layers = None
        if hasattr(self.model, 'decoder'):
            self.decoder = self.model.decoder
            print(f"  Found decoder: {type(self.decoder).__name__}")
            decoder_attrs = [a for a in dir(self.decoder) if not a.startswith('_')]
            print(f"  Decoder attributes: {decoder_attrs[:30]}")
            # Try multiple attribute names for layers
            for layer_attr in ['layers', '_layers', 'layer', 'blocks']:
                if hasattr(self.decoder, layer_attr):
                    self.layers = getattr(self.decoder, layer_attr)
                    print(f"  Found layers at decoder.{layer_attr}: {len(self.layers)} layers")
                    break
            # If layers is a ModuleList-like object in a different structure
            if self.layers is None:
                # Check if decoder itself is iterable (has layers)
                try:
                    if hasattr(self.decoder, '__len__'):
                        self.layers = self.decoder
                        print(f"  Using decoder directly as layers: {len(self.layers)} layers")
                except:
                    pass
        elif hasattr(self.model, 'language_model'):
            if hasattr(self.model.language_model, 'decoder'):
                self.decoder = self.model.language_model.decoder
                if hasattr(self.decoder, 'layers'):
                    self.layers = self.decoder.layers
            elif hasattr(self.model.language_model, 'encoder'):
                self.decoder = self.model.language_model.encoder
                if hasattr(self.decoder, 'layers'):
                    self.layers = self.decoder.layers

        # Find output layer - check multiple locations
        self.output_layer = None
        for attr in ['output_layer', 'lm_head', 'head']:
            if hasattr(self.model, attr):
                self.output_layer = getattr(self.model, attr)
                print(f"  Found output layer at model.{attr}")
                break
            if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, attr):
                self.output_layer = getattr(self.model.language_model, attr)
                print(f"  Found output layer at model.language_model.{attr}")
                break

        # If still not found, search by name
        if self.output_layer is None:
            for name, module in self.model.named_modules():
                if 'output' in name.lower() and hasattr(module, 'weight'):
                    self.output_layer = module
                    print(f"  Found output layer via search: {name}")
                    break

        print(f"Model structure analyzed:")
        print(f"  - Embedding: {type(self.embedding).__name__ if self.embedding else 'Not found'}")
        print(f"  - Layers: {len(self.layers) if self.layers else 'Not found'}")
        print(f"  - Output layer: {type(self.output_layer).__name__ if self.output_layer else 'Not found'}")
        import sys; sys.stdout.flush()

    def _move_to_device(self, module, device):
        """Move module to device."""
        module.to(device)
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    def _move_to_cpu(self, module):
        """Move module back to CPU and clear GPU cache."""
        module.to(self.cpu_device)
        torch.cuda.empty_cache()
        gc.collect()

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with CPU offloading.

        Moves each layer to GPU, computes, then moves back to CPU.
        """
        batch_size, seq_len = input_ids.shape

        # 1. Embedding layer
        print("  [Offload] Processing embedding...")
        self._move_to_device(self.embedding, self.device)
        input_ids_gpu = input_ids.to(self.device)
        hidden_states = self.embedding(input_ids_gpu)
        self._move_to_cpu(self.embedding)

        # hidden_states stays on GPU for layer processing
        hidden_states = hidden_states.to(self.device)

        # 2. Transformer layers
        if self.layers is not None:
            num_layers = len(self.layers)
            for i, layer in enumerate(self.layers):
                if i % 10 == 0:
                    print(f"  [Offload] Processing layer {i}/{num_layers}...")

                # Move layer to GPU
                self._move_to_device(layer, self.device)

                # Forward through layer
                # Megatron layer signature may vary
                try:
                    layer_output = layer(
                        hidden_states,
                        attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                    )
                    if isinstance(layer_output, tuple):
                        hidden_states = layer_output[0]
                    else:
                        hidden_states = layer_output
                except Exception as e:
                    print(f"  [Offload] Layer {i} forward error: {e}")
                    # Try simpler forward
                    hidden_states = layer(hidden_states)
                    if isinstance(hidden_states, tuple):
                        hidden_states = hidden_states[0]

                # Move layer back to CPU
                self._move_to_cpu(layer)
        elif self.decoder is not None:
            # Use decoder directly if layers not available
            print(f"  [Offload] Processing decoder block (no layer-by-layer offload)...")
            self._move_to_device(self.decoder, self.device)
            hidden_states = self.decoder(hidden_states, attention_mask=attention_mask)
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            self._move_to_cpu(self.decoder)
        else:
            print("  [Offload] WARNING: No layers or decoder found!")

        # 3. Output layer (lm_head)
        if self.output_layer is not None:
            print("  [Offload] Processing output layer...")
            self._move_to_device(self.output_layer, self.device)
            logits = self.output_layer(hidden_states)
            # Handle tuple output (some models return (logits, aux_loss) or similar)
            if isinstance(logits, tuple):
                logits = logits[0]
            self._move_to_cpu(self.output_layer)
        else:
            # If no separate output layer, use embedding weight (tied embeddings)
            print("  [Offload] Using tied embeddings for output...")
            self._move_to_device(self.embedding, self.device)
            # Megatron LanguageModelEmbedding: embedding.word_embeddings.weight
            # Megatron VocabParallelEmbedding: embedding.weight
            if hasattr(self.embedding, 'word_embeddings'):
                emb_weight = self.embedding.word_embeddings.weight
            else:
                emb_weight = self.embedding.weight
            logits = F.linear(hidden_states, emb_weight)
            self._move_to_cpu(self.embedding)

        # Ensure logits is a tensor
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.to(self.cpu_device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def load_megatron_model_cpu(checkpoint_path: str):
    """
    Load Megatron model from checkpoint with CPU initialization.
    """
    from megatron.bridge import AutoBridge
    import json

    # Load config from checkpoint
    run_config_path = Path(checkpoint_path) / "iter_0000000" / "run_config.yaml"
    config_json_path = Path('/models-local/DeepSeek-V3.2-fp8') / "config.json"

    # Try to load HF config without transformers
    if config_json_path.exists():
        with open(config_json_path) as f:
            hf_config_dict = json.load(f)
        print(f"Loaded config from {config_json_path}")
        print(f"  architectures: {hf_config_dict.get('architectures')}")
        print(f"  num_hidden_layers: {hf_config_dict.get('num_hidden_layers')}")
        print(f"  hidden_size: {hf_config_dict.get('hidden_size')}")

    # Still need AutoConfig for bridge (minimal HF dependency)
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        '/models-local/DeepSeek-V3.2-fp8',
        trust_remote_code=True
    )

    print(f"Model config: {config.architectures}")
    print(f"Loading model with CPU initialization...")

    # Create bridge and load model on CPU
    bridge = AutoBridge.from_hf_config(config)

    start_time = time.time()
    model = bridge.load_megatron_model(
        checkpoint_path,
        wrap_with_ddp=False,
        use_cpu_initialization=True,  # KEY: Initialize on CPU
    )
    load_time = time.time() - start_time

    if isinstance(model, list):
        model = model[0]

    model.eval()

    # Ensure model is on CPU
    model.to(torch.device("cpu"))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded in {load_time:.1f}s: {num_params:,} parameters")

    return model, config


@torch.inference_mode()
def generate_with_offload(
    model_wrapper: CPUOffloadWrapper,
    tokenizer,
    tokenizer_type: str,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.95,
) -> str:
    """
    Generate text using CPU offloading.
    """
    device = model_wrapper.device

    # Encode prompt
    input_ids = encode(tokenizer, tokenizer_type, prompt)
    print(f"Prompt tokens: {len(input_ids)}")

    generated_ids = input_ids.copy()
    eos_token_id = 1  # DeepSeek EOS

    for step in range(max_new_tokens):
        print(f"\n--- Generation step {step + 1}/{max_new_tokens} ---")

        seq_len = len(generated_ids)
        tokens = torch.tensor([generated_ids], dtype=torch.long)

        # Forward pass with offloading
        start_time = time.time()
        logits = model_wrapper(tokens)
        forward_time = time.time() - start_time
        print(f"  Forward pass: {forward_time:.2f}s")

        # Get logits for last position
        next_logits = logits[0, -1, :].float()

        # Apply temperature
        if temperature > 0:
            next_logits = next_logits / temperature

        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = float('-inf')

        # Sample next token
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_token)
        print(f"  Generated token: {next_token}")

        # Decode and show progress
        partial_text = decode(tokenizer, tokenizer_type, generated_ids)
        print(f"  Current output: {partial_text[-100:]}")  # Last 100 chars

        # Check for EOS
        if next_token == eos_token_id:
            print("  [EOS reached]")
            break

    return decode(tokenizer, tokenizer_type, generated_ids)


def main():
    parser = argparse.ArgumentParser(description="Pure Megatron DeepSeek V3.2 Inference with CPU Offloading")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Megatron checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                       help="Path to tokenizer (default: /models-local/DeepSeek-V3.2-fp8)")
    parser.add_argument("--prompt", type=str, default="What is 2 + 2?",
                       help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=20,
                       help="Maximum tokens to generate (keep low for CPU offload)")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p (nucleus) sampling")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="GPU device to use for computation")

    args = parser.parse_args()

    print("=" * 70)
    print("PURE MEGATRON DEEPSEEK V3.2 INFERENCE (CPU OFFLOAD)")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Prompt: {args.prompt}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Device: {args.device}")
    print()

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU memory available: {gpu_mem:.1f} GB")

    # Initialize distributed
    rank, world_size = initialize_distributed()
    print(f"Initialized: rank={rank}, world_size={world_size}")

    # Load tokenizer
    tokenizer_path = args.tokenizer or '/models-local/DeepSeek-V3.2-fp8'
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer, tokenizer_type = load_tokenizer(tokenizer_path)
    print(f"Tokenizer type: {tokenizer_type}")

    # Load model on CPU
    print(f"\nLoading model from {args.checkpoint}...")
    print("This will load 1.3TB of weights to CPU RAM...")
    model, config = load_megatron_model_cpu(args.checkpoint)

    # Create offload wrapper
    print("\nCreating CPU offload wrapper...")
    device = torch.device(args.device)
    model_wrapper = CPUOffloadWrapper(model, device=device)

    # Generate
    print("\n" + "=" * 70)
    print("GENERATING...")
    print("=" * 70)

    start_time = time.time()
    output = generate_with_offload(
        model_wrapper=model_wrapper,
        tokenizer=tokenizer,
        tokenizer_type=tokenizer_type,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("OUTPUT:")
    print("=" * 70)
    print(output)
    print()
    print(f"Total generation time: {total_time:.1f}s")
    print(f"Tokens per second: {args.max_tokens / total_time:.2f}")


if __name__ == "__main__":
    main()
