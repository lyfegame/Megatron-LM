#!/usr/bin/env python3
"""
Pure Megatron-Core 16-GPU Inference for DeepSeek V3.2 (671B)

Uses Megatron-Bridge model provider (no HF transformers).
Uses Megatron distributed checkpoint with resharding.

Usage:
    # Node 1 (master):
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
        --master_addr=10.0.0.17 --master_port=29500 \
        pure_megatron_16gpu_v2.py

    # Node 2:
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
        --master_addr=10.0.0.17 --master_port=29500 \
        pure_megatron_16gpu_v2.py
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time
import torch
import torch.distributed as dist
from pathlib import Path


# ============================================================================
# Tokenizer (using tokenizers library, no HF)
# ============================================================================

def load_tokenizer(tokenizer_path: str):
    """Load tokenizer using tokenizers library."""
    from tokenizers import Tokenizer
    tokenizer_file = Path(tokenizer_path) / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_file))


def encode(tokenizer, text: str):
    return tokenizer.encode(text).ids


def decode(tokenizer, ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return tokenizer.decode(ids)


# ============================================================================
# Distributed Setup
# ============================================================================

def setup_distributed():
    """Setup NCCL distributed environment."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, local_rank, world_size


def initialize_megatron(tp_size: int, ep_size: int):
    """Initialize Megatron-Core parallel state."""
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    try:
        parallel_state.destroy_model_parallel()
    except:
        pass

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
    )
    model_parallel_cuda_manual_seed(1234)


# ============================================================================
# Model Building and Loading
# ============================================================================

def create_model(tp_size: int, ep_size: int):
    """Create DeepSeek V3.2 model using Megatron-Bridge provider."""
    from megatron.bridge.models.deepseek.deepseek_v32_bridge import DeepSeekV32ModelProvider
    from megatron.bridge.models.model_provider import get_model
    from megatron.bridge.training.config import DistributedDataParallelConfig

    rank = dist.get_rank()

    # Create model provider with parallelism config
    provider = DeepSeekV32ModelProvider(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        sequence_parallel=tp_size > 1,
        bf16=True,
    )

    # CRITICAL: finalize() must be called to trigger __post_init__ which sets init_method
    provider.finalize()

    if rank == 0:
        print(f"Creating model with DeepSeekV32ModelProvider")
        print(f"  num_layers: {provider.num_layers}")
        print(f"  hidden_size: {provider.hidden_size}")
        print(f"  num_moe_experts: {provider.num_moe_experts}")
        print(f"  TP: {tp_size}, EP: {ep_size}")

    # Create DDP config (we will not use DDP for inference)
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        use_distributed_optimizer=False,
    )

    # Build model with meta device init (weights will be loaded from checkpoint)
    models = get_model(
        model_provider=provider,
        ddp_config=ddp_config,
        wrap_with_ddp=False,
        bf16=True,
        init_model_with_meta_device=True,
    )

    model = models[0]
    model.eval()

    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model created: {num_params:,} params on this rank")

    return model


def load_checkpoint(model, checkpoint_path: str):
    """Load Megatron distributed checkpoint with resharding."""
    from megatron.core.dist_checkpointing import load

    rank = dist.get_rank()

    if rank == 0:
        print(f"\nLoading checkpoint from {checkpoint_path}...")
        start_time = time.time()

    # Get sharded state dict
    sharded_state_dict = model.sharded_state_dict()

    # Load with resharding
    ckpt_dir = Path(checkpoint_path) / "iter_0000000"
    load(
        sharded_state_dict=sharded_state_dict,
        checkpoint_dir=str(ckpt_dir),
    )

    # Load into model
    model.load_state_dict(sharded_state_dict, strict=False)

    if rank == 0:
        elapsed = time.time() - start_time
        print(f"Checkpoint loaded in {elapsed:.1f}s")

    return model


# ============================================================================
# Inference
# ============================================================================

@torch.inference_mode()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """Generate text."""
    from megatron.core import parallel_state

    rank = dist.get_rank()
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    device = torch.cuda.current_device()

    # Encode on rank 0 and broadcast
    if rank == 0:
        input_ids = torch.tensor([encode(tokenizer, prompt)], dtype=torch.long, device=device)
        seq_len = torch.tensor([input_ids.shape[1]], device=device)
    else:
        seq_len = torch.tensor([0], device=device)
        input_ids = None

    dist.broadcast(seq_len, src=0)

    if rank != 0:
        input_ids = torch.zeros(1, seq_len.item(), dtype=torch.long, device=device)
    dist.broadcast(input_ids, src=0)

    if rank == 0:
        print(f"Input tokens: {input_ids.shape[1]}")

    generated = input_ids.clone()

    for step in range(max_new_tokens):
        if rank == 0:
            print(f"  Step {step + 1}/{max_new_tokens}...", end=" ", flush=True)

        # Forward
        cur_seq_len = generated.shape[1]
        position_ids = torch.arange(cur_seq_len, device=device).unsqueeze(0)
        attention_mask = torch.ones(1, cur_seq_len, device=device)

        output = model(
            input_ids=generated,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        # Get next token
        if tp_rank == 0:
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            next_token = torch.zeros(1, 1, dtype=torch.long, device=device)

        tp_src = parallel_state.get_tensor_model_parallel_src_rank()
        dist.broadcast(next_token, src=tp_src)

        generated = torch.cat([generated, next_token], dim=1)

        if rank == 0:
            token_str = decode(tokenizer, next_token[0])
            print(f'"{token_str}"')

        if next_token.item() == 1:  # EOS
            break

    if rank == 0:
        return decode(tokenizer, generated[0])
    return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/mnt/models-disk/DeepSeek-V3.2-megatron")
    parser.add_argument("--tokenizer", default="/mnt/models-disk/DeepSeek-V3.2-fp8")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--prompt", default="What is 2 + 2?")
    parser.add_argument("--max-tokens", type=int, default=50)
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()

    if rank == 0:
        print("=" * 70)
        print("PURE MEGATRON-CORE INFERENCE - DeepSeek V3.2")
        print("=" * 70)
        print(f"World size: {world_size}, TP: {args.tp}, EP: {args.ep}")

    expected = args.tp * args.ep
    if world_size != expected:
        if rank == 0:
            print(f"ERROR: World size {world_size} != TP*EP = {expected}")
        dist.destroy_process_group()
        return

    initialize_megatron(args.tp, args.ep)

    # Load tokenizer (no HF)
    tokenizer = load_tokenizer(args.tokenizer)

    # Create model
    model = create_model(args.tp, args.ep)

    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint)

    dist.barrier()

    # Generate
    if rank == 0:
        print(f"\nPrompt: {args.prompt}")

    start = time.time()
    output = generate(model, tokenizer, args.prompt, args.max_tokens)
    elapsed = time.time() - start

    if rank == 0 and output:
        print(f"\nOutput: {output}")
        print(f"Time: {elapsed:.2f}s")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
