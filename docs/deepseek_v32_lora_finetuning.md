# DeepSeek V3.2 LoRA Fine-Tuning with Megatron

This document covers LoRA fine-tuning for DeepSeek V3.2 using Megatron-Bridge, including integration with VERL for reinforcement learning.

## Table of Contents

1. [Overview](#overview)
2. [How LoRA Works with Megatron](#how-lora-works-with-megatron)
3. [Layer Type Detection](#layer-type-detection)
4. [V3.2 Specific Considerations](#v32-specific-considerations)
5. [VERL Integration](#verl-integration)
6. [Adapter-Only Checkpointing](#adapter-only-checkpointing)
7. [Configuration Examples](#configuration-examples)

---

## Overview

### Key Points

- **No HuggingFace transformers required** for the base model
- **No modifications to V3.2 Megatron code** - LoRA is injected at runtime
- **Adapter-only checkpoints** - only ~100MB saved, not the full 1TB model
- **VERL support** - Megatron backend + LoRA via PR #4063

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Megatron-Bridge LoRA Architecture                                  │
│                                                                      │
│  Base Model (V3.2)           LoRA Adapters                          │
│  ─────────────────           ─────────────                          │
│  - Frozen weights            - Trainable lora_A, lora_B             │
│  - ~672B parameters          - ~100M parameters (rank=64)           │
│  - Never modified            - Saved separately                     │
│                                                                      │
│  Forward: output = base_linear(x) + lora_B(lora_A(x)) * scale       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## How LoRA Works with Megatron

### Core Mechanism

LoRA wraps target linear layers with an adapter wrapper:

```python
class LoRALinear(AdapterWrapper):
    def forward(self, x, *args, **kwargs):
        # 1. Run base linear (frozen)
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)

        # 2. Run LoRA adapter (trainable)
        adapter_output = self.adapter(layernorm_output.contiguous())

        # 3. Sum outputs
        return linear_output + adapter_output, bias
```

### Source Files

| File | Description |
|------|-------------|
| `megatron/bridge/peft/lora.py` | Main `LoRA` class definition |
| `megatron/bridge/peft/lora_layers.py` | `LoRALinear`, `TELinearAdapter` wrappers |
| `megatron/bridge/peft/utils.py` | `ParallelLinearAdapter` for Megatron parallel layers |
| `megatron/bridge/peft/base.py` | Base `PEFT` class, adapter filtering |
| `megatron/bridge/training/checkpointing.py` | Adapter-only checkpoint logic |

---

## Layer Type Detection

Megatron-Bridge automatically detects layer types and applies the appropriate adapter:

| Base Layer Type | Adapter Used | Parallelism |
|-----------------|--------------|-------------|
| `nn.Linear` | `LinearAdapter` | None (replicated) |
| `te.Linear` | `TELinearAdapter` | None (replicated) |
| `ColumnParallelLinear` | `ParallelLinearAdapter` | Column-parallel (split output dim) |
| `RowParallelLinear` | `ParallelLinearAdapter` | Row-parallel (split input dim) |
| `TEColumnParallelLinear` | `ParallelLinearAdapter` | Column-parallel + TE optimizations |
| `TERowParallelLinear` | `ParallelLinearAdapter` | Row-parallel + TE optimizations |

### Detection Logic (from `lora.py:112-143`)

```python
def transform(self, module, name, prefix):
    if isinstance(module, nn.Linear) or module.__class__ == te.Linear:
        # Simple adapter for non-parallel layers
        return LinearAdapter(module, dim=self.dim, alpha=self.alpha, ...)

    # For Megatron parallel layers, detect parallelism type
    input_is_parallel, in_features, out_features, ... = get_adapter_attributes_from_linear(module)

    # Create parallel-aware adapter
    return LoRALinear(module, ParallelLinearAdapter(
        in_features, out_features, self.dim,
        input_is_parallel=input_is_parallel,
        ...
    ))
```

---

## V3.2 Specific Considerations

### Lightning Indexer Layers

DeepSeek V3.2 adds the Lightning Indexer with mixed layer types:

| Layer | Type | LoRA Adapter |
|-------|------|--------------|
| `lightning_indexer.linear_wq_b` | `ColumnParallelLinear` | `ParallelLinearAdapter` |
| `lightning_indexer.linear_wk` | `nn.Linear` | `LinearAdapter` |
| `lightning_indexer.k_layernorm` | `nn.LayerNorm` | Not applicable (normalization) |
| `lightning_indexer.linear_weights_proj` | `ColumnParallelLinear` | `ParallelLinearAdapter` |

### Recommended Target Modules for V3.2

```python
from megatron.bridge.peft.lora import LoRA

lora_config = LoRA(
    target_modules=[
        # Standard attention/MLP (recommended)
        "linear_qkv",
        "linear_proj",
        "linear_fc1",
        "linear_fc2",

        # Optional: Lightning Indexer layers
        # "lightning_indexer.linear_wq_b",
        # "lightning_indexer.linear_wk",
        # "lightning_indexer.linear_weights_proj",
    ],
    dim=64,      # LoRA rank
    alpha=128,   # Scaling factor
    dropout=0.0,
)
```

---

## VERL Integration

### Status (as of December 2025)

| Feature | Status |
|---------|--------|
| VERL + Megatron backend | ✅ Supported |
| LoRA with Megatron | ✅ Added via Megatron-Bridge (PR #4063) |
| Adapter-only sync to vLLM | ✅ Works (`layered_summon`) |
| Adapter-only sync to SGLang | 🔜 Coming soon |

### Architecture for RL Training

```
┌─────────────────────────────────────────────────────────────────────┐
│  VERL RL Loop with LoRA                                             │
│                                                                      │
│  ┌───────────────────────────┐   ┌────────────────────────────────┐ │
│  │  Megatron Training        │   │  vLLM/SGLang Rollout           │ │
│  │                           │   │                                │ │
│  │  V3.2 Base (frozen)       │   │  V3.2 Base (loaded once)       │ │
│  │  + LoRA Adapter           │──▶│  + LoRA Adapter                │ │
│  │  (trainable, ~100MB)      │   │  (layered_summon sync)         │ │
│  │                           │   │                                │ │
│  └───────────────────────────┘   └────────────────────────────────┘ │
│                                                                      │
│  Key: Only adapter weights are synced (~100MB), not full model      │
└─────────────────────────────────────────────────────────────────────┘
```

### VERL Configuration Example

```yaml
actor_rollout_ref:
  rollout:
    layered_summon: true  # Sync adapters layer-by-layer (memory efficient)
    load_format: "safetensors"
  model:
    lora_rank: 64
    lora_alpha: 128
    lora_target_modules:
      - "linear_qkv"
      - "linear_proj"
      - "linear_fc1"
      - "linear_fc2"
```

---

## Adapter-Only Checkpointing

### How It Works

Megatron-Bridge filters checkpoints to save only adapter parameters:

```python
# From checkpointing.py:554-556
if cfg.peft is not None:
    state_dict = apply_peft_adapter_filter_to_state_dict(state_dict, cfg.peft)
```

The filter identifies adapter params via (from `base.py:200`):

```python
def adapter_key_filter(self, key):
    return key in self.params_to_save or ".adapter." in key or key.endswith(".adapters")
```

### Checkpoint Structure

```
/checkpoints/step_1000/
├── adapter_model.safetensors   # ~100MB (LoRA weights only)
├── adapter_config.json         # LoRA hyperparameters
└── optimizer_states/           # Optimizer state for adapters
```

### Loading Adapters

```python
# Load base model (once)
model = load_megatron_model(base_checkpoint_path)

# Apply LoRA structure
lora_config = LoRA(dim=64, alpha=128, target_modules=[...])
model = lora_config(model)

# Load adapter weights
adapter_state = torch.load("adapter_model.safetensors")
model.load_state_dict(adapter_state, strict=False)
```

---

## Configuration Examples

### Basic LoRA Training

```python
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import ConfigContainer, CheckpointConfig

config = ConfigContainer(
    # ... other configs ...
    peft=LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=64,
        alpha=128,
        dropout=0.1,
    ),
    checkpoint=CheckpointConfig(
        pretrained_checkpoint="/path/to/v32/base/checkpoint",
        save="/path/to/lora/checkpoints",
    ),
)
```

### Wildcard Targeting (Specific Layers)

```python
lora_config = LoRA(
    target_modules=[
        "*.layers.0.*.linear_qkv",   # First layer only
        "*.layers.1.*.linear_qkv",   # Second layer only
        "*.layers.*.mlp.linear_fc1", # All MLP fc1 layers
    ],
    dim=32,
    alpha=64,
)
```

### LoRA Recommendations

| Model Size | Recommended Rank | Notes |
|------------|------------------|-------|
| < 10B | 16-32 | Lower rank sufficient |
| 10B-70B | 32-64 | Balance quality/efficiency |
| 70B+ | 64-128 | Higher rank for large models |
| V3.2 (672B) | 128+ | MoE model, may need higher rank |

**Note:** For V3.2, a `lora_rank=128` with standard attention/MLP targets typically achieves near full-parameter training quality.

---

## References

- [VERL LoRA Documentation](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)
- [VERL Megatron-Bridge PR #4063](https://github.com/volcengine/verl/pull/4063)
- [Megatron-Bridge PEFT Guide](../Megatron-Bridge-fork/docs/training/peft.md)
- [SGLang LoRA Serving](https://docs.sglang.io/advanced_features/lora.html)
