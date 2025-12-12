# DeepSeek V3 Checkpoint Conversion

This document describes checkpoint conversion procedures for DeepSeek V3/V3.2 models.

## HuggingFace to Megatron Conversion

### Command

```bash
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader deepseek_hf \
    --saver core \
    --load-dir /path/to/DeepSeek-V3-bf16 \
    --save-dir /path/to/megatron-checkpoint \
    --tokenizer-model /path/to/DeepSeek-V3-bf16 \
    --target-tensor-parallel-size 8 \
    --target-pipeline-parallel-size 1 \
    --bf16
```

### Weight Mapping

The converter maps HuggingFace weight names to Megatron weight names.

#### MLA Attention Weights

| HuggingFace | Megatron |
|-------------|----------|
| `self_attn.q_a_proj` | `linear_q_down_proj` |
| `self_attn.q_a_layernorm` | `q_layernorm` |
| `self_attn.q_b_proj` | `linear_q_up_proj` |
| `self_attn.kv_a_proj_with_mqa` | `linear_kv_down_proj` |
| `self_attn.kv_a_layernorm` | `kv_layernorm` |
| `self_attn.kv_b_proj` | `linear_kv_up_proj` |
| `self_attn.o_proj` | `linear_proj` |

#### MoE Weights

| HuggingFace | Megatron |
|-------------|----------|
| `mlp.gate.weight` | `mlp.router.weight` |
| `mlp.experts[i].gate_proj` + `up_proj` | `linear_fc1` (fused SwiGLU) |
| `mlp.experts[i].down_proj` | `linear_fc2` |
| `mlp.shared_experts.*` | `shared_experts.*` |

### Memory Requirements

DeepSeek V3 (671B parameters) checkpoint conversion memory requirements:

| Configuration | CPU RAM Required |
|---------------|------------------|
| BF16 weights (loading only) | ~1.3 TB |
| BF16 weights (loading + building) | ~2.6 TB |
| FP32 weights | ~2.7 TB |

**OOM Issue Observed**: On a3-megagpu-8g instance (1.8 TB RAM), conversion was OOM-killed during Megatron model building phase. The loader loads the entire HuggingFace model (~1.3 TB) then creates the entire Megatron model (~1.3 TB), requiring ~2.6 TB total. dmesg showed:
```
Out of memory: Killed process 130339 (python) total-vm:2489078352kB, anon-rss:1895955648kB
```

**Recommendation**: Use instance with >2.6 TB RAM or modify loader to use layer-by-layer conversion.

### Parallel State Initialization

The checkpoint converter requires fake process groups for single-process conversion. The following groups must be initialized in both `loader_deepseek_hf.py` and `saver_base.py`:

```python
from utils import _ConverterFakeProcessGroup

fake_tp_group = _ConverterFakeProcessGroup(size=tensor_parallel_size)
fake_ep_group = _ConverterFakeProcessGroup(size=expert_parallel_size)
fake_dp_group = _ConverterFakeProcessGroup(size=1)

# Core parallel groups
mpu._TENSOR_MODEL_PARALLEL_GROUP = fake_tp_group
mpu._PIPELINE_MODEL_PARALLEL_GROUP = fake_dp_group
mpu._MODEL_PARALLEL_GROUP = fake_dp_group
mpu._DATA_PARALLEL_GROUP = fake_dp_group
mpu._DATA_PARALLEL_GROUP_GLOO = fake_dp_group
mpu._TENSOR_AND_DATA_PARALLEL_GROUP = fake_dp_group

# Expert parallel groups
mpu._EXPERT_MODEL_PARALLEL_GROUP = fake_ep_group
mpu._EXPERT_TENSOR_PARALLEL_GROUP = fake_dp_group
mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = fake_dp_group
mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP = fake_dp_group
mpu._EXPERT_DATA_PARALLEL_GROUP = fake_dp_group
mpu._EXPERT_DATA_PARALLEL_GROUP_GLOO = fake_dp_group
mpu._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = fake_dp_group
mpu._INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_GLOO = fake_dp_group
mpu._INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = fake_dp_group

# Context parallel groups
mpu._CONTEXT_PARALLEL_GROUP = fake_dp_group
mpu._DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group
mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO = fake_dp_group
mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group
mpu._INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_GLOO = fake_dp_group
mpu._TENSOR_AND_CONTEXT_PARALLEL_GROUP = fake_dp_group
mpu._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = fake_dp_group

# Embedding groups
mpu._EMBEDDING_GROUP = fake_dp_group
mpu._POSITION_EMBEDDING_GROUP = fake_dp_group

# Distributed optimizer group
mpu._INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = fake_dp_group
```

These groups are required by `ProcessGroupCollection.use_mpu_process_groups()` which is called during model initialization.

## FP8 to BF16 Conversion

DeepSeek V3/V3.2 FP8 checkpoints use block-wise quantization with 128x128 block size.

### Quantization Format

- **Data type**: FP8 E4M3 (`torch.float8_e4m3fn`)
- **Block size**: 128x128
- **Scale storage**: Each weight tensor has a corresponding `{weight_name}_scale_inv` tensor
- **Scale shape**: `(M // 128, N // 128)` for weight shape `(M, N)`

### Dequantization Formula

```
bf16_weight = fp8_weight * scale_inv
```

Where `scale_inv` is expanded to match weight dimensions via block repetition.

### CPU-Based Conversion Script

Location: `scripts/fp8_to_bf16_convert.py`

```bash
# Run with resume capability (skips already converted shards)
python scripts/fp8_to_bf16_convert.py \
    --fp8-gcs-path gs://fundamental_ml_shared_storage/models/DeepSeek-V3.2-fp8 \
    --bf16-gcs-path gs://fundamental_ml_shared_storage/models/DeepSeek-V3.2-bf16 \
    --temp-dir /tmp/fp8_convert \
    --skip-existing

# Only generate metadata files (config.json, model index)
python scripts/fp8_to_bf16_convert.py \
    --fp8-gcs-path gs://bucket/DeepSeek-V3.2-fp8 \
    --bf16-gcs-path gs://bucket/DeepSeek-V3.2-bf16 \
    --only-metadata
```

**Features:**
- **Resume capability**: Checks GCS for existing shards and skips them
- **Memory efficient**: Processes one shard at a time (~12GB peak for 4GB FP8 → 8GB BF16)
- **Cross-file scale_inv handling**: Downloads scale tensors from other shards when needed
- **Size validation**: Warns if output size ratio is outside expected range (1.3x-2.5x)
- **Automatic cleanup**: Deletes local files after each shard upload

**Script workflow per shard:**
1. Check if BF16 version exists in GCS → skip if yes
2. Download FP8 shard from GCS (~4GB, ~30s)
3. Load and identify FP8 tensors (1-byte element size)
4. Download cross-file scale_inv tensors if needed
5. Dequantize: `bf16 = fp8.float32() * scale_inv.repeat_interleave(128)`
6. Save BF16 shard and upload to GCS (~8GB, ~60s)
7. Delete local files

**After all shards:**
- Copies config.json (removes `quantization_config`, sets `torch_dtype: bfloat16`)
- Creates model.safetensors.index.json (without `_scale_inv` entries)

### Conversion Progress (Dec 12, 2025)

**V3.2 FP8→BF16 conversion on `deepseek-convert-v5` (n2-highmem-16, 125GB RAM):**

| Timestamp (UTC) | Shards Complete | Status |
|-----------------|-----------------|--------|
| 16:52 | 122/163 | Started conversion |
| 16:59 | 124/163 | Shards 51-56 uploaded |
| 17:02 | 125/163 | Processing shard 59 |
| 17:11 | 128/163 | Processing shard 67 |
| 17:59 | 145/163 | Uploading shard 114 |
| 18:05 | 148/163 | 91% complete |
| ~18:20 (est) | 163/163 | Expected completion |

**Rate:** ~2.5 minutes per shard (includes download, convert, upload)

**Memory usage:** ~15GB peak out of 125GB available (very efficient)

### Conversion Time

| Model | Files | Time per File | Total Time |
|-------|-------|---------------|------------|
| DeepSeek V3.2 (671B) | 163 | ~1-2 minutes | ~3-5 hours |

## CPU-Only Conversion (No GPU/CUDA)

When running the HuggingFace to Megatron conversion on a CPU-only system (no GPU, CUDA, or Transformer Engine), several issues must be addressed.

### Required Fixes (Already Applied)

The following fixes have been committed to the `loader_deepseek_hf.py`:

#### 1. Gradient Accumulation Fusion (Commit `338d421fd`)

**Problem:** APEX gradient accumulation fusion requires CUDA, causing:
```
RuntimeError: Attempting to run ApexGradScaler when CUDA is unavailable
```

**Solution:** Add `--no-gradient-accumulation-fusion` to the Megatron args in the loader.

#### 2. Transformer Engine Detection (Commit `22c3f5991`)

**Problem:** Incorrect TE availability detection when TE is not installed:
```
ImportError: megatron.core.extensions.transformer_engine
```

**Solution:** Import `HAVE_TE` from `megatron.core.extensions` to correctly detect TE availability.

#### 3. MLA CPU-Only Mode (Commit `84121374b`)

**Problem:** MLA module tries to use TE-specific features when TE is not installed.

**Solution:** Fix MLA module to properly handle CPU-only mode with local attention implementation.

#### 4. YarnRotaryEmbedding (Commit `87c01d80e`)

**Problem:** YarnRotaryEmbedding forward pass fails on CPU-only systems.

**Solution:** Fix the forward pass to work without CUDA kernels.

#### 5. IdentityOp LayerNorm Weights (Commit `5d6e56e78`)

**Problem:** When QK-norm is disabled, `q_layernorm` and `kv_layernorm` become `IdentityOp` objects that don't have a `weight` attribute:
```
AttributeError: 'IdentityOp' object has no attribute 'weight'
```

**Solution:** Add check for `weight` attribute before attempting to copy:
```python
if hasattr(attn, 'q_layernorm') and hasattr(attn.q_layernorm, 'weight'):
    attn.q_layernorm.weight.data.copy_(hf_attn.q_a_layernorm.weight)
```

#### 6. MoE Layer Frequency for Dense Layers (Commit `06e453cc3`)

**Problem:** DeepSeek V3 uses `first_k_dense_replace=3` to make the first 3 layers dense (not MoE), but Megatron builds all layers as MoE when `num_experts > 1`:
```
RuntimeError: Layer architecture mismatch: HF has dense MLP but Megatron has MoE
```

**Solution:** Build explicit `moe_layer_freq` list based on `first_k_dense_replace`:
```python
if args.first_k_dense_replace > 0:
    # First k layers are dense (0), rest are MoE (1)
    moe_layer_pattern = [0] * args.first_k_dense_replace + [1] * (args.num_layers - args.first_k_dense_replace)
    args.moe_layer_freq = moe_layer_pattern
```

#### 7. MoE FFN Hidden Size (Commit `04cbcba10`)

**Problem:** Megatron uses `moe_ffn_hidden_size` for MoE expert FFN dimensions, but this wasn't being set:
```
RuntimeError: The size of tensor a (36864) must match the size of tensor b (4096)
```

**Solution:** Set `moe_ffn_hidden_size` from HF config's `moe_intermediate_size`:
```python
args.moe_ffn_hidden_size = args.moe_intermediate_size
```

#### 8. Shared Expert Intermediate Size (Commit `0442e052f`)

**Problem:** Shared experts use the same intermediate size as MoE experts (2048), not the dense FFN size (18432):
```
RuntimeError: The size of tensor a (36864) must match the size of tensor b (4096)
```

**Solution:** Set `moe_shared_expert_intermediate_size = num_shared_experts * moe_intermediate_size`:
```python
if args.num_shared_experts > 0:
    args.moe_shared_expert_intermediate_size = args.num_shared_experts * args.moe_intermediate_size
```

### DeepSeek V3 Architecture Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_hidden_layers` | 61 | Total transformer layers |
| `hidden_size` | 7168 | Model hidden dimension |
| `intermediate_size` | 18432 | Dense FFN hidden size |
| `moe_intermediate_size` | 2048 | MoE expert FFN hidden size |
| `n_routed_experts` | 256 | Number of routed experts per MoE layer |
| `n_shared_experts` | 1 | Number of shared experts per MoE layer |
| `num_experts_per_tok` | 8 | Top-k experts per token |
| `first_k_dense_replace` | 3 | First 3 layers use dense MLP (not MoE) |
| `q_lora_rank` | 1536 | MLA Q projection LoRA rank |
| `kv_lora_rank` | 512 | MLA KV projection LoRA rank |

#### 9. Pipeline Parallel Size (Runtime Issue)

**Problem:** DeepSeek V3 has 61 layers, which is a **prime number**. Pipeline parallel size must divide the number of layers:
```
AssertionError: Number of layers should be divisible by the pipeline-model-parallel size
```

**Solution:** Use `--target-pipeline-parallel-size 1` since 61 only divides by 1 and 61.

#### 10. GPU Architecture Version Check (Commit `d941ca9e1`)

**Problem:** The saver's `validate_args` calls `get_device_arch_version()` which requires CUDA:
```
RuntimeError: Found no NVIDIA driver on your system
```

**Solution:** Modify `get_device_arch_version()` in `megatron/training/utils.py` to return 0 when CUDA is unavailable:
```python
def get_device_arch_version():
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.get_device_properties(torch.device("cuda:0")).major
    except RuntimeError:
        return 0
```

#### 11. CUDA_DEVICE_MAX_CONNECTIONS Check (Commit `25c042362`)

**Problem:** Tensor parallel validation requires `CUDA_DEVICE_MAX_CONNECTIONS=1`:
```
AssertionError: Using tensor model parallelism or context parallelism require setting CUDA_DEVICE_MAX_CONNECTIONS to 1
```

**Solution:** Add `torch.cuda.is_available()` check in `megatron/training/arguments.py` to skip this validation for CPU-only mode.

### Conversion Progress (Dec 12, 2025)

**V3 HF→Megatron conversion on `deepseek-v3-converter` (m2-ultramem-208, 5.7TB RAM):**

| Time (UTC) | Status |
|------------|--------|
| 07:52 | Started, hit gradient_accumulation_fusion error |
| 16:36 | Fixed fusion, hit IdentityOp error |
| 16:48 | Fixed IdentityOp, hit dense/MoE mismatch |
| 16:50 | Fixed moe_layer_freq, hit shared expert size error |
| 16:54 | Fixed shared expert size, hit args transfer error |
| 17:02 | Fixed args, conversion running successfully |
| 17:07 | All 61 layers copied, hit pp=4 error (61 is prime) |
| 17:08 | Restarted with pp=1 |

**Performance:** ~4.2 seconds per layer, ~4.3 minutes for all 61 layers (after HF model loading)

## Files

| File | Purpose |
|------|---------|
| `tools/checkpoint/convert.py` | Main conversion orchestrator |
| `tools/checkpoint/loader_deepseek_hf.py` | HuggingFace to Megatron loader |
| `tools/checkpoint/saver_core.py` | Megatron Core checkpoint saver |
| `tools/checkpoint/saver_base.py` | Base saver with parallel state setup |
| `tools/checkpoint/utils.py` | `_ConverterFakeProcessGroup` class |

## GCS Locations

| Model | GCS Path |
|-------|----------|
| DeepSeek V3 BF16 (HF) | `gs://fundamental_ml_shared_storage/models/DeepSeek-V3-bf16/` |
| DeepSeek V3.2 FP8 | `gs://fundamental_ml_shared_storage/models/DeepSeek-V3.2-fp8/` |
| DeepSeek V3.2 BF16 | `gs://fundamental_ml_shared_storage/models/DeepSeek-V3.2-bf16/` |
