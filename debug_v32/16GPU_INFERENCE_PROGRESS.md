# DeepSeek V3.2 16-GPU Inference Progress

## Goal
Validate Megatron-core implementation for DeepSeek V3.2 using pure Megatron (no HF transformers dependency).

## Configuration
- **Nodes**: 2 nodes × 8 H200 GPUs = 16 GPUs total (2.26TB VRAM)
- **Model**: DeepSeek V3.2 (671B params, 1.34TB in BF16)
- **Parallelism**: TP=8, EP=2
- **Checkpoint**: `/mnt/models-disk/DeepSeek-V3.2-megatron` (Megatron format, TP=1)
- **Script**: `debug_v32/pure_megatron_16gpu_v2.py`
- **Branch**: `shuyingl/deepseek-v32-bridge` on `lyfegame/Megatron-LM`

## Cluster Nodes
| Node | Internal IP | Zone |
|------|-------------|------|
| h200-mig-cluster-rn1h | 10.0.0.17 | europe-west1-b |
| h200-mig-cluster-91l8 | 10.0.0.20 | europe-west1-b |

## Progress Log

### 2024-12-16 01:00 UTC - Initial Setup
- Created `pure_megatron_16gpu_v2.py` script using `DeepSeekV32ModelProvider`
- Script uses `tokenizers` library (not HF transformers) for tokenization
- Script uses `dist_checkpointing.load()` for checkpoint loading with resharding

### 2024-12-16 01:04 UTC - Error: init_method is None
- **Error**: `TypeError: 'NoneType' object is not callable` in `_initialize_affine_weight_gpu`
- **Root Cause**: `provider.finalize()` was not called before `get_model()`
- **Fix**: Added `provider.finalize()` after creating `DeepSeekV32ModelProvider`
- **Commit**: `7dc8fdfb2` - "Fix init_method None error - add provider.finalize() call"

### 2024-12-16 01:05 UTC - Error: gradient_accumulation_fusion requires APEX
- **Error**: `RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion set to True but the custom CUDA extension fused_weight_gradient_mlp_cuda module is not found`
- **Root Cause**: `DeepSeekModelProvider` has `gradient_accumulation_fusion=True` by default, but APEX is not installed
- **Fix**: Added `gradient_accumulation_fusion=False` to provider initialization
- **Commit**: `826f2e7ae` - "Disable gradient_accumulation_fusion (requires APEX)"

### 2024-12-16 01:07 UTC - Error: NCCL Bootstrap no socket interface
- **Error**: `NCCL error: Bootstrap : no socket interface found`
- **Root Cause**: NCCL couldn't find network interface for cross-node communication
- **Fix**: Set `NCCL_SOCKET_IFNAME=enp0s19` (interface for 10.0.0.x network)

### 2024-12-16 01:08 UTC - Model Creation Successful on Node 0
- Model created successfully: `44,066,470,144 params on this rank`
- Total params: 44B × 16 ranks ≈ 705B (matches expected ~671B with overhead)
- NCCL using RoCE/IB network: `NET/IB : Using [0]rocep145s0:1/RoCE...`

### 2024-12-16 01:08 UTC - Error: Node 1 missing megatron.bridge
- **Error on Node 1**: `ModuleNotFoundError: No module named 'megatron.bridge'`
- **Root Cause**: PYTHONPATH used `Megatron-Bridge-fork` but actual directory is `Megatron-Bridge`
- **Fix**: Change PYTHONPATH to use `Megatron-Bridge` instead of `Megatron-Bridge-fork`

### 2024-12-16 01:10 UTC - Directory Structure Confirmed
- Both nodes have `Megatron-Bridge/src/megatron/bridge/models/deepseek/deepseek_v32_bridge.py`
- Node 0: `/home/shuyingluo/Megatron-LM/Megatron-Bridge/`
- Node 1: `/home/shuyingluo/Megatron-LM/Megatron-Bridge/`

### 2024-12-16 01:15 UTC - Model Creation Successful on Node 0
- Model config verified:
  - num_layers: 61
  - hidden_size: 7168
  - num_moe_experts: 256
  - TP: 8, EP: 2
- Parameters per rank: 44,066,470,144 (44B)
- Total: 44B × 16 ranks ≈ 705B parameters

### 2024-12-16 01:15 UTC - Error: Node 1 missing Python packages
- **Errors**:
  - `ModuleNotFoundError: No module named 'rich'`
  - `ModuleNotFoundError: No module named 'omegaconf'`
- **Root Cause**: Node 1 has different Python environment than Node 0
- **Fix**: Install missing packages on node 1: `pip install rich omegaconf`

### Missing Packages/Files on Node 1 (vs Node 0)
```bash
pip install rich omegaconf
# Also copy missing file:
cp /mnt/models-disk/slurm_utils.py /home/shuyingluo/Megatron-LM/Megatron-Bridge/src/megatron/bridge/utils/
```

### 2024-12-16 01:30 UTC - Synced Megatron-Bridge utils
- **Issue**: Node 1 Megatron-Bridge was missing `slurm_utils.py`
- **Fix**: Copied file from Node 0 via shared storage (/mnt/models-disk/)

### 2024-12-16 01:35 UTC - Error: Node 1 missing `transformer_engine`
- **Error**: `ModuleNotFoundError: No module named 'transformer_engine'`
- **Root Cause**: Node 1 doesn't have NVIDIA Transformer Engine installed
- **Status**: BLOCKING - transformer_engine is critical for MLA attention

## Environment Differences Between Nodes (RESOLVED)
| Component | Node 0 | Node 1 | Resolution |
|-----------|--------|--------|------------|
| rich | installed | missing | `pip install rich` |
| omegaconf | installed | missing | `pip install omegaconf` |
| slurm_utils.py | present | missing | copy from /mnt/models-disk/ |
| transformer_engine_cu12 | 2.10.0 | missing | `pip install transformer_engine_cu12==2.10.0` |
| transformer_engine_torch | 2.10.0 | missing | `pip install transformer_engine_torch==2.10.0` |

### 2024-12-16 01:40 UTC - transformer_engine installed on Node 1
- Installed transformer_engine_cu12==2.10.0 and transformer_engine_torch==2.10.0

### 2024-12-16 01:45 UTC - CRITICAL: PyTorch version mismatch
- **Node 0**: PyTorch 2.9.1+cu128
- **Node 1**: PyTorch 2.6.0+cu124
- **Impact**: transformer_engine compiled against wrong PyTorch version, causing ImportError
- **Status**: RESOLVED

### 2024-12-16 02:00 UTC - Node 1 Environment Fixed
- Reinstalled torch==2.9.1 (matching Node 0)
- Installed nvidia-modelopt==0.40.0
- Installed nvidia-cuda-cccl, nvidia-cuda-runtime, nvidia-nvjitlink, nvidia-nvvm
- Updated torchvision==0.24.1
- All imports now work: torch, transformer_engine, DeepSeekV32ModelProvider

## Required Environment Setup for Both Nodes

### Python Packages (Critical)
```bash
pip install torch==2.9.1 torchvision==0.24.1
pip install transformer_engine==2.10.0 transformer_engine_cu12==2.10.0 transformer_engine_torch==2.10.0
pip install nvidia-modelopt==0.40.0
pip install rich omegaconf
pip install nvidia-cuda-cccl==13.1.78 nvidia-cuda-runtime==13.1.80 nvidia-nvjitlink==13.1.80 nvidia-nvvm==13.1.80 nvidia-cuda-nvrtc==13.1.80
```

### Megatron-Bridge Setup
```bash
# Ensure slurm_utils.py exists
ls /home/shuyingluo/Megatron-LM/Megatron-Bridge/src/megatron/bridge/utils/slurm_utils.py
```

## Environment Variables Required
```bash
export PYTHONPATH=/home/shuyingluo/Megatron-LM:/home/shuyingluo/Megatron-LM/Megatron-Bridge/src:$PYTHONPATH
export NCCL_SOCKET_IFNAME=enp0s19
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/targets/x86_64-linux/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.10/dist-packages/cusparselt/lib:$LD_LIBRARY_PATH
```

## Launch Commands
```bash
# Node 0 (master):
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=10.0.0.17 --master_port=29503 \
    debug_v32/pure_megatron_16gpu_v2.py --tp 8 --ep 2

# Node 1:
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=10.0.0.17 --master_port=29503 \
    debug_v32/pure_megatron_16gpu_v2.py --tp 8 --ep 2
```

### 2024-12-16 03:16 UTC - Error: Meta device loading unsupported for ShardedTensor
- **Error**: `RuntimeError: Found unsupported type <class 'torch.distributed._shard.sharded_tensor.api.ShardedTensor'> for meta device loading.`
- **Root Cause**: Model was created with `init_model_with_meta_device=True`, but MoE experts use ShardedTensor which doesn't support meta device loading
- **Fix**: Changed to `init_model_with_meta_device=False` in `pure_megatron_16gpu_v2.py`
- **Note**: This means model initializes directly on GPU (2.26TB VRAM available for 1.34TB model)

### 2024-12-16 03:24 UTC - Error: Lightning Indexer weights missing from checkpoint
- **Error**: `KeyError: "decoder.layers.0.self_attention.lightning_indexer.linear_wq_b.weight from model not in state dict"`
- **Root Cause**: Using `DeepSeekV32ModelProvider` (which includes Lightning Indexer) but checkpoint was converted for V3 (no indexer)
- **Fix**: Changed to `DeepSeekV3ModelProvider` (import from `deepseek_provider` instead of `deepseek_v32_bridge`)
- **Note**: The Megatron checkpoint at `/mnt/models-disk/DeepSeek-V3.2-megatron` is actually for V3 architecture (no sparse attention indexer)

### 2024-12-16 03:28 UTC - Error: Checkpoint has indexer weights model doesn't expect
- **Error**: `Invalid access pattern for ShardedTensor(key='decoder.layers.0.self_attention.lightning_indexer.linear_wq_b.weight')`
- **Root Cause**: Checkpoint HAS lightning_indexer weights, but model doesn't (because we set index_topk=None)
- **Fix**: Set `validate_access_integrity=False` and `strict=StrictHandling.LOG_UNEXPECTED` in dist_checkpointing.load()
- **Additional Fix**: Explicitly set `index_topk=None` in provider to ensure sparse attention is disabled

### 2024-12-16 03:45 UTC - Model creation successful, checkpoint loading in progress
- **Status**: Model created with 43.9B params/rank (no lightning_indexer)
- **Config Verified**: index_topk=None, use_sparse_attention=False
- **GPU Memory**: 96GB/GPU used
- **Progress**: Checkpoint loading from /mnt/models-disk/DeepSeek-V3.2-megatron with resharding TP1→TP8, EP2

### 2024-12-16 03:59 UTC - NCCL timeout during checkpoint loading
- **Error**: `WorkNCCL(SeqNum=2, OpType=ALLGATHER, ..., Timeout(ms)=600000) ran for 600064 milliseconds before timing out`
- **Root Cause**: Default NCCL timeout (10 minutes) exceeded during slow checkpoint loading from persistent disk
- **Fix**: Increase NCCL timeout with `NCCL_TIMEOUT=1800000` (30 minutes)

## Validation Prompts (Pending)
1. Simple math: "What is 2 + 2?"
2. Greeting
3. Code generation
4. Explanation
5. Long context
6. Sparse trigger
