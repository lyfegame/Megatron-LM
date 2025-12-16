# Claude Code Instructions for DeepSeek V3.2 Megatron Project

## Fire and Forget Pattern (IMPORTANT)

**Always run long-running commands in the background** so the user can interrupt and ask questions.

```bash
# GOOD: Fire and forget pattern
nohup python3 script.py > /tmp/output.log 2>&1 &
echo "Started PID: $!"

# Then monitor separately
tail -f /tmp/output.log
```

```bash
# BAD: Blocking command that prevents interaction
python3 script.py  # User cannot interrupt
```

### Why This Matters
- Model loading can take 5-10+ hours
- Checkpoint conversion takes hours
- User needs to be able to ask questions while tests run
- Background processes survive SSH disconnections

## Cluster Configuration

### Available Nodes

| Node | Internal IP | Role | Status |
|------|-------------|------|--------|
| `h200-mig-cluster-rn1h` | 10.0.0.17 | Primary (NFS server) | Active |
| `h200-mig-cluster-91l8` | 10.0.0.20 | Secondary (NFS client) | Active |

### Hardware Per Node
- 8x NVIDIA H200 (141GB each = 1.13TB VRAM)
- 2.8 TiB CPU RAM
- 32x 375GB local NVMe (12TB total, NOT MOUNTED by default)
- 20TB persistent disk at `/mnt/models-disk`

### Storage

| Path | Type | Speed | Notes |
|------|------|-------|-------|
| `/mnt/models-disk` | Persistent disk (NFS shared) | ~500 MB/s | Contains checkpoints |
| `/dev/nvme0n1` - `/dev/nvme32n1` | Local NVMe | 3-7 GB/s | **NOT MOUNTED** - need to create RAID |
| `/models-local` | Symlink to `/mnt/models-disk` | ~500 MB/s | Legacy path |

### Pre-flight Checklist for Large Model Tests

1. **Verify storage location**:
   ```bash
   df -h /path/to/checkpoint
   # Should show local NVMe for fast loading
   ```

2. **Check if using local NVMe or persistent disk**:
   ```bash
   lsblk | grep nvme
   mount | grep nvme
   ```

3. **For fastest loading**, copy checkpoint to local NVMe RAID (if available)

## Model Checkpoints

| Checkpoint | Path | Size | Format |
|------------|------|------|--------|
| DeepSeek V3.2 FP8 (original) | `/mnt/models-disk/DeepSeek-V3.2-fp8` | 643 GB | safetensors |
| DeepSeek V3.2 Megatron | `/mnt/models-disk/DeepSeek-V3.2-megatron` | 1.3 TB | .distcp (BF16) |

## Multi-Node Inference

For 16-GPU inference across 2 nodes:

```bash
# On primary node (rn1h):
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
    --master_addr=10.0.0.17 --master_port=29500 \
    script.py

# On secondary node (91l8):
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
    --master_addr=10.0.0.17 --master_port=29500 \
    script.py
```

## Lessons Learned

1. **Always use local NVMe for large checkpoint loading** - persistent disk is 6-14x slower
2. **`.distcp` format has high deserialization overhead** - safetensors is much faster
3. **CPU initialization is slow** - GPU loading is faster when model fits in VRAM
4. **Always run long operations in background** - use nohup and redirect to log files
