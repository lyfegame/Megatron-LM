#!/bin/bash
# DeepSeek V3 Checkpoint Conversion Script for GCP
# Uses m2-ultramem-208 with Flex Start (Spot) for cost savings
#
# Usage:
#   ./scripts/gcp_deepseek_conversion.sh create   # Create VM and start conversion
#   ./scripts/gcp_deepseek_conversion.sh status   # Check conversion status
#   ./scripts/gcp_deepseek_conversion.sh ssh      # SSH into VM
#   ./scripts/gcp_deepseek_conversion.sh shutdown # Shutdown VM (after completion)
#   ./scripts/gcp_deepseek_conversion.sh delete   # Delete VM completely

set -e

# Configuration
VM_NAME="deepseek-v3-converter"
ZONE="us-central1-a"
MACHINE_TYPE="m2-ultramem-208"
BOOT_DISK_SIZE="500GB"
PROJECT=$(gcloud config get-value project)

# GCS paths
GCS_INPUT="gs://fundamental_ml_shared_storage/models/DeepSeek-V3-bf16"
GCS_OUTPUT="gs://fundamental_ml_shared_storage/models/DeepSeek-V3-megatron"

# Local paths on VM
LOCAL_INPUT="/mnt/disks/data/deepseek-hf"
LOCAL_OUTPUT="/mnt/disks/data/deepseek-megatron"

echo "============================================"
echo "DeepSeek V3 Checkpoint Conversion on GCP"
echo "============================================"
echo "Project: $PROJECT"
echo "VM: $VM_NAME"
echo "Zone: $ZONE"
echo "Machine: $MACHINE_TYPE (5.8TB RAM)"
echo "============================================"

create_vm() {
    echo ""
    echo "Creating VM with Flex Start (Spot) for cost savings..."
    echo "Estimated cost: ~\$10-15/hr (vs ~\$20/hr on-demand)"
    echo ""

    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --boot-disk-size="$BOOT_DISK_SIZE" \
        --boot-disk-type=pd-ssd \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --scopes=storage-full,compute-rw \
        --metadata=startup-script='#!/bin/bash
set -e
exec > /var/log/startup.log 2>&1

echo "=== Starting setup at $(date) ==="

# Install dependencies
apt-get update
apt-get install -y python3-pip git screen htop

# Install Python packages
pip3 install torch transformers tqdm safetensors

# Create data directory
mkdir -p /mnt/disks/data
chmod 777 /mnt/disks/data

echo "=== Setup complete at $(date) ==="
echo "SETUP_COMPLETE" > /tmp/setup_status
'

    echo ""
    echo "VM created! Waiting for startup script to complete..."
    echo "This may take 5-10 minutes for package installation."
    echo ""

    # Wait for VM to be ready
    sleep 30

    # Wait for setup to complete
    for i in {1..30}; do
        echo "Checking setup status (attempt $i/30)..."
        if gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="cat /tmp/setup_status 2>/dev/null" 2>/dev/null | grep -q "SETUP_COMPLETE"; then
            echo "Setup complete!"
            break
        fi
        sleep 20
    done

    echo ""
    echo "============================================"
    echo "VM is ready! Next steps:"
    echo "============================================"
    echo ""
    echo "1. SSH into the VM:"
    echo "   gcloud compute ssh $VM_NAME --zone=$ZONE"
    echo ""
    echo "2. Run the conversion (inside VM):"
    echo "   screen -S convert"
    echo "   # Then run the conversion script (see below)"
    echo ""
}

run_conversion() {
    echo "SSHing into VM and starting conversion..."

    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
set -e
cd /mnt/disks/data

echo '============================================'
echo 'DeepSeek V3 Checkpoint Conversion'
echo '============================================'

# Clone megatron repo if not exists
if [ ! -d 'megatron-lm' ]; then
    echo 'Cloning megatron-lm repository...'
    git clone https://github.com/NVIDIA/Megatron-LM.git megatron-lm
    cd megatron-lm
    # Checkout your branch - update this!
    git fetch origin shuyingl/add-transformers-dir
    git checkout shuyingl/add-transformers-dir
else
    cd megatron-lm
    git pull
fi

echo ''
echo 'Downloading DeepSeek V3 weights from GCS...'
echo 'This may take 30-60 minutes for 1.3TB...'
mkdir -p $LOCAL_INPUT
gsutil -m cp -r $GCS_INPUT/* $LOCAL_INPUT/

echo ''
echo 'Starting checkpoint conversion...'
echo 'This may take 2-4 hours...'
mkdir -p $LOCAL_OUTPUT

python3 tools/checkpoint/convert.py \\
    --model-type GPT \\
    --loader deepseek_hf \\
    --saver core \\
    --load-dir $LOCAL_INPUT \\
    --save-dir $LOCAL_OUTPUT \\
    --tokenizer-model $LOCAL_INPUT \\
    --target-tensor-parallel-size 8 \\
    --target-pipeline-parallel-size 4 \\
    --bf16 \\
    --megatron-path /mnt/disks/data/megatron-lm

echo ''
echo 'Conversion complete! Uploading to GCS...'
gsutil -m cp -r $LOCAL_OUTPUT/* $GCS_OUTPUT/

echo ''
echo '============================================'
echo 'CONVERSION COMPLETE!'
echo '============================================'
echo 'Output uploaded to: $GCS_OUTPUT'
echo ''
echo 'You can now shutdown the VM:'
echo '  ./scripts/gcp_deepseek_conversion.sh shutdown'
"
}

check_status() {
    echo "Checking VM status..."
    gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --format="table(name,status,machineType.basename())"

    echo ""
    echo "Checking conversion progress (if running)..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="
        echo 'Disk usage:'
        df -h /mnt/disks/data 2>/dev/null || echo 'Data disk not mounted'
        echo ''
        echo 'Memory usage:'
        free -h
        echo ''
        echo 'Running processes:'
        ps aux | grep -E 'python|convert' | grep -v grep || echo 'No conversion running'
        echo ''
        echo 'Recent logs:'
        tail -20 /var/log/startup.log 2>/dev/null || echo 'No startup log'
    " 2>/dev/null || echo "Cannot connect to VM (may be stopped)"
}

ssh_vm() {
    echo "SSHing into VM..."
    gcloud compute ssh "$VM_NAME" --zone="$ZONE"
}

shutdown_vm() {
    echo "Stopping VM (preserves disk, can restart later)..."
    gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
    echo ""
    echo "VM stopped! To restart: gcloud compute instances start $VM_NAME --zone=$ZONE"
}

delete_vm() {
    echo "WARNING: This will delete the VM and all its data!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" = "yes" ]; then
        gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
        echo "VM deleted."
    else
        echo "Cancelled."
    fi
}

# Main command handler
case "${1:-help}" in
    create)
        create_vm
        ;;
    convert)
        run_conversion
        ;;
    status)
        check_status
        ;;
    ssh)
        ssh_vm
        ;;
    shutdown)
        shutdown_vm
        ;;
    delete)
        delete_vm
        ;;
    *)
        echo "Usage: $0 {create|convert|status|ssh|shutdown|delete}"
        echo ""
        echo "Commands:"
        echo "  create   - Create VM with Flex Start (Spot)"
        echo "  convert  - SSH in and run conversion"
        echo "  status   - Check VM and conversion status"
        echo "  ssh      - SSH into the VM"
        echo "  shutdown - Stop VM (preserves disk)"
        echo "  delete   - Delete VM completely"
        ;;
esac
