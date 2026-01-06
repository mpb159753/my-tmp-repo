#!/bin/bash
set -e

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "/Users/mpb/WorkSpace/local_job/venv" ]; then
    source /Users/mpb/WorkSpace/local_job/venv/bin/activate
else
    echo "Warning: No venv found."
fi

echo "Step 1: Preparing Dataset Split..."
python train/prepare_explicit_split.py

echo "Step 2: Training (v3 - Segmentation)..."
# Using caffeinate -i to prevent system sleep during training
# Configured for M4 16GB: Batch 4, using system python/venv
caffeinate -i python train/train_tacta_v3.py \
    --data train/tacta.yaml \
    --epochs 100 \
    --batch 4 \
    --imgsz 1024 \
    --device mps \
    --workers 0 \
    --cache False \
    --save-period 1 \
    --project train/runs \
    --name tacta_m4_v3 \
    --resume

echo "Training Complete."
