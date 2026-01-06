#!/bin/bash
set -e

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Info: No local 'venv' directory found. Using system python."
fi

echo "Step 1: Preparing Dataset Split..."
# This will generate train_full.txt and val_full.txt with correct absolute paths for THIS machine
python train/prepare_explicit_split.py

echo "Step 2: Training (v3 - Segmentation)..."

# Detect if caffeinate is available (macOS) to prevent sleep
if command -v caffeinate &> /dev/null; then
    LAUNCHER="caffeinate -i python"
else
    LAUNCHER="python"
fi

# Run training
# Note: device arg removed to let YOLO auto-detect (mps, cuda, or cpu)
$LAUNCHER train/train_tacta_v3.py \
    --data train/tacta.yaml \
    --epochs 100 \
    --batch 4 \
    --imgsz 1024 \
    --workers 0 \
    --cache False \
    --save-period 1 \
    --project train/runs \
    --name tacta_m4_v3 \
    --resume

echo "Training Complete."
