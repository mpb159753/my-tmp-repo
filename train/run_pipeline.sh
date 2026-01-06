#!/bin/bash
set -e

# activate venv
source /Users/mpb/WorkSpace/local_job/venv/bin/activate

echo "Step 1: Preparing Dataset Split..."
python /Users/mpb/WorkSpace/local_job/train/prepare_explicit_split.py

echo "Step 2: Checking Data (Visual Inspection)..."
# Just generates a few debug images in train/debug to verify
# Checking Train set
python /Users/mpb/WorkSpace/local_job/train/debug/check_data.py \
    --data /Users/mpb/WorkSpace/local_job/train/tacta.yaml \
    --images /Users/mpb/WorkSpace/local_job/synthesis_data/synthesis_data/dataset/images \
    --labels /Users/mpb/WorkSpace/local_job/synthesis_data/synthesis_data/dataset/labels

echo "Step 3: Training..."
# Using M4 optimized parameters: mps, batch 8, workers 4, cache Disk (save CPU)
# New experiment: tacta_m4_v2 (Fresh Start)
caffeinate -i python /Users/mpb/WorkSpace/local_job/train/train_tacta.py \
    --data /Users/mpb/WorkSpace/local_job/train/tacta.yaml \
    --epochs 50 \
    --batch 4 \
    --imgsz 1024 \
    --device mps \
    --workers 4 \
    --cache False \
    --save-period 1 \
    --project /Users/mpb/WorkSpace/local_job/train/runs \
    --name tacta_m4_v2

echo "Pipeline Complete."
