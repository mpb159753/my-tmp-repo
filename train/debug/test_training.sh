#!/bin/bash
set -e

# 1. Create Mock Data
echo "Creating mock data..."
/Users/mpb/WorkSpace/local_job/venv/bin/python /Users/mpb/WorkSpace/local_job/train/debug/create_mock.py

# 2. Run Training Script (Dry Run)
echo "Running training script test..."
# We override args to make it fast
/Users/mpb/WorkSpace/local_job/venv/bin/python /Users/mpb/WorkSpace/local_job/train/train_tacta.py \
    --data /Users/mpb/WorkSpace/local_job/train/debug/mock_tacta.yaml \
    --epochs 1 \
    --imgsz 64 \
    --batch 2 \
    --name debug_run \
    --project /Users/mpb/WorkSpace/local_job/train/debug/runs

echo "Test complete!"
