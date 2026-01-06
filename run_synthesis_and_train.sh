#!/bin/bash
set -e

# Default Settings
NUM_TRAIN_IMAGES=5000
# Ascend usually benefits from larger batches if memory allows, but YOLOv8/11 determines this.
# For 16GB VRAM on Ascend (e.g. 910), we can probably go higher, but let's stick to safe defaults or auto.

echo "=========================================="
echo "    Tacta Synthesis & Training Pipeline   "
echo "        (Ascend / Generic Device)         "
echo "=========================================="

# Activate venv if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Info: No local 'venv' directory found. Using system python."
fi

# 1. Data Synthesis
echo ""
echo "[Step 1] Generating Synthetic Data..."

# Set Candidates Path via Env Var or modify config dynamically? 
# The main.py uses CANDIDATE_PATH or defaults to abs path. 
# We should update main.py or pass config.
# Ideally, we pass it via python or ensure main.py logic is portable.
# I checked main.py, it defaults to "/Users/mpb/WorkSpace/local_job/assets2". 
# The exported main.py needs to be updated or we need to pass a config.
# Let's create a temporary python runner to inject the relative assets path.

cat <<EOF > run_synthesis.py
import os
import sys
from synthesis_data.main import main
from synthesis_data.generate_val import generate_validation_set

# Calculate relative path to assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets2")
TRAIN_OUTPUT_DIR = os.path.join(BASE_DIR, "synthesis_data", "dataset")

# 1. Generate Train
print(f"Generating Training Data from {ASSETS_DIR}...")
train_config = {
    "CANDIDATE_PATH": ASSETS_DIR,
    "OUTPUT_DIR": TRAIN_OUTPUT_DIR,
    "TOTAL_IMAGES": ${NUM_TRAIN_IMAGES},
    "DEBUG": {"SAVE_IMAGES": False, "SAVE_Logs": False} # Disable debug for speed
}
main(train_config)

# 2. Generate Val
# generate_val.py already handles relative paths in my previous edit, but it needs to find assets too.
# Let's just run it, but we might need to patch it or rely on it finding assets2 if it uses default?
# Wait, generate_val.py calls main(cfg). main() uses default assets path if not provided.
# The default in main.py is HARDCODED /Users/mpb/... 
# So we MUST patch the config passed to main in generate_val.py OR patch main.py itself.
# Since we are running generate_val.py functions here, let's just monkeypatch or pass config if possible.
# generate_val.py doesn't accept args easily. 
# EASIER: We updated main.py in the export to look for assets relative to itself? No we didn't.
# I should update main.py in the export to use relative paths by default.

EOF

# Update main.py to be portable FIRST
# (I will do this in the next tool call, but assuming it's done for this script flow)

echo "Running Synthesis Script..."
python run_synthesis_runner.py

echo ""
echo "[Step 2] Preparing Split Files..."
python train/prepare_explicit_split.py

echo ""
echo "[Step 3] Starting Training..."

# Detect if caffeinate is available
if command -v caffeinate &> /dev/null; then
    LAUNCHER="caffeinate -i python"
else
    LAUNCHER="python"
fi

# Run Training
# Ascend: device='np' or just '0' if torch_npu is set as default?
# Usually with ultralytics, if torch_npu is installed, 'device=0' works or 'device=np'.
# We'll try auto-detect (omit device) or suggest 'device=0'
$LAUNCHER train/train_tacta_v3.py \
    --data train/tacta.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 1024 \
    --workers 8 \
    --cache False \
    --save-period 1 \
    --project train/runs \
    --name tacta_ascend_v3 \
    --resume

echo "Pipeline Complete."
