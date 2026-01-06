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

# Calculate relative path to assets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets2")
TRAIN_OUTPUT_DIR = os.path.join(BASE_DIR, "synthesis_data", "dataset")

# Add paths for imports
sys.path.append(os.path.join(BASE_DIR, "synthesis_data"))
sys.path.append(BASE_DIR)

try:
    # Importing main directly since synthesis_data is in path
    from main import main
    # Importing generate_val directly
    from generate_val import generate_validation_set
except ImportError:
    # Fallback/Alternative depending on how checking path resolves
    from synthesis_data.main import main
    from synthesis_data.generate_val import generate_validation_set

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
print("Generating Validation Data...")
# We need to ensure generate_val uses the correct assets path too.
# Inspecting generate_val.py: It calls main(cfg). Since we updated main.py (in Step 87)
# to default to relative ../assets2 if CANDIDATE_PATH matches default, it should work fine
# as long as we run it from the correct working directory OR main.py logic holds.
# main.py logic: config["CANDIDATE_PATH"] = os.path.join(base_dir, "..", "assets2")
# If base_dir is synthesis_data/, .. is export/. assets2 is export/assets2. Correct.
generate_validation_set()

EOF

echo "Running Synthesis Script..."
python run_synthesis.py

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
