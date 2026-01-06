#!/bin/bash
set -e

# Parse Arguments
FORCE_SYNTH=false
for arg in "$@"; do
    case $arg in
        --force-synth)
        FORCE_SYNTH=true
        shift
        ;;
    esac
done

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
echo "[Step 1] Checking Synthetic Data..."

DATASET_DIR="synthesis_data/dataset"
VAL_DIR="synthesis_data/dataset_val"

# Check Training Data
SKIP_TRAIN=false
if [ -d "$DATASET_DIR/images" ] && [ "$(ls -A $DATASET_DIR/images)" ]; then
    SKIP_TRAIN=true
fi

# Check Validation Data
SKIP_VAL=false
if [ -d "$VAL_DIR/images" ] && [ "$(ls -A $VAL_DIR/images)" ]; then
    SKIP_VAL=true
fi

# Force Override
if [ "$FORCE_SYNTH" = true ]; then
    SKIP_TRAIN=false
    SKIP_VAL=false
fi

# Feedback
if [ "$SKIP_TRAIN" = true ]; then
    echo "-> Training data exists. Skipping."
else
    echo "-> Training data missing (or forced). Queued for generation."
fi

if [ "$SKIP_VAL" = true ]; then
    echo "-> Validation data exists. Skipping."
else
    echo "-> Validation data missing (or forced). Queued for generation."
fi

if [ "$SKIP_TRAIN" = true ] && [ "$SKIP_VAL" = true ]; then
    echo "All data exists. Skipping synthesis step."
else
    echo "Running Synthesis Logic..."
    export SKIP_TRAIN=$SKIP_TRAIN
    export SKIP_VAL=$SKIP_VAL

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
    from main import main
    from generate_val import generate_validation_set
except ImportError:
    from synthesis_data.main import main
    from synthesis_data.generate_val import generate_validation_set

# Read Env Vars (Bash sets them as strings 'true'/'false')
skip_train = os.environ.get("SKIP_TRAIN", "false") == "true"
skip_val = os.environ.get("SKIP_VAL", "false") == "true"

# 1. Generate Train
if not skip_train:
    print(f"Generating Training Data from {ASSETS_DIR}...")
    train_config = {
        "CANDIDATE_PATH": ASSETS_DIR,
        "OUTPUT_DIR": TRAIN_OUTPUT_DIR,
        "TOTAL_IMAGES": ${NUM_TRAIN_IMAGES},
        "DEBUG": {"SAVE_IMAGES": False, "SAVE_LOGS": False}
    }
    main(train_config)
else:
    print("Skipping Training Data Generation (already exists)")

# 2. Generate Val
if not skip_val:
    print("Generating Validation Data...")
    generate_validation_set()
else:
    print("Skipping Validation Data Generation (already exists)")

EOF

echo "Running Synthesis Script..."
python run_synthesis.py

fi # End of synthesis check

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
