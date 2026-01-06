# Exported Tacta Training Project (Ascend / Universal)

This package contains the **Synthesis Engine**, **Raw Assets**, and **Training Scripts** tailored for the Tacta Scoring System. It is designed to run on the Ascend platform (or any generic CUDA/MPS device) by generating data on-the-fly to avoid large file transfers.

## Contents
- `run_synthesis_and_train.sh`: **Master script** that runs the entire pipeline (Synthesis -> Split -> Train).
- `synthesis_data/`: The synthesis engine source code.
- `assets2/`: Raw card assets used to generate the dataset.
- `train/`: Training scripts and configuration.
- `train/runs/`: Previous training checkpoints (if copied).

## Setup on Ascend / New Machine

1.  **Environment Setup**:
    Ensure Python 3.8+ and PyTorch are installed.
    *   **Ascend (NPU)**: Ensure the CANN toolkit and `torch_npu` are installed and configured.
    *   **Generic (GPU)**: Ensure standard PyTorch with CUDA is installed.

    Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline**:
    The `run_synthesis_and_train.sh` script handles everything:
    1.  Generates 5000 training images from `assets2`.
    2.  Generates 1000 validation images (Easy/Normal/Hard/Extreme).
    3.  Creates the training split files.
    4.  Starts YOLOv11 Instance Segmentation training.

    Execute:
    ```bash
    chmod +x run_synthesis_and_train.sh
    ./run_synthesis_and_train.sh
    ```

### 3. Offline Environment (Important)
Since this repository was stripped of large weight files (`.pt`) to comply with GitHub limits, you must **manually import the base model** if the target machine has no internet access.

1.  **Download Base Model**:
    On a machine with internet, download `yolo11m-seg.pt` from the Ultralytics release page:
    *   URL: `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt`
    *   (Or search "YOLO11m-seg.pt download")

2.  **Transfer & Place**:
    Copy `yolo11m-seg.pt` to the root of this `export` directory on the target machine.
    ```
    export/
      ├── run_synthesis_and_train.sh
      ├── yolo11m-seg.pt  <-- Place here
      ...
    ```

3.  **Run**:
    The script will automatically find and use the local `yolo11m-seg.pt` instead of trying to download it.

## Customization
- **Training Config**: Edit `train/tacta.yaml` or `run_synthesis_and_train.sh` to adjust batch size (default 16 for Ascend) or epochs.
- **Data Size**: Edit `run_synthesis_and_train.sh` (`NUM_TRAIN_IMAGES`) to change the dataset size.
- **Hardware**: The scripts are configured to auto-detect the device. On Ascend, ensure `torch_npu` is importable; YOLO should handle `device=0` or `device=np` automatically if the environment is standard. If needed, edit the training command in `run_synthesis_and_train.sh`.

