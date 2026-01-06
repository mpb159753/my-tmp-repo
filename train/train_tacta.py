import argparse
from ultralytics import YOLO
import os

def main(args):
    # 1. Load the model
    # Using YOLO11-m as requested (assuming yolo11m.pt is the correct name for v11 in ultralytics package, 
    # checking recent versions, usually it is strictly yolo11m.pt if v11 is supported, else yolov8m.pt as fallback? 
    # Readme says "YOLO11-m". I will use 'yolo11m.pt'. If it's not found, user might need to update or use v8)
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # 2. Train
    print("Starting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        cache=args.cache,
        save_period=args.save_period,
        resume=args.resume,
        
        # Augmentation Settings (Optimized for Synthetic Data)
        mosaic=0.0,       # Disabled for synthetic data
        degrees=0.0,      # Disabled: Synthetic data already covers 0-360 rotation; avoids corner cropping/padding
        scale=0.0,        # Disabled: Prevent shrinking which loses detail on small anchors
        hsv_h=0.05,       # Increased HSV variance (default 0.015)
        hsv_s=0.7,        # default 0.7
        hsv_v=0.5,        # Increased from 0.4
        
        # Other settings
        exist_ok=True,    # Overwrite existing experiment
        patience=5,       # Aggressive early stopping (10 -> 5 as per request)
        save=True,        # Ensure best/last are saved
    )
    
    # 3. Validation (Auto runs after training, but explicit call here if needed)
    # results = model.val()

    # 4. Export
    print("Exporting to ONNX...")
    # Opset 12, Dynamic shapes disabled
    path = model.export(format="onnx", opset=12, dynamic=False)
    print(f"Model exported to: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11-m for Tacta Board Game")
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Base model path")
    parser.add_argument("--data", type=str, default="/Users/mpb/WorkSpace/local_job/train/tacta.yaml", help="Data config path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (Approx 16-32 for 16GB RAM M4)")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu, mps, cuda)") 
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (keep low for Mac)")
    parser.add_argument("--cache", type=str, default="False", help="Cache images (False/ram/disk). Use False for low RAM.")
    parser.add_argument("--save-period", type=int, default=1, help="Save checkpoint every X epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--project", type=str, default="/Users/mpb/WorkSpace/local_job/train/runs", help="Project name")
    parser.add_argument("--name", type=str, default="tacta_v1", help="Experiment name")
    
    args = parser.parse_args()
    
    # Convert "False"/"True" strings to bool if applicable
    if args.cache == "False": args.cache = False
    elif args.cache == "True": args.cache = True
    
    # Check if data files exist
    import yaml
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
        train_path = data_cfg.get('train')
        if not os.path.exists(train_path):
             print(f"WARNING: Train file list not found at {train_path}. Please run prepare_split.py first.")

    # Resume Logic
    if args.resume:
        # Construct expected path
        last_ckpt = os.path.join(args.project, args.name, "weights", "last.pt")
        if os.path.exists(last_ckpt):
            print(f"Resuming from checkpoint: {last_ckpt}")
            args.model = last_ckpt
        else:
            print(f"Resume requested but no checkpoint found at {last_ckpt}. Starting fresh.")
            args.resume = False # Reset to false so we don't pass resume=True with a fresh model which might error or be confusing

    main(args)
