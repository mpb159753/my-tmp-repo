import argparse
from ultralytics import YOLO
import os
import yaml

def main(args):
    # 1. Load the model
    # Using YOLO11-m-seg as requested
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # 2. Train
    print(f"Starting training experiment: {args.name}")
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
        degrees=0.0,      # Disabled: Synthetic data already covers rotation
        scale=0.0,        # Disabled: Prevent shrinking which loses detail on small anchors
        hsv_h=0.05,       # Increased HSV variance
        hsv_s=0.7,        # default 0.7
        hsv_v=0.5,        # Increased from 0.4
        
        # Other settings
        exist_ok=True,    # Overwrite existing experiment folder if name collides (we use distinct names)
        patience=5,       # Aggressive early stopping (Matched to v2 config)
        save=True,        # Ensure best/last are saved
    )
    
    # 4. Export
    print("Exporting to ONNX...")
    # Opset 12, Dynamic shapes disabled
    path = model.export(format="onnx", opset=12, dynamic=False)
    print(f"Model exported to: {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11-m-seg for Tacta Board Game (v3)")
    parser.add_argument("--model", type=str, default="yolo11m-seg.pt", help="Base model path (Segmentation)")
    parser.add_argument("--data", type=str, default="/Users/mpb/WorkSpace/local_job/train/tacta.yaml", help="Data config path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--device", type=str, default="mps", help="Device (cpu, mps, cuda)") 
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--cache", type=str, default="False", help="Cache images (False/ram/disk)")
    parser.add_argument("--save-period", type=int, default=1, help="Save checkpoint every X epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--project", type=str, default="/Users/mpb/WorkSpace/local_job/train/runs", help="Project name")
    parser.add_argument("--name", type=str, default="tacta_m4_v3", help="Experiment name (v3)")
    
    args = parser.parse_args()
    
    # Convert "False"/"True" strings to bool if applicable
    if args.cache == "False": args.cache = False
    elif args.cache == "True": args.cache = True
    
    # Check if data files exist
    # Check if data files exist
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
        train_path = data_cfg.get('train')
        
        # Resolve path relative to yaml file if it's not absolute
        yaml_dir = os.path.dirname(os.path.abspath(args.data))
        if train_path and not os.path.isabs(train_path):
            check_path = os.path.join(yaml_dir, train_path)
        else:
            check_path = train_path
            
        if check_path and not os.path.exists(check_path):
             print(f"WARNING: Train file list not found at {check_path}. Please run prepare_split.py first.")

    # Resume Logic
    if args.resume:
        last_ckpt = os.path.join(args.project, args.name, "weights", "last.pt")
        if os.path.exists(last_ckpt):
            print(f"Resuming from checkpoint: {last_ckpt}")
            args.model = last_ckpt
        else:
            print(f"Resume requested but no checkpoint found at {last_ckpt}. Starting fresh.")
            args.resume = False

    main(args)
