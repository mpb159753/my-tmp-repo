import argparse
from ultralytics import YOLO
import os
import yaml
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

def main(args):
    # Models to train
    models = ['yolo11n-seg.pt', 'yolo11s-seg.pt', 'yolo11m-seg.pt']
    
    for base_model in models:
        model_name = base_model.split('.')[0]
        run_name = f"{args.base_name}_{model_name}"
        
        print(f"\n{'='*50}")
        print(f"Starting Training for {base_model} -> {run_name}")
        print(f"{'='*50}\n")
        
        try:
            # 1. Load the model logic
            resume_run = False
            if args.resume:
                last_ckpt = os.path.join(args.project, run_name, "weights", "last.pt")
                if os.path.exists(last_ckpt):
                    print(f"Resuming {run_name} from {last_ckpt}")
                    model = YOLO(last_ckpt)
                    resume_run = True
                else:
                    print(f"Resume requested but no checkpoint found for {run_name}. Starting fresh.")
                    model = YOLO(base_model)
            else:
                model = YOLO(base_model)

            # 2. Train
            results = model.train(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=args.project,
                name=run_name,
                workers=args.workers,
                cache=args.cache,
                save_period=args.save_period,
                
                # Augmentation Settings (Optimized for Synthetic Data)
                mosaic=0.5,       # Enabled: Helps with small object detection and context
                degrees=0.0,      # Disabled: Synthetic data already covers rotation
                scale=0.5,        # Enabled: Essential for dealing with varying object sizes (missing link for full image inference)
                hsv_h=0.05,       # Increased HSV variance
                hsv_s=0.7,        # default 0.7
                hsv_v=0.5,        # Increased from 0.4
                
                # Other settings
                exist_ok=True,    # Overwrite existing experiment folder
                patience=5,       # Aggressive early stopping
                patience=5,       # Aggressive early stopping
                save=True,        # Ensure best/last are saved
                resume=resume_run,
            )
            
            # 3. Export
            # Note: Best weights are at project/name/weights/best.pt
            best_weight_path = os.path.join(args.project, run_name, "weights", "best.pt")
            if os.path.exists(best_weight_path):
                print(f"Exporting best model for {model_name}...")
                best_model = YOLO(best_weight_path)
                path = best_model.export(format="onnx", opset=12, dynamic=False)
                print(f"Model exported to: {path}")
            else:
                print(f"Warning: Could not find best.pt for {model_name}")

        except Exception as e:
            print(f"Error training {base_model}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11 variants for Tacta")
    parser.add_argument("--data", type=str, default="tacta.yaml", help="Data config path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs (default 20)")
    parser.add_argument("--imgsz", type=int, default=2048, help="Image size (default 2048)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--device", type=str, default="0", help="Device (cpu, mps, cuda, 0, npu:0)") 
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--cache", type=str, default="False", help="Cache images (False/ram/disk)")
    parser.add_argument("--save-period", type=int, default=1, help="Save checkpoint every X epochs")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    # Use path relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_project = os.path.join(script_dir, "runs")
    parser.add_argument("--project", type=str, default=default_project, help="Project name")
    parser.add_argument("--base_name", type=str, default="tacta_v3", help="Base experiment name prefix")
    
    args = parser.parse_args()
    
    # Convert "False"/"True" strings to bool if applicable
    if args.cache == "False": args.cache = False
    elif args.cache == "True": args.cache = True
    
    # Check if data files exist
    with open(args.data, 'r') as f:
        data_cfg = yaml.safe_load(f)
        train_path = data_cfg.get('train')
        if not os.path.exists(train_path):
             print(f"WARNING: Train file list not found at {train_path}. Please run prepare_split.py first.")

    main(args)
