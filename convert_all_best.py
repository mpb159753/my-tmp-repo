import os
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Export all best.pt models in a directory to ONNX")
    parser.add_argument("--runs_dir", type=str, default="train/runs", help="Directory containing run folders")
    parser.add_argument("--output_dir", type=str, default="export/onnx", help="Directory to save ONNX models")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Walk through runs dir
    print(f"Scanning {args.runs_dir} for best.pt files...")
    
    count = 0
    
    if not os.path.exists(args.runs_dir):
        print(f"Directory {args.runs_dir} does not exist.")
        return

    for root, dirs, files in os.walk(args.runs_dir):
        if "best.pt" in files:
            weight_path = os.path.join(root, "best.pt")
            
            # Infer model name from run folder name (parent directory)
            # Structure: runs/tacta_m4_v3_yolo11n/weights/best.py
            # Parent of weights is the run name
            parent_dir = os.path.dirname(root) # .../weights
            run_name = os.path.basename(parent_dir) # run_folder_name
            
            output_name = f"{run_name}.onnx"
            output_path = os.path.join(args.output_dir, output_name)
            
            print(f"Found: {weight_path}")
            print(f"Exporting to: {output_path}")
            
            try:
                model = YOLO(weight_path)
                # Export
                # Note: model.export saves to the SAME directory as the pt file by default
                exported_path = model.export(format="onnx", opset=12, dynamic=False, half=True)
                
                # Move to target dir
                if exported_path and os.path.exists(exported_path):
                    os.rename(exported_path, output_path)
                    print(f"Success: {output_path}")
                    count += 1
                else:
                    print("Export failed (file not found).")
                    
            except Exception as e:
                print(f"Error exporting {weight_path}: {e}")
                
    print(f"\nDone. Exported {count} models to {args.output_dir}")

if __name__ == "__main__":
    main()
