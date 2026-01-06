import argparse
from ultralytics import YOLO
import os
import yaml

def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on granular validation splits")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (e.g. best.pt)")
    parser.add_argument("--data", type=str, default="/Users/mpb/WorkSpace/local_job/train/tacta.yaml", help="Base data config")
    parser.add_argument("--splits-dir", type=str, default="/Users/mpb/WorkSpace/local_job/train", help="Directory containing val_*.txt files")
    args = parser.parse_args()

    # Load base config to understand class names
    with open(args.data, 'r') as f:
        base_data = yaml.safe_load(f)
    
    # Define splits to evaluate
    splits = ["val_easy.txt", "val_normal.txt", "val_hard.txt", "val_extreme.txt", "val_full.txt"]
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    results_summary = []
    
    for split_file in splits:
        split_path = os.path.join(args.splits_dir, split_file)
        if not os.path.exists(split_path):
            print(f"Skipping {split_file} (Not found)")
            continue
            
        print(f"\n=== Evaluating on {split_file} ===")
        
        # Create a temporary yaml for this split
        temp_yaml = f"temp_{split_file.replace('.txt', '.yaml')}"
        split_data = base_data.copy()
        split_data['val'] = split_path
        # We don't strictly need 'train' for validation, but YOLO config might require it. keep it pointing to something valid.
        
        with open(temp_yaml, 'w') as f:
            yaml.dump(split_data, f)
            
        try:
            # Run validation
            metrics = model.val(data=temp_yaml, verbose=False)
            
            # Extract key metrics
            map50 = metrics.box.map50
            map50_95 = metrics.box.map
            
            results_summary.append({
                "split": split_file,
                "mAP50": map50,
                "mAP50-95": map50_95,
                "precision": metrics.box.mp, # mean precision
                "recall": metrics.box.mr     # mean recall
            })
            
        except Exception as e:
            print(f"Error evaluating {split_file}: {e}")
        finally:
            if os.path.exists(temp_yaml):
                os.remove(temp_yaml)

    # Print Summary Table
    print("\n\n" + "="*65)
    print(f"{'Split Name':<20} | {'mAP50':<10} | {'mAP50-95':<10} | {'P':<8} | {'R':<8}")
    print("-" * 65)
    for res in results_summary:
        print(f"{res['split']:<20} | {res['mAP50']:.4f}     | {res['mAP50-95']:.4f}     | {res['precision']:.4f}   | {res['recall']:.4f}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
