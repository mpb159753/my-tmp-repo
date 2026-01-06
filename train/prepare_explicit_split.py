import os
import glob

# Config - Relative paths for portability
# This script determines the absolute paths based on its own location
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # .../train
PROJECT_ROOT = os.path.dirname(BASE_DIR) # .../ (The export root)
TRAIN_ROOT = os.path.join(PROJECT_ROOT, "synthesis_data", "dataset")
VAL_ROOT = os.path.join(PROJECT_ROOT, "synthesis_data", "dataset_val")
OUTPUT_DIR = BASE_DIR

def get_images(root):
    if not os.path.exists(root):
        return []
    files = glob.glob(os.path.join(root, "images", "*.jpg"))
    files.extend(glob.glob(os.path.join(root, "images", "*.png")))
    return sorted(files) # Sort for consistency

def main():
    print(f"Preparing explicit train/val lists...")
    print(f"  Base Dir: {BASE_DIR}")
    print(f"  Train Root: {TRAIN_ROOT}")
    
    # 1. Train List (All images in synthesis_data/dataset)
    train_files = get_images(TRAIN_ROOT)
    if not train_files:
        print(f"WARNING: No training images found in {TRAIN_ROOT}")
    else:
        print(f"Found {len(train_files)} training images.")
        
    # 2. Val List (All images in synthesis_data/dataset_val)
    val_files = get_images(VAL_ROOT)
    if not val_files:
        print(f"WARNING: No validation images found in {VAL_ROOT}")
    else:
        print(f"Found {len(val_files)} validation images.")
        
    # Write full lists
    # We write absolute paths so that YOLO can find them easily regardless of where verification is run
    with open(os.path.join(OUTPUT_DIR, "train_full.txt"), "w") as f:
        f.write("\n".join(train_files))
        
    with open(os.path.join(OUTPUT_DIR, "val_full.txt"), "w") as f:
        f.write("\n".join(val_files))
        
    print(f"Created {os.path.join(OUTPUT_DIR, 'train_full.txt')}")
    print(f"Created {os.path.join(OUTPUT_DIR, 'val_full.txt')}")

    # Write Granular Validation Lists (Assumes generation order: Easy -> Normal -> Hard -> Extreme)
    # Counts: Easy=200, Normal=300, Hard=300, Extreme=200
    if len(val_files) == 1000:
        val_easy = val_files[0:200]
        val_normal = val_files[200:500]
        val_hard = val_files[500:800]
        val_extreme = val_files[800:1000]
        
        splits = {
            "val_easy.txt": val_easy,
            "val_normal.txt": val_normal,
            "val_hard.txt": val_hard,
            "val_extreme.txt": val_extreme
        }
        
        for name, files in splits.items():
            path = os.path.join(OUTPUT_DIR, name)
            with open(path, "w") as f:
                f.write("\n".join(files))
            print(f"Created {path} ({len(files)} images)")
    else:
        print(f"WARNING: Validation set size is {len(val_files)} (expected 1000). Skipping granular splits.")

if __name__ == "__main__":
    main()
