import glob
import os


def prepare_splits():
    # Base paths
    train_dir = "/Users/mpb/WorkSpace/local_job/synthesis_data/dataset/images"
    val_dir = "/Users/mpb/WorkSpace/local_job/synthesis_data/dataset_val/images"
    
    output_train_txt = "/Users/mpb/WorkSpace/local_job/train/train_full.txt"
    output_val_txt = "/Users/mpb/WorkSpace/local_job/train/val_full.txt"
    
    # Collect Train
    print(f"Scanning {train_dir}...")
    train_images = sorted(glob.glob(os.path.join(train_dir, "*.jpg")))
    print(f"Found {len(train_images)} training images.")
    
    # Collect Val
    print(f"Scanning {val_dir}...")
    val_images = sorted(glob.glob(os.path.join(val_dir, "*.jpg")))
    print(f"Found {len(val_images)} validation images.")
    
    # Write to files
    print(f"Writing {output_train_txt}...")
    with open(output_train_txt, 'w') as f:
        f.write("\n".join(train_images))
        
    print(f"Writing {output_val_txt}...")
    with open(output_val_txt, 'w') as f:
        f.write("\n".join(val_images))
        
    print("Done.")

if __name__ == "__main__":
    prepare_splits()
