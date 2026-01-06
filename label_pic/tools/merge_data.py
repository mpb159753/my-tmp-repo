import os
import shutil
import json

def merge_datasets(original_dir, augmented_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 1. Process Original Data
    print(f"Processing original data from {original_dir}...")
    files_orig = os.listdir(original_dir)
    for f in files_orig:
        src_path = os.path.join(original_dir, f)
        dst_path = os.path.join(target_dir, f)
        
        # Determine if it's a file before copying
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    print("Original data copied.")

    # 2. Process Augmented Data (rename to avoid conflict)
    print(f"Processing augmented data from {augmented_dir}...")
    files_aug = os.listdir(augmented_dir)
    
    count_renamed = 0
    
    # Identify json and images pairs to handle them together logic
    # But simple iteration works if we just rename everything consistently
    
    for f in files_aug:
        src_path = os.path.join(augmented_dir, f)
        
        if not os.path.isfile(src_path):
            continue
            
        new_name = f"mirror_{f}"
        dst_path = os.path.join(target_dir, new_name)
        
        shutil.copy2(src_path, dst_path)
        
        # If it's a JSON file, we need to update the imagePath inside it
        if f.endswith('.json'):
            with open(dst_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            
            # Update imagePath
            # The corresponding image would also have been renamed to mirror_{original_image_name}
            old_image_name = data['imagePath']
            new_image_name = f"mirror_{os.path.basename(old_image_name)}"
            data['imagePath'] = new_image_name
            
            with open(dst_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=2, ensure_ascii=False)
                
        count_renamed += 1

    print(f"Augmented data processed. {count_renamed} files processed.")
    
    # Final count
    total_files = len(os.listdir(target_dir))
    print(f"Total files in {target_dir}: {total_files}")
    
    # Verification check (expect 108 * 2  images + 108 * 2 jsons = 432 files)
    expected = 432
    if total_files == expected:
        print("SUCCESS: File count matches expected (216 images + 216 jsons).")
    else:
        print(f"WARNING: File count {total_files} does not match expected {expected}.")

if __name__ == "__main__":
    original_data = "dataset_final"
    augmented_data = "dataset_final2"
    target_assets = "assets"
    
    merge_datasets(original_data, augmented_data, target_assets)