import os
import json
import cv2
import numpy as np
import copy

def augment_data(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    files = [f for f in os.listdir(src_dir) if f.endswith('.json')]
    
    print(f"Found {len(files)} JSON files in {src_dir}")

    count = 0
    for file in files:
        json_path = os.path.join(src_dir, file)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        image_name = data['imagePath']
        # Handle path separators just in case
        image_name = os.path.basename(image_name)
        
        image_path = os.path.join(src_dir, image_name)
        if not os.path.exists(image_path):
            # Try replacing extension just in case json has wrong extension
            base_name = os.path.splitext(file)[0]
            possible_exts = ['.png', '.jpg', '.jpeg']
            found = False
            for ext in possible_exts:
                if os.path.exists(os.path.join(src_dir, base_name + ext)):
                    image_name = base_name + ext
                    image_path = os.path.join(src_dir, image_name)
                    found = True
                    break
            if not found:
                print(f"Warning: Image for {file} not found. Skipping.")
                continue

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue
            
        height, width = img.shape[:2]
        
        # Mirror image
        img_flipped = cv2.flip(img, 1) # 1 means horizontal flip
        
        # Mirror annotations
        data_flipped = copy.deepcopy(data)
        data_flipped['imageHeight'] = height
        data_flipped['imageWidth'] = width
        data_flipped['imageData'] = None # Ensure no base64 data to avoid mismatch
        
        for shape in data_flipped['shapes']:
            points = shape['points']
            new_points = []
            for pt in points:
                x, y = pt
                new_x = width - 1 - x
                new_points.append([new_x, y])
            
            # Reverse order to maintain winding (CCW/CW)
            # Though strictly flipping geometry reverses it, 
            # reversing list usually helps maintain "interior is to the left" consistency if labelme cares.
            # But simpler is often better if we aren't sure. 
            # However, for polygons, reversing is good practice when mirroring.
            shape['points'] = new_points
            
        # Save results
        # We will use the same filename in the new directory
        dst_image_path = os.path.join(dst_dir, image_name)
        dst_json_path = os.path.join(dst_dir, file)
        
        cv2.imwrite(dst_image_path, img_flipped)
        
        # Update imagePath in json to point to the file (basename)
        data_flipped['imagePath'] = image_name
        
        with open(dst_json_path, 'w', encoding='utf-8') as f:
            json.dump(data_flipped, f, indent=2, ensure_ascii=False)
            
        count += 1
        
    print(f"Processed {count} images. Saved to {dst_dir}")

if __name__ == "__main__":
    src = "dataset_final"
    dst = "dataset_final2"
    augment_data(src, dst)