
import cv2
import numpy as np
import os
import json
import random
import glob

def verify_results(assets2_dir, output_path):
    image_files = glob.glob(os.path.join(assets2_dir, "*.png"))
    if not image_files:
        print("No images found in assets2")
        return

    # Pick 9 random images
    samples = random.sample(image_files, min(len(image_files), 9))
    
    vis_images = []
    
    for img_path in samples:
        basename = os.path.basename(img_path)
        json_path = img_path.replace(".png", ".json")
        
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Draw annotations
        for shape in data.get('shapes', []):
            points = np.array(shape['points'], dtype=np.int32)
            cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=3)
            
        # Resize for grid
        img = cv2.resize(img, (200, 300))
        # Add border
        img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        vis_images.append(img)
        
    # Create grid 3x3
    # Pad if less than 9
    while len(vis_images) < 9:
        vis_images.append(np.zeros((304, 204, 3), dtype=np.uint8))
        
    row1 = np.hstack(vis_images[:3])
    row2 = np.hstack(vis_images[3:6])
    row3 = np.hstack(vis_images[6:9])
    
    grid = np.vstack([row1, row2, row3])
    
    cv2.imwrite(output_path, grid)
    print(f"Saved verification grid to {output_path}")

if __name__ == "__main__":
    verify_results("/Users/mpb/WorkSpace/local_job/assets2", "verification_grid.png")
