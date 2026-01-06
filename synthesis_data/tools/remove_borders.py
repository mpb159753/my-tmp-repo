
import cv2
import numpy as np
import os
import json
import glob
from tqdm import tqdm

def remove_borders(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all png files
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    success_count = 0
    fail_count = 0
    
    for img_path in tqdm(image_files):
        basename = os.path.basename(img_path)
        json_path = os.path.join(input_dir, basename.replace(".png", ".json"))
        
        if not os.path.exists(json_path):
            print(f"Skipping {basename}: No corresponding JSON found.")
            continue
            
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            fail_count += 1
            continue

        # Load JSON
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load JSON {json_path}: {e}")
            fail_count += 1
            continue

        # --- Background Removal Algorithm ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 21, 5)
        
        # Morphological Close
        kernel = np.ones((5,5),np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find Contours
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"No contours found for {basename}")
            fail_count += 1
            continue
            
        # Filter largest contour
        # Must be meaningful size
        img_area = img.shape[0] * img.shape[1]
        
        valid_contours = []
        for c in contours:
            if cv2.contourArea(c) > 0.1 * img_area:
                valid_contours.append(c)
                
        if not valid_contours:
             print(f"No valid contours (large enough) found for {basename}")
             fail_count += 1
             continue
             
        c = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop Image
        cropped_img = img[y:y+h, x:x+w]
        
        # Update JSON Annotations
        new_shapes = []
        for shape in data.get('shapes', []):
            new_points = []
            for pt in shape['points']:
                new_pt = [pt[0] - x, pt[1] - y]
                new_points.append(new_pt)
            shape['points'] = new_points
            new_shapes.append(shape)
            
        data['shapes'] = new_shapes
        data['imageHeight'] = h
        data['imageWidth'] = w
        # Update imagePath to be consistent (just the filename)
        data['imagePath'] = basename 
        
        # Save output
        out_img_path = os.path.join(output_dir, basename)
        out_json_path = os.path.join(output_dir, basename.replace(".png", ".json"))
        
        cv2.imwrite(out_img_path, cropped_img)
        with open(out_json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        success_count += 1

    print(f"Processing Complete.")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove borders from card images.")
    parser.add_argument("--input", default="/Users/mpb/WorkSpace/local_job/assets", help="Input directory")
    parser.add_argument("--output", default="/Users/mpb/WorkSpace/local_job/assets2", help="Output directory")
    args = parser.parse_args()
    
    remove_borders(args.input, args.output)
