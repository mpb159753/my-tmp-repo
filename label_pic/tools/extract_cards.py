import cv2
import numpy as np
import os
import glob

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    if maxWidth > maxHeight:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

def sort_contours_grid(cnts):
    # Reliable 3x3 Grid Sorting
    # 1. Calculate centers
    centers = []
    for c in cnts:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centers.append((c, cX, cY))
    
    # 2. Sort primarily by Y
    centers.sort(key=lambda x: x[2])
    
    # 3. Chunk into rows of 3 (assuming 9 cards total)
    # If we don't have exactly 9, this logic might be weak, so we fallback to simple Y sort if not 9
    if len(cnts) == 9:
        rows = []
        for i in range(0, 9, 3):
            row = centers[i:i+3]
            # 4. Sort each row by X
            row.sort(key=lambda x: x[1])
            rows.extend(row)
        return [r[0] for r in rows]
    else:
        # Fallback for non-9 counts: Sort by Y
        return [x[0] for x in centers]

def extract_cards(input_dir, output_dir):
    ensure_dir(output_dir)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    print(f"Found {len(image_files)} images in {input_dir}")
    
    for img_path in image_files:
        print(f"Processing: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast: CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(gray, (7, 7), 0) # Slightly stronger blur
        
        img_area = img.shape[0] * img.shape[1]

        # Robust Candidate Filtering
        def get_best_candidates(contours_input, k=9):
            valid = []
            valid_metrics = [] # Stores (cnt, rect_area)

            for cnt in contours_input:
                area = cv2.contourArea(cnt)
                
                # Broad Filtering
                if area < (img_area * 0.005): continue

                rect_rot = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect_rot
                rect_area = w * h
                
                if w == 0 or h == 0: continue
                ar = max(w, h) / min(w, h)
                if ar < 1.0 or ar > 5.0: continue
                
                # Check Solidity (Area / Rect_Area)
                # Real cards are solid rectangles, so ratio should be close to 1.0 (e.g. > 0.8)
                # A bounding box around a thin "L" frame will have low solidity.
                if rect_area > 0:
                     solidity = area / rect_area
                     if solidity < 0.7:
                         continue # Skip hollow/sparse shapes

                valid.append(cnt)
                valid_metrics.append(rect_area) # Use Rect Area for median stats
            
            if not valid:
                return []
            
            # Zip and Sort by Rect Area Descending
            # (Using Rect Area is more representative of the final image size)
            combined = sorted(zip(valid, valid_metrics), key=lambda x: x[1], reverse=True)
            
            # Median Filtering logic
            # Take top 15 candidates to estimate median card size
            candidates_top = combined[:min(len(combined), 15)]
            rect_areas_top = [m[1] for m in candidates_top]
            
            if not rect_areas_top: return []

            median_area = np.median(rect_areas_top)
            
            final_candidates = []
            for cnt, r_area in combined:
                # Filter limits: Tighter bounds (0.7x to 1.3x) for standard cards
                # Also cap absolute max at 95% of image (just in case)
                if r_area > (median_area * 0.7) and r_area < (median_area * 1.3) and r_area < (img_area * 0.95):
                    final_candidates.append(cnt)
                else:
                    # print(f"DEBUG: Rejected rect_area {r_area:.0f}")
                    pass
            
            # Sort final by Area (or Rect Area)
            final_candidates = sorted(final_candidates, key=cv2.contourArea, reverse=True)
            
            return final_candidates[:k]

        # Single Robust Pipeline: Adaptive Thresholding
        # Canny is too sensitive to texture inside cards. Adaptive is better for outlines.
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 51, 10)
        
        # Morphological Cleanup
        kernel_large = np.ones((5,5), np.uint8)
        # Close gaps
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        # Remove small noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_large, iterations=1)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_candidates = get_best_candidates(contours, k=9)

        # Fallback: if < 9 found, try looser threshold parameters
        if len(card_candidates) < 9:
             print(f"  First pass found only {len(card_candidates)}. Trying looser threshold...")
             thresh_loose = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 99, 5)
             thresh_loose = cv2.morphologyEx(thresh_loose, cv2.MORPH_CLOSE, kernel_large, iterations=3)
             contours_loose, _ = cv2.findContours(thresh_loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
             card_candidates = get_best_candidates(contours_loose, k=9)

        print(f"  Selected top {len(card_candidates)} valid candidates.")
        
        if len(card_candidates) > 0:
             # Sort: Top-Left to Bottom-Right (Grid Sort)
            card_candidates = sort_contours_grid(card_candidates)
        
        card_count = 0
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        for i, cnt in enumerate(card_candidates):
            rect = cv2.minAreaRect(cnt)
            
            # Add Padding to the Box
            (x, y), (w, h), angle = rect
            padding_percent = 0.05 # 5% padding
            w_pad = w * (1 + padding_percent)
            h_pad = h * (1 + padding_percent)
            rect_padded = ((x, y), (w_pad, h_pad), angle)
            
            box = cv2.boxPoints(rect_padded)
            box = np.array(box, dtype="int")
            
            try:
                warped = four_point_transform(img, box)
                
                # Verify warped image isn't empty or tiny (edge case where padding pushes it out)
                if warped.shape[0] == 0 or warped.shape[1] == 0:
                     print(f"  Warning: Warped image invalid for card {i}, retry without padding.")
                     box_orig = cv2.boxPoints(rect)
                     box_orig = np.array(box_orig, dtype="int")
                     warped = four_point_transform(img, box_orig)

                output_filename = f"{img_name}_card_{str(card_count).zfill(2)}.png"
                output_path = os.path.join(output_dir, output_filename)
                cv2.imwrite(output_path, warped)
                print(f"  Saved: {output_filename}")
                card_count += 1
            except Exception as e:
                print(f"  Error warping card {i}: {e}")

if __name__ == "__main__":
    INPUT_DIR = "unlable_pic"
    OUTPUT_DIR = "assets/cards"
    extract_cards(INPUT_DIR, OUTPUT_DIR)