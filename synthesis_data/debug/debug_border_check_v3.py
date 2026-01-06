
import cv2
import numpy as np

def debug_border_v3(image_path, json_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive Threshold
    # CVS_ADAPTIVE_THRESH_GAUSSIAN_C or MEAN_C
    # blockSize: Neighborhood size (must be odd). Try 11 or 21.
    # C: Constant subtracted from mean. Try 2.
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5) # Increased C to 5 to avoid noise
    
    # Morphological operations to close bits
    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vis_img = img.copy()
    
    # Filter contours
    found = False
    img_area = img.shape[0] * img.shape[1]
    
    candidates = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.1 * img_area: # Must be at least 10% of image
            x, y, w, h = cv2.boundingRect(c)
            candidates.append((area, (x, y, w, h)))
            
    if candidates:
        # Pick largest
        max_c = max(candidates, key=lambda item: item[0])
        x, y, w, h = max_c[1]
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print(f"Bounding Box: {x},{y}, {w}x{h}")
        found = True
    else:
        print("No suitable contours found.")

    cv2.imwrite(output_path, vis_img)
    cv2.imwrite(output_path.replace(".png", "_thresh.png"), thresh)

if __name__ == "__main__":
    debug_border_v3(
        "/Users/mpb/WorkSpace/local_job/assets/blue_1_card_00.png",
        "/Users/mpb/WorkSpace/local_job/assets/blue_1_card_00.json",
        "debug_vis_v3.png"
    )
