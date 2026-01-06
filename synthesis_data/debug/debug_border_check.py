
import json
import cv2
import numpy as np
import os

def visualize_and_check(image_path, json_path, output_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Draw annotations
    vis_img = img.copy()
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.polylines(vis_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        # Put label
        cv2.putText(vis_img, shape['label'], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Try background removal / card detection logic
    # Assume white background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Invert so white becomes black (0) and content is brighter? 
    # Or just threshold.
    # If background is white (255, 255, 255), we can look for non-white pixels.
    # Actually, simpler: threshold near 255.
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the card
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        print(f"Detected bounding box: x={x}, y={y}, w={w}, h={h}")
        print(f"Original size: {img.shape[1]}x{img.shape[0]}")
    else:
        print("No contours found.")

    cv2.imwrite(output_path, vis_img)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    visualize_and_check(
        "/Users/mpb/WorkSpace/local_job/assets/blue_1_card_00.png",
        "/Users/mpb/WorkSpace/local_job/assets/blue_1_card_00.json",
        "debug_vis.png"
    )
