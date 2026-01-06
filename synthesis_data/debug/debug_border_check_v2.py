
import cv2
import numpy as np

def debug_border_v2(image_path, json_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load")
        return

    # Flood fill from corners
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # We want to flood fill the background. 
    # Use a tolerance. Corner pixel variance was ~15. So let's try 20 or 30.
    
    # FloodFill modifies image in place or uses a mask.
    # We'll use a copy for floodfill to see the mask.
    ff_img = img.copy()
    
    # New color for background: (0, 255, 0) green
    # loDiff and upDiff: tolerance
    tolerance = (30, 30, 30)
    
    # Fill from all 4 corners
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    
    # Corner 1
    cv2.floodFill(ff_img, mask, (0, 0), (0, 255, 0), tolerance, tolerance, flags)
    # Corner 2
    cv2.floodFill(ff_img, mask, (w-1, 0), (0, 255, 0), tolerance, tolerance, flags)
    # Corner 3
    cv2.floodFill(ff_img, mask, (0, h-1), (0, 255, 0), tolerance, tolerance, flags)
    # Corner 4
    cv2.floodFill(ff_img, mask, (w-1, h-1), (0, 255, 0), tolerance, tolerance, flags)
    
    # Mask is now 255 where background is. 
    # We want the content.
    # Clip mask to image size
    mask_real = mask[1:-1, 1:-1]
    
    # Invert mask -> 255 is content
    content_mask = cv2.bitwise_not(mask_real)
    
    # Find contours
    contours, _ = cv2.findContours(content_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    vis_img = img.copy()
    
    if contours:
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print(f"Bounding Box: {x},{y}, {w}x{h}")
    else:
        print("No contours")

    # Save mask visualization
    cv2.imwrite(output_path, vis_img)
    cv2.imwrite(output_path.replace(".png", "_mask.png"), content_mask)

if __name__ == "__main__":
    debug_border_v2(
        "/Users/mpb/WorkSpace/local_job/assets/blue_1_card_00.png",
        "/Users/mpb/WorkSpace/local_job/assets/blue_1_card_00.json",
        "debug_vis_v2.png"
    )
