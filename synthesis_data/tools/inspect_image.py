
import cv2
import numpy as np

def inspect_pixels(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to load image")
        return

    print(f"Image shape: {img.shape}")
    
    # Check corners
    h, w = img.shape[:2]
    corners = [
        (0, 0), (0, w-1), (h-1, 0), (h-1, w-1)
    ]
    
    for y, x in corners:
        pixel = img[y, x]
        print(f"Pixel at ({y}, {x}): {pixel}")

    # Check approximate border color
    # Sample top row
    top_row = img[0, :]
    print(f"Top row mean: {np.mean(top_row, axis=0)}")
    
    if img.shape[2] == 4:
        print("Image has Alpha channel")
    else:
        print("Image does NOT have Alpha channel")

if __name__ == "__main__":
    inspect_pixels("/Users/mpb/WorkSpace/local_job/assets/blue_1_card_00.png")
