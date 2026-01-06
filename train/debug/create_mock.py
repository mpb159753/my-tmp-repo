import os
import cv2
import numpy as np
import random

def create_mock_dataset(root_dir, num_images=10):
    images_dir = os.path.join(root_dir, "images")
    labels_dir = os.path.join(root_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for i in range(num_images):
        # Create a random image
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Add some shapes
        num_objs = random.randint(1, 5)
        lines = []
        for _ in range(num_objs):
            cls_id = random.randint(0, 3)
            # YOLO format: cls, cx, cy, w, h (normalized)
            cx = random.random()
            cy = random.random()
            w = random.uniform(0.05, 0.2)
            h = random.uniform(0.05, 0.2)
            
            # constrain within image
            cx = max(w/2, min(1-w/2, cx))
            cy = max(h/2, min(1-h/2, cy))
            
            lines.append(f"{cls_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")
            
            # Draw on image for sanity
            x1 = int((cx - w/2) * 640)
            y1 = int((cy - h/2) * 640)
            x2 = int((cx + w/2) * 640)
            y2 = int((cy + h/2) * 640)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
        filename = f"mock_{i:03d}"
        cv2.imwrite(os.path.join(images_dir, filename + ".jpg"), img)
        with open(os.path.join(labels_dir, filename + ".txt"), "w") as f:
            f.write("\n".join(lines))
            
    print(f"Created {num_images} mock images in {root_dir}")

if __name__ == "__main__":
    create_mock_dataset("/Users/mpb/WorkSpace/local_job/train/debug/mock_dataset")
