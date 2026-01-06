import os
import cv2
import random
import yaml
import numpy as np

def draw_yolo_labels(image_path, label_path, classes, output_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    if not os.path.exists(label_path):
        print(f"Label not found: {label_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    h, w, c = img.shape
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    print(f"Visualizing {len(lines)} labels for {os.path.basename(image_path)}")
    
    # Color map for 4 classes
    colors = [
        (0, 0, 255),    # 0: Red
        (0, 255, 0),    # 1: Green
        (255, 0, 0),    # 2: Blue
        (0, 255, 255)   # 3: Yellow
    ]
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
            
        cls_id = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        cw = float(parts[3])
        ch = float(parts[4])
        
        # Denormalize
        x_center = cx * w
        y_center = cy * h
        width = cw * w
        height = ch * h
        
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        color = colors[cls_id % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{classes.get(cls_id, str(cls_id))}"
        cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(output_path, img)
    print(f"Saved visualization to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize YOLO labels")
    parser.add_argument("--data", type=str, default="/Users/mpb/WorkSpace/local_job/train/tacta.yaml", help="Path to data yaml")
    parser.add_argument("--images", type=str, default="/Users/mpb/WorkSpace/local_job/synthesis_data/dataset/images", help="Path to images dir")
    parser.add_argument("--labels", type=str, default="/Users/mpb/WorkSpace/local_job/synthesis_data/dataset/labels", help="Path to labels dir")
    args = parser.parse_args()

    # Load config
    with open(args.data, 'r') as f:
        data = yaml.safe_load(f)
        
    names = data.get('names', {})
    
    images_dir = args.images
    labels_dir = args.labels
    
    if not os.path.exists(images_dir):
        print(f"Images dir not found: {images_dir}")
        return

    all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    if not all_images:
        print("No images found in dataset directory.")
        return

    # Pick random 3 images
    samples = random.sample(all_images, min(3, len(all_images)))
    
    out_dir = "/Users/mpb/WorkSpace/local_job/train/debug"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for i, img_name in enumerate(samples):
        img_path = os.path.join(images_dir, img_name)
        
        # Infer label path
        # Assuming synthesis follows same basename
        basename = os.path.splitext(img_name)[0]
        label_name = basename + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        
        out_path = os.path.join(out_dir, f"vis_check_{i}.jpg")
        
        draw_yolo_labels(img_path, label_path, names, out_path)

if __name__ == "__main__":
    main()
