import cv2
import os

def visualize_yolo(img_dir, lbl_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
    
    for img_file in img_files[:5]: # Visualize first 5
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, img_file.replace(".jpg", ".txt"))
        
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                cls_id = parts[0]
                x_c, y_c, bw, bh = map(float, parts[1:])
                
                x1 = int((x_c - bw/2) * w)
                y1 = int((y_c - bh/2) * h)
                x2 = int((x_c + bw/2) * w)
                y2 = int((y_c + bh/2) * h)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, cls_id, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(output_dir, f"check_{img_file}"), img)

if __name__ == "__main__":
    img_dir = "/Users/mpb/WorkSpace/local_job/synthesis_data/dataset/images"
    lbl_dir = "/Users/mpb/WorkSpace/local_job/synthesis_data/dataset/labels"
    output_dir = "/Users/mpb/WorkSpace/local_job/synthesis_data/final_verification"
    visualize_yolo(img_dir, lbl_dir, output_dir)
