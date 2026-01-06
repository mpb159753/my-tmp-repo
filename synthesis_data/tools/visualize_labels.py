import cv2
import os
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_all_cards, TYPE_TRIANGLE, TYPE_SQUARE, TYPE_RECT_LONG, TYPE_RECT_SHORT

def visualize_cards(cards, output_dir, assets_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for card in cards[:10]: # Just process first 10 for quick verification
        img_path = os.path.join(assets_dir, card.image_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image {img_path}")
            continue
            
        for anchor in card.anchors:
            # Draw polygon
            pts = anchor.points.astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            
            # Draw centroid
            cx, cy = int(anchor.centroid[0]), int(anchor.centroid[1])
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            
            # Draw direction vector
            angle_rad = np.radians(anchor.angle)
            vx, vy = int(cx + 40 * np.cos(angle_rad)), int(cy + 40 * np.sin(angle_rad))
            cv2.arrowedLine(img, (cx, cy), (vx, vy), (255, 0, 0), 2, tipLength=0.3)
            
            # Label canonical type and score
            text = f"{anchor.canonical_type.split('_')[-1]} ({anchor.score_value})"
            cv2.putText(img, text, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        output_path = os.path.join(output_dir, f"vis_{card.card_id}.png")
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    assets_dir = "/Users/mpb/WorkSpace/local_job/assets"
    output_dir = "/Users/mpb/WorkSpace/local_job/synthesis_data/debug_vis"
    cards = load_all_cards(assets_dir)
    visualize_cards(cards, output_dir, assets_dir)
