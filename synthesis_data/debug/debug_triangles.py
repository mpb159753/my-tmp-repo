import cv2
import os
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import load_all_cards, TYPE_TRIANGLE

def visualize_triangle_cards(cards, output_dir, assets_dir):
    """
    Visualize all non-mirror cards that contain triangles.
    Large text, bright colors for visibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter non-mirror cards with triangles
    triangle_cards = []
    for card in cards:
        if 'mirror' in card.card_id.lower():
            continue
        has_triangle = any(a.canonical_type == TYPE_TRIANGLE for a in card.anchors)
        if has_triangle:
            triangle_cards.append(card)
    
    print(f"Found {len(triangle_cards)} non-mirror cards with triangles")
    
    for card in triangle_cards:
        img_path = os.path.join(assets_dir, card.image_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image {img_path}")
            continue
        
        # Make image brighter for visibility
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        
        # Draw each anchor
        for idx, anchor in enumerate(card.anchors):
            # Draw polygon with thick bright line
            pts = anchor.points.astype(np.int32)
            if anchor.canonical_type == TYPE_TRIANGLE:
                color = (0, 255, 0)  # Green for triangles
            else:
                color = (255, 255, 0)  # Cyan for others
            cv2.polylines(img, [pts], True, color, 3)
            
            # Draw centroid - large red circle
            cx, cy = int(anchor.centroid[0]), int(anchor.centroid[1])
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
            cv2.circle(img, (cx, cy), 12, (255, 255, 255), 2)
            
            # Draw direction vector - thick bright arrow
            angle_rad = np.radians(anchor.angle)
            arrow_len = 80
            vx = int(cx + arrow_len * np.cos(angle_rad))
            vy = int(cy + arrow_len * np.sin(angle_rad))
            
            # White outline for visibility
            cv2.arrowedLine(img, (cx, cy), (vx, vy), (255, 255, 255), 6, tipLength=0.3)
            # Yellow fill
            cv2.arrowedLine(img, (cx, cy), (vx, vy), (0, 255, 255), 4, tipLength=0.3)
            
            # Label with LARGE text
            label = f"A{idx}: {anchor.canonical_type.split('_')[-1]}"
            label2 = f"angle={anchor.angle:.1f}"
            
            # Background rectangle for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(img, (cx + 15, cy - 35), (cx + 20 + tw, cy + 5), (0, 0, 0), -1)
            cv2.putText(img, label, (cx + 20, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(img, label2, (cx + 20, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Card ID at top
        cv2.rectangle(img, (0, 0), (500, 50), (0, 0, 0), -1)
        cv2.putText(img, card.card_id, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        output_path = os.path.join(output_dir, f"{card.card_id}.jpg")
        cv2.imwrite(output_path, img)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    assets_dir = "/Users/mpb/WorkSpace/local_job/assets"
    output_dir = "/Users/mpb/WorkSpace/local_job/synthesis_data/triangle_debug"
    cards = load_all_cards(assets_dir)
    visualize_triangle_cards(cards, output_dir, assets_dir)
