import cv2
import os
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def visualize_synthesis_debug(placed_cards, assets_dir, output_path, canvas_size=(2048, 2048)):
    """
    Create a debug visualization with unique IDs for each card and anchor.
    """
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    canvas[:] = [40, 40, 40]
    
    # Color palette for different cards
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), 
        (255, 100, 255), (100, 255, 255), (200, 255, 100), (255, 200, 100),
        (200, 100, 255), (100, 200, 255), (255, 100, 200), (100, 255, 200)
    ]
    
    # First pass: render card images
    for pc in placed_cards:
        img_path = os.path.join(assets_dir, pc.card_data.image_path)
        card_img = cv2.imread(img_path)
        if card_img is None:
            continue
        
        M = pc.transform_matrix[:2, :]
        warped = cv2.warpAffine(card_img, M, canvas_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        
        mask = np.ones((pc.card_data.height, pc.card_data.width), dtype=np.uint8) * 255
        warped_mask = cv2.warpAffine(mask, M, canvas_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        mask_3c = cv2.merge([warped_mask, warped_mask, warped_mask])
        np.copyto(canvas, warped, where=(mask_3c > 0))
    
    # Second pass: draw debug annotations
    for card_idx, pc in enumerate(placed_cards):
        color = colors[card_idx % len(colors)]
        
        # Draw card polygon outline - THICK
        poly_coords = np.array(pc.polygon.exterior.coords, dtype=np.int32)
        cv2.polylines(canvas, [poly_coords], True, color, 4)
        
        # Label card with ID at center - LARGE TEXT
        centroid = pc.polygon.centroid
        cx, cy = int(centroid.x), int(centroid.y)
        card_label = f"C{card_idx}"
        
        # Draw black background for text
        (tw, th), _ = cv2.getTextSize(card_label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
        cv2.rectangle(canvas, (cx - tw//2 - 5, cy - th//2 - 10), (cx + tw//2 + 5, cy + th//2 + 10), (0, 0, 0), -1)
        cv2.putText(canvas, card_label, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        # Draw each anchor with unique ID
        for anchor_idx, anchor in enumerate(pc.anchors):
            anchor_poly = anchor.mask_global
            if anchor_poly.is_empty:
                continue
            
            # Draw anchor polygon
            anchor_coords = np.array(anchor_poly.exterior.coords, dtype=np.int32)
            cv2.polylines(canvas, [anchor_coords], True, (0, 255, 255), 3)
            
            # Anchor centroid
            acx, acy = int(anchor.centroid_global[0]), int(anchor.centroid_global[1])
            cv2.circle(canvas, (acx, acy), 12, (0, 0, 255), -1)
            cv2.circle(canvas, (acx, acy), 14, (255, 255, 255), 2)
            
            # Direction arrow - THICK and BRIGHT
            angle_rad = np.radians(anchor.angle_global)
            arrow_len = 100
            vx = int(acx + arrow_len * np.cos(angle_rad))
            vy = int(acy + arrow_len * np.sin(angle_rad))
            
            # White outline
            cv2.arrowedLine(canvas, (acx, acy), (vx, vy), (255, 255, 255), 8, tipLength=0.3)
            # Yellow fill
            cv2.arrowedLine(canvas, (acx, acy), (vx, vy), (0, 255, 255), 5, tipLength=0.3)
            
            # Anchor ID label - LARGE
            anchor_label = f"{card_idx}.{anchor_idx}"
            anchor_type = anchor.original.canonical_type.split('_')[-1][:3].upper()  # TRI, SQU, REC
            full_label = f"A{anchor_label}:{anchor_type}"
            angle_label = f"{anchor.angle_global:.0f}°"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            label_x = acx + 20
            label_y = acy - 30
            cv2.rectangle(canvas, (label_x - 3, label_y - th - 5), (label_x + tw + 3, label_y + 30), (0, 0, 0), -1)
            cv2.putText(canvas, full_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(canvas, angle_label, (label_x, label_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Add legend at top
    cv2.rectangle(canvas, (0, 0), (canvas_size[0], 60), (0, 0, 0), -1)
    legend = "C=Card, A=Anchor, Arrow=Direction | Format: A[Card].[AnchorIdx]:[Type] [Angle]"
    cv2.putText(canvas, legend, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, canvas)
    return canvas

if __name__ == "__main__":
    from data_utils import load_all_cards
    from engine import SynthesisEngine
    
    assets_dir = "/Users/mpb/WorkSpace/local_job/assets"
    output_dir = "/Users/mpb/WorkSpace/local_job/synthesis_data/debug_synthesis"
    os.makedirs(output_dir, exist_ok=True)
    
    cards = load_all_cards(assets_dir)
    engine = SynthesisEngine(cards)
    
    for i in range(3):
        placed_cards = engine.generate(min_cards=10)
        output_path = os.path.join(output_dir, f"debug_{i:04d}.jpg")
        visualize_synthesis_debug(placed_cards, assets_dir, output_path)
        engine.save_logs(os.path.join(output_dir, f"debug_{i:04d}.log"))
        print(f"Saved {output_path}")
