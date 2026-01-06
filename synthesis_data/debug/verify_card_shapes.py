import os
import sys
import numpy as np
import cv2
from collections import defaultdict
import argparse

# Add project root to path so we can import synthesis_data
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from synthesis_data.data_utils import load_all_cards, TYPE_SQUARE, TYPE_RECT_LONG, TYPE_RECT_SHORT, TYPE_TRIANGLE

def calculate_metrics(points):
    """
    Calculate geometric metrics for a polygon.
    Returns: aspect_ratio, solidity, area, angle
    """
    points = np.array(points, dtype=np.float32)
    
    # 1. Rotated Rectangle (Min Area Rect)
    rect = cv2.minAreaRect(points)
    (center), (width, height), angle = rect
    
    # Normalize width/height (width is always the smaller dimension for aspect ratio calc)
    short_side = min(width, height)
    long_side = max(width, height)
    
    aspect_ratio = short_side / long_side if long_side > 0 else 0
    box_area = width * height
    
    # 2. Contour Area
    contour_area = cv2.contourArea(points)
    
    # 3. Solidity
    hull = cv2.convexHull(points)
    hull_area = cv2.contourArea(hull)
    solidity = contour_area / hull_area if hull_area > 0 else 0
    
    # 4. Angle (from data_utils logic or raw minAreaRect)
    # The user asked to check orientation, we can report minAreaRect angle directly for now
    # or use the data_utils logic if we imported Anchor class properly.
    # We will use the raw minAreaRect angle normalized to 0-90 for shape consistency specific checks
    # But for "orientation" checks, we rely on the visual inspection implied by outliers in aspect ratio/solidity first.
    
    return {
        "aspect_ratio": aspect_ratio,
        "solidity": solidity,
        "area": contour_area,
        "rect_width": short_side,
        "rect_height": long_side,
        "min_area_angle": angle
    }

def main():
    parser = argparse.ArgumentParser(description="Verify card shape annotations.")
    parser.add_argument("--assets_dir", type=str, default="assets2", help="Directory containing asset json files")
    args = parser.parse_args()

    assets_path = os.path.join(project_root, args.assets_dir)
    if not os.path.exists(assets_path):
        print(f"Error: Directory {assets_path} not found.")
        return

    print(f"Loading cards from {assets_path}...")
    cards = load_all_cards(assets_path)
    print(f"Loaded {len(cards)} cards.")

    # Collect stats by type
    stats_by_type = defaultdict(list)
    
    for card in cards:
        for anchor in card.anchors:
            metrics = calculate_metrics(anchor.points)
            # Add metadata for reporting
            metrics["file"] = card.card_id
            metrics["label"] = anchor.raw_label
            metrics["canonical_type"] = anchor.canonical_type
            
            stats_by_type[anchor.canonical_type].append(metrics)

    # Analyze each type
    print("\n" + "="*80)
    print("ANALYSIS REPORT")
    print("="*80)

    for c_type, shapes in stats_by_type.items():
        if not shapes:
            continue
            
        print(f"\nType: {c_type} (Count: {len(shapes)})")
        
        # Extract arrays for vector ops
        aspect_ratios = np.array([s["aspect_ratio"] for s in shapes])
        solidities = np.array([s["solidity"] for s in shapes])
        areas = np.array([s["area"] for s in shapes])
        
        ar_mean, ar_std = np.mean(aspect_ratios), np.std(aspect_ratios)
        sol_mean, sol_std = np.mean(solidities), np.std(solidities)
        
        print(f"  Aspect Ratio: Mean={ar_mean:.4f}, Std={ar_std:.4f}")
        print(f"  Solidity:     Mean={sol_mean:.4f}, Std={sol_std:.4f}")
        
        # 1. Check for Aspect Ratio Violations (Square vs Rect separation)
        # SQUARES should have AR close to 1.0
        # RECTANGLES should have AR clearly < 1.0 (depending on the card aspect ratio)
        
        outliers = []
        
        if c_type == TYPE_SQUARE:
            # Squared should be roughly 1:1. Flag if AR < 0.85
            for s in shapes:
                if s["aspect_ratio"] < 0.85:
                    outliers.append(f"[BAD SQUARE AR] {s['file']} ({s['label']}): AR={s['aspect_ratio']:.3f} < 0.85")
                    
        elif c_type in [TYPE_RECT_LONG, TYPE_RECT_SHORT]:
            # Rects should not be squares. Flag if AR > 0.92
             for s in shapes:
                if s["aspect_ratio"] > 0.92:
                    outliers.append(f"[RECT LOOKS LIKE SQUARE] {s['file']} ({s['label']}): AR={s['aspect_ratio']:.3f} > 0.92")

        # 2. General Statistical Outliers (Z-score > 3)
        for s in shapes:
            z_ar = abs(s["aspect_ratio"] - ar_mean) / (ar_std + 1e-6)
            z_sol = abs(s["solidity"] - sol_mean) / (sol_std + 1e-6)
            
            if z_ar > 3.0:
                 outliers.append(f"[STAT OUTLIER AR] {s['file']} ({s['label']}): AR={s['aspect_ratio']:.3f} (Z={z_ar:.1f})")
            if z_sol > 3.0:
                 outliers.append(f"[STAT OUTLIER SOLIDITY] {s['file']} ({s['label']}): Sol={s['solidity']:.3f} (Z={z_sol:.1f})")

        if outliers:
            print("  WARNINGS:")
            # Deduplicate
            for warning in sorted(list(set(outliers))):
                print(f"    - {warning}")
        else:
            print("  > No significant outliers found.")

if __name__ == "__main__":
    main()
