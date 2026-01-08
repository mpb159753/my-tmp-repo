import cv2
import numpy as np
import os
from shapely.geometry import Polygon
from data_utils import get_score_value

class Renderer:
    def __init__(self, canvas_size=(1024, 1024), visibility_threshold=0.4):
        self.canvas_size = canvas_size
        self.visibility_threshold = visibility_threshold
        self.class_map = {
            1: 0, 2: 1, 3: 2, 4: 3, 
            10: 0, 20: 1, 30: 2, 40: 3, 50: 4  # Keeping these just in case, but 1-4 are primary
        }
        
        # Load background textures
        self.bg_textures = []
        # Try to find backgrounds relative to assets dir or project root
        # Default assumption: assets/backgrounds is usually where it is.
        # But we need to be flexible.
        possible_bg_dirs = [
            "/Users/mpb/WorkSpace/local_job/assets/backgrounds", # Keep legacy for safety locally
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "backgrounds"), # ../assets/backgrounds from synthesis_data/
            os.path.join(os.getcwd(), "assets", "backgrounds"),
            "./assets/backgrounds"
        ]
        
        bg_dir = None
        for d in possible_bg_dirs:
            if os.path.exists(d) and os.path.isdir(d):
                bg_dir = d
                break
        
        if bg_dir:
            for fname in os.listdir(bg_dir):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img = cv2.imread(os.path.join(bg_dir, fname))
                    if img is not None:
                        # Resize to canvas size once
                        img = cv2.resize(img, self.canvas_size)
                        self.bg_textures.append(img)
                        
        self.lighting_dist = None
        self.perspective_dist = None

    def set_distributions(self, lighting_dist, perspective_dist):
        self.lighting_dist = lighting_dist
        self.perspective_dist = perspective_dist
                        
    def render(self, placed_cards, assets_dir, output_img_path, lighting_level=None):
        """
        lighting_level: 0 (Minimal), 1 (Weak), 2 (Medium), 3 (Strong)
        If None, picked randomly with weighted probability.
        """
        # Determine Lighting Level if not provided
        if lighting_level is None:
            if self.lighting_dist:
                # Use configured distribution
                levels = list(self.lighting_dist.keys())
                probs = list(self.lighting_dist.values())
                lighting_level = np.random.choice(levels, p=probs)
            else:
                # Default fallback (Hardcoded legacy)
                r = np.random.random()
                if r < 0.6: lighting_level = 0
                elif r < 0.8: lighting_level = 1
                elif r < 0.9: lighting_level = 2
                else: lighting_level = 3
            
        # Create canvas
        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        
        # Background Selection
        # 80% chance to use texture if available, 20% solid color
        use_texture = len(self.bg_textures) > 0 and np.random.random() < 0.8
        
        if use_texture:
            idx = np.random.randint(0, len(self.bg_textures))
            bg_img = self.bg_textures[idx]
            canvas = bg_img.copy()
        else:
            # Simple procedural background (dark gray with some noise)
            canvas[:] = [40, 40, 40]
            noise = np.random.randint(0, 20, canvas.shape, dtype=np.uint8)
            canvas = cv2.add(canvas, noise)

        # Global occlusion map (cumulative union of card polygons)
        # We render bottom to top.
        
        # Step 1: Render images with variable shadows
        for pc in placed_cards:
            img_path = os.path.join(assets_dir, pc.card_data.image_path)
            card_img = cv2.imread(img_path)
            if card_img is None: continue
            
            # Warp card image
            M = pc.transform_matrix[:2, :]
            warped = cv2.warpAffine(card_img, M, self.canvas_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            
            # Warp mask
            mask = np.ones((pc.card_data.height, pc.card_data.width), dtype=np.uint8) * 255
            warped_mask = cv2.warpAffine(mask, M, self.canvas_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            # --- Shadow Rendering Logic ---
            if lighting_level > 0:
                # Base Params
                if lighting_level == 1: # Weak
                    offset_min, offset_max = 5, 10
                    opacity_min, opacity_max = 0.2, 0.3
                    blur_val = 11
                elif lighting_level == 2: # Medium
                    offset_min, offset_max = 10, 20
                    opacity_min, opacity_max = 0.4, 0.5
                    blur_val = 15
                else: # Strong (3)
                    offset_min, offset_max = 20, 35
                    opacity_min, opacity_max = 0.6, 0.8
                    blur_val = 21

                shadow_offset = np.random.randint(offset_min, offset_max + 1)
                shadow_blur = blur_val
                shadow_opacity = np.random.uniform(opacity_min, opacity_max)
                
                # Shift mask for shadow
                M_shadow = np.float32([[1, 0, shadow_offset], [0, 1, shadow_offset]]) # Diagonal shift
                shadow_layer = cv2.warpAffine(warped_mask, M_shadow, self.canvas_size)
                
                # Blur shadow
                shadow_layer = cv2.GaussianBlur(shadow_layer, (shadow_blur, shadow_blur), 0)
                
                # Composite shadow: darken canvas where shadow is
                shadow_norm = shadow_layer.astype(np.float32) / 255.0
                shadow_factor = 1.0 - (shadow_opacity * shadow_norm)
                shadow_factor = np.stack([shadow_factor]*3, axis=2) # 3 channels
                
                # Apply shadow to existing canvas
                canvas = (canvas.astype(np.float32) * shadow_factor).astype(np.uint8)
            # --- End Shadow ---

            # Paste card using mask
            mask_3c = cv2.merge([warped_mask, warped_mask, warped_mask])
            np.copyto(canvas, warped, where=(mask_3c > 0))

        # cv2.imwrite(output_img_path, canvas) # Moved to after effects in main
        return canvas, lighting_level

    def augment_lighting(self, canvas, lighting_level=0):
        # Scale effects based on level
        if lighting_level == 0:
            # Minimal/None: Just tiny noise maybe, or clean
            return canvas
            
        rows, cols = canvas.shape[:2]
        
        if lighting_level == 1: # Weak
            vignette_power = 0.5
            spot_brightness = 10
            noise_sigma = 5
        elif lighting_level == 2: # Mid
            vignette_power = 1.0
            spot_brightness = 30
            noise_sigma = 15
        else: # Strong
            vignette_power = 1.5
            spot_brightness = 50
            noise_sigma = 30
            
        # Add gradient vignette
        # Random center
        center_x = np.random.randint(cols // 4, 3 * cols // 4)
        center_y = np.random.randint(rows // 4, 3 * rows // 4)
        
        # Max radius
        radius = max(rows, cols) * np.random.uniform(0.8, 1.2)
        
        Y, X = np.ogrid[:rows, :cols]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Radial gradient: 1 at center, 0 at radius
        # We want center to be bright, edges dark
        mask = 1 - (dist_from_center / radius)
        mask = np.clip(mask, 0, 1)
        
        # Variable vignette strength
        mask = mask ** vignette_power
        
        mask_3c = np.stack([mask]*3, axis=2)
        
        # Apply vignetting
        canvas = (canvas * mask_3c).astype(np.uint8)
        
        # Spotlight
        if spot_brightness > 0:
            spotlight = (spot_brightness * mask_3c).astype(np.uint8)
            canvas = cv2.add(canvas, spotlight)

        # Add Gaussian Noise
        if noise_sigma > 0:
            noise = np.random.normal(0, noise_sigma, canvas.shape).astype(np.int16)
            canvas_int = canvas.astype(np.int16) + noise
            canvas = np.clip(canvas_int, 0, 255).astype(np.uint8)
        
        # Note: Color cast is now applied separately in main.py pipeline
        return canvas

    def apply_color_cast(self, canvas, mode="NEUTRAL"):
        """
        Simulate white balance errors / color casts.
        Modes: NEUTRAL, WARM, COOL, FLUORESCENT, MAGENTA
        """
        if mode == "NEUTRAL":
            # Neutral - no cast
            return canvas
        elif mode == "WARM":
            # Warm: Boost R slightly, cut B
            r_gain = np.random.uniform(1.05, 1.15)
            g_gain = np.random.uniform(1.00, 1.05)
            b_gain = np.random.uniform(0.85, 0.95)
        elif mode == "COOL":
            # Cool: Boost B, cut R
            r_gain = np.random.uniform(0.85, 0.95)
            g_gain = np.random.uniform(0.95, 1.00)
            b_gain = np.random.uniform(1.05, 1.15)
        elif mode == "FLUORESCENT":
            # Fluorescent: Green tint
            r_gain = np.random.uniform(0.90, 1.00)
            g_gain = np.random.uniform(1.05, 1.12)
            b_gain = np.random.uniform(0.90, 1.00)
        elif mode == "MAGENTA":
            # Magenta: Cut G
            r_gain = np.random.uniform(1.00, 1.08)
            g_gain = np.random.uniform(0.88, 0.95)
            b_gain = np.random.uniform(1.00, 1.08)
        else:
            # Unknown mode, return unchanged
            return canvas
        
        # Apply gains (OpenCV uses BGR order)
        canvas_float = canvas.astype(np.float32)
        canvas_float[:, :, 0] *= b_gain  # B
        canvas_float[:, :, 1] *= g_gain  # G
        canvas_float[:, :, 2] *= r_gain  # R
        
        canvas = np.clip(canvas_float, 0, 255).astype(np.uint8)
        return canvas

    def apply_perspective(self, canvas, placed_cards):
        """
        Apply variable random perspective distortion.
        New Strategy: "Crop to Fill"
        - We define a Trapezoid inside the source image (simulating the camera viewport on the table).
        - We map this Trapezoid to the Full Destination Canvas.
        - Because the Source is strictly inside the texture, there are NO black borders.
        - To simulate perspective (far objects smaller), the Source Trapezoid Top Width must be WIDER than Bottom Width.
          (Mapping a wide area to fixed width = Compression = Smaller appearance)
        """
        h, w = canvas.shape[:2]
        
        # Decide mode
        mode = "NONE"
        if self.perspective_dist:
            modes = list(self.perspective_dist.keys())
            probs = list(self.perspective_dist.values())
            mode = np.random.choice(modes, p=probs)
        else:
            # Legacy fallback
            r = np.random.random()
            if r < 0.5: mode = "NONE"
            elif r < 0.8: mode = "NORMAL"
            else: mode = "HEAVY"
        
        if mode == "NONE":
            valid_poly = Polygon([(0,0), (w,0), (w,h), (0,h)])
            return canvas, valid_poly
            
        elif mode == "NORMAL":
            # 10-20% narrowing at bottom
            narrow_factor_min = 0.10
            narrow_factor_max = 0.20
        else: # HEAVY
            # 25-40% narrowing at bottom
            narrow_factor_min = 0.25
            narrow_factor_max = 0.40
            
        # Define Source Trapezoid (The crop)
        # Top Edge: Nearly full width (Keep "Far" objects normal/small)
        # Random minimal crop at top to vary framing
        top_margin = int(h * np.random.uniform(0.0, 0.05))
        top_x_shift = int(w * np.random.uniform(0.0, 0.05))
        
        p1 = [top_x_shift, top_margin] # TL
        p2 = [w - top_x_shift, top_margin] # TR
        
        # Bottom Edge: Narrower (The "Near" objects magnified)
        # We narrow the bottom source selection.
        narrow_pct = np.random.uniform(narrow_factor_min, narrow_factor_max)
        bottom_margin = int(h * np.random.uniform(0.0, 0.05))
        
        x_inset = int(w * narrow_pct)
        p3 = [w - x_inset, h - bottom_margin] # BR
        p4 = [x_inset, h - bottom_margin] # BL
        
        src_pts = np.float32([p1, p2, p3, p4])
        
        # Destination: Full Canvas
        dst_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Compute Homography
        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp Image
        # Using INTER_LINEAR for speed/quality balance. 
        # No border artifacts because we are zooming in!
        warped_canvas = cv2.warpPerspective(canvas, H, (w, h), flags=cv2.INTER_LINEAR)
        
        # Update Geometry (In-Place)
        for pc in placed_cards:
            pc.polygon = self._transform_shapely_poly(pc.polygon, H)
            for anchor in pc.anchors:
                anchor.mask_global = self._transform_shapely_poly(anchor.mask_global, H)
                if not anchor.mask_global.is_empty:
                     c = anchor.mask_global.centroid
                     anchor.centroid_global = np.array([c.x, c.y])
                     
        # Valid Polygon is technically the Full Canvas now, 
        # but effectively the "Visible Area" of the original world is the inverse of src_pts.
        # But for labeling purposes, the valid area IS the full canvas (0,0)-(w,h).
        valid_poly = Polygon([(0,0), (w,0), (w,h), (0,h)])
        
        return warped_canvas, valid_poly

    def _transform_shapely_poly(self, poly, H):
        if poly.is_empty: return poly
        
        coords = np.array(poly.exterior.coords)
        if len(coords) == 0: return poly
        
        # Perspective transform of points:
        # [x', y', w'] = H * [x, y, 1]
        # x_out = x'/w', y_out = y'/w'
        
        ones = np.ones((coords.shape[0], 1))
        coords_h = np.hstack([coords, ones]) # Nx3
        
        # (3x3) @ (3xN) -> (3xN)
        res = H @ coords_h.T
        res = res.T # Nx3
        
        # Divide by w (3rd column)
        w_vec = res[:, 2:3]
        # Avoid div by zero just in case
        w_vec[w_vec == 0] = 1.0
        
        uv = res[:, :2] / w_vec
        
        return Polygon(uv)

    def save_yolo_labels(self, placed_cards, output_txt_path, valid_poly=None):
        # Legacy Wrapper
        return self.save_yolo_segmentation_labels(placed_cards, output_txt_path, valid_poly)

    def save_yolo_segmentation_labels(self, placed_cards, output_txt_path, valid_poly=None):
        labels = []
        
        canvas_poly = Polygon([(0,0), (self.canvas_size[0], 0), (self.canvas_size[0], self.canvas_size[1]), (0, self.canvas_size[1])])
        if valid_poly is not None:
             canvas_poly = canvas_poly.intersection(valid_poly)
        
        for i, pc in enumerate(placed_cards):
            for anchor in pc.anchors:
                # 1. Skip if anchor is used for connection (logically hidden/occupied)
                if hasattr(anchor, 'is_occupied') and anchor.is_occupied:
                    continue

                if anchor.original.score_value == 0:
                    continue
                
                poly = anchor.mask_global
                if not poly.is_valid: poly = poly.buffer(0)
                
                original_area = poly.area
                if original_area <= 0: continue
                
                # Visible part (intersection with canvas)
                visible_poly = poly.intersection(canvas_poly)
                if visible_poly.is_empty: continue
                
                # EDGE CLIPPING CHECK:
                # If significant portion is lost just by canvas intersection (e.g. < 98% inside),
                # it means the pattern is cut by the edge. Skip it.
                if visible_poly.area / original_area < 0.98:
                    continue
                    
                # Subtract upper layers
                for j in range(i + 1, len(placed_cards)):
                    upper_card_poly = placed_cards[j].polygon
                    if not upper_card_poly.is_valid: upper_card_poly = upper_card_poly.buffer(0)
                    
                    if visible_poly.intersects(upper_card_poly):
                        visible_poly = visible_poly.difference(upper_card_poly)
                        if visible_poly.is_empty: break
                
                if visible_poly.is_empty: continue

                # Ratio Check
                ratio = visible_poly.area / original_area
                if ratio > 0.5:
                    # Convert to Multipolygon handling if needed, usually it's a single polygon
                    if visible_poly.geom_type == 'MultiPolygon':
                        # Take largest part
                        visible_poly = max(visible_poly.geoms, key=lambda a: a.area)
                        
                    # Normalize coords
                    coords = np.array(visible_poly.exterior.coords)
                    # Normalize x by width, y by height
                    coords[:, 0] /= self.canvas_size[0]
                    coords[:, 1] /= self.canvas_size[1]
                    
                    # Clip to 0-1
                    coords = np.clip(coords, 0, 1)
                    
                    # Flatten
                    flat_coords = coords.flatten()
                    
                    color_id = anchor.original.color_id if hasattr(anchor.original, 'color_id') else 0
                    score_val = anchor.original.score_value
                    
                    # Safety clamp
                    if score_val < 1: score_val = 1
                    if score_val > 4: score_val = 4
                    
                    class_id = color_id * 4 + (score_val - 1)
                    
                    # Format: class x1 y1 x2 y2 ...
                    coords_str = " ".join([f"{c:.6f}" for c in flat_coords])
                    labels.append(f"{class_id} {coords_str}")

        with open(output_txt_path, 'w') as f:
            f.write("\n".join(labels))
        return labels

    def render_debug(self, placed_cards, assets_dir, output_img_path, labels_path=None):
        """
        Render debug image with YOLO labels overlaid.
        """
        # NOTE: render_debug needs to use the FINAL transformed image, not re-render from scratch.
        # But our current flow separates rendering and logic. 
        # Ideally, we should pass the final canvas to this.
        # For now, to support the separate calls in main.py, we might have an issue
        # because `render()` creates a FRESH canvas without perspective.
        
        # Hack solution: If perspective was applied, `placed_cards` geometry is already distorted.
        # But `render()` will draw them flat. This will cause mismatch in debug view.
        #
        # CORRECT FIX: Main loop should pass the final image to render_debug.
        # I'll update render_debug to accept an optional 'canvas' argument.
        pass

    def render_debug_v2(self, canvas, labels_path, output_img_path):
        """
        Draw segmentation polygons on an EXISTING canvas.
        """
        debug_canvas = canvas.copy()
        
        # Parse labels
        labels = []
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        class_id = int(parts[0])
                        # Remaining are x y x y ...
                        coords = [float(x) for x in parts[1:]]
                        labels.append((class_id, coords))

        h_img, w_img = debug_canvas.shape[:2]
        
        # Draw Polygons
        for class_id, coords in labels:
            # Rescale coords
            pts = []
            for i in range(0, len(coords), 2):
                x = int(coords[i] * w_img)
                y = int(coords[i+1] * h_img)
                pts.append([x, y])
            
            pts = np.array(pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            color = (0, 255, 0) 
            if class_id == 0: color = (100, 100, 255) 
            elif class_id == 1: color = (255, 100, 100)
            elif class_id == 2: color = (255, 255, 100)
            elif class_id == 3: color = (100, 255, 255)

            # Draw polygon
            cv2.polylines(debug_canvas, [pts], True, color, 2)
            
            # Label
            if len(pts) > 0:
                label_text = f"ID: {class_id}"
                x1, y1 = pts[0][0]
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(debug_canvas, (x1, y1 - 20), (x1 + tw + 10, y1), color, -1)
                cv2.putText(debug_canvas, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(output_img_path, debug_canvas)
        return debug_canvas


    def render_debug(self, placed_cards, assets_dir, output_img_path, labels_path=None):
        """
        Render debug image with YOLO labels overlaid.
        """
        # First render the base image as usual
        canvas = self.render(placed_cards, assets_dir, output_img_path)
        
        # Parse labels
        labels = []
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        xc = float(parts[1])
                        yc = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        labels.append((class_id, xc, yc, w, h))
        else:
             # Or generate them on the fly if not passed? 
             # For now assume they are passed or re-generated.
             # Actually simplest is to regenerate or use what we derived.
             temp_labels = self.save_yolo_labels(placed_cards, "/dev/null") # Quick re-calc
             for l in temp_labels:
                 parts = l.split()
                 labels.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

        h_img, w_img = canvas.shape[:2]
        
        # Draw boxes
        for class_id, xc, yc, w, h in labels:
            x1 = int((xc - w/2) * w_img)
            y1 = int((yc - h/2) * h_img)
            x2 = int((xc + w/2) * w_img)
            y2 = int((yc + h/2) * h_img)
            
            color = (0, 255, 0) # Green default
            if class_id == 0: color = (100, 100, 255) # Red-ish
            elif class_id == 1: color = (255, 100, 100) # Blue-ish
            elif class_id == 2: color = (255, 255, 100) # Cyan-ish
            elif class_id == 3: color = (100, 255, 255) # Yellow-ish

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label_text = f"ID: {class_id}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, y1 - 20), (x1 + tw + 10, y1), color, -1)
            cv2.putText(canvas, label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imwrite(output_img_path, canvas)
        return canvas
