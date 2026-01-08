# Standard library imports
import logging
import random

# Third-party imports
import cv2
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPolygon, Polygon

# Local imports
from data_utils import TYPE_TRIANGLE, TYPE_SQUARE, TYPE_RECT_LONG, TYPE_RECT_SHORT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class PlacedCard:
    def __init__(self, card_data, transform_matrix, layer):
        self.card_data = card_data
        self.transform_matrix = transform_matrix # 3x3
        self.layer = layer
        self.polygon = self._transform_polygon(Polygon([(0,0), (card_data.width, 0), (card_data.width, card_data.height), (0, card_data.height)]))
        self.anchors = []
        for anchor in card_data.anchors:
            ta = TransformedAnchor(anchor, transform_matrix)
            self.anchors.append(ta)

    def _transform_polygon(self, poly):
        coords = np.array(poly.exterior.coords)
        ones = np.ones((coords.shape[0], 1))
        coords_h = np.hstack([coords, ones])
        transformed_coords = coords_h @ self.transform_matrix.T
        return Polygon(transformed_coords[:, :2])

class TransformedAnchor:
    def __init__(self, original_anchor, transform_matrix):
        self.original = original_anchor
        
        # Transform centroid
        pt = np.array([original_anchor.centroid[0], original_anchor.centroid[1], 1.0])
        self.centroid_global = (pt @ transform_matrix.T)[:2]
        
        # Transform direction vector properly
        # Create a point along the direction from centroid
        angle_rad = np.radians(original_anchor.angle)
        dir_local = np.array([
            original_anchor.centroid[0] + 100 * np.cos(angle_rad),
            original_anchor.centroid[1] + 100 * np.sin(angle_rad),
            1.0
        ])
        dir_global = (dir_local @ transform_matrix.T)[:2]
        
        # Calculate new angle from transformed direction
        vec_global = dir_global - self.centroid_global
        self.angle_global = np.degrees(np.arctan2(vec_global[1], vec_global[0])) % 360
        
        # Transform polygon mask
        coords = np.array(original_anchor.points)
        ones = np.ones((coords.shape[0], 1))
        coords_h = np.hstack([coords, ones])
        transformed_coords = coords_h @ transform_matrix.T
        self.mask_global = Polygon(transformed_coords[:, :2])
        self.is_occupied = False

class SynthesisEngine:
    def __init__(self, card_library, canvas_size=(2048, 2048), collision_config=None):
        self.card_library = card_library
        self.canvas_size = canvas_size
        self.collision_config = collision_config or {
            "PARENT_OVERLAP_MAX": 0.30,
            "NON_PARENT_OVERLAP_MAX": 0.005,
            "PRESERVE_PARENT_ANCHORS": True
        }
        self.global_scale = collision_config.get("GLOBAL_SCALE", 1.0) if collision_config else 1.0
        self.candidate_index = self._build_index()
        self.placed_cards = []
        self.active_anchors = []
        self.logs = []

    def _build_index(self):
        # Index: Type -> Color_ID -> List of (Card, AnchorIndex)
        index = {
            TYPE_TRIANGLE: {}, TYPE_SQUARE: {}, 
            TYPE_RECT_LONG: {}, TYPE_RECT_SHORT: {}
        }
        for card in self.card_library:
            for i, anchor in enumerate(card.anchors):
                ctype = anchor.canonical_type
                cid = card.color_id # Use card color
                
                if ctype in index:
                    if cid not in index[ctype]:
                        index[ctype][cid] = []
                    index[ctype][cid].append((card, i))
                    
        # Log stats
        msg = "Built candidate index (Types/Colors): "
        for t, colors in index.items():
            count = sum(len(v) for v in colors.values())
            msg += f"{t}={count} "
        logger.info(msg)
        return index

    def reset(self):
        self.placed_cards = []
        self.active_anchors = []
        self.logs = []

    def _log(self, msg):
        self.logs.append(msg)
        logger.info(msg)

    def generate(self, min_cards=10):
        self.reset()
        
        # 1. Place first card at center with random rotation
        first_card = random.choice(self.card_library)
        angle = random.uniform(0, 360)
        
        M_rot = cv2.getRotationMatrix2D((first_card.width/2, first_card.height/2), angle, 1.0)
        M_rot_3x3 = np.eye(3)
        M_rot_3x3[:2, :] = M_rot
        
        tx = self.canvas_size[0]/2 - first_card.width/2
        ty = self.canvas_size[1]/2 - first_card.height/2
        M_trans = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        
        # Apply Global Scaling
        # Scale -> Rotate -> Translate
        M_scale = np.diag([self.global_scale, self.global_scale, 1.0])
        M_final = M_trans @ M_rot_3x3 @ M_scale
        
        pc = PlacedCard(first_card, M_final, 0)
        self.placed_cards.append(pc)
        self.active_anchors.extend([(pc, a) for a in pc.anchors])
        self._log(f"┌── [Layer 0] Placed Base Card: {first_card.card_id} (ID: {first_card.card_id})")
        self._log(f"│   Pos: ({tx:.1f}, {ty:.1f}), Rot: {angle:.1f}°")

        attempts = 0
        while len(self.placed_cards) < min_cards and attempts < 5000:
            attempts += 1
            if not self.active_anchors:
                self._log("│   [Info] No more active anchors available.")
                break
                
            # Pick a target anchor
            target_idx = random.randrange(len(self.active_anchors))
            target_pc, target_anchor = self.active_anchors[target_idx]
            target_type = target_anchor.original.canonical_type
            
            # Pick a candidate:
            # 1. Look up candidates for this TYPE
            type_candidates = self.candidate_index.get(target_type, {})
            if not type_candidates:
                self._log(f"│   [Warn] No candidates for type {target_type}")
                continue
                
            # 2. Pick a random COLOR (Uniform Distribution)
            available_colors = list(type_candidates.keys())
            if not available_colors:
                continue
                
            chosen_color = random.choice(available_colors)
            color_candidates = type_candidates[chosen_color]
            
            # 3. Pick random card of that color
            candidate_card, cand_anchor_idx = random.choice(color_candidates)
            candidate_anchor = candidate_card.anchors[cand_anchor_idx]
            
            # Verify type match (should always be true, but log for debugging)
            if candidate_anchor.canonical_type != target_type:
                self._log(f"│   [Error] TYPE MISMATCH: target={target_type}, candidate={candidate_anchor.canonical_type}")
                continue
            
            # Symmetry-based rotations
            # For triangles: anchors should point in SAME direction when aligned
            if target_type == TYPE_TRIANGLE:
                possible_rotations = [0]  # Triangles: same direction
            elif target_type == TYPE_SQUARE:
                possible_rotations = [0, 90, 180, 270]
            elif target_type in [TYPE_RECT_LONG, TYPE_RECT_SHORT]:
                possible_rotations = [0, 180]
            else:
                possible_rotations = [0]
                
            success = False
            random.shuffle(possible_rotations)
            
            for sym_rot in possible_rotations:
                delta_theta = (target_anchor.angle_global - (candidate_anchor.angle + sym_rot))
                
                # Correct Matrix Calculation with Global Scale
                # 1. Scale Local Centroid
                c_x_local, c_y_local = candidate_anchor.centroid
                c_x_scaled = c_x_local * self.global_scale
                c_y_scaled = c_y_local * self.global_scale
                
                # 4. Jitter: "Directional Shift" Strategy (User Request)
                # Goal: Maximize exposure while staying under 30% area limit.
                
                # A. Apply Base Rotation (Aligned)
                base_angle = -delta_theta
                
                # B. Translation Jitter
                # Step 1: Align Centers First (Base)
                base_tx = target_anchor.centroid_global[0] - c_x_scaled
                base_ty = target_anchor.centroid_global[1] - c_y_scaled
                
                # Step 2: Choose Random cardinal direction (0, 90, 180, 270) relative to card
                # We move the CARD in this direction.
                # Dimensions are scaled.
                scaled_w = candidate_card.width * self.global_scale
                scaled_h = candidate_card.height * self.global_scale
                
                shift_dir_idx = random.choice([0, 1, 2, 3]) # 0=Right, 1=Down, 2=Left, 3=Up (Local)
                
                # Distance to edge in that direction
                if shift_dir_idx % 2 == 0: # Horizontal (Right/Left)
                    dist_to_edge = scaled_w / 2.0
                else: # Vertical (Down/Up)
                    dist_to_edge = scaled_h / 2.0
                    
                # Step 3: Shift magnitude 0% to 30% of distance to edge
                shift_pct = random.uniform(0.0, 0.30)
                shift_dist = shift_pct * dist_to_edge
                
                # Calculate Local Shift Vector
                # 0 deg = (1,0), 90 deg = (0,1), etc.
                angles_rad = [0, np.pi/2, np.pi, 3*np.pi/2]
                local_angle = angles_rad[shift_dir_idx]
                
                dx_local = shift_dist * np.cos(local_angle)
                dy_local = shift_dist * np.sin(local_angle)
                
                # Rotate Local Shift Vector by the Card's Global Angle (base_angle)
                # to get Global Shift Vector
                # Note: 'base_angle' is the rotation APPLIED to the card.
                # So Local (1,0) becomes Global (cos(theta), sin(theta))
                
                rad_base = np.radians(base_angle)
                cos_b = np.cos(rad_base)
                sin_b = np.sin(rad_base)
                
                dx_global = dx_local * cos_b - dy_local * sin_b
                dy_global = dx_local * sin_b + dy_local * cos_b
                
                tx_jit = base_tx + dx_global
                ty_jit = base_ty + dy_global
                
                # C. Rotation Jitter (Step 3: "Now perform angle rotation")
                # Apply small jitter to the angle
                rotation_jitter = random.uniform(-5.0, 5.0)
                final_angle = base_angle + rotation_jitter
                
                # Re-calculate Rotation Matrix with final angle
                M_rot_cand = cv2.getRotationMatrix2D(
                    (c_x_scaled, c_y_scaled), 
                    final_angle, 1.0
                )
                M_rot_cand_3x3 = np.eye(3)
                M_rot_cand_3x3[:2, :] = M_rot_cand
                
                M_trans_cand = np.array([[1, 0, tx_jit], [0, 1, ty_jit], [0, 0, 1]], dtype=np.float64)
                
                # 5. Global Scale Matrix
                M_scale = np.diag([self.global_scale, self.global_scale, 1.0])
                
                # 6. Combine: Scale -> Rotate -> Translate
                M_final_cand = M_trans_cand @ M_rot_cand_3x3 @ M_scale
                
                new_pc = PlacedCard(candidate_card, M_final_cand, len(self.placed_cards))
                
                # Check if mostly on canvas
                if not self._is_mostly_on_canvas(new_pc.polygon):
                    continue
                
                # COVERAGE CHECK: Ensure new card covers at least 70% of the target anchor
                # (User Constraint: max 30% exposed area)
                target_mask = target_anchor.mask_global
                if not target_mask.is_valid: target_mask = target_mask.buffer(0)
                
                # Area of target anchor
                tm_area = target_mask.area
                if tm_area > 0:
                     # Calculate intersection with new card
                     if new_pc.polygon.intersects(target_mask):
                         covered_area = new_pc.polygon.intersection(target_mask).area
                         exposure_ratio = 1.0 - (covered_area / tm_area)
                         
                         if exposure_ratio > 0.30:
                             # self._log(f"│   -> Rejected: Jitter exposed too much of anchor ({exposure_ratio:.1%})")
                             continue
                     else:
                         # No intersection at all? Should be impossible with correct alignment, but reject
                         continue
                
                # COLLISION DETECTION: Check if new card abnormally occludes existing anchors
                # Rule: New card should ONLY occlude the target anchor, not other active anchors
                collision_ok = self._check_collision(new_pc, target_anchor, target_pc)
                if not collision_ok:
                    # self._log(f"Collision detected: {candidate_card.card_id} would occlude non-target anchors")
                    # Reduce spam for collision failures unless needed
                    continue
                
                # Commit
                self.placed_cards.append(new_pc)
                target_anchor.is_occupied = True
                self.active_anchors.pop(target_idx)
                
                # Add new card's anchors except the one used for connection
                for i, a in enumerate(new_pc.anchors):
                    if i != cand_anchor_idx:
                        self.active_anchors.append((new_pc, a))
                
                # Detail logging
                new_anchor = new_pc.anchors[cand_anchor_idx]
                indent = "│   " * (new_pc.layer + 1)
                self._log(f"├── [Layer {new_pc.layer}] Placed {candidate_card.card_id}")
                self._log(f"│   └── Connects to {target_pc.card_data.card_id} via {target_type} anchor")
                self._log(f"│       Pos: ({tx:.0f}, {ty:.0f}), Angle: {new_anchor.angle_global:.0f}°")
                success = True
                break
            
            if not success:
                # Mark this anchor as failed too many times
                pass
        
        self._log(f"Generation complete: {len(self.placed_cards)} cards placed")
        return self.placed_cards

    def _check_collision(self, new_card, target_anchor, target_pc):
        """
        Check if the new card abnormally occludes existing cards' anchors.
        
        Rules:
        1. New card CAN occlude the target_anchor (that's the connection point)
        2. New card should NOT significantly occlude other ACTIVE anchors
        3. New card should NOT significantly occlude scoring regions of cards below
        4. Stricter overlap tolerance for non-parent cards (0.5%)
        5. Limited overlap with parent card (max 30% area) and NO occlusion of other parent anchors
        """
        new_poly = new_card.polygon
        if not new_poly.is_valid:
            new_poly = new_poly.buffer(0)
        
        # --- Check 1: Interaction with Parent Card (target_pc) ---
        parent_poly = target_pc.polygon
        if not parent_poly.is_valid:
            parent_poly = parent_poly.buffer(0)
            
        if new_poly.intersects(parent_poly):
             intersection = new_poly.intersection(parent_poly)
             overlap_area = intersection.area
             parent_area = parent_poly.area if parent_poly.area > 0 else 1.0
             
             # Sub-rule A: Overlap with parent cannot exceed 30% of NEW card's area (or parent's? usually new card context)
             # User Request: "with parent card overlap cannot be > 30%"
             # Let's check against new card area for logical consistency of "placement validity"
             new_card_area = new_poly.area if new_poly.area > 0 else 1.0
             threshold = self.collision_config.get("PARENT_OVERLAP_MAX", 0.30)
             if overlap_area / new_card_area > threshold:
                 self._log(f"  -> Rejected: Overlap with parent > {threshold:.0%} ({overlap_area/new_card_area:.1%})")
                 return False

             # Sub-rule B: Cannot occlude any other anchors on the parent card (excluding target_anchor)
             for anchor in target_pc.anchors:
                 if anchor is target_anchor:
                     continue # This one IS being connected to
                 
                 anchor_poly = anchor.mask_global
                 if not anchor_poly.is_valid:
                     anchor_poly = anchor_poly.buffer(0)
                 
                 if new_poly.intersects(anchor_poly) and not new_poly.intersection(anchor_poly).is_empty:
                     # Even a tiny overlap is forbidden per "any occlusion"
                     if new_poly.intersection(anchor_poly).area > 1.0: # Tolerance of 1 pixel area
                        self._log(f"  -> Rejected: Occludes non-target anchor {anchor.original.canonical_type} on parent")
                        return False

        # --- Check 2: Interaction with Active Anchors (Global) ---
        for placed_pc, anchor in self.active_anchors:
            if anchor is target_anchor:
                continue  # OK to occlude target
            
            # If it's a sibling anchor on the same parent, we already checked it above, but double check acts as safeguard
            
            anchor_poly = anchor.mask_global
            if not anchor_poly.is_valid:
                anchor_poly = anchor_poly.buffer(0)
            
            if new_poly.intersects(anchor_poly):
                # Calculate overlap ratio
                intersection = new_poly.intersection(anchor_poly)
                overlap_ratio = intersection.area / anchor_poly.area if anchor_poly.area > 0 else 0
                
                # If more than 30% occluded, reject (Standard rule for other card anchors)
                threshold = self.collision_config.get("PARENT_OVERLAP_MAX", 0.30)
                if overlap_ratio > threshold:
                    self._log(f"  -> Rejected: Would occlude {placed_pc.card_data.card_id}'s {anchor.original.canonical_type} by {overlap_ratio:.1%}")
                    return False
        
        # --- Check 3: Overlap with Non-Parent Cards ---
        for placed_pc in self.placed_cards:
            if placed_pc is target_pc:
                continue  # Handled in Check 1
            
            placed_poly = placed_pc.polygon
            if not placed_poly.is_valid:
                placed_poly = placed_poly.buffer(0)
            
            if new_poly.intersects(placed_poly):
                intersection = new_poly.intersection(placed_poly)
                overlap_ratio = intersection.area / placed_poly.area if placed_poly.area > 0 else 0
                
                # Stricter 0.5% threshold for non-parent cards
                threshold = self.collision_config.get("NON_PARENT_OVERLAP_MAX", 0.005)
                if overlap_ratio > threshold: 
                    self._log(f"  -> Rejected: Would overlap {overlap_ratio:.1%} of non-parent card {placed_pc.card_data.card_id}")
                    return False
        
        return True

    def _is_mostly_on_canvas(self, poly):
        canvas_poly = Polygon([(0,0), (self.canvas_size[0], 0), (self.canvas_size[0], self.canvas_size[1]), (0, self.canvas_size[1])])
        if not poly.is_valid:
            poly = poly.buffer(0)
        intersection = poly.intersection(canvas_poly)
        return intersection.area / poly.area > 0.5 if poly.area > 0 else False

    def save_logs(self, filepath):
        with open(filepath, 'w') as f:
            f.write('\n'.join(self.logs))
