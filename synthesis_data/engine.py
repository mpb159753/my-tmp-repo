import numpy as np
import random
import logging
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, translate
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
        self.candidate_index = self._build_index()
        self.placed_cards = []
        self.active_anchors = []
        self.logs = []

    def _build_index(self):
        index = {TYPE_TRIANGLE: [], TYPE_SQUARE: [], TYPE_RECT_LONG: [], TYPE_RECT_SHORT: []}
        for card in self.card_library:
            for i, anchor in enumerate(card.anchors):
                if anchor.canonical_type in index:
                    index[anchor.canonical_type].append((card, i))
        logger.info(f"Built candidate index: TRIANGLE={len(index[TYPE_TRIANGLE])}, SQUARE={len(index[TYPE_SQUARE])}, RECT_LONG={len(index[TYPE_RECT_LONG])}, RECT_SHORT={len(index[TYPE_RECT_SHORT])}")
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
        
        M_final = M_trans @ M_rot_3x3
        
        pc = PlacedCard(first_card, M_final, 0)
        self.placed_cards.append(pc)
        self.active_anchors.extend([(pc, a) for a in pc.anchors])
        self._log(f"[Layer 0] Placed first card: {first_card.card_id}")

        attempts = 0
        while len(self.placed_cards) < min_cards and attempts < 200:
            attempts += 1
            if not self.active_anchors:
                self._log("No more active anchors available")
                break
                
            # Pick a target anchor
            target_idx = random.randrange(len(self.active_anchors))
            target_pc, target_anchor = self.active_anchors[target_idx]
            target_type = target_anchor.original.canonical_type
            
            # Pick a candidate card/anchor of SAME TYPE
            candidates = self.candidate_index.get(target_type, [])
            if not candidates:
                self._log(f"No candidates for type {target_type}")
                continue
                
            candidate_card, cand_anchor_idx = random.choice(candidates)
            candidate_anchor = candidate_card.anchors[cand_anchor_idx]
            
            # Verify type match (should always be true, but log for debugging)
            if candidate_anchor.canonical_type != target_type:
                self._log(f"TYPE MISMATCH ERROR: target={target_type}, candidate={candidate_anchor.canonical_type}")
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
                
                # Step 1: Rotate card around the candidate anchor centroid
                # NOTE: cv2.getRotationMatrix2D uses CCW rotation in screen coords (Y-down),
                # which is effectively CW in math coords. Our angles are in math coords,
                # so we need to NEGATE delta_theta for OpenCV.
                M_rot_cand = cv2.getRotationMatrix2D(
                    (candidate_anchor.centroid[0], candidate_anchor.centroid[1]), 
                    -delta_theta, 1.0  # NEGATE for correct direction
                )
                M_rot_cand_3x3 = np.eye(3)
                M_rot_cand_3x3[:2, :] = M_rot_cand
                
                # Step 2: After rotation, the anchor centroid stays at the same LOCAL position
                # because we rotated around it. Now translate so rotated centroid goes to target.
                # The centroid after rotation is still at candidate_anchor.centroid in local coords,
                # but we need to move the whole rotated card so that point aligns with target.
                tx = target_anchor.centroid_global[0] - candidate_anchor.centroid[0]
                ty = target_anchor.centroid_global[1] - candidate_anchor.centroid[1]
                M_trans_cand = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
                
                M_final_cand = M_trans_cand @ M_rot_cand_3x3
                
                new_pc = PlacedCard(candidate_card, M_final_cand, len(self.placed_cards))
                
                # Check if mostly on canvas
                if not self._is_mostly_on_canvas(new_pc.polygon):
                    continue
                
                # COLLISION DETECTION: Check if new card abnormally occludes existing anchors
                # Rule: New card should ONLY occlude the target anchor, not other active anchors
                collision_ok = self._check_collision(new_pc, target_anchor, target_pc)
                if not collision_ok:
                    self._log(f"Collision detected: {candidate_card.card_id} would occlude non-target anchors")
                    continue
                
                # Commit
                self.placed_cards.append(new_pc)
                target_anchor.is_occupied = True
                self.active_anchors.pop(target_idx)
                
                # Add new card's anchors except the one used for connection
                for i, a in enumerate(new_pc.anchors):
                    if i != cand_anchor_idx:
                        self.active_anchors.append((new_pc, a))
                
                # Detail logging with angles
                new_anchor = new_pc.anchors[cand_anchor_idx]
                self._log(f"[Layer {new_pc.layer}] Placed {candidate_card.card_id}")
                self._log(f"  Target: {target_pc.card_data.card_id}.{target_type} angle_global={target_anchor.angle_global:.0f}°")
                self._log(f"  Candidate: {candidate_card.card_id}.{target_type} local={candidate_anchor.angle:.0f}° -> global={new_anchor.angle_global:.0f}°")
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

import cv2 # Required for getRotationMatrix2D
