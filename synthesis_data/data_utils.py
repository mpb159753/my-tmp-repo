import json
import os
import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPoint

# Canonical Type Constants
TYPE_TRIANGLE = "TYPE_TRIANGLE"
TYPE_SQUARE = "TYPE_SQUARE"
TYPE_RECT_LONG = "TYPE_RECT_LONG"
TYPE_RECT_SHORT = "TYPE_RECT_SHORT"

# Label mapping to Canonical Type
LABEL_MAP = {
    "score_1_triangle": TYPE_TRIANGLE,
    "anchor_triangle": TYPE_TRIANGLE,
    "score_2_square": TYPE_SQUARE,
    "anchor_square": TYPE_SQUARE,
    "score_3_rect_short": TYPE_RECT_SHORT,
    "anchor_rect_short": TYPE_RECT_SHORT,
    "score_4_rect_long": TYPE_RECT_LONG,
    "anchor_rect_long": TYPE_RECT_LONG
}

def get_canonical_type(label):
    return LABEL_MAP.get(label)

def get_score_value(label):
    if "score_" in label:
        try:
            return int(label.split("_")[1])
        except (ValueError, IndexError):
            return 0
    return 0

class Anchor:
    def __init__(self, raw_label, points, width, height):
        self.raw_label = raw_label
        self.canonical_type = get_canonical_type(raw_label)
        self.score_value = get_score_value(raw_label)
        self.points = np.array(points)
        self.mask = Polygon(points)
        self.width = width
        self.height = height
        
        # Geometry calculations
        self.centroid = np.array(self.mask.centroid.coords[0])
        self.angle = self._calculate_angle()
        
    def _calculate_angle(self):
        """
        Calculate local orientation angle (0-360).
        """
        if self.canonical_type == TYPE_TRIANGLE:
            return self._calculate_triangle_angle()
        else:
            return self._snap_angle(self._calculate_rect_angle(), 90)

    def _snap_angle(self, angle, step):
        """Snap angle to nearest multiple of step degrees."""
        return round(angle / step) * step % 360

    def _calculate_triangle_angle(self):
        """
        Algorithm per user specification:
        1. Simplify polygon to exactly 3 vertices
        2. Find the vertex with the LARGEST interior angle
        3. Direction = midpoint of longest edge → largest-angle vertex
        """
        # Step 1: Simplify to 3 vertices
        tri_vertices = self._simplify_to_triangle()
        if len(tri_vertices) != 3:
            return self._fallback_triangle_angle()
        
        # Step 2: Find angles at each vertex and edge lengths
        angles = []
        edges = []
        n = 3
        for i in range(n):
            p1 = tri_vertices[i - 1]
            p2 = tri_vertices[i]
            p3 = tri_vertices[(i + 1) % n]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Interior angle at p2
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            angles.append((angle, i))
            
            # Edge length from p2 to p3
            edge_len = np.linalg.norm(p3 - p2)
            edges.append((edge_len, i, (i + 1) % n))
        
        # Find vertex with LARGEST angle
        max_angle_vertex_idx = max(angles, key=lambda x: x[0])[1]
        max_angle_vertex = tri_vertices[max_angle_vertex_idx]
        
        # Find the LONGEST edge (opposite to the largest angle vertex)
        # The longest edge is the one that does NOT include the max_angle_vertex
        longest_edge = None
        longest_len = -1
        for edge_len, i, j in edges:
            if max_angle_vertex_idx not in (i, j):
                longest_edge = (i, j)
                longest_len = edge_len
                break
        
        if longest_edge is None:
            # Fallback: just use longest edge overall
            longest_edge = max(edges, key=lambda x: x[0])[1:3]
        
        # Midpoint of longest edge
        midpoint = (tri_vertices[longest_edge[0]] + tri_vertices[longest_edge[1]]) / 2
        
        # Direction: midpoint → largest angle vertex
        vec = max_angle_vertex - midpoint
        angle = np.degrees(np.arctan2(vec[1], vec[0]))
        # Snap to 45° multiples for triangles
        return self._snap_angle(angle, 45)

    def _simplify_to_triangle(self):
        """
        Simplify polygon to exactly 3 vertices using convex hull + 
        approximation.
        """
        from shapely.geometry import MultiPoint
        from scipy.spatial import ConvexHull
        
        points = self.points
        if len(points) <= 3:
            return points
        
        # Get convex hull
        try:
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
        except:
            hull_points = points
        
        if len(hull_points) == 3:
            return hull_points
        
        # If more than 3 points, find the 3 most spread-out points
        # Use the 3 points that form the largest area triangle
        from itertools import combinations
        max_area = -1
        best_tri = hull_points[:3]
        
        for combo in combinations(range(len(hull_points)), 3):
            p1, p2, p3 = hull_points[combo[0]], hull_points[combo[1]], hull_points[combo[2]]
            # Shoelace formula for triangle area
            area = abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2)
            if area > max_area:
                max_area = area
                best_tri = np.array([p1, p2, p3])
        
        return best_tri
    
    def _fallback_triangle_angle(self):
        """Fallback to corner-based heuristic."""
        corners = np.array([
            [0, 0], [self.width, 0],
            [self.width, self.height], [0, self.height]
        ])
        min_dist = float('inf')
        closest_vertex = self.points[0]
        for pt in self.points:
            for corner in corners:
                dist = np.linalg.norm(pt - corner)
                if dist < min_dist:
                    min_dist = dist
                    closest_vertex = pt
        vec = closest_vertex - self.centroid
        return np.degrees(np.arctan2(vec[1], vec[0])) % 360

    def _calculate_rect_angle(self):
        # Use OpenCV minAreaRect
        rect = cv2.minAreaRect(self.points.astype(np.float32))
        angle = rect[2] # -90 to 0
        size = rect[1]
        if size[0] < size[1]: # width < height
            angle += 90
        return angle % 360

class CardData:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.image_path = data.get("imagePath")
        self.width = data.get("imageWidth")
        self.height = data.get("imageHeight")
        self.card_id = os.path.splitext(os.path.basename(json_path))[0]
        
        self.anchors = []
        for shape in data.get("shapes", []):
            label = shape.get("label")
            if label in LABEL_MAP:
                self.anchors.append(Anchor(label, shape.get("points"), self.width, self.height))

def load_all_cards(directory):
    cards = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            cards.append(CardData(os.path.join(directory, filename)))
    return cards
