import json
import os
import shutil
import cv2
import numpy as np

def via_polygon_to_coco_segmentation(shape_attr):
    points_x = shape_attr['all_points_x']
    points_y = shape_attr['all_points_y']
    segmentation = []
    for x, y in zip(points_x, points_y):
        segmentation.extend([x, y])
    return [segmentation]

def via_rect_to_coco_bbox(shape_attr):
    return [shape_attr['x'], shape_attr['y'], shape_attr['width'], shape_attr['height']]

def polygon_to_bbox(segmentation):
    x_coords = segmentation[0][0::2]
    y_coords = segmentation[0][1::2]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    return [min_x, min_y, max_x - min_x, max_y - min_y]

def main():
    via_json_path = "Card Scoring Zones.json"
    output_dir = "dataset_blue1"
    images_dir = os.path.join(output_dir, "images")
    annotations_path = os.path.join(output_dir, "annotations.json")
    
    # Create directories
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(images_dir)
    
    with open(via_json_path, 'r') as f:
        via_data = json.load(f)
        
    coco_data = {
        "info": {
            "description": "Blue 1 Card Annotations for x-labelanything",
            "year": 2026,
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 1. Collect Categories
    # Hardcode or extract? Safe to extract but order matters for IDs.
    # Let's define a fixed map based on previous plan to ensure consistency
    categories_map = {
        "score_1_triangle": 1,
        "score_2_square": 2,
        "score_3_rect_short": 3,
        "score_4_rect_long": 4,
        "anchor_triangle": 5,
        "anchor_square": 6,
        "anchor_rect_short": 7,
        "anchor_rect_long": 8,
        "unknown": 9,
        # In case user used custom labels in VIA, we might need dynamic collection.
        # But let's start with these. If a label is not found, we add it.
    }
    
    # Filter blue_1 images
    img_metadata = via_data.get('_via_img_metadata', {})
    
    blue1_keys = [k for k in img_metadata.keys() if img_metadata[k]['filename'].startswith("blue_1")]
    
    print(f"Found {len(blue1_keys)} blue_1 images.")
    
    annotation_id = 1
    image_id = 1
    
    # Pre-populate categories list based on map
    # We might update this map if we find new labels
    next_cat_id = max(categories_map.values()) + 1
    
    for key in blue1_keys:
        item = img_metadata[key]
        filename = item['filename']
        regions = item['regions']
        
        # Copy image
        src_image_path = os.path.join("assets/cards", filename)
        dst_image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(src_image_path):
            print(f"Warning: Image {src_image_path} not found, skipping.")
            continue
            
        shutil.copy2(src_image_path, dst_image_path)
        
        # Get image dimensions
        img = cv2.imread(src_image_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        
        image_info = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": filename
        }
        coco_data["images"].append(image_info)
        
        # Process regions
        for region in regions:
            shape_attr = region.get('shape_attributes', {})
            region_attr = region.get('region_attributes', {})
            
            label = region_attr.get('label', 'unknown')
            
            # Update category map if new label found
            if label not in categories_map:
                categories_map[label] = next_cat_id
                next_cat_id += 1
                
            category_id = categories_map[label]
            shape_name = shape_attr.get('name')
            
            segmentation = []
            bbox = []
            
            if shape_name == 'polygon' or shape_name == 'polyline':
                segmentation = via_polygon_to_coco_segmentation(shape_attr)
                bbox = polygon_to_bbox(segmentation)
            elif shape_name == 'rect':
                # Convert rect to bbox
                x, y, w, h = shape_attr['x'], shape_attr['y'], shape_attr['width'], shape_attr['height']
                bbox = [x, y, w, h]
                # Convert rect to polygon segmentation for consistency
                segmentation = [[x, y, x+w, y, x+w, y+h, x, y+h]]
            else:
                print(f"Skipping unsupported shape: {shape_name}")
                continue
                
            area = bbox[2] * bbox[3] # Approximate area
            
            ann = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            coco_data["annotations"].append(ann)
            annotation_id += 1
            
        image_id += 1

    # Finalize categories list
    for label, cid in categories_map.items():
        coco_data["categories"].append({
            "id": cid,
            "name": label,
            "supercategory": "shape"
        })
        
    with open(annotations_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
        
    print(f"Successfully exported {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations to {output_dir}")

if __name__ == "__main__":
    main()