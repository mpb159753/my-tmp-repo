import os
import random
import cv2
import logging
from data_utils import load_all_cards
from engine import SynthesisEngine
from renderer import Renderer

# Default Configuration
CONFIG = {
    "OUTPUT_DIR": "./synthesis_data/dataset",
    "TOTAL_IMAGES": 5000,
    "MIN_CARDS": 40,
    "MAX_CARDS": 80,
    # Simulate zooming out by shrinking cards (0.4x)
    "GLOBAL_SCALE": 0.4, 
    "COLLISION": {
        "PARENT_OVERLAP_MAX": 0.30,
        "NON_PARENT_OVERLAP_MAX": 0.005,
        "PRESERVE_PARENT_ANCHORS": True
    },
    "VISIBILITY_THRESHOLD": 0.5, # 严格阈值
    
    # 概率分布配置 (Probabilities)
    "LIGHTING_DIST": {
        0: 0.30, # Minimal/Clean
        1: 0.30, # Weak
        2: 0.25, # Medium
        3: 0.15  # Strong
    },
    "PERSPECTIVE_DIST": {
        "NONE": 0.20,
        "NORMAL": 0.50, # 10-15%
        "HEAVY": 0.30   # 20-35%
    },
    
    # 调试选项
    "DEBUG": {
        "SAVE_IMAGES": True, # 输出 debug 图片
        "SAVE_LOGS": True    # 输出生成日志
    },
    
    "CANDIDATE_PATH": None, # 默认使用 global loading，可指定特定目录
    
    # YOLO 类别映射
    # YOLO 类别映射 (Color-Aware 24 Class Taxonomy)
    # This is mainly for documentation here, the actual ID generation happens in renderer.py
    # But we need this map to generate the dataset.yaml file later.
    "CLASS_MAP": {
        # Format: Color_Score (0-23)
        # Red (0-3)
        "Red_1": 0, "Red_2": 1, "Red_3": 2, "Red_4": 3,
        # Blue (4-7)
        "Blue_1": 4, "Blue_2": 5, "Blue_3": 6, "Blue_4": 7,
        # Purple (8-11)
        "Purple_1": 8, "Purple_2": 9, "Purple_3": 10, "Purple_4": 11,
        # Green (12-15)
        "Green_1": 12, "Green_2": 13, "Green_3": 14, "Green_4": 15,
        # Yellow (16-19)
        "Yellow_1": 16, "Yellow_2": 17, "Yellow_3": 18, "Yellow_4": 19,
        # Pink (20-23)
        "Pink_1": 20, "Pink_2": 21, "Pink_3": 22, "Pink_4": 23
    }
}

import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import time
import logging
from engine import SynthesisEngine # Ensure updated engine is imported

# Global context for worker processes
worker_ctx = {}

def init_worker(config, cards, assets_dir, out_dirs):
    """
    Initialize worker process with its own Engine and Renderer instances.
    """
    # Re-seed random number generators to avoid identical sequences in forks
    seed = (os.getpid() * int(time.time() * 1000)) % 123456789
    random.seed(seed)
    np.random.seed(seed)
    
    # Silence detailed logs from engine in console (keep only errors/warnings if any)
    # The engine writes to self.logs list for file saving, independent of logger.info if we tweak it.
    # However, engine uses logging.basicConfig. We should adjust the logger for "synthesis_data.engine".
    logging.getLogger("synthesis_data.engine").setLevel(logging.WARNING)
    logging.getLogger("engine").setLevel(logging.WARNING)
    
    worker_ctx['config'] = config
    worker_ctx['assets_dir'] = assets_dir
    worker_ctx['dirs'] = out_dirs
    
    # Initialize Engine and Renderer locally for this process
    worker_ctx['engine'] = SynthesisEngine(
        cards, 
        canvas_size=(2048, 2048), 
        collision_config={**config["COLLISION"], "GLOBAL_SCALE": config.get("GLOBAL_SCALE", 1.0)}
    )
    
    worker_ctx['renderer'] = Renderer(canvas_size=(2048, 2048))
    worker_ctx['renderer'].set_distributions(
        lighting_dist=config["LIGHTING_DIST"],
        perspective_dist=config["PERSPECTIVE_DIST"]
    )

def process_one_image(i):
    """
    Generate a single image.
    """
    cfg = worker_ctx['config']
    engine = worker_ctx['engine']
    renderer = worker_ctx['renderer']
    assets_dir = worker_ctx['assets_dir']
    img_dir, lbl_dir, log_dir, debug_dir = worker_ctx['dirs']
    
    min_c = cfg["MIN_CARDS"]
    max_c = cfg["MAX_CARDS"]
    
    try:
        # Generate stacking
        # Retry logic for strict density enforcement
        placed_cards = []
        gen_attempts = 0
        valid_gen = False
        
        while gen_attempts < 10 and not valid_gen:
             # Randomly select range based on weights:
             # 20%: 10-30, 35%: 35-65, 45%: 65-99
             r = random.random()
             if r < 0.20:
                 n_cards = random.randint(10, 30)
             elif r < 0.55: # 0.20 + 0.35
                 n_cards = random.randint(35, 65)
             else:
                 n_cards = random.randint(65, 99)
                 
             # engine.generate usually respects min_cards if attempts limit allows
             placed_cards = engine.generate(min_cards=n_cards)
             
             if len(placed_cards) >= min_c:
                 valid_gen = True
             else:
                 gen_attempts += 1
        
        if not valid_gen:
             print(f"[Warning] Image {i}: Could not reach min cards ({min_c}). Generated {len(placed_cards)}.")
        
        # Save name
        base_name = f"synth_{i:04d}"
        img_path = os.path.join(img_dir, f"{base_name}.jpg")
        txt_path = os.path.join(lbl_dir, f"{base_name}.txt")
        
        # Render and save
        # 1. Render base with shadows
        canvas, lighting_level = renderer.render(placed_cards, assets_dir, img_path)
        
        # 2. Lighting Augmentation
        canvas = renderer.augment_lighting(canvas, lighting_level=lighting_level)
        
        # 3. Perspective
        canvas, valid_poly = renderer.apply_perspective(canvas, placed_cards)
        
        # Save final image
        cv2.imwrite(img_path, canvas)
        
        # 4. Save labels
        renderer.save_yolo_labels(placed_cards, txt_path, valid_poly=valid_poly)
        
        # Save Debug Data (Only 10% to save space, or configured)
        # Using a deterministic check based on index to ensure consistent debug output
        debug_interval = cfg["DEBUG"].get("INTERVAL", 10)
        should_debug = (i % debug_interval == 0)

        # Logs
        if cfg["DEBUG"]["SAVE_LOGS"] and should_debug:
            log_path = os.path.join(log_dir, f"{base_name}.log")
            engine.save_logs(log_path)
        
        # Debug Images
        if cfg["DEBUG"]["SAVE_IMAGES"] and should_debug:
            debug_path = os.path.join(debug_dir, f"{base_name}_debug.jpg")
            renderer.render_debug_v2(canvas, txt_path, debug_path)
            
        return True
    except Exception as e:
        print(f"Error generating image {i}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main(user_config=None):
    # Merge user config with default
    config = CONFIG.copy()
    if user_config:
        # Shallow update for top-level
        for k, v in user_config.items():
            if k in ["COLLISION", "LIGHTING_DIST", "PERSPECTIVE_DIST", "DEBUG", "CLASS_MAP"] and isinstance(v, dict):
                 # Deep merge for dictionary fields
                 config[k] = config.get(k, {}).copy()
                 config[k].update(v)
            else:
                 config[k] = v

    # Auto-adjust DEBUG settings for small batches (Verification Mode)
    num_images = config["TOTAL_IMAGES"]
    if num_images <= 200:
        # If user didn't explicitly set interval, default to 1 (all images)
        if "INTERVAL" not in config["DEBUG"]:
            config["DEBUG"]["INTERVAL"] = 1
        print(f"[Info] Small batch detected ({num_images} images). Setting Debug Interval to {config['DEBUG']['INTERVAL']}.")

    if not config.get("CANDIDATE_PATH"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Default: ../assets/cards (standardized structure)
        candidate_path = os.path.join(base_dir, "..", "assets", "cards")
        if os.path.exists(candidate_path):
             config["CANDIDATE_PATH"] = candidate_path
        else:
             # Fallback: maybe we are in export/root?
             # If running from export/synthesis_data, ../assets/cards should work.
             # If legacy path exists?
             legacy_path = os.path.join(base_dir, "..", "assets2")
             if os.path.exists(legacy_path):
                 config["CANDIDATE_PATH"] = legacy_path 
             else:
                 config["CANDIDATE_PATH"] = "/Users/mpb/WorkSpace/local_job/assets/cards"
    
    assets_dir = config["CANDIDATE_PATH"]
    output_dir = config["OUTPUT_DIR"]
    
    # Ensure absolute path for output
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    log_dir = os.path.join(output_dir, "logs")
    debug_dir = os.path.join(output_dir, "debug")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    if config["DEBUG"]["SAVE_LOGS"]:
        os.makedirs(log_dir, exist_ok=True)
    if config["DEBUG"]["SAVE_IMAGES"]:
        os.makedirs(debug_dir, exist_ok=True)
    
    print(f"Loading cards from {assets_dir}...")
    cards = load_all_cards(assets_dir)
    print(f"Loaded {len(cards)} cards.")
    
    num_images = config["TOTAL_IMAGES"]
    print(f"Generating {num_images} images with config: {config}")
    print(f"Using Parallel Synthesis with {mp.cpu_count()} cores.")
    
    start_idx = config.get("START_INDEX", 0)
    indices = range(start_idx, start_idx + num_images)
    
    # Prepare output directories list for worker
    out_dirs = [img_dir, lbl_dir, log_dir, debug_dir]

    # Use multiprocessing Pool
    # Determine workers
    use_parallel = config.get("USE_PARALLEL", True)
    # Use half of available cores as requested
    num_workers = max(1, mp.cpu_count() // 2)
    
    if use_parallel and num_workers > 1:
        print(f"Using Parallel Synthesis with {num_workers} cores.")
        with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(config, cards, assets_dir, out_dirs)) as pool:
            # Use tqdm for progress bar
            results = list(tqdm(pool.imap(process_one_image, indices), total=num_images))
    else:
        print("Using Sequential Synthesis (Single Core).")
        # Initialize worker global state for this process
        init_worker(config, cards, assets_dir, out_dirs)
        results = []
        for idx in tqdm(indices, total=num_images):
            results.append(process_one_image(idx))
    
    success_count = sum(results)
    print(f"Done. Successfully generated {success_count}/{num_images} images.")

import argparse
import json

if __name__ == "__main__":
    # Support spawn for consistency across platforms (optional but good for Mac)
    # mp.set_start_method('spawn') 
    
    parser = argparse.ArgumentParser(description="Synthetic Data Generator")
    parser.add_argument("--user_config", type=str, help="JSON string for configuration override")
    args = parser.parse_args()
    
    u_config = None
    if args.user_config:
        try:
            u_config = json.loads(args.user_config)
        except json.JSONDecodeError as e:
            print(f"Error parsing user_config JSON: {e}")
            
    main(u_config)
