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
    "MIN_CARDS": 5,
    "MAX_CARDS": 10,
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
    "CLASS_MAP": {
        1: 0,  # Triangle
        2: 1,  # Square
        3: 2,  # Rect Short
        4: 3   # Rect Long
    }
}

import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import time
import logging

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
    
    worker_ctx['config'] = config
    worker_ctx['assets_dir'] = assets_dir
    worker_ctx['dirs'] = out_dirs
    
    # Initialize Engine and Renderer locally for this process
    worker_ctx['engine'] = SynthesisEngine(
        cards, 
        canvas_size=(2048, 2048), 
        collision_config=config["COLLISION"]
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
        n_cards = random.randint(min_c, max_c)
        placed_cards = engine.generate(min_cards=n_cards)
        
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
        should_debug = (i % 10 == 0)

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
        config.update(user_config)
        if "COLLISION" in user_config: config["COLLISION"] = user_config["COLLISION"]
        if "LIGHTING_DIST" in user_config: config["LIGHTING_DIST"] = user_config["LIGHTING_DIST"]
        if "PERSPECTIVE_DIST" in user_config: config["PERSPECTIVE_DIST"] = user_config["PERSPECTIVE_DIST"]
        if "DEBUG" in user_config: config["DEBUG"] = user_config["DEBUG"]

    # Determine default assets path relative to this script
    if not config.get("CANDIDATE_PATH"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming folder structure: synthesis_data/main.py -> ../assets2
        candidate_path = os.path.join(base_dir, "..", "assets2")
        if os.path.exists(candidate_path):
             config["CANDIDATE_PATH"] = candidate_path
        else:
             # Fallback to absolute if simple relative check fails (e.g. running from odd location)
             config["CANDIDATE_PATH"] = "/Users/mpb/WorkSpace/local_job/assets2"
    
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
    # We leave 1 core free if more than 4, else use all
    num_workers = mp.cpu_count()
    if num_workers > 4:
        num_workers -= 1
        
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(config, cards, assets_dir, out_dirs)) as pool:
        # Use tqdm for progress bar
        results = list(tqdm(pool.imap(process_one_image, indices), total=num_images))
    
    success_count = sum(results)
    print(f"Done. Successfully generated {success_count}/{num_images} images.")

if __name__ == "__main__":
    # Support spawn for consistency across platforms (optional but good for Mac)
    # mp.set_start_method('spawn') 
    main()
