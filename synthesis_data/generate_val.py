
import os
import sys

# Ensure we can import from synthesis_data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from synthesis_data.main import main

# Base Output Dir
VAL_OUTPUT_DIR = "/Users/mpb/WorkSpace/local_job/synthesis_data/dataset_val"

# Define batches
batches = [
    {
        "name": "Easy",
        "count": 200,
        "config": {
            "LIGHTING_DIST": {0: 1.0}, # Minimal
            "PERSPECTIVE_DIST": {"NONE": 1.0},
            "OUTPUT_DIR": VAL_OUTPUT_DIR
        }
    },
    {
        "name": "Normal",
        "count": 300,
        "config": {
            "LIGHTING_DIST": {1: 1.0}, # Weak
            "PERSPECTIVE_DIST": {"NORMAL": 1.0},
            "OUTPUT_DIR": VAL_OUTPUT_DIR
        }
    },
    {
        "name": "Hard",
        "count": 300,
        "config": {
            "LIGHTING_DIST": {2: 1.0}, # Medium
            "PERSPECTIVE_DIST": {"NORMAL": 1.0},
            "OUTPUT_DIR": VAL_OUTPUT_DIR
        }
    },
    {
        "name": "Extreme",
        "count": 200,
        "config": {
            "LIGHTING_DIST": {3: 1.0}, # Strong
            "PERSPECTIVE_DIST": {"HEAVY": 1.0},
            "OUTPUT_DIR": VAL_OUTPUT_DIR
        }
    }
]

def generate_validation_set():
    current_idx = 0
    print(f"Starting Generation of Validation Set to {VAL_OUTPUT_DIR}")
    
    for batch in batches:
        print(f"--- Generating {batch['name']} Batch ({batch['count']} images) ---")
        cfg = batch['config']
        cfg["TOTAL_IMAGES"] = batch['count']
        cfg["START_INDEX"] = current_idx
        # Disable debug images to save space/time for val set if desired, 
        # but user might want to inspect. Let's keep defaults or ensure they are set.
        # Inherit default debug settings from main.py default config
        
        main(cfg)
        current_idx += batch['count']
        
    print(f"Validation Set Generation Complete. Total: {current_idx} images.")

if __name__ == "__main__":
    generate_validation_set()
