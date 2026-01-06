
import sys
import os
# Ensure synthesis_data is in path
sys.path.append(os.path.join(os.getcwd(), 'synthesis_data'))
from main import main

if __name__ == "__main__":
    # Override config for speed and ensure high density
    main({
        'TOTAL_IMAGES': 100,
        'MIN_CARDS': 40,
        'MAX_CARDS': 80,
        'USE_PARALLEL': False,
        'DEBUG': {'SAVE_IMAGES': True, 'SAVE_LOGS': True, 'INTERVAL': 1}
    })
