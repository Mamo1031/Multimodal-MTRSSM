#!/usr/bin/env python3
"""Training script for MoPoE-MMTRSSM."""

import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from _train_common import run_training

if __name__ == "__main__":
    default_config = "src/multimodal_rssm/models/mmtrssm/mopoe_mmtrssm/configs/default.yaml"
    run_training(default_config)
