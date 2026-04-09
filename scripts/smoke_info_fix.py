#!/usr/bin/env python3
"""
Smoke test: aggregate info_buffer gaze metrics in a standalone script.
Run with: python scripts/smoke_info_fix.py
"""
import os
import sys

# Ensure repository root is on sys.path so we can import top-level modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import main27_train_final as m

# Simulate info_buffer with mixed dict and list forms
info_buffer = [
    {'dbg_seek_gaze_cos_front_max': 0.8, 'dbg_seek_gaze_cos_front_dist_max': 0.12},
    [
        {'dbg_seek_gaze_cos_front_max': 0.3, 'dbg_seek_gaze_cos_front_dist_max': 0.05},
        {'dbg_seek_gaze_cos_front_max': 0.6},
    ],
]

num_envs = 2
# Aggregation logic copied from main27_train_final

print('gaze metrics have been removed from the environment; this smoke script is no longer applicable.')
