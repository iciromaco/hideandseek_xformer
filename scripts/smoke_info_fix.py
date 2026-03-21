#!/usr/bin/env python3
"""
Smoke test: aggregate info_buffer gaze metrics in a standalone script.
Run with: python scripts/smoke_info_fix.py
"""

import os
import sys

# Ensure repository root is on sys.path so we can import top-level modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Simulate info_buffer with mixed dict and list forms
info_buffer = [
    {"dbg_seek_gaze_cos_front_max": 0.8, "dbg_seek_gaze_cos_front_dist_max": 0.12},
    [
        {"dbg_seek_gaze_cos_front_max": 0.3, "dbg_seek_gaze_cos_front_dist_max": 0.05},
        {"dbg_seek_gaze_cos_front_max": 0.6},
    ],
]

num_envs = 2
# Aggregation logic copied from main27_train_final

gaze_count = 0
gaze_sum = 0.0
gaze_max = 0.0
gaze_dist_sum = 0.0
gaze_dist_max = 0.0

for info in info_buffer:
    if isinstance(info, (list, tuple)):
        for i in range(num_envs):
            try:
                info_i = info[i]
            except Exception:
                info_i = None
            if info_i is None:
                val = 0.0
                val2 = 0.0
            else:
                val = float(info_i.get("dbg_seek_gaze_cos_front_max", 0.0)) if isinstance(info_i, dict) else float(0.0)
                val2 = float(info_i.get("dbg_seek_gaze_cos_front_dist_max", 0.0)) if isinstance(info_i, dict) else float(0.0)
            gaze_sum += val
            gaze_dist_sum += val2
            gaze_max = max(gaze_max, val)
            gaze_dist_max = max(gaze_dist_max, val2)
            gaze_count += 1
    elif isinstance(info, dict):
        val = float(info.get("dbg_seek_gaze_cos_front_max", 0.0))
        val2 = float(info.get("dbg_seek_gaze_cos_front_dist_max", 0.0))
        gaze_sum += val
        gaze_dist_sum += val2
        gaze_max = max(gaze_max, val)
        gaze_dist_max = max(gaze_dist_max, val2)
        gaze_count += 1

if gaze_count > 0:
    print("gaze_count", gaze_count)
    print("env/dbg_seek_gaze_cos_front_max_mean", gaze_sum / float(gaze_count))
    print("env/dbg_seek_gaze_cos_front_max", gaze_max)
    print("env/dbg_seek_gaze_cos_front_dist_max_mean", gaze_dist_sum / float(gaze_count))
    print("env/dbg_seek_gaze_cos_front_dist_max", gaze_dist_max)
else:
    print("no gaze data")
