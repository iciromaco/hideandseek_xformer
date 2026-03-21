#!/usr/bin/env python3
import csv
import json
import sys

INPUT = sys.argv[1] if len(sys.argv) > 1 else "debug_step_logs_ep0.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bad_frames_obs_summary.csv"

criteria_det_thresh = -0.2
criteria_wall_dist = 0.6

with open(INPUT, "r") as f, open(OUT, "w", newline="") as out:
    writer = csv.writer(out)
    header = [
        "step",
        "det_action0",
        "action0",
        "sampled_action0",
        "wall_distance",
        "vx",
        "vy",
        "speed",
        "gaze",
        "enemy_visible",
        "nearest_enemy_dist",
        "base_reward",
        "reward",
        "obs",
        "next_obs",
    ]
    writer.writerow(header)
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        det = obj.get("det_action") or obj.get("deterministic_action") or None
        wall = obj.get("wall_distance")
        if det and isinstance(det, (list, tuple)):
            det0 = float(det[0])
        else:
            det0 = None
        if det0 is None or wall is None:
            continue
        if det0 < criteria_det_thresh and float(wall) < criteria_wall_dist:
            row = [
                obj.get("step"),
                det0,
                (obj.get("action") or [None])[0] if obj.get("action") else None,
                ((obj.get("sampled_action") or [None])[0] if obj.get("sampled_action") else None),
                obj.get("wall_distance"),
                obj.get("vx"),
                obj.get("vy"),
                obj.get("speed"),
                obj.get("gaze"),
                obj.get("enemy_visible"),
                obj.get("nearest_enemy_dist"),
                obj.get("base_reward"),
                obj.get("reward"),
                json.dumps(obj.get("obs")) if "obs" in obj else "",
                json.dumps(obj.get("next_obs")) if "next_obs" in obj else "",
            ]
            writer.writerow(row)

print(f"Wrote bad frames to {OUT} (criteria: det_action0<{criteria_det_thresh} & wall_distance<{criteria_wall_dist})")
