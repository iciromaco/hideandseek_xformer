"""Run parameter sweep for wall-repulsion settings and report simple stats.
Usage: python scripts/diag_sweep.py
"""

import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from envs.hns_environment import TeamCosEnv


def run_once(cfg, steps=300, fwd=1.0):
    env = TeamCosEnv(mode="initial", render_mode=None, debug_mode=False)
    # apply runtime overrides if provided in cfg
    for k, v in cfg.items():
        setattr(env, k, v)
    obs, info = env.reset()
    learnable = env.learnable_agent_key
    bid = env.body_ids[learnable]
    act_fwd_id = env.actuator_ids[f"{learnable}_fwd"]

    fb_vals = []
    ctrl_vals = []
    xfrc_vals = []

    for step in range(steps):
        action = np.array([fwd, 0.0, 0.0, 0.0], dtype=np.float32)
        res = env.step(action)
        if len(res) == 4:
            obs, rew, done, info = res
        else:
            obs, rew, term, trunc, info = res
            done = bool(term or trunc)
        # try to parse debug entries from env.last_debug_ctrl if present
        last_debug = env.last_debug_ctrl.get(learnable, (0.0, 0.0))
        # fb stored in last_debug[0]? we can't assume format; instead capture env._prev_wall_rep
        prev = float(getattr(env, "_prev_wall_rep", {}).get(learnable, 0.0))
        ctrl_vals.append(prev)
        # read current applied ctrl actuator value
        try:
            ctrl_act = float(env.data.ctrl[act_fwd_id])
        except Exception:
            ctrl_act = 0.0
        fb_vals.append(ctrl_act)
        try:
            xfrc = env.data.xfrc_applied[bid, :2].copy()
            xfrc_vals.append([float(xfrc[0]), float(xfrc[1])])
        except Exception:
            xfrc_vals.append([0.0, 0.0])
        if done:
            break
    env.viewer = None
    return {
        "cfg": cfg,
        "ctrl_stats": stats_np(np.array(ctrl_vals)),
        "act_ctrl_stats": stats_np(np.array(fb_vals)),
        "xfrc_stats": stats_np(np.array(xfrc_vals)),
    }


def stats_np(arr):
    if arr.size == 0:
        return {"count": 0}
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        a = arr
        return {"count": int(a.size), "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()), "max": float(a.max()), "95pct": float(np.percentile(a, 95))}
    else:
        return {
            "count": int(arr.shape[0]),
            "mean": list(map(float, arr.mean(axis=0))),
            "std": list(map(float, arr.std(axis=0))),
            "min": list(map(float, arr.min(axis=0))),
            "max": list(map(float, arr.max(axis=0))),
            "95pct": list(map(float, np.percentile(arr, 95, axis=0))),
        }


if __name__ == "__main__":
    # parameter grids
    scales = [0.5, 0.2, 0.1, 0.05]
    lps = [0.2, 0.5, 0.8]
    clips = [None, 100.0, 50.0]
    accum_clamps = [None, 150.0, 100.0]

    results = []
    # baseline (class defaults)
    results.append(run_once({}))

    # sweep scale x lp
    for s in scales:
        for lp in lps:
            cfg = {"WALL_REPULSION_CTRL_SCALE": float(s), "WALL_REPULSION_CTRL_LP": float(lp)}
            results.append(run_once(cfg))

    # test clipping
    for clip in clips:
        cfg = {"WALL_REPULSION_CTRL_CLIP": clip} if clip is not None else {}
        results.append(run_once(cfg))

    # test accum clamps
    for a in accum_clamps:
        cfg = {"WALL_REPULSION_ACCUM_CLAMP_MAX": a} if a is not None else {}
        results.append(run_once(cfg))

    print(json.dumps(results, indent=2))
