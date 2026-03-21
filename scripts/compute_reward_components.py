#!/usr/bin/env python3
"""Compute team/distance/total reward breakdown by enemy visibility from debug_step_logs_ep0.jsonl"""

import json
import math
from pathlib import Path

import numpy as np

IN = Path("debug_step_logs_ep0.jsonl")
OUT = Path("analysis_reward_components.txt")
CSV = Path("reward_components_by_frame.csv")


def build_idx_cache():
    # mirror of _build_reward_index_cache for ObsIdx used in logs
    # SELF.VEL_X = 0, SELF.VEL_Y = 1
    cache = {}
    cache["self_vel_x"] = 0
    cache["self_vel_y"] = 1
    cache["lidar"] = list(range(5, 17))
    # enemy others start at 17, each AgentSchema is 8 dims
    # we don't know n_others; detect from obs length per line
    return cache


def estimate_min_enemy_dist(obs):
    # obs is list
    if obs is None:
        return None
    L = len(obs)
    # others start at 17
    start = 17
    if L <= start:
        return None
    rem = L - start
    n_others = rem // 8
    dmin = None
    for i in range(n_others):
        base = start + i * 8
        rx = float(obs[base + 0])
        ry = float(obs[base + 1])
        dist = math.hypot(rx, ry) * P_SCALE
        if dmin is None or dist < dmin:
            dmin = dist
    return dmin


# constants from main27_train_final
P_SCALE = 1.0  # obs positions already in world scale in logs? assume 1.0; adjust if needed


def dist_bonus_from_min_dist(min_dist):
    if min_dist is None:
        return 0.0
    ratio = min(min_dist / 12.0, 1.0)
    return 0.2 * ratio


def main():
    if not IN.exists():
        print("Missing input", IN)
        return
    rows = []
    with IN.open() as f:
        for line in f:
            d = json.loads(line)
            vis = bool(d.get("enemy_visible", False))
            base = float(d.get("base_reward", float("nan")))
            reward = float(d.get("reward", float("nan")))
            obs = d.get("obs", None)
            min_dist = estimate_min_enemy_dist(obs)
            db = dist_bonus_from_min_dist(min_dist) if not vis else 0.0
            rows.append((vis, base, db, reward, min_dist))

    # aggregate
    by_vis = {False: [], True: []}
    for r in rows:
        by_vis[r[0]].append(r)

    def stats(lst):
        if not lst:
            return (0, float("nan"), float("nan"))
        bases = [x[1] for x in lst]
        dbs = [x[2] for x in lst]
        rewards = [x[3] for x in lst]
        return (
            len(lst),
            float(np.mean(bases)),
            float(np.mean(dbs)),
            float(np.mean(rewards)),
        )

    with OUT.open("w") as o:
        o.write("Visibility, N, mean_base_reward, mean_est_dist_bonus, mean_total_reward\n")
        for vis in (False, True):
            n, mb, md, mr = stats(by_vis[vis])
            o.write(f"{vis}, {n}, {mb:.6g}, {md:.6g}, {mr:.6g}\n")
        o.write("\nPer-frame details (vis,base,est_dist_bonus,total,min_dist)\n")
        for vis, base, db, reward, mdist in rows:
            o.write(f"{vis},{base:.6g},{db:.6g},{reward:.6g},{'' if mdist is None else f'{mdist:.6g}'}\n")

    # write CSV too
    with CSV.open("w") as c:
        c.write("vis,base_reward,est_dist_bonus,total_reward,min_dist\n")
        for vis, base, db, reward, mdist in rows:
            c.write(f"{int(vis)},{base:.6g},{db:.6g},{reward:.6g},{'' if mdist is None else f'{mdist:.6g}'}\n")

    print(f"Wrote {OUT} and {CSV}")


if __name__ == "__main__":
    main()
