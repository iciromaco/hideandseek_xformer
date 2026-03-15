#!/usr/bin/env python3
"""短い調査スクリプト: Seeker で +1/-1 前進入力を投げ、位置差と報酬列を比較する。"""
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from src.envs.hns_environment import TeamCosEnv


def test_action(action, steps=10):
    env = TeamCosEnv(mode="initial", target="seeker", n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1, render_mode=None)
    obs, info = env.reset()
    seeker_key = env.seeker_keys[0]
    bid = env.body_ids[seeker_key]
    pos_before = env.data.xpos[bid].copy()
    rewards = []
    for i in range(steps):
        obs, r, _, done, info = env.step(np.asarray(action, dtype=np.float32))
        rewards.append(float(r))
        if done:
            break
    pos_after = env.data.xpos[bid].copy()
    env.close()
    return pos_before, pos_after, rewards, info


if __name__ == "__main__":
    for a, name in [([1, 0, 0, 0], "forward"), ([-1, 0, 0, 0], "backward")]:
        pb, pa, rewards, info = test_action(a, steps=10)
        delta = pa - pb
        print(f"=== {name} ===")
        print("pos_before:", pb)
        print("pos_after :", pa)
        print("delta     :", delta)
        print("rewards   :", rewards)
        print("info keys :", sorted(info.keys()))
        print()
