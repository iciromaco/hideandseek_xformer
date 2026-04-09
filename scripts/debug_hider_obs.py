#!/usr/bin/env python3
import os
import sys
import numpy as np

# Ensure project root is on sys.path so `src` package imports work when run
# from the scripts/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.hns28_environment import TeamCosEnv


def print_hider_obs(env):
    h_key = env.hider_keys[0]
    h_idx = env.agent_keys.index(h_key)
    o = env._get_obs(h_idx)
    ak = h_key
    ens = [k for k in env.agent_keys if k != ak]
    if ak.startswith("s"):
        ens.sort(key=lambda k: (0 if k.startswith("h") else 1, k))
    else:
        ens.sort(key=lambda k: (0 if k.startswith("s") else 1, k))
    print("agent_keys:", env.agent_keys)
    for i, enm in enumerate(ens[:len(env.idx.OTHERS)]):
        en_idx = env.idx.OTHERS[i]
        relx = float(o[en_idx.REL_X])
        rely = float(o[en_idx.REL_Y])
        vis = float(o[en_idx.VISIBLE])
        print(f"  OTHERS[{i}] -> {enm}: VISIBLE={vis:.1f} REL=({relx:.3f},{rely:.3f})")


def main():
    env = TeamCosEnv(n_seekers=1, n_hiders=1, debug_mode=True, render_mode=None)
    obs, info = env.reset()
    print("After reset:")
    print_hider_obs(env)
    for step in range(1, 4):
        obs, reward, _, done, info = env.step(np.zeros(4))
        print(f"After step {step}: reward={reward:.3f} is_detected={info.get('is_detected')}")
        print_hider_obs(env)
        if done:
            break


if __name__ == '__main__':
    main()
