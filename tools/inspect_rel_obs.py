#!/usr/bin/env python3
import sys
import os
import numpy as np

# Ensure project root is on sys.path so `src` can be imported when
# running scripts from the tools/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.hns28_environment import TeamCosEnv


def main():
    env = TeamCosEnv()
    env.reset()
    # 1ステップ進めて観測を更新
    env.step(np.zeros(4))
    s_key = env.seeker_keys[0]
    h_key = env.hider_keys[0]
    s_idx = env.agent_keys.index(s_key)
    obs_s = env._get_obs(s_idx)
    ens = env._ens_orderings[s_key]
    en_idx = None
    for i, e in enumerate(ens[: len(env.idx.OTHERS)]):
        if e == h_key:
            en_idx = env.idx.OTHERS[i]
            break

    bid_s = env._resolve_body_id(s_key)
    bid_h = env._resolve_body_id(h_key)
    d_w = env._body_pos(bid_h) - env._body_pos(bid_s)
    print("world delta (h - s):", float(d_w[0]), float(d_w[1]))
    if en_idx is not None:
        print("obs REL_X, REL_Y:", float(obs_s[en_idx.REL_X]), float(obs_s[en_idx.REL_Y]))
    else:
        print("hider not in OTHERS window")


if __name__ == '__main__':
    main()
