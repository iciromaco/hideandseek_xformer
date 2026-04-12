#!/usr/bin/env python3
import numpy as np

import sys
import os
# Ensure project root is on sys.path so `src` can be imported when
# running scripts from the tools/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.hns28_environment import TeamCosEnv

def main():
    env = TeamCosEnv()
    env.reset()
    env.override_learnable_policy = False
    print("learnable:", env.learnable_agent_key)
    for step in range(10):
        ret = env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        if len(ret) == 5:
            obs, rew, _, done, info = ret
        else:
            obs, rew, done, info = ret
        bid = env._resolve_body_id(env.learnable_agent_key)
        pos = env.data.xpos[bid][:2].copy()
        rot = env._agent_rot(env.learnable_agent_key)
        print(step, "pos", pos.tolist(), "rot", float(rot), "last_ctrl", env.last_debug_ctrl[env.learnable_agent_key])
        if done:
            env.reset()


if __name__ == '__main__':
    main()
