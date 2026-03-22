#!/usr/bin/env python3
"""
Simple ramp-climb verification:
- create env with one ramp
- place seeker on the lower side of ramp, facing uphill
- apply forward action repeatedly and log agent world z over time
"""

import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.hns_environment import TeamCosEnv


def run_climb(steps=80, forward_val=1.0, debug=False):
    # must keep n_hiders>=1 per env checks
    env = TeamCosEnv(mode="initial", target="seeker", n_seekers=1, n_hiders=1, n_boxes=0, n_ramps=1, render_mode=None, debug_mode=debug, action_repeat=1)
    try:
        obs, info = env.reset()
        # identify ramp body id
        ramp_key = "ramp1"
        if ramp_key not in env.obj_body_map:
            print("No ramp found in env; aborting")
            return
        rid = env.obj_body_map[ramp_key]
        # uphill direction (unit vector)
        up = env._ramp_uphill_dir(rid)
        # ramp position (body world pos)
        rpos = env.data.xpos[rid][:2].copy()

        seeker = env.seeker_keys[0]
        bid = env.body_ids[seeker]

        # compute spawn position: place behind ramp a bit along -up
        offset = env.R_RAMP + env.R_AGENT + 0.05
        spawn_xy = rpos - up * offset
        # set qpos joints for anchor
        jx = env.qpos_indices[seeker]["x"]
        jy = env.qpos_indices[seeker]["y"]
        jz = env.qpos_indices[seeker]["z"]
        jr = env.qpos_indices[seeker]["rot"]
        env.data.qpos[env.model.jnt_qposadr[jx]] = float(spawn_xy[0])
        env.data.qpos[env.model.jnt_qposadr[jy]] = float(spawn_xy[1])
        env.data.qpos[env.model.jnt_qposadr[jz]] = 0.5
        # face uphill
        ang = math.atan2(up[1], up[0])
        env.data.qpos[env.model.jnt_qposadr[jr]] = float(ang)
        # forward physics
        import mujoco

        mujoco.mj_forward(env.model, env.data)

        zs = []
        print(f"spawn_xy={spawn_xy} ramp_pos={rpos} up={up} ang={ang:.3f}")
        for t in range(steps):
            a = np.zeros(env.action_space.shape[0], dtype=np.float32)
            a[0] = float(forward_val)
            obs, r, _, done, info = env.step(a)
            z = float(env.data.xpos[bid][2])
            zs.append(z)
            print(f"step={t:03d} z={z:.4f} reward={float(r):.4f}")
            if done:
                break
        print("z_sequence:", zs)
    finally:
        env.close()


if __name__ == "__main__":
    run_climb(steps=160, forward_val=1.0, debug=False)
