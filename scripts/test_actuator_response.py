#!/usr/bin/env python3
"""Place seeker at (2,2) facing north and apply s_fwd=1.0, log response."""

import math
import os
import sys
import time

import numpy as np


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from src.envs.hns_environment import TeamCosEnv
    except Exception:
        from envs.hns_environment import TeamCosEnv

    env = TeamCosEnv(mode="train", target="seeker", debug_mode=True)
    ak = env.learnable_agent_key
    print("learnable_agent_key:", ak)
    body_id = env.body_ids[ak]
    print("body_id:", body_id)

    # set position (2,2) and rotation north (pi/2)
    jx = env.qpos_indices[ak]["x"]
    jy = env.qpos_indices[ak]["y"]
    jz = env.qpos_indices[ak]["z"]
    jr = env.qpos_indices[ak]["rot"]
    qx_adr = int(env.model.jnt_qposadr[jx])
    qy_adr = int(env.model.jnt_qposadr[jy])
    qz_adr = int(env.model.jnt_qposadr[jz])
    qr_adr = int(env.model.jnt_qposadr[jr])

    # place at (2,2) keeping current body z
    try:
        cur_z = float(env.data.xpos[body_id][2])
    except Exception:
        cur_z = 0.5
    env.data.qpos[qx_adr] = 2.0
    env.data.qpos[qy_adr] = 2.0
    env.data.qpos[qz_adr] = cur_z
    env.data.qpos[qr_adr] = math.pi / 2.0
    try:
        import mujoco

        mujoco.mj_forward(env.model, env.data)
    except Exception:
        pass

    print("initial xpos:", env.data.xpos[body_id])

    # action: [fwd, turn, lock, grab]
    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    steps = 40
    for s in range(steps):
        next_obs, r, term, trun, info = env.step(action)
        pos = env.data.xpos[body_id].copy()
        xfrc = env.data.xfrc_applied[int(body_id), :2].copy()
        applied = info.get("applied_forward", None)
        print(f"STEP {s:02d}: pos=({pos[0]:.3f},{pos[1]:.3f}) z={pos[2]:.3f} applied_forward={applied} xfrc=({xfrc[0]:.3f},{xfrc[1]:.3f}) reward={r:.3f}")
        time.sleep(0.02)
        if term or trun:
            print("episode ended", term, trun)
            break

    env.close()


if __name__ == "__main__":
    main()
