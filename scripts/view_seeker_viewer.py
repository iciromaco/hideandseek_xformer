#!/usr/bin/env python3
"""Place seeker at (2,-2) facing north and open viewer for manual inspection."""

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

    env = TeamCosEnv(mode="train", target="seeker", render_mode="human", debug_mode=True)
    ak = env.learnable_agent_key
    print("learnable_agent_key:", ak)
    body_id = env.body_ids[ak]
    print("body_id:", body_id)

    # prepare qpos addresses
    jx = env.qpos_indices[ak]["x"]
    jy = env.qpos_indices[ak]["y"]
    jz = env.qpos_indices[ak]["z"]
    jr = env.qpos_indices[ak]["rot"]
    qx_adr = int(env.model.jnt_qposadr[jx])
    qy_adr = int(env.model.jnt_qposadr[jy])
    qz_adr = int(env.model.jnt_qposadr[jz])
    qr_adr = int(env.model.jnt_qposadr[jr])

    # place seeker at (2, -2) facing north (pi/2)
    try:
        cur_z = float(env.data.xpos[body_id][2])
    except Exception:
        cur_z = 0.5
    env.data.qpos[qx_adr] = 2.0
    env.data.qpos[qy_adr] = -2.0
    env.data.qpos[qz_adr] = cur_z
    env.data.qpos[qr_adr] = math.pi / 2.0

    try:
        import mujoco

        mujoco.mj_forward(env.model, env.data)
    except Exception:
        pass

    print("initial xpos:", env.data.xpos[body_id])

    # run viewer loop, apply forward continuously
    action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    try:
        while True:
            obs, r, term, trun, info = env.step(action)
            env.render()
            # print small status line
            pos = env.data.xpos[body_id]
            print(f"pos=({pos[0]:.3f},{pos[1]:.3f}) z={pos[2]:.3f} applied_forward={info.get('applied_forward',None)}")
            time.sleep(0.03)
            if term or trun:
                print("episode ended", term, trun)
                break
    except KeyboardInterrupt:
        print("viewer interrupted by user")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
