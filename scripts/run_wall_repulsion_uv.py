#!/usr/bin/env python3
import os
import sys
import time

import mujoco
import numpy as np

# ensure repo root on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.envs.hns_environment import TeamCosEnv


def main():
    env = TeamCosEnv(debug_mode=True)
    obs, info = env.reset()
    print("[RUN] reset wall_distance=", info.get("wall_distance"))

    ak = env.learnable_agent_key
    bid = env.body_ids[ak]

    # debug: show class attrs related to wall/agent constants
    print("[RUN] class attrs:", [a for a in dir(env.__class__) if ("AGENT" in a or "WALL" in a)])
    try:
        modname = env.__class__.__module__
        print("[RUN] class module:", modname, "file=", sys.modules[modname].__file__)
    except Exception as _:
        pass
    print("[RUN] effect/clear attrs:", [a for a in dir(env.__class__) if ("EFFECT" in a or "CLEARANCE" in a)])

    # place agent near east wall (arena half - margin)
    try:
        x_idx = env.model.jnt_qposadr[env.qpos_indices[ak]["x"]]
        y_idx = env.model.jnt_qposadr[env.qpos_indices[ak]["y"]]
        z_idx = env.model.jnt_qposadr[env.qpos_indices[ak]["z"]]
        r_idx = env.model.jnt_qposadr[env.qpos_indices[ak]["rot"]]
        # try a sequence of offsets toward the east wall until clearance < WALL_REPULSION_CLEARANCE
        offsets = [0.35, 0.2, 0.1, 0.05, 0.02, 0.01]
        placed = False
        for off in offsets:
            env.data.qpos[x_idx] = float(env.ARENA_HALF) - off
            env.data.qpos[y_idx] = 0.0
            env.data.qpos[z_idx] = 0.5
            env.data.qpos[r_idx] = 0.0
            mujoco.mj_forward(env.model, env.data)
            dist, nx, ny = env.vis_engine.sample_sdf_with_normal(env.data.xpos[bid][0], env.data.xpos[bid][1])
            # use explicit effective radius (fallback to 0.4 if class attr missing)
            try:
                r_eff = float(env.AGENT_EFFECTIVE_RADIUS)
            except Exception:
                r_eff = 0.4
            clearance = float(dist) - r_eff
            print(f"[RUN] try offset={off:.3f} pos={env.data.xpos[bid][:2]} dist={dist:.3f} clearance={clearance:.3f}")
            if clearance < float(getattr(env, "WALL_REPULSION_CLEARANCE", 0.2)):
                placed = True
                break
        if not placed:
            # keep the last attempted placement and proceed
            print("[RUN] could not reach clearance threshold; proceeding with closest placement")
        else:
            print(f"[RUN] placed {ak} at", env.data.xpos[bid][:2], f"(clearance={clearance:.3f})")
    except Exception as e:
        print("[RUN] failed to place agent:", e)

    # run short loop with constant forward command to trigger repulsion
    act = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for step in range(120):
        # sample SDF/clearance before stepping
        dist, nx, ny = env.vis_engine.sample_sdf_with_normal(env.data.xpos[bid][0], env.data.xpos[bid][1])
        r_eff = float(getattr(env, "AGENT_EFFECTIVE_RADIUS", 0.4))
        clearance = float(dist) - r_eff
        last_ctrl = env.last_debug_ctrl.get(env.learnable_agent_key, (0.0, 0.0))
        print(f"[STEP {step:03d}] pre dist={dist:.3f} clearance={clearance:.3f} last_ctrl_f={last_ctrl[0]:.3f}")
        obs, rew, term, done, info = env.step(act)
        # inspect applied force on body
        try:
            fx = float(env.data.xfrc_applied[bid, 0])
            fy = float(env.data.xfrc_applied[bid, 1])
        except Exception:
            fx = fy = 0.0
        print(f"[STEP {step:03d}] post info_last_ctrl_f={info.get('dbg_last_ctrl_f', None)} fx={fx:.1f} fy={fy:.1f}")
        time.sleep(0.01)
    print("[RUN] finished")


if __name__ == "__main__":
    main()
