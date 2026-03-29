import math
import os
import sys
import time

import numpy as np

# Ensure repository root is on sys.path so `from src...` works when run from tools/
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.envs.hns_environment import TeamCosEnv

# Enable detailed visibility debug for this diagnostic run
try:
    from src.core import visibility_engine as _viseng

    _viseng.VIS_DEBUG = True
except Exception:
    pass


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    # Create env with viewer enabled and optional seed for determinism
    env = TeamCosEnv(render_mode=None, USE_VIEWER=True, seed=args.seed)
    obs, info = env.reset(seed=args.seed)

    # set positions per user's description
    s_xy = (-2.5, -2.5)
    h_xy = (0.5, 0.5)

    seekers = [k for k in env.agent_keys if k.startswith("s")]
    hiders = [k for k in env.agent_keys if k.startswith("h")]
    if not seekers or not hiders:
        print("no agents in env")
        return
    sk = seekers[0]
    hk = hiders[0]

    s_bid = env.body_ids[sk]
    h_bid = env.body_ids[hk]

    # place agents
    env.data.xpos[s_bid][0] = float(s_xy[0])
    env.data.xpos[s_bid][1] = float(s_xy[1])
    env.data.xpos[s_bid][2] = 0.4
    env.data.xpos[h_bid][0] = float(h_xy[0])
    env.data.xpos[h_bid][1] = float(h_xy[1])
    env.data.xpos[h_bid][2] = 0.4

    # set seeker rotation to face south (-pi/2)
    ridx = env.model.jnt_qposadr[env.qpos_indices[sk]["rot"]]
    env.data.qpos[ridx] = float(-math.pi / 2)

    # Ensure MuJoCo updates derived quantities (geom positions, etc.)
    try:
        mujoco.mj_forward(env.model, env.data)
    except Exception:
        pass

    print(f"Positions set. Entering render loop. Close viewer window to exit. seed={args.seed}")

    # print initial visibility
    p1 = np.array([env.data.xpos[s_bid][0], env.data.xpos[s_bid][1], 0.4])
    p2 = np.array([env.data.xpos[h_bid][0], env.data.xpos[h_bid][1], 0.4])
    try:
        vis = bool(env.vis_engine.is_visible(p1, p2, body_exclude=s_bid, target_body_id=h_bid))
    except Exception as e:
        vis = False
        print("vis_engine raised:", e)
    print("vis_engine initial:", vis)
    print("_is_vis wrapper initial:", env._is_vis(env.data.xpos[s_bid][:2], float(env.data.qpos[ridx]), env.data.xpos[h_bid][:2], s_bid, h_bid))

    # Try to render for a short period
    try:
        for i in range(300):
            # Ensure our manual placement is applied inside the simulation
            if i == 0:
                try:
                    env.data.xpos[s_bid][0] = float(s_xy[0])
                    env.data.xpos[s_bid][1] = float(s_xy[1])
                    env.data.xpos[s_bid][2] = 0.4
                    env.data.xpos[h_bid][0] = float(h_xy[0])
                    env.data.xpos[h_bid][1] = float(h_xy[1])
                    env.data.xpos[h_bid][2] = 0.4
                    env.data.qpos[ridx] = float(-math.pi / 2)
                    try:
                        mujoco.mj_forward(env.model, env.data)
                    except Exception:
                        pass
                except Exception:
                    pass

            # render (may launch passive viewer); keep CPU loop slow
            try:
                env.render()
            except Exception as e:
                print("render error:", e)
                break
            # periodically print vis state
            if i % 30 == 0:
                try:
                    # Recompute positions from the live simulation state so
                    # this external check uses the exact same inputs as
                    # the render-time instrumentation.
                    p1_live = np.array([env.data.xpos[s_bid][0], env.data.xpos[s_bid][1], 0.4])
                    p2_live = np.array([env.data.xpos[h_bid][0], env.data.xpos[h_bid][1], 0.4])
                    vis = bool(env.vis_engine.is_visible(p1_live, p2_live, body_exclude=s_bid, target_body_id=h_bid))
                except Exception:
                    vis = False
                try:
                    _is = env._is_vis(env.data.xpos[s_bid][:2], float(env.data.qpos[ridx]), env.data.xpos[h_bid][:2], s_bid, h_bid)
                except Exception:
                    _is = False
                print(f"step {i}: vis_engine={vis} _is_vis={_is}")
            time.sleep(0.05)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
