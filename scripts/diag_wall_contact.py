"""診断スクリプト: Seeker を北向きに進め、壁接触時の値を出力する
実行: python scripts/diag_wall_contact.py
"""

import os
import sys
import time

import numpy as np

# ensure workspace `src` is importable when running from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from envs.hns_environment import TeamCosEnv


def run(steps=300, fwd=1.0):
    env = TeamCosEnv(mode="initial", render_mode=None, debug_mode=True)
    obs, info = env.reset()
    learnable = env.learnable_agent_key
    bid = env.body_ids[learnable]
    act_fwd_id = env.actuator_ids[f"{learnable}_fwd"]

    print("learnable_agent_key:", learnable, "body_id:", bid)
    for step in range(steps):
        # apply constant forward action
        action = np.array([fwd, 0.0, 0.0, 0.0], dtype=np.float32)
        res = env.step(action)
        # support both old (obs, rew, done, info) and gymnasium-style
        # (obs, rew, terminated, truncated, info)
        if len(res) == 4:
            obs, rew, done, info = res
        else:
            obs, rew, term, trunc, info = res
            done = bool(term or trunc)
        pos = env.data.xpos[bid]
        # try to read wall SDF and normal
        try:
            dist, nx, ny = env.vis_engine.sample_sdf_with_normal(pos[0], pos[1])
        except Exception:
            dist, nx, ny = (None, None, None)
        # read stored debug/control values
        last_model = getattr(env, "applied_forward_model", None)
        last_env = getattr(env, "applied_forward_env", None)
        last_debug = env.last_debug_ctrl.get(learnable, (None, None))
        xfrc = env.data.xfrc_applied[bid, :2].copy() if hasattr(env.data, "xfrc_applied") else (None, None)
        wall_obs = None
        try:
            # obs idx mapping
            wi = env.idx.WALL_DIST
            wall_obs = float(obs[wi])
        except Exception:
            wall_obs = None
        print(
            f"STEP {step:03d}: pos=({pos[0]:.3f},{pos[1]:.3f}) z={pos[2]:.3f} wall_dist_sdf={dist} wall_obs={wall_obs} nx={nx} ny={ny} applied_model={last_model} applied_env={last_env} last_debug={last_debug} ctrl_fwd={env.data.ctrl[act_fwd_id]:.3f} xfrc={xfrc}"
        )
        # detect contacts involving the floor geom and print agent z when touching
        try:
            ncon = int(env.data.ncon)
            for ci in range(ncon):
                c = env.data.contact[ci]
                try:
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                except Exception:
                    g1 = int(c.geom1)
                    g2 = int(c.geom2)
                if g1 == "floor" or g2 == "floor":
                    print(f"-- FLOOR CONTACT at step={step:03d} agent_z={pos[2]:.3f} contact_pos=({c.pos[0]:.3f},{c.pos[1]:.3f},{c.pos[2]:.3f}) geom1={g1} geom2={g2}")
                    break
        except Exception:
            pass
        if done:
            print("episode ended", step, done)
            break


if __name__ == "__main__":
    run()
