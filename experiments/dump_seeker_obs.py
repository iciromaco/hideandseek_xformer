#!/usr/bin/env python3
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.envs.hns_environment import TeamCosEnv


def pretty_print(obs, env):
    idx = env.idx
    print("obs length:", len(obs), "expected total_dim:", idx.total_dim)
    # SELF
    s = idx.SELF.SLICE
    print("\n-- SELF --")
    print(
        f"VEL_X={obs[s.start + idx.SELF.VEL_X]:.4f}",
        f"VEL_Y={obs[s.start + idx.SELF.VEL_Y]:.4f}",
        f"ROT={obs[s.start + idx.SELF.ROT]:.4f}",
        f"COS_ROT={obs[s.start + idx.SELF.COS_ROT]:.4f}",
        f"SIN_ROT={obs[s.start + idx.SELF.SIN_ROT]:.4f}",
    )

    # LIDAR
    print("\n-- LIDAR (first 12) --")
    lidar = obs[5:17]
    print(np.round(lidar, 3))

    # BOXES
    if idx.B:
        print("\n-- BOXES --")
        for i, b in enumerate(idx.B):
            sl = b.SLICE
            vals = obs[sl]
            # Note: `IS_MOVING` here is an object-level flag (boxes/ramps).
            # Agent-to-agent motion flag was removed and replaced by
            # `BEING_HIT` in the agent schema; see `src/core/obs_indices.py`.
            print(f"box{i+1}: REL_X={vals[0]:.3f} REL_Y={vals[1]:.3f} VEL_X={vals[2]:.3f} IS_MOVING={vals[6]:.1f}")

    # RAMPS
    if idx.RAMP:
        print("\n-- RAMPS --")
        for i, r in enumerate(idx.RAMP):
            sl = r.SLICE
            vals = obs[sl]
            # `IS_MOVING` for ramps (object-level)
            print(f"ramp{i+1}: REL_X={vals[0]:.3f} REL_Y={vals[1]:.3f} IS_MOVING={vals[6]:.1f}")

    # OTHERS (agents)
    if idx.OTHERS:
        print("\n-- OTHER AGENTS --")
        for i, a in enumerate(idx.OTHERS):
            sl = a.SLICE
            vals = obs[sl]
            # Agent schema fields: REL_X, REL_Y, VEL_X, VEL_Y, QUAT_0, QUAT_1, BEING_HIT, VISIBLE
            being_hit = int(vals[6])
            visible = int(vals[7])
            print(f"agent[{i}]: REL_X={vals[0]:.3f} REL_Y={vals[1]:.3f} VEL_X={vals[2]:.3f} BEING_HIT={being_hit} VISIBLE={visible}")


if __name__ == "__main__":
    env = TeamCosEnv()
    # ensure learnable is seeker
    env.learnable_agent_key = env.seeker_keys[0]
    env.learnable_agent_index = env.agent_keys.index(env.learnable_agent_key)
    # reset and compute raw obs
    env.current_step = 0
    env._init_agent_intelligence()
    env._init_interaction_state()
    obs = env._get_obs(env.learnable_agent_index)
    # obs is raw (not normalized) as _get_obs returns before normalize
    pretty_print(obs, env)
    # Also show visibility flags from info by doing a single step with rule-based actions
    # run one step to get info
    a = env.action_space.sample() if hasattr(env, "action_space") else np.zeros(4)
    try:
        _obs_norm, info = env.reset()
    except Exception:
        # some reset signatures return differently; call reset like earlier scripts
        _obs_norm = env.reset()
        info = {}
    print("\nreset info:", info)
    # also do env.step with zero action to collect info
    out = env.step([0.0, 0.0, 0.0, 0.0])
    if len(out) == 5:
        next_obs, reward, done, truncated, info = out
    else:
        next_obs, reward, done, info = out
    print("\nstep info keys:", list(info.keys()))
    print(
        "sample info:",
        {
            k: info.get(k)
            for k in [
                "dbg_seek_gaze_cos_front_max",
                "dbg_seek_gaze_cos_front_dist_max",
                "dbg_learnable_hider_seen",
            ]
        },
    )
