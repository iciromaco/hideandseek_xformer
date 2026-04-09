#!/usr/bin/env python3
"""Detailed diagnosis of differences between state-based and observation-based
team reward computations.

Run from project root:
    python tools/diagnose_team_reward.py --steps 200

Outputs per-step summaries and per (seeker,hider) comparisons.
"""
import sys
import os
import math
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_PATH)

from src.envs.hns28_environment import TeamCosEnv


def compare_step(env, step, eps=1e-6):
    s_out = env._compute_team_reward_state()
    o_out = s_out
    print(f"step={step} cur_step={env.current_step}")
    print("  state_out=", s_out)

    # per pair diagnostics
    for sk in env.seeker_keys:
        for hk in env.hider_keys:
            # state-based metrics
            sid = env.body_ids[sk]
            hid = env.body_ids[hk]
            spos = env.data.xpos[sid][:2]
            hpos = env.data.xpos[hid][:2]
            dx = float(hpos[0] - spos[0])
            dy = float(hpos[1] - spos[1])
            state_dist = math.hypot(dx, dy)
            srot = float(env.data.qpos[env.model.jnt_qposadr[env.qpos_indices[sk]['rot']]])
            dist_with_margin = state_dist + 1e-8
            state_cos_align = (math.cos(srot) * (dx / dist_with_margin)) + (math.sin(srot) * (dy / dist_with_margin))
            state_frontness = max(float(state_cos_align), 0.0)
            state_vis = bool(env._is_vis(spos, srot, hpos, sid, hid))

            # obs-based metrics (from seeker's obs)
            sk_idx = env.agent_keys.index(sk)
            obs_sk = env._get_obs(sk_idx) 
            # preserve env.agent_keys ordering (obs OTHERS uses this order)
            ens = [k for k in env.agent_keys if k != sk]
            try:
                pos = ens.index(hk)
            except ValueError:
                continue
            if pos >= len(env.idx.OTHERS):
                continue
            en_idx = env.idx.OTHERS[pos]
            obs_rel_x = float(obs_sk[en_idx.REL_X])
            obs_rel_y = float(obs_sk[en_idx.REL_Y])
            obs_dist = math.hypot(obs_rel_x, obs_rel_y)
            obs_front = max(obs_rel_x / (obs_dist + 1e-8), 0.0)
            obs_vis_flag = float(obs_sk[en_idx.VISIBLE])
            obs_vis = obs_vis_flag > 0.5

            mismatch = False
            reasons = []
            if state_vis != obs_vis:
                mismatch = True
                reasons.append('vis')
            if abs(state_frontness - obs_front) > 1e-3:
                mismatch = True
                reasons.append('front')
            if abs(state_dist - obs_dist) > 0.05:
                mismatch = True
                reasons.append('dist')

            if mismatch:
                # print index mapping and raw obs values for debugging
                try:
                    rx_idx = en_idx.REL_X 
                    ry_idx = en_idx.REL_Y
                    v_idx = en_idx.VISIBLE
                    rx_val = float(obs_sk[rx_idx])
                    ry_val = float(obs_sk[ry_idx])
                    v_val = float(obs_sk[v_idx])
                except Exception:
                    rx_idx = getattr(en_idx, 'REL_X', None)
                    ry_idx = getattr(en_idx, 'REL_Y', None)
                    v_idx = getattr(en_idx, 'VISIBLE', None)
                    rx_val = obs_sk[rx_idx] if rx_idx is not None and rx_idx < len(obs_sk) else None
                    ry_val = obs_sk[ry_idx] if ry_idx is not None and ry_idx < len(obs_sk) else None
                    v_val = obs_sk[v_idx] if v_idx is not None and v_idx < len(obs_sk) else None

                print(f"   [{sk} -> {hk}] state_vis={state_vis} obs_vis={obs_vis} | state_dist={state_dist:.3f} obs_dist={obs_dist:.3f} | state_front={state_frontness:.3f} obs_front={obs_front:.3f} | reasons={reasons}")
                print(f"      en_idx={{REL_X:{rx_idx},REL_Y:{ry_idx},VISIBLE:{v_idx}}} obs_vals={{REL_X:{rx_val},REL_Y:{ry_val},VISIBLE:{v_val}}}") 


def main(steps=200, start=0):
    env = TeamCosEnv(debug_mode=True)
    obs, info = env.reset()
    i = 0
    # prefer env.current_step as authoritative; stop when it reaches steps
    while True:
        if steps is not None and env.current_step >= steps:
            break
        action = np.zeros(4, dtype=np.float32)
        obs, reward, done, _, info = env.step(action)
        # skip verbose comparison until we reach `start` step
        if env.current_step >= start:
            compare_step(env, i)
        i += 1
        if done:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', '-s', type=int, default=200, help='max env.current_step to run')
    parser.add_argument('--start', type=int, default=0, help='skip verbose output until env.current_step >= START')
    args = parser.parse_args()
    print(f"diagnose_team_reward: running with steps={args.steps} start={args.start}")
    main(steps=args.steps, start=args.start)
