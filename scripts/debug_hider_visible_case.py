#!/usr/bin/env python3
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.envs.hns28_environment import TeamCosEnv
from src.agents.scripted_agents import RuleBasedHider
from src.core.constants import P_SCALE


def analyze_visible(obs, idx):
    # REL values returned by env._get_obs are already in world units (meters).
    # Do not multiply by P_SCALE here; use raw values.
    p_scale = 1.0
    visible = [en for en in idx.OTHERS if obs[en.VISIBLE] > 0.5]
    if not visible:
        return None
    # find nearest
    nearest_tx, nearest_ty = 0.0, 0.0
    nearest_dist = float('inf')
    for en in visible:
        tx = float(obs[en.REL_X]) * p_scale
        ty = float(obs[en.REL_Y]) * p_scale
        d = math.hypot(tx, ty)
        if d < nearest_dist:
            nearest_dist = d
            nearest_tx, nearest_ty = tx, ty
    angle_to_seeker = math.atan2(nearest_ty, nearest_tx)
    return {
        'nearest_tx': nearest_tx,
        'nearest_ty': nearest_ty,
        'nearest_dist': nearest_dist,
        'angle_to_seeker': angle_to_seeker,
    }


def main():
    found = False
    attempts = 0
    max_attempts = 200
    while not found and attempts < max_attempts:
        attempts += 1
        env = TeamCosEnv(n_seekers=1, n_hiders=1, debug_mode=False, render_mode=None)
        obs, info = env.reset()
        h_key = env.hider_keys[0]
        h_idx = env.agent_keys.index(h_key)
        hider = env.npcs[h_key]
        # check initial and a few steps
        for step in range(0, 6):
            o = env._get_obs(h_idx)
            analysis = analyze_visible(o, env.idx)
            if analysis is not None:
                # compute hider internal values similarly to get_action
                lidar_raw = o[env.idx.LIDAR] * env.L_SCALE if hasattr(env, 'L_SCALE') else o[env.idx.LIDAR]
                cur_rot = o[env.idx.SELF.ROT] * env.R_SCALE if hasattr(env, 'R_SCALE') else o[env.idx.SELF.ROT]
                front_min = np.min(lidar_raw[env.idx.LIDAR_FRONT_IDX])
                back_min = np.min(lidar_raw[env.idx.LIDAR_BACK_IDX])
                l_gap = np.sum(lidar_raw[env.idx.LIDAR_LEFT_IDX])
                r_gap = np.sum(lidar_raw[env.idx.LIDAR_RIGHT_IDX])
                speed_scale = np.clip((front_min - 0.45) / 0.8, 0.15, 1.0)
                angle_to_seeker = analysis['angle_to_seeker']
                escape_base = (angle_to_seeker + math.pi + math.pi) % (2*math.pi) - math.pi
                side_bias = 1.2 if l_gap > r_gap else -1.2
                target_angle = (escape_base + side_bias + math.pi) % (2*math.pi) - math.pi
                fwd_val = math.cos(target_angle)
                manual_fwd = 0.0 if (fwd_val < 0 and back_min < 0.5) else 0.8 * fwd_val * speed_scale

                action = hider.get_action(o, env.idx)

                print('Found visible case on attempt', attempts, 'step', step)
                print('obs nearest:', analysis)
                print('front_min, back_min, l_gap, r_gap:', front_min, back_min, l_gap, r_gap)
                print('escape_base, side_bias, target_angle:', escape_base, side_bias, target_angle)
                print('manual_fwd, hider_action:', manual_fwd, action)
                found = True
                break
            if step < 5:
                env.step(np.zeros(4))
        # avoid keeping env around
        del env
    if not found:
        print('No visible cases found in', max_attempts, 'attempts')

if __name__ == '__main__':
    main()
