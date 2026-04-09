#!/usr/bin/env python3
import os
import sys
import math
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.envs.hns28_environment import TeamCosEnv
from src.agents.scripted_agents import RuleBasedHider

MAX_ATTEMPTS = 500
TRACE_FRAMES = 10


def find_and_trace():
    attempts = 0
    while attempts < MAX_ATTEMPTS:
        attempts += 1
        env = TeamCosEnv(n_seekers=3, n_hiders=3, debug_mode=False, render_mode=None)
        obs, info = env.reset()
        h_key = env.hider_keys[0]
        h_idx = env.agent_keys.index(h_key)
        hider = env.npcs[h_key]

        # prepare `ens` ordering and inform ObsIdx about SEEKERS/HIDERS
        ens = [k for k in env.agent_keys if k != h_key]
        print(ens)
        if h_key.startswith("s"):
            ens.sort(key=lambda k: (0 if k.startswith("h") else 1, k))
        else:
            ens.sort(key=lambda k: (0 if k.startswith("s") else 1, k))
        try:
            env.idx.set_others_keys(ens)
        except Exception:
            pass

        # check initial and a few early steps for visibility (use SEEKERS only)
        for step0 in range(0, 4):
            o = env._get_obs(h_idx)
            visible = [en for en in env.idx.SEEKERS if o[en.VISIBLE] > 0.5]
            if visible:
                print(f"=== Found visible case: attempt={attempts} step={step0} ===")
                trace(env, h_key, h_idx, TRACE_FRAMES, ens)
                return True
            # advance a frame
            env.step(np.zeros(4))
        del env
    print(f"No visible case found in {MAX_ATTEMPTS} attempts")
    return False


def trace(env, h_key, h_idx, frames, ens=None):
    hider = env.npcs[h_key]
    print('Header: step | rel=(tx,ty) | dist | visible | front_min | back_min | l_gap | r_gap | manual_fwd | action')
    for f in range(frames):
        o = env._get_obs(h_idx)
        # consider only SEEKERS for hider visibility
        visible = [en for en in env.idx.SEEKERS if o[en.VISIBLE] > 0.5]
        if visible:
            # nearest
            nearest_tx = nearest_ty = 0.0
            nearest_dist = float('inf')
            for en in visible:
                tx = float(o[en.REL_X])
                ty = float(o[en.REL_Y])
                d = math.hypot(tx, ty)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_tx, nearest_ty = tx, ty
            # lidar and geometry
            lidar_raw = o[env.idx.LIDAR] * 1.0
            front_min = float(np.min(lidar_raw[env.idx.LIDAR_FRONT_IDX]))
            back_min = float(np.min(lidar_raw[env.idx.LIDAR_BACK_IDX]))
            l_gap = float(np.sum(lidar_raw[env.idx.LIDAR_LEFT_IDX]))
            r_gap = float(np.sum(lidar_raw[env.idx.LIDAR_RIGHT_IDX]))
            # internal manual computations
            angle_to_seeker = math.atan2(nearest_ty, nearest_tx)
            escape_base = (angle_to_seeker + math.pi + math.pi) % (2*math.pi) - math.pi
            side_bias = 1.2 if l_gap > r_gap else -1.2
            target_angle = (escape_base + side_bias + math.pi) % (2*math.pi) - math.pi
            speed_scale = np.clip((front_min - 0.45) / 0.8, 0.15, 1.0)
            fwd_val = math.cos(target_angle)
            manual_fwd = 0.0 if (fwd_val < 0 and back_min < 0.5) else 0.8 * fwd_val * speed_scale
            # pass agent_key and ens to get_action so scripted agent can filter correctly
            action = hider.get_action(o, env.idx, agent_key=h_key, ens=ens)
            # agent world positions
            seeker_key = env.seeker_keys[0]
            s_pos = env.data.xpos[env.body_ids[seeker_key]][:2]
            h_pos = env.data.xpos[env.body_ids[h_key]][:2]
            print(f"{f:2d} | rel=({nearest_tx:.3f},{nearest_ty:.3f}) | d={nearest_dist:.3f} | V=1 | fm={front_min:.3f} bm={back_min:.3f} l={l_gap:.3f} r={r_gap:.3f} | mfwd={manual_fwd:.3f} | act=({action[0]:.3f},{action[1]:.3f}) | s_pos=({s_pos[0]:.3f},{s_pos[1]:.3f}) h_pos=({h_pos[0]:.3f},{h_pos[1]:.3f})")
        else:
            print(f"{f:2d} | V=0 (no visible) - skipping detailed log")
        env.step(np.zeros(4))
        time.sleep(0.01)


if __name__ == '__main__':
    find_and_trace()
