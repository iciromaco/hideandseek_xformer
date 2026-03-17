#!/usr/bin/env python3
import os, sys, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.envs.hns_environment import TeamCosEnv


def run_stats(target, steps=200):
    print(f"=== target={target} steps={steps} ===")
    try:
        env = TeamCosEnv(mode='initial', target=target, n_seekers=1, n_hiders=2, render_mode=None)
    except Exception as e:
        print('Env construction failed:', e)
        return
    counts = {'total':0, 'present':0, 'nonzero':0}
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        counts['total'] += 1
        if isinstance(info, dict):
            if 'dbg_seek_gaze_cos_front_max' in info or 'dbg_seek_gaze_cos_front_dist_max' in info:
                counts['present'] += 1
                v = info.get('dbg_seek_gaze_cos_front_max', None)
                d = info.get('dbg_seek_gaze_cos_front_dist_max', None)
                if (v is not None and float(v) != 0.0) or (d is not None and float(d) != 0.0):
                    counts['nonzero'] += 1
        elif isinstance(info, (list, tuple)):
            for item in info:
                if item is None:
                    continue
                if 'dbg_seek_gaze_cos_front_max' in item or 'dbg_seek_gaze_cos_front_dist_max' in item:
                    counts['present'] += 1
                    v = item.get('dbg_seek_gaze_cos_front_max', None)
                    d = item.get('dbg_seek_gaze_cos_front_dist_max', None)
                    if (v is not None and float(v) != 0.0) or (d is not None and float(d) != 0.0):
                        counts['nonzero'] += 1
    env.close()
    print('total steps:', counts['total'])
    print('present count:', counts['present'])
    print('nonzero count:', counts['nonzero'])
    print('present ratio:', counts['present']/counts['total'])
    print('nonzero ratio:', counts['nonzero']/counts['total'])

if __name__ == '__main__':
    run_stats('seeker', steps=200)
    print('\n')
    run_stats('hider', steps=200)
