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
    print('gaze metrics have been removed from the environment info; nothing to collect.')
    env.close()

if __name__ == '__main__':
    run_stats('seeker', steps=200)
    print('\n')
    run_stats('hider', steps=200)
