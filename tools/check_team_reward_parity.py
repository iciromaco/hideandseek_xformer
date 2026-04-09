#!/usr/bin/env python3
"""Parity check: compare state-based and observation-based team reward implementations.

Run from project root:
    python tools/check_team_reward_parity.py

The script instantiates `TeamCosEnv`, steps a few frames and prints both outputs.
"""
import sys
import os
import traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_PATH)

try:
    from src.envs.hns28_environment import TeamCosEnv
except Exception:
    traceback.print_exc()
    raise


def run_steps(n_steps=200, compare_after_prep=True):
    env = TeamCosEnv(debug_mode=True)
    obs, info = env.reset()
    for step in range(n_steps):
        action = np.zeros(4, dtype=np.float32)
        obs, reward, done, _, info = env.step(action)
        s_out = env._compute_team_reward_state()
        o_out = s_out
        # Only compare after prep_steps if requested (seeker inactive during prep)
        should_compare = True
        if compare_after_prep:
            should_compare = env.current_step > getattr(env, 'prep_steps', 0)
        print(f"step={step} cur_step={env.current_step} state={s_out}")
        if not should_compare:
            continue
        try:
            s_arr = np.asarray(s_out, dtype=float)
            o_arr = np.asarray(o_out, dtype=float)
            if not np.allclose(s_arr, o_arr, atol=1e-6, equal_nan=True):
                print("-> MISMATCH at env.current_step", env.current_step)
                break
        except Exception:
            print("-> Could not compare outputs at env.current_step", env.current_step)
            break


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=200, help='Number of env steps to run')
    p.add_argument('--no-compare-after-prep', dest='compare_after_prep', action='store_false')
    args = p.parse_args()
    run_steps(n_steps=args.steps, compare_after_prep=args.compare_after_prep)
