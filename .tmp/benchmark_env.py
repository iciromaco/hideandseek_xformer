import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from src.envs.hns_environment import TeamCosEnv


def run(seed=None, steps=200, warmup=10):
    env = TeamCosEnv(debug_mode=False, seed=seed)
    obs = env.reset()
    act_dim = env.action_space.shape[0]
    # warmup
    for _ in range(warmup):
        a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
        res = env.step(a)
        if isinstance(res, tuple):
            if len(res) == 4:
                obs, r, d, info = res
            elif len(res) == 5:
                obs, r, _, d, info = res
            else:
                # fallback: assume last element is info
                obs = res[0]
                r = res[1] if len(res) > 1 else 0.0
                d = res[2] if len(res) > 2 else False
                info = res[-1]
        else:
            # non-tuple return
            obs, r, d, info = res
    # benchmark
    t0 = time.perf_counter()
    for i in range(steps):
        a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
        res = env.step(a)
        if isinstance(res, tuple):
            if len(res) == 4:
                obs, r, d, info = res
            elif len(res) == 5:
                obs, r, _, d, info = res
            else:
                # fallback: assume last element is info
                obs = res[0]
                r = res[1] if len(res) > 1 else 0.0
                d = res[2] if len(res) > 2 else False
                info = res[-1]
        else:
            # non-tuple return
            obs, r, d, info = res
    t1 = time.perf_counter()
    total = t1 - t0
    sps = steps / total if total > 0 else float("inf")
    print(f"Benchmark: steps={steps} total_time={total:.4f}s SPS={sps:.2f}")


if __name__ == "__main__":
    run(seed=123, steps=200, warmup=5)
