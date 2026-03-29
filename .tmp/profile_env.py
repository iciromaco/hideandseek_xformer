import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.getcwd())
import numpy as np

from src.envs.hns_environment import TeamCosEnv


def bench():
    env = TeamCosEnv(debug_mode=False, seed=123)
    obs = env.reset()
    act_dim = env.action_space.shape[0]
    steps = 200
    pr = cProfile.Profile()
    pr.enable()
    for i in range(steps):
        a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
        res = env.step(a)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    print(s.getvalue())


if __name__ == "__main__":
    bench()
