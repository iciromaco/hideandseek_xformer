#!/usr/bin/env python3
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.envs.hns_environment import TeamCosEnv


def inspect_target(target):
    print("=== target =", target, "===")
    try:
        env = TeamCosEnv(mode="initial", target=target, n_seekers=1, n_hiders=2, render_mode=None)
    except Exception as e:
        print("Env construction failed:", e)
        return
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    print("info type:", type(info))
    if isinstance(info, dict):
        keys = list(info.keys())
        print("dict keys:", keys)
        for k in keys:
            v = info[k]
            print(
                f"key={k} type={type(v)} sample=",
                (v if isinstance(v, (int, float, str)) else (str(type(v)))),
            )
    elif isinstance(info, (list, tuple)):
        print("list length:", len(info))
        for i, item in enumerate(info):
            print("--- env", i, "---")
            if item is None:
                print(" None")
                continue
            print(" keys:", list(item.keys()))
            for k, v in item.items():
                print(
                    f"  {k}: type={type(v)} val=",
                    (v if isinstance(v, (int, float, str)) else str(type(v))),
                )
    else:
        print("Unknown info format:", info)
    env.close()


if __name__ == "__main__":
    inspect_target("seeker")
    print("\n")
    inspect_target("hider")
