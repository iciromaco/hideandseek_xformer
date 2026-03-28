import os
import sys

import numpy as np

# ensure project src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from envs.hns_environment_flip import FlippedForwardTeamCosEnv


def main():
    env = FlippedForwardTeamCosEnv(debug_mode=True, target="seeker")
    print("Env created:", type(env).__name__)
    obs = env.reset()
    print("Reset done")

    a = np.array([0.25, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(5):
        obs, r, term, done, info = env.step(a)
        print(
            "Step {}: applied_forward_model={}, applied_forward={}, agent_vx={}".format(
                i,
                info.get("applied_forward_model"),
                info.get("applied_forward"),
                info.get("agent_vx"),
            )
        )
    print("Smoke test finished")


if __name__ == "__main__":
    main()
