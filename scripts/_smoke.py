import sys
import traceback

import numpy as np

try:
    from src.envs.hns_environment import TeamCosEnv
except Exception:
    # try package-style import
    from envs.hns_environment import TeamCosEnv

print("Creating env...")
# minimal config: 1 seeker, 1 hider, no boxes/ramps to keep light
env = TeamCosEnv(mode="initial", target="hider", n_seekers=1, n_hiders=1, n_boxes=0, n_ramps=0, debug_mode=True, action_repeat=1)
print("Resetting...")
obs, info = env.reset()
print("Reset OK. obs.shape=", getattr(obs, "shape", type(obs)))
print("Info keys:", list(info.keys()))
# prepare zero action for learnable agent (4-dim)
action = np.zeros(4, dtype=np.float32)
print("Stepping...")
res = env.step(action)
print("Step result:", type(res))
# print summary
try:
    if isinstance(res, tuple):
        print("Step tuple lengths:", len(res))
        # show types and small samples
        for i, v in enumerate(res):
            if isinstance(v, np.ndarray):
                print(f"  item {i}: ndarray shape={v.shape} dtype={v.dtype}")
            else:
                print(f"  item {i}: {type(v)}")
except Exception:
    traceback.print_exc()

print("Closing env")
try:
    env.close()
except Exception:
    pass
print("SMOKE_OK")
