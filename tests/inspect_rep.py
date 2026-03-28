import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np

from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=True)
print("instantiated")
obs = env.reset()
print("reset done")
learnable_agent_body_id = env.body_ids[env.learnable_agent_key]
pos = env.data.xpos[learnable_agent_body_id]
last = env.last_debug_ctrl.get(env.learnable_agent_key, (0.0, 0.0))[0]
try:
    before = env.data.xfrc_applied.copy()
except Exception:
    before = None
rep = env._compute_and_apply_wall_repulsion(learnable_agent_body_id, pos, last, 0.45)
try:
    after = env.data.xfrc_applied.copy()
except Exception:
    after = None
print("rep:", rep)
print("sum before:", None if before is None else before.sum())
print("sum after :", None if after is None else after.sum())
print("delta sum:", None if before is None or after is None else (after - before).sum())
print("delta L1 :", None if before is None or after is None else (abs(after - before)).sum())
print("done")
