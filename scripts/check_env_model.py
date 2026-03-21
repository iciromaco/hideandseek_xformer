#!/usr/bin/env python3
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(mode="initial", target="seeker", n_seekers=1, n_hiders=2, render_mode=None)
print("RUN_PROFILE:", os.environ.get("RUN_PROFILE", ""))
print("override_learnable_policy:", env.override_learnable_policy)
print("_inference_models keys:", list(env._inference_models.keys()))
print("learnable_agent_key:", env.learnable_agent_key)
env.close()
