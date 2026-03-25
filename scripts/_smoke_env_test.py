# smoke test: construct TeamCosEnv, reset, take one zero action step
import os
import sys
import traceback

import numpy as np

# ensure repo root is on sys.path so main27_train_final can be imported
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from main27_train_final import ENV_CONFIG, MODE, _resolve_runtime_target, build_env

    rt = _resolve_runtime_target()
    print("MODE=", MODE, "runtime_target=", rt)
    cfg = dict(ENV_CONFIG)
    cfg["num_envs"] = 1
    env = build_env(MODE, rt, cfg, render_mode=None)
    print("Env constructed:", type(env))
    obs, info = env.reset()
    print("Reset OK. obs type:", type(obs))
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    step_res = env.step(action)
    print("Step returned len:", len(step_res))
    env.close()
except Exception:
    traceback.print_exc()
