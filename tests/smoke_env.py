import os
import sys
import traceback
from pprint import pprint

# add project root so 'src' can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.envs.hns_environment import TeamCosEnv
except Exception as e:
    print("Failed to import TeamCosEnv:", e)
    traceback.print_exc()
    raise

try:
    env = TeamCosEnv(debug_mode=True)
    print("Env instantiated")
    try:
        res = env.reset()
        print("reset() returned:")
        pprint(res)
    except Exception as e:
        print("reset() raised:", e)
        traceback.print_exc()

    # prepare a default action
    action = None
    try:
        if hasattr(env, "action_space") and env.action_space is not None:
            action = env.action_space.sample()
        else:
            action = [0.0, 0.0, 0.0, 0.0]
    except Exception:
        action = [0.0, 0.0, 0.0, 0.0]

    print("Using action:", action)
    try:
        out = env.step(action)
        print("step() returned type:", type(out))
        pprint(out)
    except Exception as e:
        print("step() raised:", e)
        traceback.print_exc()

    try:
        env.close()
    except Exception:
        pass

    print("SMOKE TEST DONE")
except Exception:
    traceback.print_exc()
    sys.exit(2)
