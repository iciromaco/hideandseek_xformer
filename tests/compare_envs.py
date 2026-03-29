import importlib
import importlib.util
import os
import sys
import traceback
from importlib.machinery import SourceFileLoader
from pprint import pprint

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# load current module (installed in src)
try:
    from src.envs.hns_environment import TeamCosEnv as TeamCosEnvCurrent
except Exception as e:
    print("Failed to import current TeamCosEnv:", e)
    traceback.print_exc()
    raise

# load previous module from dumped file
prev_path = os.path.join(ROOT, "tests", "hns_env_prev.py")
if not os.path.exists(prev_path):
    print("Previous file not found at", prev_path)
    raise SystemExit(2)

loader = SourceFileLoader("hns_env_prev", prev_path)
spec = importlib.util.spec_from_loader(loader.name, loader)
prev_mod = importlib.util.module_from_spec(spec)
loader.exec_module(prev_mod)
TeamCosEnvPrev = getattr(prev_mod, "TeamCosEnv", None)
if TeamCosEnvPrev is None:
    print("Prev module does not contain TeamCosEnv")
    raise SystemExit(3)

print("Both modules loaded:")
print(" - current:", TeamCosEnvCurrent)
print(" - prev:", TeamCosEnvPrev)

# helper to run a single reset+step and capture key outputs
import numpy as np


def sample_run(EnvClass, seed=1234):
    try:
        try:
            env = EnvClass(debug_mode=True, seed=seed)
        except TypeError:
            # EnvClass doesn't accept seed at construction; seed globals and instantiate
            import random

            random.seed(seed)
            np.random.seed(seed)
            try:
                import torch

                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            env = EnvClass(debug_mode=True)
    except Exception as e:
        print(f"Failed to instantiate {EnvClass}:", e)
        raise
    try:
        # env was seeded at construction; call reset without re-seeding
        try:
            obs = env.reset()
        except TypeError:
            obs = env.reset()
        # record copies of key arrays/state before step
        try:
            xfrc_before = None if not hasattr(env.data, "xfrc_applied") else env.data.xfrc_applied.copy()
        except Exception:
            xfrc_before = None
        try:
            qpos_before = None if not hasattr(env.data, "qpos") else env.data.qpos.copy()
        except Exception:
            qpos_before = None
        try:
            qvel_before = None if not hasattr(env.data, "qvel") else env.data.qvel.copy()
        except Exception:
            qvel_before = None
        try:
            # wall_distance helper if available
            if hasattr(env, "vis_engine") and hasattr(env, "body_ids") and hasattr(env, "learnable_agent_key"):
                bid = env.body_ids[env.learnable_agent_key]
                wp = env.data.xpos[bid]
                wall_before = float(env.vis_engine.wall_distance(wp[0], wp[1]))
            else:
                wall_before = None
        except Exception:
            wall_before = None
        # use a fixed action
        action = [0.5, 0.0, 0.0, 0.0]
        try:
            out = env.step(action)
        except Exception as e:
            print(f"step() raised for {EnvClass}:", e)
            raise
        # capture copies of key arrays/state after step
        try:
            xfrc_after = None if not hasattr(env.data, "xfrc_applied") else env.data.xfrc_applied.copy()
        except Exception:
            xfrc_after = None
        try:
            qpos_after = None if not hasattr(env.data, "qpos") else env.data.qpos.copy()
        except Exception:
            qpos_after = None
        try:
            qvel_after = None if not hasattr(env.data, "qvel") else env.data.qvel.copy()
        except Exception:
            qvel_after = None
        try:
            if hasattr(env, "vis_engine") and hasattr(env, "body_ids") and hasattr(env, "learnable_agent_key"):
                bid = env.body_ids[env.learnable_agent_key]
                wp = env.data.xpos[bid]
                wall_after = float(env.vis_engine.wall_distance(wp[0], wp[1]))
            else:
                wall_after = None
        except Exception:
            wall_after = None
        # tidy
        try:
            env.close()
        except Exception:
            pass
        return {
            "obs": obs,
            "out": out,
            "xfrc_before": xfrc_before,
            "xfrc_after": xfrc_after,
            "qpos_before": qpos_before,
            "qpos_after": qpos_after,
            "qvel_before": qvel_before,
            "qvel_after": qvel_after,
            "wall_before": wall_before,
            "wall_after": wall_after,
        }

    except Exception:
        traceback.print_exc()
        raise


print("Running prev env")
prev_res = sample_run(TeamCosEnvPrev)
print("Running current env")
curr_res = sample_run(TeamCosEnvCurrent)

print("\n--- Comparison Summary ---")
# compare obs shapes
try:
    print("obs prev type:", type(prev_res["obs"]))
    print("obs curr type:", type(curr_res["obs"]))
except Exception:
    pass

# compare out (step return)
print("\nStep return (prev):")
try:
    pprint(prev_res["out"])
except Exception:
    print("unable to pprint prev out")
print("\nStep return (curr):")
try:
    pprint(curr_res["out"])
except Exception:
    print("unable to pprint curr out")

# compare key info dict fields if present
try:
    prev_info = prev_res["out"][-1] if isinstance(prev_res["out"], tuple) else None
    curr_info = curr_res["out"][-1] if isinstance(curr_res["out"], tuple) else None
    print("\nInfo keys difference:")
    if prev_info is not None and curr_info is not None and isinstance(prev_info, dict) and isinstance(curr_info, dict):
        pk = set(prev_info.keys())
        ck = set(curr_info.keys())
        print("only_in_prev =", sorted(list(pk - ck)))
        print("only_in_curr =", sorted(list(ck - pk)))
        common = pk & ck
        for k in sorted(list(common)):
            pv = prev_info.get(k)
            cv = curr_info.get(k)
            if isinstance(pv, (float, int)) and isinstance(cv, (float, int)):
                if abs(float(pv) - float(cv)) > 1e-6:
                    print(f"DIFF {k}: prev={pv} curr={cv}")
            else:
                if pv != cv:
                    print(f"DIFF {k}: prev={pv} curr={cv}")
    else:
        print("info not present in one of runs")
except Exception:
    traceback.print_exc()

# compare xfrc_applied arrays
print("\nCompare xfrc_applied before/after (prev vs curr):")
try:
    print("prev before:", None if prev_res["xfrc_before"] is None else prev_res["xfrc_before"].shape)
    print("curr before:", None if curr_res["xfrc_before"] is None else curr_res["xfrc_before"].shape)
    print("prev after:", None if prev_res["xfrc_after"] is None else prev_res["xfrc_after"].shape)
    print("curr after:", None if curr_res["xfrc_after"] is None else curr_res["xfrc_after"].shape)
    if prev_res["xfrc_before"] is not None and curr_res["xfrc_before"] is not None:
        # compare sums
        import numpy as np

        pb = np.asarray(prev_res["xfrc_before"])
        cb = np.asarray(curr_res["xfrc_before"])
        pa = np.asarray(prev_res["xfrc_after"])
        ca = np.asarray(curr_res["xfrc_after"])
        print("sum(prev delta) =", float(np.nansum(pa - pb)))
        print("sum(curr delta) =", float(np.nansum(ca - cb)))
        diff = np.nansum(np.abs((pa - pb) - (ca - cb)))
        print("L1 diff of applied delta =", float(diff))
except Exception:
    traceback.print_exc()

print("\n--- Done")
