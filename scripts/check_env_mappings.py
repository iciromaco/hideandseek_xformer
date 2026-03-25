#!/usr/bin/env python3
"""Dump environment model mappings to help debug geom/actuator index mismatches.

Usage:
  python scripts/check_env_mappings.py
  or
  uv run mjpython scripts/check_env_mappings.py
"""

import inspect
import os
import sys


def main():
    # ensure repo root on path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    TeamCosEnv = None
    try:
        # try package import used in repo
        from src.envs.hns_environment import TeamCosEnv
    except Exception:
        try:
            from envs.hns_environment import TeamCosEnv
        except Exception as exc:
            print("Failed to import TeamCosEnv:", exc)
            raise

    print("TeamCosEnv __init__ signature:", inspect.signature(TeamCosEnv.__init__))

    # Try a few sensible constructor patterns
    env = None
    tried = []
    for args in [
        {"mode": "train", "target": "seeker"},
        {"mode": "debug", "target": "seeker"},
        ("train", "seeker"),
        (),
    ]:
        try:
            if isinstance(args, dict):
                env = TeamCosEnv(**args)
            else:
                env = TeamCosEnv(*args)
            print("Created env with:", args)
            break
        except Exception as e:
            tried.append((args, str(e)))

    if env is None:
        print("Failed to construct TeamCosEnv. Attempts:")
        for t, e in tried:
            print("  tried:", t, " -> ", e)
        return 1

    # Dump common mappings if present
    def safe_print(name, getter):
        try:
            val = getter()
            print(f"{name}:\n", val)
        except Exception as exc:
            print(f"{name}: <error> {exc}")

    safe_print("agent_keys", lambda: getattr(env, "agent_keys", None))
    safe_print("learnable_agent_key", lambda: getattr(env, "learnable_agent_key", None))
    safe_print("body_ids (dict)", lambda: getattr(env, "body_ids", None))
    safe_print("actuator_ids (dict)", lambda: getattr(env, "actuator_ids", None))
    safe_print("qpos_indices (dict)", lambda: getattr(env, "qpos_indices", None))
    safe_print(
        "model.ngeom/model.nbody/model.nu/model.njnt", lambda: (getattr(env.model, "ngeom", None), getattr(env.model, "nbody", None), getattr(env.model, "nu", None), getattr(env.model, "njnt", None))
    )

    # Attempt to list geom names and their body attachments
    try:
        ngeom = int(env.model.ngeom)
        names = []
        for gid in range(ngeom):
            try:
                gname = env.model.geom_names[gid]
            except Exception:
                try:
                    gname = env.model._geom_name[gid]
                except Exception:
                    gname = f"geom#{gid}"
            try:
                body = int(env.model.geom_bodyid[gid])
            except Exception:
                body = None
            names.append((gid, gname, body))
        print("geoms (gid, name, body):")
        for item in names[:200]:
            print(" ", item)
    except Exception as exc:
        print("Failed to list geoms:", exc)

    # List geoms attached to each agent body
    try:
        print("\nPer-agent body -> geoms")
        for ak in getattr(env, "agent_keys", []):
            bid = env.body_ids.get(ak, None)
            print(f"agent {ak} body_id={bid}")
            if bid is None:
                continue
            attached = [g for g in names if g[2] == bid]
            for g in attached:
                print("   ", g)
    except Exception:
        pass

    # dump actuator info
    try:
        print("\nActuators (name -> id):")
        for k, v in sorted(getattr(env, "actuator_ids", {}).items()):
            print(" ", k, "->", v)
    except Exception:
        pass

    # joints / qpos mapping
    try:
        print("\nqpos_indices for learnable agent:")
        print(env.qpos_indices.get(env.learnable_agent_key, {}))
        print("model.jnt_qposadr (first 100):", list(env.model.jnt_qposadr[:100]))
    except Exception:
        pass

    # quick check: actuator names that include '_fwd' should point to expected actuator ids
    try:
        print("\nActuators referencing agent forward channels:")
        for ak in getattr(env, "agent_keys", []):
            name = f"{ak}_fwd"
            aid = env.actuator_ids.get(name, None)
            print(" ", name, "->", aid)
    except Exception:
        pass

    # keep env alive briefly so resources are not immediately freed
    try:
        env.close()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
