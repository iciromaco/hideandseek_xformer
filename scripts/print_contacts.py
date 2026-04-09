#!/usr/bin/env python3
"""Print contact info between agents and the floor for quick debugging.

Usage: python3 scripts/print_contacts.py
"""
import sys

def geom_name(model, idx):
    # Try a few ways to get a human-readable geom name
    try:
        return model.geom_id2name(idx)
    except Exception:
        names = getattr(model, "geom_names", None)
        if names is not None:
            n = names[idx]
            return n.decode() if isinstance(n, bytes) else n
    return f"geom#{idx}"


def find_floor_id(model):
    try:
        return model.geom_name2id("floor")
    except Exception:
        for i, n in enumerate(getattr(model, "geom_names", [])):
            name = n.decode() if isinstance(n, bytes) else n
            if name == "floor":
                return i
    return None


def print_agent_floor_contacts(env, agent_prefixes=None):
    try:
        env.reset()
    except Exception as e:
        print("env.reset() failed:", e)
        return

    model = getattr(env, "model", None)
    sim = getattr(env, "sim", None)
    data = getattr(env, "data", None)
    if model is None:
        print("env has no model attribute")
        return
    if sim is None:
        if data is None:
            print("env has neither sim nor data attributes")
            return
        # shim so code can reference sim.data and sim.forward()
        class _Shim:
            def __init__(self, data):
                self.data = data
            def forward(self):
                return
        sim = _Shim(data)

    if agent_prefixes is None:
        agent_prefixes = getattr(env, "agent_keys", None) or getattr(env, "seeker_keys", None) or []

    floor_id = find_floor_id(model)
    if floor_id is None:
        # Fallback: treat geom#0 as floor (observed in some builds)
        floor_id = 0
        print("floor not found by name; assuming geom#0 is floor (id=0)")
    else:
        print("floor_id:", floor_id)

    sim.forward()
    ncon = int(getattr(sim.data, "ncon", 0))
    # Only print per-agent floor contacts (suppress full contact table)
    print("ncon (total):", ncon)

    # Check per-agent geoms vs floor
    for ak in agent_prefixes:
        candidates = [f"{ak}_capsule", f"{ak}_btm", f"{ak}_nose", f"{ak}_tail"]
        # build name->id map
        ids = {}
        for i, n in enumerate(getattr(model, "geom_names", [])):
            name = n.decode() if isinstance(n, bytes) else n
            if name in candidates:
                ids[name] = i
        for gname in candidates:
            gid = ids.get(gname)
            if gid is None:
                continue
            print(f"---- checking {gname} (geom id {gid}) ----")
            found = False
            for i in range(ncon):
                con = sim.data.contact[i]
                g1, g2 = int(con.geom1), int(con.geom2)
                if (g1 == floor_id and g2 == gid) or (g2 == floor_id and g1 == gid):
                    found = True
                    pos = tuple(getattr(con, "pos", (None, None, None)))
                    dist = float(getattr(con, "dist", 0.0))
                    print(f"  CONTACT floor <-> {gname}: dist={dist:.6f} pos={pos}")
            if not found:
                print(f"  NO CONTACT with floor for {gname}")


def main():
    # Try importing common env classes
    env = None
    tried = []
    candidates = [
        "src.envs.hns_environment.TeamCosEnv",
        "src.envs.hns28_environment.TeamCosEnv",
        "src.envs.hns_environment.HnsEnv",
    ]
    for cand in candidates:
        modname, clsname = cand.rsplit(".", 1)
        tried.append(cand)
        try:
            mod = __import__(modname, fromlist=[clsname])
            cls = getattr(mod, clsname)
            env = cls()
            print("Instantiated", cand)
            break
        except Exception as e:
            print("Could not instantiate", cand, e)

    if env is None:
        print("Failed to construct env. Tried:", tried)
        sys.exit(2)

    print_agent_floor_contacts(env, agent_prefixes=[getattr(env, "learnable_agent_key", None)])


if __name__ == "__main__":
    main()
