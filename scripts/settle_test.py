#!/usr/bin/env python3
"""Run a settling test: compare agent bottom z and contacts before/after stepping."""
import sys
import time
import numpy as np

sys.path.insert(0, '.')
from src.envs.hns_environment import TeamCosEnv


def geom_world_bottom_z(model, data, gid):
    # Prefer fromto endpoints
    try:
        ft = np.asarray(model.geom_fromto[gid])
        if ft is not None and ft.size >= 6 and not np.allclose(ft, 0.0):
            local_from = ft[0:3].astype(float)
            local_to = ft[3:6].astype(float)
            xmat = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
            wp_from = np.asarray(data.geom_xpos[gid]) + xmat.dot(local_from)
            wp_to = np.asarray(data.geom_xpos[gid]) + xmat.dot(local_to)
            return float(min(wp_from[2], wp_to[2]))
    except Exception:
        pass
    try:
        sz = np.asarray(model.geom_size[gid])
        r = float(sz[0]) if sz.size >= 1 else 0.0
        halflen = float(sz[1]) if sz.size >= 2 else 0.0
    except Exception:
        r = 0.0
        halflen = 0.0
    try:
        xmat = np.asarray(data.geom_xmat[gid]).reshape(3, 3)
        axis_z = float(xmat[2, 2])
    except Exception:
        axis_z = 1.0
    mid_z = float(data.geom_xpos[gid, 2])
    bottom = mid_z - abs(axis_z) * halflen - r
    return bottom


def print_agent_bottoms(env):
    m, d = env.model, env.data
    for ak in env.agent_keys:
        bid = env.body_ids.get(ak)
        bottoms = []
        if bid is None:
            print(ak, 'no body id')
            continue
        gadr = int(m.body_geomadr[bid])
        gnum = int(m.body_geomnum[bid])
        for gi in range(gadr, gadr + gnum):
            b = geom_world_bottom_z(m, d, gi)
            bottoms.append((gi, b))
        lowest = min(b for _, b in bottoms) if bottoms else None
        print(f"{ak}: lowest bottom z = {lowest:.6f}; geom bottoms: {bottoms}")


def print_contacts(env):
    sim = getattr(env, 'sim', None)
    data = getattr(env, 'data')
    if sim is None:
        # use data.ncon if available
        ncon = int(getattr(data, 'ncon', 0))
    else:
        ncon = int(getattr(sim.data, 'ncon', 0))
    print('ncon:', ncon)
    # print contact pairs and dist if available
    for i in range(ncon):
        con = None
        try:
            con = (sim.data.contact[i] if sim is not None else data.contact[i])
        except Exception:
            continue
        g1, g2 = int(con.geom1), int(con.geom2)
        pos = tuple(getattr(con, 'pos', (None, None, None)))
        dist = float(getattr(con, 'dist', 0.0))
        print(f"con {i}: geom{g1} <-> geom{g2} dist={dist:.6f} pos={pos}")


def main():
    env = TeamCosEnv()
    print('AGENT_Z_MIN, AGENT_Z_MAX:', env.AGENT_Z_MIN, env.AGENT_Z_MAX)
    env.reset()
    print('\n-- after reset (no stepping) --')
    print_agent_bottoms(env)
    print_contacts(env)

    steps = 50
    print(f'\nStepping {steps} times to let solver settle...')
    for i in range(steps):
        try:
            # use mj_step loop within env.step to maintain constraints; call a simplified step
            # perform a minimal step: no control
            env.data.ctrl[:] = 0
            env.model.step = getattr(env.model, 'step', None)
            # run a raw mj_step
            import mujoco
            mujoco.mj_step(env.model, env.data)
        except Exception:
            break

    print('\n-- after stepping --')
    print_agent_bottoms(env)
    print_contacts(env)


if __name__ == '__main__':
    main()
