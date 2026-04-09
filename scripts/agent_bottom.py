#!/usr/bin/env python3
"""Compute agent COM z and lowest geom bottom z for the learnable agent."""
import sys
import numpy as np

sys.path.insert(0, '.')
from src.envs.hns_environment import TeamCosEnv


def geom_world_bottom_z(model, data, gid):
    # Try fromto endpoints first
    try:
        if hasattr(model, 'geom_fromto'):
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

    # Fallback: use geom_size (radius, half-length for capsules)
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


def main():
    env = TeamCosEnv()
    env.reset()
    model = env.model
    data = env.data
    ak = env.learnable_agent_key

    # COM (body origin) z
    bid = env.body_ids.get(ak)
    com_z = float(data.xpos[bid, 2]) if bid is not None else None

    # body geoms
    bottoms = []
    if bid is not None:
        gadr = int(model.body_geomadr[bid])
        gnum = int(model.body_geomnum[bid])
        for gi in range(gadr, gadr + gnum):
            b = geom_world_bottom_z(model, data, gi)
            bottoms.append((gi, b))

    print(f"agent={ak} body_id={bid} COM_z={com_z}")
    if bottoms:
        for gi, b in bottoms:
            # try to get geom name
            try:
                gname = model.geom_id2name(gi)
            except Exception:
                gname = f"geom#{gi}"
            print(f"  geom {gi} ({gname}) bottom_z={b}")
        lowest = min(b for _, b in bottoms)
        print(f"lowest bottom z = {lowest}")
    else:
        print("no geoms for agent body found")


if __name__ == '__main__':
    main()
