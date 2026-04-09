#!/usr/bin/env python3
"""Print world z of learnable agent geoms and its z-joint qpos."""
from __future__ import annotations
import sys
from importlib import import_module

sys.path.insert(0, '.')
from src.envs.hns_environment import TeamCosEnv

def main():
    env = TeamCosEnv()
    env.reset()
    model = env.model
    data = env.data
    ak = env.learnable_agent_key
    print('learnable agent key:', ak)
    # collect geom ids
    names = []
    for i in range(int(model.ngeom)):
        try:
            n = model.geom_id2name(i)
        except Exception:
            # fallback
            n = None
        names.append((i, n))
    agent_geom_ids = [(i, n) for i, n in names if n is not None and n.startswith(ak + '_')]
    print('agent geom ids:', agent_geom_ids)
    if not agent_geom_ids:
        print('no agent geoms found by prefix; dumping all geoms:')
        for i, n in names:
            print(i, n)
    else:
        for gid, name in agent_geom_ids:
            print(name, 'world z=', float(data.geom_xpos[gid, 2]))
    jz = env.qpos_indices[ak]['z']
    qadr = model.jnt_qposadr[jz]
    print('z joint qpos:', float(data.qpos[qadr]))
    # print learnable agent body world z
    bid = env.body_ids.get(ak)
    if bid is not None:
        print('agent body world z:', float(data.xpos[bid, 2]))
    # print all geom world zs (index only)
    print('\nAll geom world z (index: z):')
    for i in range(int(model.ngeom)):
        try:
            z = float(data.geom_xpos[i, 2])
        except Exception:
            z = None
        print(i, z)

if __name__ == '__main__':
    main()
