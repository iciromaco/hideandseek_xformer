#!/usr/bin/env python3
"""Detailed diagnostics to find why agents appear to float.

Prints:
- env.AGENT_Z_MIN / AGENT_Z_MAX
- the generated XML fragment for agents
- for each agent: joint ranges for *_z, qpos adr and initial qpos
- for each geom on agent body: geom type, size/fromto, contype/conaffinity, world z
"""
import sys
import re
import os
import numpy as np

sys.path.insert(0, '.')
from src.envs.hns_environment import TeamCosEnv


def dump_agent_xml(xml, agent_key):
    # simple search for the agent body block
    m = re.search(rf"<body name=\"{agent_key}_anchor\".*?<\/body>\s*<\/body>", xml, re.S)
    if m:
        print(f"--- XML fragment for {agent_key} ---")
        print(m.group(0))
    else:
        print(f"(no xml fragment found for {agent_key})")


def main():
    env = TeamCosEnv()
    print('ENV AGENT_Z_MIN, AGENT_Z_MAX:', env.AGENT_Z_MIN, env.AGENT_Z_MAX)
    # get raw xml
    try:
        xml = env._build_dynamic_xml()
    except Exception:
        # fallback: try to access model.orig_xml if present
        xml = None
        try:
            xml = env.model_xml
        except Exception:
            pass
    if xml is None:
        print('Could not obtain XML string from env')
    else:
        # write to tmp file for inspection
        path = os.path.join('scripts', 'diagnose_agent_xml.xml')
        with open(path, 'w') as f:
            f.write(xml)
        print('Wrote XML to', path)

    # list agents
    print('\nAgent keys:', env.agent_keys)

    # For each agent, show joint range, qpos adr and qpos
    m = env.model
    d = env.data
    for ak in env.agent_keys:
        print('\n==', ak, '==')
        # dump xml fragment for the specific agent
        if xml:
            dump_agent_xml(xml, ak)
        # joint info
        for jname in (f"{ak}_x", f"{ak}_y", f"{ak}_z", f"{ak}_rot"):
            try:
                j = m.joint(jname)
                jid = j.id
                qadr = m.jnt_qposadr[jid]
                qv = float(d.qpos[qadr])
                jrange = None
                try:
                    jrange = m.jnt_range[jid]
                except Exception:
                    jrange = None
                print(f"joint {jname}: id={jid} qposadr={qadr} qpos={qv} range={jrange}")
            except Exception as e:
                print(f"  cannot query joint {jname}: {e}")

        # geom info for body
        try:
            bid = env.body_ids[ak]
            gadr = int(m.body_geomadr[bid])
            gnum = int(m.body_geomnum[bid])
            print(' body_id', bid, 'geom adr/num', gadr, gnum)
            for gi in range(gadr, gadr + gnum):
                try:
                    gname = m.geom_id2name(gi)
                except Exception:
                    gname = f'geom#{gi}'
                gtype = int(getattr(m, 'geom_type', [None])[gi]) if hasattr(m, 'geom_type') else None
                try:
                    size = np.asarray(m.geom_size[gi])
                except Exception:
                    size = None
                try:
                    fromto = np.asarray(m.geom_fromto[gi])
                except Exception:
                    fromto = None
                try:
                    contype = int(m.geom_contype[gi])
                    conaff = int(m.geom_conaffinity[gi])
                except Exception:
                    contype = None
                    conaff = None
                world_z = float(d.geom_xpos[gi, 2]) if hasattr(d, 'geom_xpos') else None
                print(f"  geom {gi} {gname} type={gtype} size={size} fromto={fromto} contype={contype} conaff={conaff} world_z={world_z}")
        except Exception as e:
            print('  cannot enumerate geoms for body:', e)

    # Also dump global joint defaults and body positions
    print('\nGlobal ---')
    try:
        print(' model.njnt, ngeom, nbody =', int(m.njnt), int(m.ngeom), int(m.nbody))
    except Exception:
        pass
    # print joint ranges for all z joints
    for i in range(int(m.njnt)):
        try:
            name = m.joint_id2name(i)
        except Exception:
            name = f'j{i}'
        if name.endswith('_z'):
            try:
                r = m.jnt_range[i]
            except Exception:
                r = None
            print(' z-joint', name, 'range', r)


if __name__ == '__main__':
    main()
