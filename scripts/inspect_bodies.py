#!/usr/bin/env python3
"""Inspect body mass and joint metadata for agents."""
import sys
sys.path.insert(0, '.')

from src.envs.hns_environment import TeamCosEnv


def main():
    env = TeamCosEnv()
    env.reset()
    m = env.model
    print('nbody', m.nbody, 'njnt', m.njnt)
    for ak in env.agent_keys:
        bid = env.body_ids.get(ak)
        if bid is None:
            print(ak, 'no body id')
            continue
        mass = float(m.body_mass[bid])
        weld = int(m.body_weldid[bid])
        print(f"{ak}: body {bid}, mass={mass:.6f}, weldid={weld}")

    print('\n-- joints --')
    for j in range(m.njnt):
        try:
            rang = tuple(m.jnt_range[j])
        except Exception:
            rang = None
        qadr = int(m.jnt_qposadr[j]) if m.jnt_qposadr[j] >= 0 else None
        print(f"j{j}: type={int(m.jnt_type[j])}, qposadr={qadr}, range={rang}")

    print('\n-- joint mapping for agent bodies --')
    for ak in env.agent_keys:
        bid = env.body_ids.get(ak)
        if bid is None:
            continue
        jadr = int(m.body_jntadr[bid])
        jnum = int(m.body_jntnum[bid])
        print(f"{ak}: jadr={jadr} jnum={jnum}")
        for ji in range(jadr, jadr + jnum):
            print('  j', ji, 'qposadr', int(m.jnt_qposadr[ji]), 'type', int(m.jnt_type[ji]), 'range', tuple(m.jnt_range[ji]))


if __name__ == '__main__':
    main()
