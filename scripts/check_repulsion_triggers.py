"""Check how often wall-repulsion conditions are met for a given setting.
Usage:
  PYTHONPATH=. uv run python3 scripts/check_repulsion_triggers.py
"""

import time

from src.envs.hns_environment import TeamCosEnv


def run_check(radius, alpha, fmax, steps=500):
    env = TeamCosEnv(mode="initial", debug_mode=False)
    env.WALL_REPULSION_RADIUS = float(radius)
    env.WALL_REPULSION_ALPHA = float(alpha)
    env.WALL_REPULSION_FMAX = float(fmax)

    env.reset()
    triggers = 0
    total = 0
    for t in range(steps):
        a = env.action_space.sample()
        res = env.step(a)
        # normalize step return handling
        if isinstance(res, tuple):
            if len(res) == 2:
                o, info = res
            elif len(res) == 3:
                o, _r, info = res
            elif len(res) == 4:
                o, _r, _done, info = res
            elif len(res) == 5:
                o, _r, _term, _trunc, info = res
            else:
                o = res
                info = {}
        else:
            o = res
            info = {}
        # compute same condition as env: dist < r and abs(last_ctrl_f)>1e-6
        bid = env.body_ids[env.learnable_agent_key]
        pos = env.data.xpos[bid]
        try:
            dist, nx, ny = env.vis_engine.sample_sdf_with_normal(pos[0], pos[1])
        except Exception:
            dist = 999.0
        last = env.last_debug_ctrl.get(env.learnable_agent_key, (0.0, 0.0))
        last_ctrl_f = float(last[0])
        r = float(env.WALL_REPULSION_RADIUS)
        if dist < r and abs(last_ctrl_f) > 1e-6:
            triggers += 1
        total += 1
    try:
        env.close()
    except Exception:
        pass
    return triggers, total


if __name__ == "__main__":
    settings = [
        (0.4, 0.5, 1500.0),  # best-mean from sweep
        (0.6, 1.0, 1000.0),  # mid
        (0.8, 1.5, 1000.0),  # another
    ]
    for r, a, f in settings:
        t, tot = run_check(r, a, f, steps=1000)
        print(f"radius={r},alpha={a},fmax={f} -> triggers={t}/{tot} ({t/tot:.3%})")
