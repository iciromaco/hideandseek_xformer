"""Sweep WALL_REPULSION parameters and collect simple metrics.

Usage:
  uv run python3 scripts/sweep_wall_repulsion.py

This runs short episodes for combinations of parameters and prints CSV-like results.
"""

import csv
import itertools
import statistics
import time

import numpy as np

from src.envs.hns_environment import TeamCosEnv


def eval_setting(radius, alpha, fmax, episodes=3, steps=200):
    metrics = []
    for ep in range(episodes):
        env = TeamCosEnv(mode="initial", debug_mode=False)
        # apply params
        env.WALL_REPULSION_RADIUS = float(radius)
        env.WALL_REPULSION_ALPHA = float(alpha)
        env.WALL_REPULSION_FMAX = float(fmax)

        obs = env.reset()
        learnable_key = env.learnable_agent_key
        stuck_flag = False
        stuck_events = 0
        stuck_streak = 0
        turns = []
        wall_dists = []

        for t in range(steps):
            # simple probing policy: random actions
            a = env.action_space.sample()
            res = env.step(a)
            # support different env.step return signatures: (obs, info) or (obs, reward, done, info) or (obs, reward, info)
            if isinstance(res, tuple):
                if len(res) == 2:
                    o, info = res
                elif len(res) == 4:
                    o, _r, _done, info = res
                elif len(res) == 3:
                    o, _r, info = res
                elif len(res) == 5:
                    # gymnasium-style: (obs, reward, terminated, truncated, info)
                    o, _r, _term, _trunc, info = res
                else:
                    raise ValueError(f"Unexpected env.step() return tuple length: {len(res)}")
            else:
                o = res
                info = {}
            # record
            wd = info.get("wall_distance")
            if isinstance(wd, (list, tuple, np.ndarray)):
                wd = float(wd[0])
            wall_dists.append(wd if wd is not None else 999.0)

            # read last turn control for learnable agent if available
            last = env.last_debug_ctrl.get(env.learnable_agent_key, (0.0, 0.0))
            turns.append(abs(float(last[1])))

            # simple wall-stick detector: distance < 0.2 and low speed
            bid = env.body_ids[env.learnable_agent_key]
            # support different mujoco bindings: prefer body linear velocity `xvelp`,
            # fall back to `xvel` or `cvel` if necessary
            if hasattr(env.data, "xvelp"):
                speed = float(np.linalg.norm(env.data.xvelp[bid][:2]))
            elif hasattr(env.data, "xvel"):
                speed = float(np.linalg.norm(env.data.xvel[bid][:2]))
            elif hasattr(env.data, "cvel"):
                speed = float(np.linalg.norm(env.data.cvel[bid][:2]))
            else:
                speed = 0.0
            if wd is not None and wd < 0.2 and speed < 0.05:
                stuck_streak += 1
            else:
                if stuck_streak >= 3:
                    stuck_events += 1
                stuck_streak = 0

        # finalize
        if stuck_streak >= 3:
            stuck_events += 1

        metrics.append(
            {
                "wall_dist_mean": statistics.mean(wall_dists),
                "wall_dist_min": min(wall_dists),
                "wall_dist_median": statistics.median(wall_dists),
                "avg_turn": statistics.mean(turns) if turns else 0.0,
                "turn_std": statistics.pstdev(turns) if len(turns) > 1 else 0.0,
                "stuck_events": stuck_events,
            }
        )
        try:
            env.close()
        except Exception:
            pass

    # aggregate across episodes
    agg = {
        "wall_dist_mean": statistics.mean([m["wall_dist_mean"] for m in metrics]),
        "wall_dist_min": min([m["wall_dist_min"] for m in metrics]),
        "wall_dist_median": statistics.mean([m["wall_dist_median"] for m in metrics]),
        "avg_turn": statistics.mean([m["avg_turn"] for m in metrics]),
        "turn_std": statistics.mean([m["turn_std"] for m in metrics]),
        "stuck_events": sum([m["stuck_events"] for m in metrics]),
    }
    return agg


def main():
    radii = [0.4, 0.6, 0.8]
    alphas = [0.5, 1.0, 1.5]
    fmaxs = [500.0, 1000.0, 1500.0]

    combos = list(itertools.product(radii, alphas, fmaxs))
    out_rows = []
    print("radius,alpha,fmax,wall_dist_mean,wall_dist_min,wall_dist_median,avg_turn,turn_std,stuck_events")
    for radius, alpha, fmax in combos:
        start = time.time()
        agg = eval_setting(radius, alpha, fmax, episodes=3, steps=200)
        elapsed = time.time() - start
        print(f"{radius},{alpha},{fmax},{agg['wall_dist_mean']:.3f},{agg['wall_dist_min']:.3f},{agg['wall_dist_median']:.3f},{agg['avg_turn']:.4f},{agg['turn_std']:.4f},{agg['stuck_events']}")
        out_rows.append((radius, alpha, fmax, agg))

    # optional: write CSV
    try:
        with open("sweep_wall_repulsion_results.csv", "w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow(["radius", "alpha", "fmax", "wall_dist_mean", "wall_dist_min", "wall_dist_median", "avg_turn", "turn_std", "stuck_events"])
            for r, a, f, agg in out_rows:
                w.writerow([r, a, f, agg["wall_dist_mean"], agg["wall_dist_min"], agg["wall_dist_median"], agg["avg_turn"], agg["turn_std"], agg["stuck_events"]])
    except Exception:
        pass


if __name__ == "__main__":
    main()
