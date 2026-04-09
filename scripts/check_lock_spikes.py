import numpy as np
import sys
from pathlib import Path
import argparse
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.hns_environment import TeamCosEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Check lock/hit and post-lock spike metrics.")
    parser.add_argument("--seeds", type=int, default=8, help="Number of seeds starting from 0")
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed value")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per seed")
    parser.add_argument("--window", type=int, default=5, help="Post-lock observation window")
    parser.add_argument("--obj-th", type=float, default=1.5, help="Object spike threshold")
    parser.add_argument("--ag-th", type=float, default=0.8, help="Agent spike threshold")
    parser.add_argument("--out", type=str, default="", help="Optional CSV output path")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    steps = args.steps
    post_lock_window = args.window
    obj_spike_th = args.obj_th
    ag_spike_th = args.ag_th

    rows = []
    for seed in seeds:
        env = TeamCosEnv(mode="initial", lidar_mode=1, render_mode=None)
        obs, _ = env.reset(seed=seed)
        data = env.data
        model = env.model

        prev_h_lock = False
        post_window = 0
        lock_events = 0
        h_lock_steps = 0
        obj_spikes = 0
        ag_spikes = 0

        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(
                np.array([0, 0, 0, 0], dtype=np.float32)
            )

            owners = {v for v in env.lock_owners.values() if v is not None}
            h_lock = "h" in owners

            if h_lock and not prev_h_lock:
                lock_events += 1
                post_window = post_lock_window
            if h_lock:
                h_lock_steps += 1

            ag_max = 0.0
            for agent_key in env.agent_keys:
                v_ax = model.jnt_dofadr[env.qpos_indices[agent_key]["x"]]
                v_ar = model.jnt_dofadr[env.qpos_indices[agent_key]["rot"]]
                v_xy = float(np.linalg.norm(data.qvel[v_ax : v_ax + 2]))
                v_rot = float(abs(data.qvel[v_ar]))
                ag_max = max(ag_max, v_xy + v_rot)

            obj_max = 0.0
            object_ids = env.box_ids + ([env.ramp_id] if env.ramp_id != -1 else [])
            for object_id in object_ids:
                v_obj = model.jnt_dofadr[model.body_jntadr[object_id]]
                obj_speed = float(np.linalg.norm(data.qvel[v_obj : v_obj + 6]))
                obj_max = max(obj_max, obj_speed)

            if post_window > 0:
                if obj_max > obj_spike_th:
                    obj_spikes += 1
                if ag_max > ag_spike_th:
                    ag_spikes += 1
                post_window -= 1

            prev_h_lock = h_lock
            if terminated or truncated:
                break

        env.close()
        rows.append((seed, lock_events, h_lock_steps, obj_spikes, ag_spikes))

    print(rows)
    print("lock_hit_seeds =", sum(1 for row in rows if row[2] > 0))
    print("total_obj_spikes =", sum(row[3] for row in rows))
    print("total_ag_spikes =", sum(row[4] for row in rows))

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "lock_events", "h_lock_steps", "obj_spikes", "ag_spikes"])
            for row in rows:
                writer.writerow(row)
        print("saved_csv =", out_path)


if __name__ == "__main__":
    main()
