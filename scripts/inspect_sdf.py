"""Inspect saved SDF grid and current environment wall geom positions.
Run:
    python scripts/inspect_sdf.py
or specify a pickle: python scripts/inspect_sdf.py --path path/to/sdfgrid.pkl
"""

import argparse
import glob
import os
import pickle
import textwrap

import mujoco

from envs.hns_environment import TeamCosEnv


def find_latest_sdf(envs_dir):
    pattern = os.path.join(envs_dir, "sdfgrid_mode4_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def print_pkl_info(p):
    print(f"Loading: {p}")
    with open(p, "rb") as f:
        data = pickle.load(f)
    keys = list(data.keys())
    print("Keys:", keys)
    grid = data.get("grid")
    normals = data.get("normals")
    print(f"grid shape: {None if grid is None else grid.shape}")
    print(f"normals shape: {None if normals is None else normals.shape}")
    print(f"min_x/min_y/cell_size: {data.get('min_x')},{data.get('min_y')},{data.get('cell_size')}")
    if grid is not None:
        print("grid sample (center 5x5):")
        ny, nx = grid.shape
        cy, cx = ny // 2, nx // 2
        lo_y = max(0, cy - 2)
        hi_y = min(ny, cy + 3)
        lo_x = max(0, cx - 2)
        hi_x = min(nx, cx + 3)
        for y in range(lo_y, hi_y):
            row = grid[y, lo_x:hi_x]
            print(" ", " ".join(f"{v:6.3f}" for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    args = ap.parse_args()

    repo_envs = os.path.join(os.path.dirname(__file__), "../envs")
    src_envs = os.path.join(os.path.dirname(__file__), "../src/envs")
    p = args.path or find_latest_sdf(src_envs) or find_latest_sdf(repo_envs)
    if p is None:
        print(f"No sdf grid found in {src_envs} or {repo_envs}")
    else:
        print_pkl_info(p)

    print("\nInstantiate environment and list wall geoms:")
    env = TeamCosEnv(mode="initial", target="seeker", n_seekers=1, n_hiders=1, n_boxes=0, n_ramps=0, render_mode=None, debug_mode=False)
    try:
        m = env.model
        d = env.data
        idxs = env.vis_engine.idx_wall_geom
        if len(idxs) == 0:
            print("No wall geoms found in model.vis_engine.idx_wall_geom")
        else:
            print(f"Found {len(idxs)} wall geoms:\n")
            for i in idxs:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(i)) if "mujoco" in globals() else str(i)
                pos = d.geom_xpos[int(i)]
                size = m.geom_size[int(i)]
                print(textwrap.dedent(f"""
                    geom_id={i} name={name}
                      pos={pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}
                      size={size[0]:.3f},{size[1]:.3f},{size[2]:.3f}
                """))
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
