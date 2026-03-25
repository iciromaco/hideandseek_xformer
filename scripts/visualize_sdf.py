"""Visualize saved SDF grid and normals.
Usage:
    python scripts/visualize_sdf.py --path path/to/sdfgrid_mode4_*.pkl
If no path provided, picks the most recent sdfgrid file under envs/.
"""

import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


def find_latest_sdf(envs_dir):
    pattern = os.path.join(envs_dir, "sdfgrid_mode4_*.pkl")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="Path to sdf pickle")
    ap.add_argument("--envs-dir", default=None, help="envs dir")
    ap.add_argument("--subsample", type=int, default=8, help="quiver subsample")
    args = ap.parse_args()

    p = args.path
    if p is None:
        # prefer project-level envs/, then src/envs/
        candidates = []
        if args.envs_dir:
            candidates.append(args.envs_dir)
        repo_envs = os.path.join(os.path.dirname(__file__), "../envs")
        src_envs = os.path.join(os.path.dirname(__file__), "../src/envs")
        candidates.extend([repo_envs, src_envs])
        found = None
        for c in candidates:
            c = os.path.abspath(c)
            found = find_latest_sdf(c)
            if found is not None:
                p = found
                break
        if p is None:
            raise SystemExit(f"No sdf grid found in {candidates}")
    print(f"Loading {p}")
    with open(p, "rb") as f:
        data = pickle.load(f)
    grid = data.get("grid")
    normals = data.get("normals")
    min_x = data.get("min_x")
    min_y = data.get("min_y")
    cell = data.get("cell_size")

    ny, nx = grid.shape
    extent = (min_x, min_x + nx * cell, min_y, min_y + ny * cell)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, origin="lower", extent=extent, cmap="RdYlBu_r")
    plt.colorbar(label="SDF distance (m)")
    plt.title(os.path.basename(p))

    if normals is not None:
        step = max(1, args.subsample)
        Y, X = np.mgrid[0:ny:step, 0:nx:step]
        # convert indices to world coords (center of cell)
        xs = min_x + (X + 0.5) * cell
        ys = min_y + (Y + 0.5) * cell
        nx_s = normals[Y, X, 0]
        ny_s = normals[Y, X, 1]
        # quiver expects X, Y in world coords
        plt.quiver(xs, ys, nx_s, ny_s, color="k", scale=25)

    plt.xlabel("world x")
    plt.ylabel("world y")
    plt.show()


if __name__ == "__main__":
    main()
