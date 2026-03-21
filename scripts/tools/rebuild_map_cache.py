# rebuild_map_cache.py
# 演習第25回：main18 (OpenAI HNS) 専用・高精度回転対応キャッシュ生成

import os
import pickle
import tempfile

import mujoco
import numpy as np
from numba import njit

try:
    from main18_optimization import XML_CONTENT
except ImportError:
    XML_CONTENT = None


@njit(cache=True)
def _sdf_box_rotated_jit(p, b_pos, b_size, b_mat):
    rel_p = p - b_pos
    lx = rel_p[0] * b_mat[0] + rel_p[1] * b_mat[1]
    ly = rel_p[0] * b_mat[3] + rel_p[1] * b_mat[4]
    qx, qy = abs(lx) - b_size[0], abs(ly) - b_size[1]
    return np.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2) + min(max(qx, qy), 0.0)


@njit(cache=True)
def compute_static_sdf_map_jit(grid_coords, wall_data, wall_mats):
    num_points = len(grid_coords)
    sdf_map = np.empty(num_points, dtype=np.float32)
    for i in range(num_points):
        p = grid_coords[i]
        d_min = 1e10
        for j in range(len(wall_data)):
            d = _sdf_box_rotated_jit(p, wall_data[j, 0:2], wall_data[j, 2:4], wall_mats[j])
            if d < d_min:
                d_min = d
        sdf_map[i] = d_min
    return sdf_map


def generate_main18_cache(output_path="hns_sightmap_v25_cache_Maze.pkl"):
    if XML_CONTENT is None:
        return False
    print(f"🔄 キャッシュ生成中: {output_path} ...")
    fd, path = tempfile.mkstemp(suffix=".xml", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(XML_CONTENT)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    wall_data, wall_mats = [], []
    for i in range(model.ngeom):
        bid = model.geom_bodyid[i]
        if model.body_jntnum[bid] == 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "floor" in name:
                continue
            wall_data.append(
                [
                    data.geom_xpos[i][0],
                    data.geom_xpos[i][1],
                    model.geom_size[i][0],
                    model.geom_size[i][1],
                ]
            )
            wall_mats.append(data.geom_xmat[i].copy())

    bounds, cell_size = 6.2, 0.1
    cells = np.arange(-bounds, bounds + 0.001, cell_size)
    n_side = len(cells)
    grid_coords = np.array([[x, y] for y in reversed(cells) for x in cells], dtype=np.float32)
    sdf_map = compute_static_sdf_map_jit(
        grid_coords,
        np.array(wall_data, dtype=np.float32),
        np.array(wall_mats, dtype=np.float32),
    ).reshape((n_side, n_side))

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "layout_name": "main18",
                "bounds": bounds,
                "cell_size": cell_size,
                "n_side": n_side,
                "sdf_map": sdf_map,
            },
            f,
        )
    print("✅ 完了")
    os.remove(path)
    return True


if __name__ == "__main__":
    generate_main18_cache()
