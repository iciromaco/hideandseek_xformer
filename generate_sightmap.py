# generate_sightmap.py
# 演習第25回：統合視界テーブル（Sightmap）プリコンパイル
# 
# 1. 静的な壁の遮蔽判定マトリックスのみを計算。
# 2. 検証用として、その時の動的オブジェクトの位置スナップショットを同梱。

import numpy as np
import mujoco
import pickle
from pathlib import Path
from numba import njit
from hns_environment import TeamCosEnv

@njit(cache=True)
def get_ray_intersect_dist_jit(p_start, p_dir, a, b):
    v1 = p_start - a
    v2 = b - a
    v3 = np.array([-p_dir[1], p_dir[0]], dtype=np.float32)
    dot = v2[0] * v3[0] + v2[1] * v3[1]
    if abs(dot) < 1e-8: return 1e10
    t1 = (v2[0] * v1[1] - v2[1] * v1[0]) / dot
    t2 = (v1[0] * v3[0] + v1[1] * v3[1]) / dot
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0: return t1
    return 1e10

@njit(cache=True)
def compute_vis_matrix_jit(grid_coords, walls):
    num_points = len(grid_coords)
    vis_matrix = np.ones((num_points, num_points), dtype=np.bool_)
    for i in range(num_points):
        p1 = grid_coords[i]
        for j in range(i + 1, num_points):
            p2 = grid_coords[j]
            diff = p2 - p1
            dist = np.sqrt(np.sum(diff**2))
            if dist < 0.01: continue
            direction = diff / dist
            is_visible = True
            for k in range(len(walls)):
                d = get_ray_intersect_dist_jit(p1, direction, walls[k, 0], walls[k, 1])
                if d < dist - 0.001:
                    is_visible = False
                    break
            vis_matrix[i, j] = is_visible
            vis_matrix[j, i] = is_visible
    return vis_matrix

def generate_layout_cache(layout_name: str):
    print(f"\n--- Sightmap 生成開始: {layout_name} ---")
    env = TeamCosEnv(layout_name=layout_name)
    m = env.model; d = env.data
    
    # 物理状態確定 (検証用スナップショット)
    env.reset()
    mujoco.mj_forward(m, d)
    
    # 動的オブジェクト情報の記録
    dynamic_snapshot = []
    for b_name in ["box1_body", "box2_body", "ramp_body"]:
        try:
            b_id = m.body(b_name).id
            g_id = m.body(b_name).geomadr[0]
            dynamic_snapshot.append({
                "name": b_name,
                "pos": d.xpos[b_id][:2].copy(),
                "size": m.geom_size[g_id][:2].copy()
            })
        except: pass

    bounds = 6.0; cell_size = 0.4
    cells = np.arange(-bounds, bounds + 0.01, cell_size)
    n_side = len(cells)
    grid_coords = np.array([[x, y] for y in reversed(cells) for x in cells], dtype=np.float32)
    
    # 静的な壁のみ抽出
    walls = []
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name is None: continue
        if layout_name == "Open" and "maze" in name: continue
        if any(k in name for k in ["wall", "maze", "border"]):
            p = m.geom_pos[i][:2]; s = m.geom_size[i][:2]
            if s[0] > s[1]: walls.append([[p[0]-s[0], p[1]], [p[0]+s[0], p[1]]])
            else: walls.append([[p[0], p[1]-s[1]], [p[0], p[1]+s[1]]])
    
    walls_np = np.array(walls, dtype=np.float32)
    vis_matrix = compute_vis_matrix_jit(grid_coords, walls_np)
    
    cache_data = {
        "walls": walls_np, "grid_coords": grid_coords, "vis_matrix": vis_matrix,
        "cell_size": cell_size, "bounds": bounds, "n_side": n_side,
        "layout_name": layout_name, "dynamic_objects": dynamic_snapshot
    }
    
    cache_path = Path(f"hns_sightmap_v25_cache_{layout_name}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"✅ 保存完了: {cache_path}")

if __name__ == "__main__":
    for name in ["Open", "Maze"]:
        generate_layout_cache(name)