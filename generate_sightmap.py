# generate_sightmap.py
# 演習第25回：統合視界テーブル（Sightmap）および静的SDF Mapの生成
# 
# 1. 静的な壁に対する遮蔽判定マトリックスを生成。
# 2. 静的な壁に対する高解像度な SDF Map (距離場) をプリコンパイル。
# 3. キャッシュが存在する場合はロードして再利用し、計算時間を短縮。

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
def _sdf_box_jit(p, b_pos, b_size):
    q = np.abs(p - b_pos) - b_size
    return np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1]), 0.0)

@njit(cache=True)
def compute_static_sdf_map_jit(grid_coords, walls):
    """全グリッド点に対して、最も近い壁への距離を計算"""
    num_points = len(grid_coords)
    sdf_map = np.empty(num_points, dtype=np.float32)
    for i in range(num_points):
        p = grid_coords[i]
        d_min = 1e10
        for j in range(len(walls)):
            # 壁を矩形(pos, size)として扱う
            w_pos = walls[j, 0:2]
            w_size = walls[j, 2:4]
            d = _sdf_box_jit(p, w_pos, w_size)
            if d < d_min: d_min = d
        sdf_map[i] = d_min
    return sdf_map

@njit(cache=True)
def compute_vis_matrix_jit(grid_coords, wall_segments):
    """視界マトリックスの計算 (線分交差ベース)"""
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
            for k in range(len(wall_segments)):
                d = get_ray_intersect_dist_jit(p1, direction, wall_segments[k, 0:2], wall_segments[k, 2:4])
                if d < dist - 0.001:
                    is_visible = False
                    break
            vis_matrix[i, j] = is_visible
            vis_matrix[j, i] = is_visible
    return vis_matrix

def generate_layout_cache(layout_name: str):
    print(f"\n--- Resource Compilation: {layout_name} ---")
    
    cache_path = Path(f"hns_sightmap_v25_cache_{layout_name}.pkl")
    existing_cache = {}
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                existing_cache = pickle.load(f)
            print(f"📦 既存のキャッシュファイルをロードしました: {cache_path}")
        except Exception as e:
            print(f"⚠️ キャッシュのロードに失敗しました (再計算します): {e}")

    env = TeamCosEnv(layout_name=layout_name)
    m = env.model
    
    # グリッド設定 (SDF Map用に 0.1m 解像度)
    bounds = 6.2
    cell_size = 0.1
    cells = np.arange(-bounds, bounds + 0.001, cell_size)
    n_side = len(cells)
    grid_coords = np.array([[x, y] for y in reversed(cells) for x in cells], dtype=np.float32)
    
    # 壁データの抽出
    wall_boxes = [] # SDF計算用 (x, y, sx, sy)
    wall_segments = [] # 視界判定用 (x1, y1, x2, y2)
    
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name is None: continue
        if any(k in name for k in ["wall", "maze", "border"]):
            p = m.geom_pos[i][:2]; s = m.geom_size[i][:2]
            wall_boxes.append([p[0], p[1], s[0], s[1]])
            if s[0] > s[1]: # 横
                wall_segments.append([p[0]-s[0], p[1], p[0]+s[0], p[1]])
            else: # 縦
                wall_segments.append([p[0], p[1]-s[1], p[0], p[1]+s[1]])
    
    wall_boxes_np = np.array(wall_boxes, dtype=np.float32)
    wall_segs_np = np.array(wall_segments, dtype=np.float32)

    # 1. 静的SDF Mapの生成
    if "sdf_map" in existing_cache and existing_cache["sdf_map"].shape == (n_side, n_side):
        print(f"   - SDF Map をキャッシュから再利用します。")
        sdf_map_2d = existing_cache["sdf_map"]
    else:
        print(f"   - Static SDF Map ({n_side}x{n_side}) を計算中...")
        sdf_map_raw = compute_static_sdf_map_jit(grid_coords, wall_boxes_np)
        sdf_map_2d = sdf_map_raw.reshape((n_side, n_side))
    
    # 2. 視界マトリックスの生成
    vis_cell_size = 0.4
    vis_cells = np.arange(-bounds, bounds + 0.001, vis_cell_size)
    vis_n_side = len(vis_cells)
    
    if "vis_matrix" in existing_cache and existing_cache.get("vis_n_side") == vis_n_side:
        print(f"   - Visibility Matrix をキャッシュから再利用します。")
        vis_matrix = existing_cache["vis_matrix"]
    else:
        print(f"   - Visibility Matrix を計算中 (これには時間がかかります)...")
        vis_grid_coords = np.array([[x, y] for y in reversed(vis_cells) for x in vis_cells], dtype=np.float32)
        vis_matrix = compute_vis_matrix_jit(vis_grid_coords, wall_segs_np)
    
    cache_data = {
        "layout_name": layout_name,
        "bounds": bounds,
        "cell_size": cell_size,
        "n_side": n_side,
        "sdf_map": sdf_map_2d,
        "vis_matrix": vis_matrix,
        "vis_cell_size": vis_cell_size,
        "vis_n_side": vis_n_side
    }
    
    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)
    print(f"✅ プリコンパイル完了: {cache_path}")

if __name__ == "__main__":
    generate_layout_cache("Maze")