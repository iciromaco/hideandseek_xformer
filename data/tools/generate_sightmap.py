# generate_sightmap.py
# 演習第25回：物理構造自動解析による統合マップ生成 (フルスペック・ノイズ排除・オフセットなし版)
# 
# 修正点:
# 1. 命名規則フィルタの徹底: "wall", "maze", "border" を含むgeomのみを抽出し、原点のマーカーを排除。
# 2. メタデータ抽出の復元: 検証スクリプト(verify_sightmap_results等)用にBox/Rampの初期情報を保存。
# 3. 視界マトリックス計算の復元: 静的な壁による全グリッド点ペア間の遮蔽判定を事前計算。
# 4. オフセットなしの同期: near_clipを使用せず、t=0からt=1の範囲で正確に幾何学的な遮蔽を判定。

import numpy as np
import mujoco
import pickle
import time
import os
import tempfile
from pathlib import Path
from numba import njit

# main18のXML定義（環境に合わせてインポート）
try:
    from main18_optimization import XML_CONTENT
except ImportError:
    XML_CONTENT = ""

# --- JIT 加速された幾何演算コア ---

@njit(cache=True)
def _sdf_box_aabb_jit(p, b_pos, b_size):
    """静的な壁(AABB)に対する高速SDF"""
    qx, qy = abs(p[0]-b_pos[0]) - b_size[0], abs(p[1]-b_pos[1]) - b_size[1]
    return np.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2) + min(max(qx, qy), 0.0)

@njit(cache=True)
def _is_visible_static_jit(p1, p2, walls):
    """2点間が静的な壁によって遮蔽されているか判定 (線分交差・オフセットなし)"""
    for i in range(len(walls)):
        bx, by, sx, sy = walls[i]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        t_min, t_max = -1e10, 1e10
        
        # Xスラブ判定
        if abs(dx) < 1e-12:
            if abs(p1[0] - bx) > sx: continue
        else:
            inv_d = 1.0 / dx
            t1 = (bx - sx - p1[0]) * inv_d
            t2 = (bx + sx - p1[0]) * inv_d
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
            
        # Yスラブ判定
        if abs(dy) < 1e-12:
            if abs(p1[1] - by) > sy: continue
        else:
            inv_d = 1.0 / dy
            t1 = (by - sy - p1[1]) * inv_d
            t2 = (by + sy - p1[1]) * inv_d
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
            
        # 線分 [0.0001, 0.9999] の範囲内に障害物があれば遮蔽されているとみなす
        if t_max > t_min and t_min < 0.9999 and t_max > 0.0001:
            return False 
    return True

@njit(cache=True)
def compute_static_sdf_map_jit(grid_coords, walls):
    """グリッド全体のSDFマップを計算"""
    num_points = len(grid_coords)
    sdf_map = np.empty(num_points, dtype=np.float32)
    for i in range(num_points):
        p = grid_coords[i]
        d_min = 1e10
        for j in range(len(walls)):
            d = _sdf_box_aabb_jit(p, walls[j, 0:2], walls[j, 2:4])
            if d < d_min: d_min = d
        sdf_map[i] = d_min
    return sdf_map

@njit(cache=True)
def compute_visibility_matrix_jit(grid_coords, walls):
    """全グリッド点ペア間の可視性マトリックスを計算 (Sightmap)"""
    n = len(grid_coords)
    vis_matrix = np.ones((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_visible_static_jit(grid_coords[i], grid_coords[j], walls):
                vis_matrix[i, j] = 0
                vis_matrix[j, i] = 0
    return vis_matrix

def generate_layout_cache(layout_name: str):
    print(f"🚀 --- 統合マップキャッシュ再構築 (Full Spec Mode): {layout_name} ---")
    
    # XMLからモデルをロードして物理座標を確定
    fd, path = tempfile.mkstemp(suffix='.xml', text=True)
    with os.fdopen(fd, 'w') as f: f.write(XML_CONTENT)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    wall_data = []
    dynamic_objects = []
    
    print("🔍 物理構造を解析中 (Naming Filter: wall/maze/border)...")
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        bid = model.geom_bodyid[i]
        
        # 1. 静的な壁の抽出 (ジョイントを持たないボディ)
        if model.body_jntnum[bid] == 0:
            if any(k in name.lower() for k in ["wall", "maze", "border"]):
                if "floor" in name.lower() or "ground" in name.lower(): continue
                # 物理座標とサイズの取得
                pos = data.geom_xpos[i][:2].copy()
                size = model.geom_size[i][:2].copy()
                wall_data.append([pos[0], pos[1], size[0], size[1]])
                print(f"  [Static] Added: {name:20} at {pos}")
        
        # 2. 動的なオブジェクトの抽出 (検証メタデータ用)
        else:
            b_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if any(k in b_name.lower() for k in ["box", "ramp"]):
                # Rampはメインのスロープ表面のみを保存
                if "ramp" in b_name.lower() and "slope_surface" not in name.lower():
                    continue
                pos = data.geom_xpos[i][:2].copy()
                size = model.geom_size[i][:2].copy()
                dynamic_objects.append({
                    "name": name,
                    "pos": pos,
                    "size": size
                })
                print(f"  [Dynamic] Meta: {name:20} at {pos}")

    if not wall_data:
        print("❌ エラー: 静的な壁が見つかりませんでした。XMLの命名を確認してください。")
        return

    wall_data_np = np.array(wall_data, dtype=np.float32)
    
    # グリッドの定義
    bounds = 6.2
    cell_size = 0.1
    cells = np.arange(-bounds, bounds + 0.001, cell_size)
    n_side = len(cells)
    
    # グリッド座標の生成 (MuJoCoの+Y=Upに合わせた反転レイアウト)
    grid_coords = np.array([[x, y] for y in reversed(cells) for x in cells], dtype=np.float32)
    
    print(f"🔄 SDF Map ({n_side}x{n_side}) 計算中...")
    t0 = time.time()
    sdf_map = compute_static_sdf_map_jit(grid_coords, wall_data_np).reshape((n_side, n_side))
    print(f"✅ SDF完了 ({time.time()-t0:.2f}s)")

    print(f"🔄 視界マトリックス ({n_side**2} pairs) 計算中...")
    t1 = time.time()
    vis_matrix = compute_visibility_matrix_jit(grid_coords, wall_data_np)
    print(f"✅ 視界マトリックス完了 ({time.time()-t1:.2f}s)")

    cache_path = Path(f"hns_sightmap_v25_cache_{layout_name}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump({
            "layout_name": layout_name, 
            "bounds": bounds, 
            "cell_size": cell_size, 
            "n_side": n_side, 
            "sdf_map": sdf_map,
            "vis_matrix": vis_matrix,
            "walls": wall_data,
            "grid_coords": grid_coords,
            "dynamic_objects": dynamic_objects
        }, f)
    
    os.remove(path)
    print(f"🎉 統合キャッシュ保存完了: {cache_path}")

if __name__ == "__main__":
    generate_layout_cache("Maze")