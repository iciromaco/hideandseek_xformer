#!/usr/bin/env python3
"""
SDF Distance Field を生成
"""

import sys
sys.path.insert(0, "/Users/dan/Desktop/Semi/hideandseek_xformer")

import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path

# main23 から必要な関数をインポート
from hideandseek import XML_CONTENT
from main23_sightmap_optimized import extract_maze_walls_from_xml, walls_to_segments

# XMLから内壁を抽出
print("[1] Extracting walls from XML...")
maze_walls = extract_maze_walls_from_xml(XML_CONTENT, from_string=True)
static_walls = walls_to_segments(maze_walls)
print(f"  Found {len(maze_walls)} maze walls → {len(static_walls)} wall boxes")

# グリッド設定
grid_size = 590
cell_size = 12.0 / (grid_size - 1)
boundary = np.array([0.0, 0.0, 6.0, 6.0])

print(f"\n[2] Building SDF grid ({grid_size}×{grid_size})")
print(f"  Cell size: {cell_size:.6f}")
print(f"  Boundary: {boundary}")

# SDF 計算関数（get_sdf_static と同じロジック）
def get_sdf_static(p, boundary, static_walls):
    # 1. 外壁までの距離（AABB: center [0,0], half-size [6,6]）
    center = boundary[:2]
    half_size = boundary[2:]
    d = np.abs(p[:2] - center) - half_size
    
    # 点が外壁内部か外部かで符号を決定
    inside_boundary = np.all(d <= 0)
    
    if inside_boundary:
        # 外壁内部：最も近い壁までの距離（負の値）
        boundary_dist = np.min(d)  # すべて <= 0, なので負または0
    else:
        # 外壁外部：最短距離（正の値）
        outside_dist = np.linalg.norm(np.maximum(d, 0.0))
        boundary_dist = -outside_dist  # 外部は負
    
    # 2. 内壁（矩形AABB）までの距離
    min_wall_dist = boundary_dist
    
    if len(static_walls) > 0:
        for wall in static_walls:
            x_min = wall['x_min']
            y_min = wall['y_min']
            x_max = wall['x_max']
            y_max = wall['y_max']
            
            # 矩形の外側への最短距離
            dx = max(x_min - p[0], p[0] - x_max, 0.0)
            dy = max(y_min - p[1], p[1] - y_max, 0.0)
            dist_outside = np.sqrt(dx**2 + dy**2)
            
            # 矩形内にいるかチェック
            if p[0] >= x_min and p[0] <= x_max and p[1] >= y_min and p[1] <= y_max:
                # 矩形内：最も近い辺までの距離（負の値）
                dist_x_min = p[0] - x_min
                dist_x_max = x_max - p[0]
                dist_y_min = p[1] - y_min
                dist_y_max = y_max - p[1]
                
                # 最も近い辺までの距離（負）
                wall_dist = -min(min(dist_x_min, dist_x_max),
                                  min(dist_y_min, dist_y_max))
            else:
                # 矩形外：距離は正
                wall_dist = dist_outside
            
            # 最も近い障害物までの距離を保持
            if wall_dist < min_wall_dist:
                min_wall_dist = wall_dist
    
    return min_wall_dist

# SDF フィールドを構築
sdf_field = np.zeros((grid_size, grid_size), dtype=np.float32)

total_cells = grid_size * grid_size
print(f"\n[3] Computing SDF for {total_cells:,} cells...")

for i in tqdm(range(grid_size), desc="Building SDF"):
    for j in range(grid_size):
        # グリッド座標 → 世界座標
        x = -6.0 + i * cell_size
        y = -6.0 + j * cell_size
        
        p = np.array([x, y, 0.0])
        sdf_field[j, i] = get_sdf_static(p, boundary, static_walls)

# 統計
print(f"\n[4] SDF Statistics:")
print(f"  Shape: {sdf_field.shape}")
print(f"  Min: {sdf_field.min():.6f}")
print(f"  Max: {sdf_field.max():.6f}")
print(f"  Mean: {sdf_field.mean():.6f}")
print(f"  Negative pixels: {np.sum(sdf_field < 0):,}")
print(f"  Zero pixels: {np.sum(np.abs(sdf_field) < 1e-6):,}")
print(f"  Positive pixels: {np.sum(sdf_field > 0):,}")

# 保存
output_file = Path("sdf_distance_field.pkl")
data_dict = {
    "sdf_field": sdf_field,
    "cell_centers": None,
    "metadata": {
        "grid_size": grid_size,
        "cell_size": cell_size,
        "boundary": boundary.tolist(),
    }
}

with open(output_file, "wb") as f:
    pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

file_size_kb = output_file.stat().st_size / 1024
print(f"\n✓ Distance field saved: {output_file} ({file_size_kb:.1f} KB)")
