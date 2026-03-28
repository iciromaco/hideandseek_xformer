#!/usr/bin/env python3
"""
SDF の内壁周辺の値を詳細に検査
"""

import pickle
import sys
from pathlib import Path

import numpy as np

# SDF を読み込み
cache_file = Path("sdf_distance_field.pkl")
if not cache_file.exists():
    print("Error: sdf_distance_field.pkl not found")
    sys.exit(1)

with open(cache_file, "rb") as f:
    data_dict = pickle.load(f)
    sdf_field = data_dict.get("sdf_field")

grid_size = sdf_field.shape[0]
cell_size = 12.0 / (grid_size - 1)

print(f"SDF shape: {sdf_field.shape}")
print(f"Cell size: {cell_size:.6f}")
print()

# 内壁周辺の SDF を検査
test_points = [
    # 外壁
    (6.0, 0.0, "外壁東"),
    (-6.0, 0.0, "外壁西"),
    (0.0, 6.0, "外壁北"),
    (0.0, -6.0, "外壁南"),
    # 内壁 maze_w0: x=[1.5,4.5], y=[1.3,1.7]
    (3.0, 1.7, "maze_w0 上辺（壁の上）"),
    (3.0, 1.5, "maze_w0 中央（壁内）"),
    (3.0, 1.3, "maze_w0 下辺（壁の下）"),
    (3.0, 2.0, "maze_w0 上外側"),
    (3.0, 1.0, "maze_w0 下外側"),
    # 内壁 maze_w1: x=[-4.5,-1.5], y=[-1.7,-1.3]
    (-3.0, -1.7, "maze_w1 下辺（壁の下）"),
    (-3.0, -1.5, "maze_w1 中央（壁内）"),
    (-3.0, -1.3, "maze_w1 上辺（壁の上）"),
    # 視点周辺
    (-2.0, 0.0, "Viewpoint"),
    (-1.0, 0.0, "Viewpoint + 1m"),
    (0.0, 0.0, "Center"),
]

print("[SDF VALUES AT TEST POINTS]")
print()

for x, y, label in test_points:
    # グリッド座標に変換
    grid_x = int((x + 6.0) / cell_size + 0.5)
    grid_y = int((y + 6.0) / cell_size + 0.5)

    # 境界チェック
    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
        sdf_val = sdf_field[grid_y, grid_x]
        print(f"{label:25} ({x:6.2f}, {y:6.2f}) -> grid ({grid_x:3d}, {grid_y:3d}): SDF = {sdf_val:7.4f}")
    else:
        print(f"{label:25} ({x:6.2f}, {y:6.2f}) -> grid ({grid_x:3d}, {grid_y:3d}): OUT OF BOUNDS")

# ビーム方向のレイをトレース
print("\n[RAY TRACING SDF VALUES]")
print()

yaw = np.pi / 3
viewpoint = np.array([-2.0, 0.0])

# Beam 1: 45度（maze_w3に向かう、内壁を通り抜けている）
lidar_angle = np.pi / 4  # 45度
beam_cos = np.cos(lidar_angle) * np.cos(yaw) - np.sin(lidar_angle) * np.sin(yaw)
beam_sin = np.cos(lidar_angle) * np.sin(yaw) + np.sin(lidar_angle) * np.cos(yaw)

print(f"Beam 1 (45° from viewpoint): direction = ({beam_cos:.4f}, {beam_sin:.4f})")
print(f"Starting from: {viewpoint}")
print()

# レイに沿った SDF 値をサンプル
distances = np.linspace(0, 6, 30)
print(f"{'Distance':<10} {'Position':<20} {'SDF':<10} {'Status':<20}")
print("-" * 60)

for dist in distances:
    ray_x = viewpoint[0] + beam_cos * dist
    ray_y = viewpoint[1] + beam_sin * dist

    grid_x = int((ray_x + 6.0) / cell_size + 0.5)
    grid_y = int((ray_y + 6.0) / cell_size + 0.5)

    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
        sdf_val = sdf_field[grid_y, grid_x]

        # 壁チェック
        status = "OK"
        # maze_w3: x=[-0.1,0.1], y=[1.5,4.5]
        if -0.1 <= ray_x <= 0.1 and 1.5 <= ray_y <= 4.5:
            status = "INSIDE maze_w3!"
        # 他の壁のチェック
        elif ray_x <= -6.0 or ray_x >= 6.0 or ray_y <= -6.0 or ray_y >= 6.0:
            status = "OUTSIDE bounds!"

        print(f"{dist:<10.2f} ({ray_x:7.3f}, {ray_y:7.3f})  {sdf_val:<10.4f} {status:<20}")
    else:
        print(f"{dist:<10.2f} ({ray_x:7.3f}, {ray_y:7.3f})  OUT OF BOUNDS  OUT OF BOUNDS")
