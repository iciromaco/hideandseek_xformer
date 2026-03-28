#!/usr/bin/env python3
"""
SDF距離フィールドを生成（テスト用）
"""

import os
import sys

import numpy as np

current_script_abs_path_val = os.path.abspath(__file__)
current_script_parent_dir_val = os.path.dirname(current_script_abs_path_val)
search_dir_path_pointer_idx = current_script_parent_dir_val

for _ in range(5):
    potential_base_config_file_path = os.path.join(search_dir_path_pointer_idx, "main18_optimization.py")
    if os.path.exists(potential_base_config_file_path):
        if search_dir_path_pointer_idx not in sys.path:
            sys.path.insert(0, search_dir_path_pointer_idx)
        break
    search_dir_path_pointer_idx = os.path.dirname(search_dir_path_pointer_idx)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

import main18_optimization as base_config
import mujoco
from main23_sightmap_optimized import (
    VisibilityEngine,
    build_sdf_distance_field,
    save_sdf_distance_field,
)

# 初期化
xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# VisibilityEngine の初期化
visibility_engine = VisibilityEngine(model, data, epsilon=0.1, max_steps=15, max_dist=15.0)

# Bodies の設定
s0_body = model.body("seeker_body").id
h1_body = model.body("hider1_body").id
h2_body = model.body("hider2_body").id
box1_body = model.body("box1_body").id
box2_body = model.body("box2_body").id
ramp_body = model.body("ramp_body").id

visibility_engine.set_bodies(s0_body, h1_body, h2_body, box1_body, box2_body, ramp_body)

# 動的オブジェクト位置を設定
visibility_engine.dynamic_positions[0] = np.array([3.0, 3.0])
visibility_engine.dynamic_positions[1] = np.array([1.0, 1.0])
visibility_engine.dynamic_positions[2] = np.array([5.0, 5.0])
visibility_engine.dynamic_positions[3] = np.array([2.0, -2.0])
visibility_engine.dynamic_positions[4] = np.array([-2.0, 2.0])
visibility_engine.dynamic_positions[5] = np.array([0.0, 0.0])

# 壁を設定
wall_segments = [
    ((-6.0, 6.0), (6.0, 6.0)),
    ((-6.0, -6.0), (6.0, -6.0)),
    ((6.0, -6.0), (6.0, 6.0)),
    ((-6.0, -6.0), (-6.0, 6.0)),
    ((1.5, 1.7), (4.5, 1.7)),
    ((1.5, 1.3), (4.5, 1.3)),
    ((-4.5, -1.3), (-1.5, -1.3)),
    ((-4.5, -1.7), (-1.5, -1.7)),
    ((-0.1, -4.5), (-0.1, -1.5)),
    ((0.1, -4.5), (0.1, -1.5)),
    ((-0.1, 1.5), (-0.1, 4.5)),
    ((0.1, 1.5), (0.1, 4.5)),
]

for i, (p1, p2) in enumerate(wall_segments):
    if i >= 50:
        break
    visibility_engine.wall_segments[i, 0] = p1[0]
    visibility_engine.wall_segments[i, 1] = p1[1]
    visibility_engine.wall_segments[i, 2] = p2[0]
    visibility_engine.wall_segments[i, 3] = p2[1]
    visibility_engine.num_wall_segments += 1

visibility_engine.static_walls = wall_segments

print("Building SDF field...")
# グリッドセルを作成
grid_size = 590
cell_size = 12.0 / (grid_size - 1)
cell_centers = []
for x in np.linspace(-6, 6, grid_size):
    for y in np.linspace(-6, 6, grid_size):
        cell_centers.append([x, y])
cell_centers = np.array(cell_centers, dtype=np.float32)

# SDF を構築
sdf_field = build_sdf_distance_field(visibility_engine, cell_centers)

# 保存
print("Saving SDF field...")
metadata = {
    "grid_size": grid_size,
    "cell_size": cell_size,
    "bounds": [[-6, 6], [-6, 6]],
    "timestamp": str(np.datetime64("now")),
}
save_sdf_distance_field(sdf_field, cell_centers, metadata, "sdf_distance_field.pkl")

print("SDF field saved!")
print(f"  Shape: {sdf_field.shape}")
print(f"  Min: {sdf_field.min():.4f}")
print(f"  Max: {sdf_field.max():.4f}")
print(f"  Mean: {sdf_field.mean():.4f}")
