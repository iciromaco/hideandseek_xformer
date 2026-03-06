#!/usr/bin/env python3
"""
Sphere Tracing パラメータ最適化テスト
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
from main23_sightmap_optimized import VisibilityEngine, cast_ray_direct_numba, cast_ray_numba
import mujoco

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
    ((0.1, 1.5), (0.1, 4.5))
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

# SDF を読み込み
try:
    from pathlib import Path
    import pickle
    cache_file = Path("sdf_distance_field.pkl")
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data_dict = pickle.load(f)
            visibility_engine.sdf_field = data_dict.get("sdf_field")
except:
    pass

if visibility_engine.sdf_field is None:
    print("Error: SDF not loaded")
    sys.exit(1)

# テスト設定
viewpoint = np.array([-2.0, 0.0], dtype=np.float32)
yaw = np.pi / 3
SEEKER_BODY_ID = 5

# Lidar 設定
surround = np.linspace(0, 2*np.pi, 8, endpoint=False)
front = np.linspace(-np.pi/6, np.pi/6, 5)
lidar_angles = np.unique(np.concatenate([surround, front]))
n_beams = len(lidar_angles)

lidar_angle_cos = np.cos(lidar_angles)
lidar_angle_sin = np.sin(lidar_angles)

cy = np.cos(yaw)
sy = np.sin(yaw)
beam_cos = lidar_angle_cos * cy - lidar_angle_sin * sy
beam_sin = lidar_angle_cos * sy + lidar_angle_sin * cy

# パラメータテスト
test_params = [
    (0.01, 300, "epsilon=0.01, max_steps=300"),
    (0.005, 500, "epsilon=0.005, max_steps=500"),
    (0.001, 1000, "epsilon=0.001, max_steps=1000"),
    (0.002, 800, "epsilon=0.002, max_steps=800"),
]

print("[SPHERE TRACING PARAMETER OPTIMIZATION]")
print(f"Viewpoint: {viewpoint}, Yaw: {yaw:.3f}")
print()

for epsilon, max_steps, label in test_params:
    distances_direct = []
    distances_sphere = []
    
    for i in range(n_beams):
        direction = np.array([beam_cos[i], beam_sin[i], 0.0], dtype=np.float64)
        
        # Direct
        dist_dir, _ = cast_ray_direct_numba(
            viewpoint[0], viewpoint[1],
            direction[0], direction[1],
            visibility_engine.dynamic_positions,
            visibility_engine.dynamic_radii,
            visibility_engine.dynamic_body_ids_array,
            visibility_engine.num_dynamic_objects,
            visibility_engine.wall_segments,
            visibility_engine.num_wall_segments,
            SEEKER_BODY_ID, -1, 15.0
        )
        distances_direct.append(dist_dir)
        
        # Sphere
        grid_size = visibility_engine.sdf_field.shape
        cell_size = 12.0 / (grid_size[0] - 1)
        dist_sph, _ = cast_ray_numba(
            viewpoint[0], viewpoint[1],
            direction[0], direction[1],
            visibility_engine.sdf_field, grid_size[0], grid_size[1], cell_size,
            visibility_engine.dynamic_positions, visibility_engine.dynamic_radii,
            visibility_engine.dynamic_body_ids_array,
            visibility_engine.num_dynamic_objects,
            SEEKER_BODY_ID, -1,
            epsilon, max_steps, 15.0
        )
        distances_sphere.append(dist_sph)
    
    distances_direct = np.array(distances_direct)
    distances_sphere = np.array(distances_sphere)
    
    # 統計
    diff = np.abs(distances_direct - distances_sphere)
    mean_diff = diff.mean()
    max_diff = diff.max()
    
    print(f"{label}:")
    print(f"  Direct avg: {distances_direct.mean():.3f}m")
    print(f"  Sphere avg: {distances_sphere.mean():.3f}m")
    print(f"  Mean diff:  {mean_diff:.3f}m")
    print(f"  Max diff:   {max_diff:.3f}m")
    print()
