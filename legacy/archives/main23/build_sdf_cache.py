#!/usr/bin/env python3
"""
SDF距離場を構築してキャッシュに保存
"""

import os
import sys

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
    ENV_BOUNDS,
    LIDAR_MAX_DIST,
    SDF_CELL_SIZE,
    VisibilityEngine,
    build_sdf_distance_field,
    create_cell_grid,
    save_sdf_distance_field,
)

print("=" * 80)
print("Building SDF Distance Field")
print("=" * 80)

# 初期化
xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# VisibilityEngine
visibility_engine = VisibilityEngine(model, data, epsilon=0.1, max_steps=15, max_dist=LIDAR_MAX_DIST)

# Body IDs
s0_body = model.body("seeker_body").id
h1_body = model.body("hider1_body").id
h2_body = model.body("hider2_body").id
box1_body = model.body("box1_body").id
box2_body = model.body("box2_body").id
ramp_body = model.body("ramp_body").id

visibility_engine.set_bodies(s0_body, h1_body, h2_body, box1_body, box2_body, ramp_body)

# SDF場構築用セルグリッドを作成
print(f"\nCreating cell grid (bounds={ENV_BOUNDS}, cell_size={SDF_CELL_SIZE})...")
sdf_cell_centers, sdf_metadata = create_cell_grid(ENV_BOUNDS, SDF_CELL_SIZE)
print(f"✓ Created {len(sdf_cell_centers)} cells")

# SDF距離場を構築
print("\nBuilding SDF distance field...")
sdf_field = build_sdf_distance_field(visibility_engine, sdf_cell_centers)

# キャッシュに保存
print("\nSaving SDF distance field to cache...")
output_file = "sdf_distance_field.pkl"
save_sdf_distance_field(sdf_field, sdf_cell_centers, sdf_metadata, output_file)

print("\n" + "=" * 80)
print("✓ SDF distance field built and saved successfully!")
print(f"  Shape: {sdf_field.shape}")
print(f"  Min: {sdf_field.min():.4f}, Max: {sdf_field.max():.4f}")
print("=" * 80)
