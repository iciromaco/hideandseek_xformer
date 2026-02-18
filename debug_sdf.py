#!/usr/bin/env python3
"""
SDF場のデバッグプログラム
座標系とアクセス方法が正しいか検証
"""

import os
import sys
import numpy as np
import pickle
from pathlib import Path

# パスの設定
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

try:
    import main18_optimization as base_config
except ImportError as e:
    print(f"Error importing main18_optimization: {e}")
    raise

import mujoco
from main23_sightmap_optimized import VisibilityEngine, LIDAR_MAX_DIST

# ============================================
# 初期化
# ============================================
xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# エージェント位置
agent_positions = {
    'seeker_body': np.array([3.0, 3.0]),
    'hider1_body': np.array([1.0, 1.0]),
    'hider2_body': np.array([5.0, 5.0])
}

# visibility engineの初期化
visibility_engine = VisibilityEngine(model, data, epsilon=0.1, max_steps=15, max_dist=LIDAR_MAX_DIST)

# Body IDの設定
s0_body = model.body("seeker_body").id
h1_body = model.body("hider1_body").id
h2_body = model.body("hider2_body").id
box1_body = model.body("box1_body").id
box2_body = model.body("box2_body").id
ramp_body = model.body("ramp_body").id

visibility_engine.set_bodies(s0_body, h1_body, h2_body, box1_body, box2_body, ramp_body)

# 動的オブジェクト位置
visibility_engine.dynamic_positions[0] = agent_positions['seeker_body']
visibility_engine.dynamic_positions[1] = agent_positions['hider1_body']
visibility_engine.dynamic_positions[2] = agent_positions['hider2_body']
visibility_engine.dynamic_positions[3] = np.array([2.0, -2.0])
visibility_engine.dynamic_positions[4] = np.array([-2.0, 2.0])
visibility_engine.dynamic_positions[5] = np.array([0.0, 0.0])

# エージェント位置を data に反映
data.qpos[0:3] = [agent_positions['seeker_body'][0], agent_positions['seeker_body'][1], 0.5]
data.qpos[3:6] = [agent_positions['hider1_body'][0], agent_positions['hider1_body'][1], 0.5]
data.qpos[6:9] = [agent_positions['hider2_body'][0], agent_positions['hider2_body'][1], 0.5]

mujoco.mj_forward(model, data)

# ============================================
# SDF場のロード/構築
# ============================================

print("=" * 80)
print("SDF FIELD DEBUG")
print("=" * 80)

cache_file = Path("sdf_distance_field.pkl")
if cache_file.exists():
    print("Loading existing SDF field...")
    with open(cache_file, "rb") as f:
        data_dict = pickle.load(f)
        visibility_engine.sdf_field = data_dict.get("sdf_field")
        print(f"✓ SDF field loaded: shape={visibility_engine.sdf_field.shape}")
else:
    print("SDF field not found. Skipping...")

# ============================================
# グリッド座標系のテスト
# ============================================

print("\nTesting SDF grid coordinate system:")
print("-" * 80)

if visibility_engine.sdf_field is not None:
    grid_size = visibility_engine.sdf_field.shape
    print(f"SDF field shape: {grid_size}")
    
    # cell_sizeの計算
    # SDF場は (-6, 6) の範囲をカバーしているはず
    # つまり幅は 12.0
    cell_size = 12.0 / (grid_size[0] - 1)
    print(f"Cell size (calculated): {cell_size:.6f}")
    
    # テスト点1: (-4, -4) - 視点
    p_x, p_y = -4.0, -4.0
    grid_x = (p_x + 6.0) / cell_size
    grid_y = (p_y + 6.0) / cell_size
    print(f"\nPoint (-4.0, -4.0):")
    print(f"  Grid coords: ({grid_x:.2f}, {grid_y:.2f})")
    
    x = int(grid_x + 0.5)
    y = int(grid_y + 0.5)
    print(f"  Rounded grid indices: ({x}, {y})")
    
    if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
        sdf_val = visibility_engine.sdf_field[x, y]
        print(f"  SDF value: {sdf_val:.4f}")
    else:
        print(f"  Out of bounds!")
    
    # テスト点2: (0, 0)
    p_x, p_y = 0.0, 0.0
    grid_x = (p_x + 6.0) / cell_size
    grid_y = (p_y + 6.0) / cell_size
    print(f"\nPoint (0.0, 0.0):")
    print(f"  Grid coords: ({grid_x:.2f}, {grid_y:.2f})")
    
    x = int(grid_x + 0.5)
    y = int(grid_y + 0.5)
    print(f"  Rounded grid indices: ({x}, {y})")
    
    if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
        sdf_val = visibility_engine.sdf_field[x, y]
        print(f"  SDF value: {sdf_val:.4f}")
    else:
        print(f"  Out of bounds!")
    
    # テスト点3: (3.0, 3.0) - seeker位置
    p_x, p_y = 3.0, 3.0
    grid_x = (p_x + 6.0) / cell_size
    grid_y = (p_y + 6.0) / cell_size
    print(f"\nPoint (3.0, 3.0) - Seeker position:")
    print(f"  Grid coords: ({grid_x:.2f}, {grid_y:.2f})")
    
    x = int(grid_x + 0.5)
    y = int(grid_y + 0.5)
    print(f"  Rounded grid indices: ({x}, {y})")
    
    if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
        sdf_val = visibility_engine.sdf_field[x, y]
        print(f"  SDF value: {sdf_val:.4f}")
    else:
        print(f"  Out of bounds!")
    
    # テスト点4: (6.0, 6.0) - 境界
    p_x, p_y = 6.0, 6.0
    grid_x = (p_x + 6.0) / cell_size
    grid_y = (p_y + 6.0) / cell_size
    print(f"\nPoint (6.0, 6.0) - Boundary:")
    print(f"  Grid coords: ({grid_x:.2f}, {grid_y:.2f})")
    
    x = int(grid_x + 0.5)
    y = int(grid_y + 0.5)
    print(f"  Rounded grid indices: ({x}, {y})")
    
    if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
        sdf_val = visibility_engine.sdf_field[x, y]
        print(f"  SDF value: {sdf_val:.4f}")
    else:
        print(f"  Out of bounds!")
    
    # ============================================
    # SDF場の統計情報
    # ============================================
    print(f"\n" + "=" * 80)
    print("SDF Field Statistics:")
    print("=" * 80)
    print(f"Min value: {visibility_engine.sdf_field.min():.4f}")
    print(f"Max value: {visibility_engine.sdf_field.max():.4f}")
    print(f"Mean value: {visibility_engine.sdf_field.mean():.4f}")
    print(f"Std value: {visibility_engine.sdf_field.std():.4f}")
    
    # 負の値（内部）の割合
    negative_count = np.sum(visibility_engine.sdf_field < 0)
    positive_count = np.sum(visibility_engine.sdf_field > 0)
    total_count = visibility_engine.sdf_field.size
    print(f"\nNegative (inside): {negative_count} ({100*negative_count/total_count:.1f}%)")
    print(f"Positive (outside): {positive_count} ({100*positive_count/total_count:.1f}%)")
    
    print("\n" + "=" * 80)

else:
    print("SDF field is None - cannot test")

print("\nDebug complete!")
