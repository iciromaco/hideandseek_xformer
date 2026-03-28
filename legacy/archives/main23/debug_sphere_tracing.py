#!/usr/bin/env python3
"""
Sphere Tracing デバッグプログラム
特定のビームについてステップバイステップでSphere Tracingを実行
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

# パス設定
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
from main23_sightmap_optimized import LIDAR_MAX_DIST, VisibilityEngine, sdf_lookup_numba

# ============================================
# 初期化
# ============================================
xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

# エージェント位置
agent_positions = {
    "seeker_body": np.array([3.0, 3.0]),
    "hider1_body": np.array([1.0, 1.0]),
    "hider2_body": np.array([5.0, 5.0]),
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
visibility_engine.dynamic_positions[0] = agent_positions["seeker_body"]
visibility_engine.dynamic_positions[1] = agent_positions["hider1_body"]
visibility_engine.dynamic_positions[2] = agent_positions["hider2_body"]
visibility_engine.dynamic_positions[3] = np.array([2.0, -2.0])
visibility_engine.dynamic_positions[4] = np.array([-2.0, 2.0])
visibility_engine.dynamic_positions[5] = np.array([0.0, 0.0])

# エージェント位置を data に反映
data.qpos[0:3] = [
    agent_positions["seeker_body"][0],
    agent_positions["seeker_body"][1],
    0.5,
]
data.qpos[3:6] = [
    agent_positions["hider1_body"][0],
    agent_positions["hider1_body"][1],
    0.5,
]
data.qpos[6:9] = [
    agent_positions["hider2_body"][0],
    agent_positions["hider2_body"][1],
    0.5,
]

mujoco.mj_forward(model, data)

# SDF場のロード
print("Loading SDF field...")
cache_file = Path("sdf_distance_field.pkl")
if cache_file.exists():
    with open(cache_file, "rb") as f:
        data_dict = pickle.load(f)
        visibility_engine.sdf_field = data_dict.get("sdf_field")
        print(f"✓ SDF field loaded: shape={visibility_engine.sdf_field.shape}")
else:
    print("⚠ SDF field cache not found")
    sys.exit(1)

# ============================================
# Sphere Tracing デバッグ（特定ビームについて）
# ============================================

# 視点と方向
viewpoint = np.array([-4.0, -4.0], dtype=np.float32)
yaw = np.pi / 4

# ビーム1を選択（問題のあるビーム）
beam_index = 1
surround = np.linspace(0, 2 * np.pi, 8, endpoint=False)
front = np.linspace(-np.pi / 6, np.pi / 6, 5)
lidar_angles = np.unique(np.concatenate([surround, front]))

lidar_angle = lidar_angles[beam_index]
print("\n" + "=" * 80)
print(f"Sphere Tracing Debug - Beam {beam_index}")
print("=" * 80)
print(f"Lidar angle: {np.degrees(lidar_angle):.1f}° = {lidar_angle:.4f} rad")

# ビーム方向の計算
lidar_angle_cos = np.cos(lidar_angle)
lidar_angle_sin = np.sin(lidar_angle)

cy = np.cos(yaw)
sy = np.sin(yaw)
beam_cos = lidar_angle_cos * cy - lidar_angle_sin * sy
beam_sin = lidar_angle_cos * sy + lidar_angle_sin * cy

print(f"Beam direction (before norm): ({beam_cos:.4f}, {beam_sin:.4f})")

# 正規化
dir_mag = np.sqrt(beam_cos**2 + beam_sin**2)
dir_x_norm = beam_cos / dir_mag
dir_y_norm = beam_sin / dir_mag

print(f"Beam direction (normalized): ({dir_x_norm:.4f}, {dir_y_norm:.4f}), magnitude={dir_mag:.4f}")
print(f"Viewpoint: ({viewpoint[0]:.4f}, {viewpoint[1]:.4f})")

# ============================================
# Sphere Tracing シミュレーション
# ============================================

grid_size = visibility_engine.sdf_field.shape
cell_size = 12.0 / (grid_size[0] - 1)
epsilon = 0.1
max_steps = 15
max_dist = LIDAR_MAX_DIST

print("\nSphere Tracing parameters:")
print(f"  Grid size: {grid_size}")
print(f"  Cell size: {cell_size:.6f}")
print(f"  Epsilon: {epsilon}")
print(f"  Max steps: {max_steps}")
print(f"  Max dist: {max_dist}")

print("\n" + "-" * 80)
print(f"{'Step':<6} {'X':<10} {'Y':<10} {'d_static':<12} {'d_dynamic':<12} {'d_min':<12} {'total_d':<12}")
print("-" * 80)

curr_x = float(viewpoint[0])
curr_y = float(viewpoint[1])
total_d = 0.0
hit = False

for step in range(max_steps):
    # SDF値の取得
    d_static = sdf_lookup_numba(
        curr_x,
        curr_y,
        visibility_engine.sdf_field,
        grid_size[0],
        grid_size[1],
        cell_size,
    )

    # 負の値の処理
    if d_static < 0:
        d_static_orig = d_static
        d_static = epsilon
        print(f"[WARN] Negative SDF at step {step}: {d_static_orig:.4f} -> {epsilon}")

    # 動的SDF
    d_dynamic = 1e6
    for i in range(visibility_engine.num_dynamic_objects):
        if visibility_engine.dynamic_body_ids_array[i] != 5 and visibility_engine.dynamic_body_ids_array[i] != -1:
            dx = visibility_engine.dynamic_positions[i, 0] - curr_x
            dy = visibility_engine.dynamic_positions[i, 1] - curr_y
            dist = np.sqrt(dx * dx + dy * dy) - visibility_engine.dynamic_radii[i]
            if dist < d_dynamic:
                d_dynamic = dist

    # 最小距離
    d = min(d_static, d_dynamic)

    print(f"{step:<6} {curr_x:<10.4f} {curr_y:<10.4f} {d_static:<12.4f} {d_dynamic:<12.4f} {d:<12.4f} {total_d:<12.4f}")

    # 収束判定
    if d < epsilon:
        hit = True
        print(f"[HIT] Converged at step {step}, total_d={total_d:.4f}")
        break

    # 前進
    total_d += d
    curr_x += dir_x_norm * d
    curr_y += dir_y_norm * d

    # 距離チェック
    if total_d > max_dist:
        print(f"[MAX_DIST] Exceeded max_dist at step {step}, total_d={total_d:.4f}")
        break

print("-" * 80)
print("\nFinal result:")
print(f"  Hit: {hit}")
print(f"  Total distance: {total_d:.4f}")
print(f"  Final position: ({curr_x:.4f}, {curr_y:.4f})")
print(f"  Expected max_dist: {max_dist}")

if total_d >= max_dist:
    print(f"  Result: {max_dist} (max_dist limit)")
else:
    print(f"  Result: {total_d:.4f}")

print("\n" + "=" * 80)
