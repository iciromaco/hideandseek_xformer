#!/usr/bin/env python3
"""
Sphere Tracing のテスト（MuJoCo 環境を初期化してテスト）
"""

import sys
sys.path.insert(0, ".")
import pickle
import numpy as np
import mujoco
from hideandseek import XML_CONTENT
from main23_sightmap_optimized import VisibilityEngine

# MuJoCo モデルとデータを初期化
try:
    model = mujoco.MjModel.from_xml_string(XML_CONTENT)
    data = mujoco.MjData(model)
    print(f"✓ MuJoCo environment initialized")
except Exception as e:
    print(f"✗ Failed to initialize MuJoCo: {e}")
    exit(1)

# VisibilityEngine を初期化
try:
    engine = VisibilityEngine(model, data)
    print(f"✓ VisibilityEngine initialized")
except Exception as e:
    print(f"✗ Failed to initialize VisibilityEngine: {e}")
    exit(1)

# SDF を読み込んで設定
try:
    with open("sdf_distance_field.pkl", "rb") as f:
        data_dict = pickle.load(f)
        engine.sdf_field = data_dict.get("sdf_field")
    print(f"✓ SDF loaded: {engine.sdf_field.shape if engine.sdf_field is not None else 'None'}")
except Exception as e:
    print(f"✗ Failed to load SDF: {e}")
    exit(1)

# 動的オブジェクトの初期化（静的環境なので不要だが、numba関数に必須）
engine.dynamic_positions = np.zeros((6, 2), dtype=np.float32)
engine.dynamic_radii = np.zeros(6, dtype=np.float32)
engine.dynamic_body_ids_array = np.zeros(6, dtype=np.int32)
engine.num_dynamic_objects = 0

# テストケース
test_cases = [
    # (x, y, beam_angle, description)
    (0.0, 0.0, np.pi/2, "中央から北方向"),
    (0.0, 0.0, 0.0, "中央から東方向"),
    (0.0, 0.0, np.pi, "中央から西方向"),
    (0.0, 0.0, -np.pi/2, "中央から南方向"),
    (2.0, 1.5, 0.0, "内壁南側から東方向"),
    (2.0, 1.5, np.pi, "内壁南側から西方向"),
]

print("\n[SPHERE TRACING TEST]")
print(f"{'Position':<20} {'Angle (deg)':>12} {'Distance (m)':>14} {'Result':<15} Description")
print("-" * 90)

for x, y, angle, desc in test_cases:
    try:
        # Lidar ビームをレイキャスト
        ray_len, hit = engine.cast_ray(np.array([x, y, 0.0]), np.array([np.cos(angle), np.sin(angle)]))
        
        # 結果判定
        if ray_len >= 14.9:
            result = "✗ Max dist"
        elif ray_len > 0.1:
            result = "✓ Hit"
        else:
            result = "? Zero"
        
        angle_deg = np.degrees(angle)
        print(f"({x:6.2f}, {y:6.2f}){angle_deg:>12.1f}°{ray_len:>14.4f}m {result:<15} {desc}")
    except Exception as e:
        angle_deg = np.degrees(angle)
        print(f"({x:6.2f}, {y:6.2f}){angle_deg:>12.1f}° ERROR: {str(e)[:30]}")

# Lidar スキャン（12ビーム）をテスト
print("\n[LIDAR SCAN FROM CENTER (0, 0)]")
print(f"{'Beam':>5} {'Angle (deg)':>12} {'Distance (m)':>14} {'Status':<15}")
print("-" * 60)

try:
    lidar_readings = []
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    
    for i, angle in enumerate(angles):
        ray_len, hit = engine.cast_ray(np.array([0.0, 0.0, 0.0]), np.array([np.cos(angle), np.sin(angle)]))
        lidar_readings.append(ray_len)
        
        # 距離が reasonable か判定
        if ray_len < 14.9:
            status = "✓ Hit"
        else:
            status = "✗ Max dist"
        
        angle_deg = np.degrees(angle)
        print(f"{i:>5} {angle_deg:>12.1f}°{ray_len:>14.4f}m {status:<15}")
    
    print(f"\nMin: {min(lidar_readings):.4f}m, Max: {max(lidar_readings):.4f}m, Mean: {np.mean(lidar_readings):.4f}m")
    
    # 全て max_dist なら失敗
    max_dist_count = sum(1 for r in lidar_readings if r >= 14.9)
    if max_dist_count == len(lidar_readings):
        print("✗ CRITICAL: All beams returned max_dist - SDF is still broken")
    elif max_dist_count > 0:
        print(f"⚠ WARNING: {max_dist_count}/{len(lidar_readings)} beams returned max_dist")
    else:
        print("✓ All beams detected obstacles correctly")
        
except Exception as e:
    print(f"✗ Lidar scan failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
