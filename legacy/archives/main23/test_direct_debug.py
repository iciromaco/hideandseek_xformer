#!/usr/bin/env python3
"""
ライダーの Direct method を詳しくデバッグ
"""

import os

# Skip GL setup for headless rendering
os.environ["MUJOCO_GL"] = "glfw"  # Use default

import mujoco
import numpy as np
from main23_sightmap_optimized import (
    VisibilityEngine,
    cast_ray_direct_numba,
    ray_segment_intersection_numba,
)

# ===== MuJoCo環境のセットアップ =====
xml_path = "hideandseek.py"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ===== ライダーの視点を設定 =====
# seeker ボディの位置：(-4, -4)，yaw=45°
seeker_body_id = 5  # SEEKER_BODY_ID
seeker_pos = np.array([-4.0, -4.0, 0.5])
seeker_yaw = np.radians(45)  # 45°

data.body(seeker_body_id).xpos[:2] = seeker_pos[:2]

# ===== VisibilityEngine の初期化 =====
visibility_engine = VisibilityEngine(model, data)

# ===== ライダービームのうち、失敗しているビームをテスト =====
# Beam 1, 3, 5, 6, 7, 8, 9, 10, 11 が Direct でも 15.0 を返している

lidar_angles = np.array([-30, -15, 0, 15, 30, 45, 90, 135, 180, 225, 270, 315], dtype=float) * np.pi / 180
lidar_angles = np.sort(lidar_angles)

print("=" * 80)
print("Direct Method デバッグ - ビーム別詳細")
print("=" * 80)

for beam_idx in [1, 3, 5, 6]:
    angle = lidar_angles[beam_idx]
    dir_x = np.cos(seeker_yaw + angle)
    dir_y = np.sin(seeker_yaw + angle)

    print(f"\nBeam {beam_idx}:")
    print(f"  Angle: {np.degrees(angle):.1f}°")
    print(f"  Direction: ({dir_x:.4f}, {dir_y:.4f})")

    # キャスト実行
    dist, hit = cast_ray_direct_numba(
        seeker_pos[0],
        seeker_pos[1],
        dir_x,
        dir_y,
        visibility_engine.positions,
        visibility_engine.radii,
        visibility_engine.body_ids,
        visibility_engine.num_objects,
        visibility_engine.wall_segments,
        visibility_engine.num_walls,
        seeker_body_id,
        -1,
        15.0,
    )

    print(f"  Result Distance: {dist:.4f}")
    print(f"  Hit: {hit}")

    # 個別に壁との交点を確認
    print("  Wall intersection details:")
    for w_idx, wall in enumerate(visibility_engine.wall_segments[:5]):  # 最初の5つだけ
        d = ray_segment_intersection_numba(
            seeker_pos[0],
            seeker_pos[1],
            dir_x,
            dir_y,
            wall[0],
            wall[1],
            wall[2],
            wall[3],
        )
        if d < 15.0:
            print(f"    Wall {w_idx}: ({wall[0]:.2f},{wall[1]:.2f}) - ({wall[2]:.2f},{wall[3]:.2f}) -> distance={d:.4f}")

print("\n" + "=" * 80)
