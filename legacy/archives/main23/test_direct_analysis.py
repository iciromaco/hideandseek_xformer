#!/usr/bin/env python3
"""
各ビームについて、どの壁と交差しているかを詳しく追跡
"""

import numpy as np

# セットアップ（前と同じ）
from hideandseek import create_env
from main23_sightmap_optimized import VisibilityEngine, cast_ray_direct_numba

model, data = create_env()
visibility_engine = VisibilityEngine(model, data)

seeker_body_id = 5
seeker_x, seeker_y = -4.0, -4.0
seeker_yaw = np.radians(45)

lidar_angles = np.array([-30, -15, 0, 15, 30, 45, 90, 135, 180, 225, 270, 315], dtype=float) * np.pi / 180
lidar_angles = np.sort(lidar_angles)

print("=" * 100)
print("Direct Method - ビーム別壁交差分析")
print("=" * 100)

for beam_idx in range(12):
    angle = lidar_angles[beam_idx]
    dir_x = np.cos(seeker_yaw + angle)
    dir_y = np.sin(seeker_yaw + angle)

    dist, hit = cast_ray_direct_numba(
        seeker_x,
        seeker_y,
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

    print(f"\nBeam {beam_idx:2d} (angle={np.degrees(angle):6.1f}°): direction=({dir_x:7.4f}, {dir_y:7.4f}) -> {dist:7.4f}")

    if dist >= 15.0:
        print("  → 交点なし（15.0以上）")
