#!/usr/bin/env python3
"""
修正版の交線法が実際に使われているか確認
（Python直呼び出しで、Numbaコンパイルをバイパス）
"""

import numpy as np


# 修正版と同じロジックをPythonで実装
def ray_segment_intersection_python_debug(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2):
    """Python版の修正版交線法（デバッグ付き）"""
    seg_dx = seg_x2 - seg_x1
    seg_dy = seg_y2 - seg_y1

    # 線分が退化している
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq < 1e-12:
        print("  DEBUG: Degenerate segment")
        return 1e6

    # 2つの直線の交点計算
    p1_x = start_x
    p1_y = start_y
    p2_x = start_x + dir_x
    p2_y = start_y + dir_y
    p3_x = seg_x1
    p3_y = seg_y1
    p4_x = seg_x2
    p4_y = seg_y2

    denom = (p1_x - p2_x) * (p3_y - p4_y) - (p1_y - p2_y) * (p3_x - p4_x)

    eps = 1e-10
    if abs(denom) < eps:
        print(f"  DEBUG: Parallel (denom={denom:.6e})")
        return 1e6

    t1_num = (p1_x * p2_y - p1_y * p2_x) * (p3_x - p4_x) - (p1_x - p2_x) * (p3_x * p4_y - p3_y * p4_x)
    t2_num = (p1_x * p2_y - p1_y * p2_x) * (p3_y - p4_y) - (p1_y - p2_y) * (p3_x * p4_y - p3_y * p4_x)

    intersect_x = t1_num / denom
    intersect_y = t2_num / denom

    print(f"  DEBUG: Intersection point: ({intersect_x:.4f}, {intersect_y:.4f})")

    if abs(dir_x) > eps:
        t = (intersect_x - start_x) / dir_x
    elif abs(dir_y) > eps:
        t = (intersect_y - start_y) / dir_y
    else:
        print("  DEBUG: Zero direction vector")
        return 1e6

    print(f"  DEBUG: Ray parameter t={t:.4f}")

    if t < 1e-6:
        print("  DEBUG: Behind ray (t < 1e-6)")
        return 1e6

    if abs(seg_dx) > eps:
        u = (intersect_x - seg_x1) / seg_dx
    elif abs(seg_dy) > eps:
        u = (intersect_y - seg_y1) / seg_dy
    else:
        print("  DEBUG: Degenerate segment (seg_dy, seg_dx both zero)")
        return 1e6

    print(f"  DEBUG: Segment parameter u={u:.4f} (valid range: [-1e-6, 1+1e-6])")

    if -1e-6 <= u <= 1.0 + 1e-6:
        print("  DEBUG: HIT!")
        return t
    else:
        print("  DEBUG: Out of segment range")
        return 1e6


# Python版の修正版交線法
def ray_segment_intersection_python(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2):
    """Python版の修正版交線法"""
    seg_dx = seg_x2 - seg_x1
    seg_dy = seg_y2 - seg_y1

    # 線分が退化している
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq < 1e-12:
        return 1e6

    # 2つの直線の交点計算
    p1_x = start_x
    p1_y = start_y
    p2_x = start_x + dir_x
    p2_y = start_y + dir_y
    p3_x = seg_x1
    p3_y = seg_y1
    p4_x = seg_x2
    p4_y = seg_y2

    denom = (p1_x - p2_x) * (p3_y - p4_y) - (p1_y - p2_y) * (p3_x - p4_x)

    eps = 1e-10
    if abs(denom) < eps:
        return 1e6

    t1_num = (p1_x * p2_y - p1_y * p2_x) * (p3_x - p4_x) - (p1_x - p2_x) * (p3_x * p4_y - p3_y * p4_x)
    t2_num = (p1_x * p2_y - p1_y * p2_x) * (p3_y - p4_y) - (p1_y - p2_y) * (p3_x * p4_y - p3_y * p4_x)

    intersect_x = t1_num / denom
    intersect_y = t2_num / denom

    if abs(dir_x) > eps:
        t = (intersect_x - start_x) / dir_x
    elif abs(dir_y) > eps:
        t = (intersect_y - start_y) / dir_y
    else:
        return 1e6

    if t < 1e-6:
        return 1e6

    if abs(seg_dx) > eps:
        u = (intersect_x - seg_x1) / seg_dx
    elif abs(seg_dy) > eps:
        u = (intersect_y - seg_y1) / seg_dy
    else:
        return 1e6

    if -1e-6 <= u <= 1.0 + 1e-6:
        return t
    else:
        return 1e6


# テスト：実環境の壁と同じ設定でテスト
print("=" * 80)
print("修正版の交線法をPythonで直呼び出しテスト")
print("=" * 80)

# 視点
viewpoint = np.array([-4.1, -4.3])
yaw = np.pi / 3  # 60度

# Beam 5 (例：90度 - 真上)
lidar_angle = np.pi / 2  # 90度
cy = np.cos(yaw)
sy = np.sin(yaw)
cos_a = np.cos(lidar_angle)
sin_a = np.sin(lidar_angle)

beam_x = cos_a * cy - sin_a * sy
beam_y = cos_a * sy + sin_a * cy

print(f"\nViewpoint: {viewpoint}, Yaw: {np.degrees(yaw):.1f}°")
print(f"Beam angle: {np.degrees(lidar_angle):.1f}° (relative to lidar)")
print(f"Beam direction after yaw rotation: ({beam_x:.4f}, {beam_y:.4f})")

# トップウォール (y=6.15, x in [-3.10, 3.10])
seg_x1, seg_y1 = -3.10, 6.15
seg_x2, seg_y2 = 3.10, 6.15

print(f"\nTop wall: y=6.15, x in [{seg_x1}, {seg_x2}]")
dist = ray_segment_intersection_python_debug(viewpoint[0], viewpoint[1], beam_x, beam_y, seg_x1, seg_y1, seg_x2, seg_y2)

print(f"Result: {dist:.4f}")
if dist >= 1e6 * 0.99:
    print("→ 交点なし（15.0相当）")
else:
    print("→ 交点あり")

# 別のビーム（例：Beam 7, 135度）
print("\n" + "=" * 80)
lidar_angle = 3 * np.pi / 4  # 135度
cos_a = np.cos(lidar_angle)
sin_a = np.sin(lidar_angle)

beam_x = cos_a * cy - sin_a * sy
beam_y = cos_a * sy + sin_a * cy

print(f"\nBeam angle: {np.degrees(lidar_angle):.1f}° (relative to lidar)")
print(f"Beam direction after yaw rotation: ({beam_x:.4f}, {beam_y:.4f})")

# ライトウォール (x=3.10, y in [-6.05, 6.15])
seg_x1, seg_y1 = 3.10, -6.05
seg_x2, seg_y2 = 3.10, 6.15

print(f"\nRight wall: x=3.10, y in [{seg_y1}, {seg_y2}]")
dist = ray_segment_intersection_python_debug(viewpoint[0], viewpoint[1], beam_x, beam_y, seg_x1, seg_y1, seg_x2, seg_y2)

print(f"Result: {dist:.4f}")
if dist >= 1e6 * 0.99:
    print("→ 交点なし（15.0相当）")
else:
    print("→ 交点あり")

print("\n" + "=" * 80)
