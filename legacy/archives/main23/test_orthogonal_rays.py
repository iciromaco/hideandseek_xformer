#!/usr/bin/env python3
"""
交線法修正の効果確認
垂直線・水平線に対する特殊なテストケース
"""

import numpy as np
from main23_sightmap_optimized import ray_segment_intersection_numba

print("=" * 80)
print("交線法修正の効果確認")
print("=" * 80)

# テストケース1：垂直方向のレイが水平壁を撃つ（視点を調整）
print("\n[Test] 垂直方向のレイが水平壁を撃つ")
print("Lidar viewpoint: (0, 0)")
print("Ray direction: (0, 1) - 真上")
print("Wall: at y=5, x in [-2, 2]")

start_x, start_y = 0.0, 0.0
dir_x, dir_y = 0.0, 1.0  # 真上

# 水平線
seg_x1, seg_y1 = -2.0, 5.0
seg_x2, seg_y2 = 2.0, 5.0

dist = ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2)

expected_dist = 5.0 - 0.0  # = 5.0
print(f"\nResult: {dist:.4f}")
print(f"Expected: {expected_dist:.4f}")

if abs(dist - expected_dist) < 0.01:
    print("✓ PASS - 垂直レイが水平壁を正しく撃っている")
else:
    print(f"✗ FAIL - 差分: {abs(dist - expected_dist):.4f}")

# テストケース2：水平方向のレイが垂直壁を撃つ
print("\n" + "=" * 80)
print("[Test] 水平方向のレイが垂直壁を撃つ")
print("Lidar viewpoint: (0, 0)")
print("Ray direction: (1, 0) - 真右")
print("Wall: at x=5, y in [-2, 2]")

start_x, start_y = 0.0, 0.0
dir_x, dir_y = 1.0, 0.0  # 真右

# 垂直線
seg_x1, seg_y1 = 5.0, -2.0
seg_x2, seg_y2 = 5.0, 2.0

dist = ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2)

expected_dist = 5.0 - 0.0  # = 5.0
print(f"\nResult: {dist:.4f}")
print(f"Expected: {expected_dist:.4f}")

if abs(dist - expected_dist) < 0.01:
    print("✓ PASS - 水平レイが垂直壁を正しく撃っている")
else:
    print(f"✗ FAIL - 差分: {abs(dist - expected_dist):.4f}")

# テストケース3：斜めレイが垂直壁を撃つ
print("\n" + "=" * 80)
print("[Test] 斜めレイが垂直壁を撃つ")
print("Lidar viewpoint: (0, 0)")
print("Ray direction: (1, 1) - 45度")
print("Wall: at x=3, y in [-5, 5]")

start_x, start_y = 0.0, 0.0
dir_x, dir_y = 1.0, 1.0  # 45度

# 垂直線
seg_x1, seg_y1 = 3.0, -5.0
seg_x2, seg_y2 = 3.0, 5.0

dist = ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2)

# レイ: (0,0) + t*(1,1) = (t, t)
# 壁: x = 3 → t = 3, y = 3 ✓ (y in [-5, 5])
# 距離 = t * sqrt(2) = 3 * sqrt(2)
expected_dist = 3.0 * np.sqrt(2)
print(f"\nResult: {dist:.4f}")
print(f"Expected: {expected_dist:.4f}")

if abs(dist - expected_dist) < 0.01:
    print("✓ PASS - 斜めレイが垂直壁を正しく撃っている")
else:
    print(f"✗ FAIL - 差分: {abs(dist - expected_dist):.4f}")

print("\n" + "=" * 80)
