#!/usr/bin/env python3
"""
ray_segment_intersection_numba のテスト
垂直線・水平線との交差を検証
"""

import numpy as np
import math
from numba import njit

# 修正版 ray_segment_intersection_numba をコピー
@njit(cache=True, fastmath=True)
def ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y,
                                   seg_x1, seg_y1, seg_x2, seg_y2):
    """直線と線分の交点を計算（最短交点距離を返す）"""
    seg_dx = seg_x2 - seg_x1
    seg_dy = seg_y2 - seg_y1
    
    seg_len_sq = seg_dx * seg_dx + seg_dy * seg_dy
    if seg_len_sq < 1e-12:
        return 1e6
    
    fx = seg_x1 - start_x
    fy = seg_y1 - start_y
    
    cross = dir_x * seg_dy - dir_y * seg_dx
    
    if abs(cross) > 1e-10:
        t = (fx * seg_dy - fy * seg_dx) / cross
        if t < 1e-6:
            return 1e6
        u = (fx * dir_y - fy * dir_x) / cross
        if -1e-6 <= u <= 1.0 + 1e-6:
            return t
        else:
            return 1e6
    else:
        # 平行またはレイと線分が同一直線上
        eps = 1e-10
        is_vertical = abs(seg_dx) < eps
        is_horizontal = abs(seg_dy) < eps
        
        if is_vertical:
            if abs(dir_x) < eps:
                return 1e6
            t = (seg_x1 - start_x) / dir_x
            if t < 1e-6:
                return 1e6
            y_at_t = start_y + t * dir_y
            y_min = min(seg_y1, seg_y2)
            y_max = max(seg_y1, seg_y2)
            if y_min - 1e-6 <= y_at_t <= y_max + 1e-6:
                return t
            else:
                return 1e6
        
        elif is_horizontal:
            if abs(dir_y) < eps:
                return 1e6
            t = (seg_y1 - start_y) / dir_y
            if t < 1e-6:
                return 1e6
            x_at_t = start_x + t * dir_x
            x_min = min(seg_x1, seg_x2)
            x_max = max(seg_x1, seg_x2)
            if x_min - 1e-6 <= x_at_t <= x_max + 1e-6:
                return t
            else:
                return 1e6
        
        else:
            return 1e6

# テストケース
print("=" * 80)
print("Ray-Segment Intersection Tests")
print("=" * 80)

# Test 1: 垂直線（斜めレイが垂直線と交差）
print("\nTest 1: Diagonal ray hits vertical line at x=0")
start_x, start_y = -4.0, -4.0
dir_x, dir_y = 1.0, 1.0  # 45度方向
dir_mag = np.sqrt(dir_x**2 + dir_y**2)
dir_x, dir_y = dir_x / dir_mag, dir_y / dir_mag
seg_x1, seg_y1 = 0.0, -2.0  # 垂直線
seg_x2, seg_y2 = 0.0, 2.0
result = ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2)
expected = np.sqrt(2) * 4.0  # (-4, -4) から (0, 0) へは距離 4√2
print(f"  Result: {result:.4f}, Expected: {expected:.4f}")
print(f"  ✓ PASS" if abs(result - expected) < 0.01 else f"  ✗ FAIL")

# Test 2: 水平線（斜めレイが水平線と交差）
print("\nTest 2: Diagonal ray hits horizontal line at y=0")
start_x, start_y = -4.0, -4.0
dir_x, dir_y = 1.0, 1.0  # 45度方向
dir_mag = np.sqrt(dir_x**2 + dir_y**2)
dir_x, dir_y = dir_x / dir_mag, dir_y / dir_mag
seg_x1, seg_y1 = -2.0, 0.0  # 水平線
seg_x2, seg_y2 = 2.0, 0.0
result = ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2)
expected = np.sqrt(2) * 4.0  # (-4, -4) から (0, 0) へは距離 4√2
print(f"  Result: {result:.4f}, Expected: {expected:.4f}")
print(f"  ✓ PASS" if abs(result - expected) < 0.01 else f"  ✗ FAIL")

# Test 3: 対角線レイが垂直線と交差
print("\nTest 3: Diagonal ray hits vertical line")
start_x, start_y = -4.0, -4.0
dir_x, dir_y = 1.0, 1.0  # 45度
dir_mag = np.sqrt(dir_x**2 + dir_y**2)
dir_x, dir_y = dir_x / dir_mag, dir_y / dir_mag
seg_x1, seg_y1 = 0.0, -2.0
seg_x2, seg_y2 = 0.0, 2.0
result = ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2)
expected = np.sqrt((4.0)**2 + (4.0)**2)  # (-4,-4) から (0,0) まで
print(f"  Result: {result:.4f}, Expected: {expected:.4f}")
print(f"  ✓ PASS" if abs(result - expected) < 0.01 else f"  ✗ FAIL")

# Test 4: レイが線分の端点で交差
print("\nTest 4: Ray hits endpoint of segment")
start_x, start_y = -4.0, -4.0
dir_x, dir_y = 1.0, 1.0
dir_mag = np.sqrt(dir_x**2 + dir_y**2)
dir_x, dir_y = dir_x / dir_mag, dir_y / dir_mag
seg_x1, seg_y1 = 0.0, 0.0  # 線分の端点がレイと交差する点
seg_x2, seg_y2 = 1.0, 1.0
result = ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2)
expected = np.sqrt(2) * 4.0  # (-4,-4) から (0,0) まで
print(f"  Result: {result:.4f}, Expected: {expected:.4f}")
print(f"  ✓ PASS" if abs(result - expected) < 0.01 else f"  ✗ FAIL")

# Test 5: 垂直線（外壁）との交差
print("\nTest 5: Ray hits vertical wall at x=6.1")
start_x, start_y = -4.0, -4.0
dir_x, dir_y = 1.0, 0.0  # 水平に右へ
seg_x1, seg_y1 = 6.1, -6.0
seg_x2, seg_y2 = 6.1, 6.0
result = ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2)
expected = 6.1 - (-4.0)  # 10.1
print(f"  Result: {result:.4f}, Expected: {expected:.4f}")
print(f"  ✓ PASS" if abs(result - expected) < 0.01 else f"  ✗ FAIL")

print("\n" + "=" * 80)
