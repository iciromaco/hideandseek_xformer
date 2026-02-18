#!/usr/bin/env python3
"""
垂直線・水平線処理のデバッグ
"""

# Test 1 の詳細
print("Test 1 Debug:")
print("  start: (-4, -4), dir: (0, 1)")
print("  segment: (0, -2) -> (0, 2)")

seg_x1, seg_y1 = 0.0, -2.0
seg_x2, seg_y2 = 0.0, 2.0
seg_dx = seg_x2 - seg_x1
seg_dy = seg_y2 - seg_y1
print(f"  seg_d: ({seg_dx}, {seg_dy})")
print(f"  is_vertical: {abs(seg_dx) < 1e-10}")

start_x, start_y = -4.0, -4.0
dir_x, dir_y = 0.0, 1.0

print(f"  |dir_x| < 1e-10: {abs(dir_x) < 1e-10}")

# 垂直線に対して水平レイの場合
if abs(seg_dx) < 1e-10:  # 線分が垂直
    print(f"  線分が垂直")
    if abs(dir_x) < 1e-10:  # レイも垂直
        print(f"  レイも垂直 -> 平行")
    else:
        print(f"  レイは垂直ではない -> 計算を続行")
        t = (seg_x1 - start_x) / dir_x
        print(f"  t = ({seg_x1} - {start_x}) / {dir_x} = {t}")

print("\n---\n")

print("Test 2 Debug:")
print("  start: (-4, -4), dir: (1, 0)")
print("  segment: (-2, 0) -> (2, 0)")

seg_x1, seg_y1 = -2.0, 0.0
seg_x2, seg_y2 = 2.0, 0.0
seg_dx = seg_x2 - seg_x1
seg_dy = seg_y2 - seg_y1
print(f"  seg_d: ({seg_dx}, {seg_dy})")
print(f"  is_horizontal: {abs(seg_dy) < 1e-10}")

start_x, start_y = -4.0, -4.0
dir_x, dir_y = 1.0, 0.0

print(f"  |dir_y| < 1e-10: {abs(dir_y) < 1e-10}")

# 水平線に対して垂直レイの場合
if abs(seg_dy) < 1e-10:  # 線分が水平
    print(f"  線分が水平")
    if abs(dir_y) < 1e-10:  # レイも水平
        print(f"  レイも水平 -> 平行")
    else:
        print(f"  レイは水平ではない -> 計算を続行")
        t = (seg_y1 - start_y) / dir_y
        print(f"  t = ({seg_y1} - {start_y}) / {dir_y} = {t}")
