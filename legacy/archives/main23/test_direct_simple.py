#!/usr/bin/env python3
"""
Direct method の実際の動作確認
"""

import numpy as np
from main23_sightmap_optimized import cast_ray_direct_numba

# テスト環境設定
# 壁：y=2 の水平線、x in [0, 4]
wall_segments = np.array([
    [0.0, 2.0, 4.0, 2.0]
], dtype=np.float64)

# 円：なし
positions = np.zeros((0, 2), dtype=np.float64)
radii = np.zeros(0, dtype=np.float64)
body_ids = np.zeros(0, dtype=np.int32)

# レイ：(0, 0) -> (1, 1) 方向
start_x, start_y = 0.0, 0.0
dir_x, dir_y = 1.0, 1.0

# キャスト実行
dist, hit = cast_ray_direct_numba(
    start_x, start_y, dir_x, dir_y,
    positions, radii, body_ids, 0,  # num_objects = 0
    wall_segments, 1,  # num_walls = 1
    -1, -1,  # exclude_ids
    15.0  # max_dist
)

print(f"Distance: {dist:.4f}")
print(f"Hit: {hit}")
print(f"Expected: ~2.0 (diagonal ray hits horizontal line at t=2)")

if abs(dist - 2.0) < 0.01:
    print("✓ PASS - Direct method working correctly!")
else:
    print(f"✗ FAIL - Got {dist}, expected ~2.0")
