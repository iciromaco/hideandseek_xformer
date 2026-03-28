#!/usr/bin/env python3
"""
SDF の詳細デバッグ - 座標系と符号を確認
"""

import sys

sys.path.insert(0, "/Users/dan/Desktop/Semi/hideandseek_xformer")

import numpy as np

# XMLからダイレクトに読み込み
from hideandseek import XML_CONTENT
from main23_sightmap_optimized import extract_maze_walls_from_xml, walls_to_segments

maze_walls = extract_maze_walls_from_xml(XML_CONTENT, from_string=True)
static_walls = walls_to_segments(maze_walls)

# テスト点
test_points = [
    (0.0, 0.0, "中央"),
    (5.9, 0.0, "中央、東寄り（壁直前）"),
    (6.0, 0.0, "外壁上（東）"),
    (6.1, 0.0, "外壁外（東）"),
    (-6.0, 0.0, "外壁上（西）"),
    (-6.1, 0.0, "外壁外（西）"),
    (3.0, 1.5, "maze_w0 内部"),
    (3.0, 1.7, "maze_w0 上辺（壁上）"),
    (3.0, 2.0, "maze_w0 外（上側）"),
]

print(f"Static walls loaded: {len(static_walls)}")
print()


# SDF計算用の簡易関数
def get_sdf_static(p, boundary, static_walls):
    # 1. 外壁までの距離
    center = boundary[:2]
    half_size = boundary[2:]
    d = np.abs(p[:2] - center) - half_size
    outside_dist = np.linalg.norm(np.maximum(d, 0.0))
    inside_dist = np.minimum(np.max(d), 0.0)

    print(f"  DEBUG: p={p[:2]}, d={d}, outside_dist={outside_dist:.4f}, inside_dist={inside_dist:.4f}")

    # 修正：符号を反転
    boundary_dist = -(outside_dist + inside_dist)

    # 2. 内壁（矩形AABB）までの距離
    min_wall_dist = boundary_dist

    if len(static_walls) > 0:
        for wall in static_walls:
            x_min = wall["x_min"]
            y_min = wall["y_min"]
            x_max = wall["x_max"]
            y_max = wall["y_max"]

            # 矩形の外側への最短距離
            dx = max(x_min - p[0], p[0] - x_max, 0.0)
            dy = max(y_min - p[1], p[1] - y_max, 0.0)
            dist_outside = np.sqrt(dx**2 + dy**2)

            # 矩形内にいるかチェック
            if p[0] >= x_min and p[0] <= x_max and p[1] >= y_min and p[1] <= y_max:
                # 矩形内：最も近い辺までの距離（負の値）
                dist_x_min = p[0] - x_min
                dist_x_max = x_max - p[0]
                dist_y_min = p[1] - y_min
                dist_y_max = y_max - p[1]

                # 最も近い辺までの距離（負）
                wall_dist = -min(min(dist_x_min, dist_x_max), min(dist_y_min, dist_y_max))
            else:
                # 矩形外：距離は正
                wall_dist = dist_outside

            # 最も近い障害物までの距離を保持
            if wall_dist < min_wall_dist:
                min_wall_dist = wall_dist

    result = min_wall_dist
    print(f"  → boundary_dist={boundary_dist:.4f}, min_wall_dist={min_wall_dist:.4f}, result={result:.4f}")
    return result


boundary = np.array([0.0, 0.0, 6.0, 6.0])

print("[SDF VALUES - SIGN CHECK]")
print(f"{'Point':<20} {'SDF':<15} {'Status':<30}")
print("-" * 65)

for x, y, label in test_points:
    print(f"\n{label} ({x}, {y}):")
    p = np.array([x, y, 0.0])
    sdf = get_sdf_static(p, boundary, static_walls)

    # 期待される符号
    if x > -6 and x < 6 and y > -6 and y < 6:
        expected = "外壁内（正の値期待）"
    else:
        expected = "外壁外（負の値期待）"

    status = "✓" if (sdf > 0 and expected.startswith("外壁内")) or (sdf < 0 and expected.startswith("外壁外")) else "✗"

    print(f"{status} {expected}")
