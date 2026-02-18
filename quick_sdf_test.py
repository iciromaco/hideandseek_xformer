#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")
import numpy as np
from hideandseek import XML_CONTENT
from main23_sightmap_optimized import extract_maze_walls_from_xml, walls_to_segments

maze_walls = extract_maze_walls_from_xml(XML_CONTENT, from_string=True)
wall_segments = walls_to_segments(maze_walls)

print(f"Found {len(wall_segments)} wall segments:")
for i, seg in enumerate(wall_segments[:4]):
    print(f"  Segment {i}: {seg}")

# テスト点
test_points = [
    (0.0, 0.0, "Room center"),
    (5.9, 0.0, "Near external wall (inside)"),
    (6.1, 0.0, "Outside external wall"),
    (3.0, 1.5, "Internal wall maze_w0 (inside)"),
    (3.0, 1.7, "Internal wall maze_w0 (outside)")
]

def point_to_segment_distance(p, seg):
    p = np.array(p[:2])
    start = np.array(seg[0])
    end = np.array(seg[1])
    seg_vec = end - start
    seg_len_sq = np.dot(seg_vec, seg_vec)
    if seg_len_sq < 1e-8:
        return np.linalg.norm(p - start)
    t = np.clip(np.dot(p - start, seg_vec) / seg_len_sq, 0.0, 1.0)
    closest = start + t * seg_vec
    return np.linalg.norm(p - closest)

print("\nDistance to nearest wall segment:")
for pt in test_points:
    x, y, label = pt
    min_dist = min(point_to_segment_distance((x, y, 0), seg) for seg in wall_segments)
    print(f"  ({x:5.1f}, {y:5.1f}) {label:30s}: {min_dist:6.3f}m")
