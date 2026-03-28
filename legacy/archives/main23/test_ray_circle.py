#!/usr/bin/env python3
"""
ray_circle_intersection_numba のテスト
"""

import os
import sys

sys.path.insert(0, os.getcwd())

from main23_sightmap_optimized import ray_circle_intersection_numba

# テスト1: 明らかにヒットするケース
# start=(0,0), dir=(1,0) →x方向に進むレイ
# circle=(5,0), radius=1
# 期待値: 4.0 (circle表面までの距離)
dist = ray_circle_intersection_numba(0, 0, 1, 0, 5, 0, 1)
print(f"Test 1 (expected hit at 4.0): {dist}")

# テスト2: ヒットしないケース
# start=(0,0), dir=(1,0) →x方向
# circle=(0, 5), radius=1
# 期待値: 1e6 (no hit)
dist = ray_circle_intersection_numba(0, 0, 1, 0, 0, 5, 1)
print(f"Test 2 (expected no hit 1e6): {dist}")

# テスト3: visibility_engine のテストケース
# viewpoint=(-2, 0), direction=(0.7071, 0.7071), hider1=(1, 1), radius=0.4
dist = ray_circle_intersection_numba(-2, 0, 0.7071, 0.7071, 1, 1, 0.4)
print(f"Test 3 (hider1 from viewpoint along beam 1): {dist}")

# テスト4: 開始点が円内
# start=(4.9, 0), dir=(1, 0), circle=(5, 0), radius=1
# → 円内から出ていく場合
dist = ray_circle_intersection_numba(4.9, 0, 1, 0, 5, 0, 1)
print(f"Test 4 (start inside circle): {dist}")
