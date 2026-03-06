#!/usr/bin/env python3
"""
交線法のテスト
修正版 ray_segment_intersection_numba の検証
"""

import sys
from main23_sightmap_optimized import ray_segment_intersection_numba

def test_case(name, start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2, expected_t):
    """
    テストケースを実行
    """
    result = ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y, seg_x1, seg_y1, seg_x2, seg_y2)
    
    # 期待値との比較
    if expected_t is None:
        # 交点がないことを期待
        is_pass = result >= 1e6 * 0.99  # 1e6に近い値
        status = "✓ PASS" if is_pass else "✗ FAIL"
        print(f"{status} | {name}")
        print(f"  Result: {result:.4f} (expected: no intersection)")
    else:
        # 交点があることを期待
        error = abs(result - expected_t)
        is_pass = error < 0.01  # 許容誤差 0.01
        status = "✓ PASS" if is_pass else "✗ FAIL"
        print(f"{status} | {name}")
        print(f"  Result: {result:.4f}, Expected: {expected_t:.4f}, Error: {error:.4f}")
    
    return is_pass

print("=" * 70)
print("交線法テスト（ray_segment_intersection_numba）")
print("=" * 70)

tests_passed = 0
tests_total = 0

# ========== テスト1：水平線に対する対角線レイ ==========
print("\n[Test 1] 水平線（y=2）に対する対角線レイ")
print("  レイ: (0, 0) → (1, 1) 方向")
print("  線分: (0, 2) - (4, 2)")
# レイ: (0,0) + t*(1,1)
# 水平線: y=2 → t=2
# その時 x=2, 線分範囲内 (0<=x<=4)
tests_total += 1
if test_case("水平線との交点", 0, 0, 1, 1, 0, 2, 4, 2, 2.0):
    tests_passed += 1

# ========== テスト2：垂直線に対する対角線レイ ==========
print("\n[Test 2] 垂直線（x=3）に対する対角線レイ")
print("  レイ: (0, 0) → (1, 1) 方向")
print("  線分: (3, 0) - (3, 4)")
# レイ: (0,0) + t*(1,1)
# 垂直線: x=3 → t=3
# その時 y=3, 線分範囲内 (0<=y<=4)
tests_total += 1
if test_case("垂直線との交点", 0, 0, 1, 1, 3, 0, 3, 4, 3.0):
    tests_passed += 1

# ========== テスト3：水平線だが端点外 ==========
print("\n[Test 3] 水平線（y=2）だが交点が線分範囲外")
print("  レイ: (0, 0) → (1, 1) 方向")
print("  線分: (0, 2) - (1, 2)（x範囲 0-1, 交点は x=2）")
# レイ: (0,0) + t*(1,1) → t=2 で (2,2)
# 線分: x in [0, 1], y = 2
# 交点 (2,2) は線分範囲外
tests_total += 1
if test_case("範囲外", 0, 0, 1, 1, 0, 2, 1, 2, None):
    tests_passed += 1

# ========== テスト4：垂直線だが端点外 ==========
print("\n[Test 4] 垂直線（x=3）だが交点が線分範囲外")
print("  レイ: (0, 0) → (1, 1) 方向")
print("  線分: (3, 0) - (3, 1)（y範囲 0-1, 交点は y=3）")
# レイ: (0,0) + t*(1,1) → t=3 で (3,3)
# 線分: x = 3, y in [0, 1]
# 交点 (3,3) は線分範囲外
tests_total += 1
if test_case("範囲外（垂直線）", 0, 0, 1, 1, 3, 0, 3, 1, None):
    tests_passed += 1

# ========== テスト5：斜めの線との交点 ==========
print("\n[Test 5] 斜めの線との交点")
print("  レイ: (0, 0) → (1, 0) 方向（右方向）")
print("  線分: (2, -1) - (2, 1)（x=2の垂直線）")
# レイ: (0,0) + t*(1,0) → y は常に 0
# 垂直線: x=2, y in [-1, 1]
# 交点: (2, 0), t=2
tests_total += 1
if test_case("垂直線（右方向レイ）", 0, 0, 1, 0, 2, -1, 2, 1, 2.0):
    tests_passed += 1

# ========== テスト6：平行線（交点なし） ==========
print("\n[Test 6] 平行線（交点なし）")
print("  レイ: (0, 0) → (1, 0) 方向")
print("  線分: (0, 1) - (1, 1)（y=1, 平行）")
tests_total += 1
if test_case("平行線", 0, 0, 1, 0, 0, 1, 1, 1, None):
    tests_passed += 1

# ========== テスト7：後ろ側の交点（無視） ==========
print("\n[Test 7] レイの後ろ側の交点（無視される）")
print("  レイ: (5, 0) → (1, 0) 方向（右方向）")
print("  線分: (2, -1) - (2, 1)（x=2）")
print("  注：レイの開始点は (5,0)、線分は x=2 より左（後ろ側）")
tests_total += 1
if test_case("後ろ側", 5, 0, 1, 0, 2, -1, 2, 1, None):
    tests_passed += 1

# ========== テスト8：開始点に近い交点（1e-6より小さい） ==========
print("\n[Test 8] 開始点に極めて近い交点")
print("  レイ: (0, 0) → (1, 1) 方向")
print("  線分: (0, 1e-7) - (1, 1e-7)（ほぼy=0）")
tests_total += 1
if test_case("開始点極近", 0, 0, 1, 1, 0, 1e-7, 1, 1e-7, None):
    tests_passed += 1

# ========== テスト9：端点ちょうどの交点 ==========
print("\n[Test 9] 線分の端点ちょうどを通過")
print("  レイ: (0, 0) → (1, 1) 方向")
print("  線分: (2, 2) - (4, 2)（端点が (2,2)）")
tests_total += 1
if test_case("端点通過", 0, 0, 1, 1, 2, 2, 4, 2, 2.0):
    tests_passed += 1

# ========== テスト10：複雑な斜め線 ==========
print("\n[Test 10] 複雑な斜め線との交点")
print("  レイ: (0, 0) → (2, 1) 方向")
print("  線分: (0, 2) - (4, 0)（斜めの線）")
# レイ: (0,0) + t*(2,1) = (2t, t)
# 線分: (0,2) + u*(4,-2) = (4u, 2-2u), u in [0,1]
# 2t = 4u → t = 2u
# t = 2 - 2u → 2u = 2 - 2u → 4u = 2 → u = 0.5, t = 1
tests_total += 1
if test_case("斜め線との交点", 0, 0, 2, 1, 0, 2, 4, 0, 1.0):
    tests_passed += 1

# ========== 結果サマリー ==========
print("\n" + "=" * 70)
print(f"テスト結果: {tests_passed}/{tests_total} PASS")
print("=" * 70)

if tests_passed == tests_total:
    print("✓ 全テスト成功！")
    sys.exit(0)
else:
    print(f"✗ {tests_total - tests_passed} テスト失敗")
    sys.exit(1)
