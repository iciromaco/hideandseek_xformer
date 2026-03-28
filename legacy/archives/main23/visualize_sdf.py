#!/usr/bin/env python3
"""
SDF フィールドを可視化
距離を色に置き換えて表示
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# SDF を読み込み
cache_file = Path("sdf_distance_field.pkl")
if not cache_file.exists():
    print("Error: sdf_distance_field.pkl not found")
    sys.exit(1)

with open(cache_file, "rb") as f:
    data_dict = pickle.load(f)
    sdf_field = data_dict.get("sdf_field")

if sdf_field is None:
    print("Error: SDF field not found in pickle file")
    sys.exit(1)

print(f"SDF field shape: {sdf_field.shape}")
print(f"SDF field dtype: {sdf_field.dtype}")
print(f"SDF field min: {sdf_field.min():.4f}")
print(f"SDF field max: {sdf_field.max():.4f}")
print(f"SDF field mean: {sdf_field.mean():.4f}")

# グリッドサイズと座標の計算
grid_size_0, grid_size_1 = sdf_field.shape
cell_size = 12.0 / (grid_size_0 - 1)  # -6 to 6

# 座標軸を作成
x_coords = np.linspace(-6, 6, grid_size_1)
y_coords = np.linspace(-6, 6, grid_size_0)
X, Y = np.meshgrid(x_coords, y_coords)

# 可視化
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# プロット1: SDF 距離場（カラーマップ）
ax1 = axes[0]
# 距離値をクリップして可視化（-1 to 1 の範囲に）
sdf_clipped = np.clip(sdf_field, -1, 1)
im1 = ax1.contourf(X, Y, sdf_clipped, levels=20, cmap="RdBu_r")
ax1.set_title("SDF Distance Field (clipped -1 to 1)", fontsize=12, fontweight="bold")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_aspect("equal")
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label("Distance (m)")

# 等高線も追加
contour = ax1.contour(
    X,
    Y,
    sdf_field,
    levels=[-0.5, -0.25, 0, 0.25, 0.5, 1.0],
    colors="black",
    alpha=0.3,
    linewidths=0.5,
)
ax1.clabel(contour, inline=True, fontsize=8)

# プロット2: 絶対距離
ax2 = axes[1]
sdf_abs = np.abs(sdf_field)
sdf_abs_clipped = np.clip(sdf_abs, 0, 2)
im2 = ax2.contourf(X, Y, sdf_abs_clipped, levels=20, cmap="viridis")
ax2.set_title("Absolute Distance Field (clipped 0 to 2)", fontsize=12, fontweight="bold")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_aspect("equal")
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label("|Distance| (m)")

plt.tight_layout()
plt.savefig("sdf_visualization.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to 'sdf_visualization.png'")
plt.show()

# 統計情報
print("\n[SDF STATISTICS]")
print(f"Negative values (inside): {(sdf_field < 0).sum()} pixels")
print(f"Zero/boundary values: {(sdf_field == 0).sum()} pixels")
print(f"Positive values (outside): {(sdf_field > 0).sum()} pixels")

if (sdf_field < 0).sum() > 0:
    print(f"Negative range: {sdf_field[sdf_field < 0].min():.4f} to {sdf_field[sdf_field < 0].max():.4f}")
if (sdf_field > 0).sum() > 0:
    print(f"Positive range: {sdf_field[sdf_field > 0].min():.4f} to {sdf_field[sdf_field > 0].max():.4f}")


# 外壁付近の値をチェック
print("\n[WALL VICINITY CHECK]")
# 北壁 (y = 6)
idx_north = grid_size_0 - 1
print("North wall (y=6) SDF values:")
print(f"  At center (x=0): {sdf_field[idx_north, grid_size_1//2]:.4f}")
print(f"  At x=-5: {sdf_field[idx_north, int(((-5 + 6) / 12) * grid_size_1)]:.4f}")

# 南壁 (y = -6)
idx_south = 0
print("South wall (y=-6) SDF values:")
print(f"  At center (x=0): {sdf_field[idx_south, grid_size_1//2]:.4f}")

# 東壁 (x = 6)
idx_east = grid_size_1 - 1
print("East wall (x=6) SDF values:")
print(f"  At center (y=0): {sdf_field[grid_size_0//2, idx_east]:.4f}")

# 西壁 (x = -6)
idx_west = 0
print("West wall (x=-6) SDF values:")
print(f"  At center (y=0): {sdf_field[grid_size_0//2, idx_west]:.4f}")

# 中央付近
print(f"\nCenter (0, 0): {sdf_field[grid_size_0//2, grid_size_1//2]:.4f}")

# Viewpoint 付近
viewpoint = np.array([-2.0, 0.0])
grid_x = int((viewpoint[0] + 6.0) / cell_size + 0.5)
grid_y = int((viewpoint[1] + 6.0) / cell_size + 0.5)
if 0 <= grid_x < grid_size_1 and 0 <= grid_y < grid_size_0:
    print(f"Viewpoint (-2, 0): {sdf_field[grid_y, grid_x]:.4f}")
