#!/usr/bin/env python3
"""
現在の SDF pkl ファイルを読み込んで、距離画像を描画
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# SDF を読み込み
cache_file = Path("sdf_distance_field.pkl")
if not cache_file.exists():
    print("Error: sdf_distance_field.pkl not found")
    exit(1)

with open(cache_file, "rb") as f:
    data_dict = pickle.load(f)
    sdf_field = data_dict.get("sdf_field")
    metadata = data_dict.get("metadata", {})

grid_size = sdf_field.shape[0]
cell_size = metadata.get("cell_size", 12.0 / (grid_size - 1))

print(f"SDF Field Shape: {sdf_field.shape}")
print(f"Cell Size: {cell_size:.6f}")
print(f"Min: {sdf_field.min():.4f}")
print(f"Max: {sdf_field.max():.4f}")
print(f"Mean: {sdf_field.mean():.4f}")
print()

# 統計
print("[STATISTICS]")
negative_pixels = np.sum(sdf_field < -1e-6)
zero_pixels = np.sum(np.abs(sdf_field) <= 1e-6)
positive_pixels = np.sum(sdf_field > 1e-6)
print(f"Negative (内部/障害物): {negative_pixels:,} pixels")
print(f"Zero (境界): {zero_pixels:,} pixels")
print(f"Positive (外部/自由空間): {positive_pixels:,} pixels")
print()

# カラーマッピング用に正規化
# 負: 青系（障害物）
# ゼロ: 白（境界）
# 正: 赤系（自由空間）
vmin = sdf_field.min()
vmax = sdf_field.max()

# 図を作成
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左図：元の SDF 値
im1 = axes[0].imshow(sdf_field, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
axes[0].set_title(f'SDF Distance Field\n(Min={vmin:.4f}, Max={vmax:.4f})')
axes[0].set_xlabel('X (grid)')
axes[0].set_ylabel('Y (grid)')
cbar1 = plt.colorbar(im1, ax=axes[0])
cbar1.set_label('Distance')

# 右図：符号なし距離（絶対値）
abs_sdf = np.abs(sdf_field)
im2 = axes[1].imshow(abs_sdf, cmap='viridis', origin='lower')
axes[1].set_title(f'Absolute Distance Field\n(Max={abs_sdf.max():.4f})')
axes[1].set_xlabel('X (grid)')
axes[1].set_ylabel('Y (grid)')
cbar2 = plt.colorbar(im2, ax=axes[1])
cbar2.set_label('|Distance|')

plt.tight_layout()
plt.savefig('sdf_current_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved: sdf_current_visualization.png")

# 特定の点での値を検査
print("\n[SPECIFIC POINTS]")
test_points_world = [
    (0.0, 0.0, "中央"),
    (5.9, 0.0, "外壁東（内側）"),
    (-5.9, 0.0, "外壁西（内側）"),
    (0.0, 5.9, "外壁北（内側）"),
    (0.0, -5.9, "外壁南（内側）"),
    (6.1, 0.0, "外壁東（外側）"),
    (-6.1, 0.0, "外壁西（外側）"),
]

for wx, wy, label in test_points_world:
    gi = int((wx + 6.0) / cell_size)
    gj = int((wy + 6.0) / cell_size)
    if 0 <= gi < grid_size and 0 <= gj < grid_size:
        val = sdf_field[gj, gi]
        print(f"World ({wx:5.1f}, {wy:5.1f}) → Grid ({gi:3d}, {gj:3d}): {val:8.4f} ({label})")

# 内壁周辺の値を検査
print("\n[INNER WALLS]")
# maze_w0: center (3.0, 1.5), size (1.5, 0.2)
inner_walls = [
    (3.0, 1.5, "maze_w0（内側）"),
    (3.0, 1.7, "maze_w0（外側）"),
    (3.0, 1.3, "maze_w0（外側）"),
]

for wx, wy, label in inner_walls:
    gi = int((wx + 6.0) / cell_size)
    gj = int((wy + 6.0) / cell_size)
    if 0 <= gi < grid_size and 0 <= gj < grid_size:
        val = sdf_field[gj, gi]
        print(f"World ({wx:5.1f}, {wy:5.1f}) → Grid ({gi:3d}, {gj:3d}): {val:8.4f} ({label})")

plt.show()
