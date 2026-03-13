import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# --- 使い方 ---
# python scripts/visualize_sdf_from_checkpoint.py checkpoints/sdfgrid_mode4_xxxxx.pkl

def main(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    grid = data["grid"]
    min_x = data["min_x"]
    min_y = data["min_y"]
    cell_size = data["cell_size"]
    print(f"Loaded SDF grid: shape={grid.shape}, min=({min_x},{min_y}), cell_size={cell_size}")
    print(f"SDF min: {np.min(grid)}, max: {np.max(grid)}")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, origin='lower', cmap='viridis',
                   extent=[min_x, min_x + grid.shape[1]*cell_size,
                           min_y, min_y + grid.shape[0]*cell_size])
    plt.colorbar(im, label='SDF value')
    plt.title(f'SDF Grid Visualization\n{os.path.basename(pkl_path)}')
    plt.xlabel('World X')
    plt.ylabel('World Y')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/visualize_sdf_from_checkpoint.py <sdfgrid_mode4_xxx.pkl>")
        sys.exit(1)
    main(sys.argv[1])
