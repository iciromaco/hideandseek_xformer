
import numpy as np
import pickle
import sys
import os

if __name__ == "__main__":
    # コマンドライン引数でpklファイル指定。なければsrc/envs内の最新pklを自動選択
    if len(sys.argv) < 2:
        envs_dir = os.path.join(os.path.dirname(__file__), "../src/envs")
        import glob
        pkls = sorted(glob.glob(os.path.join(envs_dir, "sdfgrid_mode4_*.pkl")), key=os.path.getmtime, reverse=True)
        if not pkls:
            print("No SDF pkl found in src/envs. Please generate one or specify a path.")
            sys.exit(1)
        pkl_path = pkls[0]
        print(f"Auto-selected latest SDF: {pkl_path}")
    else:
        pkl_path = sys.argv[1]
        if not os.path.exists(pkl_path):
            candidate = os.path.join(os.path.dirname(__file__), "../src/envs", os.path.basename(pkl_path))
            if os.path.exists(candidate):
                pkl_path = candidate
            else:
                print(f"File not found: {pkl_path}")
                sys.exit(1)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    sdf_grid = data["grid"]
    min_x = data["min_x"]
    min_y = data["min_y"]
    cell_size = data["cell_size"]
    print("SDF grid shape:", sdf_grid.shape)
    print("SDF grid min:", np.min(sdf_grid))
    print("SDF grid max:", np.max(sdf_grid))
    print("SDF grid dtype:", sdf_grid.dtype)
    mask = sdf_grid < -0.1
    if np.any(mask):
        idxs = np.argwhere(mask)
        print(f"Found {len(idxs)} grid points < -0.1:")
        for y, x in idxs:
            print(f"  grid[{y},{x}] = {sdf_grid[y, x]}")
    else:
        print("No grid points < -0.1 found.")

    # --- SDFグリッドの異常値のみ出力 ---


    # AABB内判定・walls_xpos関連は除去

    # --- 可視化機能追加 ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    im = plt.imshow(sdf_grid, origin='lower', cmap='viridis')
    plt.colorbar(im, label='SDF value')
    plt.title('SDF Grid Visualization')
    # -0.1未満の点を赤丸で重ねて表示
    if np.any(mask):
        idxs = np.argwhere(mask)
        ys, xs = idxs[:, 0], idxs[:, 1]
        plt.scatter(xs, ys, color='red', s=10, label='SDF < -0.1')
        plt.legend()
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.tight_layout()
    plt.show()
