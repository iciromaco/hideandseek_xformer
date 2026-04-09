# verify_sightmap_results.py v2.60
# 演習第25回：Sightmapのカラー視覚化検証（フォルダ整理 ＆ 統合遮蔽監査版）
# 
# 整理後の配置: scripts/verify/verify_sightmap_results.py
# 
# 修正履歴:
# v2.59: 境界誤差修正版。
# v2.60: フォルダ整理に伴うパス解決の修正。
#        1. スクリプトの位置（scripts/verify/）から見て、プロジェクトルートにある .pkl を自動探索。
#        2. 出力先をプロジェクトルートの `data/audit/` に固定し、管理を容易化。

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pathlib import Path

def intersect_segment_with_box(p1, p2, box_pos, box_size):
    """線分 (p1, p2) が矩形 (box_pos, box_size) と交差するかを判定 (Liang-Barsky)"""
    eps = 1e-4
    min_x, max_x = box_pos[0] - box_size[0] + eps, box_pos[0] + box_size[0] - eps
    min_y, max_y = box_pos[1] - box_size[1] + eps, box_pos[1] + box_size[1] - eps
    
    seg_min_x, seg_max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
    seg_min_y, seg_max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
    
    if seg_max_x < min_x or seg_min_x > max_x or seg_max_y < min_y or seg_min_y > max_y:
        return False

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    t_min, t_max = 0.0, 1.0
    
    for edge in range(4):
        if edge == 0: p, q = -dx, p1[0] - min_x
        elif edge == 1: p, q = dx, max_x - p1[0]
        elif edge == 2: p, q = -dy, p1[1] - min_y
        else: p, q = dy, max_y - p1[1]
            
        if p == 0:
            if q < 0: return False
        else:
            t = q / p
            if p < 0:
                if t > t_max: return False
                if t > t_min: t_min = t
            else:
                if t < t_min: return False
                if t < t_max: t_max = t
                
    return t_max >= t_min

def plot_audit_view(px, py, data, output_path: Path):
    """視界マトリックスに動的オブジェクトの影を計算・合成して描画"""
    n = data.get("n_side", 125)
    matrix = data["vis_matrix"]
    bounds = data["bounds"]
    cell_size = data["cell_size"]
    grid_coords = data["grid_coords"]
    dyn_objs = data.get("dynamic_objects", [])
    
    # 座標からインデックスへの変換
    ix = int((px + bounds) / cell_size)
    iy = int((bounds - py) / cell_size)
    ix, iy = max(0, min(ix, n - 1)), max(0, min(iy, n - 1))
    origin_idx = iy * n + ix
    
    vis_row = matrix[origin_idx].reshape((n, n)).astype(float)
    grid_2d = grid_coords.reshape((n, n, 2))
    view_point = np.array([px, py])
    
    # 動的な影の合成
    for y_idx in range(n):
        for x_idx in range(n):
            if vis_row[y_idx, x_idx] < 0.5: continue
            target_pos = grid_2d[y_idx, x_idx]
            for obj in dyn_objs:
                if intersect_segment_with_box(view_point, target_pos, obj["pos"], obj["size"]):
                    vis_row[y_idx, x_idx] = 0.4 # 動的な影の色
                    break
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(vis_row, extent=[-bounds, bounds, -bounds, bounds], 
                   cmap='viridis', origin='upper', alpha=0.8, vmin=0, vmax=1)
    
    # 静的な壁の描画
    walls = data.get("walls", [])
    for i, w in enumerate(walls):
        # wall format: [cx, cy, sx, sy]
        ax.add_patch(patches.Rectangle((w[0]-w[2], w[1]-w[3]), w[2]*2, w[3]*2, color='red', alpha=0.5))
        
    # 動的オブジェクトの描画
    for obj in dyn_objs:
        pos, size = obj["pos"], obj["size"]
        rect = patches.Rectangle((pos[0]-size[0], pos[1]-size[1]), size[0]*2, size[1]*2, 
                                 linewidth=2, edgecolor='darkorange', facecolor='orange', alpha=0.8)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], obj['name'].split('_')[0], color='black', ha='center', va='center', fontweight='bold')

    ax.plot(px, py, 'ro', markersize=12, markeredgecolor='white', label="Agent")
    ax.set_title(f"Sightmap Audit: {data['layout_name']}\n(Yellow=Visible, Blue=Dynamic Shadow, Purple=Wall)")
    ax.set_aspect('equal')
    plt.savefig(output_path)
    plt.close()

def main():
    # パス解決: スクリプトの2つ上の階層（プロジェクトルート）を探索
    root = Path(__file__).resolve().parent.parent.parent
    cache_files = list(root.glob("hns_sightmap_v25_cache_*.pkl"))
    
    if not cache_files:
        print(f"❌ Error: No cache files found in {root}")
        print("   Please run 'python generate_sightmap.py' first.")
        return

    output_dir = root / "data" / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_points = [(0.0, 0.0), (3.0, 3.0), (-3.0, -1.0)]
    
    for cf in cache_files:
        print(f"📂 Auditing cache: {cf.name}")
        with open(cf, "rb") as f:
            data = pickle.load(f)
        
        for px, py in test_points:
            out_name = f"audit_{data['layout_name']}_x{px}_y{py}.png"
            out_path = output_dir / out_name
            plot_audit_view(px, py, data, out_path)
            print(f"   - Generated: {out_path.relative_to(root)}")

if __name__ == "__main__":
    main()