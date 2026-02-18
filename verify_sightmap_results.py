# verify_sightmap_results.py
# 演習第25回：Sightmapのカラー視覚化検証（ハイブリッド遮蔽監査版）
#
# 保存されたキャッシュから「壁の影」を取得し、さらにその時の「箱の配置」に
# 基づいて動的に影を計算・合成することで、視界判定の正当性を厳密に検証します。

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def intersect_segment_with_box(p1, p2, box_pos, box_size):
    """
    線分 (p1, p2) が矩形 (box_pos, box_size) と交差するかを判定します。
    """
    # 矩形の境界
    min_x, max_x = box_pos[0] - box_size[0], box_pos[0] + box_size[0]
    min_y, max_y = box_pos[1] - box_size[1], box_pos[1] + box_size[1]
    
    # 線分の AABB
    seg_min_x, seg_max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
    seg_min_y, seg_max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
    
    # AABB同士が重なっていないなら交差しない
    if seg_max_x < min_x or seg_min_x > max_x or seg_max_y < min_y or seg_min_y > max_y:
        return False

    # 線分パラメータ t (Liang-Barsky風の判定)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    t_min = 0.0
    t_max = 1.0
    
    # 各境界線に対して判定
    for edge in range(4):
        if edge == 0:   # Left
            p, q = -dx, p1[0] - min_x
        elif edge == 1: # Right
            p, q = dx, max_x - p1[0]
        elif edge == 2: # Bottom
            p, q = -dy, p1[1] - min_y
        else:           # Top
            p, q = dy, max_y - p1[1]
            
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
    n = data.get("n_side", data.get("n_cells"))
    matrix = data["vis_matrix"]
    bounds = data["bounds"]
    cell_size = data["cell_size"]
    grid_coords = data["grid_coords"]
    dyn_objs = data.get("dynamic_objects", [])
    
    # 視点(エージェント)のインデックス取得
    ix = int((px + bounds) / cell_size)
    iy = int((bounds - py) / cell_size)
    ix = max(0, min(ix, n - 1)); iy = max(0, min(iy, n - 1))
    origin_idx = iy * n + ix
    
    # 1. まず静的な壁による視界を取得 (2次元に整形して扱いやすくする)
    vis_row = matrix[origin_idx].reshape((n, n)).astype(float)
    
    # 座標データも (n, n, 2) に整形して main23 風の直感的なアクセスを可能にする
    grid_2d = grid_coords.reshape((n, n, 2))
    
    # 2. 動적オブジェクトによる影を計算 (ハイブリッド・レイキャスト)
    print(f"   - {data['layout_name']} ({px}, {py}) の動的な影を計算中...")
    
    view_point = np.array([px, py])
    
    for y_idx in range(n):
        for x_idx in range(n):
            # すでに静的な壁で遮蔽されている場所(0.0)はスキップ
            if vis_row[y_idx, x_idx] < 0.5:
                continue
                
            # 2次元グリッドから物理座標を直接取得
            target_pos = grid_2d[y_idx, x_idx]
            
            # 各動的オブジェクトとの交差をチェック
            for obj in dyn_objs:
                if intersect_segment_with_box(view_point, target_pos, obj["pos"], obj["size"]):
                    # 動的な影として 0.4 を設定 (静的な影 0.0 と区別)
                    vis_row[y_idx, x_idx] = 0.4
                    break
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 3. 合成された視界を描画
    im = ax.imshow(vis_row, extent=[-bounds, bounds, -bounds, bounds], 
                   cmap='viridis', origin='upper', alpha=0.8, vmin=0, vmax=1)
    
    # 4. 静的な壁 (赤線)
    walls = data.get("walls", [])
    for i, wall in enumerate(walls):
        p1, p2 = wall
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=3, label="Static Wall" if i==0 else "")
        
    # 5. 動的オブジェクト (オレンジ矩形)
    for obj in dyn_objs:
        pos, size = obj["pos"], obj["size"]
        rect = patches.Rectangle((pos[0]-size[0], pos[1]-size[1]), size[0]*2, size[1]*2, 
                                 linewidth=2, edgecolor='darkorange', facecolor='orange', alpha=0.9)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], obj['name'].split('_')[0], color='black', 
                ha='center', va='center', fontweight='bold', fontsize=9)

    # 6. エージェント
    ax.plot(px, py, 'ro', markersize=12, markeredgecolor='white', label="Agent Viewpoint")
    
    ax.set_title(f"Visibility Audit: {data['layout_name']}\n(Yellow=Visible, Blue=Dynamic Shadow, Purple=Static Shadow)")
    ax.set_xlim(-bounds, bounds); ax.set_ylim(-bounds, bounds)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.3)
    
    plt.colorbar(im, ax=ax, label="Visibility Value")
    plt.savefig(output_path)
    plt.close()

def main():
    cache_files = list(Path(".").glob("hns_sightmap_v25_cache_*.pkl"))
    output_dir = Path("sightmap_debug")
    output_dir.mkdir(exist_ok=True)
    
    test_points = [(0.0, 0.0), (3.0, 3.0)]
    
    for cf in cache_files:
        print(f"📂 監査実行中: {cf.name}")
        with open(cf, "rb") as f:
            data = pickle.load(f)
        
        l_name = data.get("layout_name", "Unknown")
        l_dir = output_dir / l_name
        l_dir.mkdir(exist_ok=True)
        
        for px, py in test_points:
            out_file = l_dir / f"audit_v25_x{px}_y{py}.png"
            plot_audit_view(px, py, data, out_file)
            print(f"   - 完了: {out_file}")

if __name__ == "__main__":
    main()