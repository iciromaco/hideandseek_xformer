# verify_sightmap_results.py
# 演習第25回：Sightmapのカラー視覚化検証（動的オブジェクトの重畳表示版）
#
# 壁による静的な影の上に、ランダム配置された箱やランプを重ねて描画し、
# 物理環境の整合性を視覚的に監査します。

import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def plot_visibility_with_objects(px, py, data, output_path: Path):
    """特定の視点からの視界プロットに、動的オブジェクトを重ねて表示"""
    n = data.get("n_side", data.get("n_cells"))
    matrix = data["vis_matrix"]
    bounds = data["bounds"]
    cell_size = data["cell_size"]
    layout_name = data.get("layout_name", "Unknown")

    # 視点の座標をインデックスに変換
    ix = int((px + bounds) / cell_size)
    iy = int((bounds - py) / cell_size)
    ix = max(0, min(ix, n - 1))
    iy = max(0, min(iy, n - 1))
    origin_idx = iy * n + ix

    # 視界データの抽出
    vis_row = matrix[origin_idx].reshape((n, n)).astype(float)

    plt.figure(figsize=(10, 10))
    # 1. 壁による影（Sightmap）を描画
    im = plt.imshow(
        vis_row,
        extent=[-bounds, bounds, -bounds, bounds],
        cmap="viridis",
        origin="upper",
        alpha=0.6,
    )

    # 2. 静的な壁をプロット
    walls = data.get("walls", [])
    for i, wall in enumerate(walls):
        p1, p2 = wall
        plt.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "r-",
            linewidth=3,
            label="Static Walls" if i == 0 else "",
        )

    # 3. 動的オブジェクト（箱・ランプ）をプロット
    dyn_objs = data.get("dynamic_objects", [])
    for obj in dyn_objs:
        pos = obj["pos"]
        size = obj["size"]
        # 矩形として描画
        rect = plt.Rectangle(
            (pos[0] - size[0], pos[1] - size[1]),
            size[0] * 2,
            size[1] * 2,
            angle=0,
            color="orange",
            alpha=0.8,
            label=f"Dynamic: {obj['name']}",
        )
        plt.gca().add_patch(rect)
        plt.text(
            pos[0],
            pos[1],
            obj["name"].split("_")[0],
            color="white",
            ha="center",
            va="center",
            fontsize=8,
        )

    # 4. エージェント（視点）
    plt.plot(px, py, "ro", markersize=15, markeredgecolor="white", label="Agent Viewpoint")

    plt.title(f"Hybrid Visibility Audit: {layout_name}\n(Static Shadow + Dynamic Objects)")
    plt.colorbar(im, label="Static Wall Visibility (1.0=Clear, 0.0=Shadow)")
    plt.legend(loc="upper right", fontsize="small")
    plt.axis("equal")

    plt.savefig(output_path)
    plt.close()


def run_audit():
    cache_files = list(Path(".").glob("hns_sightmap_v25_cache_*.pkl"))
    output_dir = Path("sightmap_debug")
    output_dir.mkdir(exist_ok=True)

    # 中央付近からの視界をメインに確認
    test_points = [(0.0, 0.0)]

    for cf in cache_files:
        print(f"📂 監査実行中: {cf.name}")
        with open(cf, "rb") as f:
            data = pickle.load(f)

        layout_name = data.get("layout_name", "Unknown")
        layout_dir = output_dir / layout_name
        layout_dir.mkdir(exist_ok=True)

        for px, py in test_points:
            out_file = layout_dir / f"audit_view_x{px}_y{py}.png"
            plot_visibility_with_objects(px, py, data, out_file)
            print(f"   - 監査画像を出力しました: {out_file}")


if __name__ == "__main__":
    run_audit()
