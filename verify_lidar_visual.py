# verify_lidar_visual.py
# 演習第25回：Lidar 3モード検証・SDFマップ視覚化デバッグ機能搭載版

import matplotlib
# 非対話型バックエンド設定
matplotlib.use('Agg')

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
from hns_environment import TeamCosEnv

def run_benchmarked_audit():
    """エンジンの監査と、SDFマップを含む視覚的整合性の確認"""
    print("🚀 verify_lidar_visual: Running high-performance audit with SDF Visualization...")
    env = TeamCosEnv()
    engine = env.vis_engine
    angles_deg = engine.angles_deg

    # 検証地点: 以前のご指摘に基づき、何もない空間(-3,3)と中心付近(0,1)
    test_points = [
        {"pos": np.array([-3.0, 3.0]), "heading": 0.0, "label": "First View"},
        {"pos": np.array([0.0, 1.0]), "heading": np.deg2rad(45), "label": "Last View"}
    ]
    
    modes = [0, 1, 2] 
    mode_names = ["Native", "Geometric", "Sphere"]
    
    fig, axes = plt.subplots(len(test_points), 3, figsize=(20, 12))
    print("\nMode            |   Avg (μs) |        SPS")
    print("-" * 40)

    # 1. 統計計測
    for mode in modes:
        engine.cast_lidar(np.array([0., 0.]), mode=mode) # Warmup
        n_iters = 3000 
        t_start = time.perf_counter()
        for _ in range(n_iters):
            engine.cast_lidar(test_points[0]["pos"], heading=0.0, mode=mode)
        t_total = time.perf_counter() - t_start
        avg_us = (t_total / n_iters) * 1e6
        sps = int(1.0 / (t_total / n_iters))
        print(f"{mode_names[mode]:15} | {avg_us:10.2f} | {sps:12,}")

    # 2. 精度検証およびSDFの描画
    for row, pt in enumerate(test_points):
        pos, heading = pt["pos"], pt["heading"]
        for mode in modes:
            lidar_res = engine.cast_lidar(pos, heading=heading, mode=mode)
            ax = axes[row, mode]
            ax.set_title(f"{pt['label']}: {mode_names[mode]}")
            
            # --- 背景デバッグ描画 (Mode 2 のみ SDF マップを表示) ---
            if mode == 2 and "sdf_map" in engine.cache:
                c = engine.cache
                sdf = c["sdf_map"]
                b = c["bounds"]
                # SDFをヒートマップとして描画 (青いほど障害物から遠く、白いほど近い)
                im = ax.imshow(sdf, extent=[-b, b, -b, b], cmap='GnBu_r', origin='upper', alpha=0.4)
                if row == 0:
                    plt.colorbar(im, ax=ax, shrink=0.6, label="SDF Distance (m)")

            # 静的な壁 (境界線の描画)
            for w in engine.walls_np:
                rect = plt.Rectangle((w[0]-w[2], w[1]-w[3]), w[2]*2, w[3]*2, color='gray', alpha=0.15, ec='black', linestyle='--')
                ax.add_patch(rect)
                
            # 動的オブジェクト (Box/Ramp)
            for gid in engine.idx_box_geom:
                p = env.data.geom_xpos[gid]; s = env.model.geom_size[gid]
                rect = plt.Rectangle((p[0]-s[0], p[1]-s[1]), s[0]*2, s[1]*2, color='orange', alpha=0.4, ec='black')
                ax.add_patch(rect)
                
            # エージェント描画
            for bid in engine.idx_agent_body:
                p = env.data.xpos[bid]
                name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                color = 'red' if 'seeker' in name or 's' == name else 'lime'
                circle = plt.Circle((p[0], p[1]), 0.45, color=color, alpha=0.8, ec='black', zorder=5)
                ax.add_patch(circle)
                ax.text(p[0], p[1], name[:2], ha='center', va='center', fontsize=8, fontweight='bold', zorder=6)

            # Lidarレイ描画
            for i, d in enumerate(lidar_res):
                angle_rad = np.deg2rad(angles_deg[i]) + heading
                vx, vy = np.cos(angle_rad), np.sin(angle_rad)
                color = 'black' if mode==0 else ('green' if mode==1 else 'blue')
                ax.plot([pos[0], pos[0]+vx*d], [pos[1], pos[1]+vy*d], color=color, alpha=0.4, linewidth=1)
                ax.scatter(pos[0]+vx*d, pos[1]+vy*d, color=color, s=15, zorder=7)
                if d < 19.0:
                    ax.text(pos[0]+vx*d, pos[1]+vy*d, f"{i}", fontsize=8, color=color)

            # 視点マーカー
            ax.scatter(pos[0], pos[1], color='magenta', s=120, marker='*', zorder=10)
            ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5)
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig("lidar_benchmark_final_debug.png", dpi=150)
    print(f"\n✅ Benchmark report saved: lidar_benchmark_final_debug.png")

if __name__ == "__main__":
    run_benchmarked_audit()