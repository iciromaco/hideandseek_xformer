# verify_lidar_visual.py
# 演習第25回：Lidar 3モード検証・全グループジオメトリ可視化・全エージェント描画版

import matplotlib
# 非対話型環境での警告回避
matplotlib.use('Agg')

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
from hns_environment import TeamCosEnv

def run_benchmarked_audit():
    """エンジンの監査と、全グループのジオメトリを含めた視覚的整合性の確認"""
    print("🚀 verify_lidar_visual: Running high-performance audit with ALL geom groups Visualization...")
    env = TeamCosEnv()
    engine = env.vis_engine
    angles_deg = engine.angles_deg

    # 検証地点: 上段はエージェント密集地、下段は中央付近
    test_points = [
        {"pos": np.array([-3.0, 3.0]), "heading": 0.0, "label": "First View"},
        {"pos": np.array([0.0, 1.0]), "heading": np.deg2rad(45), "label": "Last View"}
    ]
    
    modes = [0, 1, 2] 
    mode_names = ["Native", "Geometric", "Sphere"]
    
    fig, axes = plt.subplots(len(test_points), 3, figsize=(24, 14))
    print("\nMode            |   Avg (μs) |        SPS")
    print("-" * 40)

    # 1. 統計計測
    for mode in modes:
        engine.cast_lidar(np.array([0., 0.]), mode=mode) # ウォームアップ
        n_iters = 1000 
        t_start = time.perf_counter()
        for _ in range(n_iters):
            engine.cast_lidar(test_points[0]["pos"], heading=0.0, mode=mode)
        t_total = time.perf_counter() - t_start
        
        avg_us = (t_total / n_iters) * 1e6
        sps = int(1.0 / (t_total / n_iters))
        print(f"{mode_names[mode]:15} | {avg_us:10.2f} | {sps:12,}")

    # 2. 物理空間の全ジオメトリを走査して描画
    for row, pt in enumerate(test_points):
        pos, heading = pt["pos"], pt["heading"]
        for mode in modes:
            lidar_res = engine.cast_lidar(pos, heading=heading, mode=mode)
            ax = axes[row, mode]
            ax.set_title(f"{pt['label']}: {mode_names[mode]}")
            
            # --- 全ジオメトリの徹底描画 (フィルタなし) ---
            for i in range(env.model.ngeom):
                g_pos = env.data.geom_xpos[i][:2]
                g_sz = env.model.geom_size[i][:2]
                g_group = env.model.geom_group[i]
                g_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"g{i}"
                
                # 色とスタイルの決定
                if g_group == 0:
                    # グループ0（物理的実体）
                    if any(k in g_name.lower() for k in ["wall", "maze", "border"]):
                        color, alpha, ls = 'gray', 0.15, ':'
                    elif any(k in g_name.lower() for k in ["box", "ramp"]):
                        color, alpha, ls = 'orange', 0.3, '-'
                    else:
                        color, alpha, ls = 'blue', 0.1, '-' # エージェントの構成パーツなど
                else:
                    # グループ1以上（視覚用メタ / 犯人の可能性が高い）
                    color, alpha, ls = 'magenta', 0.6, '--'

                # 描画
                if g_sz[0] > 0.001 and g_sz[1] > 0.001:
                    rect = plt.Rectangle((g_pos[0]-g_sz[0], g_pos[1]-g_sz[1]), g_sz[0]*2, g_sz[1]*2, 
                                         fill=(g_group==0), facecolor=color, edgecolor=color, 
                                         linestyle=ls, alpha=alpha, linewidth=1.5 if g_group > 0 else 1)
                    ax.add_patch(rect)
                    if g_group > 0:
                        ax.text(g_pos[0], g_pos[1], f"G{g_group}:{g_name[:4]}", color='magenta', fontsize=7, ha='center')

            # --- 全エージェントの描画 (代表円) ---
            for bid in engine.idx_agent_body:
                p = env.data.xpos[bid]
                name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
                color = 'red' if 's' in name.lower() or 'seeker' in name.lower() else 'lime'
                
                # 物理サイズ 0.45m の円を最前面に描画
                circle = plt.Circle((p[0], p[1]), 0.45, color=color, alpha=0.8, ec='black', zorder=10)
                ax.add_patch(circle)
                label = "se" if 's' in name.lower() or 'seeker' in name.lower() else "hi"
                ax.text(p[0], p[1], label, ha='center', va='center', fontsize=12, fontweight='bold', zorder=11, color='black')

            # --- Lidarレイの描画 ---
            for i, d in enumerate(lidar_res):
                angle_rad = np.deg2rad(angles_deg[i]) + heading
                vx, vy = np.cos(angle_rad), np.sin(angle_rad)
                color = 'black' if mode==0 else ('green' if mode==1 else 'blue')
                
                ax.plot([pos[0], pos[0]+vx*d], [pos[1], pos[1]+vy*d], color=color, alpha=0.4, linewidth=1, zorder=5)
                ax.scatter(pos[0]+vx*d, pos[1]+vy*d, color=color, s=20, zorder=12)
                if d < 19.5:
                    ax.text(pos[0]+vx*d, pos[1]+vy*d, f"{i}", fontsize=9, color=color, zorder=13, fontweight='bold')

            # 視点
            ax.scatter(pos[0], pos[1], color='yellow', s=250, marker='*', zorder=20, edgecolors='black')
            
            ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5)
            ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig("lidar_benchmark_final_debug.png", dpi=150)
    print(f"\n✅ Benchmark report saved: lidar_benchmark_final_debug.png")

if __name__ == "__main__":
    run_benchmarked_audit()