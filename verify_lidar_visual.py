# verify_lidar_visual.py v2.26
# 演習第25回：Lidar 3モード検証・物理空間全要素可視化（VisibilityEngine v1.38 同期版）
# 
# 修正履歴:
# v2.25: 装飾パーツ含めた可視化。
# v2.26: VisibilityEngine v1.38 の変数名変更(base_angles)および属性整理に同期し AttributeError を解消。

import matplotlib
# 非対話型環境での警告回避
matplotlib.use('Agg')

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
from hns_environment import TeamCosEnv

def run_benchmarked_audit():
    """エンジンの監査と、装飾パーツ・全エージェントを含めた視覚的整合性の確認"""
    print("🚀 verify_lidar_visual: Running high-performance audit with ALL physical elements...")
    
    env = TeamCosEnv()
    engine = env.vis_engine
    # 修正: VisibilityEngine v1.38 では base_angles を使用
    angles_deg = engine.base_angles

    # 検証地点
    test_points = [
        {"pos": np.array([-3.0, 3.0]), "heading": 0.0, "label": "First View"},
        {"pos": np.array([0.0, 1.0]), "heading": np.deg2rad(45), "label": "Last View"}
    ]
    
    modes = [0, 1, 2] 
    mode_names = ["Native", "Geometric", "Sphere"]
    
    # --- 1. 詳細なターゲット統計の出力 ---
    print("\n" + "="*75)
    print("📊 LIDAR CALCULATION AUDIT (Ground Truth Physics)")
    print("="*75)
    print(f"  - Static Wall Geoms:   {len(engine.idx_wall_geom)}")
    print(f"  - Dynamic Box Geoms:   {len(engine.idx_box_geom)}")
    print(f"  - Active Agent Bodies: {len(engine.idx_agent_body)} (_btm only)")
    print("="*75)

    fig, axes = plt.subplots(len(test_points), 3, figsize=(22, 14))
    
    # --- 2. 高精度な統計計測 ---
    print("\nPerformance Comparison:")
    print("Mode            |   Avg (μs) |        SPS")
    print("-" * 40)

    for mode in modes:
        # ウォームアップ (JITコンパイル実行)
        engine.cast_lidar(np.array([0., 0.]), mode=mode)
        
        n_iters = 500 
        t_start = time.perf_counter()
        for _ in range(n_iters):
            engine.cast_lidar(test_points[0]["pos"], heading=0.0, mode=mode)
        t_total = time.perf_counter() - t_start
        
        avg_us = (t_total / n_iters) * 1e6
        sps = int(1.0 / (t_total / n_iters))
        print(f"{mode_names[mode]:15} | {avg_us:10.2f} | {sps:12,}")

    # --- 3. 視覚的な整合性チェック ---
    for row, pt in enumerate(test_points):
        pos, heading = pt["pos"], pt["heading"]
        for mode in modes:
            lidar_res = engine.cast_lidar(pos, heading=heading, mode=mode)
            ax = axes[row, mode]
            ax.set_title(f"{pt['label']}: {mode_names[mode]}", fontsize=14, fontweight='bold')
            
            # (A) 物理空間の全ジオメトリを走査
            for i in range(env.model.ngeom):
                p = env.data.geom_xpos[i][:2]
                s = env.model.geom_size[i][:2]
                n = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                
                # 壁、箱、スロープの表面
                if any(k in n.lower() for k in ["wall", "maze", "border", "box", "slope_surface"]):
                    color = 'gray' if "wall" in n.lower() else 'orange'
                    rect = plt.Rectangle((p[0]-s[0], p[1]-s[1]), s[0]*2, s[1]*2, 
                                         facecolor=color, alpha=0.15, edgecolor=color, linestyle=':')
                    ax.add_patch(rect)
                
                # エージェントの各パーツ (btm, nose, tail, capsule)
                elif any(k in n.lower() for k in ["nose", "tail", "capsule", "btm"]):
                    is_btm = "btm" in n.lower()
                    if is_btm:
                        color = 'red' if 'seeker' in n else 'lime'
                        circle = plt.Circle((p[0], p[1]), s[0], color=color, alpha=0.2, ec='black', linewidth=0.5)
                        ax.add_patch(circle)
                    else:
                        rect = plt.Rectangle((p[0]-s[0], p[1]-s[1]), s[0]*2, s[1]*2, 
                                             fill=False, edgecolor='cyan', linestyle='--', alpha=0.4, linewidth=0.8)
                        ax.add_patch(rect)

            # (B) 全エージェントの描画
            for i in range(env.model.nbody):
                bn = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
                if any(k in bn.lower() for k in ["seeker", "hider", "s_body", "h1_body", "h2_body"]):
                    if "anchor" in bn.lower(): continue
                    p_body = env.data.xpos[i][:2]
                    color = 'red' if 'se' in bn.lower() else 'lime'
                    circle = plt.Circle((p_body[0], p_body[1]), 0.45, color=color, alpha=0.6, ec='black', zorder=10)
                    ax.add_patch(circle)

            # (C) Lidarレイの描画
            for i, d in enumerate(lidar_res):
                angle_rad = np.deg2rad(angles_deg[i]) + heading
                vx, vy = np.cos(angle_rad), np.sin(angle_rad)
                color = 'black' if mode==0 else ('green' if mode==1 else 'blue')
                
                ax.plot([pos[0], pos[0]+vx*d], [pos[1], pos[1]+vy*d], color=color, alpha=0.4, linewidth=1, zorder=5)
                ax.scatter(pos[0]+vx*d, pos[1]+vy*d, color=color, s=25, zorder=12)
                if d < engine.max_dist - 0.1:
                    ax.text(pos[0]+vx*d, pos[1]+vy*d, f"{i}", fontsize=9, color=color, zorder=13, fontweight='bold')

            # (D) 視点マーカー
            ax.scatter(pos[0], pos[1], color='yellow', s=250, marker='*', zorder=20, edgecolors='black', linewidth=1)
            
            ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5)
            ax.set_aspect('equal'); ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    plt.savefig("lidar_benchmark_final_debug.png", dpi=150)
    print(f"\n✅ Benchmark report saved: lidar_benchmark_final_debug.png")

if __name__ == "__main__":
    run_benchmarked_audit()