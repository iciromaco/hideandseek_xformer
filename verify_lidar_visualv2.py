# verify_lidar_visual.py v2.27
# 演習第25回：Lidar 3モード完全検証（視覚化 ＆ 詳細数値レポート 統合版）
# 
# 修正履歴:
# v2.26: 属性整理。
# v2.27: ユーザーの指摘に基づき、詳細な数値レポート（全レイの距離・ヒット対象名）
#        の出力機能を復活。視覚的な直感と、数値的な正確性の両面で監査が可能。

import matplotlib
matplotlib.use('Agg') # 非対話環境対応

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from hns_environment import TeamCosEnv

def print_numerical_report(point_label, pos, heading, engine, env):
    """特定の地点におけるLidar全モードの数値をコンソールに詳しく出力する"""
    print(f"\n--- [Numerical Report: {point_label}] ---")
    print(f"Position: {pos}, Heading: {np.rad2deg(heading):.1f}°")
    
    # 自身のBody IDを取得（除外用）
    s_id = env.body_ids['s']
    
    # 全モードのデータを取得
    # Mode 0 (Native) はヒット名も確認するため mj_ray を伴うエンジン内部状態を利用
    l0 = engine.cast_lidar(pos, heading=heading, mode=0, body_exclude=s_id)
    # Mode 0 のヒットした geom 名を取得
    hit_names = []
    for i in range(12):
        # Native計測時の最新のヒットgeom IDを取得 (VisibilityEngine内部で mj_ray 実行時に更新される)
        # ※本来は cast_lidar 内部で取得するのが綺麗だが、互換性のためここで mj_ray を再現
        h_c, h_s = math.cos(heading), math.sin(heading)
        v_c, v_s = engine.base_cos[i], engine.base_sin[i]
        vx, vy = v_c*h_c - v_s*h_s, v_s*h_c + v_c*h_s
        p_orig = np.array([pos[0], pos[1], 0.4], dtype=np.float64)
        g_out = np.zeros(1, dtype=np.int32)
        mujoco.mj_ray(env.model, env.data, p_orig, np.array([vx, vy, 0.0]), None, 1, int(s_id), g_out)
        name = mujoco.mj_id2name(env.model, mujoco.mjtGeom.mjGEOM_GEOM, g_out[0]) if g_out[0] >= 0 else "None"
        hit_names.append(name)

    l1 = engine.cast_lidar(pos, heading=heading, mode=1, body_exclude=s_id)
    l2 = engine.cast_lidar(pos, heading=heading, mode=2, body_exclude=s_id)

    # テーブルヘッダー
    header = f"{'[Idx]':<6} | {'Angle':<6} | {'Native (M0)':<12} | {'Geom (M1)':<12} | {'Sphere (M2)':<12} | {'Hit Object (M0)'}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i in range(12):
        angle = int(engine.base_angles[i])
        print(f" {i:4d}  | {angle:4d}°  | {l0[i]:10.3f}m | {l1[i]:10.3f}m | {l2[i]:10.3f}m | {hit_names[i]}")

    # RMSEの計算
    rmse_1 = np.sqrt(np.mean((l0 - l1)**2))
    rmse_2 = np.sqrt(np.mean((l0 - l2)**2))
    print("-" * len(header))
    print(f"📊 RMSE (M1 vs M0): {rmse_1:.5f}m")
    print(f"📊 RMSE (M2 vs M0): {rmse_2:.5f}m")

def run_benchmarked_audit():
    print("🚀 verify_lidar_visual: Running full audit with visual & numerical reports...")
    
    env = TeamCosEnv()
    engine = env.vis_engine
    
    # 検証地点 (数値レポートもここを基準にする)
    test_points = [
        {"pos": np.array([-3.0, 3.0]), "heading": 0.0, "label": "Open Corner"},
        {"pos": np.array([0.0, 1.0]), "heading": np.deg2rad(45), "label": "Near Maze Center"}
    ]
    
    modes = [0, 1, 2] 
    mode_names = ["Native", "Geometric", "Sphere"]
    
    # 1. 数値レポートの出力
    for pt in test_points:
        print_numerical_report(pt["label"], pt["pos"], pt["heading"], engine, env)

    # 2. 性能計測
    print("\n" + "="*40)
    print("Performance Comparison (SPS)")
    print("-" * 40)
    for mode in modes:
        engine.cast_lidar(np.array([0., 0.]), mode=mode) # Warmup
        n_iters = 500
        t_start = time.perf_counter()
        for _ in range(n_iters):
            engine.cast_lidar(test_points[0]["pos"], heading=0.0, mode=mode)
        sps = int(n_iters / (time.perf_counter() - t_start))
        print(f"{mode_names[mode]:15} | {sps:12,} steps/sec")
    print("="*40)

    # 3. 視覚化
    fig, axes = plt.subplots(len(test_points), 3, figsize=(22, 14))
    for row, pt in enumerate(test_points):
        pos, heading = pt["pos"], pt["heading"]
        for mode in modes:
            lidar_res = engine.cast_lidar(pos, heading=heading, mode=mode)
            ax = axes[row, mode]
            ax.set_title(f"{pt['label']}: {mode_names[mode]}", fontsize=14, fontweight='bold')
            
            # 物理要素の描画
            for i in range(env.model.ngeom):
                p = env.data.geom_xpos[i][:2]; s = env.model.geom_size[i][:2]
                n = mujoco.mj_id2name(env.model, mujoco.mjtGeom.mjGEOM_GEOM, i) or ""
                if any(k in n.lower() for k in ["wall", "maze", "border", "box", "slope_surface"]):
                    color = 'gray' if "wall" in n.lower() else 'orange'
                    ax.add_patch(plt.Rectangle((p[0]-s[0], p[1]-s[1]), s[0]*2, s[1]*2, facecolor=color, alpha=0.15))
                elif "_btm" in n.lower():
                    c = 'red' if 'seeker' in n else 'lime'
                    ax.add_patch(plt.Circle((p[0], p[1]), s[0], color=c, alpha=0.2, ec='black'))

            # Lidarレイの描画
            for i, d in enumerate(lidar_res):
                angle_rad = np.deg2rad(engine.base_angles[i]) + heading
                vx, vy = np.cos(angle_rad), np.sin(angle_rad)
                color = 'black' if mode==0 else ('green' if mode==1 else 'blue')
                ax.plot([pos[0], pos[0]+vx*d], [pos[1], pos[1]+vy*d], color=color, alpha=0.4, linewidth=1)
                ax.scatter(pos[0]+vx*d, pos[1]+vy*d, color=color, s=20)

            ax.scatter(pos[0], pos[1], color='yellow', s=200, marker='*', edgecolors='black', zorder=20)
            ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lidar_benchmark_final_debug.png", dpi=150)
    print(f"\n✅ Benchmark report saved: lidar_benchmark_final_debug.png")

if __name__ == "__main__":
    run_benchmarked_audit()