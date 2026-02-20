# verify_lidar_visual.py v2.66
# 演習第25回：最速演算性能（10万SPS超）復元 ＆ 統合監査レポート
# 
# 修正履歴:
# v2.65: 統合版。
# v2.66: ユーザー報告の速度低下を解消するための最終リファクタリング。
#        1. VisibilityEngine v1.70 と同期し、ボックス計算を JIT 内へ封印。
#        2. 1000回繰り返し計測時の Python 側オーバーヘッドを極限まで排除。
#        3. コンソール出力を以前の正確なフォーマットに維持。

import matplotlib
matplotlib.use('Agg')
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import math
from hns_environment import TeamCosEnv

def get_rotated_rect_corners(pos, size, quat):
    """回転を考慮した矩形の4頂点を計算 (MuJoCoクォータニオン対応)"""
    yaw = math.atan2(2.0*(quat[0]*quat[3] + quat[1]*quat[2]), 1.0 - 2.0*(quat[2]*quat[2] + quat[3]*quat[3]))
    c, s = math.cos(yaw), math.sin(yaw)
    offsets = np.array([[-size[0], -size[1]], [size[0], -size[1]], [size[0], size[1]], [-size[0], size[1]]])
    rotated = np.zeros((4, 2))
    for i in range(4):
        rotated[i, 0] = pos[0] + offsets[i, 0] * c - offsets[i, 1] * s
        rotated[i, 1] = pos[1] + offsets[i, 0] * s + offsets[i, 1] * c
    return rotated

def run_comprehensive_audit():
    print("="*120)
    print("🚀 verify_lidar_visual v2.66: High-Performance Lidar Audit Starting...")
    print("="*120)
    
    env = TeamCosEnv()
    engine = env.vis_engine
    m, d = env.model, env.data
    s_id = env.body_ids['s']
    
    modes = [0, 1, 2]
    mode_names = ["Native (M0)", "Geometric (M1)", "Sphere (M2)"]
    mode_colors = ["red", "forestgreen", "royalblue"]
    
    fig, axes = plt.subplots(2, 3, figsize=(28, 18))
    perf_stats = {m: [] for m in modes}

    # 監査用定数
    g_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    for scenario in range(2):
        print(f"\n--- [Scenario {scenario+1}: Randomized Placement Audit] ---")
        env.reset(seed=int(time.time()*100)%1000 + scenario*11)
        mujoco.mj_forward(m, d)
        
        s_pos = d.xpos[s_id][:2].copy()
        s_heading = d.qpos[m.jnt_qposadr[env.qpos_indices['s']['rot']]]
        print(f"📍 Seeker Pose: pos={s_pos}, heading={np.rad2deg(s_heading):.1f}°")

        # 計測実行
        results = []
        for m_idx in modes:
            results.append(engine.cast_lidar(s_pos, heading=s_heading, mode=m_idx, body_exclude=s_id))

        # ヒット対象特定
        hit_names = []
        p_base = np.array([s_pos[0], s_pos[1], 0.4], dtype=np.float64)
        for i in range(12):
            ang = np.deg2rad(engine.base_angles[i]) + s_heading
            v_dir = np.array([np.cos(ang), np.sin(ang), 0.0])
            g_out = np.array([-1], dtype=np.int32)
            d_hit = mujoco.mj_ray(m, d, p_base, v_dir, g_mask, 1, int(s_id), g_out)
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g_out[0]) if (d_hit > 0.41 and g_out[0] >= 0) else "None"
            hit_names.append(name)

        # 数値テーブル表示
        header = f"{'[Idx]':<6} | {'Angle':<6} | {'Native (M0)':<12} | {'Geom (M1)':<12} | {'Sphere (M2)':<12} | {'Hit Object (M0)'}"
        print("-" * len(header)); print(header); print("-" * len(header))
        for i in range(12):
            ang = int(engine.base_angles[i])
            print(f" {i:4d}  | {ang:4d}°  | {results[0][i]:10.3f}m | {results[1][i]:10.3f}m | {results[2][i]:10.3f}m | {hit_names[i]}")
        print("-" * len(header))
        
        rmse_m1 = np.sqrt(np.mean((results[0]-results[1])**2))
        print(f"📊 Accuracy: RMSE(M1 vs M0) = {rmse_m1:.5f}m")

        # プロット描画 (可視化)
        for m_idx in modes:
            ax = axes[scenario, m_idx]
            ax.set_title(f"{mode_names[m_idx]} (Scenario {scenario+1})", fontsize=14, fontweight='bold')
            for i in range(m.ngeom):
                if m.geom_group[i] != 0: continue
                gp, gs = d.geom_xpos[i][:2], m.geom_size[i][:2]
                gn = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
                if any(k in gn for k in ["wall", "maze", "border"]):
                    ax.add_patch(plt.Rectangle((gp[0]-gs[0], gp[1]-gs[1]), gs[0]*2, gs[1]*2, color='gray', alpha=0.3, ec='black'))
                elif any(k in gn for k in ["box", "slope"]):
                    bid = m.geom_bodyid[i]
                    corners = get_rotated_rect_corners(d.xpos[bid][:2], gs, d.xquat[bid])
                    ax.add_patch(patches.Polygon(corners, closed=True, color='orange', alpha=0.4, ec='darkorange'))
                elif "_btm" in gn:
                    c = 'red' if 'seeker' in gn else 'lime'
                    ax.add_patch(plt.Circle((gp[0], gp[1]), gs[0], color=c, alpha=0.5, ec='black', zorder=10))
            for i, dist in enumerate(results[m_idx]):
                ang = np.deg2rad(engine.base_angles[i]) + s_heading
                vx, vy = np.cos(ang), np.sin(ang)
                tx, ty = s_pos[0]+vx*dist, s_pos[1]+vy*dist
                ax.plot([s_pos[0], tx], [s_pos[1], ty], color=mode_colors[m_idx], alpha=0.3)
                if dist < engine.max_dist - 0.1:
                    ax.scatter(tx, ty, color=mode_colors[m_idx], s=40, edgecolors='white', zorder=15)
            ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect('equal'); ax.grid(True, linestyle=':', alpha=0.3)

    # --- 最終パフォーマンスサマリー (1000 Iterations) ---
    print(f"\n{' Lidar Performance Benchmark Summary (1000 Iterations) ':^60}\n" + "-" * 60)
    for m_idx in modes:
        # ウォームアップ (JIT等)
        engine.cast_lidar(s_pos, heading=s_heading, mode=m_idx, body_exclude=s_id)
        t0 = time.perf_counter()
        for _ in range(1000):
            engine.cast_lidar(s_pos, heading=s_heading, mode=m_idx, body_exclude=s_id)
        t_total = time.perf_counter() - t0
        avg_ms = (t_total / 1000) * 1000
        sps = int(1000 / t_total)
        print(f" {mode_names[m_idx]:15} : {avg_ms:8.4f} ms/call ({sps:10,} SPS)")
    print("-" * 60)

    plt.tight_layout(); plt.savefig("lidar_visual_audit_v266.png", dpi=150)
    print(f"\n✅ Visual audit complete. Report saved: lidar_visual_audit_v266.png")

if __name__ == "__main__":
    run_comprehensive_audit()