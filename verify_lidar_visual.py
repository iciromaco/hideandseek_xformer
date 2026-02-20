# verify_lidar_visual.py v2.61
# 演習第25回：Lidar 3モード・完全物理監査 ＆ 自己透過（Group判定是正）版
# 
# 修正履歴:
# v2.60: 詳細デバッグ出力追加。
# v2.61: ユーザー報告の Native (M0) 15m問題を解消。
#        1. mj_ray の geomgroup に [1,0,0,0,0,0] を指定し、Group 1 (装飾パーツ) を透過。
#        2. flg_static=0 に是正し、動的な箱やハイダーを Native 検知対象に含める。
#        3. プロット図のタイトルに計算速度 (SPS) を埋め込み、情報の密度を強化。

import matplotlib
matplotlib.use('Agg')
import mujoco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import math
from hns_environment import TeamCosEnv
import main18_optimization as base_config
ml_string = base_config.XML_CONTENT
print("XXXX",ml_string.body('seeker_body').id)

def get_rotated_rect_corners(pos, size, quat):
    """回転を考慮した矩形の4頂点を計算"""
    # クォータニオンからYaw角を抽出
    yaw = math.atan2(2.0*(quat[0]*quat[3] + quat[1]*quat[2]), 1.0 - 2.0*(quat[2]*quat[2] + quat[3]*quat[3]))
    c, s = math.cos(yaw), math.sin(yaw)
    offsets = np.array([[-size[0], -size[1]], [size[0], -size[1]], [size[0], size[1]], [-size[0], size[1]]])
    rotated = np.zeros((4, 2))
    for i in range(4):
        rotated[i, 0] = pos[0] + offsets[i, 0] * c - offsets[i, 1] * s
        rotated[i, 1] = pos[1] + offsets[i, 0] * s + offsets[i, 1] * c
    return rotated

def run_comprehensive_audit():
    print("🚀 verify_lidar_visual: Running High-Fidelity Lidar Audit v2.61...")
    
    modes = [0, 1, 2]
    mode_names = ["Native (M0)", "Geometric (M1)", "SphereTr. (M2)"]
    mode_colors = ["red", "green", "blue"]
    
    env = TeamCosEnv(); engine = env.vis_engine; m, d = env.model, env.data; s_id = env.body_ids['s']
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    # ベンチマーク用統計
    perf_stats = {m: [] for m in modes}
    # Group 0 (主要パーツ) のみを検知し、Group 1 (装飾パーツ) を無視するマスク
    g_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    for r in range(2):
        # 1. 物理配置のランダム化
        env.reset(seed=int(time.time() * 1000) % 10000 + r * 123)
        # Seekerの最新座標と向きを物理エンジンから直接取得
        s_pos = d.xpos[s_id][:2].copy()
        s_heading = d.qpos[m.jnt_qposadr[env.qpos_indices['s']['rot']]]
        mujoco.mj_forward(m, d)

        print(f"\n--- [Scenario {r+1}: Randomized Placement] ---")
        print(f"📍 Seeker at: {s_pos}, Heading: {np.rad2deg(s_heading):.1f}°")

        # 3モード計測
        l_res = []
        for m_idx in modes:
            t0 = time.perf_counter()
            res = engine.cast_lidar(s_pos, heading=s_heading, mode=m_idx, body_exclude=s_id)
            perf_stats[m_idx].append(time.perf_counter() - t0)
            l_res.append(res)

        # --- 【Native ヒット対象の精密特定】 ---
        hit_details = []
        p_orig = np.array([s_pos[0], s_pos[1], 0.4], dtype=np.float64) # 中心高度0.4
        
        if r == 0: print("\n🔍 [Native Mode Debug Probe (Scenario 1)]")
        for i in range(12):
            ang = np.deg2rad(engine.base_angles[i]) + s_heading
            v_dir = np.array([np.cos(ang), np.sin(ang), 0.0])
            g_out = np.array([-1], dtype=np.int32)
            
            # g_maskを適用し、flg_static=0 (全ボディ対象) で実行
            # dist = mujoco.mj_ray(m, d, p_orig, v_dir, g_mask, 0, int(s_id), g_out)
            mujoco.MjModel.from_xml_string(XML_CONTENT)
            selfanchor = "seeker_body"
            selfid =   mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, selfanchor)
            dist = mujoco.mj_ray(m, d, p_orig, v_dir, g_mask, 0, selfid, g_out)
            hit_name = "None"
            # 0.41m以上を外界の衝突として受理
            # if dist > 0.41 and g_out[0] >= 0:
            hit_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g_out[0]) or f"ID:{g_out[0]}"
            
            hit_details.append({"name": hit_name, "dist": dist if dist > 0.41 else 15.0})
            
            if r == 0:
                print(f"  Ray {i:2d} ({int(engine.base_angles[i]):4d}°): dist={dist:8.3f}m | hit='{hit_name}'")

        # 数値レポート出力
        header = f"{'[Idx]':<6} | {'Angle':<6} | {'Native (M0)':<12} | {'Geom (M1)':<12} | {'Sphere (M2)':<12} | {'Hit Object (M0)'}"
        print("\n" + "-" * len(header)); print(header); print("-" * len(header))
        for i in range(12):
            ang = int(engine.base_angles[i])
            print(f" {i:4d}  | {ang:4d}°  | {l_res[0][i]:10.3f}m | {l_res[1][i]:10.3f}m | {l_res[2][i]:10.3f}m | {hit_details[i]['name']}")
        print("-" * len(header))
        
        rmse = np.sqrt(np.mean((l_res[0]-l_res[1])**2))
        print(f"📊 RMSE (M1 vs M0): {rmse:.5f}m")

        # プロット描画
        for m_idx in modes:
            ax = axes[r, m_idx]
            # 各モードの現在の推定平均SPSを表示
            avg_time = np.mean(perf_stats[m_idx]) if perf_stats[m_idx] else 0.001
            sps_label = f" | {int(1.0/avg_time):,} SPS"
            ax.set_title(f"{mode_names[m_idx]} (Scenario {r+1}){sps_label}", fontsize=12, fontweight='bold')
            
            # 物理要素の描画 (Group 0 のみ)
            for i in range(m.ngeom):
                if m.geom_group[i] != 0: continue
                gp, gs = d.geom_xpos[i][:2], m.geom_size[i][:2]
                name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
                
                if any(k in name for k in ["wall", "maze", "border"]):
                    ax.add_patch(plt.Rectangle((gp[0]-gs[0], gp[1]-gs[1]), gs[0]*2, gs[1]*2, color='gray', alpha=0.3, ec='black'))
                elif any(k in name for k in ["box", "slope"]):
                    bid = m.geom_bodyid[i]
                    corners = get_rotated_rect_corners(d.xpos[bid][:2], gs, d.xquat[bid])
                    ax.add_patch(patches.Polygon(corners, closed=True, color='orange', alpha=0.4, ec='darkorange'))
                elif "_btm" in name:
                    c = 'red' if 'seeker' in name else 'lime'
                    ax.add_patch(plt.Circle((gp[0], gp[1]), gs[0], color=c, alpha=0.5, ec='black', zorder=10))

            # Lidarレイ描画
            for i, dist in enumerate(l_res[m_idx]):
                ang = np.deg2rad(engine.base_angles[i]) + s_heading
                vx, vy = np.cos(ang), np.sin(ang)
                target_x, target_y = s_pos[0] + vx*dist, s_pos[1] + vy*dist
                ax.plot([s_pos[0], target_x], [s_pos[1], target_y], color=mode_colors[m_idx], alpha=0.3)
                if dist < engine.max_dist - 0.1:
                    ax.scatter(target_x, target_y, color=mode_colors[m_idx], s=35, edgecolors='white', zorder=15)
            
            ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect('equal')
            ax.grid(True, linestyle=':', alpha=0.5); ax.set_facecolor('#fdfdfd')

    # 2. パフォーマンス統計の出力 (詳細版)
    print(f"\n{' Lidar Performance Benchmark (per Lidar Call) ':^60}\n" + "-" * 60)
    for m_idx in modes:
        avg_time = np.mean(perf_stats[m_idx]) * 1000 # ms
        sps = int(1.0 / (avg_time / 1000.0))
        print(f" {mode_names[m_idx]:15} : {avg_time:8.4f} ms/call ({sps:10,} SPS)")
    print("-" * 60)

    plt.tight_layout(); plt.savefig("lidar_visual_audit_v261.png", dpi=150)
    print(f"\n✅ Visual audit report saved: lidar_visual_audit_v261.png")

if __name__ == "__main__": run_comprehensive_audit()