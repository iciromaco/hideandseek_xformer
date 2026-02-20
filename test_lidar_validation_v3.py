# test_lidar_validation_v3.py v3.2
# 演習第25回：物理中心完全同期 ＆ ランダム配置・性能統計統合監査
# 
# 修正履歴:
# v3.1: デバッグ情報強化。
# v3.2: ユーザー報告の不整合（Scenario 1 RMSE 1.4m）を解消するための完全版。
#        1. ランダム配置による多角的な物理監査。
#        2. Native(M0) の mj_ray フラグをエンジン仕様 (flg_static=1) と完全同期。
#        3. パフォーマンスベンチマーク (SPS) を監査末尾に統合。

import numpy as np
import mujoco
import time
import math
from hns_environment import TeamCosEnv

def run_lidar_triple_audit():
    print("="*120)
    print(f"🚀 LIDAR TRIPLE-MODE HIGH-FIDELITY AUDIT v3.2")
    print(f"   Focus: Physical Consistency & SPS Performance Recovery")
    print("="*120 + "\n")

    env = TeamCosEnv()
    engine = env.vis_engine
    m, d = env.model, env.data
    s_id = env.body_ids['s']

    modes = [0, 1, 2]
    mode_names = ["Native (M0)", "Geometric(M1)", "Sphere (M2)"]
    perf_log = {m_idx: [] for m_idx in modes}

    # 監査シナリオ（ランダム配置による2つの試行）
    for r in range(2):
        print(f"--- [Scenario {r+1}: Randomized Placement Audit] ---")
        # 物理配置のリセット
        env.reset(seed=int(time.time() * 100) % 5000 + r)
        s_pos = d.xpos[s_id][:2].copy()
        s_heading = d.qpos[m.jnt_qposadr[env.qpos_indices['s']['rot']]]
        mujoco.mj_forward(m, d)
        
        print(f"📍 Seeker at: {s_pos}, Heading: {np.rad2deg(s_heading):.1f}°")

        # 1. 3モード一括計測
        l_res = []
        for m_idx in modes:
            t0 = time.perf_counter()
            res = engine.cast_lidar(s_pos, heading=s_heading, mode=m_idx, body_exclude=s_id)
            perf_log[m_idx].append(time.perf_counter() - t0)
            l_res.append(res)

        # 2. Native ヒット対象詳細特定 (デバッグ用)
        hit_names = []
        p_base = np.array([s_pos[0], s_pos[1], 0.4], dtype=np.float64) # 中心高度0.4m
        g_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8) # Group 0 (主要部品) のみ検知
        
        for i in range(12):
            ang = np.deg2rad(engine.base_angles[i]) + s_heading
            vx, vy = np.cos(ang), np.sin(ang)
            # 物理エンジン側のフラグ (flg_static=1) を使用
            g_out = np.array([-1], dtype=np.int32)
            dist = mujoco.mj_ray(m, d, p_base, np.array([vx, vy, 0.0]), g_mask, 1, int(s_id), g_out)
            
            if dist > 0.41 and g_out[0] >= 0:
                hit_names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g_out[0]) or f"ID:{g_out[0]}")
            else:
                hit_names.append("None")

        # 3. 数値レポート表示
        header = f"{'[Idx]':<6} | {'Angle':<6} | {'Native (M0)':<12} | {'Geom (M1)':<12} | {'Sphere (M2)':<12} | {'Hit Object (M0)'}"
        print("-" * len(header)); print(header); print("-" * len(header))
        for i in range(12):
            print(f" {i:4d}  | {int(engine.base_angles[i]):4d}°  | {l_res[0][i]:10.3f}m | {l_res[1][i]:10.3f}m | {l_res[2][i]:10.3f}m | {hit_names[i]}")
        print("-" * len(header))
        
        rmse_m1 = np.sqrt(np.mean((l_res[0] - l_res[1])**2))
        rmse_m2 = np.sqrt(np.mean((l_res[0] - l_res[2])**2))
        print(f"📊 Accuracy: RMSE(M1 vs M0) = {rmse_m1:.5f}m | RMSE(M2 vs M0) = {rmse_m2:.5f}m")
        print("-" * 60 + "\n")

    # 4. 最終パフォーマンス・ベンチマーク
    print(f"\n{' Lidar Performance Benchmark (1000 Iterations) ':^60}")
    print("-" * 60)
    for m_idx in modes:
        # ウォームアップ
        engine.cast_lidar(s_pos, heading=s_heading, mode=m_idx, body_exclude=s_id)
        t_bench = time.perf_counter()
        for _ in range(1000):
            engine.cast_lidar(s_pos, heading=s_heading, mode=m_idx, body_exclude=s_id)
        elapsed = time.perf_counter() - t_bench
        sps = int(1000 / elapsed)
        ms_per_call = (elapsed / 1000) * 1000
        print(f" {mode_names[m_idx]:15} : {ms_per_call:8.4f} ms/call ({sps:10,} SPS)")
    print("-" * 60)

    print("\n🏁 Audit Conclusion:")
    print("If RMSE is < 0.05m and SPS for M1/M2 is > 30,000, the refactoring is SUCCESSFUL.")

if __name__ == "__main__":
    run_lidar_triple_audit()