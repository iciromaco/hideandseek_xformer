# verify_lidar_alignment.py v2.48
# 演習第25回：Lidar監査 決定版（mj_ray引数サイズ厳密是正 ＆ 3モード性能統合）
# 
# 修正履歴:
# v2.47: ベンチマークフェーズの TypeError 解消。
# v2.48: ユーザー報告の TypeError (mj_ray geomgroup size mismatch: 5 vs 6) を完全に解消。
#        1. 内部の g_mask 定義を [1, 0, 0, 0, 0, 0] (要素数6) に統一。
#        2. VisibilityEngine 側の v1.55 修正と合わせることで、ベンチマーク中の
#           engine.cast_lidar も含めて正常動作を保証。

import mujoco
import mujoco.viewer
import numpy as np
import time
import math
from hns_environment import TeamCosEnv

def run_alignment_audit():
    print("🚀 Lidar Alignment Audit v2.48 (Full Integrated Spec)")
    
    # 環境とエンジンのロード
    env = TeamCosEnv()
    m, d = env.model, env.data
    engine = env.vis_engine
    s_id = env.body_ids['s']
    
    # 1. 干渉オブジェクトの退避
    for obj in ["box1", "box2", "ramp"]:
        try:
            jnt_id = m.joint(f"{obj}_joint").id; q_adr = m.jnt_qposadr[jnt_id]
            d.qpos[q_adr:q_adr+2] = [10.0, 10.0]; d.qpos[q_adr+2] = 0.5
        except: pass

    # 2. 理論値計算
    H1_CENTER_DIST = np.sqrt(3.0**2 + (-3.0)**2)
    H1_IDEAL_DIST = H1_CENTER_DIST - 0.4
    
    # 3. テスト用ターゲットの固定配置
    planned_positions = {
        'h1': [3.0, -3.0, 0.0, 0.5], # Z=0.0 (Global Z=0.4)
        'h2': [0.0, 4.0, 0.0, 0.0],  # Z=0.0 (Global Z=0.4)
    }
    for p in ["h1", "h2"]:
        if p in env.qpos_indices:
            idxs = env.qpos_indices[p]; pos = planned_positions[p]
            d.qpos[m.jnt_qposadr[idxs['x']]] = pos[0]; d.qpos[m.jnt_qposadr[idxs['y']]] = pos[1]
            d.qpos[m.jnt_qposadr[idxs['z']]] = pos[2]; d.qpos[m.jnt_qposadr[idxs['rot']]] = pos[3]
    
    # シーカーを原点、東向き(0°)に固定
    s_x_adr, s_y_adr, s_rot_adr = m.jnt_qposadr[env.qpos_indices['s']['x']], m.jnt_qposadr[env.qpos_indices['s']['y']], m.jnt_qposadr[env.qpos_indices['s']['rot']]
    d.qpos[s_x_adr] = 0.0; d.qpos[s_y_adr] = 0.0; d.qpos[s_rot_adr] = 0.0
    
    mujoco.mj_forward(m, d)

    # 4. 理論的な正解マップ
    ideal_results = {
        0:  ("wall_e", 6.00),      
        6:  ("hider1_btm", H1_IDEAL_DIST), 
        7:  ("maze_w3", 1.50),     
        8:  ("maze_w2", 1.50),     
        9:  ("wall_n", 8.50),      
        10: ("maze_w1", 2.12),     
        11: ("wall_w", 6.00),      
    }

    print(f"🧪 Audit: Native(M0) vs Theory | Scan Height=0.4m")
    print("-" * 180)
    header = f" [Idx] | RelAng | {'Hit Target (Native)':<20} | Native(M0) | Geom(M1) | Sphere(M2) | {'Ideal Target':<15} | Ideal Dist | Diff(M0-Idl)"
    print(header)
    print("-" * 180)
    
    # --- 計測 ---
    l1 = engine.cast_lidar(np.array([0.0, 0.0]), heading=0.0, mode=1, body_exclude=s_id)
    l2 = engine.cast_lidar(np.array([0.0, 0.0]), heading=0.0, mode=2, body_exclude=s_id)
    
    l0 = np.zeros(12)
    p_orig = np.array([0.0, 0.0, 0.4], dtype=np.float64) 
    # 【重要】MuJoCo API が要求する要素数 6 の配列。
    g_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8) 
    
    angles_rad = np.deg2rad([0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180])
    
    for i in range(12):
        angle_val = int(np.rad2deg(angles_rad[i]))
        vx, vy = np.cos(angles_rad[i]), np.sin(angles_rad[i])
        g_id = np.array([-1], dtype=np.int32)
        
        # Native計測実行
        dist = mujoco.mj_ray(m, d, p_orig, np.array([vx, vy, 0.0]), g_mask, 1, int(s_id), g_id)
        
        dist_m0 = dist if dist > 0.45 else 15.0
        l0[i] = dist_m0
        
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g_id[0]) if (g_id[0] >= 0 and dist > 0.45) else "None"
        
        ideal_target, ideal_dist = ideal_results.get(i, ("-", 0.0))
        diff_m0_ideal = abs(dist_m0 - ideal_dist) if ideal_dist > 0 else 0.0
        mark = "!!" if diff_m0_ideal > 0.05 and ideal_dist > 0 else ""
        
        ideal_dist_str = f"{ideal_dist:8.2f}m" if ideal_dist > 0 else f"{'-':>9}"
        diff_str = f"{diff_m0_ideal:10.3f}m" if ideal_dist > 0 else f"{'-':>11}"

        print(f"  {i:2d}   | {angle_val:4d}°  | {name:<20} | {dist_m0:8.2f}m | {l1[i]:8.2f}m | {l2[i]:10.2f}m | {ideal_target:<15} | {ideal_dist_str} | {diff_str} {mark}")
    
    print("-" * 180)
    
    # 5. パフォーマンス・ベンチマーク
    print(f"\n{' Lidar Performance Benchmark (1000 Iterations) ':^60}")
    print("-" * 60)
    mode_names = ["Native (M0)", "Geometric (M1)", "SphereTr. (M2)"]
    for m_idx in [0, 1, 2]:
        # ウォームアップ（VisibilityEngine v1.55 によりマスクサイズ 6 が保証される）
        engine.cast_lidar(np.array([0.0, 0.0]), mode=m_idx, body_exclude=s_id) 
        t0 = time.perf_counter()
        for _ in range(1000):
            engine.cast_lidar(np.array([0.0, 0.0]), mode=m_idx, body_exclude=s_id)
        t_elapsed = time.perf_counter() - t0
        sps = int(1000 / t_elapsed)
        ms_per_op = (t_elapsed / 1000) * 1000
        print(f" {mode_names[m_idx]:15} : {ms_per_op:8.4f} ms/call ({sps:10,} SPS)")
    print("-" * 60)

    rmse = np.sqrt(np.mean((l0 - l1)**2))
    print(f"📊 Overall Consistency: RMSE(Native vs Geom) = {rmse:.4f}m")

if __name__ == "__main__":
    run_alignment_audit()