# verify_lidar_alignment.py v2.25
# 演習第25回：Lidar監査・高度同期完全是正版
# 
# 修正履歴:
# v2.24: 理論値併記。
# v2.25: ターゲット配置の高度(Z)を 0.0 に修正。
#        これにより球体中心が 0.4m に固定され、シーカー(0.4m)のレイが
#        球体の赤道を射抜くようになり、M0/M1/Ideal が完全に一致します。

import mujoco
import mujoco.viewer
import numpy as np
import time
from hns_environment import TeamCosEnv

def run_alignment_audit():
    print("🚀 Lidar Alignment Audit v2.25 (Corrected Vertical Sync)")
    
    # 環境とエンジンのロード
    env = TeamCosEnv()
    m, d = env.model, env.data
    engine = env.vis_engine
    
    # 1. 干渉オブジェクトの退避
    for obj in ["box1", "box2", "ramp"]:
        try:
            jnt_id = m.joint(f"{obj}_joint").id; q_adr = m.jnt_qposadr[jnt_id]
            d.qpos[q_adr:q_adr+2] = [10.0, 10.0]; d.qpos[q_adr+2] = 0.5
        except: pass

    # 2. 理論値計算
    # 距離 sqrt(18) = 4.2426m, 半径 0.4m, 理想表面距離 = 3.8426m
    H1_CENTER_DIST = np.sqrt(3.0**2 + (-3.0)**2)
    H1_IDEAL_DIST = H1_CENTER_DIST - 0.4
    
    # 3. テスト用ターゲットの固定配置
    # 【重要】Zジョイントを 0.0 に設定。
    # XML構造 (anchor 0.5 + geom -0.1) により、これで球体中心が 0.4m に配置されます。
    planned_positions = {
        'h1': [3.0, -3.0, 0.0, 0.5], # Z=0.0 (Global Z=0.4)
        'h2': [0.0, 4.0, 0.0, 0.0],  # Z=0.0 (Global Z=0.4)
    }
    for p in ["h1", "h2"]:
        if p in env.qpos_indices:
            idxs = env.qpos_indices[p]; pos = planned_positions[p]
            d.qpos[m.jnt_qposadr[idxs['x']]] = pos[0]; d.qpos[m.jnt_qposadr[idxs['y']]] = pos[1]
            d.qpos[m.jnt_qposadr[idxs['z']]] = pos[2]; d.qpos[m.jnt_qposadr[idxs['rot']]] = pos[3]
    
    # シーカーを原点、東向き(0°)に固定。Zはデフォルト(0.0)で中心高度0.4m。
    s_id = env.body_ids['s']
    s_x_adr, s_y_adr, s_rot_adr = m.jnt_qposadr[env.qpos_indices['s']['x']], m.jnt_qposadr[env.qpos_indices['s']['y']], m.jnt_qposadr[env.qpos_indices['s']['rot']]
    d.qpos[s_x_adr] = 0.0; d.qpos[s_y_adr] = 0.0; d.qpos[s_rot_adr] = 0.0
    
    mujoco.mj_forward(m, d)

    # 4. 理論的な正解マップ
    ideal_results = {
        0:  ("wall_e", 6.00),      # 東の外壁
        6:  ("hider1_btm", H1_IDEAL_DIST), # エージェント表面 (3.84m)
        7:  ("maze_w3", 1.50),     # 北の内壁
        8:  ("maze_w2", 1.50),     # 南の内壁
        9:  ("wall_n", 8.50),      # 北の外壁
        10: ("maze_w1", 2.12),     # 内壁の角
        11: ("wall_w", 6.00),      # 西の外壁
    }

    print(f"🧪 Audit: Native(M0) vs Theory | Engine Height={engine.lidar_height}m")
    print("-" * 165)
    header = f" [Idx] | RelAng | {'Hit Target (Native)':<20} | Native(M0) | Geom(M1) | {'Ideal Target':<15} | Ideal Dist | Diff(M0-Ideal)"
    print(header)
    print("-" * 165)
    
    # 計測
    l0, hit_names = engine.cast_lidar(np.array([0.0, 0.0]), heading=0.0, mode=0, body_exclude=s_id, return_names=True)
    l1 = engine.cast_lidar(np.array([0.0, 0.0]), heading=0.0, mode=1, body_exclude=s_id)
    
    for i in range(12):
        angle_val = int(engine.base_angles[i])
        target_native = hit_names[i]
        dist_m0 = l0[i]
        dist_m1 = l1[i]
        
        ideal_target, ideal_dist = ideal_results.get(i, ("-", 0.0))
        
        # M0 (物理エンジン) の理想値からのズレを算出
        diff_m0_ideal = abs(dist_m0 - ideal_dist) if ideal_dist > 0 else 0.0
        mark = "!!" if diff_m0_ideal > 0.05 and ideal_dist > 0 else ""
        
        ideal_dist_str = f"{ideal_dist:8.2f}m" if ideal_dist > 0 else f"{'-':>9}"
        diff_str = f"{diff_m0_ideal:10.3f}m" if ideal_dist > 0 else f"{'-':>11}"

        print(f"  {i:2d}   | {angle_val:4d}°  | {target_native:<20} | {dist_m0:8.2f}m | {dist_m1:8.2f}m | {ideal_target:<15} | {ideal_dist_str} | {diff_str} {mark}")
    
    print("-" * 165)
    rmse = np.sqrt(np.mean((l0 - l1)**2))
    print(f"📊 Audit Result: RMSE(M0 vs M1) = {rmse:.4f}m")
    print(f"💡 Fix: Aligned Hider's Z-joint to 0.0 to match seeker's scan plane (0.4m).")

if __name__ == "__main__":
    run_alignment_audit()