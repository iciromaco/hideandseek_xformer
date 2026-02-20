# test_lidar_validation_v3.py v3.1
# 演習第25回：特定部品限定フィルタリング（v1.43）後の3モード整合性・デバッグ監査
# 
# 修正点:
# 1. デバッグ情報の強化: Nativeモードで衝突した geom 名を全レイについて出力。
# 2. 距離フィルタの明示化: 0.45m未満のヒットがなぜ発生しているかを特定。

import numpy as np
import mujoco
import time
from hns_environment import TeamCosEnv

def run_lidar_triple_audit():
    print("="*100)
    print("🚀 LIDAR TRIPLE-MODE LOGIC & PERFORMANCE AUDIT v3.1")
    print("   Goal: Eliminate SELF-HIT in Native Mode & Sync Accuracy")
    print("="*100 + "\n")

    env = TeamCosEnv(lidar_mode=1)
    engine = env.vis_engine
    m, d = env.model, env.data
    seeker_body_id = env.body_ids['s']

    scenarios = [
        {"name": "Open Field  ", "s_pos": [0.0, 0.0], "h1_pos": [10.0, 10.0]},
        {"name": "Wall Proximity", "s_pos": [3.0, 1.3], "h1_pos": [10.0, 10.0]},
        {"name": "Agent Chase  ", "s_pos": [0.0, 0.0], "h1_pos": [1.1, 0.0]},
    ]

    modes = [0, 1, 2]
    mode_names = ["Native (M0)", "Geometric(M1)", "Sphere (M2)"]

    for sc in scenarios:
        print(f"📍 Scenario: {sc['name']} at {sc['s_pos']}")
        d.qpos[m.jnt_qposadr[env.qpos_indices['s']['x']]] = sc['s_pos'][0]
        d.qpos[m.jnt_qposadr[env.qpos_indices['s']['y']]] = sc['s_pos'][1]
        d.qpos[m.jnt_qposadr[env.qpos_indices['s']['z']]] = 0.0 # 地面
        d.qpos[m.jnt_qposadr[env.qpos_indices['h1']['x']]] = sc['h1_pos'][0]
        d.qpos[m.jnt_qposadr[env.qpos_indices['h1']['y']]] = sc['h1_pos'][1]
        mujoco.mj_forward(m, d)

        results = {}
        for mode in modes:
            results[mode] = engine.cast_lidar(np.array(sc['s_pos']), heading=0.0, mode=mode, body_exclude=seeker_body_id)

        # Native モードのヒット詳細デバッグ
        hit_geom_names = []
        for i in range(12):
            vx, vy = engine.base_cos[i], engine.base_sin[i]
            p_orig = np.array([sc['s_pos'][0], sc['s_pos'][1], 0.4], dtype=np.float64)
            g_out = np.zeros(1, dtype=np.int32)
            mujoco.mj_ray(m, d, p_orig, np.array([vx, vy, 0.0]), None, 1, int(seeker_body_id), g_out)
            name = mujoco.mj_id2name(m, mujoco.mjtGeom.mjGEOM_GEOM, g_out[0]) if g_out[0] >= 0 else "None"
            hit_geom_names.append(name)

        # 表示
        for m_idx in modes:
            min_d = np.min(results[m_idx])
            sh_mark = "❌ SELF-HIT!" if min_d < 0.41 else "✅ Safe"
            print(f"   [{mode_names[m_idx]}] {sh_mark} | MinDist: {min_d:.4f}m")
            if min_d < 0.41 and m_idx == 0:
                idx = np.argmin(results[0])
                print(f"      🚨 Native Hit Details: Ray {idx} hit geom '{hit_geom_names[idx]}'")

        rmse_geom = np.sqrt(np.mean((results[0] - results[1])**2))
        print(f"   📊 Accuracy: RMSE(M1 vs M0) = {rmse_geom:.5f}m")
        if sc['name'].strip() == "Agent Chase":
            print(f"   🎯 Chase Dist (0°): Native={results[0][0]:.3f}m, Geom={results[1][0]:.3f}m (Target: hider1_btm)")
        print("-" * 60)

    print("\n🏁 Final Audit Result:")
    print("If MinDist is consistently > 0.45m in all modes (except possibly Sphere rounding), the logic is FIXED.")

if __name__ == "__main__":
    run_lidar_triple_audit()