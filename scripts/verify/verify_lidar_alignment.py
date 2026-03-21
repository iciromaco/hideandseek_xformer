# verify_lidar_alignment.py v2.49
# 演習第25回：Lidar監査 決定版（パス解決 ＆ 自動解析完全同期版）
#
# 修正履歴:
# v2.48: mj_ray 引数是正。
# v2.49: KeyError: 's' を防ぐための安全なキーアクセスを導入。
#        解析によって検出された Seeker (env.seekers) が存在するかをチェックしてから実行。


import mujoco
import mujoco.viewer
import numpy as np

from envs.hns_environment import TeamCosEnv


def run_alignment_audit():
    print("=" * 120)
    print("🚀 Lidar Alignment Audit v2.49 (System Integrated Audit)")
    print("=" * 120)

    # 環境の初期化
    env = TeamCosEnv()
    m, d = env.model, env.data
    engine = env.vis_engine

    # 解析結果の確認
    if not env.seekers:
        print("❌ Error: Seeker agent ('s') could not be detected in the XML.")
        print("   Please check if main18_optimization.py has a valid XML_CONTENT.")
        return

    # 解析で最初に見つかったシーカーを使用
    s_key = env.seekers[0]
    s_id = env.body_ids[s_key]

    print(f"🎯 Using detected agent: {s_key} (Body ID: {s_id})")

    # 1. 干渉オブジェクトの退避
    for obj in ["box1", "box2", "ramp"]:
        try:
            jnt_id = m.joint(f"{obj}_joint").id
            q_adr = m.jnt_qposadr[jnt_id]
            d.qpos[q_adr : q_adr + 2] = [10.0, 10.0]
            d.qpos[q_adr + 2] = 0.5
        except:
            pass

    # 2. シーカーを原点、東向き(0°)に固定
    q_s = env.qpos_indices[s_key]
    d.qpos[m.jnt_qposadr[q_s["x"]]] = 0.0
    d.qpos[m.jnt_qposadr[q_s["y"]]] = 0.0
    d.qpos[m.jnt_qposadr[q_s["rot"]]] = 0.0

    # 他のハイダーを特定の位置へ
    planned = {"h1": [3.0, -3.0], "h2": [0.0, 4.0]}
    for h_key, pos in planned.items():
        if h_key in env.qpos_indices:
            q_h = env.qpos_indices[h_key]
            d.qpos[m.jnt_qposadr[q_h["x"]]] = pos[0]
            d.qpos[m.jnt_qposadr[q_h["y"]]] = pos[1]

    mujoco.mj_forward(m, d)

    print("\n🧪 Audit: Native(M0) vs Theory | Scan Height=0.4m")
    print("-" * 150)

    # 計測実行
    l1 = engine.cast_lidar(np.array([0.0, 0.0]), heading=0.0, mode=1, body_exclude=s_id)
    l2 = engine.cast_lidar(np.array([0.0, 0.0]), heading=0.0, mode=2, body_exclude=s_id)

    p_orig = np.array([0.0, 0.0, 0.4], dtype=np.float64)
    g_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    angles_rad = np.deg2rad([0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180])

    for i in range(12):
        vx, vy = np.cos(angles_rad[i]), np.sin(angles_rad[i])
        g_out = np.array([-1], dtype=np.int32)
        dist = mujoco.mj_ray(m, d, p_orig, np.array([vx, vy, 0.0]), g_mask, 1, int(s_id), g_out)

        dist_m0 = dist if dist > 0.41 else 15.0
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g_out[0]) if (g_out[0] >= 0 and dist > 0.41) else "None"

        print(f"  {i:2d} | {int(np.rad2deg(angles_rad[i])):4d}° | {name:<20} | M0:{dist_m0:6.2f}m | M1:{l1[i]:6.2f}m | M2:{l2[i]:6.2f}m")

    print("-" * 150)


if __name__ == "__main__":
    run_alignment_audit()
