# verify_seeker_chase.py v1.23
# 演習第25回：シーカー物理追跡・全エージェント駆動確認版
# 
# 修正履歴:
# v1.22: アクチュエータの並び順（0-1: Red, 2-3: Blue, 4-5: Cyan）を修正。
# v1.23: 起動時にアクチュエータ名を表示してインデックスを確認。
#        Hiderの動きを「その場」ではなく「一定方向への直進」に変更して視認性を向上。

import time
import mujoco
import mujoco.viewer
import numpy as np
import math
import sys
from hns_environment import TeamCosEnv
from scripted_agents import RuleBasedSeeker

# 意思決定1回につき繰り返す物理ステップ
ACTION_REPEAT = 16

def run_live_monitor():
    print(f"🚀 verify_seeker_chase v1.23: アクチュエータ順序検証モード")
    
    env = TeamCosEnv()
    seeker_agent = RuleBasedSeeker()
    m, d = env.model, env.data
    
    # --- アクチュエータ名の確認 (インデックスの真実を暴く) ---
    print("\n" + "-"*50)
    print("📋 Actuator Index Map:")
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  Index {i}: {name}")
    print("-"*50 + "\n")
    
    s_id = env.body_ids['s']
    h1_id = env.body_ids['h1']
    h2_id = env.body_ids['h2']
    h_ids = [h1_id, h2_id]
    
    h1_act = np.zeros(2)
    h2_act = np.zeros(2)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 12.0
        
        step_count = 0
        try:
            while viewer.is_running():
                step_start = time.time()
                
                # 状態取得
                s_pos_3d = d.xpos[s_id].copy()
                s_pos = s_pos_3d[:2]
                s_rot_idx = m.jnt_qposadr[env.qpos_indices['s']['rot']]
                s_heading = np.deg2rad(d.qpos[s_rot_idx])
                h_positions = [d.xpos[h1_id][:2].copy(), d.xpos[h2_id][:2].copy()]
                obs_s = env._get_obs(2) 
                
                # 意思決定
                s_action = seeker_agent.get_action(obs_s, env.vis_engine, s_pos, s_heading, h_positions, body_exclude=s_id)
                
                # ハイダーの行動 (ランダムだが一定時間直進させることで浮遊を回避)
                if step_count % 32 == 0:
                    # 前進(0.15〜0.25)を基本にし、旋回を混ぜる
                    h1_act = np.array([np.random.uniform(0.15, 0.25), np.random.uniform(-0.1, 0.1)])
                if (step_count + 16) % 32 == 0:
                    h2_act = np.array([np.random.uniform(0.15, 0.25), np.random.uniform(-0.1, 0.1)])
                
                # アクチュエータ記述順 (Seeker, Hider1, Hider2) に基づく代入
                ctrl = np.zeros(6)
                ctrl[0:2] = s_action    # Red
                ctrl[2:4] = h1_act      # Blue
                ctrl[4:6] = h2_act      # Cyan
                d.ctrl[:] = ctrl
                
                # 物理ステップ実行
                for _ in range(ACTION_REPEAT):
                    mujoco.mj_step(m, d)
                
                # 視界の線とラベルの描画
                visible_targets = []
                if hasattr(viewer, 'add_marker'):
                    viewer.add_marker(pos=s_pos_3d + [0, 0, 1.5], label=seeker_agent.debug_text, rgba=[1, 1, 1, 1], type=mujoco.mjtGeom.mjGEOM_NONE)
                    for i, hid in enumerate(h_ids):
                        h_pos_3d = d.xpos[hid].copy()
                        visible, dist, _ = seeker_agent._check_visibility(env.vis_engine, s_pos, s_heading, h_pos_3d[:2], s_id)
                        if visible:
                            visible_targets.append(f"H{i+1}")
                            # 視界を表す赤い線
                            mid = (s_pos_3d + h_pos_3d) / 2
                            # 方向行列の計算
                            diff = h_pos_3d - s_pos_3d
                            z_axis = diff / (dist + 1e-8)
                            x_axis = np.array([0, 0, 1])
                            y_axis = np.cross(z_axis, x_axis); y_axis /= (np.linalg.norm(y_axis) + 1e-8)
                            x_axis = np.cross(y_axis, z_axis)
                            mat = np.stack([x_axis, y_axis, z_axis], axis=1)
                            
                            viewer.add_marker(pos=s_pos_3d + diff/2, size=[0.01, 0.01, dist/2], mat=mat, type=mujoco.mjtGeom.mjGEOM_LINE, rgba=[1, 0, 0, 1])
                            viewer.add_marker(pos=h_pos_3d + [0,0,1.2], label=f"SAW H{i+1}", rgba=[1,0,0,1], type=mujoco.mjtGeom.mjGEOM_NONE)

                # ログ
                seen_str = ",".join(visible_targets) if visible_targets else "None"
                sys.stdout.write(f"\r[Stp:{step_count:4d}] {seeker_agent.debug_text:30s} | Saw:[{seen_str:5s}] | Act:[{s_action[0]:.1f},{s_action[1]:.1f}]    ")
                sys.stdout.flush()

                viewer.sync()
                step_count += 1
                time.sleep(max(0, (ACTION_REPEAT * 0.002) - (time.time() - step_start)))
                
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    run_live_monitor()