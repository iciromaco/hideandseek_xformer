# verify_seeker_chase.py v1.7
# 修正点：ACTION_REPEAT=16 の導入によりエージェントを物理的に移動させる。
#        表示の簡略化のためステータス表示幅を短縮。

import time
import mujoco
import mujoco.viewer
import numpy as np
import math
from hns_environment import TeamCosEnv
from scripted_agents import RuleBasedSeeker

# 物理定数：1回の意思決定を何ステップ繰り返すか
ACTION_REPEAT = 160

def run_live_monitor():
    print(f"🚀 verify_seeker_chase v1.7: 監査開始 (ACTION_REPEAT={ACTION_REPEAT}, ログ整理版)")
    env = TeamCosEnv()
    seeker_agent = RuleBasedSeeker()
    m, d = env.model, env.data
    
    s_id = env.body_ids['s']
    h_ids = [env.body_ids['h1'], env.body_ids['h2']]
    
    # ハイダーの行動維持用
    h_actions = np.zeros(4)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 12.0
        
        step_count = 0
        while viewer.is_running():
            step_start = time.time()
            
            # --- 意思決定フェーズ ---
            s_pos_3d = d.xpos[s_id].copy()
            s_pos = s_pos_3d[:2]
            s_rot_idx = m.jnt_qposadr[env.qpos_indices['s']['rot']]
            s_heading = np.deg2rad(d.qpos[s_rot_idx])
            h_positions = [d.xpos[hid][:2].copy() for hid in h_ids]
            
            # Seekerの意思決定
            obs_s = env._get_obs(2) 
            s_action = seeker_agent.get_action(obs_s, env.vis_engine, s_pos, s_heading, h_positions, body_exclude=s_id)
            
            # Hiderの意思決定 (ACTION_REPEAT ごとにランダム変更)
            if step_count % (ACTION_REPEAT * 4) == 0:
                h_actions = np.random.uniform(-0.3, 0.3, 4) 
            
            # --- 物理実行フェーズ (ACTION_REPEAT分繰り返す) ---
            ctrl_input = np.zeros(6)
            ctrl_input[0:4] = h_actions
            ctrl_input[4:6] = s_action
            d.ctrl[:] = ctrl_input
            
            # ACTION_REPEAT 回ステップを進めることで、慣性が乗り「浮遊」が解消される
            mujoco.mj_step(m, d, ACTION_REPEAT)
            
            # --- 視覚化とログ ---
            for hid in h_ids:
                h_pos_3d = d.xpos[hid].copy()
                visible, dist, _ = seeker_agent._check_visibility(env.vis_engine, s_pos, s_heading, h_pos_3d[:2], s_id)
                if visible:
                    diff = h_pos_3d - s_pos_3d
                    z_axis = diff / (dist + 1e-8)
                    x_axis = np.array([0, 0, 1])
                    y_axis = np.cross(z_axis, x_axis); y_axis /= (np.linalg.norm(y_axis) + 1e-8)
                    x_axis = np.cross(y_axis, z_axis)
                    mat = np.stack([x_axis, y_axis, z_axis], axis=1)
                    viewer.add_marker(pos=s_pos_3d + (diff/2), size=[0.01, 0.01, dist/2], mat=mat, type=mujoco.mjtGeom.mjGEOM_LINE, rgba=[1, 0, 0, 0.8])
                    viewer.add_marker(pos=h_pos_3d + np.array([0, 0, 1.2]), label="VISIBLE", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0.05, 0.05], rgba=[1, 0, 0, 0])

            # ターミナル出力の整理 (桁数を絞り視認性向上。ステータス幅を5に短縮)
            if step_count % 10 == 0:
                h1_dist = np.linalg.norm(s_pos - h_positions[0])
                print(f"Step:{step_count:4} | {seeker_agent.status:5} | Dist:{h1_dist:4.1} | Act:[{s_action[0]:.1}, {s_action[1]:.1}]")

            viewer.sync()
            step_count += 1
            
            # 実時間同期
            elapsed = time.time() - step_start
            if elapsed < (ACTION_REPEAT * 0.002): # MuJoCoデフォルト 2ms * 16
                time.sleep((ACTION_REPEAT * 0.002) - elapsed)

if __name__ == "__main__":
    run_live_monitor()