# verify_seeker_chase.py v1.62
# 演習第25回：Lidar Mode 1 (Geometric) 強制・垂直スタック解消版
#
# 修正履歴:
# v1.61: 進捗監視型スタック解消。
# v1.62: ユーザーの提案に基づき、不確実な Mode 0 (Native) を廃止し、
#        数学的に厳密な Mode 1 (Geometric) で全エージェントを駆動。
#        壁に垂直に止まる現象を Mode 1 の正確な距離測定により検知・打破する。

import time
import mujoco
import mujoco.viewer
import numpy as np
import sys
import math
from hns_environment import TeamCosEnv
from scripted_agents import RuleBasedSeeker

ACTION_REPEAT = 2 
STEP_DURATION = 0.01 * ACTION_REPEAT 
EPISODE_LIMIT = 1500 

class SimpleWanderer:
    """Mode 1 の正確な距離を前提とした、壁を怖がらないランダムエージェント"""
    def __init__(self, name="H"):
        self.name = name
        self.reflex_timer = 0
        self.reflex_action = np.array([0.0, 0.0])
        self.wander_timer = 0
        self.current_wander = np.array([0.3, 0.0])
        self.status = "WANDER"
        self.last_check_pos = np.zeros(2)
        self.stuck_counter = 0
        
    def get_action(self, obs, current_pos):
        lidar = obs[5:17] # すでに半径分を引いた自由空間(自由距離)
        
        # 1. リフレックスモード（切り返し中）
        if self.reflex_timer > 0:
            self.reflex_timer -= 1
            self.status = f"MANV({self.reflex_timer})"
            # 正面(Mode1なら正確)が開ければ復帰
            if min(lidar[0], lidar[1], lidar[2]) > 0.4:
                self.reflex_timer = 0
                return self.current_wander
            return self.reflex_action

        # 2. スタック判定（Mode 1 なら壁に密着しても Lidar が 0 に固定されない）
        # 自由空間が 0.05m(実距離0.5m) 未満、または進捗がない場合
        is_front_crash = min(lidar[0], lidar[1], lidar[2]) < 0.05
        
        if step_count % 15 == 0:
            dist = np.linalg.norm(current_pos - self.last_check_pos)
            self.last_check_pos = current_pos.copy()
            if dist < 0.01: self.stuck_counter += 1
            else: self.stuck_counter = 0

        if is_front_crash or self.stuck_counter > 3:
            self.reflex_timer = 40
            self.stuck_counter = 0
            # 垂直スタック対策：左右どちらかより開いている方へ大きく舵を切る
            turn_dir = 1.0 if lidar[1] > lidar[2] else -1.0
            # 少し強めの後退と旋回
            self.reflex_action = np.array([-0.4, turn_dir * 0.8])
            return self.reflex_action

        # 3. 通常の放浪
        self.status = "WANDER"
        if self.wander_timer <= 0:
            self.current_wander = np.array([np.random.uniform(0.25, 0.4), np.random.uniform(-0.4, 0.4)])
            self.wander_timer = np.random.randint(40, 100)
        self.wander_timer -= 1
        return self.current_wander

def run_live_monitor():
    # 明示的に lidar_mode=1 を指定
    print(f"🚀 verify_seeker_chase v1.62: Mode 1 (Geometric) Only Strategy")
    env = TeamCosEnv(lidar_mode=1)
    seeker_agent = RuleBasedSeeker()
    h1_agent = SimpleWanderer(name="H1")
    h2_agent = SimpleWanderer(name="H2")
    m, d = env.model, env.data
    global step_count
    step_count = 0
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]; viewer.cam.distance = 15.0
        try:
            while viewer.is_running():
                if step_count % EPISODE_LIMIT == 0: env.reset()
                loop_start = time.time()
                
                # 観測取得 (Mode 1 が適用される)
                obs_s = env._get_obs(2); obs_h1 = env._get_obs(0); obs_h2 = env._get_obs(1)
                h1_pos, h2_pos, s_pos = d.xpos[env.body_ids['h1']][:2].copy(), d.xpos[env.body_ids['h2']][:2].copy(), d.xpos[env.body_ids['s']][:2].copy()

                # --- 物理補正 (Nudge) ---
                # Mode 1 でも物理エンジン側の癒着は起こりうるため Nudge は維持
                for idx, p in enumerate(["h1", "h2", "s"]):
                    obs = [obs_h1, obs_h2, obs_s][idx]
                    if np.linalg.norm(obs[0:2]) < 0.015 and (obs[5] < 0.05 or obs[16] < 0.05):
                        q_x, q_y = m.jnt_qposadr[env.qpos_indices[p]['x']], m.jnt_qposadr[env.qpos_indices[p]['y']]
                        ang = obs[2] + (np.pi if obs[5] < 0.05 else 0)
                        d.qpos[q_x] -= math.cos(ang) * 0.04; d.qpos[q_y] -= math.sin(ang) * 0.04
                        mujoco.mj_forward(m, d)

                s_heading = d.qpos[m.jnt_qposadr[env.qpos_indices['s']['rot']]]
                s_act = seeker_agent.get_action(obs_s, env.vis_engine, s_pos, s_heading, [h1_pos, h2_pos], body_exclude=env.body_ids['s'])
                h1_act = h1_agent.get_action(obs_h1, h1_pos); h2_act = h2_agent.get_action(obs_h2, h2_pos)
                
                full_action = np.concatenate([h1_act, h2_act, s_act])
                for _ in range(ACTION_REPEAT):
                    env.step(full_action); step_count += 1
                
                if hasattr(viewer, 'user_scn'):
                    viewer.user_scn.ngeom = 0
                    s_label = f"S:{seeker_agent.status}"
                    for bid, text, color in [(env.body_ids['h1'], f"H1:{h1_agent.status}", [0,1,0,1]), 
                                             (env.body_ids['h2'], f"H2:{h2_agent.status}", [0,0.8,1,1]), 
                                             (env.body_ids['s'], s_label, [1,0,0,1])]:
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.zeros(3), pos=d.xpos[bid]+np.array([0,0,1.2]), mat=np.eye(3).flatten(), rgba=np.array(color))
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].label = text; viewer.user_scn.ngeom += 1

                viewer.sync()
                wait = STEP_DURATION - (time.time() - loop_start)
                if wait > 0: time.sleep(wait)
        except KeyboardInterrupt: 
            print("\nStopped.")

if __name__ == "__main__": 
    run_live_monitor()