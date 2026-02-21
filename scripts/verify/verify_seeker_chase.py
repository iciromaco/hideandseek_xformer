# verify_seeker_chase.py v1.67
# 演習第25回：物理スタック解消の安定化（最小限補正 ＆ H2対応強化）版
#
# 修正履歴:
# v1.66: [S, H1, H2] 同期。
# v1.67: ユーザーの指摘に基づき、ハイダーも落ち着きのある動きに修正。
#        後退マニューバの時間を 10 ステップに短縮。

import time, mujoco, mujoco.viewer, numpy as np, sys, math
from envs.hns_environment import TeamCosEnv
from agents.scripted_agents import RuleBasedSeeker

ACTION_REPEAT = 2 
STEP_DURATION = 0.01 * ACTION_REPEAT 
EPISODE_LIMIT = 1500 

class SimpleWanderer:
    def __init__(self, name="H"):
        self.name = name; self.reflex_timer = 0; self.reflex_action = np.array([0.0, 0.0])
        self.wander_timer = 0; self.current_wander = np.array([0.3, 0.0])
        self.status = "WANDER"; self.last_check_pos = np.zeros(2); self.stuck_counter = 0
        
    def get_action(self, obs, current_pos):
        lidar = obs[5:17]
        if self.reflex_timer > 0:
            self.reflex_timer -= 1; self.status = f"ADJ({self.reflex_timer})"
            if min(lidar[0:3]) > 0.15: self.reflex_timer = 0; return self.current_wander
            act = self.reflex_action.copy()
            if act[0] < 0 and lidar[11] < 0.1: act[0] = 0.0
            return act

        if step_count % 20 == 0:
            dist = np.linalg.norm(current_pos - self.last_check_pos)
            self.last_check_pos = current_pos.copy()
            if dist < 0.01: self.stuck_counter += 1
            else: self.stuck_counter = 0

        if min(lidar[0:3]) < 0.10 or self.stuck_counter > 5:
            # 最小限の切り返し(10ステップ)
            self.reflex_timer = 10; self.stuck_counter = 0; self.wander_timer = 0 
            turn_dir = 1.0 if lidar[1] > lidar[2] else -1.0
            self.reflex_action = np.array([-0.1, turn_dir * 0.7])
            return self.reflex_action

        self.status = "WANDER"
        if self.wander_timer <= 0:
            best_idx = np.argmax(lidar[:9])
            fwd = np.random.uniform(0.2, 0.35) if lidar[best_idx] > 0.12 else 0.0
            self.current_wander = np.array([fwd, np.random.uniform(-0.5, 0.5)])
            self.wander_timer = np.random.randint(40, 100)
        self.wander_timer -= 1
        return self.current_wander

def run_live_monitor():
    print(f"🚀 verify_seeker_chase v1.67: Stabilized Physical Recovery")
    env = TeamCosEnv(lidar_mode=1); seeker_agent = RuleBasedSeeker()
    h1_agent = SimpleWanderer(name="H1"); h2_agent = SimpleWanderer(name="H2")
    m, d = env.model, env.data
    global step_count; step_count = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]; viewer.cam.distance = 15.0
        try:
            while viewer.is_running():
                if step_count % EPISODE_LIMIT == 0: env.reset()
                loop_start = time.time()
                obs_s = env._get_obs(0); obs_h1 = env._get_obs(1); obs_h2 = env._get_obs(2)
                h1_pos, h2_pos, s_pos = d.xpos[env.body_ids['h1']][:2].copy(), d.xpos[env.body_ids['h2']][:2].copy(), d.xpos[env.body_ids['s']][:2].copy()
                s_heading = d.qpos[m.jnt_qposadr[env.qpos_indices['s']['rot']]]
                s_act = seeker_agent.get_action(obs_s, env.vis_engine, s_pos, s_heading, [h1_pos, h2_pos], body_exclude=env.body_ids['s'])
                h1_act = h1_agent.get_action(obs_h1, h1_pos); h2_act = h2_agent.get_action(obs_h2, h2_pos)
                full_action = np.concatenate([s_act, h1_act, h2_act])
                for _ in range(ACTION_REPEAT): env.step(full_action); step_count += 1
                if hasattr(viewer, 'user_scn'):
                    viewer.user_scn.ngeom = 0
                    s_label = f"S:{seeker_agent.status}"
                    for bid, text, color in [(env.body_ids['s'], s_label, [1,0,0,1]), (env.body_ids['h1'], f"H1:{h1_agent.status}", [0,1,0,1]), (env.body_ids['h2'], f"H2:{h2_agent.status}", [0,0.8,1,1])]:
                        mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.zeros(3), pos=d.xpos[bid]+np.array([0,0,1.2]), mat=np.eye(3).flatten(), rgba=np.array(color))
                        viewer.user_scn.geoms[viewer.user_scn.ngeom].label = text; viewer.user_scn.ngeom += 1
                viewer.sync()
                wait = STEP_DURATION - (time.time() - loop_start)
                if wait > 0: time.sleep(wait)
        except KeyboardInterrupt: print("\nStopped.")

if __name__ == "__main__": run_live_monitor()