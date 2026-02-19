# verify_seeker_rule.py
# 演習第25回：本番アクチュエータ構成・スタック回避・同時駆動検証

import mujoco
import mujoco.viewer
import numpy as np
import time
from hns_environment import TeamCosEnv

class RuleBasedSeeker:
    def __init__(self):
        self.state = "EXPLORE"
        self.stuck_timer = 0
        self.reverse_timer = 0

    def get_action(self, obs):
        speed = np.linalg.norm(obs[0:2])
        lidar = obs[5:17]
        h1_vis, h2_vis = obs[44] > 0.5, obs[51] > 0.5
        target_rel = obs[40:42] if h1_vis else (obs[45:47] if h2_vis else None)

        if target_rel is not None:
            self.state = "CHASE"
            angle = np.arctan2(target_rel[1], target_rel[0])
            return np.array([1.0, np.clip(angle * 2.0, -1.0, 1.0)])

        self.state = "EXPLORE"
        if speed < 0.1: self.stuck_timer += 1
        else: self.stuck_timer = 0

        if self.stuck_timer > 30 or self.reverse_timer > 0:
            if self.reverse_timer == 0: self.reverse_timer = 20
            self.reverse_timer -= 1
            return np.array([-0.6, 1.0])

        best_idx = np.argmax(lidar)
        angles = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
        target_angle = np.deg2rad(angles[best_idx])
        
        if lidar[0] < 1.5:
            turn = 1.0 if np.sum(lidar[[1,3,5]]) > np.sum(lidar[[2,4,6]]) else -1.0
            return np.array([0.2 if lidar[0] < 0.8 else 0.4, turn])
        
        return np.array([0.8, np.clip(target_angle * 1.5, -0.6, 0.6)])

def main():
    env = TeamCosEnv()
    seeker = RuleBasedSeeker()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            obs = env._get_obs(2)
            s_act = seeker.get_action(obs)
            full_ctrl = np.zeros(6)
            full_ctrl[0:2] = s_act
            full_ctrl[2:6] = np.random.uniform(-0.1, 0.1, 4) # Hiders 微動
            env.step(full_ctrl)
            viewer.sync()
            if int(env.data.time * 20) % 10 == 0:
                print(f"Time: {env.data.time:.1f}s | Mode: {seeker.state:8} | Ctrl: {s_act}")
            time.sleep(max(0, 0.05 - (time.time() - t0)))
            if env.data.time > 40.0: env.reset()

if __name__ == "__main__":
    main()