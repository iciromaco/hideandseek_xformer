# scripted_agents.py v1.54
# 演習第25回：安定巡航・穏やかな衝突回避版
# 
# 修正履歴:
# v1.53: 自己衝突解消同期。
# v1.54: 環境側のLidarバグが完全に解消されたため、不必要なリフレックスを
#        誘発しないよう、衝突回避の最大出力をさらに抑え、
#        「詰まった場所からスッと逃げる」自然な挙動に回帰。

import numpy as np
import math

class RuleBasedSeeker:
    def __init__(self):
        self.last_target_pos = None
        self.target_lost_timer = 0
        self.reflex_timer = 0
        self.cooldown = 0
        self.reflex_action = np.array([0.0, 0.0])
        self.status = "IDLE"
        self.last_check_pos = np.zeros(2)
        self.stuck_counter = 0
        
    def get_action(self, obs, engine, current_pos, current_heading, target_positions, body_exclude):
        lidar = obs[5:17]
        is_chasing = any(engine.is_visible(current_pos, t, body_exclude=body_exclude) for t in target_positions)

        # 1. 動的リフレックス (切り返し)
        if self.reflex_timer > 0:
            self.reflex_timer -= 1
            if min(lidar[0:3]) > 0.20:
                self.reflex_timer = 0; self.cooldown = 15; return np.array([0.1, 0.0])
            act = self.reflex_action.copy()
            # 真後ろが近いときは後退を止めて旋回に専念
            if act[0] < 0 and lidar[11] < 0.1: act[0] = 0.0
            return act
        
        if self.cooldown > 0:
            self.cooldown -= 1; return self._compute_explore_action(obs)
        
        # 2. スタック判定
        dist_moved = np.linalg.norm(current_pos - self.last_check_pos)
        self.last_check_pos = current_pos.copy()
        if dist_moved < 0.01: self.stuck_counter += 1
        else: self.stuck_counter = 0
            
        # 自己衝突が消えたため、反射距離は 10cm 以下で十分
        reflex_dist = 0.05 if is_chasing else 0.10
        if min(lidar[0:3]) < reflex_dist or self.stuck_counter > 25:
            self.reflex_timer = 12; self.status = "ADJUST"; self.stuck_counter = 0
            best_idx = np.argmax(lidar)
            angles = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
            fwd = -0.05 if abs(angles[best_idx]) > 90 and lidar[11] > 0.2 else 0.08
            self.reflex_action = np.array([fwd, np.clip(np.deg2rad(angles[best_idx]) * 1.3, -0.6, 0.6)])
            return self.reflex_action

        # 3. ターゲット追跡
        best_target = None; min_dist = float('inf')
        for t_pos in target_positions:
            visible = engine.is_visible(current_pos, t_pos, body_exclude=body_exclude)
            dist = np.linalg.norm(t_pos - current_pos)
            if visible and dist < min_dist: min_dist = dist; best_target = t_pos

        if best_target is not None:
            self.status = "CHASE"; self.last_target_pos = best_target.copy(); self.target_lost_timer = 0
            return self._compute_pursuit_action(current_pos, current_heading, best_target, lidar)
        elif self.last_target_pos is not None and self.target_lost_timer < 60:
            self.status = "TRACK"; self.target_lost_timer += 1
            return self._compute_pursuit_action(current_pos, current_heading, self.last_target_pos, lidar)
        else:
            self.status = "EXPLR"; return self._compute_explore_action(obs)

    def _compute_pursuit_action(self, pos, heading, target, lidar):
        diff = target - pos; dist = np.linalg.norm(diff)
        target_angle = math.atan2(diff[1], diff[0])
        err = (target_angle - heading + math.pi) % (2 * math.pi) - math.pi
        # ターゲットに肉薄する
        fwd = 0.35 if dist > 0.55 else 0.0
        return np.array([fwd, np.clip(err * 2.0, -0.6, 0.6)])

    def _compute_explore_action(self, obs):
        lidar = obs[5:17]; best_idx = np.argmax(lidar[:9])
        angles = [0, 15, -15, 30, -30, 45, -45, 90, -90]
        # 狭い隙間でも積極的に狙う
        fwd = 0.3 if lidar[best_idx] > 0.12 else 0.0
        turn = np.deg2rad(angles[best_idx]) * 1.5
        return np.array([fwd, np.clip(turn, -0.6, 0.6)])