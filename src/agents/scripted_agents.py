# scripted_agents.py v2.12
# 演習第26回：NPC道具操作（ランプ・箱） ＆ v1.70機動性統合版
# 
# 修正内容:
# 1. ランプ操作知能: シーカーがランプ（スロープ）を見つけたら Grab してハイダーへ運ぶ基本戦略を追加。
# 2. ロック sabotoge: 箱が近くにあれば Lock して妨害。
# 3. 4次元出力: [fwd, turn, lock, grab] を正確に出力。

import numpy as np
import math

class RuleBasedSeeker:
    """高度な戦略的シーカー。スロープ搬送と箱のロック妨害を行う。"""
    def __init__(self):
        self.last_target_pos = None
        self.target_lost_timer = 0
        self.reflex_timer = 0
        self.brake_timer = 0
        self.cooldown = 0
        self.reflex_action = np.array([0.0, 0.0])
        self.status = "IDLE"
        self.last_check_pos = np.zeros(2)
        self.stuck_counter = 0
        
    def get_action(self, obs, engine, current_pos, current_heading, target_positions, body_exclude, target_body_ids):
        lidar = obs[5:17] 
        
        # 物理制御
        if self.brake_timer > 0:
            self.brake_timer -= 1; self.status = "BRAKE"; return np.array([0.0, 0.0, 0.0, 0.0])
        if self.reflex_timer > 0:
            self.reflex_timer -= 1; self.status = "ADJUST"
            if np.min(lidar[0:3]) > 0.6: self.reflex_timer = 0; self.cooldown = 10; return np.array([0.1, 0.0, 0.0, 0.0])
            return np.array([self.reflex_action[0], self.reflex_action[1], 0.0, 0.0])
        
        if self.cooldown > 0: self.cooldown -= 1
        
        # スタック検知
        dist_moved = np.linalg.norm(current_pos - self.last_check_pos)
        self.last_check_pos = current_pos.copy()
        if dist_moved < 0.005: self.stuck_counter += 1
        else: self.stuck_counter = 0

        # ターゲット探索
        best_target = None; min_dist = float('inf')
        for i, t_pos in enumerate(target_positions):
            visible = engine.is_visible(current_pos, t_pos, body_exclude=body_exclude, target_body_id=target_body_ids[i])
            dist = np.linalg.norm(t_pos - current_pos)
            if visible and dist < min_dist: min_dist = dist; best_target = t_pos

        # 回避判定
        front_min = np.min(lidar[0:3])
        is_enemy_in_front = (best_target is not None and front_min > (min_dist - 0.9))

        if (front_min < 0.22 and not is_enemy_in_front) or self.stuck_counter > 45:
            self.status = "ADJUST"; self.brake_timer = 4; self.reflex_timer = 15
            best_idx = np.argmax(lidar)
            target_deg = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180][best_idx]
            if abs(target_deg) >= 90:
                self.reflex_action = np.array([-0.08, (1.0 if target_deg > 0 else -1.0) * 0.6])
            else:
                self.reflex_action = np.array([0.0, np.clip(np.deg2rad(target_deg) * 2.5, -0.7, 0.7)])
            self.stuck_counter = 0; return np.array([0.0, 0.0, 0.0, 0.0])

        # 💡 [道具操作知能]
        lock_cmd, grab_cmd = 0.0, 0.0
        # 1. 箱の妨害
        for base in [17, 25]:
            if obs[base + 7] > 0.5: # Visible
                d = np.linalg.norm(obs[base : base+2])
                if d < 1.3 and obs[base + 6] < 0.5: # 近くにあって未ロックならロックする
                    lock_cmd = 1.0; self.status = "SABOTAGE"
        
        # 2. ランプの利用
        ramp_rel = obs[33:35]
        if obs[39] > 0.5 and np.linalg.norm(ramp_rel) < 1.4: # ランプが見えていて近い
            grab_cmd = 1.0; self.status = "RAMP_CARRY"

        # 基本移動
        res_act = np.zeros(2)
        if best_target is not None:
            self.status = "CHASE"; self.last_target_pos = best_target.copy(); self.target_lost_timer = 0
            res_act = self._compute_pursuit(current_pos, current_heading, best_target)
        else:
            self.status = "EXPLR"; res_act = self._compute_explore(obs)

        return np.array([res_act[0], res_act[1], lock_cmd, grab_cmd])

    def _compute_pursuit(self, pos, heading, target, speed_limit=0.35):
        diff = target - pos; dist = np.linalg.norm(diff)
        err = (math.atan2(diff[1], diff[0]) - heading + math.pi) % (2 * math.pi) - math.pi
        ideal = 0.90 
        fwd = np.clip((dist - ideal) * 0.8, 0.0, speed_limit) if dist > ideal else 0.0
        if abs(err) > np.deg2rad(20): fwd *= 0.1
        return np.array([fwd, np.clip(err * 3.0, -0.9, 0.9)])

    def _compute_explore(self, obs):
        lidar = obs[5:17]; scores = [min(lidar[i], 3.0) - abs([0,15,-15,30,-30,45,-45,90,-90,135,-135,180][i]) * 0.015 for i in range(12)]
        best_idx = np.argmax(np.array(scores))
        fwd = 0.15 if lidar[best_idx] > 0.5 else 0.05
        turn = np.clip(np.deg2rad([0,15,-15,30,-30,45,-45,90,-90,135,-135,180][best_idx]) * 1.5, -0.6, 0.6)
        return np.array([fwd, turn])

class RuleBasedHider:
    def __init__(self):
        self.status = "WANDER"; self.reflex_timer = 0; self.grab_timer = 0
    def get_action(self, obs, pos, heading, seeker_pos):
        lidar = obs[5:17]; front = min(lidar[0:3])
        if self.reflex_timer > 0:
            self.reflex_timer -= 1; return np.array([self.reflex_action[0], self.reflex_action[1], 0.0, 0.0])
        if front < 0.25:
            self.reflex_timer = 15; best_idx = np.argmax(lidar); ang = [0,15,-15,30,-30,45,-45,90,-90,135,-135,180][best_idx]
            self.reflex_action = np.array([-0.05, np.clip(np.deg2rad(ang), -0.6, 0.6)])
            return np.array([self.reflex_action[0], self.reflex_action[1], 0.0, 0.0])
        
        lock_cmd, grab_cmd = 0.0, 0.0
        for base in [17, 25]:
            if obs[base + 7] > 0.5 and np.linalg.norm(obs[base:base+2]) < 1.4:
                if obs[base + 6] < 0.5:
                    grab_cmd = 1.0; self.grab_timer += 1
                    if self.grab_timer > 100: lock_cmd = 1.0; grab_cmd = 0.0
        
        diff = seeker_pos - pos; dist = np.linalg.norm(diff)
        err = (math.atan2(-diff[1], -diff[0]) - heading + math.pi) % (2 * math.pi) - math.pi
        fwd = 0.35 if dist < 4.0 else 0.15
        return np.array([fwd, np.clip(err * 1.8, -0.6, 0.6), lock_cmd, grab_cmd])