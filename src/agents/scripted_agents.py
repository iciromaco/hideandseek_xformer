# src/agents/scripted_agents.py
# scripted_agents.py v5.4 (お見合い回避ロジックの導入)

import numpy as np
import math


class RuleBasedSeeker:
    """比例制御に加え、お見合いを回避するバイアスを持つ Seeker。"""

    def __init__(self):
        self.reflex_timer = 0
        self.patrol_step = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.is_grabbing = False
        self.last_known_rel_pos_x = 0.0
        self.last_known_rel_pos_y = 0.0
        self.memory_timer = 0
        self.last_l_sum = 0.0
        self.stuck_counter = 0
        self.current_state = "Idle"

    def get_action(self, obs, idx):
        self.patrol_step += 1
        lidar = obs[idx.LIDAR]
        front_min = np.min(lidar[0:3])

        r_gap = lidar[1] + lidar[3] + lidar[5]
        l_gap = lidar[2] + lidar[4] + lidar[6]
        side_diff = r_gap - l_gap

        # --- 比例制御とデッドロック回避 ---
        speed_scale = np.clip((front_min - 0.2) / 0.8, 0.0, 1.0)
        # 正面に障害物がある場合、右方向（または左）への微小バイアスを付与
        # これにより、完全に左右対称な状況でもお見合いが解消されます
        omiai_bias = 0.25 * (1.0 - speed_scale) if front_min < 0.6 else 0.0
        
        avoid_w = np.clip(1.0 - (front_min / 1.5), 0.0, 1.0)
        avoid_torque = (side_diff * 5.0) + omiai_bias

        curr_l_sum = np.sum(lidar)
        if abs(curr_l_sum - self.last_l_sum) < 0.0005:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_l_sum = curr_l_sum

        visible_enemies = [en for en in idx.OTHERS if obs[en.VISIBLE] > 0.5]
        enemy_seen = len(visible_enemies) > 0

        fwd, trn, lck, grb = 0.0, 0.0, 0.0, 0.0

        if self.reflex_timer > 0:
            self.current_state = "Reflex"
            self.reflex_timer -= 1
            fwd, trn = -0.4, 0.8
        elif front_min < 0.15:
            self.current_state = "AvoidWall"
            self.reflex_timer = 10
        elif self.stuck_counter > 45:
            self.current_state = "StuckEscape"
            self.reflex_timer = 20
            self.stuck_counter = 0
        elif enemy_seen:
            self.current_state = "Chasing"
            self.memory_timer = 150
            target_en = visible_enemies[0]
            tx, ty = obs[target_en.REL_X], obs[target_en.REL_Y]
            self.last_known_rel_pos_x, self.last_known_rel_pos_y = tx, ty
            fwd = 0.65 * speed_scale
            # 回避バイアスを含めて目標角度を計算
            raw_trn = math.atan2(ty, tx) * 3.5
            trn = np.clip(raw_trn + avoid_torque * avoid_w, -0.9, 0.9)
        elif self.memory_timer > 0:
            self.current_state = "MemorySearch"
            self.memory_timer -= 1
            mx, my = self.last_known_rel_pos_x, self.last_known_rel_pos_y
            fwd = 0.45 * speed_scale
            raw_trn = math.atan2(my, mx) * 2.8
            trn = np.clip(raw_trn + avoid_torque * avoid_w, -0.7, 0.7)
        else:
            self.current_state = "Patrol"
            self.wander_timer -= 1
            if self.wander_timer <= 0:
                self.wander_angle = np.random.uniform(-np.pi, np.pi)
                self.wander_timer = np.random.randint(100, 250)
            a_err = (self.wander_angle - obs[idx.SELF.ROT] + np.pi) % (2*np.pi)-np.pi
            fwd = 0.4 * speed_scale
            trn = np.clip(a_err * 2.5 + avoid_torque * avoid_w, -0.7, 0.7)

        return np.array([fwd, trn, lck, grb])


class RuleBasedHider:
    """お見合いを回避し、比例制御で荷物を運ぶ Hider。"""

    def __init__(self):
        self.reflex_timer = 0
        self.is_grabbing = False
        self.is_locking = False
        self.grab_time = 0
        self.patrol_step = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.last_l_sum = 0.0
        self.stuck_counter = 0
        self.current_state = "Idle"

    def get_action(self, obs, idx):
        self.patrol_step += 1
        lidar = obs[idx.LIDAR]
        front_min = np.min(lidar[0:3])
        
        speed_scale = np.clip((front_min - 0.2) / 0.8, 0.0, 1.0)
        # 対称性を破るバイアスを付与
        omiai_bias = 0.22 * (1.0 - speed_scale) if front_min < 0.6 else 0.0
        
        seekers = [en for en in idx.OTHERS if obs[en.VISIBLE] > 0.5]
        s_vis = len(seekers) > 0

        side_diff = (lidar[1]+lidar[3]+lidar[5]) - (lidar[2]+lidar[4]+lidar[6])
        avoid_w = np.clip(1.0 - (front_min / 1.5), 0.0, 1.0)
        avoid_t = (side_diff * 6.0) + omiai_bias

        curr_l_sum = np.sum(lidar)
        if abs(curr_l_sum - self.last_l_sum) < 0.0005:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_l_sum = curr_l_sum

        min_d, target_obj = 999.0, None
        for obj_sc in idx.B + idx.RAMP:
            if obs[obj_sc.IS_LOCKED] > 0.5:
                d = math.sqrt(obs[obj_sc.REL_X]**2 + obs[obj_sc.REL_Y]**2)
                if d < min_d:
                    min_d, target_obj = d, obj_sc

        fwd, trn, lck, grb = 0.0, 0.0, 0.0, 0.0
        if target_obj and min_d < 1.05 and not self.is_locking:
            if not self.is_grabbing:
                if min_d < 0.88:
                    fwd, grb, self.is_grabbing = 0.0, 1.0, True
                else:
                    fwd = 0.15
            else:
                self.grab_time += 1
                if self.grab_time > 70:
                    lck, grb, self.is_locking, self.is_grabbing = 1, 1, True, False
                    self.grab_time = 0

        if self.reflex_timer > 0:
            self.current_state = "Reflex"
            self.reflex_timer -= 1
            fwd, trn = -0.35, 0.75
        elif front_min < 0.15:
            self.current_state = "AvoidWall"
            self.reflex_timer = 10
        elif self.stuck_counter > 40:
            self.current_state = "StuckEscape"
            self.reflex_timer = 18
            self.stuck_counter = 0
        elif s_vis:
            self.current_state = "Escape"
            tx, ty = obs[seekers[0].REL_X], obs[seekers[0].REL_Y]
            fwd = 0.55 * speed_scale
            trn = np.clip(math.atan2(-ty, -tx)*3.5 + avoid_t*avoid_w, -0.85, 0.85)
        elif target_obj and not self.is_locking:
            if self.is_grabbing:
                self.current_state = "Carrying"
                fwd = 0.2 * speed_scale
                trn = 0.1 * math.sin(self.patrol_step * 0.15)
            else:
                self.current_state = "Approach"
                tx, ty = obs[target_obj.REL_X], obs[target_obj.REL_Y]
                fwd = 0.4 * speed_scale
                trn = np.clip(math.atan2(ty, tx)*2.8 + avoid_t*avoid_w, -0.7, 0.7)
        else:
            self.current_state = "Patrol"
            self.wander_timer -= 1
            if self.wander_timer <= 0:
                self.wander_angle = np.random.uniform(-np.pi, np.pi)
                self.wander_timer = np.random.randint(120, 280)
            a_err = (self.wander_angle - obs[idx.SELF.ROT] + np.pi) % (2*np.pi)-np.pi
            fwd = 0.4 * speed_scale
            trn = np.clip(a_err * 2.6 + avoid_t * avoid_w, -0.7, 0.7)

        return np.array([fwd, trn, lck, grb])