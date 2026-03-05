# src/agents/scripted_agents.py
# scripted_agents.py v5.43 (論理の整理とHider追いかけ挙動の根本排除)

import numpy as np
import math


class RuleBasedSeeker:
    """目的地への方位偏差を計算し、壁を避けながら追従する Seeker。"""

    def __init__(self):
        self.reflex_timer = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.last_known_rel_pos_x = 0.0
        self.last_known_rel_pos_y = 0.0
        self.memory_timer = 0
        self.stuck_counter = 0
        self.escape_turn_dir = 1.0
        self.escape_fwd_dir = -1.0

    def get_action(self, obs, idx):
        L_SCALE, P_SCALE, R_SCALE = 15.0, 12.0, 5.0
        lidar_raw = obs[idx.LIDAR] * L_SCALE
        cur_rot = obs[idx.SELF.ROT] * R_SCALE
        
        front_min = np.min(lidar_raw[idx.LIDAR_FRONT_IDX])
        back_min = np.min(lidar_raw[idx.LIDAR_BACK_IDX])
        l_gap = np.sum(lidar_raw[idx.LIDAR_LEFT_IDX])
        r_gap = np.sum(lidar_raw[idx.LIDAR_RIGHT_IDX])
        
        norm_speed = np.linalg.norm(obs[idx.SELF.VEL_X:idx.SELF.VEL_Y+1])
        if 0.001 < norm_speed < 0.015: self.stuck_counter += 1
        else: self.stuck_counter = 0

        speed_scale = np.clip((front_min - 0.45) / 0.5, 0.0, 1.0)
        avoid_torque = (l_gap - r_gap) * 3.5
        avoid_w = np.clip(1.0 - (front_min / 1.5), 0.0, 1.0)

        visible_enemies = [en for en in idx.OTHERS if obs[en.VISIBLE] > 0.5]
        fwd, trn, lck, grb = 0.0, 0.0, 0.0, 0.0

        if self.reflex_timer > 0:
            self.reflex_timer -= 1
            if self.reflex_timer > 8: fwd, trn = 0.6 * self.escape_fwd_dir, 0.0
            else: fwd, trn = 0.2 * self.escape_fwd_dir, 0.85 * self.escape_turn_dir
        elif self.stuck_counter > 40 or front_min < 0.42:
            self.reflex_timer, self.stuck_counter = 14, 0
            self.escape_fwd_dir = -1.0 if front_min < back_min else 1.0
            self.escape_turn_dir = 1.0 if l_gap >= r_gap else -1.0
        else:
            if len(visible_enemies) > 0:
                target = visible_enemies[0]
                tx, ty = obs[target.REL_X] * P_SCALE, obs[target.REL_Y] * P_SCALE
                self.last_known_rel_pos_x, self.last_known_rel_pos_y = tx, ty
                self.memory_timer = 180
                target_angle = math.atan2(ty, tx)
                fwd = 0.85 * max(0.1, math.cos(target_angle)) * speed_scale
            elif self.memory_timer > 0:
                self.memory_timer -= 1
                target_angle = math.atan2(self.last_known_rel_pos_y, self.last_known_rel_pos_x)
                fwd = 0.55 * max(0.1, math.cos(target_angle)) * speed_scale
            else:
                self.wander_timer -= 1
                if avoid_w > 0.5: self.wander_timer = 0
                if self.wander_timer <= 0:
                    if l_gap > r_gap: self.wander_angle = cur_rot + np.random.uniform(0.3, np.pi)
                    else: self.wander_angle = cur_rot + np.random.uniform(-np.pi, -0.3)
                    self.wander_timer = np.random.randint(150, 400)
                target_angle = (self.wander_angle - cur_rot + np.pi) % (2*np.pi) - np.pi
                fwd = 0.45 * speed_scale

            trn = np.clip(target_angle * 2.8 + avoid_torque * avoid_w, -0.9, 0.9)

        return np.array([fwd, trn, lck, grb])


class RuleBasedHider:
    """シーカーから側方へ逃げ、壁際では適切に後退してスタックを回避する Hider。"""

    def __init__(self):
        self.reflex_timer = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.stuck_counter = 0
        self.escape_turn_dir = 1.0
        self.escape_fwd_dir = -1.0

    def get_action(self, obs, idx):
        L_SCALE, P_SCALE, R_SCALE = 15.0, 12.0, 5.0
        lidar_raw = obs[idx.LIDAR] * L_SCALE
        cur_rot = obs[idx.SELF.ROT] * R_SCALE
        
        front_min = np.min(lidar_raw[idx.LIDAR_FRONT_IDX])
        back_min = np.min(lidar_raw[idx.LIDAR_BACK_IDX])
        l_gap = np.sum(lidar_raw[idx.LIDAR_LEFT_IDX])
        r_gap = np.sum(lidar_raw[idx.LIDAR_RIGHT_IDX])
        
        norm_speed = np.linalg.norm(obs[idx.SELF.VEL_X:idx.SELF.VEL_Y+1])
        if 0.001 < norm_speed < 0.015: self.stuck_counter += 1
        else: self.stuck_counter = 0

        speed_scale = np.clip((front_min - 0.45) / 0.8, 0.15, 1.0)
        avoid_torque = (l_gap - r_gap) * 4.5
        avoid_w = np.clip(1.0 - (front_min / 1.5), 0.0, 1.0)

        seeker_vis = obs[idx.OTHERS[0].VISIBLE] > 0.5
        fwd, trn, lck, grb = 0.0, 0.0, 0.0, 0.0

        if self.reflex_timer > 0:
            self.reflex_timer -= 1
            if self.reflex_timer > 8: fwd, trn = 0.6 * self.escape_fwd_dir, 0.0
            else: fwd, trn = 0.2 * self.escape_fwd_dir, 0.85 * self.escape_turn_dir
        elif self.stuck_counter > 40 or front_min < 0.42:
            self.reflex_timer, self.stuck_counter = 14, 0
            self.escape_fwd_dir = -1.0 if front_min < back_min else 1.0
            self.escape_turn_dir = 1.0 if l_gap >= r_gap else -1.0
        else:
            if seeker_vis:
                tx, ty = obs[idx.OTHERS[0].REL_X] * P_SCALE, obs[idx.OTHERS[0].REL_Y] * P_SCALE
                angle_to_seeker = math.atan2(ty, tx)
                escape_base = (angle_to_seeker + np.pi + np.pi) % (2*np.pi) - np.pi
                side_bias = 1.2 if l_gap > r_gap else -1.2
                target_angle = (escape_base + side_bias + np.pi) % (2*np.pi) - np.pi
                
                fwd_val = math.cos(target_angle)
                if fwd_val < 0 and back_min < 0.5: fwd = 0.0
                else: fwd = 0.8 * fwd_val * speed_scale
            else:
                self.wander_timer -= 1
                if avoid_w > 0.5: self.wander_timer = 0
                if self.wander_timer <= 0:
                    if l_gap > r_gap: self.wander_angle = cur_rot + np.random.uniform(0.3, np.pi)
                    else: self.wander_angle = cur_rot + np.random.uniform(-np.pi, -0.3)
                    self.wander_timer = np.random.randint(200, 500)
                target_angle = (self.wander_angle - cur_rot + np.pi) % (2*np.pi) - np.pi
                fwd = 0.45 * speed_scale

            trn = np.clip(target_angle * 2.8 + avoid_torque * avoid_w, -0.9, 0.9)

        return np.array([fwd, trn, lck, grb])