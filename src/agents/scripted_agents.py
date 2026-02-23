# scripted_agents.py v1.70
# 演習第26回：【最終完全版】NPCエージェント・ライブラリ統合
# 
# 修正履歴:
# v1.69: RuleBasedHider 追加。
# v1.70: ユーザー指摘の AttributeError を解消するため SimpleWanderer を統合。
#        1. SimpleWanderer クラスを追加し、get_action メソッドを実装。
#        2. スタック検知（物理的本能）を備えたランダム放浪ロジックを提供。
#        3. これにより、この1ファイルで「シーカー」「賢いハイダー」「アホなハイダー」が揃う。

import numpy as np
import math

class RuleBasedSeeker:
    """ハイダーを精密に追い詰めるプロフェッショナル・シーカー"""
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
        
        if self.brake_timer > 0:
            self.brake_timer -= 1; self.status = "BRAKE"; return np.array([0.0, 0.0])

        if self.reflex_timer > 0:
            self.reflex_timer -= 1; self.status = "ADJUST"
            if np.min(lidar[0:3]) > 0.6: self.reflex_timer = 0; self.cooldown = 10; return np.array([0.1, 0.0])
            return self.reflex_action
        
        if self.cooldown > 0: self.cooldown -= 1
        
        dist_moved = np.linalg.norm(current_pos - self.last_check_pos)
        self.last_check_pos = current_pos.copy()
        if dist_moved < 0.005: self.stuck_counter += 1
        else: self.stuck_counter = 0

        best_target = None; min_dist = float('inf')
        for i, t_pos in enumerate(target_positions):
            t_id = target_body_ids[i]
            visible = engine.is_visible(current_pos, t_pos, body_exclude=body_exclude, target_body_id=t_id)
            dist = np.linalg.norm(t_pos - current_pos)
            if visible and dist < min_dist: min_dist = dist; best_target = t_pos

        front_min = np.min(lidar[0:3])
        is_enemy_in_front = (best_target is not None and front_min > (min_dist - 0.9))

        if (front_min < 0.22 and not is_enemy_in_front) or self.stuck_counter > 45:
            self.status = "ADJUST"; self.brake_timer = 4; self.reflex_timer = 15
            best_idx = np.argmax(lidar)
            angles_deg = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
            target_deg = angles_deg[best_idx]
            if abs(target_deg) >= 90:
                self.reflex_action = np.array([-0.1, (1.0 if target_deg > 0 else -1.0) * 0.6])
            else:
                self.reflex_action = np.array([0.0, np.clip(np.deg2rad(target_deg) * 2.5, -0.7, 0.7)])
            self.stuck_counter = 0; return np.array([0.0, 0.0])

        if best_target is not None:
            self.status = "CHASE"; self.last_target_pos = best_target.copy(); self.target_lost_timer = 0
            return self._compute_pursuit_action(current_pos, current_heading, best_target)
        elif self.last_target_pos is not None and self.target_lost_timer < 100:
            self.status = "TRACK"; self.target_lost_timer += 1
            if np.linalg.norm(self.last_target_pos - current_pos) < 0.8:
                self.status = "SCAN"; return np.array([0.0, 0.5])
            return self._compute_pursuit_action(current_pos, current_heading, self.last_target_pos, speed_limit=0.2)
        else:
            self.status = "EXPLR"; return self._compute_explore_action(obs)

    def _compute_pursuit_action(self, pos, heading, target, speed_limit=0.35):
        diff = target - pos; dist = np.linalg.norm(diff)
        target_angle = math.atan2(diff[1], diff[0])
        err = (target_angle - heading + math.pi) % (2 * math.pi) - math.pi
        ideal_dist = 0.90 
        if dist < (ideal_dist - 0.15): fwd = -0.10 
        elif dist < ideal_dist: fwd = 0.0
        else: fwd = np.clip((dist - ideal_dist) * 0.8, 0.0, speed_limit)
        if abs(err) > np.deg2rad(20): fwd *= 0.1
        turn = np.clip(err * 3.0, -0.9, 0.9)
        return np.array([fwd, turn])

    def _compute_explore_action(self, obs):
        lidar = obs[5:17]; angles_deg = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
        scores = [min(lidar[i], 3.0) - abs(angles_deg[i]) * 0.015 for i in range(12)]
        best_idx = np.argmax(np.array(scores))
        target_ang = np.deg2rad(angles_deg[best_idx])
        fwd = 0.15 if lidar[best_idx] > 0.5 and abs(angles_deg[best_idx]) < 45 else 0.05
        turn = np.clip(target_ang * 1.5, -0.6, 0.6)
        return np.array([fwd, turn])

class RuleBasedHider:
    """シーカーから逃げつつ、壁を避けて生存する知恵あるHider"""
    def __init__(self):
        self.status = "WANDER"
        self.timer = 0
        self.action = np.array([0.15, 0.0])
        self.reflex_timer = 0
        
    def get_action(self, obs, current_pos, current_heading, seeker_pos=None):
        lidar = obs[5:17]; front_dist = min(lidar[0:3])
        
        # 1. 壁回避 (最優先)
        if self.reflex_timer > 0:
            self.reflex_timer -= 1
            if front_dist > 0.8: self.reflex_timer = 0
            return self.reflex_action

        if front_dist < 0.25:
            self.reflex_timer = 15; self.status = "ESCAPE_WALL"
            best_idx = np.argmax(lidar)
            angles_deg = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
            self.reflex_action = np.array([-0.05, np.clip(np.deg2rad(angles_deg[best_idx]), -0.6, 0.6)])
            return self.reflex_action

        # 2. シーカー回避ロジック
        if seeker_pos is not None:
            diff = current_pos - seeker_pos
            dist = np.linalg.norm(diff)
            if dist < 3.0: # 3m以内で警戒
                self.status = "DANGER"
                escape_angle = math.atan2(diff[1], diff[0])
                err = (escape_angle - current_heading + math.pi) % (2 * math.pi) - math.pi
                fwd = 0.25 if dist > 1.2 else 0.35
                turn = np.clip(err * 2.0, -0.8, 0.8)
                return np.array([fwd, turn])

        # 3. 通常巡回 (ランダム)
        self.status = "WANDER"
        self.timer -= 1
        if self.timer <= 0:
            self.timer = np.random.randint(60, 120)
            self.action = np.array([0.18, np.random.uniform(-0.4, 0.4)])
        return self.action

class SimpleWanderer:
    """知恵を持たない、純粋なランダム放浪者（スタック回避機能付き）"""
    def __init__(self, agent_id=0):
        self.agent_id = agent_id
        self.timer = 0
        self.mode = "WANDER"
        self.action = np.array([0.15, 0.0])
        self.last_pos = np.zeros(2)
        self.stuck_counter = 0
        np.random.seed(42 + agent_id)
    
    def get_action(self, obs, current_pos):
        lidar = obs[5:17]; front_dist = min(lidar[0:3])
        
        # スタック検知
        dist_moved = np.linalg.norm(current_pos - self.last_pos)
        self.last_pos = current_pos.copy()
        if dist_moved < 0.005: self.stuck_counter += 1
        else: self.stuck_counter = 0

        # 緊急回避 (物理的本能)
        if self.mode == "BRAKE":
            self.timer -= 1
            if self.timer <= 0:
                self.mode = "ESCAPE"; self.timer = 30
                best_idx = np.argmax(lidar)
                angles_deg = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
                self.action = np.array([-0.05, np.clip(np.deg2rad(angles_deg[best_idx]), -0.5, 0.5)])
            return np.array([0.0, 0.0])

        if self.mode == "ESCAPE":
            self.timer -= 1; 
            if self.timer <= 0 or front_dist > 0.8: self.mode = "WANDER"
            return self.action

        if front_dist < 0.22 or self.stuck_counter > 40:
            self.mode = "BRAKE"; self.timer = 8; return np.array([0.0, 0.0])

        # 通常巡回
        self.timer -= 1
        if self.timer <= 0:
            self.timer = np.random.randint(80, 150)
            self.action = np.array([0.18, np.random.uniform(-0.4, 0.4)])
        return self.action