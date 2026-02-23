# scripted_agents.py v1.59
# 演習第25回：制動フェーズ(BRAKE)導入・スタック脱出シーケンス強化版
# 
# 修正履歴:
# v1.58: 後退旋回強化。
# v1.59: ユーザーの指摘に基づき「後退前の静止フェーズ」を導入。
#        1. 壁検知・スタック時にまず 5ステップの 'BRAKE' を実行（速度入力を0にする）。
#        2. 完全に静止または減速してから、後退旋回マニューバに移行することで物理的な「弾け」を防止。
#        3. 旋回中のバック速度を少し抑え（-0.3 -> -0.2）、より精密な離脱挙動へ。

import numpy as np
import math

class RuleBasedSeeker:
    def __init__(self):
        self.last_target_pos = None
        self.target_lost_timer = 0
        self.reflex_timer = 0
        self.brake_timer = 0 # 💡 制動用タイマー
        self.cooldown = 0
        self.reflex_action = np.array([0.0, 0.0])
        self.status = "IDLE"
        self.last_check_pos = np.zeros(2)
        self.stuck_counter = 0
        
    def get_action(self, obs, engine, current_pos, current_heading, target_positions, body_exclude):
        lidar = obs[5:17]
        
        # 1. 制動フェーズ (BRAKE): 後退前に速度を殺す
        if self.brake_timer > 0:
            self.brake_timer -= 1
            self.status = "BRAKE"
            # すべての入力を 0 にして静止を待つ
            return np.array([0.0, 0.0])

        # 2. 衝突回避・後退フェーズ (ADJUST)
        if self.reflex_timer > 0:
            self.reflex_timer -= 1
            self.status = "ADJUST"
            # 前方が十分に（0.8m）空いたら探索へ復帰
            if np.min(lidar[0:3]) > 0.8:
                self.reflex_timer = 0
                return np.array([0.2, 0.0])
            return self.reflex_action
        
        # スタック検知（移動距離の監視）
        dist_moved = np.linalg.norm(current_pos - self.last_check_pos)
        self.last_check_pos = current_pos.copy()
        if dist_moved < 0.005:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # 3. 回避トリガー判定
        front_min = np.min(lidar[0:3])
        if front_min < 0.25 or self.stuck_counter > 20:
            # 💡 修正：いきなり下がらず、まず「止まる」
            self.brake_timer = 6 # 6ステップ（約0.03秒）静止入力を入れる
            self.reflex_timer = 25
            
            best_idx = np.argmax(lidar)
            angles_deg = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
            target_deg = angles_deg[best_idx]
            
            # 後退旋回アクションの準備（BRAKE終了後に実行される）
            if abs(target_deg) >= 90:
                turn_dir = 1.0 if target_deg > 0 else -1.0
                # 静止状態からのスタートなので、速度は少し控えめでも確実に動く
                self.reflex_action = np.array([-0.2, turn_dir * 0.75])
            else:
                self.reflex_action = np.array([0.0, np.clip(np.deg2rad(target_deg) * 3.0, -0.8, 0.8)])
            
            self.stuck_counter = 0
            return np.array([0.0, 0.0]) # 最初のステップは BRAKE

        # 4. ターゲット追跡
        best_target = None
        min_dist = float('inf')
        for t_pos in target_positions:
            visible = engine.is_visible(current_pos, t_pos, body_exclude=body_exclude)
            dist = np.linalg.norm(t_pos - current_pos)
            if visible and dist < min_dist: 
                min_dist = dist
                best_target = t_pos

        if best_target is not None:
            self.status = "CHASE"
            self.last_target_pos = best_target.copy()
            self.target_lost_timer = 0
            return self._compute_pursuit_action(current_pos, current_heading, best_target)
        
        elif self.last_target_pos is not None and self.target_lost_timer < 60:
            self.status = "TRACK"
            self.target_lost_timer += 1
            return self._compute_pursuit_action(current_pos, current_heading, self.last_target_pos)
        
        else:
            # 5. 巡航・探索
            self.status = "EXPLR"
            return self._compute_explore_action(obs)

    def _compute_pursuit_action(self, pos, heading, target):
        diff = target - pos
        dist = np.linalg.norm(diff)
        target_angle = math.atan2(diff[1], diff[0])
        err = (target_angle - heading + math.pi) % (2 * math.pi) - math.pi
        fwd = 0.5 if dist > 0.4 else 0.1
        return np.array([fwd, np.clip(err * 2.5, -0.8, 0.8)])

    def _compute_explore_action(self, obs):
        lidar = obs[5:17]
        angles_deg = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
        scores = []
        for i in range(12):
            score = lidar[i] - abs(angles_deg[i]) * 0.008
            scores.append(score)
        best_idx = np.argmax(np.array(scores))
        target_ang = np.deg2rad(angles_deg[best_idx])
        fwd = 0.4 if lidar[best_idx] > 0.5 and abs(angles_deg[best_idx]) < 30 else 0.15
        turn = np.clip(target_ang * 1.5, -0.6, 0.6)
        return np.array([fwd, turn])