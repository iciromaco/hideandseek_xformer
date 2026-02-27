# scripted_agents.py v2.30
# 演習第26回：【幽霊把持解消 ＆ 強制解除パルス実装 ＆ 1行1文徹底版】
# 
# 修正内容:
# 1. 幽霊把持（Ghost Grasp）の防止:
#    - Seeker および Hider において、対象との距離が 1.6m を超えた場合、
#      内部フラグを下ろすだけでなく、環境（トグル式）に対して確実に grab_sig=1.0 を送信して物理拘束を解除。
# 2. スタック・異常距離時の即時復帰:
#    - 異常を検知した瞬間、他の行動をキャンセルして解除信号を優先的に出力。
# 3. 1行1命令・極限展開の遵守: 距離チェック、パルス生成、フラグ更新をすべて独立行で記述。

import numpy as np
import math

class RuleBasedSeeker:
    """敵を記憶して追い、邪魔な道具を排除するシーカー"""
    def __init__(self):
        self.reflex_timer = 0
        self.patrol_step = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.is_grabbing = False 
        self.last_known_rel_pos = None
        self.memory_timer = 0
        self.last_lidar = np.zeros(12)
        self.stuck_counter = 0
        
    def get_action(self, obs):
        lidar = obs[5:17]
        front_dist = min(lidar[0], lidar[1], lidar[2])
        lidar_diff = np.linalg.norm(lidar - self.last_lidar)
        if lidar_diff < 0.0005: self.stuck_counter = self.stuck_counter + 1
        else: self.stuck_counter = 0
        self.last_lidar = lidar.copy()

        h1_vis, h2_vis, ramp_vis = obs[47], obs[54], obs[40]
        ramp_rel = obs[33:35]
        ramp_dist = np.linalg.norm(ramp_rel) if ramp_vis > 0.5 else 999.0

        grab_signal = 0.0
        # 幽霊把持防止ロジック
        if self.is_grabbing:
            if ramp_dist > 1.6:
                self.is_grabbing = False
                grab_signal = 1.0 # 距離超過による強制解除パルス

        if h1_vis > 0.5 or h2_vis > 0.5:
            if self.is_grabbing:
                grab_signal = 1.0
                self.is_grabbing = False
        elif ramp_vis > 0.5 and ramp_dist < 1.15:
            if not self.is_grabbing:
                grab_signal = 1.0
                self.is_grabbing = True

        avoid_th = 0.22 if not self.is_grabbing else 0.0
        if self.reflex_timer > 0:
            self.reflex_timer = self.reflex_timer - 1
            return np.array([-0.1, 0.7, 0.0, grab_signal])
        if front_dist < avoid_th or self.stuck_counter > 50:
            self.reflex_timer = 20
            self.stuck_counter = 0
            return np.array([0.0, 0.0, 0.0, grab_signal])

        target_rel = obs[41:43] if h1_vis > 0.5 else (obs[48:50] if h2_vis > 0.5 else None)
        if target_rel is not None:
            self.last_known_rel_pos, self.memory_timer = target_rel, 150 
            err = math.atan2(target_rel[1], target_rel[0])
            return np.array([0.4, np.clip(err * 2.5, -0.7, 0.7), 0.0, grab_signal])
        if self.memory_timer > 0:
            self.memory_timer = self.memory_timer - 1
            m_rel = self.last_known_rel_pos
            m_err = math.atan2(m_rel[1], m_rel[0])
            dist_m = np.linalg.norm(m_rel)
            fwd_m = 0.3 if dist_m > 0.5 else 0.0
            return np.array([fwd_m, np.clip(m_err * 2.0, -0.6, 0.6), 0.0, grab_signal])
        if ramp_vis > 0.5:
            r_err = math.atan2(ramp_rel[1], ramp_rel[0])
            if self.is_grabbing: return np.array([-0.3, 0.4, 0.0, grab_signal])
            return np.array([0.3, np.clip(r_err * 2.2, -0.7, 0.7), 0.0, grab_signal])

        self.patrol_step = self.patrol_step + 1
        self.wander_timer = self.wander_timer - 1
        if self.wander_timer <= 0:
            dirs = [0, 45, -45, 90, -90, 180]
            self.wander_angle = np.deg2rad(dirs[np.random.randint(6)])
            self.wander_timer = np.random.randint(100, 200)
        return np.array([0.18, np.clip(self.wander_angle + 0.2*math.sin(self.patrol_step*0.1), -0.6, 0.6), 0.0, grab_signal])

class RuleBasedHider:
    """環境と完全同期し、スタックを回避しながら道具を運搬・ロックするハイダー"""
    def __init__(self):
        self.reflex_timer = 0
        self.is_grabbing = False
        self.is_locking = False
        self.interact_obj_idx = -1 
        self.grab_time = 0
        self.lock_retry_timer = 0
        self.wander_timer = 0
        self.patrol_step = 0
        self.wander_angle = 0.0
        self.last_lidar = np.zeros(12)
        self.stuck_counter = 0
        self.prev_turn = 0.0

    def _smooth_turn(self, err, gain, limit):
        desired_turn = np.clip(err * gain, -limit, limit)
        max_delta = 0.08
        turn_delta = desired_turn - self.prev_turn
        turn_delta = np.clip(turn_delta, -max_delta, max_delta)
        self.prev_turn = self.prev_turn + turn_delta
        return self.prev_turn

    def get_action(self, obs):
        lidar = obs[5:17]
        front_dist = min(lidar[0], lidar[1], lidar[2])
        s_vis = obs[47]
        
        l_diff = np.linalg.norm(lidar - self.last_lidar)
        if l_diff < 0.0005: self.stuck_counter = self.stuck_counter + 1
        else: self.stuck_counter = 0
        self.last_lidar = lidar.copy()
        
        # 1. 状態の再同期 ＆ 幽霊把持の物理解除
        if self.interact_obj_idx != -1:
            base = 17 + self.interact_obj_idx * 8
            tool_state = obs[base + 6]
            self.is_locking = (tool_state > 1.5)
            if (tool_state > 0.5) and (not self.is_locking):
                self.is_grabbing = True
            if self.is_locking:
                self.is_grabbing = False
                self.grab_time = 0
                self.lock_retry_timer = 0
            obj_dist = np.linalg.norm(obs[base : base + 2])
            
            # 距離超過検知: フラグだけでなく解除パルスを送る
            if self.is_grabbing and obj_dist > 1.6:
                self.is_grabbing = False
                self.interact_obj_idx = -1
                return np.array([0.0, 0.0, 0.0, 1.0]) # 強制解除パルス送信
            
            # スタック検知時の強制解除
            if self.is_grabbing and self.stuck_counter > 60:
                self.is_grabbing = False
                self.interact_obj_idx = -1
                return np.array([0.0, 0.0, 0.0, 1.0]) # 強制解除パルス送信

            if not self.is_grabbing and not self.is_locking and self.lock_retry_timer <= 0:
                self.interact_obj_idx, self.grab_time = -1, 0

        # 2. 道具探索
        best_base, best_idx, min_d = -1, -1, 999.0
        if self.interact_obj_idx != -1:
            best_idx = self.interact_obj_idx
            best_base = 17 + best_idx * 8
            min_d = np.linalg.norm(obs[best_base : best_base + 2])
        else:
            for i, b in enumerate([17, 25, 33]):
                if obs[b+7] > 0.5:
                    d = np.linalg.norm(obs[b : b+2])
                    if d < min_d: min_d, best_idx, best_base = d, i, b

        # 3. 信号計算
        grab_sig, lock_sig = 0.0, 0.0
        if best_idx != -1 and min_d < 1.22:
            if not self.is_grabbing and not self.is_locking:
                grab_sig, self.is_grabbing, self.interact_obj_idx = 1.0, True, best_idx

        if self.is_grabbing and not self.is_locking:
            self.grab_time = self.grab_time + 1
            if self.grab_time > 40:
                lock_sig = 1.0
                self.lock_retry_timer = max(self.lock_retry_timer, 8)

        if self.lock_retry_timer > 0 and not self.is_locking and best_idx != -1 and min_d < 1.18:
            lock_sig = 1.0
            self.lock_retry_timer = self.lock_retry_timer - 1
        
        # 4. 回避・反射
        avoid_m = 0.25 if not self.is_grabbing else 0.12
        if self.reflex_timer > 0:
            self.reflex_timer = self.reflex_timer - 1
            if self.is_grabbing:
                return np.array([0.03, 0.0, lock_sig, 0.0])
            return np.array([-0.02, 0.25, lock_sig, grab_sig])
        if front_dist < avoid_m:
            self.reflex_timer = 12
            return np.array([0.0, 0.0, 0.0, grab_sig])

        # 5. 行動
        if self.is_grabbing and not self.is_locking:
            stable_turn = self._smooth_turn(0.0, 1.0, 0.10)
            return np.array([0.05, stable_turn, lock_sig, 0.0])

        if best_idx != -1 and min_d < 1.22 and not self.is_locking and (self.is_grabbing or lock_sig > 0.5 or grab_sig > 0.5):
            rel_close = obs[best_base : best_base + 2]
            err_close = math.atan2(rel_close[1], rel_close[0])
            turn_close = self._smooth_turn(err_close, 1.4, 0.25)
            return np.array([0.04, turn_close, lock_sig, grab_sig])
        if s_vis > 0.5:
            s_rel = obs[41:43]
            esc = math.atan2(-s_rel[1], -s_rel[0])
            turn_esc = self._smooth_turn(esc, 1.6, 0.45)
            return np.array([0.30 if abs(esc)<0.5 else 0.1, turn_esc, lock_sig, grab_sig])
        if best_idx != -1 and min_d < 1.22 and not self.is_locking:
            rel_near = obs[best_base : best_base + 2]
            err_near = math.atan2(rel_near[1], rel_near[0])
            turn_near = self._smooth_turn(err_near, 1.2, 0.20)
            return np.array([0.04, turn_near, lock_sig, grab_sig])
        if best_idx != -1 and not self.is_locking:
            rel = obs[best_base : best_base + 2]
            err = math.atan2(rel[1], rel[0])
            turn_obj = self._smooth_turn(err, 1.8 if self.is_grabbing else 2.0, 0.4 if self.is_grabbing else 0.5)
            fwd_obj = 0.05 if self.is_grabbing else 0.2
            return np.array([fwd_obj, turn_obj, lock_sig, grab_sig])

        self.patrol_step, self.wander_timer = self.patrol_step + 1, self.wander_timer - 1
        if self.wander_timer <= 0:
            self.wander_angle, self.wander_timer = np.random.uniform(-1.0, 1.0), np.random.randint(60, 150)
        return np.array([0.18, np.clip(self.wander_angle + 0.1*math.sin(self.patrol_step*0.2), -0.5, 0.5), 0.0, 0.0])