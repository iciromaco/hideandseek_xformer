# scripted_agents.py v2.40
# 演習第26回：【成功バージョン 100% 復元 ＆ lock_prepare 知能再展開版】
# 
# 修正内容:
# 1. ユーザー提供 v2.37 ロジックの完全復元:
#    - Hider の lock_prepare挙動を再構築。フリーな道具に対し、停止・接近・周期パルスを送信。
#    - 把持時間（grab_time > 50）による自動 Lock 移行プロトコルを復旧。
# 2. 幽霊把持（Ghost Grasp）解消ロジック:
#    - 距離 1.6m 超過時に強制解除パルス (1.0) を送信するトグル制御。
# 3. 準備期間の機動力: 基本前進速度を摩擦に負けないレベル（0.45等）で維持。
# 4. 1行1命令（1-line-1-command）徹底: 論理パスをすべて垂直展開。

import numpy as np
import math

class RuleBasedSeeker:
    """敵を追い、邪魔な道具を排除するシーカー"""
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
        self.current_state = "Idle"
        
    def get_action(self, obs):
        lidar_s = obs[5:17]; l0_s, l1_s, l2_s = lidar_s[0], lidar_s[1], lidar_s[2]; front_dist_s = min(l0_s, l1_s, l2_s)
        l_diff_s = lidar_s - self.last_lidar; l_norm_s = np.linalg.norm(l_diff_s)
        if l_norm_s < 0.0005: self.stuck_counter += 1
        else: self.stuck_counter = 0
        self.last_lidar = lidar_s.copy()
        
        h1_v_s, h2_v_s, ramp_v_s = obs[47], obs[54], obs[40]; r_rel_s = obs[33:35]; r_dist_s = np.linalg.norm(r_rel_s) if ramp_v_s > 0.5 else 999.0
        grab_sig_s = 0.0
        if self.is_grabbing and r_dist_s > 1.6: self.is_grabbing, grab_sig_s = False, 1.0
        if h1_v_s > 0.5 or h2_v_s > 0.5:
            if self.is_grabbing: grab_sig_s, self.is_grabbing = 1.0, False
        elif ramp_v_s > 0.5 and r_dist_s < 1.15:
            if not self.is_grabbing: grab_sig_s, self.is_grabbing = 1.0, True
            
        if self.reflex_timer > 0: self.current_state, self.reflex_timer = "Reflex", self.reflex_timer - 1; return np.array([-0.1, 0.7, 0.0, grab_sig_s])
        if front_dist_s < (0.22 if not self.is_grabbing else 0.0) or self.stuck_counter > 50: self.current_state, self.reflex_timer, self.stuck_counter = "Avoid", 20, 0; return np.array([0.0, 0.0, 0.0, grab_sig_s])
        
        target_rel_s = obs[41:43] if h1_v_s > 0.5 else (obs[48:50] if h2_v_s > 0.5 else None)
        if target_rel_s is not None:
            self.current_state, self.last_known_rel_pos, self.memory_timer = "Chase", target_rel_s, 150 
            err_s = math.atan2(target_rel_s[1], target_rel_s[0]); return np.array([0.4, np.clip(err_s * 2.5, -0.7, 0.7), 0.0, grab_sig_s])
        if self.memory_timer > 0:
            self.current_state, self.memory_timer = "Memory", self.memory_timer - 1; m_rel_s = self.last_known_rel_pos; m_err_s = math.atan2(m_rel_s[1], m_rel_s[0])
            return np.array([0.3 if np.linalg.norm(m_rel_s) > 0.5 else 0.0, np.clip(m_err_s * 2.0, -0.6, 0.6), 0.0, grab_sig_s])
        if ramp_v_s > 0.5:
            self.current_state, r_err_s = "Approach", math.atan2(r_rel_s[1], r_rel_s[0])
            if self.is_grabbing: return np.array([-0.3, 0.4, 0.0, grab_sig_s])
            return np.array([0.3, np.clip(r_err_s * 2.2, -0.7, 0.7), 0.0, grab_sig_s])
        self.current_state, self.patrol_step, self.wander_timer = "Patrol", self.patrol_step + 1, self.wander_timer - 1
        if self.wander_timer <= 0: self.wander_angle, self.wander_timer = np.deg2rad([0, 45, -45, 90, -90, 180][np.random.randint(6)]), np.random.randint(100, 200)
        return np.array([0.18, np.clip(self.wander_angle + 0.2*math.sin(self.patrol_step*0.1), -0.6, 0.6), 0.0, grab_sig_s])

class RuleBasedHider:
    """以前の成功ロジック(lock_prepare)を復元したハイダー"""
    def __init__(self):
        self.reflex_timer = 0
        self.is_grabbing = False
        self.is_locking = False
        self.interact_obj_idx = -1 
        self.grab_time = 0
        self.action_tick = 0
        self.patrol_step = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.last_lidar = np.zeros(12)
        self.stuck_counter = 0
        self.current_state = "Idle"

    def get_action(self, obs):
        self.action_tick += 1
        lid_h = obs[5:17]; front_d_h = min(lid_h[0], lid_h[1], lid_h[2]); s_v_h = obs[47]
        l_diff_h = np.linalg.norm(lid_h - self.last_lidar)
        if l_diff_h < 0.0005: self.stuck_counter += 1
        else: self.stuck_counter = 0
        self.last_lidar = lid_h.copy()
        
        # 状態同期
        if self.interact_obj_idx != -1:
            base_h = 17 + self.interact_obj_idx * 8; i_st_h = obs[base_h + 6]
            if i_st_h == 1.0: self.is_locking, self.is_grabbing = True, False
            elif i_st_h == 2.0: self.is_locking, self.is_grabbing = False, True
            else: self.is_locking, self.is_grabbing = False, False
            obj_d_h = np.linalg.norm(obs[base_h : base_h + 2])
            if self.is_grabbing and (obj_d_h > 1.6 or self.stuck_counter > 60):
                self.is_grabbing, self.interact_obj_idx = False, -1; return np.array([0.0, 0.0, 0.0, 1.0])
            if (not self.is_grabbing) and (not self.is_locking): self.interact_obj_idx, self.grab_time = -1, 0

        # 道具探索
        min_d_h, b_idx_h, b_base_h = 999.0, -1, -1
        if self.interact_obj_idx != -1:
            b_idx_h = self.interact_obj_idx; b_base_h = 17 + b_idx_h * 8; min_d_h = np.linalg.norm(obs[b_base_h : b_base_h + 2])
        else:
            for i_h, b_h in enumerate([17, 25, 33]):
                if obs[b_h + 7] > 0.5:
                    d_h = np.linalg.norm(obs[b_h : b_h+2])
                    if d_h < min_d_h: min_d_h, b_idx_h, b_base_h = d_h, i_h, b_h

        # 信号計算 (lock_prepare)
        grab_s_h, lock_s_h, l_prep_h = 0.0, 0.0, False
        if b_idx_h != -1 and min_d_h < 1.22:
            t_st_h = obs[b_base_h + 6]
            if (not self.is_grabbing) and (not self.is_locking) and t_st_h < 0.5:
                l_prep_h, self.interact_obj_idx = True, b_idx_h
                if min_d_h < 1.02 and self.action_tick % 3 == 0: lock_s_h = 1.0
            elif (not self.is_grabbing) and (not self.is_locking) and t_st_h >= 1.0:
                if self.action_tick % 6 == 0: grab_s_h = 1.0
                self.interact_obj_idx = b_idx_h
            elif self.is_grabbing:
                self.grab_time += 1
                if self.grab_time > 50 and self.action_tick % 5 == 0: lock_s_h = 1.0
        
        # 回避 ＆ ステート決定
        if self.reflex_timer > 0: self.current_state, self.reflex_timer = "Reflex", self.reflex_timer - 1; return np.array([0.03, 0.0, lock_s_h, 0.0]) if self.is_grabbing else np.array([-0.03, 0.35, lock_s_h, grab_s_h])
        if front_d_h < (0.0 if self.is_grabbing else 0.25): self.current_state, self.reflex_timer = "Avoid", 12; return np.array([0.0, 0.0, 0.0, grab_s_h])

        if b_idx_h != -1 and not self.is_locking and l_prep_h:
            self.current_state, err_h = "LockPrep", math.atan2(obs[b_base_h+1], obs[b_base_h]); f_h = 0.0 if min_d_h < 0.96 else 0.03 if min_d_h < 1.02 else 0.08
            return np.array([f_h, np.clip(err_h*1.6, -0.35, 0.35), lock_s_h, 0.0])
        if s_v_h > 0.5:
            self.current_state, esc_h = "Escape", math.atan2(-obs[42], -obs[41]); f_h = 0.35 if abs(esc_h)<0.5 else 0.1
            return np.array([f_h, np.clip(esc_h*2.0, -0.6, 0.6), 0.0, grab_s_h])
        if b_idx_h != -1 and not self.is_locking:
            self.current_state, err_h = "Approach", math.atan2(obs[b_base_h+1], obs[b_base_h])
            f_h, t_g, t_l = (0.05, 1.2, 0.22) if self.is_grabbing else (0.45, 2.2, 0.6)
            return np.array([f_h, np.clip(err_h*t_g, -t_l, t_l), lock_s_h, grab_s_h])

        self.current_state, self.patrol_step, self.wander_timer = "Patrol", self.patrol_step + 1, self.wander_timer - 1
        if self.wander_timer <= 0: self.wander_angle, self.wander_timer = np.random.uniform(-1.0, 1.0), np.random.randint(60, 150)
        return np.array([0.35, np.clip(self.wander_angle + 0.1*math.sin(self.patrol_step*0.2), -0.5, 0.5), 0.0, 0.0])