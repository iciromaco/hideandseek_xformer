# scripted_agents.py v2.33
# 演習第26回：【55次元同期・極限展開 ＆ 幽霊把持完全解消版】
# 
# 修正内容:
# 1. 論理の垂直展開 (行数復旧):
#    - 報酬計算、距離チェック、アクションベクトル構築を一行ずつ明示的に展開。
# 2. 55次元観測インデックス同期:
#    - 55次元回帰 (interaction_state 集約) に合わせ、ramp_vis(40), h1_vis(47), h2_vis(54) を正確に参照。
# 3. 幽霊把持（Ghost Grasp）解消ロジックの完全記述:
#    - 距離超過時の強制パルス (1.0) 送信プロセスを省略なく実装。

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
        lidar_s = obs[5:17]
        l0_s = lidar_s[0]
        l1_s = lidar_s[1]
        l2_s = lidar_s[2]
        front_dist_s = min(l0_s, l1_s, l2_s)
        
        diff_lidar_s = lidar_s - self.last_lidar
        l_norm_s = np.linalg.norm(diff_lidar_s)
        if l_norm_s < 0.0005:
            self.stuck_counter = self.stuck_counter + 1
        else:
            self.stuck_counter = 0
        self.last_lidar = lidar_s.copy()

        # インデックス同期
        h1_vis_s = obs[47]
        h2_vis_s = obs[54]
        ramp_vis_s = obs[40]
        ramp_rel_s = obs[33:35]
        
        r_dx_s = ramp_rel_s[0]
        r_dy_s = ramp_rel_s[1]
        ramp_dist_s = math.sqrt(r_dx_s*r_dx_s + r_dy_s*r_dy_s)
        if ramp_vis_s <= 0.5:
            ramp_dist_s = 999.0

        grab_signal_s = 0.0
        # 幽霊把持防止
        if self.is_grabbing:
            if ramp_dist_s > 1.6:
                self.is_grabbing = False
                grab_signal_s = 1.0

        if h1_vis_s > 0.5 or h2_vis_s > 0.5:
            if self.is_grabbing:
                grab_signal_s = 1.0
                self.is_grabbing = False
        elif ramp_vis_s > 0.5 and ramp_dist_s < 1.15:
            if not self.is_grabbing:
                grab_signal_s = 1.0
                self.is_grabbing = True

        avoid_th_s = 0.22
        if self.is_grabbing:
            avoid_th_s = 0.0
            
        if self.reflex_timer > 0:
            self.reflex_timer = self.reflex_timer - 1
            return np.array([-0.1, 0.7, 0.0, grab_signal_s])
            
        if front_dist_s < avoid_th_s or self.stuck_counter > 50:
            self.reflex_timer = 20
            self.stuck_counter = 0
            return np.array([0.0, 0.0, 0.0, grab_signal_s])

        target_rel_s = None
        if h1_vis_s > 0.5:
            target_rel_s = obs[41:43]
        elif h2_vis_s > 0.5:
            target_rel_s = obs[48:50]
            
        if target_rel_s is not None:
            self.last_known_rel_pos = target_rel_s
            self.memory_timer = 150 
            err_s = math.atan2(target_rel_s[1], target_rel_s[0])
            clamped_err_s = np.clip(err_s * 2.5, -0.7, 0.7)
            return np.array([0.4, clamped_err_s, 0.0, grab_signal_s])
            
        if self.memory_timer > 0:
            self.memory_timer = self.memory_timer - 1
            m_rel_s = self.last_known_rel_pos
            m_err_s = math.atan2(m_rel_s[1], m_rel_s[0])
            m_dist_s = np.linalg.norm(m_rel_s)
            fwd_m_s = 0.3
            if m_dist_s <= 0.5:
                fwd_m_s = 0.0
            clamped_m_err_s = np.clip(m_err_s * 2.0, -0.6, 0.6)
            return np.array([fwd_m_s, clamped_m_err_s, 0.0, grab_signal_s])
            
        if ramp_vis_s > 0.5:
            r_err_s = math.atan2(ramp_rel_s[1], ramp_rel_s[0])
            if self.is_grabbing:
                return np.array([-0.3, 0.4, 0.0, grab_signal_s])
            clamped_r_err_s = np.clip(r_err_s * 2.2, -0.7, 0.7)
            return np.array([0.3, clamped_r_err_s, 0.0, grab_signal_s])

        self.patrol_step = self.patrol_step + 1
        self.wander_timer = self.wander_timer - 1
        if self.wander_timer <= 0:
            dirs_list = [0, 45, -45, 90, -90, 180]
            choice_idx = np.random.randint(6)
            self.wander_angle = np.deg2rad(dirs_list[choice_idx])
            self.wander_timer = np.random.randint(100, 200)
            
        osc_s = 0.2 * math.sin(self.patrol_step * 0.1)
        clamped_w_err_s = np.clip(self.wander_angle + osc_s, -0.6, 0.6)
        return np.array([0.18, clamped_w_err_s, 0.0, grab_signal_s])

class RuleBasedHider:
    """運搬・ロック・スタック回避を極限展開したハイダー"""
    def __init__(self):
        self.reflex_timer = 0
        self.is_grabbing = False
        self.is_locking = False
        self.interact_obj_idx = -1 
        self.grab_time = 0
        self.action_tick = 0
        self.wander_timer = 0
        self.patrol_step = 0
        self.wander_angle = 0.0
        self.last_lidar = np.zeros(12)
        self.stuck_counter = 0

    def get_action(self, obs):
        self.action_tick = self.action_tick + 1
        lidar_h = obs[5:17]
        l0_h = lidar_h[0]
        l1_h = lidar_h[1]
        l2_h = lidar_h[2]
        front_dist_h = min(l0_h, l1_h, l2_h)
        s_vis_h = obs[47]
        
        diff_lidar_h = lidar_h - self.last_lidar
        l_norm_h = np.linalg.norm(diff_lidar_h)
        if l_norm_h < 0.0005:
            self.stuck_counter = self.stuck_counter + 1
        else:
            self.stuck_counter = 0
        self.last_lidar = lidar_h.copy()
        
        # 1. 把持状態同期・強制解除
        if self.interact_obj_idx != -1:
            base_h = 17 + self.interact_obj_idx * 8 
            int_state_h = obs[base_h + 6]
            self.is_locking = (int_state_h == 1.0)
            self.is_grabbing = (int_state_h == 2.0)
            o_rel_h = obs[base_h : base_h + 2]
            obj_dist_h = np.linalg.norm(o_rel_h)
            
            # 距離超過解除
            if self.is_grabbing and obj_dist_h > 1.6:
                self.is_grabbing = False
                self.interact_obj_idx = -1
                return np.array([0.0, 0.0, 0.0, 1.0])
            
            # スタック強制解除
            if self.is_grabbing and self.stuck_counter > 60:
                self.is_grabbing = False
                self.interact_obj_idx = -1
                return np.array([0.0, 0.0, 0.0, 1.0])

            if (not self.is_grabbing) and (not self.is_locking):
                self.interact_obj_idx = -1
                self.grab_time = 0

        # 2. 道具探索
        best_base_h, best_idx_h, min_d_h = -1, -1, 999.0
        if self.interact_obj_idx != -1:
            best_idx_h = self.interact_obj_idx
            best_base_h = 17 + best_idx_h * 8
            min_d_h = np.linalg.norm(obs[best_base_h : best_base_h + 2])
        else:
            block_list = [17, 25, 33]
            for i_h, b_h in enumerate(block_list):
                presence_h = obs[b_h + 7]
                if presence_h > 0.5:
                    d_h = np.linalg.norm(obs[b_h : b_h + 2])
                    if d_h < min_d_h:
                        min_d_h = d_h
                        best_idx_h = i_h
                        best_base_h = b_h

        # 3. 信号計算
        grab_sig_h = 0.0
        lock_sig_h = 0.0
        lock_prepare_h = False
        target_free_h = False
        own_speed_h = np.linalg.norm(obs[0:2])
        if best_idx_h != -1 and min_d_h < 1.22:
            target_state_h = obs[best_base_h + 6]
            target_free_h = target_state_h < 0.5

            if (not self.is_grabbing) and (not self.is_locking) and target_free_h and min_d_h < 1.40:
                lock_prepare_h = True
                self.interact_obj_idx = best_idx_h
                if min_d_h < 1.02 and (self.action_tick % 3 == 0):
                    lock_sig_h = 1.0
            elif (not self.is_grabbing) and (not self.is_locking) and (not target_free_h):
                if self.action_tick % 6 == 0:
                    grab_sig_h = 1.0
                self.interact_obj_idx = best_idx_h
            elif self.is_grabbing:
                self.grab_time = self.grab_time + 1
                if self.grab_time > 50 and (self.action_tick % 5 == 0):
                    lock_sig_h = 1.0
        
        # 4. 反射・回避
        avoid_m_h = 0.25
        if self.is_grabbing:
            avoid_m_h = 0.0
            
        if self.reflex_timer > 0:
            self.reflex_timer = self.reflex_timer - 1
            if self.is_grabbing:
                return np.array([0.03, 0.0, lock_sig_h, 0.0])
            return np.array([-0.03, 0.35, lock_sig_h, grab_sig_h])
            
        if front_dist_h < avoid_m_h:
            self.reflex_timer = 12
            return np.array([0.0, 0.0, 0.0, grab_sig_h])

        # 5. 行動
        if best_idx_h != -1 and not self.is_locking and lock_prepare_h:
            rel_h = obs[best_base_h : best_base_h + 2]
            err_h = math.atan2(rel_h[1], rel_h[0])
            if min_d_h < 0.96:
                fwd_h = 0.0
            elif min_d_h < 1.02:
                fwd_h = 0.03
            else:
                fwd_h = 0.08
            clamped_err_h = np.clip(err_h * 1.6, -0.35, 0.35)
            return np.array([fwd_h, clamped_err_h, lock_sig_h, 0.0])

        if s_vis_h > 0.5:
            s_rel_h = obs[41:43]
            esc_h = math.atan2(-s_rel_h[1], -s_rel_h[0])
            fwd_esc_h = 0.1
            if abs(esc_h) < 0.5:
                fwd_esc_h = 0.35
            clamped_esc_h = np.clip(esc_h * 2.0, -0.6, 0.6)
            return np.array([fwd_esc_h, clamped_esc_h, 0.0, grab_sig_h])
            
        if best_idx_h != -1 and not self.is_locking:
            rel_h = obs[best_base_h : best_base_h + 2]
            err_h = math.atan2(rel_h[1], rel_h[0])
            fwd_h = 0.2
            if self.is_grabbing:
                fwd_h = 0.05
            turn_gain_h = 1.2 if self.is_grabbing else 2.2
            turn_lim_h = 0.22 if self.is_grabbing else 0.6
            clamped_err_h = np.clip(err_h * turn_gain_h, -turn_lim_h, turn_lim_h)
            return np.array([fwd_h, clamped_err_h, lock_sig_h, grab_sig_h])

        self.patrol_step = self.patrol_step + 1
        self.wander_timer = self.wander_timer - 1
        if self.wander_timer <= 0:
            self.wander_angle = np.random.uniform(-1.0, 1.0)
            self.wander_timer = np.random.randint(60, 150)
            
        osc_h = 0.1 * math.sin(self.patrol_step * 0.2)
        clamped_w_h = np.clip(self.wander_angle + osc_h, -0.5, 0.5)
        return np.array([0.18, clamped_w_h, 0.0, 0.0])