# scripted_agents.py v2.37
# 演習第26回：【論理整合性・極限展開 ＆ Lock準備挙動(lock_prepare) ＆ 55次元完全同期版】
# 
# 修正内容:
# 1. 1行1命令（1-line-1-command）の徹底:
#    - すべてのアクション計算、インデックス参照、条件判定を独立した行に展開。
#    - リスト内包表記や三項演算子を排除し、ロジックのステップを物理的に可視化。
# 2. Lock意図仕様の刷新:
#    - 近距離（<1.22m）のフリー対象に対し、接近・減速しつつ周期的な Lock パルスを送る lock_prepare 状態を実装。
#    - Lock 準備中は Grab 信号を抑制し、意図しない把持を防止する仕様を厳守。
# 3. 把持中 Lock への移行: 
#    - 保持時間が一定値（50ステップ）を超えた場合に、Lock パルスを送信して固定へ移行。
# 4. 幽霊把持（Ghost Grasp）解消の継承: 
#    - 対象との距離超過（>1.6m）時にフラグ解除だけでなく、強制解除パルス(1.0)を送信。

import numpy as np
import math

class RuleBasedSeeker:
    """敵を追い、邪魔な道具を解除するシーカー"""
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
        # 1. Lidar データの取得とスタック判定
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

        # 2. インデックス同期 (55次元カテゴリカル)
        # 40: ramp_vis, 47: h1_vis, 54: h2_vis
        h1_vis_s = obs[47]
        h2_vis_s = obs[54]
        ramp_vis_s = obs[40]
        
        # 33-34: ramp_rel_pos
        ramp_rel_s = obs[33:35]
        r_dx = ramp_rel_s[0]
        r_dy = ramp_rel_s[1]
        ramp_dist_s = math.sqrt(r_dx * r_dx + r_dy * r_dy)
        
        if ramp_vis_s <= 0.5:
            ramp_dist_s = 999.0

        grab_signal_s = 0.0
        
        # 3. 把持状態の更新と解除パルス (幽霊把持防止)
        if self.is_grabbing:
            if ramp_dist_s > 1.6:
                self.is_grabbing = False
                grab_signal_s = 1.0

        # 4. インタラクション判定
        if h1_vis_s > 0.5 or h2_vis_s > 0.5:
            # 敵を見つけたら道具を離す
            if self.is_grabbing:
                grab_signal_s = 1.0
                self.is_grabbing = False
        elif ramp_vis_s > 0.5 and ramp_dist_s < 1.15:
            # 道具に近づいたら掴む
            if not self.is_grabbing:
                grab_signal_s = 1.0
                self.is_grabbing = True

        # 5. 反射回避
        avoid_th_s = 0.22
        if self.is_grabbing:
            avoid_th_s = 0.0
            
        if self.reflex_timer > 0:
            self.reflex_timer = self.reflex_timer - 1
            act_reflex = np.array([-0.1, 0.7, 0.0, grab_signal_s])
            return act_reflex
            
        if front_dist_s < avoid_th_s or self.stuck_counter > 50:
            self.reflex_timer = 20
            self.stuck_counter = 0
            act_stuck = np.array([0.0, 0.0, 0.0, grab_signal_s])
            return act_stuck

        # 6. ターゲット追跡
        target_rel_s = None
        if h1_vis_s > 0.5:
            target_rel_s = obs[41:43]
        elif h2_vis_s > 0.5:
            target_rel_s = obs[48:50]
            
        if target_rel_s is not None:
            self.last_known_rel_pos = target_rel_s
            self.memory_timer = 150 
            t_x = target_rel_s[0]
            t_y = target_rel_s[1]
            err_s = math.atan2(t_y, t_x)
            steer_s = err_s * 2.5
            clamped_steer = np.clip(steer_s, -0.7, 0.7)
            act_chase = np.array([0.4, clamped_steer, 0.0, grab_signal_s])
            return act_chase
            
        # 記憶に基づく追跡
        if self.memory_timer > 0:
            self.memory_timer = self.memory_timer - 1
            m_rel_s = self.last_known_rel_pos
            m_x = m_rel_s[0]
            m_y = m_rel_s[1]
            m_err_s = math.atan2(m_y, m_x)
            m_dist_s = np.linalg.norm(m_rel_s)
            
            fwd_m_s = 0.3
            if m_dist_s <= 0.5:
                fwd_m_s = 0.0
                
            steer_m = m_err_s * 2.0
            clamped_m_steer = np.clip(steer_m, -0.6, 0.6)
            act_memory = np.array([fwd_m_s, clamped_m_steer, 0.0, grab_signal_s])
            return act_memory
            
        # 道具への接近
        if ramp_vis_s > 0.5:
            r_x = ramp_rel_s[0]
            r_y = ramp_rel_s[1]
            r_err_s = math.atan2(r_y, r_x)
            
            if self.is_grabbing:
                # 運搬中は少し回転
                act_ramp = np.array([-0.3, 0.4, 0.0, grab_signal_s])
                return act_ramp
                
            steer_r = r_err_s * 2.2
            clamped_r_steer = np.clip(steer_r, -0.7, 0.7)
            act_approach = np.array([0.3, clamped_r_steer, 0.0, grab_signal_s])
            return act_approach

        # 7. 巡回
        self.patrol_step = self.patrol_step + 1
        self.wander_timer = self.wander_timer - 1
        if self.wander_timer <= 0:
            dirs_s = [0, 45, -45, 90, -90, 180]
            idx_s = np.random.randint(6)
            angle_deg = dirs_s[idx_s]
            self.wander_angle = np.deg2rad(angle_deg)
            self.wander_timer = np.random.randint(100, 200)
            
        osc_s = 0.2 * math.sin(self.patrol_step * 0.1)
        steer_w = self.wander_angle + osc_s
        clamped_w_steer = np.clip(steer_w, -0.6, 0.6)
        act_wander = np.array([0.18, clamped_w_steer, 0.0, grab_signal_s])
        return act_wander

class RuleBasedHider:
    """Lock準備挙動(lock_prepare)を実装したハイダー"""
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
        
        # 1. Lidar とスタック判定
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
        
        # 2. 状態同期 ＆ 幽霊把持解消
        if self.interact_obj_idx != -1:
            base_h = 17 + self.interact_obj_idx * 8 
            # 6: interaction_state (0:None, 1:Locked, 2:Me, 3:Others)
            int_state_h = obs[base_h + 6]
            
            if int_state_h == 1.0:
                self.is_locking = True
                self.is_grabbing = False
            elif int_state_h == 2.0:
                self.is_locking = False
                self.is_grabbing = True
            else:
                self.is_locking = False
                self.is_grabbing = False
                
            o_rel_h = obs[base_h : base_h + 2]
            obj_dist_h = np.linalg.norm(o_rel_h)
            
            # 距離超過による強制解除
            if self.is_grabbing and obj_dist_h > 1.6:
                self.is_grabbing = False
                self.interact_obj_idx = -1
                act_release_d = np.array([0.0, 0.0, 0.0, 1.0])
                return act_release_d
            
            # スタック検知時の強制解除
            if self.is_grabbing and self.stuck_counter > 60:
                self.is_grabbing = False
                self.interact_obj_idx = -1
                act_release_s = np.array([0.0, 0.0, 0.0, 1.0])
                return act_release_s

            if (not self.is_grabbing) and (not self.is_locking):
                self.interact_obj_idx = -1
                self.grab_time = 0

        # 3. 道具の探索
        best_base_h = -1
        best_idx_h = -1
        min_d_h = 999.0
        
        if self.interact_obj_idx != -1:
            best_idx_h = self.interact_obj_idx
            best_base_h = 17 + best_idx_h * 8
            min_d_h = np.linalg.norm(obs[best_base_h : best_base_h + 2])
        else:
            # 17: b1, 25: b2, 33: ramp
            block_indices = [17, 25, 33]
            for i_h, b_h in enumerate(block_indices):
                presence_flag = obs[b_h + 7]
                if presence_flag > 0.5:
                    d_h = np.linalg.norm(obs[b_h : b_h + 2])
                    if d_h < min_d_h:
                        min_d_h = d_h
                        best_idx_h = i_h
                        best_base_h = b_h

        # 4. 信号計算（Lock意図仕様：lock_prepare）
        grab_sig_h = 0.0
        lock_sig_h = 0.0
        lock_prepare_h = False
        
        if best_idx_h != -1 and min_d_h < 1.22:
            target_st_h = obs[best_base_h + 6]
            
            # フリー対象への Lock 意図
            if (not self.is_grabbing) and (not self.is_locking) and target_st_h < 0.5:
                lock_prepare_h = True
                self.interact_obj_idx = best_idx_h
                # 至近距離で周期的な Lock パルス
                if min_d_h < 1.02:
                    mod_tick = self.action_tick % 3
                    if mod_tick == 0:
                        lock_sig_h = 1.0
                        
            # 他者の固定・把持への介入パルス
            elif (not self.is_grabbing) and (not self.is_locking) and target_st_h >= 1.0:
                mod_grab = self.action_tick % 6
                if mod_grab == 0:
                    grab_sig_h = 1.0
                self.interact_obj_idx = best_idx_h
                
            # 把持中からの Lock 移行
            elif self.is_grabbing:
                self.grab_time = self.grab_time + 1
                if self.grab_time > 50:
                    mod_l = self.action_tick % 5
                    if mod_l == 0:
                        lock_sig_h = 1.0
        
        # 5. 反射・回避
        avoid_m_h = 0.25
        if self.is_grabbing:
            avoid_m_h = 0.0
            
        if self.reflex_timer > 0:
            self.reflex_timer = self.reflex_timer - 1
            if self.is_grabbing:
                act_r_g = np.array([0.03, 0.0, lock_sig_h, 0.0])
                return act_r_g
            act_r_h = np.array([-0.03, 0.35, lock_sig_h, grab_sig_h])
            return act_r_h
            
        if front_dist_h < avoid_m_h:
            self.reflex_timer = 12
            act_avoid = np.array([0.0, 0.0, 0.0, grab_sig_h])
            return act_avoid

        # 6. アクション決定（垂直展開）
        
        # [A] Lock準備挙動 (停止志向)
        if best_idx_h != -1 and not self.is_locking and lock_prepare_h:
            rel_h = obs[best_base_h : best_base_h + 2]
            err_h = math.atan2(rel_h[1], rel_h[0])
            
            # 段階的な減速
            if min_d_h < 0.96:
                f_h = 0.0
            elif min_d_h < 1.02:
                f_h = 0.03
            else:
                f_h = 0.08
                
            steer_h = err_h * 1.6
            clamped_steer_h = np.clip(steer_h, -0.35, 0.35)
            act_l_prep = np.array([f_h, clamped_steer_h, lock_sig_h, 0.0])
            return act_l_prep

        # [B] 敵視認時の逃走
        if s_vis_h > 0.5:
            s_rel_h = obs[41:43]
            # 逆方向にステアリング
            esc_h = math.atan2(-s_rel_h[1], -s_rel_h[0])
            
            if abs(esc_h) < 0.5:
                fwd_esc_h = 0.35
            else:
                fwd_esc_h = 0.1
                
            steer_esc = esc_h * 2.0
            clamped_esc = np.clip(steer_esc, -0.6, 0.6)
            act_escape = np.array([fwd_esc_h, clamped_esc, 0.0, grab_sig_h])
            return act_escape
            
        # [C] 道具への接近・運搬
        if best_idx_h != -1 and not self.is_locking:
            rel_h = obs[best_base_h : best_base_h + 2]
            err_h = math.atan2(rel_h[1], rel_h[0])
            
            if self.is_grabbing:
                f_h = 0.05
                turn_g = 1.2
                turn_l = 0.22
            else:
                f_h = 0.2
                turn_g = 2.2
                turn_l = 0.6
            
            steer_val = err_h * turn_g
            clamped_steer_val = np.clip(steer_val, -turn_l, turn_l)
            act_interact = np.array([f_h, clamped_steer_val, lock_sig_h, grab_sig_h])
            return act_interact

        # [D] 巡回・ランダムウォーク
        self.patrol_step = self.patrol_step + 1
        self.wander_timer = self.wander_timer - 1
        if self.wander_timer <= 0:
            self.wander_angle = np.random.uniform(-1.0, 1.0)
            self.wander_timer = np.random.randint(60, 150)
            
        osc_h = 0.1 * math.sin(self.patrol_step * 0.2)
        steer_w_h = self.wander_angle + osc_h
        clamped_w_h = np.clip(steer_w_h, -0.5, 0.5)
        act_wander_h = np.array([0.18, clamped_w_h, 0.0, 0.0])
        return act_wander_h