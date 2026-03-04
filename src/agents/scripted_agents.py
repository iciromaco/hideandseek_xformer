# scripted_agents.py v3.47
# 修正内容:
# 1. ターゲット識別型回避: 接近中、Lidarの壁回避がターゲット(箱)を避けないよう修正。
# 2. 強制制動ロジック: 1.1m以内かつ高速時に逆噴射(fwd < 0)をかけて精密停止。
# 3. ドッキング安定化: 0.9m以内で移動指令を完全遮断し、Grab/Lock条件を確実に満たす。
# 4. キツツキ運動の再発防止: Pivot/ReflexがApproachステートを阻害しないよう優先度調整。
# 5. PEP 8 & 極限展開: 1行1命令を徹底し、制動・停止・アクションの工程を垂直展開。

import numpy as np
import math


class RuleBasedSeeker:
    """ターゲットに正確に密着し、ロックを解除する高度な Seeker NPC。"""

    def __init__(self):
        self.reflex_timer = 0
        self.patrol_step = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.stop_wait_timer = 0
        self.last_known_rel_pos_x = 0.0
        self.last_known_rel_pos_y = 0.0
        self.memory_timer = 0
        self.last_l_sum = 0.0
        self.stuck_counter = 0
        self.current_state = "Idle"

    def _compute_navigation(self, lidar, current_rot, target_dist=999.0):
        """壁の斥力を計算。ただしターゲットが障害物より近い場合は無視する。"""
        f_min = np.min(lidar[0:5])
        
        # ターゲットが目前(1.2m以内)にあり、Lidar値とターゲット距離が近い場合、
        # それは壁ではなく目標物なので回避重みを下げる。
        mask_avoid = bool(target_dist < 1.2 and f_min > (target_dist - 0.2))
        
        if mask_avoid:
            av_w = 0.0
            avoid_t = 0.0
            fwd_lim = 1.0
        else:
            av_w = np.clip(1.0 - (f_min - 0.2) / 1.0, 0.0, 1.0)
            r_dist = lidar[1] + lidar[3] + lidar[5] + lidar[7]
            l_dist = lidar[2] + lidar[4] + lidar[6] + lidar[8]
            side_diff = r_dist - l_dist
            t_gain = 7.0 - (av_w * 3.0)
            avoid_t = side_diff * t_gain
            fwd_lim = 1.0 - (av_w * 0.9)
            
        return av_w, avoid_t, fwd_lim

    def _find_best_open_angle(self, lidar, current_rot):
        """Lidarから最も安全な進行方向を特定。"""
        degs = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
        best_idx = np.argmax(lidar)
        rel_rad = np.deg2rad(degs[best_idx])
        return current_rot + rel_rad

    def get_action(self, obs):
        self.patrol_step += 1
        lidar = obs[5:17]
        vx, vy = obs[0], obs[1]
        v_mag = math.sqrt(vx**2 + vy**2)
        
        # 視認情報
        h1_s = bool(obs[47] > 0.5)
        h2_s = bool(obs[54] > 0.5)
        
        # ロック対象 (接近距離を計算)
        locked_targets = []
        if obs[24] > 0.5 and obs[23] > 0.5:
            locked_targets.append((obs[17], obs[18]))
        if obs[32] > 0.5 and obs[31] > 0.5:
            locked_targets.append((obs[25], obs[26]))
        if obs[40] > 0.5 and obs[39] > 0.5:
            locked_targets.append((obs[33], obs[34]))
            
        t_dist_near = 999.0
        if len(locked_targets) > 0:
            tx, ty = min(locked_targets, key=lambda t: t[0]**2 + t[1]**2)
            t_dist_near = math.sqrt(tx**2 + ty**2)

        # 物理ナビゲーション情報の計算
        av_w, av_t, f_lim = self._compute_navigation(lidar, obs[2], t_dist_near)

        # スタック判定
        c_l_sum = np.sum(lidar)
        l_diff = abs(c_l_sum - self.last_l_sum)
        if l_diff < 0.001 and v_mag < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_l_sum = c_l_sum

        # 巡航速度
        p_speed = 0.85 if self.patrol_step < 350 else 0.98
        fwd, trn, lck, grb = 0.0, 0.0, 0.0, 0.0

        # --- 意思決定レイヤー (優先順位を整理) ---

        # 1. スタック回復
        if self.stuck_counter > 30 and self.reflex_timer <= 0:
            self.reflex_timer = 15
            self.wander_angle = self._find_best_open_angle(lidar, obs[2])

        if self.reflex_timer > 0:
            self.current_state = "Reflex"
            self.reflex_timer -= 1
            fwd = -0.35
            diff_r = self.wander_angle - obs[2] + np.pi
            a_err_r = (diff_r % (2.0 * np.pi)) - np.pi
            trn = np.clip(a_err_r * 8.0, -0.9, 0.9)

        # 2. ロック解除接近 (精密停止ロジック)
        elif len(locked_targets) > 0:
            self.current_state = "Unlocking"
            ox, oy = min(locked_targets, key=lambda t: t[0]**2 + t[1]**2)
            dist = math.sqrt(ox**2 + oy**2)
            a_err = math.atan2(oy, ox)
            
            # 旋回制御
            trn = np.clip(a_err * 4.5, -0.85, 0.85)
            
            # 停止制御: 1.1mから減速開始。0.9m以下で強制ブレーキ
            if dist < 0.88:
                fwd = 0.0
                if v_mag < 0.05:
                    self.stop_wait_timer += 1
                else:
                    # まだ動いているなら逆噴射
                    fwd = -0.15
                    self.stop_wait_timer = 0
                    
                if self.stop_wait_timer > 6:
                    lck, self.stop_wait_timer = 1.0, 0
            else:
                # 接近速度: 距離に比例させる
                fwd = np.clip((dist - 0.85) * 1.5, 0.0, 0.45)

        # 3. 追跡
        elif h1_s or h2_s:
            self.current_state = "Chasing"
            h1_d = math.sqrt(obs[41]**2 + obs[42]**2)
            h2_d = math.sqrt(obs[48]**2 + obs[49]**2)
            tx, ty = (obs[41], obs[42]) if (h1_s and not h2_s) or (h1_s and h1_d < h2_d) else (obs[48], obs[49])
            self.last_known_rel_pos_x, self.last_known_rel_pos_y = tx, ty
            self.memory_timer = 150
            a_err = math.atan2(ty, tx)
            trn = np.clip((a_err * 7.0 * (1.0 - av_w)) + (av_t * av_w), -0.95, 0.95)
            fwd = p_speed * f_lim

        # 4. パトロール / 衝突回避
        elif np.min(lidar[0:3]) < 0.15:
            self.current_state = "Pivot"
            fwd = 0.0
            tgt_ang = self._find_best_open_angle(lidar, obs[2])
            diff_p = tgt_ang - obs[2] + np.pi
            a_err_p = (diff_p % (2.0 * np.pi)) - np.pi
            trn = np.clip(a_err_p * 10.0, -1.0, 1.0)
            if np.min(lidar[0:3]) > 0.6:
                self.wander_angle = tgt_ang

        else:
            self.current_state = "Patrol"
            self.wander_timer -= 1
            if self.wander_timer <= 0 or av_w > 0.8:
                self.wander_angle = self._find_best_open_angle(lidar, obs[2])
                self.wander_timer = np.random.randint(80, 160)
            diff = self.wander_angle - obs[2] + np.pi
            a_err = (diff % (2.0 * np.pi)) - np.pi
            trn = np.clip(a_err * 4.5 + av_t * av_w, -0.85, 0.85)
            fwd = p_speed * f_lim

        # Ramp Assist
        if obs[39] > 0.5:
            rx, ry = obs[33], obs[34]
            if math.sqrt(rx**2 + ry**2) < 1.6 and abs(math.atan2(ry, rx)) < 0.5:
                fwd = max(fwd, 0.65)
        return np.array([fwd, trn, lck, grb])


class RuleBasedHider:
    """箱をターゲットとして精密に接近し、確実に把持・固定する Hider NPC。"""

    def __init__(self):
        self.reflex_timer = 0
        self.is_grabbing = False
        self.target_id = -1
        self.patrol_step = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.last_l_sum = 0.0
        self.stuck_counter = 0
        self.grab_time = 0
        self.current_state = "Idle"

    def _compute_navigation(self, lidar, target_dist=999.0):
        """Hider用のターゲット考慮型ナビゲーション。"""
        f_min = np.min(lidar[0:5])
        
        # ターゲットが近い場合は壁回避を抑える
        mask_avoid = bool(target_dist < 1.2 and f_min > (target_dist - 0.2))
        
        if mask_avoid:
            av_w = 0.0
            av_t = 0.0
            f_lim = 1.0
        else:
            av_w = np.clip(1.0 - (f_min - 0.3) / 0.9, 0.0, 1.0)
            r_dist = np.sum(lidar[1:6:2])
            l_dist = np.sum(lidar[2:7:2])
            av_t = (r_dist - l_dist) * (8.0 - (av_w * 4.0))
            f_lim = 1.0 - (av_w * 0.85)
            
        return av_w, av_t, f_lim

    def _find_best_open_angle(self, lidar, current_rot):
        degs = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
        max_idx = np.argmax(lidar)
        return current_rot + np.deg2rad(degs[max_idx])

    def get_action(self, obs):
        self.patrol_step += 1
        lidar = obs[5:17].copy()
        if self.is_grabbing:
            lidar[0:3] = np.clip(lidar[0:3] - 0.06, 0.0, 1.0)
        vx, vy = obs[0], obs[1]
        v_mag = math.sqrt(vx**2 + vy**2)
        
        # ターゲット選定
        if self.target_id == -1:
            min_d, best_i = 999.0, -1
            for i, b in enumerate([17, 25, 33]):
                if obs[b+6] > 0.5 and obs[b+7] < 0.5:
                    d = math.sqrt(obs[b]**2 + obs[b+1]**2)
                    if d < min_d: min_d, best_i = d, i
            self.target_id = best_i

        t_dist = 999.0
        if self.target_id != -1:
            b_base = 17 + self.target_id * 8
            t_dist = math.sqrt(obs[b_base]**2 + obs[b_base+1]**2)

        av_w, av_t, f_lim = self._compute_navigation(lidar, t_dist)
        
        # スタック判定
        c_l_sum = np.sum(lidar)
        if abs(c_l_sum - self.last_l_sum) < 0.001 and v_mag < 0.05:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_l_sum = c_l_sum
        
        s_vis = bool(obs[47] > 0.5)
        rel_sx, rel_sy = obs[41], obs[42]
        fwd, trn, lck, grb = 0.0, 0.0, 0.0, 0.0

        # 1. スタックリカバリー
        if self.stuck_counter > 30 and self.reflex_timer <= 0:
            if self.is_grabbing:
                lck, grb, self.is_grabbing = 1.0, 1.0, False
            self.reflex_timer = 15
            self.wander_angle = self._find_best_open_angle(lidar, obs[2])

        if self.reflex_timer > 0:
            self.current_state = "Reflex"
            self.reflex_timer -= 1
            fwd = -0.35
            diff_r = self.wander_angle - obs[2] + np.pi
            trn = np.clip(((diff_r + np.pi) % (2.0 * np.pi) - np.pi) * 8.0, -0.9, 0.9)

        # 2. Seeker逃走 (最高優先)
        elif s_vis:
            self.current_state = "Escape"
            self.target_id = -1
            if self.is_grabbing:
                grb, self.is_grabbing = 1.0, False
            e_rad = math.atan2(-rel_sy, -rel_sx)
            trn = np.clip(e_rad * 4.5 + av_t * av_w, -0.98, 0.98)
            fwd = 0.8 * f_lim

        # 3. 箱への接近・精密停止
        elif self.target_id != -1:
            b_idx = 17 + self.target_id * 8
            if obs[b_idx+6] < 0.5 or obs[b_idx+7] > 0.5:
                self.target_id = -1
            else:
                self.current_state = "Approach"
                tx, ty = obs[b_idx], obs[b_idx+1]
                dist = math.sqrt(tx**2 + ty**2)
                a_err = math.atan2(ty, tx)
                trn = np.clip(a_err * 5.0, -0.95, 0.95)
                
                # 接近と制動
                if dist < 0.88:
                    fwd = 0.0
                    if v_mag < 0.05:
                        self.grab_time += 1
                    else:
                        fwd = -0.2 # ブレーキ
                        self.grab_time = 0
                        
                    if self.grab_time > 5:
                        # 条件成立でアクション
                        if self.patrol_step < 350 and not self.is_grabbing:
                            grb, self.is_grabbing = 1.0, True
                        else:
                            lck = 1.0
                        self.target_id, self.grab_time = -1, 0
                else:
                    fwd = np.clip((dist - 0.82) * 2.0, 0.0, 0.65)

        # 4. 正面衝突の「停止旋回」
        elif np.min(lidar[0:3]) < 0.15:
            self.current_state = "Pivot"
            fwd = 0.0
            tgt_ang = self._find_best_open_angle(lidar, obs[2])
            diff_p = tgt_ang - obs[2] + np.pi
            a_err_p = (diff_p % (2.0 * np.pi)) - np.pi
            trn = np.clip(a_err_p * 10.0, -1.0, 1.0)
            if np.min(lidar[0:3]) > 0.6:
                self.wander_angle = tgt_ang

        # 5. 通常パトロール
        else:
            self.current_state = "Patrol"
            self.wander_timer -= 1
            if self.wander_timer <= 0 or av_w > 0.7:
                self.wander_angle = self._find_best_open_angle(lidar, obs[2])
                self.wander_timer = np.random.randint(60, 150)
            diff = self.wander_angle - obs[2] + np.pi
            a_err = (diff % (2.0 * np.pi)) - np.pi
            trn = np.clip(a_err * 4.0 + av_t * av_w, -0.85, 0.85)
            fwd = 0.65 * f_lim

        # Ramp
        if obs[39] > 0.5:
            rx, ry = obs[33], obs[34]
            if math.sqrt(rx**2 + ry**2) < 1.6 and abs(math.atan2(ry, rx)) < 0.5:
                fwd = max(fwd, 0.65)
        return np.array([fwd, trn, lck, grb])