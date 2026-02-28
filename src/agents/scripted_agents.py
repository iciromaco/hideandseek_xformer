# scripted_agents.py v3.21
# 修正内容:
# 1. 追跡・逃走優先ロジック: Chasing/Escape中は壁回避バイアスを50%抑制。
# 2. 精密インタラクションの復旧: 道具接近時の停止判定距離を 1.05m から開始。
# 3. Lock頻度の向上: 固定待機時間を 120 から 80 ステップへ短縮。
# 4. 垂直展開の維持: 意思決定の全工程を独立した行で記述。

import numpy as np
import math


class RuleBasedSeeker:
    """壁を予見して回避し、Hiderを追跡する Seeker NPC。"""

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

    def get_action(self, obs):
        self.patrol_step += 1
        lidar = obs[5:17]
        front_min = np.min(lidar[0:3])

        # 左右スペース評価
        r_gap = lidar[1] + lidar[3] + lidar[5]
        l_gap = lidar[2] + lidar[4] + lidar[6]
        side_diff = r_gap - l_gap

        # 回避バイアス: 標準は 1.2m から
        avoid_w = np.clip(1.0 - (front_min / 1.2), 0.0, 1.0)
        avoid_torque = side_diff * 4.2

        # スタック判定
        curr_l_sum = np.sum(lidar)
        l_diff = abs(curr_l_sum - self.last_l_sum)

        if l_diff < 0.001:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_l_sum = curr_l_sum

        h1_vis = obs[47]
        h2_vis = obs[54]
        enemy_seen = bool(h1_vis > 0.5 or h2_vis > 0.5)

        r_rel_x = obs[33]
        r_rel_y = obs[34]
        r_dist = math.sqrt(r_rel_x**2 + r_rel_y**2)

        fwd = 0.0
        trn = 0.0
        lck = 0.0
        grb = 0.0

        # --- 精密インタラクションロジック ---
        if enemy_seen:
            # 追跡優先のため、持っているなら離す
            if self.is_grabbing:
                grb = 1.0
                self.is_grabbing = False
        elif obs[40] > 0.5 and r_dist < 1.15:
            if not self.is_grabbing:
                if r_dist < 0.9:
                    fwd = 0.0
                    grb = 1.0
                    self.is_grabbing = True
                else:
                    fwd = 0.15

        # --- ステートマシン判定 ---
        if self.reflex_timer > 0:
            self.current_state = "Reflex"
            self.reflex_timer -= 1
            fwd = -0.15
            trn = 0.72

        elif front_min < 0.12:
            self.current_state = "AvoidWall"
            self.reflex_timer = 15

        elif self.stuck_counter > 50:
            self.current_state = "StuckEscape"
            self.reflex_timer = 22
            self.stuck_counter = 0

        elif enemy_seen:
            self.current_state = "Chasing"
            self.memory_timer = 150
            if h1_vis > 0.5:
                tx, ty = obs[41], obs[42]
            else:
                tx, ty = obs[48], obs[49]
            self.last_known_rel_pos_x, self.last_known_rel_pos_y = tx, ty

            # 追跡時は壁回避よりもターゲット方向を優先
            # 回避バイアスを50%に抑制
            v_scale = 1.0 - (avoid_w * 0.2)
            fwd = 0.55 * v_scale
            c_err = math.atan2(ty, tx)
            # ブレンド計算
            trn_val = (c_err * 3.2) + (avoid_torque * avoid_w * 0.5)
            trn = np.clip(trn_val, -0.78, 0.78)

        elif self.memory_timer > 0:
            self.current_state = "MemorySearch"
            self.memory_timer -= 1
            mx, my = self.last_known_rel_pos_x, self.last_known_rel_pos_y
            fwd = 0.4
            m_err = math.atan2(my, mx)
            trn_val = (m_err * 2.4) + (avoid_torque * avoid_w)
            trn = np.clip(trn_val, -0.65, 0.65)

        else:
            self.current_state = "Patrol"
            self.wander_timer -= 1
            if self.wander_timer <= 0:
                angles = [0, 45, -45, 90, -90, 135, -135, 180]
                self.wander_angle = np.deg2rad(np.random.choice(angles))
                self.wander_timer = np.random.randint(150, 300)

            cur_rot = obs[2]
            a_err = (self.wander_angle - cur_rot + np.pi) % (2.0 * np.pi) - np.pi
            v_scale = 1.0 - (avoid_w * 0.5)
            fwd = 0.4 * v_scale
            trn = np.clip((a_err * 2.4) + (avoid_torque * avoid_w), -0.65, 0.65)

        return np.array([fwd, trn, lck, grb])


class RuleBasedHider:
    """壁を予見して逃げ回り、道具を操作する Hider NPC。"""

    def __init__(self):
        self.reflex_timer = 0
        self.is_grabbing = False
        self.is_locking = False
        self.interact_obj_idx = -1
        self.grab_time = 0
        self.patrol_step = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.last_l_sum = 0.0
        self.stuck_counter = 0
        self.current_state = "Idle"

    def get_action(self, obs):
        self.patrol_step += 1
        lidar = obs[5:17]
        front_min = np.min(lidar[0:3])
        s_vis = obs[47]

        # 回避操舵
        side_diff = (lidar[1]+lidar[3]+lidar[5]) - (lidar[2]+lidar[4]+lidar[6])
        avoid_w = np.clip(1.0 - (front_min / 1.4), 0.0, 1.0)
        avoid_t = side_diff * 4.8

        curr_l_sum = np.sum(lidar)
        l_diff = abs(curr_l_sum - self.last_l_sum)
        if l_diff < 0.001:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_l_sum = curr_l_sum

        # 道具探索
        min_d = 999.0
        t_id = -1
        for i, b in enumerate([17, 25, 33]):
            if obs[b+7] > 0.5:
                d = math.sqrt(obs[b]**2 + obs[b+1]**2)
                if d < min_d:
                    min_d, t_id = d, i

        fwd = 0.0
        trn = 0.0
        lck = 0.0
        grb = 0.0

        if t_id != -1 and min_d < 1.05:
            if not self.is_grabbing and not self.is_locking:
                if min_d < 0.9:
                    fwd, grb = 0.0, 1.0
                    self.is_grabbing, self.interact_obj_idx = True, t_id
                else:
                    fwd = 0.15
            elif self.is_grabbing:
                self.grab_time += 1
                # 固定待機時間を短縮して頻度向上
                if self.grab_time > 80:
                    lck, grb = 1.0, 1.0
                    self.is_locking, self.is_grabbing = True, False
                    self.grab_time = 0

        # --- 意思決定 ---
        if self.reflex_timer > 0:
            self.current_state, self.reflex_timer = "Reflex", self.reflex_timer - 1
            fwd, trn = -0.1, 0.65
        elif front_min < 0.12:
            self.current_state, self.reflex_timer = "AvoidWall", 14
        elif self.stuck_counter > 40:
            self.current_state, self.reflex_timer, self.stuck_counter = "StuckEscape", 20, 0
        elif s_vis > 0.5:
            self.current_state = "Escape"
            rel_sx, rel_sy = obs[41], obs[42]
            e_rad = math.atan2(-rel_sy, -rel_sx)
            # 逃走時も壁回避の干渉を50%に抑制
            fwd = 0.5 * (1.0 - avoid_w * 0.3)
            trn = np.clip(e_rad * 3.0 + (avoid_t * avoid_w * 0.5), -0.75, 0.75)
        elif t_id != -1 and not self.is_locking:
            if self.is_grabbing:
                self.current_state, fwd = "Carrying", 0.18
                trn = 0.12 * math.sin(self.patrol_step * 0.1)
            else:
                self.current_state = "Approach"
                base_idx = 17 + t_id * 8
                tx, ty = obs[base_idx], obs[base_idx + 1]
                a_err = math.atan2(ty, tx)
                fwd = 0.35 * (1.0 - avoid_w * 0.5)
                trn = np.clip(a_err * 2.4 + (avoid_t * avoid_w), -0.65, 0.65)
        else:
            self.current_state = "Patrol"
            self.wander_timer -= 1
            if self.wander_timer <= 0:
                self.wander_angle = np.random.uniform(-np.pi, np.pi)
                self.wander_timer = np.random.randint(150, 300)
            a_err = (self.wander_angle - obs[2] + np.pi) % (2.0 * np.pi) - np.pi
            fwd = 0.32 * (1.0 - avoid_w * 0.5)
            trn = np.clip(a_err * 2.4 + (avoid_t * avoid_w), -0.65, 0.65)

        return np.array([fwd, trn, lck, grb])