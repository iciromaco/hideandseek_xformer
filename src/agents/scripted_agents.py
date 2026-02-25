# scripted_agents.py v2.27
# 演習第26回：【論理等価性復元 ＆ 高速パルス制御統合版】
# 
# 修正内容:
# 1. 欠落ロジックの完全復元: 
#    - Seeker: 視界から消えた後の「記憶追跡（Memory）」と「スタック回復」を再実装。
#    - Hider: 敵視認時の「逃走（Flee）」ロジックを復元。
# 2. 最新制御方式の維持:
#    - 環境側のトグル仕様に対応した「立ち上がりパルス信号」を継続採用。
# 3. 可読性と速度の両立:
#    - ヘルパーメソッド化を避け、1行1命令のフラットな記述で実行速度を担保。

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
        
        # 記憶ロジック用
        self.last_known_rel_pos = None
        self.memory_timer = 0
        
        # スタック検知用
        self.last_lidar = np.zeros(12)
        self.stuck_counter = 0
        
    def get_action(self, obs):
        lidar = obs[5:17]
        front_dist = min(lidar[0], lidar[1], lidar[2])
        
        # 1. スタック検知
        lidar_diff = np.linalg.norm(lidar - self.last_lidar)
        if lidar_diff < 0.0005:
            self.stuck_counter = self.stuck_counter + 1
        else:
            self.stuck_counter = 0
        self.last_lidar = lidar.copy()

        # ターゲット情報の抽出
        h1_vis, h2_vis, ramp_vis = obs[47], obs[54], obs[40]
        ramp_rel = obs[33:35]
        ramp_dist = np.linalg.norm(ramp_rel) if ramp_vis > 0.5 else 999.0

        # 2. 把持パルス信号の計算 (状態変化時のみ 1.0)
        grab_signal = 0.0
        
        # 敵を見つけたら即座に道具を離す
        if h1_vis > 0.5 or h2_vis > 0.5:
            if self.is_grabbing:
                grab_signal = 1.0
                self.is_grabbing = False
        # 道具が近くにあり、敵がいないなら掴む
        elif ramp_vis > 0.5 and ramp_dist < 1.15:
            if not self.is_grabbing:
                grab_signal = 1.0
                self.is_grabbing = True

        # 3. 反射・回避行動
        avoid_th = 0.22 if not self.is_grabbing else 0.0
        if self.reflex_timer > 0:
            self.reflex_timer = self.reflex_timer - 1
            return np.array([-0.1, 0.7, 0.0, grab_signal])

        if front_dist < avoid_th or self.stuck_counter > 50:
            self.reflex_timer = 20
            self.stuck_counter = 0
            return np.array([0.0, 0.0, 0.0, grab_signal])

        # 4. メイン行動ロジック (Chase > Memory > Tool > Patrol)
        
        # A. 直接追跡
        target_rel = None
        if h1_vis > 0.5:
            target_rel = obs[41:43]
        elif h2_vis > 0.5:
            target_rel = obs[48:50]
            
        if target_rel is not None:
            self.last_known_rel_pos = target_rel
            self.memory_timer = 150 # 150ステップ記憶
            err = math.atan2(target_rel[1], target_rel[0])
            return np.array([0.4, np.clip(err * 2.5, -0.7, 0.7), 0.0, grab_signal])

        # B. 記憶追跡 (見失った場所へ向かう)
        if self.memory_timer > 0:
            self.memory_timer = self.memory_timer - 1
            m_rel = self.last_known_rel_pos
            m_err = math.atan2(m_rel[1], m_rel[0])
            # 記憶地点へ近づく
            dist_m = np.linalg.norm(m_rel)
            fwd_m = 0.3 if dist_m > 0.5 else 0.0
            return np.array([fwd_m, np.clip(m_err * 2.0, -0.6, 0.6), 0.0, grab_signal])

        # C. 道具排除
        if ramp_vis > 0.5:
            r_err = math.atan2(ramp_rel[1], ramp_rel[0])
            if self.is_grabbing:
                # 掴んでいるなら引きずる
                return np.array([-0.3, 0.4, 0.0, grab_signal])
            return np.array([0.3, np.clip(r_err * 2.2, -0.7, 0.7), 0.0, grab_signal])

        # D. 巡回
        self.patrol_step = self.patrol_step + 1
        self.wander_timer = self.wander_timer - 1
        if self.wander_timer <= 0:
            dirs = [0, 45, -45, 90, -90, 180]
            self.wander_angle = np.deg2rad(dirs[np.random.randint(6)])
            self.wander_timer = np.random.randint(100, 200)
            
        wiggle = 0.2 * math.sin(self.patrol_step * 0.1)
        turn = np.clip(self.wander_angle + wiggle, -0.6, 0.6)
        return np.array([0.18, turn, 0.0, grab_signal])

class RuleBasedHider:
    """敵から逃げ、道具を確保してロックするハイダー"""
    def __init__(self):
        self.reflex_timer = 0
        self.is_grabbing = False
        self.is_locking = False
        self.grab_time = 0
        self.wander_timer = 0
        self.patrol_step = 0
        self.wander_angle = 0.0

    def get_action(self, obs):
        lidar = obs[5:17]
        front_dist = min(lidar[0], lidar[1], lidar[2])
        
        # 敵の視認
        s_vis = obs[47]
        
        # 最も近い道具の探索
        best_base, min_d = -1, 999.0
        for b in [17, 25, 33]:
            if obs[b+7] > 0.5:
                # ロックされていない、または自分がロック中のものを優先
                d = np.linalg.norm(obs[b:b+2])
                if d < min_d:
                    min_d = d
                    best_base = b

        # 信号パルスの計算
        grab_sig, lock_sig = 0.0, 0.0
        
        if best_base != -1 and min_d < 1.15:
            if not self.is_grabbing and not self.is_locking:
                grab_sig = 1.0
                self.is_grabbing = True
            elif self.is_grabbing:
                self.grab_time = self.grab_time + 1
                if self.grab_time > 120:
                    # ロックして、把持を解除するパルスを同時送信
                    lock_sig = 1.0
                    grab_sig = 1.0
                    self.is_locking = True
                    self.is_grabbing = False
        
        # 回避ロジック
        avoid = 0.25 if not self.is_grabbing else 0.0
        if self.reflex_timer > 0:
            self.reflex_timer = self.reflex_timer - 1
            return np.array([-0.05, 0.5, lock_sig, grab_sig])

        if front_dist < avoid:
            self.reflex_timer = 12
            return np.array([0.0, 0.0, 0.0, grab_sig])

        # 行動選択 (Flee > Tool > Patrol)
        
        # A. 敵から逃げる
        if s_vis > 0.5:
            s_rel = obs[41:43]
            # 敵と反対方向を向く
            escape_angle = math.atan2(-s_rel[1], -s_rel[0])
            fwd = 0.35 if abs(escape_angle) < 0.5 else 0.1
            return np.array([fwd, np.clip(escape_angle * 2.0, -0.6, 0.6), 0.0, grab_sig])

        # B. 道具へ向かう / 操作
        if best_base != -1 and not self.is_locking:
            rel = obs[best_base:best_base+2]
            err = math.atan2(rel[1], rel[0])
            fwd_speed = -0.25 if self.is_grabbing else 0.2
            return np.array([fwd_speed, np.clip(err * 2.2, -0.6, 0.6), lock_sig, grab_sig])

        # C. 巡回
        self.patrol_step = self.patrol_step + 1
        self.wander_timer = self.wander_timer - 1
        if self.wander_timer <= 0:
            self.wander_angle = np.random.uniform(-1.0, 1.0)
            self.wander_timer = np.random.randint(60, 150)
            
        turn = np.clip(self.wander_angle + 0.1 * math.sin(self.patrol_step * 0.2), -0.5, 0.5)
        return np.array([0.18, turn, 0.0, 0.0])