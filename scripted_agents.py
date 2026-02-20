# scripted_agents.py v1.19
# 演習第25回：物理駆動型ルールベース・シーカー（緊急回避・スタック完全防止版）
# 
# 修正履歴:
# v1.18: 直進性強化。
# v1.19: 壁へのスタック防止。真正面が詰まった場合の後退(Reverse)と急旋回を実装。

import numpy as np
import math

# --- 共通定数 ---
FOV_DEG = 135
FOV_HALF_RAD = FOV_DEG * 0.5 * math.pi / 180.0
FOV_COS_HALF = math.cos(FOV_HALF_RAD)
MAX_SENSE_DIST = 15.0

# 物理出力
MAX_CTRL = 0.4  
CRUISE_FWD = 0.3 # 巡航速度

class RuleBasedSeeker:
    def __init__(self):
        self.last_target_pos = None
        self.target_lost_timer = 0
        self.status = "IDLE"
        self.last_reason = "NONE"
        self.explore_timer = 0
        self.explore_action = np.array([0.0, 0.0])
        self.debug_text = ""
        
    def get_action(self, obs, engine, current_pos, current_heading, target_positions, body_exclude):
        best_target = None
        min_dist = float('inf')
        found_name = "None"
        
        # 1. 視認判定
        for i, t_pos in enumerate(target_positions):
            visible, dist, reason = self._check_visibility(engine, current_pos, current_heading, t_pos, body_exclude)
            if i == 0: self.last_reason = reason
            if visible and dist < min_dist:
                min_dist = dist
                best_target = t_pos
                found_name = f"H{i+1}"

        # 2. 状態遷移
        if best_target is not None:
            self.status = "CHASE"
            self.last_target_pos = best_target.copy()
            self.target_lost_timer = 0
            self.explore_timer = 0
            action = self._compute_pursuit_action(current_pos, current_heading, best_target)
        elif self.last_target_pos is not None and self.target_lost_timer < 60:
            self.status = "TRACK"
            self.target_lost_timer += 1
            action = self._compute_pursuit_action(current_pos, current_heading, self.last_target_pos)
        else:
            self.status = "EXPLR"
            action = self._compute_persistent_explore(obs, current_heading)

        self.debug_text = f"{self.status}|Saw:{found_name}|Rsn:{self.last_reason}"
        return action

    def _check_visibility(self, engine, viewer_pos, viewer_heading, target_pos, body_exclude):
        diff = target_pos - viewer_pos
        dist = np.linalg.norm(diff)
        if dist > MAX_SENSE_DIST: return False, dist, "FAR"
        view_dir = np.array([math.cos(viewer_heading), math.sin(viewer_heading)])
        rel_dir = diff / (dist + 1e-8)
        if np.dot(view_dir, rel_dir) < FOV_COS_HALF: return False, dist, "FOV_OUT"
        eye_pos = viewer_pos + view_dir * 0.45
        if not engine.is_visible(eye_pos, target_pos, body_exclude=body_exclude):
            return False, dist, "OCCLUD"
        return True, dist, "VISIBLE"

    def _compute_pursuit_action(self, pos, heading, target):
        diff = target - pos
        dist = np.linalg.norm(diff)
        target_angle = math.atan2(diff[1], diff[0])
        angle_err = (target_angle - heading + math.pi) % (2 * math.pi) - math.pi
        
        turn = np.clip(angle_err * 4.0, -MAX_CTRL, MAX_CTRL)
        # 追跡中も激突しそうなら止まる
        forward = CRUISE_FWD if abs(angle_err) < 0.5 and dist > 0.5 else 0.1
        return np.array([forward, turn])

    def _compute_persistent_explore(self, obs, heading):
        """
        探索：壁を避ける、詰まったら下がる。
        """
        lidar = obs[5:17]
        
        # 緊急避難：真正面が極端に短い(0.6m以下)
        if lidar[0] < 0.6:
            self.status = "ESCAPE"
            # 左右どちらか空いている方へ急旋回しながら、少し下がる
            turn_dir = 1.0 if lidar[7] > lidar[8] else -1.0
            self.explore_action = np.array([-0.15, turn_dir * MAX_CTRL])
            self.explore_timer = 10 # 短い回避
        
        # 通常の探索更新
        elif self.explore_timer <= 0 or lidar[0] < 1.5:
            self.status = "EXPLR"
            angles = [0, 15, -15, 30, -30, 45, -45, 90, -90]
            # 正面ボーナスを入れつつ、最も開けた方向を探す
            lidar_weighted = lidar[0:9].copy()
            lidar_weighted[0] *= 1.3
            best_idx = np.argmax(lidar_weighted)
            
            target_rel_angle = np.deg2rad(angles[best_idx])
            turn = np.clip(target_rel_angle * 2.0, -MAX_CTRL, MAX_CTRL)
            
            # 正面が空いているなら前進、そうでなければ旋回に専念
            forward = CRUISE_FWD if lidar[0] > 1.2 else 0.05
            self.explore_action = np.array([forward, turn])
            self.explore_timer = 30
            
        self.explore_timer -= 1
        return self.explore_action