# src/agents/scripted_agents.py
# scripted_agents.py v5.44

import math

import numpy as np

from core.constants import L_SCALE, P_SCALE, R_SCALE


def _nearest_pushable(obs, idx):
    p_scale = P_SCALE
    best = None
    for obj in list(idx.B) + list(idx.RAMP):
        rel_x = float(obs[obj.REL_X]) * p_scale
        rel_y = float(obs[obj.REL_Y]) * p_scale
        dist = math.hypot(rel_x, rel_y)
        if dist > 2.1 or rel_x <= -0.15:
            continue
        if best is None or dist < best[0]:
            best = (dist, rel_x, obj)
    return best


def _interaction_buttons(obs, idx, interact_cooldown, interact_focus_steps, suppress=False):
    """近距離前方オブジェクトに対して lock/grab ボタンを生成する。"""
    if suppress:
        return 0.0, 0.0, max(interact_cooldown - 1, 0), max(interact_focus_steps - 2, 0)

    if interact_cooldown > 0:
        return 0.0, 0.0, interact_cooldown - 1, max(interact_focus_steps - 1, 0)

    best = _nearest_pushable(obs, idx)

    if best is None:
        return 0.0, 0.0, 0, max(interact_focus_steps - 1, 0)

    _, rel_x, obj = best
    is_locked = float(obs[obj.IS_LOCKED]) > 0.5
    is_moving = float(obs[obj.IS_MOVING]) > 0.5
    lidar_raw = obs[idx.LIDAR] * L_SCALE
    front_min = float(np.min(lidar_raw[idx.LIDAR_FRONT_IDX]))

    if rel_x > 0.05:
        interact_focus_steps = min(interact_focus_steps + 1, 70)
    else:
        interact_focus_steps = max(interact_focus_steps - 1, 0)

    # まずは押して動かす挙動を優先し、一定時間接触が続いた場合のみ相互作用する
    if interact_focus_steps < 30:
        return 0.0, 0.0, 0, interact_focus_steps

    if is_locked:
        if rel_x > 0.05 and front_min < 0.42:
            return 1.0, 0.0, 22, 0
        return 0.0, 0.0, 0, interact_focus_steps

    if is_moving:
        if front_min < 0.35:
            return 1.0, 0.0, 22, 0
        return 0.0, 0.0, 0, interact_focus_steps

    if not is_moving:
        return 0.0, 1.0, 20, 0
    return 0.0, 0.0, 0, interact_focus_steps


class RuleBasedSeeker:
    """目的地への方位偏差を計算し、壁を避けながら追従する Seeker。"""

    def __init__(self):
        self.reflex_timer = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.last_known_rel_pos_x = 0.0
        self.last_known_rel_pos_y = 0.0
        self.memory_timer = 0
        self.stuck_counter = 0
        self.escape_turn_dir = 1.0
        self.escape_fwd_dir = -1.0
        self.interact_cooldown = 0
        self.interact_focus_steps = 0

    def get_action(self, obs, idx):
        p_scale = P_SCALE
        lidar_raw = obs[idx.LIDAR] * L_SCALE
        cur_rot = obs[idx.SELF.ROT] * R_SCALE

        front_min = np.min(lidar_raw[idx.LIDAR_FRONT_IDX])
        back_min = np.min(lidar_raw[idx.LIDAR_BACK_IDX])
        l_gap = np.sum(lidar_raw[idx.LIDAR_LEFT_IDX])
        r_gap = np.sum(lidar_raw[idx.LIDAR_RIGHT_IDX])

        norm_speed = np.linalg.norm(obs[idx.SELF.VEL_X : idx.SELF.VEL_Y + 1])
        if 0.001 < norm_speed < 0.015:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        push_candidate = _nearest_pushable(obs, idx)
        if push_candidate is not None and push_candidate[0] < 1.4 and push_candidate[1] > 0.15:
            front_min = max(front_min, 1.2)
        speed_scale = np.clip((front_min - 0.45) / 0.5, 0.0, 1.0)
        avoid_torque = (l_gap - r_gap) * 3.5
        avoid_w = np.clip(1.0 - (front_min / 1.5), 0.0, 1.0)
        if push_candidate is not None and push_candidate[0] < 1.4 and push_candidate[1] > 0.15:
            avoid_w *= 0.35

        visible_enemies = [en for en in idx.OTHERS if obs[en.VISIBLE] > 0.5]
        fwd, trn, lck, grb = 0.0, 0.0, 0.0, 0.0

        if self.reflex_timer > 0:
            self.reflex_timer -= 1
            if self.reflex_timer > 8:
                fwd, trn = 0.6 * self.escape_fwd_dir, 0.0
            else:
                fwd, trn = 0.2 * self.escape_fwd_dir, 0.85 * self.escape_turn_dir
        elif self.stuck_counter > 40 or front_min < 0.42:
            self.reflex_timer, self.stuck_counter = 14, 0
            self.escape_fwd_dir = -1.0 if front_min < back_min else 1.0
            self.escape_turn_dir = 1.0 if l_gap >= r_gap else -1.0
        else:
            if len(visible_enemies) > 0:
                target = visible_enemies[0]
                tx, ty = obs[target.REL_X] * p_scale, obs[target.REL_Y] * p_scale
                self.last_known_rel_pos_x, self.last_known_rel_pos_y = tx, ty
                self.memory_timer = 180
                target_angle = math.atan2(ty, tx)
                fwd = 0.85 * max(0.1, math.cos(target_angle)) * speed_scale
            elif self.memory_timer > 0:
                self.memory_timer -= 1
                target_angle = math.atan2(self.last_known_rel_pos_y, self.last_known_rel_pos_x)
                fwd = 0.55 * max(0.1, math.cos(target_angle)) * speed_scale
            else:
                # wall-follow heuristic for seekers: prefer sliding along wall when close
                try:
                    wall_dist = float(obs[idx.WALL_DIST])
                    nx_loc = float(obs[idx.WALL_NORM_X])
                    ny_loc = float(obs[idx.WALL_NORM_Y])
                except Exception:
                    wall_dist = 1.0
                    nx_loc = 0.0
                    ny_loc = 0.0
                if wall_dist < 0.9 and (abs(nx_loc) > 1e-6 or abs(ny_loc) > 1e-6):
                    tx, ty = -ny_loc, nx_loc
                    if tx < 0:
                        tx, ty = -tx, -ty
                    desired_angle = math.atan2(ty, tx)
                    fwd = max(fwd, 0.35 * speed_scale)
                    trn = np.clip(desired_angle * 2.2, -0.9, 0.9)
                    target_angle = None
                else:
                    self.wander_timer -= 1
                if avoid_w > 0.5:
                    self.wander_timer = 0
                if self.wander_timer <= 0:
                    if l_gap > r_gap:
                        self.wander_angle = cur_rot + np.random.uniform(0.3, np.pi)
                    else:
                        self.wander_angle = cur_rot + np.random.uniform(-np.pi, -0.3)
                    self.wander_timer = np.random.randint(150, 400)
                target_angle = (self.wander_angle - cur_rot + np.pi) % (2 * np.pi) - np.pi
                fwd = 0.45 * speed_scale

            if target_angle is not None:
                trn = np.clip(target_angle * 2.8 + avoid_torque * avoid_w, -0.9, 0.9)
            else:
                # target_angle None indicates wall-follow branch already set trn
                trn = np.clip(trn + avoid_torque * avoid_w, -0.9, 0.9)

        chase_priority = len(visible_enemies) > 0
        lck, grb, self.interact_cooldown, self.interact_focus_steps = _interaction_buttons(
            obs,
            idx,
            self.interact_cooldown,
            self.interact_focus_steps,
            suppress=chase_priority,
        )
        return np.array([fwd, trn, lck, grb])


class RuleBasedHider:
    """シーカーから側方へ逃げ、壁際では適切に後退してスタックを回避する Hider。"""

    def __init__(self):
        self.reflex_timer = 0
        self.wander_timer = 0
        self.wander_angle = 0.0
        self.stuck_counter = 0
        self.escape_turn_dir = 1.0
        self.escape_fwd_dir = -1.0
        self.interact_cooldown = 0
        self.interact_focus_steps = 0

    def get_action(self, obs, idx):
        p_scale = P_SCALE
        lidar_raw = obs[idx.LIDAR] * L_SCALE
        cur_rot = obs[idx.SELF.ROT] * R_SCALE

        front_min = np.min(lidar_raw[idx.LIDAR_FRONT_IDX])
        back_min = np.min(lidar_raw[idx.LIDAR_BACK_IDX])
        l_gap = np.sum(lidar_raw[idx.LIDAR_LEFT_IDX])
        r_gap = np.sum(lidar_raw[idx.LIDAR_RIGHT_IDX])

        norm_speed = np.linalg.norm(obs[idx.SELF.VEL_X : idx.SELF.VEL_Y + 1])
        if 0.001 < norm_speed < 0.015:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        push_candidate = _nearest_pushable(obs, idx)
        if push_candidate is not None and push_candidate[0] < 1.4 and push_candidate[1] > 0.15:
            front_min = max(front_min, 1.2)
        speed_scale = np.clip((front_min - 0.45) / 0.8, 0.15, 1.0)
        avoid_torque = (l_gap - r_gap) * 4.5
        avoid_w = np.clip(1.0 - (front_min / 1.5), 0.0, 1.0)
        if push_candidate is not None and push_candidate[0] < 1.4 and push_candidate[1] > 0.15:
            avoid_w *= 0.35

        seeker_vis = obs[idx.OTHERS[0].VISIBLE] > 0.5
        fwd, trn, lck, grb = 0.0, 0.0, 0.0, 0.0

        if self.reflex_timer > 0:  # スタック回避のための緊急後退行動
            self.reflex_timer -= 1  # タイマーが0になるまで回転を固定して後退を続ける。タイマーの長さは状況に応じて変化させる。
            if self.reflex_timer > 8:
                fwd, trn = 0.6 * self.escape_fwd_dir, 0.0
            else:
                fwd, trn = 0.2 * self.escape_fwd_dir, 0.85 * self.escape_turn_dir
        elif self.stuck_counter > 5 or front_min < 0.45:
            self.reflex_timer, self.stuck_counter = 10, 0
            self.escape_fwd_dir = -1.0 if front_min < back_min else 1.0
            self.escape_turn_dir = 1.0 if l_gap >= r_gap else -1.0
        else:  # 通常行動
            if seeker_vis:  # 追跡されている場合はシーカーから逃げる
                tx, ty = (
                    obs[idx.OTHERS[0].REL_X] * p_scale,
                    obs[idx.OTHERS[0].REL_Y] * p_scale,
                )
                angle_to_seeker = math.atan2(ty, tx)
                escape_base = (angle_to_seeker + np.pi + np.pi) % (2 * np.pi) - np.pi
                side_bias = 1.2 if l_gap > r_gap else -1.2
                target_angle = (escape_base + side_bias + np.pi) % (2 * np.pi) - np.pi

                fwd_val = math.cos(target_angle)
                if fwd_val < 0 and back_min < 0.5:
                    fwd = 0.0
                else:
                    fwd = 0.8 * fwd_val * speed_scale
            else:  # 非可視時はランダムに徘徊
                # wall-follow: prefer sliding along wall when close
                try:
                    wall_dist = float(obs[idx.WALL_DIST])
                    nx_loc = float(obs[idx.WALL_NORM_X])
                    ny_loc = float(obs[idx.WALL_NORM_Y])
                except Exception:
                    wall_dist = 1.0
                    nx_loc = 0.0
                    ny_loc = 0.0
                if wall_dist < 0.85 and (abs(nx_loc) > 1e-6 or abs(ny_loc) > 1e-6):
                    tx, ty = -ny_loc, nx_loc
                    if tx < 0:
                        tx, ty = -tx, -ty
                    target_angle = math.atan2(ty, tx)
                    fwd = 0.5 * speed_scale
                else:
                    self.wander_timer -= 1
                    if avoid_w > 0.5:
                        self.wander_timer = 0
                    if self.wander_timer <= 0:
                        if l_gap > r_gap:
                            self.wander_angle = cur_rot + np.random.uniform(0.3, np.pi)
                        else:
                            self.wander_angle = cur_rot + np.random.uniform(-np.pi, -0.3)
                        self.wander_timer = np.random.randint(200, 500)
                    target_angle = (self.wander_angle - cur_rot + math.pi) % (2 * math.pi) - math.pi
                    fwd = 0.45 * speed_scale

            trn = np.clip(target_angle * 2.8 + avoid_torque * avoid_w, -0.9, 0.9)

        seeker_dist = float("inf")
        if len(idx.OTHERS) > 0:
            sx = float(obs[idx.OTHERS[0].REL_X]) * p_scale
            sy = float(obs[idx.OTHERS[0].REL_Y]) * p_scale
            seeker_dist = math.hypot(sx, sy)
        escape_priority = seeker_vis and seeker_dist < 7.0
        lck, grb, self.interact_cooldown, self.interact_focus_steps = _interaction_buttons(
            obs,
            idx,
            self.interact_cooldown,
            self.interact_focus_steps,
            suppress=escape_priority,
        )
        return np.array([fwd, trn, lck, grb])
