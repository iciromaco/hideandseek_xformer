"""
seeker_course.py
================
Seeker 1体でコースを自律走行するデモ。
HideAndSeekEnv の seeker_specs に開始位置・向きを渡すだけで配置できる。

使い方:
    python seeker_course.py
    python seeker_course.py --speed 0.7
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hide_and_seek_env import HideAndSeekEnv

# ---------------------------------------------------------------------------
# コース定義  (x, y, arrive_radius)
# ---------------------------------------------------------------------------
WAYPOINTS = [
    (-5.0,  5.5, 0.6),   # A  NW スタート
    ( 0.5,  5.5, 0.18), # B  ランプ上側まで進んでから次へ
    ( 0.0,  0.8, 0.8),   # C  中央縦壁を避けて南下
    ( 0.2,  1.5, 0.7),   # D  隘路通過中
    ( 2.8, -3.2, 0.8),   # E  隘路抜け
    ( 4.5, -4.8, 0.8),   # F  SE コーナー → 右旋回(西)
    (-4.8, -4.8, 0.8),   # G  SW コーナー → 北東へ
    ( 0.0, -1.0, 0.9),   # H  NE エリア  → 右旋回(西)
    (-5.0,  5.2, 0.8),   # I  帰還
]

# ---------------------------------------------------------------------------
# ウェイポイント追従コントローラー
# ---------------------------------------------------------------------------
class WaypointController:
    TURN_GAIN    = 2.2
    ALIGN_THRESH = np.deg2rad(20)
    SLOW_THRESH  = np.deg2rad(65)
    NEAR_GAIN    = 2.5
    MIN_FWD_NEAR = 0.15
    MIN_TURN_NEAR = 0.45

    def __init__(self, model, data, speed_scale: float = 1.0):
        self.model       = model
        self.data        = data
        self.speed_scale = speed_scale
        self.wp_idx      = 1  # WP 0 はスタート位置なのでスキップ

        # 関節 qpos アドレス
        self._qadr_x   = _jnt_qposadr(model, "seeker_0_x")
        self._qadr_y   = _jnt_qposadr(model, "seeker_0_y")
        self._qadr_rot = _jnt_qposadr(model, "seeker_0_rot")

        # アクチュエータ ID
        self._act_fwd  = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "seeker_0_fwd")
        self._act_turn = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "seeker_0_turn")

        # anchor の XML 基準座標 (= seeker_specs で指定した値)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seeker_0_anchor")
        self._ref_x = float(model.body_pos[bid, 0])
        self._ref_y = float(model.body_pos[bid, 1])

    @property
    def world_x(self) -> float:
        return self._ref_x + float(self.data.qpos[self._qadr_x])

    @property
    def world_y(self) -> float:
        return self._ref_y + float(self.data.qpos[self._qadr_y])

    @property
    def heading(self) -> float:
        return float(self.data.qpos[self._qadr_rot])

    def step(self) -> bool:
        if self.wp_idx >= len(WAYPOINTS):
            self.data.ctrl[self._act_fwd]  = 0.0
            self.data.ctrl[self._act_turn] = 0.0
            return False

        tx, ty, arrive_r = WAYPOINTS[self.wp_idx]
        dx, dy = tx - self.world_x, ty - self.world_y
        dist   = np.hypot(dx, dy)

        if dist < arrive_r:
            print(f"  WP {self.wp_idx} 到達: ({self.world_x:.2f}, {self.world_y:.2f})")
            self.wp_idx += 1
            return self.wp_idx < len(WAYPOINTS)

        target_angle = np.arctan2(dy, dx)
        err          = _angle_diff(target_angle, self.heading)
        ctrl_turn    = float(np.clip(err * self.TURN_GAIN, -1.0, 1.0))

        abs_err = abs(err)
        if abs_err < self.ALIGN_THRESH:
            ctrl_fwd = 1.0
        elif abs_err < self.SLOW_THRESH:
            t        = (abs_err - self.ALIGN_THRESH) / (self.SLOW_THRESH - self.ALIGN_THRESH)
            ctrl_fwd = 1.0 - 0.85 * t
        else:
            ctrl_fwd = 0.0

        # 目標近傍では前進と旋回の両方を弱めてオーバーシュートを抑制
        near_dist = max(arrive_r * self.NEAR_GAIN, arrive_r + 0.25)
        if dist < near_dist:
            near_alpha = (dist - arrive_r) / (near_dist - arrive_r + 1e-8)
            near_alpha = float(np.clip(near_alpha, 0.0, 1.0))

            fwd_scale = self.MIN_FWD_NEAR + (1.0 - self.MIN_FWD_NEAR) * near_alpha
            turn_scale = self.MIN_TURN_NEAR + (1.0 - self.MIN_TURN_NEAR) * near_alpha

            ctrl_fwd *= fwd_scale
            ctrl_turn *= turn_scale

        self.data.ctrl[self._act_fwd]  = float(np.clip(ctrl_fwd * self.speed_scale, -1.0, 1.0))
        self.data.ctrl[self._act_turn] = ctrl_turn
        return True


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def _jnt_qposadr(model, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[jid])


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return float((d + np.pi) % (2 * np.pi) - np.pi)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--assist-scale", type=float, default=2.4)
    parser.add_argument("--assist-top-extra", type=float, default=1.1)
    parser.add_argument("--assist-stall-extra", type=float, default=1.8)
    parser.add_argument("--stall-speed", type=float, default=0.08)
    parser.add_argument("--assist-z-hold", type=float, default=0.06)
    parser.add_argument("--assist-min-fwd-top", type=float, default=0.45)
    args = parser.parse_args()

    # ---- 開始位置を seeker_specs で指定 ------------------------------------
    #   make_spec(x, y, rot_deg)  ← 度単位で向きを指定できる便利メソッド
    env = HideAndSeekEnv(
        n_seekers    = 1,
        n_hiders     = 0,
        n_boxes      = 1,
        n_ramps      = 1,
        seeker_specs = [HideAndSeekEnv.make_spec(-5.0, 4.8, rot_deg=0.0)],  # NW, 東向き
        box_specs    = [HideAndSeekEnv.make_spec(3.0, 5.2, rot_deg=0.0)],   # Ramp右側に離して配置
        ramp_specs   = [HideAndSeekEnv.make_spec(0.0, 5.2, rot_deg=0.0)],   # 低い側を -X 向き
    )
    model, data = env.model, env.data
    ctrl = WaypointController(model, data, speed_scale=args.speed)
    qadr_z = _jnt_qposadr(model, "seeker_0_z")
    dadr_x = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "seeker_0_x")])
    dadr_y = int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "seeker_0_y")])
    seeker_body_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "seeker_0_body")
    ramp_body_bids = []
    for i in range(env.n_ramps):
        ramp_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ramp_{i}_body")
        if ramp_bid >= 0:
            ramp_body_bids.append(ramp_bid)

    base_fwd_gain = float(model.actuator_gainprm[ctrl._act_fwd, 0])
    assist_gain_scale = float(args.assist_scale)
    assist_top_extra = float(args.assist_top_extra)
    assist_stall_extra = float(args.assist_stall_extra)
    stall_speed_thresh = float(args.stall_speed)
    assist_z_hold = float(args.assist_z_hold)
    assist_min_fwd_top = float(args.assist_min_fwd_top)
    assist_release_z = max(0.02, assist_z_hold * 0.5)
    slope_top_center_local = np.array([0.0, 0.0, 0.52], dtype=float)
    slope_top_dist_thresh = 1.15
    slope_top_y_abs = 0.62
    slope_top_z_min = 0.38
    slope_top_z_max = 1.05
    lower_entry_x_min = -1.05
    lower_entry_x_max = 0.30
    lower_entry_y_abs = 0.62
    lower_entry_z_min = 0.20
    lower_entry_z_max = 0.70
    ramp_env_x_min = -1.15
    ramp_env_x_max = 1.05
    ramp_env_y_abs = 0.70
    ramp_env_z_min = 0.10
    ramp_env_z_max = 1.20
    assist_uphill_v_min = -0.02
    assist_downhill_v_thresh = -0.04
    crest_taper_start_x = 0.58
    crest_taper_end_x = 0.95
    crest_ctrl_min = 0.78
    z_climb_thresh = 0.02

    print("===== Seeker コース走行 =====")
    print(f"速度倍率: {args.speed:.2f} | WP 数: {len(WAYPOINTS)}")
    print(f"アシスト倍率(進入): x{1.0 + assist_gain_scale:.2f}")
    print(f"アシスト倍率(上面): x{1.0 + assist_gain_scale + assist_top_extra:.2f}")
    print(f"失速ブースト: +{assist_stall_extra:.2f} (速度<{stall_speed_thresh:.2f} m/s)")
    print(f"斜面アシスト最小前進: {assist_min_fwd_top:.2f}")
    print(f"登坂保持Zしきい値: {assist_z_hold:.3f}")
    print(f"アシスト解除Zしきい値: {assist_release_z:.3f}")
    print("アシスト方向条件: ランプ局所X速度ベース(下り速度で解除)")
    print("Viewer を起動します。ウィンドウを閉じると終了します。\n")

    print("[DBG] launch_passive 呼び出し前")
    viewer = mujoco.viewer.launch_passive(model, data)
    print(f"[DBG] launch_passive 完了  is_running={viewer.is_running()}")

    viewer.cam.lookat[:]  = [0.0, 0.0, 0.0]
    viewer.cam.distance   = 19.0
    viewer.cam.elevation  = -65.0
    viewer.cam.azimuth    = 90.0

    print("[DBG] sync 呼び出し前")
    viewer.sync()
    print(f"[DBG] sync 完了  is_running={viewer.is_running()}")

    time.sleep(0.2)
    print(f"[DBG] 0.2s sleep 後  is_running={viewer.is_running()}")

    sim_time   = 0.0
    wall_start = time.perf_counter()
    ctrl_done  = False
    loop_count = 0
    was_on_ramp = False
    was_assist_on = False
    assist_latched = False

    print("[DBG] while True ループ開始")
    try:
        while True:
            loop_count += 1
            if loop_count <= 3 or loop_count % 200 == 0:
                wx = ctrl.world_x
                wy = ctrl.world_y
                hd = ctrl.heading
                tx, ty, _ = WAYPOINTS[ctrl.wp_idx] if ctrl.wp_idx < len(WAYPOINTS) else (0, 0, 0)
                dist = (wx-tx)**2 + (wy-ty)**2
                print(f"[DBG] loop={loop_count}  sim={sim_time:.3f}  "                      f"pos=({wx:.2f},{wy:.2f})  heading={hd:.2f}rad  "                      f"wp={ctrl.wp_idx} target=({tx},{ty})  dist2={dist:.2f}  "                      f"ctrl_fwd={ctrl.data.ctrl[ctrl._act_fwd]:.2f}  "                      f"ctrl_turn={ctrl.data.ctrl[ctrl._act_turn]:.2f}  "                      f"assist_on={int(was_assist_on)}  "                      f"assist_latched={int(assist_latched)}")
            # リアルタイム制御のためスリープ
            time.sleep(0.001)

            wall_now = time.perf_counter()
            target_t = wall_now - wall_start

            while sim_time < target_t:
                if not ctrl_done:
                    ctrl_done = not ctrl.step()

                z_rel = float(data.qpos[qadr_z])
                seeker_pos = data.xpos[seeker_body_bid]
                on_slope_top = False
                in_lower_entry = False
                best_top_dist = 1e9
                best_ramp_pos = None
                best_ramp_rot = None
                best_rel_local = None
                top_local_x = 0.0
                for ramp_bid in ramp_body_bids:
                    ramp_pos = data.xpos[ramp_bid]
                    ramp_rot = data.xmat[ramp_bid].reshape(3, 3)
                    rel_world = seeker_pos - ramp_pos
                    rel_local = ramp_rot.T @ rel_world
                    d_top = float(np.linalg.norm(rel_local - slope_top_center_local))
                    if d_top < best_top_dist:
                        best_top_dist = d_top
                        best_ramp_pos = ramp_pos.copy()
                        best_ramp_rot = ramp_rot.copy()
                        best_rel_local = rel_local.copy()
                    if (
                        d_top <= slope_top_dist_thresh
                        and abs(rel_local[1]) <= slope_top_y_abs
                        and slope_top_z_min <= rel_local[2] <= slope_top_z_max
                    ):
                        on_slope_top = True
                        top_local_x = float(rel_local[0])
                        break

                    if (
                        lower_entry_x_min <= rel_local[0] <= lower_entry_x_max
                        and abs(rel_local[1]) <= lower_entry_y_abs
                        and lower_entry_z_min <= rel_local[2] <= lower_entry_z_max
                    ):
                        in_lower_entry = True

                climb_hold = z_rel > assist_z_hold
                in_ramp_envelope = False
                if best_rel_local is not None:
                    in_ramp_envelope = (
                        ramp_env_x_min <= best_rel_local[0] <= ramp_env_x_max
                        and abs(best_rel_local[1]) <= ramp_env_y_abs
                        and ramp_env_z_min <= best_rel_local[2] <= ramp_env_z_max
                    )
                ramp_region = on_slope_top or in_lower_entry or in_ramp_envelope
                target_local_dx = 0.0
                local_vx = 0.0
                if best_ramp_pos is not None and best_ramp_rot is not None:
                    vel_world = np.array([data.qvel[dadr_x], data.qvel[dadr_y], 0.0], dtype=float)
                    local_vx = float(np.dot(best_ramp_rot[:, 0], vel_world))

                if best_ramp_pos is not None and best_ramp_rot is not None and ctrl.wp_idx < len(WAYPOINTS):
                    tx, ty, _ = WAYPOINTS[ctrl.wp_idx]
                    target_world = np.array([tx, ty, seeker_pos[2]], dtype=float)
                    target_rel_local = best_ramp_rot.T @ (target_world - best_ramp_pos)
                    target_local_dx = float(target_rel_local[0] - (best_ramp_rot.T @ (seeker_pos - best_ramp_pos))[0])

                uphill_motion_ok = local_vx >= assist_uphill_v_min
                descending_on_slope = on_slope_top and (local_vx <= assist_downhill_v_thresh)

                if (ramp_region or climb_hold) and uphill_motion_ok:
                    assist_latched = True
                elif assist_latched and (ramp_region or z_rel > assist_release_z) and (not descending_on_slope):
                    assist_latched = True
                else:
                    assist_latched = False

                should_assist = assist_latched and (not descending_on_slope)
                if in_lower_entry and assist_latched and (not on_slope_top) and data.ctrl[ctrl._act_fwd] < 1.0:
                    data.ctrl[ctrl._act_fwd] = 1.0

                vxy = float(np.hypot(data.qvel[dadr_x], data.qvel[dadr_y]))

                if on_slope_top:
                    crest_alpha = float(np.clip(
                        (top_local_x - crest_taper_start_x)
                        / (crest_taper_end_x - crest_taper_start_x + 1e-8),
                        0.0,
                        1.0,
                    ))
                    crest_ctrl_cap = 1.0 - (1.0 - crest_ctrl_min) * crest_alpha
                    if data.ctrl[ctrl._act_fwd] > crest_ctrl_cap:
                        data.ctrl[ctrl._act_fwd] = crest_ctrl_cap
                else:
                    crest_alpha = 0.0
                    crest_ctrl_cap = 1.0

                if should_assist and on_slope_top:
                    min_fwd_cap = min(assist_min_fwd_top, crest_ctrl_cap)
                    if data.ctrl[ctrl._act_fwd] < min_fwd_cap:
                        data.ctrl[ctrl._act_fwd] = min_fwd_cap

                if should_assist:
                    gain_scale = assist_gain_scale + (assist_top_extra * (1.0 - crest_alpha))
                    stall_boost_on = vxy < stall_speed_thresh
                    if stall_boost_on:
                        gain_scale += assist_stall_extra
                    target_gain = base_fwd_gain * (1.0 + gain_scale)
                else:
                    gain_scale = 0.0
                    stall_boost_on = False
                    target_gain = base_fwd_gain
                if abs(float(model.actuator_gainprm[ctrl._act_fwd, 0]) - target_gain) > 1e-9:
                    model.actuator_gainprm[ctrl._act_fwd, 0] = target_gain

                if should_assist and not was_assist_on:
                    print(
                        f"[LOG] Assist ON sim={sim_time:.3f} "
                        f"gain={target_gain:.1f} "
                        f"gain_scale={gain_scale:.2f} "
                        f"on_slope_top={int(on_slope_top)} "
                        f"in_lower_entry={int(in_lower_entry)} "
                        f"climb_hold={int(climb_hold)} "
                        f"uphill_motion={int(uphill_motion_ok)} "
                        f"target_dx={target_local_dx:.3f} "
                        f"local_vx={local_vx:.3f} "
                        f"descending={int(descending_on_slope)} "
                        f"ramp_env={int(in_ramp_envelope)} "
                        f"ramp_region={int(ramp_region)} "
                        f"latched={int(assist_latched)} "
                        f"top_x={top_local_x:.3f} "
                        f"crest_alpha={crest_alpha:.2f} "
                        f"stall_boost={int(stall_boost_on)} "
                        f"vxy={vxy:.3f} "
                        f"ctrl_fwd={float(data.ctrl[ctrl._act_fwd]):.2f} "
                        f"ctrl_cap={crest_ctrl_cap:.2f} "
                        f"top_dist={best_top_dist:.3f} "
                        f"z_rel={z_rel:.4f} pos=({ctrl.world_x:.2f},{ctrl.world_y:.2f})"
                    )
                elif (not should_assist) and was_assist_on:
                    print(
                        f"[LOG] Assist OFF sim={sim_time:.3f} "
                        f"gain={base_fwd_gain:.1f} "
                        f"gain_scale=0.00 "
                        f"in_lower_entry={int(in_lower_entry)} "
                        f"climb_hold={int(climb_hold)} "
                        f"uphill_motion={int(uphill_motion_ok)} "
                        f"target_dx={target_local_dx:.3f} "
                        f"local_vx={local_vx:.3f} "
                        f"descending={int(descending_on_slope)} "
                        f"ramp_env={int(in_ramp_envelope)} "
                        f"ramp_region={int(ramp_region)} "
                        f"latched={int(assist_latched)} "
                        f"top_x={top_local_x:.3f} "
                        f"crest_alpha={crest_alpha:.2f} "
                        f"stall_boost=0 "
                        f"vxy={vxy:.3f} "
                        f"ctrl_fwd={float(data.ctrl[ctrl._act_fwd]):.2f} "
                        f"ctrl_cap={crest_ctrl_cap:.2f} "
                        f"top_dist={best_top_dist:.3f} "
                        f"z_rel={z_rel:.4f} pos=({ctrl.world_x:.2f},{ctrl.world_y:.2f})"
                    )
                was_assist_on = should_assist

                mujoco.mj_step(model, data)
                z_rel_after = float(data.qpos[qadr_z])
                is_on_ramp = on_slope_top and (z_rel_after > z_climb_thresh)
                if is_on_ramp and not was_on_ramp:
                    print(
                        f"[LOG] Ramp乗上げ 検知 sim={sim_time:.3f} "
                        f"z_rel={z_rel_after:.4f} "
                        f"pos=({ctrl.world_x:.2f},{ctrl.world_y:.2f}) wp={ctrl.wp_idx}"
                    )
                was_on_ramp = is_on_ramp
                sim_time += model.opt.timestep

            viewer.sync()

            if not viewer.is_running():
                print(f"[DBG] is_running=False で break (loop={loop_count})")
                break

    except KeyboardInterrupt:
        print("[DBG] KeyboardInterrupt で停止")
    except Exception as e:
        import traceback
        print(f"[DBG] 例外発生: {e}")
        traceback.print_exc()
    finally:
        if was_assist_on:
            print(f"[LOG] Assist OFF sim={sim_time:.3f} gain={base_fwd_gain:.1f} (finally)")
        if abs(float(model.actuator_gainprm[ctrl._act_fwd, 0]) - base_fwd_gain) > 1e-9:
            model.actuator_gainprm[ctrl._act_fwd, 0] = base_fwd_gain
        print(f"[DBG] finally: loop_count={loop_count}")
        print("完了。")


if __name__ == "__main__":
    main()