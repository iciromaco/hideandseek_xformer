# src/envs/hns_environment.py
# flake8: noqa
# hns_environment.py v4.5９
import logging
import math

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch
from gymnasium import spaces
from numba import njit

from agents.scripted_agents import RuleBasedHider, RuleBasedSeeker
from core.constants import L_SCALE, P_SCALE, R_SCALE, V_SCALE
from core.obs_indices import ObsIdx
from core.visibility_engine import VisibilityEngine
from models.ppo_transformer_v2 import AgentV2

# module logger
logger = logging.getLogger(__name__)

# Pylint: mujoco C-extension exposes members dynamically; suppress no-member and
# broad-exception warnings here to reduce false positives from static analysis.
# pylint: disable=E1101,W0718


# --- DebugLoggerクラス ---
class DebugLogger:
    def __init__(self, enabled=False, console=True, log_interval_steps=100):
        # `enabled` controls whether the logger records/handles debug events
        # `console` controls whether messages are printed to stdout
        self.enabled = enabled
        self.console = console
        self.log_interval_steps = log_interval_steps
        self._log_last_step = {}
        # use short name consistently
        self._policy_src_logged = set()

    def print(self, message):
        if self.console:
            print(message)

    def print_throttled(self, key, message, current_step, force=False):
        # preserve previous behavior: suppressed unless logger enabled
        if not self.enabled and not force:
            return
        # original implementation had no-op; keep suppressed behaviour
        return

    def log_policy_src(self, agent_key, source, current_step, force=False):
        if not self.enabled and not force:
            return
        # preserved as no-op for now (external sinks may implement later)
        return

    def clear_policy_src_log(self):
        self._policy_src_logged.clear()

    def clear_last_step(self):
        self._log_last_step.clear()


def _euler_z_to_quat(yaw):
    """Z軸回転角からクォータニオン文字列を生成。"""
    w, z = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return f"{w:.6f} 0 0 {z:.6f}"


@njit(cache=True)
def _blocked_by_walls_numba(p1x, p1y, p2x, p2y, walls, margin):
    dx = p2x - p1x
    dy = p2y - p1y
    for i in range(walls.shape[0]):
        cx = walls[i, 0]
        cy = walls[i, 1]
        hx = walls[i, 2]
        hy = walls[i, 3]
        x_min = cx - hx - margin
        x_max = cx + hx + margin
        y_min = cy - hy - margin
        y_max = cy + hy + margin
        t0 = 0.0
        t1 = 1.0

        if abs(dx) < 1e-9:
            if p1x < x_min or p1x > x_max:
                continue
        else:
            inv_dx = 1.0 / dx
            tx1 = (x_min - p1x) * inv_dx
            tx2 = (x_max - p1x) * inv_dx
            t_enter = tx1 if tx1 < tx2 else tx2
            t_exit = tx2 if tx1 < tx2 else tx1
            if t_enter > t0:
                t0 = t_enter
            if t_exit < t1:
                t1 = t_exit
            if t0 > t1:
                continue

        if abs(dy) < 1e-9:
            if p1y < y_min or p1y > y_max:
                continue
        else:
            inv_dy = 1.0 / dy
            ty1 = (y_min - p1y) * inv_dy
            ty2 = (y_max - p1y) * inv_dy
            t_enter = ty1 if ty1 < ty2 else ty2
            t_exit = ty2 if ty1 < ty2 else ty1
            if t_enter > t0:
                t0 = t_enter
            if t_exit < t1:
                t1 = t_exit
            if t0 > t1:
                continue

        return True
    return False


# (short name only) previously there was a longer name; keep short name


class TeamCosEnv(gym.Env):
    # --- 統計バッファ初期化 ---
    def _init_debug_stats(self):
        self._dbg_reward_buf = []
        self._dbg_hide_buf = []
        self._dbg_wall_dist_buf = []
        self._dbg_step_ctr = 0
        self._dbg_log_int = getattr(self, "debug_logger", None) and getattr(self.debug_logger, "log_interval_steps", 100) or 100
        # store last ramp boost per agent for toggle detection
        self._last_ramp_boost = {k: 0.0 for k in getattr(self, "agent_keys", [])}
        # hold counter per-agent to ignore brief CLEAR transitions
        self._ramp_boost_hold = {k: 0 for k in getattr(self, "agent_keys", [])}
        # previous agent world z for detecting upward motion (climbing)
        self._prev_agent_z = {k: None for k in getattr(self, "agent_keys", [])}

    def _dbg_collect_stats(self, reward, info):
        if not self.debug_mode:
            return
        self._dbg_reward_buf.append(reward)
        # 隠れ率: is_detected==False なら隠れているとみなす
        self._dbg_hide_buf.append(0 if info.get("is_detected", False) else 1)
        wd = info.get("wall_distance", None)
        if wd is not None:
            if isinstance(wd, (list, tuple, np.ndarray)):
                self._dbg_wall_dist_buf.extend([float(v) for v in wd])
            else:
                self._dbg_wall_dist_buf.append(float(wd))
        self._dbg_step_ctr += 1
        # log_intervalごとに統計出力
        if self._dbg_step_ctr % self._dbg_log_int == 0:
            self._dbg_print_stats()

    def _dbg_print_stats(self):
        if not self.debug_mode:
            return
        avg_reward = float(np.mean(self._dbg_reward_buf)) if self._dbg_reward_buf else 0.0
        hide_rate = float(np.mean(self._dbg_hide_buf)) if self._dbg_hide_buf else 0.0
        wd_arr = np.array(self._dbg_wall_dist_buf, dtype=np.float32) if self._dbg_wall_dist_buf else np.array([0.0])
        wd_mean = float(wd_arr.mean())
        wd_min = float(wd_arr.min())
        wd_max = float(wd_arr.max())
        self.debug_logger.print(f"[DEBUG] Step={self.current_step} AvgR={avg_reward:.3f} HideRate={hide_rate:.2f} " f"WD(mean/min/max)={wd_mean:.3f}/{wd_min:.3f}/{wd_max:.3f}")
        self._dbg_reward_buf.clear()
        self._dbg_hide_buf.clear()
        self._dbg_wall_dist_buf.clear()

    # Hide and Seek 高度物理環境 (単一エージェント学習最適化版)
    ARENA_HALF = 6.0
    SAFE_HALF = 5.0
    R_AGENT = 0.55
    R_BOX = 0.95
    R_RAMP = 1.30

    AGENT_DAMPING_XY = 25  #
    AGENT_DAMPING_Z = 20  # 16.0
    AGENT_DAMPING_ROT = 25
    AGENT_ACTUATOR_FWD = 1500  # 700
    AGENT_ACTUATOR_TURN_GAIN = 150
    AGENT_BOTTOM_MASS = 20.0
    AGENT_HEAD_MASS = 10.0
    AGENT_Z_MIN = -0.05
    AGENT_Z_MAX = 1.20
    AGENT_MAX_VZ = 2.2
    RAMP_JOINT_DAMPING = 100.0
    BOX_JOINT_DAMPING = 100.0
    RAMP_MASS = 60.0
    RAMP_INNER_WEIGHT_MASS = 30.0
    BOX_MASS = 100.0
    INTERACT_RANGE = 1.95
    BTN_ON = 0.1
    BTN_COOLDOWN = 8
    GRAB_OFFSET = 1.45
    GRAB_FOLLOW_GAIN = 8.0
    GRAB_MAX_SPEED = 2.6
    GRAB_BREAK_DIST = 2.8
    RAMP_BOOST_FWD = 0.35
    # number of simulation steps to hold a previously-engaged boost when a
    # transient CLEAR condition is observed (mitigates brief overshoots/noise)
    RAMP_BOOST_HOLD_STEPS = 8
    OBJECT_PLANAR_LOCK = True
    RAMP_LOCKED_SPEED_EPS = 0.035
    FREE_OBJ_LINEAR_DAMP = 0.90
    FREE_OBJ_STOP_EPS = 0.045
    INTERACT_OCCLUSION_MARGIN = 0.03
    # When an inference model is present for the learnable agent, the model's
    # forward channel can be multiplied by this sign. Set to 1.0 for no change
    # or -1.0 to flip the forward direction at apply-time. Keeping this as a
    # tunable class attribute avoids editing code logic elsewhere.
    INFERENCE_FORWARD_SIGN = 1.0
    # --- Wall repulsion parameters ---
    # distance within which wall repulsion is applied (m)
    WALL_REPULSION_RADIUS = 0.25
    # effective agent radius for clearance-based checks (separate from spawn R_AGENT)
    AGENT_EFFECTIVE_RADIUS = 0.4  # エージェント半径
    # clearance (m) from agent surface below which repulsion begins
    WALL_REPULSION_CLEARANCE = 0.20
    # scaling factor for repulsion relative to estimated agent forward force
    WALL_REPULSION_ALPHA = 0.3
    # maximum repulsion force (N)
    WALL_REPULSION_FMAX = 800.0
    # --- Runtime defaults for smoothing/scaling/clamping (recommended safe defaults) ---
    # Global multiplier applied to the estimate-based repulsion (0 = disable)
    WALL_REPULSION_CTRL_SCALE = 0.05
    # EMA low-pass coefficient applied to the repulsion estimate (0..1)
    WALL_REPULSION_CTRL_LP = 0.2
    # Optional explicit upper clip on smoothed ctrl (None disables)
    WALL_REPULSION_CTRL_CLIP = 50.0
    # Optional per-step accumulated force clamp applied before writing data.xfrc_applied
    WALL_REPULSION_ACCUM_CLAMP_MAX = 100.0
    # world-z above which ramp-based repulsion suppression is considered
    WALL_REPULSION_SUPPRESS_Z = 0.5 + 0.2  # z margin below which suppression is applied even if not currently climbing

    def __init__(
        self,
        mode="initial",
        target="hider",
        n_seekers=1,
        n_hiders=2,
        n_boxes=2,
        n_ramps=1,
        render_mode=None,
        inference_policies=None,
        show_turn_lines=True,
        policy_source_log=False,
        policy_src_log_each_reset=False,
        dbg_log_interval_steps=200,
        mode4_sdf_cell_size=0.05,
        debug_mode=False,
        debug_console=True,
        action_repeat=16,
        learnable_turn_scale=1.0,
    ):
        super().__init__()
        self.n_seekers = int(n_seekers)
        self.n_hiders = int(n_hiders)
        self.n_boxes = int(n_boxes)
        self.n_ramps = int(n_ramps)
        if self.n_seekers < 1 or self.n_hiders < 1:
            raise ValueError("n_seekers and n_hiders must be >= 1")
        if self.n_ramps < 0:
            raise ValueError("n_ramps must be >= 0")
        self.mode, self.target, self.render_mode = mode, target, render_mode
        self.show_turn_lines = bool(show_turn_lines)
        self.mode4_sdf_cell_size = float(mode4_sdf_cell_size)
        self.current_step, self.prep_steps, self.max_episode_steps = 0, 80, 500
        # use private attr and property to keep debug_logger.enabled in sync
        self._debug_mode = bool(debug_mode)
        self.debug_logger = DebugLogger(enabled=self._debug_mode, console=bool(debug_console), log_interval_steps=dbg_log_interval_steps)
        self.action_repeat = action_repeat
        # multiplier applied to the learnable agent's turn channel (runtime-only)
        self.learnable_turn_scale = float(learnable_turn_scale)
        # preserve flags passed for backward compatibility / debugging
        self.policy_source_log = bool(policy_source_log)
        self.policy_src_log_each_reset = bool(policy_src_log_each_reset)
        # temporal caches for visibility / being-hit freshness
        self._prev_vis = {}
        self._prev_being_hit = {}
        # render disabled flag (set if passive viewer cannot be launched on macOS)
        self._render_disabled = False
        # counters for persisting BEING_HIT in observations
        self.being_hit_persist = 3
        self._being_hit_counters = {}
        # cached last observation returned for the learnable agent
        self._cached_obs = None
        self._init_debug_stats()
        if self.n_seekers == 1:
            self.seeker_keys = ["s"]
        else:
            self.seeker_keys = [f"s{i}" for i in range(1, self.n_seekers + 1)]
        self.hider_keys = [f"h{i}" for i in range(1, self.n_hiders + 1)]
        self.agent_keys = self.seeker_keys + self.hider_keys
        self.learnable_agent_key = self.seeker_keys[0] if target == "seeker" else self.hider_keys[0]
        self.learnable_agent_idx = self.agent_keys.index(self.learnable_agent_key)
        self.idx = ObsIdx(n_boxes, n_ramps, n_others=len(self.agent_keys) - 1)

        self.model = mujoco.MjModel.from_xml_string(self._build_dynamic_xml())
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(
            self.model,
            self.data,
            mode4_sdf_cell_size=self.mode4_sdf_cell_size,
        )
        self.viewer = None
        self.inference_policies = inference_policies or {}
        self._inference_models = {}
        self._inference_seq_lens = {}
        self.shared_team_policy = False
        self.shared_policy_model = None
        self.shared_policy_seq_len = 8
        self.shared_policy_hdim = 128
        self.shared_team_prefix = "h" if self.learnable_agent_key.startswith("h") else "s"
        self._policy_histories = {}
        self.override_learnable_policy = False
        self.model_policy_deterministic = True
        # デバッグ用ロガーに移譲

        self.last_debug_ctrl = dict.fromkeys(self.agent_keys, (0.0, 0.0))

        # Ensure runtime instance attributes for wall-repulsion constants exist
        # Some run-time imports/overrides may omit class-level constants; set
        # instance fallbacks here to guarantee availability during tests.
        try:
            self.AGENT_EFFECTIVE_RADIUS = float(getattr(self.__class__, "AGENT_EFFECTIVE_RADIUS", 0.4))
        except Exception:
            self.AGENT_EFFECTIVE_RADIUS = 0.4
        try:
            self.WALL_REPULSION_CLEARANCE = float(getattr(self.__class__, "WALL_REPULSION_CLEARANCE", 0.2))
        except Exception:
            self.WALL_REPULSION_CLEARANCE = 0.2

        self.body_ids, self.qpos_indices, self.actuator_ids = {}, {}, {}
        self.obj_body_map = {}
        self.obj_geom_ids = {}
        self.obj_default_rgba = {}
        self.maze_walls = [
            (3, 1.5, 1.5, 0.2),
            (-3, -1.5, 1.5, 0.2),
            (0, -3, 0.2, 1.5),
            (0, 3, 0.2, 1.5),
        ]
        s = self.ARENA_HALF
        self.static_wall_aabbs = np.asarray(
            [
                (0.0, 6.1, s + 0.15, 0.1),
                (0.0, -6.1, s + 0.15, 0.1),
                (6.1, 0.0, 0.1, s),
                (-6.1, 0.0, 0.1, s),
                *self.maze_walls,
            ],
            dtype=np.float64,
        )
        self._analyze_structure()
        self._init_agent_intelligence()
        self._init_interaction_state()

        # 観測空間
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.idx.total_dim,), np.float32)

        # 【修正】アクションスペースを 4 次元に固定。
        # 外部（学習アルゴリズム）からは常に 1 体分の入力を受け取る。
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    def set_shared_team_policy_state(self, state_dict, seq_len=8, hidden_dim=128):
        if state_dict is None:
            self.shared_team_policy = False
            self.shared_policy_model = None
            return False

        self.shared_team_policy = True
        self.shared_policy_seq_len = int(seq_len)
        self.shared_policy_hdim = int(hidden_dim)
        obs_dim = int(self.observation_space.shape[0])
        act_dim = int(self.action_space.shape[0])
        policy_model = AgentV2(obs_dim, act_dim, self.shared_policy_hdim, self.shared_policy_seq_len)
        policy_model.load_state_dict(state_dict)
        policy_model.eval()
        self.shared_policy_model = policy_model
        for ak in self.agent_keys:
            self._policy_histories.pop((ak, self.shared_policy_seq_len), None)
        return True

    def set_inference_policy_state(self, agent_keys, state_dict, seq_len=8, hidden_dim=128):
        if state_dict is None:
            return False

        keys = [k for k in agent_keys if k in self.agent_keys]
        if not keys:
            return False

        obs_dim = int(self.observation_space.shape[0])
        act_dim = int(self.action_space.shape[0])
        policy_model = AgentV2(obs_dim, act_dim, int(hidden_dim), int(seq_len))
        policy_model.load_state_dict(state_dict)
        policy_model.eval()

        for ak in keys:
            self._inference_models[ak] = policy_model
            self._inference_seq_lens[ak] = int(seq_len)
            self._policy_histories.pop((ak, int(seq_len)), None)
        return True

    def set_override_learnable_policy(self, enabled):
        self.override_learnable_policy = bool(enabled)
        return True

    def set_model_policy_deterministic(self, enabled):
        self.model_policy_deterministic = bool(enabled)
        return True

    def _ensure_policy_history(self, agent_key, seq_len):
        sl = int(seq_len)
        obs_dim = int(self.observation_space.shape[0])
        key = (agent_key, sl)
        hist = self._policy_histories.get(key)
        if hist is None or hist["buffer"].shape != (sl * 2, obs_dim):
            hist = {
                "buffer": np.zeros((sl * 2, obs_dim), dtype=np.float32),
                "ptr": 0,
            }
            self._policy_histories[key] = hist
        return hist

    def _prime_policy_history(self, agent_key, seq_len, norm_obs):
        hist = self._ensure_policy_history(agent_key, seq_len)
        obs_np = np.asarray(norm_obs, dtype=np.float32).reshape(-1)
        sl = int(seq_len)
        for i in range(sl):
            hist["buffer"][i] = obs_np
            hist["buffer"][i + sl] = obs_np
        hist["ptr"] = 0

    def _update_policy_history(self, agent_key, seq_len, norm_obs):
        hist = self._ensure_policy_history(agent_key, seq_len)
        obs_np = np.asarray(norm_obs, dtype=np.float32).reshape(-1)
        sl = int(seq_len)
        ptr = int(hist["ptr"])
        hist["buffer"][ptr] = obs_np
        hist["buffer"][ptr + sl] = obs_np
        hist["ptr"] = (ptr + 1) % sl

    def _get_policy_history_seq(self, agent_key, seq_len, norm_obs):
        hist = self._ensure_policy_history(agent_key, seq_len)
        if not np.any(hist["buffer"]):
            self._prime_policy_history(agent_key, seq_len, norm_obs)
            hist = self._ensure_policy_history(agent_key, seq_len)
        ptr = int(hist["ptr"])
        sl = int(seq_len)
        return hist["buffer"][ptr : ptr + sl]

    def _build_dynamic_xml(self):
        arena = self._xml_static_scene()
        ramps = "".join(self._xml_ramp(i, [0, 0], 0) for i in range(1, self.n_ramps + 1))
        # place boxes on a small radius around the origin to avoid spawning directly above agents
        box_positions = self._default_box_positions()
        boxes = "".join(self._xml_box(i, box_positions[i - 1], 0) for i in range(1, self.n_boxes + 1))
        seekers = "".join(
            self._xml_agent(
                ak,
                [0, 0],
                0,
                ((1.0, 0.35, 0.35) if ak == self.learnable_agent_key else (0.75, 0.10, 0.10)),
            )
            for ak in self.seeker_keys
        )
        hiders = "".join(
            self._xml_agent(
                ak,
                [0, 0],
                0,
                ((0.35, 0.7, 1.0) if ak == self.learnable_agent_key else (0.1, 0.2 + 0.4 * (i % 2), 0.9)),
            )
            for i, ak in enumerate(self.hider_keys)
        )
        acts = "".join(self._xml_actuators_fixed(ak) for ak in self.agent_keys)

        ramp_mesh_vertex = "-0.6666 -0.5 0.0 0.6666 -0.5 0.0 0.6666 -0.5 1.0 " + "-0.6666 0.5 0.0 0.6666 0.5 0.0 0.6666 0.5 1.0"

        return f"""
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
    <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>
        <mesh name="ramp_mesh"
                    vertex="{ramp_mesh_vertex}"
                    face="0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0"/>
    </asset>
    <worldbody>
        <light pos="0 0 12" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <camera name="overview" pos="0 13 13" euler="2.35 0 -3.14" mode="fixed" />
        {arena} {ramps} {boxes} {seekers} {hiders}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""

    def _xml_static_scene(self):
        s = self.ARENA_HALF
        attr = 'friction="0.05 0.05 0.05" solref="0.01 1" solimp="0.95 0.99 0.001"'
        return f"""
    <geom name="floor" type="plane" size="{s} {s} 0.1" material="grid" friction="1.0 0.05 0.0001"/>
    <geom name="wall_n" type="box" size="{s + 0.15} 0.1 2.0" pos="0 6.1 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_s" type="box" size="{s + 0.15} 0.1 2.0" pos="0 -6.1 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_e" type="box" size="0.1 {s} 2.0" pos="6.1 0 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_w" type="box" size="0.1 {s} 2.0" pos="-6.1 0 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="maze_w0" type="box" size="1.4 0.2 0.5" pos="3.0 1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w1" type="box" size="1.4 0.2 0.5" pos="-3.0 -1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w2" type="box" size="0.2 1.4 0.5" pos="0.0 -3.0 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w3" type="box" size="0.2 1.4 0.5" pos="0.0 3.0 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_corner0" type="cylinder" size="0.2 0.5" pos="1.6 1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_corner1" type="cylinder" size="0.2 0.5" pos="4.4 1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_corner2" type="cylinder" size="0.2 0.5" pos="-4.4 -1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_corner3" type="cylinder" size="0.2 0.5" pos="-1.6 -1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_corner4" type="cylinder" size="0.2 0.5" pos="0 -4.4 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_corner5" type="cylinder" size="0.2 0.5" pos="0 -1.6 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_corner6" type="cylinder" size="0.2 0.5" pos="0 1.6 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_corner7" type="cylinder" size="0.2 0.5" pos="0 4.4 0.5" rgba="0 0.7 0.7 1" {attr}/>
"""

    def _xml_ramp(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        inner_weight = (
            f'<geom name="ramp{i}_inner_weight" type="box" size="0.3333 0.5 0.25" '
            f'pos="0.3333 0 0.25" rgba="0 1 0 0.3" mass="{self.RAMP_INNER_WEIGHT_MASS}" '
            'solimp="0.95 0.99 0.001" friction="1.35 0.22 0.01"/>'
        )
        return f"""
        <body name="ramp{i}_body" pos="{xy[0]} {xy[1]} 0" quat="{q}">
            <inertial pos="0.3 0 0.25" mass="{self.RAMP_MASS}" diaginertia="10 10 20"/>
            <joint name="ramp{i}_joint_x" type="slide" axis="1 0 0" damping="{self.RAMP_JOINT_DAMPING}"/>
            <joint name="ramp{i}_joint_y" type="slide" axis="0 1 0" damping="{self.RAMP_JOINT_DAMPING}"/>
            <joint name="ramp{i}_joint_z" type="hinge" axis="0 0 1" damping="{self.RAMP_JOINT_DAMPING}"/>
            <geom name="ramp{i}_geom" type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0" rgba="0 1 0 1"/>
            <geom name="ramp{i}_slope_surface" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.3" friction="1.35 0.22 0.01"/>
            <geom name="ramp{i}_back_panel" type="box" size="0.02 0.5 0.5" pos="0.6666 0 0.5" rgba="0 1 0 0.3" friction="1.35 0.22 0.01"/>
            {inner_weight}
    </body>"""

    def _xml_box(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        return f"""
    <body name="box{i}_body" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
            <joint name="box{i}_joint" type="free" damping="{self.BOX_JOINT_DAMPING}"/>
            <geom name="box{i}_geom" type="box" size="0.6 0.6 0.5" mass="{self.BOX_MASS}" solref="0.02 1" condim="3" rgba="0.75 0.55 0.3 1" friction="1.2 0.08 0.003"/>
    </body>"""

    def _default_box_positions(self):
        # deterministic positions around a circle, away from the origin
        # keeps boxes from spawning directly above agents at reset
        radius = min(max(2.5, self.ARENA_HALF - 2.0), self.ARENA_HALF - 1.0)
        angles = np.linspace(0.0, 2.0 * math.pi, num=max(1, self.n_boxes), endpoint=False)
        poses = [[float(math.cos(a) * radius), float(math.sin(a) * radius)] for a in angles]
        # if n_boxes==0 this returns [], but caller won't iterate
        return poses

    def _xml_agent(self, pre, xy, rot, color):
        q = _euler_z_to_quat(rot)
        r, g, b = color
        return f"""
    <body name="{pre}_anchor" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
      <joint name="{pre}_x" type="slide" axis="1 0 0" damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{pre}_y" type="slide" axis="0 1 0" damping="{self.AGENT_DAMPING_XY}"/>
    <joint name="{pre}_z" type="slide" axis="0 0 1" damping="{self.AGENT_DAMPING_Z}" limited="true" range="{self.AGENT_Z_MIN} {self.AGENT_Z_MAX}"/>
    <joint name="{pre}_rot" type="hinge" axis="0 0 1" damping="{self.AGENT_DAMPING_ROT}" armature="3.0"/>
      <body name="{pre}_body">
        <site name="{pre}_thrust" pos="0 0 0"/>
        <geom name="{pre}_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="{self.AGENT_BOTTOM_MASS}" friction="1.2 0.12 0.003"/>
        <geom name="{pre}_capsule" type="capsule" size="0.3 0.2" rgba="{r} {g} {b} 1" mass="{self.AGENT_HEAD_MASS}" contype="0" conaffinity="0"/>
        <geom name="{pre}_nose" type="capsule" fromto="0 0 0.3 0.3 0 0.3" size="0.09" rgba="1 1 1 1" contype="0" conaffinity="0"/>
        <geom name="{pre}_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="{r} {g} {b} 1" contype="0" conaffinity="0"/>
      </body>
    </body>"""

    def _xml_actuators(self, pre):
        return f"""
    <general name="{pre}_fwd" site="{pre}_thrust" gear="1 0 0 0 0 0" gainprm="{self.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{pre}_turn" joint="{pre}_rot" gear="0.5" gainprm="{self.AGENT_ACTUATOR_TURN_GAIN}" ctrlrange="-1 1"/>
    """

    def _analyze_structure(self):
        m = self.model
        for ak in self.agent_keys:
            self.body_ids[ak] = m.body(f"{ak}_body").id
            self.qpos_indices[ak] = {
                "x": m.joint(f"{ak}_x").id,
                "y": m.joint(f"{ak}_y").id,
                "z": m.joint(f"{ak}_z").id,
                "rot": m.joint(f"{ak}_rot").id,
            }
            self.actuator_ids[f"{ak}_fwd"] = m.actuator(f"{ak}_fwd").id
            self.actuator_ids[f"{ak}_turn"] = m.actuator(f"{ak}_turn").id
        # cache agent geom ids and default colors for runtime recoloring
        self.agent_geom_ids = {}
        self.agent_default_rgba = {}
        for ak in self.agent_keys:
            geom_names = [f"{ak}_capsule", f"{ak}_nose", f"{ak}_tail"]
            ids = []
            cols = []
            for gname in geom_names:
                try:
                    gid = m.geom(gname).id
                    ids.append(gid)
                    cols.append(m.geom_rgba[gid].copy())
                except Exception:
                    logger.exception("failed to lookup geom or rgba in _analyze_structure")
                    raise
            self.agent_geom_ids[ak] = ids
            # store per-geom default rgba array list
            self.agent_default_rgba[ak] = cols
        self.ramp_ids = [m.body(f"ramp{i}_body").id for i in range(1, self.n_ramps + 1)]
        self.ramp_keys = [f"ramp{i}" for i in range(1, self.n_ramps + 1)]
        self.box_ids = [m.body(f"box{i}_body").id for i in range(1, self.n_boxes + 1)]
        self.obj_body_map = {f"b{i}": bid for i, bid in enumerate(self.box_ids, start=1)}
        for i, rid in enumerate(self.ramp_ids, start=1):
            self.obj_body_map[f"ramp{i}"] = rid
        self.obj_geom_ids = {f"b{i}": [m.geom(f"box{i}_geom").id] for i in range(1, self.n_boxes + 1)}
        for i in range(1, self.n_ramps + 1):
            self.obj_geom_ids[f"ramp{i}"] = [m.geom(f"ramp{i}_geom").id]
        self.obj_default_rgba = {k: m.geom_rgba[v[0]].copy() for k, v in self.obj_geom_ids.items()}

    def _init_agent_intelligence(self):
        self.npcs = {ak: (RuleBasedSeeker() if ak.startswith("s") else RuleBasedHider()) for ak in self.agent_keys}

    def _init_interaction_state(self):
        self.object_state = {
            tk: {
                "mode": "free",
                "owner": None,
                "locked_pose": None,
                "planar_z": None,
                "planar_quat": None,
            }
            for tk in self.obj_body_map
        }
        self.prev_action_btns = {ak: np.zeros(2, dtype=np.float32) for ak in self.agent_keys}
        self.btn_cooldown = dict.fromkeys(self.agent_keys, 0)

    # --- debug_mode property to keep logger.enabled synced ---
    @property
    def debug_mode(self):
        return bool(getattr(self, "_debug_mode", False))

    @debug_mode.setter
    def debug_mode(self, v):
        self._debug_mode = bool(v)
        if hasattr(self, "debug_logger") and self.debug_logger is not None:
            self.debug_logger.enabled = bool(v)

    @property
    def debug_console(self):
        return bool(getattr(self, "debug_logger", None) and getattr(self.debug_logger, "console", False))

    @debug_console.setter
    def debug_console(self, v):
        if hasattr(self, "debug_logger") and self.debug_logger is not None:
            self.debug_logger.console = bool(v)

    def _cache_planar_object_pose(self):
        # 実行中にデータ形状が変わることはないので、外部で一度 shape を取っておく
        qpos_len = self.data.qpos.shape[0]
        xquat_len = self.data.xquat.shape[0]

        for tk, bid in self.obj_body_map.items():
            qadr, _ = self._obj_addr(tk)

            # z軸の基準設定
            self.object_state[tk]["planar_z"] = 0.0 if tk.startswith("ramp") else 0.5

            # オブジェクトがロック状態なら、planar lock は常に水平向きの恒等クォータニオンを使う
            if self.object_state[tk].get("mode") == "locked":
                pq = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                self.object_state[tk]["planar_quat"] = pq
                if self.debug_mode:
                    self.debug_logger.print(f"[DEBUG][_cache_planar_object_pose] tk={tk} mode=locked -> forcing identity planar_quat")
                continue
            if tk[0] in ("b", "r"):  # 'b'(box/b1), 'r'(ramp) に合致
                pq = None

                # 1. xquat 優先 (もっとも直接的で安全)
                if bid < xquat_len:
                    xq = self.data.xquat[bid]
                    # 基本的に MuJoCo の xquat は常に size 4 です
                    pq = xq.copy()

                # 2. qpos スライス (fallback)
                if pq is None and qadr >= 0:
                    if tk.startswith("ramp"):
                        # ramp は slide x/y + hinge z になったため qpos=[x,y,rot]
                        if (qadr + 3) <= qpos_len:
                            rot = float(self.data.qpos[qadr + 2])
                            pq = np.array([math.cos(rot / 2.0), 0.0, 0.0, math.sin(rot / 2.0)], dtype=np.float32)
                    else:
                        if (qadr + 7) <= qpos_len:
                            pq = self.data.qpos[qadr + 3 : qadr + 7].copy()

                self.object_state[tk]["planar_quat"] = pq

                # デバッグログは失敗時のみ、かつ簡潔に
                if self.debug_mode and pq is None:
                    self.debug_logger.print(f"[DEBUG] Pose cache failed for {tk}: bid={bid}, qadr={qadr}")
            else:
                # Agent
                self.object_state[tk]["planar_quat"] = None

    def _obj_addr(self, obj_key):
        bid = self.obj_body_map[obj_key]
        jadr = self.model.body_jntadr[bid]
        return self.model.jnt_qposadr[jadr], self.model.jnt_dofadr[jadr]

    def _body_speed_xy(self, bid):
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[bid]]
        qlen = self.data.qvel.shape[0]
        vx = self.data.qvel[vadr] if vadr < qlen else 0.0
        vy = self.data.qvel[vadr + 1] if (vadr + 1) < qlen else 0.0
        return math.sqrt(vx**2 + vy**2)

    def _ramp_uphill_dir(self, rid):
        quat = self.data.xquat[rid]
        yaw = math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)

    def _compute_ramp_lx_bounds(self, rid, up):
        """
        Compute lx bounds (min, max) in ramp-local uphill coordinates.
        Attempts to use geom positions if available; falls back to sensible defaults.
        Returns (lx_min, lx_max).
        """
        # try to map rid -> rkey (ramp1..n)
        try:
            idx = int(self.ramp_ids.index(rid))
            rkey = f"ramp{idx+1}"
        except Exception:
            rkey = None
        geom_ids = []
        if rkey is not None:
            geom_ids = self.obj_geom_ids.get(rkey, [])

        # try to read geom_size for a representative geom
        gsize = None
        for gid in geom_ids:
            try:
                gsize = np.asarray(self.model.geom_size[gid], dtype=np.float32)
                if gsize.size > 0:
                    break
            except Exception:
                gsize = None

        # determine ramp length L and width W
        if gsize is not None and gsize.size >= 1:
            # MuJoCo geom_size is half-extents; assume index 0 is along ramp length
            half_length = float(gsize[0])
            L = 2.0 * half_length
            half_width = float(gsize[1]) if gsize.size > 1 else 0.5
        else:
            # fall back to class constant if present
            try:
                L = float(self.R_RAMP)
            except Exception:
                L = 0.833
            half_length = 0.5 * L
            # fallback width
            half_width = 0.5

        # compute pitch from quaternion (safe asin argument clamp)
        q = self.data.xquat[rid]
        # q = [w, x, y, z]
        sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
        if sinp >= 1.0:
            pitch = math.pi / 2
        elif sinp <= -1.0:
            pitch = -math.pi / 2
        else:
            pitch = math.asin(sinp)

        # projected half-length on XY plane
        projected_half = half_length * float(math.cos(pitch))
        # allow boosting from slightly before the ramp to overcome the gap
        # below the ramp: subtract an approach margin roughly equal to the
        # agent radius (≈0.4m). Also keep a small numerical EPS margin.
        APPROACH_MARGIN = 0.4
        EPS_MARGIN = 0.0
        lx_min = -projected_half - APPROACH_MARGIN - EPS_MARGIN
        lx_max = projected_half + EPS_MARGIN
        ly_thresh = half_width + 0.05
        return lx_min, lx_max, ly_thresh

    def _compute_ramp_top_z(self, rid):
        """
        Estimate ramp top z (world) from geom_size and quaternion.
        Returns a single float (top z). Falls back to body z if computation fails.
        """
        try:
            # Prefer using the explicit slope surface geom if present; this geom
            # gives a reliable center position and half-length along the ramp.
            slope_name = None
            try:
                idx = int(self.ramp_ids.index(rid))
                slope_name = f"ramp{idx+1}_slope_surface"
            except Exception:
                slope_name = None

            half_length = None
            geom_center_z = None
            geom_thickness = 0.0
            if slope_name is not None:
                try:
                    gid = self.model.geom(slope_name).id
                    gsize = np.asarray(self.model.geom_size[gid], dtype=np.float32)
                    if gsize.size >= 1:
                        half_length = float(gsize[0])
                    if gsize.size >= 3:
                        geom_thickness = float(gsize[2])
                    # use data.geom_xpos for current world position of geom center
                    geom_center_z = float(self.data.geom_xpos[gid][2])
                except Exception:
                    half_length = None
                    geom_center_z = None

            # fallback to body-based estimate if slope geom not available
            if half_length is None:
                try:
                    # try to find a representative geom size from stored obj_geom_ids
                    idx = int(self.ramp_ids.index(rid))
                    rkey = f"ramp{idx+1}"
                    geom_ids = self.obj_geom_ids.get(rkey, [])
                except Exception:
                    geom_ids = []
                gsize = None
                for gid in geom_ids:
                    try:
                        gsize = np.asarray(self.model.geom_size[gid], dtype=np.float32)
                        if gsize.size > 0:
                            break
                    except Exception:
                        gsize = None
                if gsize is not None and gsize.size >= 1:
                    half_length = float(gsize[0])
                else:
                    try:
                        L = float(self.R_RAMP)
                    except Exception:
                        L = 0.833
                    half_length = 0.5 * L
                geom_center_z = float(self.data.xpos[rid][2])

            # compute pitch from body quaternion (same as before)
            q = self.data.xquat[rid]
            sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
            if sinp >= 1.0:
                pitch = math.pi / 2
            elif sinp <= -1.0:
                pitch = -math.pi / 2
            else:
                pitch = math.asin(sinp)

            # top is at geom center + absolute vertical projection of half_length
            vertical_half = abs(half_length * float(math.sin(pitch)))
            # include half-thickness of the slope geom if available
            top_z = float(geom_center_z) + vertical_half + float(geom_thickness)
            return top_z
        except Exception:
            try:
                return float(self.data.xpos[rid][2])
            except Exception:
                return 0.0

    def _is_ramp_blocked_or_locked(self, ramp_key, rid):
        if self.object_state[ramp_key]["mode"] == "locked":
            return True
        rpos = self.data.xpos[rid][:2]
        rspeed = self._body_speed_xy(rid)
        if rspeed > self.RAMP_LOCKED_SPEED_EPS:
            return False
        # 静止しているランプが壁・箱に背中を預けているかを簡易判定
        margin = 0.9
        if abs(rpos[0]) > self.ARENA_HALF - margin or abs(rpos[1]) > self.ARENA_HALF - margin:
            return True
        for bx in self.box_ids:
            if np.linalg.norm(self.data.xpos[bx][:2] - rpos) < 1.7:
                return True
        for wx, wy, sx, sy in self.maze_walls:
            if abs(rpos[0] - wx) <= sx + 0.9 and abs(rpos[1] - wy) <= sy + 0.9:
                return True
        return False

    def _ramp_boost_gain(self, ak):
        apos = self.data.xpos[self.body_ids[ak]][:2]
        arot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]]
        afwd = np.array([math.cos(arot), math.sin(arot)], dtype=np.float32)
        gain = 0.0
        per_ramp = []
        for i, rid in enumerate(self.ramp_ids, start=1):
            rkey = f"ramp{i}"
            # ランプが他のエージェントにホールドされている（locked + owner が存在する）場合のみ除外する。
            # そうでなければ（壁際で固定されている等のケースを含む）判定対象にする。
            st = self.object_state.get(rkey, {})
            # エージェントが持ち運んでいる（grabbed）ランプは除外する。
            if st.get("mode") == "grabbed" and st.get("owner") is not None:
                continue
            rpos = self.data.xpos[rid][:2]
            up = self._ramp_uphill_dir(rid)
            side = np.array([-up[1], up[0]], dtype=np.float32)
            rel = apos - rpos
            lx = float(np.dot(rel, up))
            ly = float(np.dot(rel, side))
            facing = float(np.dot(afwd, up))
            # record this ramp candidate for diagnostics
            # compute dynamic bounds and record this ramp candidate for diagnostics
            try:
                lx_min, lx_max, ly_thresh = self._compute_ramp_lx_bounds(rid, up)
            except Exception:
                lx_min, lx_max, ly_thresh = -1.15, 0.666, 0.95
            full_thresh = lx_min + 0.5 * (lx_max - lx_min)
            per_ramp.append(
                {
                    "rkey": rkey,
                    "lx": lx,
                    "ly": ly,
                    "facing": facing,
                    "lx_min": lx_min,
                    "lx_max": lx_max,
                    "ly_thresh": ly_thresh,
                    "full_thresh": full_thresh,
                }
            )

            if abs(ly) > ly_thresh or facing < 0.55:
                continue
            # determine gain: full boost closer to ramp start, partial boost near top
            if lx_min <= lx <= full_thresh:
                gain = max(gain, 1.0)
            elif full_thresh < lx <= lx_max:
                gain = max(gain, 0.6)

        # detect toggle and print diagnostics when boost value changes for this agent
        # apply hysteresis: set_thresh to engage, clear_thresh to disengage
        SET_THRESH = 0.6
        CLEAR_THRESH = 0.4
        try:
            prev = float(self._last_ramp_boost.get(ak, 0.0))
        except Exception:
            prev = 0.0
        prev_on = prev >= SET_THRESH
        cand_on = gain >= SET_THRESH
        # determine final_on using hysteresis and short hold to avoid spurious clears
        try:
            hold = int(self._ramp_boost_hold.get(ak, 0))
        except Exception:
            hold = 0

        # normal hysteresis decision
        if prev_on:
            would_clear = gain < CLEAR_THRESH
        else:
            would_clear = False

        # if previously on and we would clear now, start/continue a short hold
        if prev_on and would_clear:
            if hold <= 0:
                # begin hold period
                hold = int(self.RAMP_BOOST_HOLD_STEPS)
            # keep boost on during hold
            final_on = True
        elif hold > 0:
            # decrement hold while forcing boost on
            hold = max(0, hold - 1)
            final_on = True
        else:
            # not in hold, use standard hysteresis engage rule
            final_on = cand_on

        # choose magnitude: prefer current gain; do not preserve previous magnitude
        # when current gain is zero (prevents boost magnitude lingering after exit)
        if final_on:
            final_gain = gain if gain > 0.0 else 0.0
        else:
            final_gain = 0.0

        # persist hold state
        try:
            self._ramp_boost_hold[ak] = int(hold)
        except Exception:
            pass

        # update last seen boost magnitude
        try:
            self._last_ramp_boost[ak] = float(final_gain)
        except Exception:
            pass

        return final_gain

    def _stabilize_agent_vertical_motion(self):
        for ak in self.agent_keys:
            jz = self.qpos_indices[ak]["z"]
            qz_adr = self.model.jnt_qposadr[jz]
            vz_adr = self.model.jnt_dofadr[jz]
            z = float(self.data.qpos[qz_adr])
            vz = float(self.data.qvel[vz_adr])

            if z > self.AGENT_Z_MAX:
                self.data.qpos[qz_adr] = self.AGENT_Z_MAX
                if vz > 0.0:
                    self.data.qvel[vz_adr] = 0.0
            elif z < self.AGENT_Z_MIN:
                self.data.qpos[qz_adr] = self.AGENT_Z_MIN
                if vz < 0.0:
                    self.data.qvel[vz_adr] = 0.0
            else:
                if vz > self.AGENT_MAX_VZ:
                    self.data.qvel[vz_adr] = self.AGENT_MAX_VZ
                elif vz < -self.AGENT_MAX_VZ:
                    self.data.qvel[vz_adr] = -self.AGENT_MAX_VZ

    def _interaction_blocked_by_static_walls(self, p1, p2):
        return bool(
            _blocked_by_walls_numba(
                float(p1[0]),
                float(p1[1]),
                float(p2[0]),
                float(p2[1]),
                self.static_wall_aabbs,
                float(self.INTERACT_OCCLUSION_MARGIN),
            )
        )

    def _select_target(self, ak, for_grab=False):
        apos = self.data.xpos[self.body_ids[ak]][:2]
        rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]]
        fwd = np.array([math.cos(rot), math.sin(rot)], dtype=np.float32)
        aid = self.body_ids[ak]
        best_key, best_dist = None, 1e9
        for tk, bid in self.obj_body_map.items():
            st = self.object_state[tk]
            if for_grab and st["mode"] == "locked" and st["owner"] != ak:
                continue
            opos = self.data.xpos[bid][:2]
            rel = opos - apos
            dist = float(np.linalg.norm(rel))
            if dist > self.INTERACT_RANGE:
                continue
            if self._interaction_blocked_by_static_walls(apos, opos):
                continue
            if for_grab and float(np.dot(fwd, rel)) <= -0.2:
                continue
            if not self.vis_engine.is_visible(apos, opos, body_exclude=aid, target_body_id=bid):
                continue
            if dist < best_dist:
                best_key, best_dist = tk, dist
        return best_key

    def _current_grabbed_by(self, ak):
        for tk, st in self.object_state.items():
            if st["mode"] == "grabbed" and st["owner"] == ak:
                return tk
        return None

    def _toggle_lock(self, ak):
        tk = self._select_target(ak, for_grab=False)
        if tk is None:
            return False
        st = self.object_state[tk]
        qadr, _ = self._obj_addr(tk)
        if st["mode"] == "locked" and st["owner"] == ak:
            st["mode"], st["owner"], st["locked_pose"] = "free", None, None
            return True

        if st["mode"] in ("free", "grabbed") and (st["owner"] is None or st["owner"] == ak):
            st["mode"] = "locked"
            st["owner"] = ak
            if tk.startswith("ramp"):
                st["locked_pose"] = self.data.qpos[qadr : qadr + 3].copy()
            else:
                st["locked_pose"] = self.data.qpos[qadr : qadr + 7].copy()
            return True

        return False

    def _toggle_grab(self, ak):
        cur = self._current_grabbed_by(ak)
        tk = self._select_target(ak, for_grab=True)

        if cur is not None:
            aid = self.body_ids[ak]
            cid = self.obj_body_map[cur]
            apos = self.data.xpos[aid][:2]
            cpos = self.data.xpos[cid][:2]
            if self._interaction_blocked_by_static_walls(apos, cpos):
                return False
            if not self.vis_engine.is_visible(apos, cpos, body_exclude=aid, target_body_id=cid):
                return False

        if cur is not None and (tk is None or tk == cur):
            self.object_state[cur]["mode"] = "free"
            self.object_state[cur]["owner"] = None
            return True
        if tk is None:
            return False
        st = self.object_state[tk]
        if st["mode"] == "free":
            if cur is not None and cur != tk:
                self.object_state[cur]["mode"] = "free"
                self.object_state[cur]["owner"] = None
            st["mode"] = "grabbed"
            st["owner"] = ak
            return True
        return False

    def _handle_buttons(self, ak, lock_btn, grab_btn):
        prev = self.prev_action_btns[ak]
        lock_edge = lock_btn > self.BTN_ON and prev[0] <= self.BTN_ON
        grab_edge = grab_btn > self.BTN_ON and prev[1] <= self.BTN_ON
        self.prev_action_btns[ak][0] = lock_btn
        self.prev_action_btns[ak][1] = grab_btn
        if self.btn_cooldown[ak] > 0:
            return False, False
        if lock_edge:
            lock_evt = self._toggle_lock(ak)
            self.btn_cooldown[ak] = self.BTN_COOLDOWN
            return lock_evt, False
        if grab_edge:
            grab_evt = self._toggle_grab(ak)
            self.btn_cooldown[ak] = self.BTN_COOLDOWN
            return False, grab_evt
        return False, False

    def _apply_object_constraints(self):
        """
        より単純なGrab/Lock拘束ロジック。
        - Grab: ownerの位置に線形追従（弱い溶接的）
        - Lock: pose固定
        - Free: 減衰のみ
        マジックナンバーは定数化。
        """
        # --- 定数定義 ---
        # POSE_SIZE / VEL_SIZE はオブジェクト毎に変わる（ramp は 3、box は 7）
        XY_START, XY_STOP = 0, 2  # x, y成分
        # 以下のインデックスはオブジェクト種別に応じてループ内で決定する

        for ak in self.agent_keys:
            if self.btn_cooldown[ak] > 0:
                self.btn_cooldown[ak] -= 1
        for tk, st in self.object_state.items():
            qadr, vadr = self._obj_addr(tk)
            is_ramp = tk.startswith("ramp")
            pose_size = 3 if is_ramp else 7
            vel_size = 3 if is_ramp else 6
            # velocity index layout per-object
            XY_VEL_START, XY_VEL_STOP = 0, 2
            if is_ramp:
                Z_VEL_IDX = 2
                # ramps have no separate angular-velocity block beyond index 2
                ANG_VEL_START, ANG_VEL_STOP = 2, 3
            else:
                Z_VEL_IDX = 2
                ANG_VEL_START, ANG_VEL_STOP = 3, 6

            if st["mode"] == "locked" and st["locked_pose"] is not None:
                # locked_pose は object 作成時の pose_size に合わせて保存される
                self.data.qpos[qadr : qadr + pose_size] = st["locked_pose"]
                self.data.qvel[vadr : vadr + vel_size] = 0.0
            elif st["mode"] == "grabbed" and st["owner"] is not None:
                owner = st["owner"]
                opos = self.data.xpos[self.body_ids[owner]][XY_START:XY_STOP]
                cur_xy = self.data.qpos[qadr : qadr + pose_size][XY_START:XY_STOP].copy()
                # --- 距離・壁判定によるgrab解除 ---
                grab_dist = float(np.linalg.norm(cur_xy - opos))
                if grab_dist > self.GRAB_BREAK_DIST or self._interaction_blocked_by_static_walls(opos, cur_xy):
                    st["mode"] = "free"
                    st["owner"] = None
                    self.data.qvel[vadr : vadr + vel_size][XY_VEL_START:XY_VEL_STOP] *= 0.5
                    continue
                err_xy = opos - cur_xy
                self.data.qvel[vadr : vadr + vel_size][XY_VEL_START:XY_VEL_STOP] = err_xy  # 速度を直接目標方向に
                # Z 軸速度（垂直 or hinge 回転）を 0 にする
                self.data.qvel[vadr : vadr + vel_size][Z_VEL_IDX] = 0.0
                # 角速度ブロックがある場合は 0 にする（boxes: indices 3..6、ramps は短い範囲）
                self.data.qvel[vadr : vadr + vel_size][ANG_VEL_START:ANG_VEL_STOP] = 0.0
            else:
                self.data.qvel[vadr : vadr + vel_size][XY_VEL_START:XY_VEL_STOP] *= 0.9
                speed_xy = float(np.linalg.norm(self.data.qvel[vadr : vadr + vel_size][XY_VEL_START:XY_VEL_STOP]))
                if speed_xy < 1e-3:
                    self.data.qvel[vadr : vadr + vel_size][XY_VEL_START:XY_VEL_STOP] = 0.0
            # Planar lock
            if self.OBJECT_PLANAR_LOCK and st.get("planar_z") is not None:
                pq = st.get("planar_quat")
                try:
                    if is_ramp:
                        # ramps: qpos layout = [x, y, rot]
                        # planar_z は ground 用に保存されているが qpos に高さ成分はないためスキップ
                        # 回転は planar_quat から yaw を計算して hinge 値に適用する
                        if pq is not None and np.all(np.isfinite(pq)) and len(pq) >= 4:
                            w, xq, yq, zq = float(pq[0]), float(pq[1]), float(pq[2]), float(pq[3])
                            yaw = math.atan2(2.0 * (w * zq + xq * yq), 1.0 - 2.0 * (yq * yq + zq * zq))
                            # hinge joint qpos is at qadr + 2
                            if (qadr + 3) <= self.data.qpos.shape[0]:
                                self.data.qpos[qadr + 2] = yaw
                        else:
                            if self.debug_mode:
                                self.debug_logger.print(f"[DEBUG][_apply_object_constraints] skipping invalid planar_quat for ramp {tk}: {pq}")
                    else:
                        # boxes / free bodies: set z and quaternion in qpos
                        self.data.qpos[qadr + 2] = st["planar_z"]
                        if pq is not None and np.all(np.isfinite(pq)) and len(pq) >= 4:
                            self.data.qpos[qadr + 3 : qadr + 7] = pq
                        else:
                            if self.debug_mode:
                                self.debug_logger.print(f"[DEBUG][_apply_object_constraints] skipping invalid planar_quat for {tk}: {pq}")
                except Exception:
                    if self.debug_mode:
                        self.debug_logger.print(f"[DEBUG][_apply_object_constraints] exception validating planar_quat for {tk}")
                # zero velocities for the vertical/rotational components
                self.data.qvel[vadr : vadr + vel_size][Z_VEL_IDX] = 0.0
                self.data.qvel[vadr : vadr + vel_size][ANG_VEL_START:ANG_VEL_STOP] = 0.0
        for tk, geom_ids in self.obj_geom_ids.items():
            mode = self.object_state[tk]["mode"]
            if mode == "locked":
                rgba = np.array([0.2, 0.2, 0.2, 1.0])
            elif mode == "grabbed":
                rgba = np.array([1.0, 0.85, 0.1, 1.0])
            else:
                rgba = self.obj_default_rgba[tk]
            for gid in geom_ids:
                self.model.geom_rgba[gid] = rgba

    def _policy_action(self, agent_key, norm_obs):
        """モデルが設定されている場合は必ずそのモデルを使う。"""
        model = self._inference_models.get(agent_key)
        if model is not None:
            self._log_policy_src(agent_key, "model")
            seq_len = int(self._inference_seq_lens.get(agent_key, 8))
            seq_np = self._get_policy_history_seq(agent_key, seq_len, norm_obs)
            seq_t = torch.as_tensor(seq_np[None, :, :], dtype=torch.float32)
            with torch.no_grad():
                if self.model_policy_deterministic and hasattr(model, "get_deterministic_action_and_value"):
                    out = model.get_deterministic_action_and_value(seq_t)
                else:
                    out = model.get_action_and_value(seq_t)
                arr = out[0].cpu().numpy().reshape(-1)
            if arr.size >= 4:
                return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            if arr.size >= 2:
                return float(arr[0]), float(arr[1]), 0.0, 0.0
            raise RuntimeError(f"Invalid model action size: agent={agent_key}, size={arr.size}")

        policy = self.inference_policies.get(agent_key)
        if policy is not None:
            try:
                self._log_policy_src(agent_key, "callable")
                pred = policy(norm_obs)
                arr = np.asarray(pred).reshape(-1)
                if arr.size >= 4:
                    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
                if arr.size >= 2:
                    return float(arr[0]), float(arr[1]), 0.0, 0.0
            except Exception:
                pass

        if self.shared_team_policy and self.shared_policy_model is not None and agent_key != self.learnable_agent_key and agent_key.startswith(self.shared_team_prefix):
            self._log_policy_src(agent_key, "shared_model")
            seq_np = self._get_policy_history_seq(agent_key, self.shared_policy_seq_len, norm_obs)
            seq_t = torch.as_tensor(seq_np[None, :, :], dtype=torch.float32)
            with torch.no_grad():
                if self.model_policy_deterministic and hasattr(self.shared_policy_model, "get_deterministic_action_and_value"):
                    out = self.shared_policy_model.get_deterministic_action_and_value(seq_t)
                else:
                    out = self.shared_policy_model.get_action_and_value(seq_t)
                arr = out[0].cpu().numpy().reshape(-1)
            if arr.size >= 4:
                return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            if arr.size >= 2:
                return float(arr[0]), float(arr[1]), 0.0, 0.0
            raise RuntimeError(f"Invalid shared action size: agent={agent_key}, size={arr.size}")

        self._log_policy_src(agent_key, "rule")
        arr = np.asarray(self.npcs[agent_key].get_action(norm_obs, self.idx)).reshape(-1)
        if arr.size >= 4:
            return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        return float(arr[0]), float(arr[1]), 0.0, 0.0

    def _log_policy_src(self, agent_key, source):
        if not self.debug_mode:
            return
        self.debug_logger.log_policy_src(agent_key, source, self.current_step, force=True)

    def _dbg_print_throttled(self, key, message, force=False):
        self.debug_logger.print_throttled(key, message, self.current_step, force=force)

    def _is_spawn_position_valid(self, pos_xy, radius, placed, margin):
        px = float(pos_xy[0])
        py = float(pos_xy[1])
        lim = float(self.SAFE_HALF - radius - margin)
        if abs(px) > lim or abs(py) > lim:
            return False
        # 内壁（maze_walls）との重なりチェック
        for wx, wy, sx, sy in self.maze_walls:
            if abs(px - wx) <= (sx + radius + margin) and abs(py - wy) <= (sy + radius + margin):
                return False
        # static_wall_aabbs（外壁・内壁）との重なりチェック
        for cx, cy, hx, hy in getattr(self, "static_wall_aabbs", []):
            if abs(px - cx) <= (hx + radius + margin) and abs(py - cy) <= (hy + radius + margin):
                return False
        for pp, pr in placed:
            if np.linalg.norm(pos_xy - pp) < (radius + pr + margin):
                return False
        return True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._policy_histories.clear()
        self.debug_logger.clear_last_step()
        if self.debug_mode:
            self.debug_logger.clear_policy_src_log()
        mujoco.mj_resetData(self.model, self.data)
        if self.debug_mode:
            # quick sanity checks after reset
            try:
                qpos = self.data.qpos
                qvel = self.data.qvel
                self.debug_logger.print(f"[DEBUG][reset] qpos.shape={qpos.shape} qvel.shape={qvel.shape}")
                if np.any(~np.isfinite(qpos)):
                    self.debug_logger.print(f"[DEBUG][reset] qpos contains non-finite values at indices: {np.where(~np.isfinite(qpos))[0][:10]}")
                if np.any(~np.isfinite(qvel)):
                    self.debug_logger.print(f"[DEBUG][reset] qvel contains non-finite values at indices: {np.where(~np.isfinite(qvel))[0][:10]}")
            except Exception:
                logger.exception("exception while doing debug sanity checks in reset")
                raise
        self._init_agent_intelligence()
        self._init_interaction_state()
        placed = []
        ramp_specs = [(rid, self.R_RAMP, 0.0) for rid in self.ramp_ids]
        box_specs = [(b, self.R_BOX, 0.5) for b in self.box_ids]
        for bid, rad, z in ramp_specs + box_specs:
            for _ in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                if self._is_spawn_position_valid(p, rad, placed, margin=0.2):
                    try:
                        jadr = self.model.body_jntadr[bid]
                        if jadr is None or int(jadr) < 0:
                            if self.debug_mode:
                                self.debug_logger.print(f"[DEBUG][reset] invalid body_jntadr for body {bid}: {jadr}")
                            continue
                        adr = int(self.model.jnt_qposadr[jadr])
                        if adr is None or adr < 0 or (adr + 7) > self.data.qpos.shape[0]:
                            if self.debug_mode:
                                self.debug_logger.print(f"[DEBUG][reset] invalid qpos adr for body {bid}: {adr}")
                            continue
                        # ランプは free から slide x/y + hinge z に変更したため
                        # qpos の長さが異なる（3）点に注意する
                        if bid in getattr(self, "ramp_ids", []):
                            # slide x, slide y, hinge z
                            self.data.qpos[adr : adr + 3] = [p[0], p[1], 0.0]
                        else:
                            # boxes 等は従来通り free joint (pos + quat)
                            self.data.qpos[adr : adr + 7] = [p[0], p[1], z, 1, 0, 0, 0]
                        placed.append((p, rad))
                    except Exception:
                        if self.debug_mode:
                            self.debug_logger.print(f"[DEBUG][reset] exception placing body {bid}")
                        continue
                    # ランプのみx, y出力とqpos値も出力
                    # if (bid, rad, z) in ramp_specs:
                    #    print(f"[reset] ramp placed: x={p[0]:.3f}, y={p[1]:.3f}, qpos={self.data.qpos[adr]:.3f},{self.data.qpos[adr+1]:.3f}")
                    # 正常に配置できたら inner loop を抜け、次のオブジェクト配置に進む
                    break
        for ak in self.agent_keys:
            for _ in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                rot = np.random.uniform(-np.pi, np.pi)
                if self._is_spawn_position_valid(p, self.R_AGENT, placed, margin=0.3):
                    jx = self.qpos_indices[ak]["x"]
                    jy = self.qpos_indices[ak]["y"]
                    jz = self.qpos_indices[ak]["z"]
                    jr = self.qpos_indices[ak]["rot"]
                    self.data.qpos[self.model.jnt_qposadr[jx]] = p[0]
                    self.data.qpos[self.model.jnt_qposadr[jy]] = p[1]
                    self.data.qpos[self.model.jnt_qposadr[jz]] = 0.5
                    self.data.qpos[self.model.jnt_qposadr[jr]] = rot
                    placed.append((p, self.R_AGENT))
                    break
        mujoco.mj_forward(self.model, self.data)
        if self.debug_mode:
            # report any objects or agents still at origin (likely placement failure)
            try:
                origins = []
                for name, bid in self.obj_body_map.items():
                    pos = self.data.xpos[bid]
                    if np.allclose(pos, 0.0, atol=1e-6):
                        origins.append(name)
                for ak in self.agent_keys:
                    bid = self.body_ids.get(ak, None)
                    if bid is not None:
                        pos = self.data.xpos[bid]
                        if np.allclose(pos, 0.0, atol=1e-6):
                            origins.append(ak)
                if origins:
                    self.debug_logger.print(f"[DEBUG][reset] objects/agents at origin after reset: {origins}")
                # also quick qpos/qvel NaN check
                if np.any(~np.isfinite(self.data.qpos)) or np.any(~np.isfinite(self.data.qvel)):
                    self.debug_logger.print("[DEBUG][reset] Non-finite detected in qpos/qvel after forward")
            except Exception:
                pass
        # --- update prev_vis / prev_being_hit to reflect post-step state ---
        try:
            # refresh caches based on current (post-physics) positions/rotations
            self._prev_vis.clear()
            self._prev_being_hit.clear()
            for viewer in self.agent_keys:
                try:
                    v_bid = self.body_ids[viewer]
                    v_pos = self.data.xpos[v_bid][:2]
                    v_idx_r = self.model.jnt_qposadr[self.qpos_indices[viewer]["rot"]]
                    v_rot = float(self.data.qpos[v_idx_r])
                except Exception:
                    logger.exception("failed to read viewer body/joint data in reset visibility cache; skipping viewer")
                    continue
                for target in self.agent_keys:
                    if target == viewer:
                        continue
                    try:
                        t_bid = self.body_ids[target]
                        t_pos = self.data.xpos[t_bid][:2]
                    except Exception:
                        logger.exception("failed to read target body data in reset visibility cache; marking not visible")
                        self._prev_vis[(viewer, target)] = False
                        continue
                    vis = False
                    try:
                        vis = bool(self._is_vis(v_pos, v_rot, t_pos, v_bid, t_bid))
                    except Exception:
                        logger.exception("_is_vis failed while populating _prev_vis")
                        vis = False
                    self._prev_vis[(viewer, target)] = vis
                    if viewer.startswith("h") and target.startswith("s"):
                        # check whether seeker (target) sees the hider (viewer)
                        is_hit = False
                        try:
                            s_idx_r = self.model.jnt_qposadr[self.qpos_indices[target]["rot"]]
                            s_rot = float(self.data.qpos[s_idx_r])
                            try:
                                is_hit = bool(self._is_vis(t_pos, s_rot, v_pos, t_bid, v_bid))
                            except Exception:
                                logger.exception("_is_vis failed when checking seeker->hider hit in reset cache")
                                is_hit = False
                        except Exception:
                            logger.exception("failed to read seeker rotation while computing being_hit in reset cache")
                            is_hit = False
                        self._prev_being_hit[(viewer, target)] = 1.0 if is_hit else 0.0
        except Exception:
            logger.exception("failed while refreshing prev_vis/prev_being_hit caches after forward")
            raise
        # initialize previous-step visibility / being-hit caches so _get_obs
        # can use a well-defined 'previous' value immediately after reset
        self._prev_vis.clear()
        self._prev_being_hit.clear()
        for viewer in self.agent_keys:
            try:
                v_bid = self.body_ids[viewer]
                v_pos = self.data.xpos[v_bid][:2]
                v_idx_r = self.model.jnt_qposadr[self.qpos_indices[viewer]["rot"]]
                v_rot = float(self.data.qpos[v_idx_r])
            except Exception:
                logger.exception("failed to access viewer state when initializing prev_vis; skipping viewer")
                continue

            for target in self.agent_keys:
                if target == viewer:
                    continue
                try:
                    t_bid = self.body_ids[target]
                    t_pos = self.data.xpos[t_bid][:2]
                except Exception:
                    logger.exception("failed to access target state when initializing prev_vis; marking not visible")
                    self._prev_vis[(viewer, target)] = False
                    continue

                # 視界判定 (局所的に例外を捕捉)
                vis = False
                try:
                    vis = bool(self._is_vis(v_pos, v_rot, t_pos, v_bid, t_bid))
                except Exception:
                    logger.exception("_is_vis exception during initialization of prev_vis; marking not visible")
                    vis = False
                self._prev_vis[(viewer, target)] = vis

                # 被弾フラグ: viewer が Hider で target が Seeker の場合、
                # seeker の視界で自分(viewer)が見られているかを判定して保存する
                if viewer.startswith("h") and target.startswith("s"):
                    try:
                        s_idx_r = self.model.jnt_qposadr[self.qpos_indices[target]["rot"]]
                        s_rot = float(self.data.qpos[s_idx_r])
                        is_hit = False
                        try:
                            is_hit = bool(self._is_vis(t_pos, s_rot, v_pos, t_bid, v_bid))
                        except Exception:
                            is_hit = False
                    except Exception:
                        is_hit = False
                    # store as (hider, seeker)
                    self._prev_being_hit[(viewer, target)] = 1.0 if is_hit else 0.0
                    # set persistence counter when hit detected
                    if is_hit:
                        self._being_hit_counters[(viewer, target)] = int(self.being_hit_persist)

        self._cache_planar_object_pose()

        for i, ak in enumerate(self.agent_keys):
            if ak == self.learnable_agent_key and not self.override_learnable_policy:
                continue
            norm_obs = self._normalize_obs(self._get_obs(i))
            if ak in self._inference_models:
                self._prime_policy_history(
                    ak,
                    int(self._inference_seq_lens.get(ak, 8)),
                    norm_obs,
                )
            if self.shared_team_policy and self.shared_policy_model is not None and ak.startswith(self.shared_team_prefix):
                self._prime_policy_history(ak, self.shared_policy_seq_len, norm_obs)

        idx_to_obs = self.learnable_agent_idx

        # wall_distance を計算
        learnable_agent_body_id = self.body_ids[self.learnable_agent_key]
        learnable_agent_pos = self.data.xpos[learnable_agent_body_id]
        wall_dist = self.vis_engine.wall_distance(learnable_agent_pos[0], learnable_agent_pos[1])
        obs = self._normalize_obs(self._get_obs(idx_to_obs))
        # cache current observation
        self._cached_obs = obs
        return obs, {
            "is_detected": False,
            "wall_distance": wall_dist,
        }

    def step(self, action):
        self.current_step += 1
        af, cv = np.ravel(action), np.zeros(self.model.nu)
        any_lock_event = False
        any_grab_event = False
        any_lock_pressed = False
        any_grab_pressed = False
        any_lock_target = False
        any_grab_target = False
        max_lock_btn = 0.0
        max_grab_btn = 0.0
        boosted_agents = 0
        # keep both the raw model forward and the actual applied forward (after any inference-only transform)
        applied_forward_model = 0.0
        applied_forward_env = 0.0

        # (original behavior) do not zero external force buffer here; individual
        # contributors historically accumulated into `data.xfrc_applied`.

        for i, ak in enumerate(self.agent_keys):
            is_seeker = ak.startswith("s")
            if ak == self.learnable_agent_key:
                if is_seeker and self.current_step <= self.prep_steps:
                    f, t, lck, grb = 0.0, 0.0, 0.0, 0.0
                elif self.override_learnable_policy:
                    norm_obs = self._normalize_obs(self._get_obs(i))
                    f, t, lck, grb = self._policy_action(ak, norm_obs)
                else:
                    # 外部アクション（常に4要素）を適用
                    f = af[0] if len(af) > 0 else 0.0
                    t = af[1] if len(af) > 1 else 0.0
                    lck = af[2] if len(af) > 2 else 0.0
                    grb = af[3] if len(af) > 3 else 0.0
            else:
                # 非学習エージェントは推論モデル優先、なければRuleBased
                if is_seeker and self.current_step <= self.prep_steps:
                    f, t, lck, grb = 0.0, 0.0, 0.0, 0.0
                else:
                    norm_obs = self._normalize_obs(self._get_obs(i))
                    f, t, lck, grb = self._policy_action(ak, norm_obs)

            boost = self._ramp_boost_gain(ak)
            if boost > 0.0:
                f = float(np.clip(f + self.RAMP_BOOST_FWD * boost, -1.0, 1.0))
                boosted_agents += 1

            # record the final model forward for the learnable agent (before any env-side transform)
            if ak == self.learnable_agent_key:
                applied_forward_model = float(f)

            max_lock_btn = max(max_lock_btn, float(lck))
            max_grab_btn = max(max_grab_btn, float(grb))
            lock_pressed = bool(lck > self.BTN_ON)
            grab_pressed = bool(grb > self.BTN_ON)
            any_lock_pressed = any_lock_pressed or lock_pressed
            any_grab_pressed = any_grab_pressed or grab_pressed
            if lock_pressed and self._select_target(ak, for_grab=False) is not None:
                any_lock_target = True
            if grab_pressed and self._select_target(ak, for_grab=True) is not None:
                any_grab_target = True

            lock_evt, grab_evt = self._handle_buttons(ak, lck, grb)
            any_lock_event = any_lock_event or lock_evt
            any_grab_event = any_grab_event or grab_evt

            # For inference-only compatibility: some checkpoints were trained with the
            # opposite sign convention for the forward channel. Apply a runtime-only
            # transform controlled by `INFERENCE_FORWARD_SIGN` so this behavior can be
            # toggled without editing logic below.
            if ak == self.learnable_agent_key and self._inference_models.get(ak) is not None:
                f_env = float(self.INFERENCE_FORWARD_SIGN * f)
            else:
                f_env = float(f)

            applied_forward_env = f_env
            # Turn channel: no env-side scaling — let学習で回転を学ばせる
            t_env = float(t)
            cv[self.actuator_ids[f"{ak}_fwd"]], cv[self.actuator_ids[f"{ak}_turn"]] = (
                f_env,
                t_env,
            )
            self.last_debug_ctrl[ak] = (f_env, t_env)

        self.data.ctrl[:] = cv
        # --- capture pre-physics visibility / being-hit state (used as 'previous' freshness) ---
        try:
            self._prev_vis.clear()
            self._prev_being_hit.clear()
            for viewer in self.agent_keys:
                v_bid = self.body_ids[viewer]
                v_pos = self.data.xpos[v_bid][:2]
                v_rot = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[viewer]["rot"]]])
                for target in self.agent_keys:
                    if target == viewer:
                        continue
                    t_bid = self.body_ids[target]
                    t_pos = self.data.xpos[t_bid][:2]
                    try:
                        vis = bool(self._is_vis(v_pos, v_rot, t_pos, v_bid, t_bid))
                    except Exception:
                        vis = False
                    self._prev_vis[(viewer, target)] = vis
                    # being_hit: store as (hider, seeker) -> flag
                    if viewer.startswith("s") and target.startswith("h"):
                        self._prev_being_hit[(target, viewer)] = 1.0 if vis else 0.0
        except Exception:
            pass
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
            self._apply_object_constraints()
            self._stabilize_agent_vertical_motion()
        mujoco.mj_forward(self.model, self.data)
        if self.debug_mode:
            try:
                bad_qpos = np.where(~np.isfinite(self.data.qpos))[0]
                bad_qvel = np.where(~np.isfinite(self.data.qvel))[0]
                if bad_qpos.size or bad_qvel.size:
                    self.debug_logger.print(f"[DEBUG][step] Non-finite qpos indices: {bad_qpos}")
                    self.debug_logger.print(f"[DEBUG][step] Non-finite qvel indices: {bad_qvel}")
                    # map qpos indices back to joints/bodies where possible
                    qpos_to_j = {}
                    for jid in range(self.model.njnt):
                        try:
                            qadr = int(self.model.jnt_qposadr[jid])
                        except Exception:
                            continue
                        qpos_to_j[qadr] = jid
                    # print joint qposadr table to help map indices to joints
                    try:
                        self.debug_logger.print(f"[DEBUG][step] model.nq={int(self.model.nq)} model.njnt={int(self.model.njnt)}")
                        rows = []
                        for jid in range(min(200, int(self.model.njnt))):
                            try:
                                qadr = int(self.model.jnt_qposadr[jid])
                            except Exception:
                                qadr = -1
                            try:
                                dadr = int(self.model.jnt_dofadr[jid])
                            except Exception:
                                dadr = -1
                            rows.append((jid, qadr, dadr))
                        self.debug_logger.print("[DEBUG][step] sample joint qposadr/dofadr:")
                        for r in rows[:50]:
                            self.debug_logger.print(f"  jid={r[0]:3d} qposadr={r[1]:3d} dofadr={r[2]:3d}")
                    except Exception:
                        pass
                    # show qpos around bad indices
                    for idx in bad_qpos.tolist():
                        lo = max(0, idx - 3)
                        hi = min(self.data.qpos.shape[0], idx + 4)
                        self.debug_logger.print(f"[DEBUG][step] qpos[{lo}:{hi}] = {self.data.qpos[lo:hi]}")
            except Exception:
                pass

        box_speeds = [self._body_speed_xy(bid) for bid in self.box_ids]
        ramp_speeds = [self._body_speed_xy(rid) for rid in self.ramp_ids]
        moving_box_count = int(sum(1 for v in box_speeds if v > 0.06))
        moving_ramp_count = int(sum(1 for v in ramp_speeds if v > 0.06))
        blocked_ramp_count = int(sum(1 for i, rid in enumerate(self.ramp_ids, start=1) if self._is_ramp_blocked_or_locked(f"ramp{i}", rid)))

        rb, find, gaze_cos_front_max, gaze_dist_max, learnable_hider_seen = self._compute_team_reward()

        for i, ak in enumerate(self.agent_keys):
            if ak == self.learnable_agent_key and not self.override_learnable_policy:
                continue
            norm_obs_next = self._normalize_obs(self._get_obs(i))
            if ak in self._inference_models:
                self._update_policy_history(
                    ak,
                    int(self._inference_seq_lens.get(ak, 8)),
                    norm_obs_next,
                )
            if self.shared_team_policy and self.shared_policy_model is not None and ak.startswith(self.shared_team_prefix):
                self._update_policy_history(ak, self.shared_policy_seq_len, norm_obs_next)

        # 学習対象に合わせて観測を生成
        idx_to_obs = self.learnable_agent_idx

        # 壁までの最短距離を計算
        learnable_agent_body_id = self.body_ids[self.learnable_agent_key]
        learnable_agent_pos = self.data.xpos[learnable_agent_body_id]
        wall_dist = self.vis_engine.wall_distance(learnable_agent_pos[0], learnable_agent_pos[1])

        # z方向速度をinfoに追加
        jz = self.qpos_indices[self.learnable_agent_key]["z"]
        vz_idx = self.model.jnt_dofadr[jz]
        agent_vz = float(self.data.qvel[vz_idx])
        # agent world z (body position) and debug vz
        agent_body_id = learnable_agent_body_id
        agent_z = float(self.data.xpos[agent_body_id][2])
        dbg_agent_z = float(agent_z)
        dbg_agent_vz = float(agent_vz)
        # xy 速度と直近コントロールも取得して info に含める
        # Use stored joint indices for the learnable agent (joints live on the
        # "{agent}_anchor" body). Avoid using model.body_jntadr on the child
        # body, which can yield the next body's joint adr.
        jx = self.qpos_indices[self.learnable_agent_key]["x"]
        vadr = self.model.jnt_dofadr[jx]
        qlen = self.data.qvel.shape[0]
        agent_vx = float(self.data.qvel[vadr]) if vadr < qlen else 0.0
        agent_vy = float(self.data.qvel[vadr + 1]) if (vadr + 1) < qlen else 0.0
        last_ctrl = self.last_debug_ctrl.get(self.learnable_agent_key, (0.0, 0.0))
        last_ctrl_f = float(last_ctrl[0])
        last_ctrl_t = float(last_ctrl[1])
        # Apply wall repulsion proportional to agent's current forward effort
        # Wrap the whole repulsion block in a single try/except to avoid
        # syntax/indentation fragility from nested try/excepts.
        try:
            dist, nx, ny = self.vis_engine.sample_sdf_with_normal(learnable_agent_pos[0], learnable_agent_pos[1])
            # use clearance = (surface distance) - (agent effective radius)
            r_clear = float(self.WALL_REPULSION_CLEARANCE)
            r_eff = float(self.AGENT_EFFECTIVE_RADIUS)
            clearance = float(dist) - r_eff

            # determine applied forward (prefer env-applied value)
            try:
                applied_fwd = float(applied_forward_env)
            except Exception:
                applied_fwd = float(last_ctrl_f)

            # Feedback controller to maintain a minimum clearance (target_dist)
            # This acts whenever the agent is closer than `target_dist` to the surface.
            target_dist = 0.5
            kp = 150.0
            f_max = float(self.WALL_REPULSION_FMAX)

            apos_xy = learnable_agent_pos[:2]
            near_arena_edge_x = (self.ARENA_HALF - abs(float(apos_xy[0]))) < 1.0
            near_arena_edge_y = (self.ARENA_HALF - abs(float(apos_xy[1]))) < 1.0

            # suppression: require both elevated z AND evidence of ramp interaction (boost)
            try:
                boost_val = float(self._ramp_boost_gain(self.learnable_agent_key))
            except Exception:
                boost_val = 0.0
            BOOST_THRESH = 0.05
            if float(agent_z) >= float(self.WALL_REPULSION_SUPPRESS_Z) and (boost_val > BOOST_THRESH) and (not near_arena_edge_x) and (not near_arena_edge_y):
                if self.debug_mode:
                    try:
                        self.debug_logger.print(
                            f"[DEBUG][wall_repulsion] suppressed_ramp step={self.current_step} agent={self.learnable_agent_key} z={agent_z:.3f} boost={boost_val:.3f} pos=({apos_xy[0]:.3f},{apos_xy[1]:.3f}) dist={dist:.3f} clearance={clearance:.3f}"
                        )
                    except Exception:
                        print(f"[DEBUG][wall_repulsion] suppressed_ramp agent={self.learnable_agent_key} z={agent_z:.3f} boost={boost_val:.3f} pos=({apos_xy[0]:.3f},{apos_xy[1]:.3f})")
            else:
                # controller contribution: act when clearance < target_dist
                fb_fx = 0.0
                fb_fy = 0.0
                if clearance < float(target_dist):
                    err = float(target_dist) - float(clearance)
                    f_fb = min(max(0.0, kp * err), f_max)
                    fb_fx = f_fb * float(nx)
                    fb_fy = f_fb * float(ny)

                # existing repulsion (estimate-based) still useful when agent is actively pushing
                ctrl_fx = 0.0
                ctrl_fy = 0.0
                try:
                    applied_forward = float(applied_forward_env) if "applied_forward_env" in locals() else float(last_ctrl_f)
                except Exception:
                    applied_forward = float(last_ctrl_f)
                if not hasattr(self, "_prev_wall_rep"):
                    self._prev_wall_rep = {}
                prev_rep = float(self._prev_wall_rep.get(self.learnable_agent_key, 0.0))

                # Parameters to control smoothing and scaling of estimated repulsion
                ctrl_scale = float(getattr(self, "WALL_REPULSION_CTRL_SCALE", 0.5))
                ctrl_lp = float(getattr(self, "WALL_REPULSION_CTRL_LP", 0.2))

                if clearance < r_clear and (abs(applied_forward) > 1e-6):
                    # estimate agent forward force from control magnitude
                    f_agent_est = abs(last_ctrl_f) * float(self.AGENT_ACTUATOR_FWD)
                    # scale factor [0..1] based on how deep into the clearance band we are
                    s = max(0.0, (r_clear - clearance) / max(1e-6, r_clear))
                    f_rep = min(float(self.WALL_REPULSION_ALPHA) * f_agent_est * s, f_max)
                    # apply a global scale to reduce peak magnitude
                    f_rep_scaled = f_rep * ctrl_scale
                    # low-pass (EMA) smoothing to avoid step jumps
                    f_rep_sm = prev_rep * (1.0 - ctrl_lp) + f_rep_scaled * ctrl_lp
                    # optional explicit clip on the smoothed ctrl (runtime override)
                    try:
                        ctrl_clip = getattr(self, "WALL_REPULSION_CTRL_CLIP", None)
                        if ctrl_clip is not None:
                            f_rep_sm = min(float(ctrl_clip), float(f_rep_sm))
                    except Exception:
                        pass
                    # write back smoothed value for next step
                    self._prev_wall_rep[self.learnable_agent_key] = float(f_rep_sm)
                    ctrl_fx = f_rep_sm * float(nx)
                    ctrl_fy = f_rep_sm * float(ny)
                else:
                    # decay previous smoothed rep toward zero when not actively applied
                    f_rep_sm = prev_rep * (1.0 - ctrl_lp)
                    self._prev_wall_rep[self.learnable_agent_key] = float(f_rep_sm)
                    ctrl_fx = f_rep_sm * float(nx)
                    ctrl_fy = f_rep_sm * float(ny)

                # combine feedback and control contributions, clip to f_max
                fx_tot = fb_fx + ctrl_fx
                fy_tot = fb_fy + ctrl_fy
                f_tot = math.hypot(fx_tot, fy_tot)
                # optional per-step accumulation clamp (runtime override)
                try:
                    accum_max = getattr(self, "WALL_REPULSION_ACCUM_CLAMP_MAX", None)
                    if accum_max is not None:
                        if f_tot > float(accum_max) and f_tot > 0.0:
                            scale = float(accum_max) / f_tot
                            fx_tot *= scale
                            fy_tot *= scale
                    else:
                        if f_tot > f_max and f_tot > 0.0:
                            scale = f_max / f_tot
                            fx_tot *= scale
                            fy_tot *= scale
                except Exception:
                    if f_tot > f_max and f_tot > 0.0:
                        scale = f_max / f_tot
                        fx_tot *= scale
                        fy_tot *= scale

                bid = int(learnable_agent_body_id)
                # data.xfrc_applied stores [Fx,Fy,Fz, Mx,My,Mz]
                try:
                    # single write to the external force buffer to avoid partial accumulation
                    self.data.xfrc_applied[bid, 0] = fx_tot
                    self.data.xfrc_applied[bid, 1] = fy_tot
                except Exception:
                    # best-effort: ignore if structure differs
                    pass
                if self.debug_mode:
                    try:
                        self.debug_logger.print(
                            f"[DEBUG][wall_repulsion] step={self.current_step} agent={self.learnable_agent_key} dist={dist:.3f} clearance={clearance:.3f} fb=({fb_fx:.1f},{fb_fy:.1f}) ctrl=({ctrl_fx:.1f},{ctrl_fy:.1f}) fx={fx_tot:.1f} fy={fy_tot:.1f}"
                        )
                    except Exception:
                        print(f"[DEBUG][wall_repulsion] agent={self.learnable_agent_key} dist={dist:.3f} clearance={clearance:.3f} f_tot={f_tot:.1f}")
        except Exception:
            # swallow any error in repulsion logic to avoid breaking simulation step
            pass

        obs = self._normalize_obs(self._get_obs(idx_to_obs))
        reward = float(rb if self.target == "hider" else -rb)
        done = self.current_step >= self.max_episode_steps
        # ランプ関連のデバッグ指標を計算（ランプ上にいると判断できる場合のみ）
        dbg_ramp_progress = None
        dbg_ramp_lx = None
        dbg_ramp_ly = None
        dbg_ramp_facing = None
        dbg_ramp_rpos = None
        try:
            # _ramp_boost_gain が高いとランプ上にいる可能性が高いのでそれをフィルタに使う
            boost = float(self._ramp_boost_gain(self.learnable_agent_key))
            if boost > 0.0:
                # agent の位置・高さ — qpos の z ではなく body の world z を使う
                # (qpos.z はジョイント制約で AGENT_Z_MIN にクリップされるため、
                #  実際のボディ高さを使う方がランプ接触の指標として有用)
                agent_body_id = learnable_agent_body_id
                agent_z = float(self.data.xpos[agent_body_id][2])
                apos2 = learnable_agent_pos[:2]
                # 学習対象の回転角（z）を取得して facing 計算に使う
                arot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[self.learnable_agent_key]["rot"]]]
                # 進捗正規化に使う範囲はランプごとの実ジオメトリから算出
                for _i, rid in enumerate(self.ramp_ids, start=1):
                    rpos = self.data.xpos[rid][:2]
                    up = self._ramp_uphill_dir(rid)
                    side = np.array([-up[1], up[0]], dtype=np.float32)
                    rel = apos2 - rpos
                    lx = float(np.dot(rel, up))
                    ly = float(np.dot(rel, side))
                    # 横ずれが大きければ無視
                    if abs(ly) > 1.5:
                        continue
                    try:
                        lx_min, lx_max, _ = self._compute_ramp_lx_bounds(rid, up)
                    except Exception:
                        lx_min, lx_max = -1.15, 0.666
                    # 正規化された進捗（clamp 0..1）
                    prog = (lx - lx_min) / max(1e-6, (lx_max - lx_min))
                    prog = max(0.0, min(1.0, prog))
                    # ランプの高さはブースト判定には関係しないため、
                    # `dbg_agent_z` にはエージェントの world z をそのまま記録する。
                    # 最も進捗が大きいランプを採用
                    if dbg_ramp_progress is None or prog > dbg_ramp_progress:
                        dbg_ramp_progress = float(prog)
                        dbg_agent_z = float(agent_z)
                        dbg_ramp_lx = float(lx)
                        dbg_ramp_ly = float(ly)
                        # facing はエージェント向きとランプ上り方向のコサイン
                        afwd = np.array([math.cos(arot), math.sin(arot)], dtype=np.float32)
                        dbg_ramp_facing = float(np.dot(afwd, up))
                        dbg_ramp_rpos = rpos.copy()
                        dbg_ramp_rid = rid
                # end per-ramp loop
                # If boost suggests a ramp but we didn't select any, emit per-ramp diagnostics
                # to help tune filters (temporary verbose diagnostic).
                if dbg_ramp_progress is None:
                    try:
                        details = []
                        afwd = np.array([math.cos(arot), math.sin(arot)], dtype=np.float32)
                        for i2, rid2 in enumerate(self.ramp_ids, start=1):
                            rkey2 = f"ramp{i2}"
                            st2 = self.object_state.get(rkey2, {})
                            rpos2 = self.data.xpos[rid2][:2]
                            up2 = self._ramp_uphill_dir(rid2)
                            side2 = np.array([-up2[1], up2[0]], dtype=np.float32)
                            rel2 = apos2 - rpos2
                            lx2 = float(np.dot(rel2, up2))
                            ly2 = float(np.dot(rel2, side2))
                            facing2 = float(np.dot(afwd, up2))
                            details.append(f"{rkey2}(mode={st2.get('mode')},owner={st2.get('owner')}," f"lx={lx2:.3f},ly={ly2:.3f},f={facing2:.3f},rpos=({rpos2[0]:.3f},{rpos2[1]:.3f}))")
                        s = ", ".join(details)
                        if getattr(self, "debug_mode", False):
                            print(f"[RAMP_DBG_REJECT] step={getattr(self,'current_step',-1)} ak={self.learnable_agent_key} " f"boost={boost:.3f} candidates=[{s}]")
                    except Exception:
                        pass
            # detect climbing (agent z increasing while on/near ramp) and reaching top
            prev_z = self._prev_agent_z.get(self.learnable_agent_key, None)
            dbg_ramp_climbing = False
            dbg_ramp_reached_top = False
            if dbg_ramp_progress is not None:
                if prev_z is not None and (agent_z - prev_z) > 0.01 and dbg_ramp_progress > 0.05:
                    dbg_ramp_climbing = True
                # reached-top is only debug info; use height-only criterion
                try:
                    HEIGHT_TOL = 0.02
                    if dbg_ramp_rid is not None:
                        top_z = self._compute_ramp_top_z(dbg_ramp_rid)
                        dbg_ramp_reached_top = agent_z >= (top_z - HEIGHT_TOL)
                    else:
                        dbg_ramp_reached_top = False
                except Exception:
                    dbg_ramp_reached_top = False
            # store for next step
            try:
                self._prev_agent_z[self.learnable_agent_key] = float(agent_z)
            except Exception:
                pass
        except Exception:
            # ランプ指標の計算で失敗しても主処理に影響を出さない。
            # ただしエージェントの位置/速度は常に報告したいため
            # `dbg_agent_z` と `dbg_agent_vz` は上書きしない。
            dbg_ramp_progress = None
            dbg_ramp_climbing = False
            dbg_ramp_reached_top = False
        # Ensure agent z is reported when we have a computed progress but z wasn't set
        try:
            if dbg_ramp_progress is not None and dbg_agent_z is None:
                agent_body_id = learnable_agent_body_id
                dbg_agent_z = float(self.data.xpos[agent_body_id][2])
        except Exception:
            dbg_agent_z = None
        info = {
            "is_detected": find,
            "lock_event": any_lock_event,
            "grab_event": any_grab_event,
            "dbg_lock_pressed": any_lock_pressed,
            "dbg_grab_pressed": any_grab_pressed,
            "dbg_lock_target": any_lock_target,
            "dbg_grab_target": any_grab_target,
            "dbg_lock_btn_max": max_lock_btn,
            "dbg_grab_btn_max": max_grab_btn,
            "dbg_box_moving_count": moving_box_count,
            "dbg_ramp_moving_count": moving_ramp_count,
            "dbg_max_box_speed": float(max(box_speeds) if box_speeds else 0.0),
            "dbg_max_ramp_speed": float(max(ramp_speeds) if ramp_speeds else 0.0),
            "dbg_blocked_ramp_count": blocked_ramp_count,
            "dbg_boosted_agents": boosted_agents,
            "dbg_override_learnable_policy": bool(self.override_learnable_policy),
            "dbg_model_policy_deterministic": bool(self.model_policy_deterministic),
            "dbg_seek_gaze_cos_front_max": float(gaze_cos_front_max),
            "dbg_seek_gaze_cos_front_dist_max": float(gaze_dist_max),
            "dbg_learnable_hider_seen": bool(learnable_hider_seen),
            "wall_distance": wall_dist,
            "agent_vz": agent_vz,
            "agent_vx": agent_vx,
            "agent_vy": agent_vy,
            "dbg_last_ctrl_f": last_ctrl_f,
            "dbg_last_ctrl_t": last_ctrl_t,
            # whether current step is within prep/warmup
            "in_prep": bool(self.current_step <= self.prep_steps),
            # 'applied_forward_model' is the raw model output; 'applied_forward' is what was
            # actually applied to the actuator (may be transformed at runtime for compatibility)
            "applied_forward_model": applied_forward_model,
            "applied_forward": applied_forward_env,
        }
        # デバッグ用: ランプ上判定に使った boost 値を常に出力する
        try:
            info["dbg_ramp_boost"] = float(self._ramp_boost_gain(self.learnable_agent_key))
        except Exception:
            pass
        # 追加デバッグ出力
        if dbg_ramp_lx is not None:
            info["dbg_ramp_lx"] = dbg_ramp_lx
        if dbg_ramp_ly is not None:
            info["dbg_ramp_ly"] = dbg_ramp_ly
        if dbg_ramp_facing is not None:
            info["dbg_ramp_facing"] = dbg_ramp_facing
        if dbg_ramp_rpos is not None:
            info["dbg_ramp_rpos_x"] = float(dbg_ramp_rpos[0])
            info["dbg_ramp_rpos_y"] = float(dbg_ramp_rpos[1])
        info["dbg_agent_z"] = dbg_agent_z
        info["dbg_agent_vz"] = dbg_agent_vz
        if dbg_ramp_progress is not None:
            info["dbg_ramp_progress"] = dbg_ramp_progress
        # climbing / reached-top diagnostics
        info["dbg_ramp_climbing"] = bool(dbg_ramp_climbing)
        info["dbg_ramp_reached_top"] = bool(dbg_ramp_reached_top)
        self._dbg_collect_stats(reward, info)

        # cache the observation returned for the learnable agent
        try:
            self._cached_obs = obs
        except Exception:
            pass

        # decrement being_hit counters
        try:
            if self._being_hit_counters:
                for k in list(self._being_hit_counters.keys()):
                    self._being_hit_counters[k] = max(0, int(self._being_hit_counters[k]) - 1)
                    if self._being_hit_counters[k] == 0:
                        del self._being_hit_counters[k]
        except Exception:
            pass

        return obs, reward, False, done, info

    def _compute_team_reward(self):
        """
        基本報酬の再定義:
        - Seeker: 視野内かつ正面・近距離で捉えるほど高報酬。視野外は 0。
        - Hider: 生存ボーナス + 被弾ペナルティ（Seeker報酬の裏返し）。
        """
        if self.current_step <= self.prep_steps:
            return 0.0, False, 0.0, 0.0, False

        total_hider_reward = 0.0
        any_hider_seen = False
        gaze_cos_front_max = 0.0
        gaze_cos_front_dist_max = 0.0
        learnable_hider_seen_flag = False

        for hk in self.hider_keys:
            hid = self.body_ids[hk]
            hpos = self.data.xpos[hid][:2]
            h_reward = 0.00  # 基本生存ボーナス ステップ数誇張なので0.0

            for sk in self.seeker_keys:
                sid = self.body_ids[sk]
                spos = self.data.xpos[sid][:2]
                srot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[sk]["rot"]]]

                dx, dy = float(hpos[0] - spos[0]), float(hpos[1] - spos[1])
                dist = math.sqrt(dx * dx + dy * dy)
                dist_m = dist + 1e-8

                # 正面度 (gaze_cos)
                cos_align = (math.cos(srot) * (dx / dist_m)) + (math.sin(srot) * (dy / dist_m))
                frontness = max(float(cos_align), 0.0)

                # シーカーの視界判定
                if self._is_vis(spos, srot, hpos, sid, hid):
                    any_hider_seen = True
                    # 捕捉の質に基づく報酬 (Seeker:+, Hider:-)
                    capture_reward = frontness / (dist + 0.5)
                    h_reward -= capture_reward

                    if sk == self.learnable_agent_key or hk == self.learnable_agent_key:
                        learnable_hider_seen_flag = True

                    gaze_cos_front_max = max(gaze_cos_front_max, frontness)
                    gaze_cos_front_dist_max = max(gaze_cos_front_dist_max, capture_reward)

            total_hider_reward += h_reward

        return (
            total_hider_reward / len(self.hider_keys),
            any_hider_seen,
            gaze_cos_front_max,
            gaze_cos_front_dist_max,
            learnable_hider_seen_flag,
        )

    def _is_vis(self, pos, rot, t_pos, my_id, t_id):
        rel = t_pos - pos
        dist = math.sqrt(np.sum(rel**2)) + 1e-8
        if dist > L_SCALE or (math.cos(rot) * (rel[0] / dist) + math.sin(rot) * (rel[1] / dist)) < 0.38:
            return False
        return self.vis_engine.is_visible(pos, t_pos, body_exclude=my_id, target_body_id=t_id)

    def _normalize_obs(self, o):
        v = o.copy()
        idx = self.idx
        v[idx.SELF.VEL_X] /= V_SCALE
        v[idx.SELF.VEL_Y] /= V_SCALE
        v[idx.SELF.ROT] /= R_SCALE
        v[idx.LIDAR] /= L_SCALE
        for b in idx.B:
            v[b.REL_X] /= P_SCALE
            v[b.REL_Y] /= P_SCALE
            v[b.VEL_X] /= V_SCALE
            v[b.VEL_Y] /= V_SCALE
        for r in idx.RAMP:
            v[r.REL_X] /= P_SCALE
            v[r.REL_Y] /= P_SCALE
            v[r.VEL_X] /= V_SCALE
            v[r.VEL_Y] /= V_SCALE
        for en in idx.OTHERS:
            v[en.REL_X] /= P_SCALE
            v[en.REL_Y] /= P_SCALE
            v[en.VEL_X] /= V_SCALE
            v[en.VEL_Y] /= V_SCALE
        return v

    def _get_obs(self, idx):
        """
        観測情報の生成:
        名前ベースの絶対インデックス参照により、情報の取り違えとIndexErrorを完全に排除。
        """
        o = np.zeros(self.idx.total_dim, dtype=np.float32)
        ak, m, d = self.agent_keys[idx], self.model, self.data
        ps, rv = (
            d.xpos[self.body_ids[ak]],
            float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]["rot"]]]),
        )

        # 自己情報の速度 (名前から正確な位置を特定)
        vax_self = m.jnt_dofadr[self.qpos_indices[ak]["x"]]
        vay_self = m.jnt_dofadr[self.qpos_indices[ak]["y"]]
        cos_r, sin_r = math.cos(-rv), math.sin(-rv)

        si = self.idx.SELF
        o[si.VEL_X] = d.qvel[vax_self] * cos_r - d.qvel[vay_self] * sin_r
        o[si.VEL_Y] = d.qvel[vax_self] * sin_r + d.qvel[vay_self] * cos_r
        o[si.ROT] = rv
        o[si.COS_ROT], o[si.SIN_ROT] = math.cos(rv), math.sin(rv)

        # LiDAR
        gb = self._current_grabbed_by(ak)
        ignore_id = self.obj_body_map[gb] if gb else -1
        o[self.idx.LIDAR] = self.vis_engine.cast_lidar(ps[:2], rv, 1, self.body_ids[ak], ignore_id)

        qvel_limit = d.qvel.shape[0]

        # オブジェクト情報 (Box, Ramp) - 境界チェックを追加
        for i, tid in enumerate(self.box_ids):
            b_idx = self.idx.B[i]
            d_w = d.xpos[tid][:2] - ps[:2]
            o[b_idx.REL_X] = d_w[0] * cos_r - d_w[1] * sin_r
            o[b_idx.REL_Y] = d_w[0] * sin_r + d_w[1] * cos_r
            b_vadr = m.jnt_dofadr[m.body_jntadr[tid]]
            if b_vadr >= 0 and b_vadr + 1 < qvel_limit:
                o[b_idx.IS_MOVING] = 1.0 if math.sqrt(d.qvel[b_vadr] ** 2 + d.qvel[b_vadr + 1] ** 2) > 0.05 else 0.0
            o[b_idx.IS_LOCKED] = 1.0 if self.object_state[f"b{i + 1}"]["mode"] == "locked" else 0.0
            q = d.xquat[tid]
            yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
            o[b_idx.QUAT_0], o[b_idx.QUAT_1] = math.cos(yaw - rv), math.sin(yaw - rv)

        for i, rid in enumerate(self.ramp_ids):
            r_idx = self.idx.RAMP[i]
            d_w_r = d.xpos[rid][:2] - ps[:2]
            o[r_idx.REL_X] = d_w_r[0] * cos_r - d_w_r[1] * sin_r
            o[r_idx.REL_Y] = d_w_r[0] * sin_r + d_w_r[1] * cos_r
            r_vadr = m.jnt_dofadr[m.body_jntadr[rid]]
            if r_vadr >= 0 and r_vadr + 1 < qvel_limit:
                o[r_idx.IS_MOVING] = 1.0 if math.sqrt(d.qvel[r_vadr] ** 2 + d.qvel[r_vadr + 1] ** 2) > 0.05 else 0.0
            o[r_idx.IS_LOCKED] = 1.0 if self.object_state[f"ramp{i + 1}"]["mode"] == "locked" else 0.0
            q = d.xquat[rid]
            yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] ** 2 + q[3] ** 2))
            o[r_idx.QUAT_0], o[r_idx.QUAT_1] = math.cos(yaw - rv), math.sin(yaw - rv)

        # 他エージェント情報 (痛覚メタファー & 名前ベース参照)
        ens = [k for k in self.agent_keys if k != ak]
        ens.sort(
            key=lambda k: (
                0 if k.startswith("s" if ak.startswith("h") else "h") else 1,
                k,
            )
        )

        for i, enm in enumerate(ens[: len(self.idx.OTHERS)]):
            en_idx = self.idx.OTHERS[i]
            eid = self.body_ids[enm]
            e_pos = d.xpos[eid][:2]

            # 1. 視界判定 (自分が相手を見ているか)
            prev_vis = self._prev_vis.get((ak, enm), None)
            if prev_vis is not None:
                visible = bool(prev_vis)
            else:
                visible = self._is_vis(ps[:2], rv, e_pos, self.body_ids[ak], eid)

            # 2. 被弾判定 (自分がHider、相手がSeeker、かつ相手が自分を見ているか)
            being_hit_flag = 0.0
            if ak.startswith("h") and enm.startswith("s"):
                # Use persistence counter if present (preferred)
                cnt = int(self._being_hit_counters.get((ak, enm), 0))
                if cnt > 0:
                    being_hit_flag = 1.0
                else:
                    prev_bh = self._prev_being_hit.get((ak, enm), None)
                    if prev_bh is not None and float(prev_bh) > 0.0:
                        being_hit_flag = 1.0
                    else:
                        s_rot = float(d.qpos[m.jnt_qposadr[self.qpos_indices[enm]["rot"]]])
                        if self._is_vis(e_pos, s_rot, ps[:2], eid, self.body_ids[ak]):
                            being_hit_flag = 1.0

            if visible:
                # 【視覚情報】
                o[en_idx.VISIBLE] = 1.0
                d_w = e_pos - ps[:2]
                o[en_idx.REL_X] = d_w[0] * cos_r - d_w[1] * sin_r
                o[en_idx.REL_Y] = d_w[0] * sin_r + d_w[1] * cos_r

                vax_en = m.jnt_dofadr[self.qpos_indices[enm]["x"]]
                vay_en = m.jnt_dofadr[self.qpos_indices[enm]["y"]]
                o[en_idx.VEL_X] = d.qvel[vax_en] * cos_r - d.qvel[vay_en] * sin_r
                o[en_idx.VEL_Y] = d.qvel[vax_en] * sin_r + d.qvel[vay_en] * cos_r

                e_rot = float(d.qpos[m.jnt_qposadr[self.qpos_indices[enm]["rot"]]])
                o[en_idx.QUAT_0], o[en_idx.QUAT_1] = math.cos(e_rot - rv), math.sin(e_rot - rv)
                # エージェント間の IS_MOVING は廃止し、0.0 固定（または別の用途）へ
                o[en_idx.BEING_HIT] = being_hit_flag

            elif being_hit_flag > 0.5:
                # 【痛覚情報】視界外だが撃たれている
                o[en_idx.VISIBLE] = 0.0
                o[en_idx.BEING_HIT] = 1.0  # 明示的に「被弾中」を示す

                d_w = e_pos - ps[:2]
                rel_x = d_w[0] * cos_r - d_w[1] * sin_r
                rel_y = d_w[0] * sin_r + d_w[1] * cos_r
                o[en_idx.REL_X], o[en_idx.REL_Y] = rel_x, rel_y

                # 向きは位置から逆算（自分を狙っていると仮定）
                dist = math.sqrt(rel_x**2 + rel_y**2) + 1e-8
                o[en_idx.QUAT_0], o[en_idx.QUAT_1] = -rel_x / dist, -rel_y / dist
                o[en_idx.VEL_X], o[en_idx.VEL_Y] = 0.0, 0.0
            else:
                # 非検知
                o[en_idx.REL_X], o[en_idx.REL_Y] = L_SCALE, L_SCALE
                o[en_idx.VISIBLE] = 0.0
                o[en_idx.BEING_HIT] = 0.0
            # 壁情報：距離と法線（エージェント局所座標系）を追加
            try:
                dist_w, nx_w, ny_w = self.vis_engine.sample_sdf_with_normal(ps[0], ps[1])
                r = float(self.WALL_REPULSION_RADIUS)
                # 正規化距離 (0..1) を供給。r 超は 1.0
                dist_norm = float(min(dist_w, r) / max(r, 1e-8))
                # world -> local: rotate by -rv (cos_r, sin_r are cos(-rv), sin(-rv))
                n_loc_x = nx_w * cos_r - ny_w * sin_r
                n_loc_y = nx_w * sin_r + ny_w * cos_r
            except Exception:
                dist_norm = 1.0
                n_loc_x = 0.0
                n_loc_y = 0.0

            # write into obs vector at indices defined in ObsIdx
            try:
                o[self.idx.WALL_DIST] = dist_norm
                o[self.idx.WALL_NORM_X] = n_loc_x
                o[self.idx.WALL_NORM_Y] = n_loc_y
            except Exception:
                pass
            return o

    def render(self):  # noqa: C901
        if getattr(self, "_render_disabled", False):
            return

        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except RuntimeError as e:
                # On macOS `launch_passive` requires running under mjpython;
                # avoid raising repeatedly and disable rendering for this process.
                logger.warning("mujoco.viewer.launch_passive failed: %s; disabling rendering for this process. Run under mjpython on macOS to enable passive viewer.", e)
                self._render_disabled = True
                return
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0, 0, 0.8]
                self.viewer.cam.distance = 18.0
                self.viewer.cam.elevation = -35.0
                self.viewer.cam.azimuth = 90.0
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            for ak in self.agent_keys:
                sid = self.body_ids[ak]
                pos = self.data.xpos[sid]
                rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]]
                t_val = self.last_debug_ctrl[ak][1]
                # デバッグ可視化はdebug_logger.enabledで制御
                if self.debug_logger.enabled and self.show_turn_lines and abs(t_val) > 0.005:
                    is_seeker = ak.startswith("s")
                    h = 1.1 if is_seeker else 1.3
                    color = [0, 1, 1, 1.0] if is_seeker else [0, 0.2, 1, 1.0]
                    p_start = pos.copy()
                    p_start[2] = h
                    p_end = p_start.copy()
                    p_end[0] += -math.sin(rot) * t_val * 2.0
                    p_end[1] += math.cos(rot) * t_val * 2.0
                    if self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                        g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                        mujoco.mjv_connector(
                            g,
                            mujoco.mjtGeom.mjGEOM_LINE,
                            8.0 + abs(t_val) * 25,
                            p_start,
                            p_end,
                        )
                        g.rgba[:] = color
                        self.viewer.user_scn.ngeom += 1
                for k in self.agent_keys:
                    if k == ak:
                        continue
                    tid = self.body_ids[k]
                    if not self._is_vis(pos[:2], rot, self.data.xpos[tid][:2], sid, tid):
                        continue

                    # Determine color:
                    # - If current agent `ak` is a Seeker and target `k` is a Hider,
                    #   show a yellow line. Brightness differs so H1 and H2 are distinguishable.
                    # - Otherwise keep previous semantics (Seeker->others red, Hider->others blue).
                    if ak.startswith("s") and k.startswith("h"):
                        # brightness by index in hider_keys (H1 brighter than H2)
                        try:
                            h_idx = self.hider_keys.index(k)
                            brightness = 1.0 - 0.4 * h_idx
                            if brightness < 0.2:
                                brightness = 0.2
                        except Exception:
                            brightness = 0.8
                        color = [brightness, brightness, 0.0, 1.0]
                    else:
                        if ak.startswith("s"):
                            color = [1, 0, 0, 1]
                        else:
                            color = [0, 0, 1, 1]

                    if self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                        g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, pos, self.data.xpos[tid])
                        g.rgba[:] = color
                        self.viewer.user_scn.ngeom += 1
            # runtime recolor: if any seeker sees a hider, recolor that hider's geoms to yellow
            try:
                # determine overrides per hider key
                overrides = {}
                for s in [k for k in self.agent_keys if k.startswith("s")]:
                    s_sid = self.body_ids[s]
                    s_pos = self.data.xpos[s_sid][:2]
                    s_rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[s]["rot"]]]
                    for h in [k for k in self.agent_keys if k.startswith("h")]:
                        h_tid = self.body_ids[h]
                        if not self._is_vis(s_pos, s_rot, self.data.xpos[h_tid][:2], s_sid, h_tid):
                            continue
                        # compute brightness by hider index
                        try:
                            h_idx = self.hider_keys.index(h)
                            brightness = 1.0 - 0.4 * h_idx
                            if brightness < 0.2:
                                brightness = 0.2
                        except Exception:
                            brightness = 0.8
                        overrides[h] = [brightness, brightness, 0.0, 1.0]

                # apply overrides or restore defaults
                for ak in self.agent_keys:
                    geom_ids = self.agent_geom_ids.get(ak, [])
                    if ak in overrides:
                        col = overrides[ak]
                        for gid in geom_ids:
                            try:
                                self.model.geom_rgba[gid][:] = col
                            except Exception:
                                pass
                    else:
                        # restore defaults
                        cols = self.agent_default_rgba.get(ak, [])
                        for gid, defcol in zip(geom_ids, cols, strict=True):
                            try:
                                self.model.geom_rgba[gid][:] = defcol
                            except Exception:
                                pass
            except Exception:
                # be resilient to any viewer/model issues
                pass

        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()


# Provide a fixed actuator XML generator and attach it to the class so
# _build_dynamic_xml can safely call it even if the original method
# contained malformed content during edits.
def _xml_actuators_fixed(self, pre):
    return f"""
    <general name="{pre}_fwd" site="{pre}_thrust" gear="1 0 0 0 0 0" gainprm="{self.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{pre}_turn" joint="{pre}_rot" gear="0.5" gainprm="{self.AGENT_ACTUATOR_TURN_GAIN}" ctrlrange="-1 1"/>
    """


# Attach to class
TeamCosEnv._xml_actuators_fixed = _xml_actuators_fixed
