# src/envs/hns_environment.py
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


# --- DebugLoggerクラス ---
class DebugLogger:
    def __init__(self, enabled=False, log_interval_steps=100):
        self.enabled = enabled
        self.log_interval_steps = log_interval_steps
        self._log_last_step = {}
        # use short name consistently
        self._policy_src_logged = set()

    def print(self, message):
        if self.enabled:
            print(message)

    def print_throttled(self, key, message, current_step, force=False):
        # debug統計以外の出力は抑制
        return

    def log_policy_src(self, agent_key, source, current_step, force=False):
        # short-name API — suppressed in this debug logger
        return

    def clear_policy_src_log(self):
        self._policy_src_logged.clear()


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
        self._debug_reward_buffer = []
        self._debug_hide_buffer = []
        self._debug_wall_distance_buffer = []
        self._debug_step_counter = 0
        self._debug_log_interval = getattr(self, "debug_logger", None) and getattr(self.debug_logger, "log_interval_steps", 100) or 100

    def _debug_collect_stats(self, reward, info):
        if not self.debug_mode:
            return
        self._debug_reward_buffer.append(reward)
        # 隠れ率: is_detected==False なら隠れているとみなす
        self._debug_hide_buffer.append(0 if info.get("is_detected", False) else 1)
        wd = info.get("wall_distance", None)
        if wd is not None:
            if isinstance(wd, (list, tuple, np.ndarray)):
                self._debug_wall_distance_buffer.extend([float(v) for v in wd])
            else:
                self._debug_wall_distance_buffer.append(float(wd))
        self._debug_step_counter += 1
        # log_intervalごとに統計出力
        if self._debug_step_counter % self._debug_log_interval == 0:
            self._debug_print_stats()

    def _debug_print_stats(self):
        if not self.debug_mode:
            return
        avg_reward = float(np.mean(self._debug_reward_buffer)) if self._debug_reward_buffer else 0.0
        hide_rate = float(np.mean(self._debug_hide_buffer)) if self._debug_hide_buffer else 0.0
        wd_arr = np.array(self._debug_wall_distance_buffer, dtype=np.float32) if self._debug_wall_distance_buffer else np.array([0.0])
        self.debug_logger.print(f"[DEBUG] Step={self.current_step} AvgR={avg_reward:.3f} HideRate={hide_rate:.2f} WallDist(mean/min/max)={wd_arr.mean():.3f}/{wd_arr.min():.3f}/{wd_arr.max():.3f}")
        self._debug_reward_buffer.clear()
        self._debug_hide_buffer.clear()
        self._debug_wall_distance_buffer.clear()

    """
    Hide and Seek 高度物理環境 (単一エージェント学習最適化版)
    """
    ARENA_HALF = 6.0
    SAFE_HALF = 5.0
    R_AGENT = 0.55
    R_BOX = 0.95
    R_RAMP = 1.30

    AGENT_DAMPING_XY = 30.0
    AGENT_DAMPING_Z = 16.0
    AGENT_DAMPING_ROT = 25.0
    AGENT_ACTUATOR_FWD = 700
    AGENT_Z_MIN = 0.35
    AGENT_Z_MAX = 1.20
    AGENT_MAX_VZ = 2.2
    RAMP_JOINT_DAMPING = 90.0
    BOX_JOINT_DAMPING = 28.0
    RAMP_MASS = 60.0
    RAMP_INNER_WEIGHT_MASS = 30.0
    BOX_MASS = 22.0
    INTERACT_RANGE = 1.95
    BTN_ON = 0.1
    BTN_COOLDOWN = 8
    GRAB_OFFSET = 1.45
    GRAB_FOLLOW_GAIN = 8.0
    GRAB_MAX_SPEED = 2.6
    GRAB_BREAK_DIST = 2.8
    RAMP_BOOST_FWD = 0.35
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
        action_repeat=16,
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
        self.debug_mode = debug_mode
        self.debug_logger = DebugLogger(enabled=debug_mode, log_interval_steps=dbg_log_interval_steps)
        self.action_repeat = action_repeat
        # temporal caches for visibility / being-hit freshness
        self._prev_vis = {}
        self._prev_being_hit = {}
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
        boxes = "".join(self._xml_box(i, [0, 0], 0) for i in range(1, self.n_boxes + 1))
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
        acts = "".join(self._xml_actuators(ak) for ak in self.agent_keys)

        return f"""
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
    <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>
    <mesh name="ramp_mesh"
          vertex="-0.6666 -0.5 0.0 0.6666 -0.5 0.0 0.6666 -0.5 1.0 -0.6666 0.5 0.0 0.6666 0.5 0.0 0.6666 0.5 1.0"
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
    <geom name="floor" type="plane" size="{s} {s} 0.1" material="grid" friction="1.1 0.15 0.003"/>
    <geom name="wall_n" type="box" size="{s + 0.15} 0.1 2.0" pos="0 6.1 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_s" type="box" size="{s + 0.15} 0.1 2.0" pos="0 -6.1 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_e" type="box" size="0.1 {s} 2.0" pos="6.1 0 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_w" type="box" size="0.1 {s} 2.0" pos="-6.1 0 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="maze_w0" type="box" size="1.5 0.2 0.5" pos="3.0 1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w1" type="box" size="1.5 0.2 0.5" pos="-3.0 -1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w2" type="box" size="0.2 1.5 0.5" pos="0.0 -3.0 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w3" type="box" size="0.2 1.5 0.5" pos="0.0 3.0 0.5" rgba="0 0.7 0.7 1" {attr}/>
"""

    def _xml_ramp(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        return f"""
        <body name="ramp{i}_body" pos="{xy[0]} {xy[1]} 0" quat="{q}">
            <inertial pos="0.3 0 0.25" mass="{self.RAMP_MASS}" diaginertia="10 10 20"/>
            <joint type="free" name="ramp{i}_joint" damping="{self.RAMP_JOINT_DAMPING}"/>
            <geom name="ramp{i}_geom" type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0" rgba="0 1 0 1"/>
            <geom name="ramp{i}_slope_surface" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.3" friction="1.35 0.22 0.01"/>
            <geom name="ramp{i}_back_panel" type="box" size="0.02 0.5 0.5" pos="0.6666 0 0.5" rgba="0 1 0 0.3" friction="1.35 0.22 0.01"/>
            <geom name="ramp{i}_inner_weight" type="box" size="0.3333 0.5 0.25" pos="0.3333 0 0.25" rgba="0 1 0 0.3" mass="{self.RAMP_INNER_WEIGHT_MASS}" solimp="0.95 0.99 0.001" friction="1.35 0.22 0.01"/>
    </body>"""

    def _xml_box(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        return f"""
    <body name="box{i}_body" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
            <joint name="box{i}_joint" type="free" damping="{self.BOX_JOINT_DAMPING}"/>
            <geom name="box{i}_geom" type="box" size="0.6 0.6 0.5" mass="{self.BOX_MASS}" rgba="0.75 0.55 0.3 1" friction="1.2 0.08 0.003"/>
    </body>"""

    def _xml_agent(self, pre, xy, rot, color):
        q = _euler_z_to_quat(rot)
        r, g, b = color
        return f"""
    <body name="{pre}_anchor" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
      <joint name="{pre}_x" type="slide" axis="1 0 0" damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{pre}_y" type="slide" axis="0 1 0" damping="{self.AGENT_DAMPING_XY}"/>
    <joint name="{pre}_z" type="slide" axis="0 0 1" damping="{self.AGENT_DAMPING_Z}" limited="true" range="{self.AGENT_Z_MIN} {self.AGENT_Z_MAX}"/>
      <joint name="{pre}_rot" type="hinge" axis="0 0 1" damping="{self.AGENT_DAMPING_ROT}"/>
      <body name="{pre}_body">
        <site name="{pre}_thrust" pos="0 0 0"/>
        <geom name="{pre}_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="12" friction="1.2 0.12 0.003"/>
        <geom name="{pre}_capsule" type="capsule" size="0.3 0.3" rgba="{r} {g} {b} 1" mass="4" contype="0" conaffinity="0"/>
        <geom name="{pre}_nose" type="capsule" fromto="0 0 0.3 0.3 0 0.3" size="0.09" rgba="1 1 1 1" contype="0" conaffinity="0"/>
        <geom name="{pre}_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="{r} {g} {b} 1" contype="0" conaffinity="0"/>
      </body>
    </body>"""

    def _xml_actuators(self, pre):
        return f"""
    <general name="{pre}_fwd" site="{pre}_thrust" gear="1 0 0 0 0 0" gainprm="{self.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{pre}_turn" joint="{pre}_rot" gear="0.5" gainprm="120" ctrlrange="-1 1"/>\n"""

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

            # Freejointオブジェクト (box/b/ramp) の判定
            # シンプルに 's' (seeker) や 'h' (hider) で始まらないものを対象とするのも手です
            if tk[0] in ("b", "r"):  # 'b'(box/b1), 'r'(ramp) に合致
                pq = None

                # 1. xquat 優先 (もっとも直接的で安全)
                if bid < xquat_len:
                    xq = self.data.xquat[bid]
                    # 基本的に MuJoCo の xquat は常に size 4 です
                    pq = xq.copy()

                # 2. qpos スライス (fallback)
                if pq is None and qadr >= 0 and (qadr + 7) <= qpos_len:
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
        for i, rid in enumerate(self.ramp_ids, start=1):
            rkey = f"ramp{i}"
            if not self._is_ramp_blocked_or_locked(rkey, rid):
                continue
            rpos = self.data.xpos[rid][:2]
            up = self._ramp_uphill_dir(rid)
            side = np.array([-up[1], up[0]], dtype=np.float32)
            rel = apos - rpos
            lx = float(np.dot(rel, up))
            ly = float(np.dot(rel, side))
            facing = float(np.dot(afwd, up))
            if abs(ly) > 0.95 or facing < 0.55:
                continue
            if -1.15 <= lx <= 0.2:
                gain = max(gain, 1.0)
            elif 0.2 < lx <= 0.95:
                gain = max(gain, 0.6)
        return gain

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
        POSE_SIZE = 7  # qposのpose成分長
        VEL_SIZE = 6  # qvelの成分長
        XY_START, XY_STOP = 0, 2  # x, y成分
        Z_IDX = 2
        QUAT_START, QUAT_STOP = 3, 7
        XY_VEL_START, XY_VEL_STOP = 0, 2
        Z_VEL_IDX = 2
        ANG_VEL_START, ANG_VEL_STOP = 3, 6

        for ak in self.agent_keys:
            if self.btn_cooldown[ak] > 0:
                self.btn_cooldown[ak] -= 1
        for tk, st in self.object_state.items():
            qadr, vadr = self._obj_addr(tk)
            if st["mode"] == "locked" and st["locked_pose"] is not None:
                self.data.qpos[qadr : qadr + POSE_SIZE] = st["locked_pose"]
                self.data.qvel[vadr : vadr + VEL_SIZE] = 0.0
            elif st["mode"] == "grabbed" and st["owner"] is not None:
                owner = st["owner"]
                opos = self.data.xpos[self.body_ids[owner]][XY_START:XY_STOP]
                cur_xy = self.data.qpos[qadr : qadr + POSE_SIZE][XY_START:XY_STOP].copy()
                # --- 距離・壁判定によるgrab解除 ---
                grab_dist = float(np.linalg.norm(cur_xy - opos))
                if grab_dist > self.GRAB_BREAK_DIST or self._interaction_blocked_by_static_walls(opos, cur_xy):
                    st["mode"] = "free"
                    st["owner"] = None
                    self.data.qvel[vadr : vadr + VEL_SIZE][XY_VEL_START:XY_VEL_STOP] *= 0.5
                    continue
                err_xy = opos - cur_xy
                self.data.qvel[vadr : vadr + VEL_SIZE][XY_VEL_START:XY_VEL_STOP] = err_xy  # 速度を直接目標方向に
                self.data.qvel[vadr : vadr + VEL_SIZE][Z_VEL_IDX] = 0.0
                self.data.qvel[vadr : vadr + VEL_SIZE][ANG_VEL_START:ANG_VEL_STOP] = 0.0
            else:
                self.data.qvel[vadr : vadr + VEL_SIZE][XY_VEL_START:XY_VEL_STOP] *= 0.9
                speed_xy = float(np.linalg.norm(self.data.qvel[vadr : vadr + VEL_SIZE][XY_VEL_START:XY_VEL_STOP]))
                if speed_xy < 1e-3:
                    self.data.qvel[vadr : vadr + VEL_SIZE][XY_VEL_START:XY_VEL_STOP] = 0.0
            # Planar lock
            if self.OBJECT_PLANAR_LOCK and st.get("planar_z") is not None:
                self.data.qpos[qadr + Z_IDX] = st["planar_z"]
                pq = st.get("planar_quat")
                try:
                    if pq is not None and np.all(np.isfinite(pq)) and len(pq) >= (QUAT_STOP - QUAT_START):
                        self.data.qpos[qadr + QUAT_START : qadr + QUAT_STOP] = pq
                    else:
                        if self.debug_mode:
                            self.debug_logger.print(f"[DEBUG][_apply_object_constraints] skipping invalid planar_quat for {tk}: {pq}")
                except Exception:
                    if self.debug_mode:
                        self.debug_logger.print(f"[DEBUG][_apply_object_constraints] exception validating planar_quat for {tk}")
                self.data.qvel[vadr + Z_VEL_IDX] = 0.0
                self.data.qvel[vadr + ANG_VEL_START : vadr + ANG_VEL_STOP] = 0.0
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
                    arr = model.get_deterministic_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
                else:
                    arr = model.get_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
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
                    arr = self.shared_policy_model.get_deterministic_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
                else:
                    arr = self.shared_policy_model.get_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
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

    def _debug_print_throttled(self, key, message, force=False):
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._policy_histories.clear()
        self.debug_logger._log_last_step.clear()
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
            cv[self.actuator_ids[f"{ak}_fwd"]], cv[self.actuator_ids[f"{ak}_turn"]] = (
                f_env,
                t,
            )
            self.last_debug_ctrl[ak] = (f_env, t)

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
        obs = self._normalize_obs(self._get_obs(idx_to_obs))
        reward = float(rb if self.target == "hider" else -rb)
        done = self.current_step >= self.max_episode_steps
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
        self._debug_collect_stats(reward, info)

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
            h_reward = 0.05  # 基本生存ボーナス

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
        return o

    def render(self):  # noqa: C901
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
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
