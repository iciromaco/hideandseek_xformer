# src/envs/hns28_environment.py
# hns28_environment.py 
import math
from src.core.constants import P_SCALE, L_SCALE, R_SCALE, V_SCALE
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch
from numba import njit
from gymnasium import spaces

from core.visibility_engine import VisibilityEngine
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider
from core.obs_indices import ObsIdx
from src.policy.policy_adapter import PolicyAdapter
from .env_xml_builder import EnvXMLBuilder

# --- DebugLoggerクラス ---
class DebugLogger:
    def __init__(self, enabled=False, log_interval_steps=100):
        self.enabled = enabled
        self.log_interval_steps = log_interval_steps
        self._log_last_step = {}
        self._policy_source_logged = set()

    def print(self, message):
        if self.enabled:
            print(message)

    def log_policy_source(self, agent_key, source, current_step, force=False):
        # policy_sourceログも抑制
        return


@njit(cache=True)
def _blocked_walls_numba(p1x, p1y, p2x, p2y, walls, margin):
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


class TeamCosEnv(gym.Env):
    # --- 統計バッファ初期化 ---
    def _init_debug_stats(self):
        self._debug_reward_buffer = []
        self._debug_hide_buffer = []
        self._debug_wall_distance_buffer = []
        self._debug_step_counter = 0
        self._debug_log_interval = getattr(self, 'debug_logger', None) and getattr(self.debug_logger, 'log_interval_steps', 100) or 100

    def _debug_collect_stats(self, reward, info):
        if not self.debug_mode:
            return
        self._debug_reward_buffer.append(reward)
        # 隠れ率: is_detected==False なら隠れているとみなす
        self._debug_hide_buffer.append(0 if info.get('is_detected', False) else 1)
        wd = info.get('wall_distance', None)
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
        n = max(len(self._debug_reward_buffer), 1)
        avg_reward = float(np.mean(self._debug_reward_buffer)) if self._debug_reward_buffer else 0.0
        hide_rate = float(np.mean(self._debug_hide_buffer)) if self._debug_hide_buffer else 0.0
        wd_arr = np.array(self._debug_wall_distance_buffer, dtype=np.float32) if self._debug_wall_distance_buffer else np.array([0.0])
        print(f"[DEBUG] Step={self.current_step} AvgR={avg_reward:.3f} HideRate={hide_rate:.2f} WallDist(mean/min/max)={wd_arr.mean():.3f}/{wd_arr.min():.3f}/{wd_arr.max():.3f}")
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
    
    AGENT_MASS = 20.0
    AGENT_DAMPING_XY = 30.0
    AGENT_DAMPING_Z = 16.0
    AGENT_DAMPING_ROT = 25.0
    AGENT_ACTUATOR_FWD = 1500
    AGENT_TURN_GAIN = 120
    AGENT_Z_POS = 0.50
    AGENT_Z_FLOAT = 0.01
    AGENT_Z_MIN = AGENT_Z_POS + AGENT_Z_FLOAT
    AGENT_Z_MAX = AGENT_Z_POS + 1.20 # エージェントのZ軸ジョイントは、初期位置から最大1.2mまで伸びる設定
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

    def __init__(self, mode="initial", target="hider", n_seekers=1,
                 n_hiders=2, n_boxes=2, n_ramps=1, render_mode=None,
                 inference_policies=None, show_turn_lines=True,
                 debug_log_interval_steps=200,
                 mode4_sdf_cell_size=0.05,
                 debug_mode=False,
                 action_repeat=16,
                 # shared policy injection (optional, external manager can pass model)
                 shared_policy_model=None,
                 shared_policy_seq_len=8,
                 shared_policy_hidden_dim=128,
                 shared_team_policy=False,
                 shared_team_prefix=None):
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
        self.debug_logger = DebugLogger(enabled=debug_mode, log_interval_steps=debug_log_interval_steps)
        self.action_repeat = action_repeat
        self._init_debug_stats()
        if self.n_seekers == 1:
            self.seeker_keys = ["s"]
        else:
            self.seeker_keys = [f"s{i}" for i in range(1, self.n_seekers + 1)]
        self.hider_keys = [f"h{i}" for i in range(1, self.n_hiders + 1)]
        self.agent_keys = self.seeker_keys + self.hider_keys
        self.learnable_agent_key = (
            self.seeker_keys[0] if target == "seeker" else self.hider_keys[0]
        )
        self.learnable_agent_index = self.agent_keys.index(self.learnable_agent_key)
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
        # Shared/team policy (can be injected from outside). If a model is
        # provided we enable shared_team_policy automatically.
        self.shared_policy_model = shared_policy_model
        self.shared_policy_seq_len = int(shared_policy_seq_len)
        self.shared_policy_hidden_dim = int(shared_policy_hidden_dim)
        self.shared_team_policy = bool(shared_team_policy) or (self.shared_policy_model is not None)
        self.shared_team_prefix = (
            shared_team_prefix if shared_team_prefix is not None
            else ("h" if self.learnable_agent_key.startswith("h") else "s")
        )
        self._policy_histories = {}
        self.override_learnable_policy = False
        self.model_policy_deterministic = True

        # adapter that centralizes policy behavior outside the env
        self.policy_adapter = PolicyAdapter(self, shared_policy_model=shared_policy_model, shared_seq_len=shared_policy_seq_len, shared_hidden_dim=shared_policy_hidden_dim, shared_team_policy=shared_team_policy, shared_team_prefix=shared_team_prefix)

        # デバッグ用ロガーに移譲
        self.last_debug_ctrl = {k: (0.0, 0.0) for k in self.agent_keys}
        
        self.body_ids, self.qpos_indices, self.actuator_ids = {}, {}, {}
        self.obj_body_map = {}
        self.obj_geom_ids = {}
        self.obj_default_rgba = {}
        self.maze_walls = [(3, 1.5, 1.5, 0.2), (-3, -1.5, 1.5, 0.2),
                          (0, -3, 0.2, 1.5), (0, 3, 0.2, 1.5)]
        s = self.ARENA_HALF
        self.static_wall_aabbs = np.asarray([
            (0.0, 6.1, s + 0.15, 0.1),
            (0.0, -6.1, s + 0.15, 0.1),
            (6.1, 0.0, 0.1, s),
            (-6.1, 0.0, 0.1, s),
            *self.maze_walls,
        ], dtype=np.float64)
        self._analyze_structure()
        self._init_agent_intelligence()
        self._init_interaction_state()
        
        # 観測空間
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.idx.total_dim,), np.float32)
        
        # 【修正】アクションスペースを 4 次元に固定。
        # 外部（学習アルゴリズム）からは常に 1 体分の入力を受け取る。
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    # 共有チームポリシーの状態を設定するメソッド
    def set_shared_team_policy_state(self, state_dict, seq_len=8, hidden_dim=128):
        return self.policy_adapter.set_shared_team_policy_state(state_dict, seq_len=seq_len, hidden_dim=hidden_dim)

    # 個別ポリシーの状態を設定するメソッド
    def set_inference_policy_state(self, agent_keys, state_dict, seq_len=8, hidden_dim=128):
        return self.policy_adapter.set_inference_policy_state(agent_keys, state_dict, seq_len=seq_len, hidden_dim=hidden_dim)

    #  学習可能エージェントのポリシーを上書きするフラグを設定するメソッド
    def set_override_learnable_policy(self, enabled):
        return self.policy_adapter.set_override_learnable_policy(enabled)

    # モデルポリシーを決定的にするかどうかのフラグを設定するメソッド
    def set_model_policy_deterministic(self, enabled):
        return self.policy_adapter.set_model_policy_deterministic(enabled)

    # ポリシー履歴の管理メソッド
    def _ensure_policy_history(self, agent_key, seq_len):
        return self.policy_adapter._ensure_policy_history(agent_key, seq_len)

    # ポリシー履歴を初期化するメソッド
    def _prime_policy_history(self, agent_key, seq_len, norm_obs):
        return self.policy_adapter._prime_policy_history(agent_key, seq_len, norm_obs)

    # ポリシー履歴を更新するメソッド
    def _update_policy_history(self, agent_key, seq_len, norm_obs):
        return self.policy_adapter._update_policy_history(agent_key, seq_len, norm_obs)

    # ポリシー履歴からシーケンスを取得するメソッド
    def _get_policy_history_seq(self, agent_key, seq_len, norm_obs):
        return self.policy_adapter._get_policy_history_seq(agent_key, seq_len, norm_obs)

    # --- 環境構築と内部状態管理のためのヘルパー関数群 ---
    def _build_dynamic_xml(self):
        return EnvXMLBuilder(self).build_dynamic_xml()

    # --- 環境構築後のモデル解析と内部状態初期化 ---
    def _analyze_structure(self):
        m = self.model
        for ak in self.agent_keys:
            self.body_ids[ak] = m.body(f"{ak}_body").id
            self.qpos_indices[ak] = {
                'x': m.joint(f"{ak}_x").id,
                'y': m.joint(f"{ak}_y").id,
                'z': m.joint(f"{ak}_z").id,
                'rot': m.joint(f"{ak}_rot").id,
            }
            self.actuator_ids[f"{ak}_fwd"] = m.actuator(f"{ak}_fwd").id
            self.actuator_ids[f"{ak}_turn"] = m.actuator(f"{ak}_turn").id
        self.ramp_ids = [
            m.body(f"ramp{i}_body").id
            for i in range(1, self.n_ramps + 1)
        ]
        self.ramp_keys = [f"ramp{i}" for i in range(1, self.n_ramps + 1)]
        self.box_ids = [m.body(f"box{i}_body").id for i in range(1, self.n_boxes + 1)]
        self.obj_body_map = {f"b{i}": bid for i, bid in enumerate(self.box_ids, start=1)}
        for i, rid in enumerate(self.ramp_ids, start=1):
            self.obj_body_map[f"ramp{i}"] = rid
        self.obj_geom_ids = {f"b{i}": [m.geom(f"box{i}_geom").id] for i in range(1, self.n_boxes + 1)}
        for i in range(1, self.n_ramps + 1):
            self.obj_geom_ids[f"ramp{i}"] = [m.geom(f"ramp{i}_geom").id]
        self.obj_default_rgba = {
            k: m.geom_rgba[v[0]].copy() for k, v in self.obj_geom_ids.items()
        }
        # エージェントのカプセルジオムIDとデフォルト色を保持（レンダリングで可視時に色を変えるため）
        self.agent_geom_ids = {ak: m.geom(f"{ak}_capsule").id for ak in self.agent_keys}
        self.agent_default_rgba = {ak: m.geom_rgba[g].copy() for ak, g in self.agent_geom_ids.items()}
        # 体IDからエージェントキーへ逆引きできるようマッピングを保持
        self.body_to_agent_key = {m.body(f"{ak}_body").id: ak for ak in self.agent_keys}

    # エージェントの知能（行動ルール）を初期化するメソッド
    def _init_agent_intelligence(self):
        self.npcs = {
            ak: (RuleBasedSeeker() if ak.startswith("s") else RuleBasedHider())
            for ak in self.agent_keys
        }

    # エージェントとオブジェクトの相互作用状態を初期化するメソッド
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
        self.btn_cooldown = {ak: 0 for ak in self.agent_keys}

    def _cache_planar_object_pose(self):
        for tk in self.obj_body_map:
            qadr, _ = self._obj_addr(tk)
            if tk.startswith("ramp"):
                self.object_state[tk]["planar_z"] = 0.0
            else:
                self.object_state[tk]["planar_z"] = 0.5
            self.object_state[tk]["planar_quat"] = self.data.qpos[qadr + 3:qadr + 7].copy()

    def _obj_addr(self, obj_key):
        bid = self.obj_body_map[obj_key]
        jadr = self.model.body_jntadr[bid]
        return self.model.jnt_qposadr[jadr], self.model.jnt_dofadr[jadr]

    def _body_speed_xy(self, bid):
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[bid]]
        qlen = self.data.qvel.shape[0]
        vx = self.data.qvel[vadr] if vadr < qlen else 0.0
        vy = self.data.qvel[vadr + 1] if (vadr + 1) < qlen else 0.0
        return math.sqrt(vx ** 2 + vy ** 2)

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
        if (
            abs(rpos[0]) > self.ARENA_HALF - margin
            or abs(rpos[1]) > self.ARENA_HALF - margin
        ):
            return True
        for bx in self.box_ids:
            if np.linalg.norm(self.data.xpos[bx][:2] - rpos) < 1.7:
                return True
        for wx, wy, sx, sy in self.maze_walls:
            if abs(rpos[0] - wx) <= sx + 0.9 and abs(rpos[1] - wy) <= sy + 0.9:
                return True
        return False

    def _ramp_boost_gain(self, ak):
        apos = self.data.xpos[self.body_ids[ak]][:2] # エージェントの位置
        apos_z = self.data.xpos[self.body_ids[ak]][2] # エージェントの高さ
        arot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]['rot']]] # エージェントの回転角
        afwd = np.array([math.cos(arot), math.sin(arot)], dtype=np.float32) # エージェントの前方ベクトル
        # prefer velocity alignment: use planar velocity to decide whether agent is moving up the ramp
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[self.body_ids[ak]]]
        vx = float(self.data.qvel[vadr]) if vadr < self.data.qvel.shape[0] else 0.0
        vy = float(self.data.qvel[vadr + 1]) if (vadr + 1) < self.data.qvel.shape[0] else 0.0
        vel_xy = np.array([vx, vy], dtype=np.float32)
        gain = 0.0
        for i, rid in enumerate(self.ramp_ids, start=1):
            rkey = f"ramp{i}"
            if not self._is_ramp_blocked_or_locked(rkey, rid):
                continue
            rpos = self.data.xpos[rid][:2] # ランプの位置
            up = self._ramp_uphill_dir(rid) # ランプの上り方向ベクトル
            side = np.array([-up[1], up[0]], dtype=np.float32) # ランプの横方向ベクトル
            rel = apos - rpos # エージェントからランプへの相対位置ベクトル
            lx = float(np.dot(rel, up)) #   ランプの上り方向への相対位置
            ly = float(np.dot(rel, side)) #   ランプの横方向への相対位置
            # decide boost based on velocity up the ramp (require positive up-component)
            vel_dot = float(np.dot(vel_xy, up))
            if abs(ly) > 0.95 or vel_dot <= 0.0 or apos_z > self.AGENT_Z_MAX: # ランプの横から大きく外れているか、上り方向への速度がない場合はブーストなし
                continue
            if -1.15 <= lx <= 0.2: # ランプの上り方向で、適度に近い位置にいる場合はブースト最大
                gain = max(gain, 1.0)
            elif 0.2 < lx <= 0.95: # ランプの上り方向で、やや遠い位置にいる場合はブーストを距離に応じて減衰
                gain = max(gain, 0.6) 
        return gain

    def _ramp_boost_gain_from_obs(self, ak, obs):
        """Observation-based計算でランプブーストを評価する（出力を現行実装と一致させる目的）。

        `obs` は `self._get_obs(i)` によって得られる未正規化の観測配列であることを想定する。
        """
        idx = self.idx
        si = idx.SELF
        vx = float(obs[si.VEL_X])
        vy = float(obs[si.VEL_Y])
        vel_xy = np.array([vx, vy], dtype=np.float32)

        # 落下中はブースト無効
        jz = self.qpos_indices[ak]['z']
        vz_adr = self.model.jnt_dofadr[jz]
        vz = float(self.data.qvel[vz_adr]) if vz_adr < self.data.qvel.shape[0] else 0.0
        if vz < -0.1:
            return 0.0

        gain = 0.0
        for i, rid in enumerate(self.ramp_ids, start=1):
            rkey = f"ramp{i}"
            if not self._is_ramp_blocked_or_locked(rkey, rid):
                continue
            r_idx = idx.RAMP[i - 1]
            rel_x = float(obs[r_idx.REL_X])
            rel_y = float(obs[r_idx.REL_Y])
            up_x = float(obs[r_idx.QUAT_0])
            up_y = float(obs[r_idx.QUAT_1])
            side_x = -up_y
            side_y = up_x

            # obs の REL は (ramp - agent) を agent-frame に回転したものなので,
            # 現行実装の rel = agent - ramp に対応させるため符号を反転する
            lx = - (rel_x * up_x + rel_y * up_y)
            ly = - (rel_x * side_x + rel_y * side_y)

            if abs(ly) > 0.8:
                continue
            vel_dot = float(np.dot(vel_xy, np.array([up_x, up_y], dtype=np.float32)))
            if vel_dot == 0.0:
                continue
            norm = float(np.linalg.norm(vel_xy))
            if norm == 0.0:
                continue
            if vel_dot / norm < 0.8:
                continue
            if vel_dot > 0.0:
                if -1.0 <= lx <= 0.2:
                    gain = max(gain, 1.0)
                elif 0.2 < lx <= 0.95:
                    gain = max(gain, 0.6)
        return gain

    def _stabilize_agent_vertical_motion(self):
        """z方向の位置・速度を定常帯に収めるための安定化処理。
        """
        for ak in self.agent_keys:
            jz = self.qpos_indices[ak]['z']
            vz_adr = self.model.jnt_dofadr[jz]
            vz = float(self.data.qvel[vz_adr])
            if vz > self.AGENT_MAX_VZ:
                self.data.qvel[vz_adr] = self.AGENT_MAX_VZ
            elif vz < -self.AGENT_MAX_VZ:
                self.data.qvel[vz_adr] = -self.AGENT_MAX_VZ

    def _blocked_by_walls(self, p1, p2):
        return bool(_blocked_walls_numba(
            float(p1[0]),
            float(p1[1]),
            float(p2[0]),
            float(p2[1]),
            self.static_wall_aabbs,
            float(self.INTERACT_OCCLUSION_MARGIN),
        ))

    def _select_target(self, ak, for_grab=False):
        apos = self.data.xpos[self.body_ids[ak]][:2]
        rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]['rot']]]
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
            if self._blocked_by_walls(apos, opos):
                continue
            if for_grab and float(np.dot(fwd, rel)) <= -0.2:
                continue
            if not self.vis_engine.is_visible(
                apos, opos, body_exclude=aid, target_body_id=bid
            ):
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
            st["locked_pose"] = self.data.qpos[qadr:qadr+7].copy()
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
            if self._blocked_by_walls(apos, cpos):
                return False
            if not self.vis_engine.is_visible(
                apos, cpos, body_exclude=aid, target_body_id=cid
            ):
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
        VEL_SIZE = 6   # qvelの成分長
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
                self.data.qpos[qadr:qadr+POSE_SIZE] = st["locked_pose"]
                self.data.qvel[vadr:vadr+VEL_SIZE] = 0.0
            elif st["mode"] == "grabbed" and st["owner"] is not None:
                owner = st["owner"]
                opos = self.data.xpos[self.body_ids[owner]][XY_START:XY_STOP]
                cur_xy = self.data.qpos[qadr:qadr+POSE_SIZE][XY_START:XY_STOP].copy()
                # --- 距離・壁判定によるgrab解除 ---
                grab_dist = float(np.linalg.norm(cur_xy - opos))
                if (
                    grab_dist > self.GRAB_BREAK_DIST or
                    self._blocked_by_walls(opos, cur_xy)
                ):
                    st["mode"] = "free"
                    st["owner"] = None
                    self.data.qvel[vadr:vadr+VEL_SIZE][XY_VEL_START:XY_VEL_STOP] *= 0.5
                    continue
                err_xy = opos - cur_xy
                self.data.qvel[vadr:vadr+VEL_SIZE][XY_VEL_START:XY_VEL_STOP] = err_xy  # 速度を直接目標方向に
                self.data.qvel[vadr:vadr+VEL_SIZE][Z_VEL_IDX] = 0.0
                self.data.qvel[vadr:vadr+VEL_SIZE][ANG_VEL_START:ANG_VEL_STOP] = 0.0
            else:
                self.data.qvel[vadr:vadr+VEL_SIZE][XY_VEL_START:XY_VEL_STOP] *= 0.9
                speed_xy = float(np.linalg.norm(self.data.qvel[vadr:vadr+VEL_SIZE][XY_VEL_START:XY_VEL_STOP]))
                if speed_xy < 1e-3:
                    self.data.qvel[vadr:vadr+VEL_SIZE][XY_VEL_START:XY_VEL_STOP] = 0.0
            # Planar lock
            if self.OBJECT_PLANAR_LOCK and st.get("planar_z") is not None:
                self.data.qpos[qadr + Z_IDX] = st["planar_z"]
                self.data.qpos[qadr + QUAT_START : qadr + QUAT_STOP] = st["planar_quat"]
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
            self._log_policy_source(agent_key, "model")
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
                self._log_policy_source(agent_key, "callable")
                pred = policy(norm_obs)
                arr = np.asarray(pred).reshape(-1)
                if arr.size >= 4:
                    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
                if arr.size >= 2:
                    return float(arr[0]), float(arr[1]), 0.0, 0.0
            except Exception:
                pass

        if (
            self.shared_team_policy
            and self.shared_policy_model is not None
            and agent_key != self.learnable_agent_key
            and agent_key.startswith(self.shared_team_prefix)
        ):
            self._log_policy_source(agent_key, "shared_model")
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

        self._log_policy_source(agent_key, "rule")
        arr = np.asarray(self.npcs[agent_key].get_action(norm_obs, self.idx)).reshape(-1)
        if arr.size >= 4:
            return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        return float(arr[0]), float(arr[1]), 0.0, 0.0


    def _log_policy_source(self, agent_key, source):
        if not self.debug_mode:
            return
        self.debug_logger.log_policy_source(agent_key, source, self.current_step, force=True)

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
        for cx, cy, hx, hy in getattr(self, 'static_wall_aabbs', []):
            if abs(px - cx) <= (hx + radius + margin) and abs(py - cy) <= (hy + radius + margin):
                return False
        for pp, pr in placed:
            if np.linalg.norm(pos_xy - pp) < (radius + pr + margin):
                return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None:
            # options parameter kept for API compatibility
            pass
        self.current_step = 0
        self._policy_histories.clear()
        self.debug_logger._log_last_step.clear()
        if self.debug_mode:
            self.debug_logger._policy_source_logged.clear()
        mujoco.mj_resetData(self.model, self.data)
        self._init_agent_intelligence()
        self._init_interaction_state()
        placed = []
        ramp_specs = [(rid, self.R_RAMP, 0.0) for rid in self.ramp_ids]
        box_specs = [(b, self.R_BOX, 0.5) for b in self.box_ids]
        for bid, rad, z in ramp_specs + box_specs:
            for _ in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                if self._is_spawn_position_valid(p, rad, placed, margin=0.2):
                    adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
                    self.data.qpos[adr:adr+7] = [p[0], p[1], z, 1, 0, 0, 0]
                    placed.append((p, rad))
                    # ランプのみx, y出力とqpos値も出力
                    # if (bid, rad, z) in ramp_specs:
                    #    print(f"[reset] ramp placed: x={p[0]:.3f}, y={p[1]:.3f}, qpos={self.data.qpos[adr]:.3f},{self.data.qpos[adr+1]:.3f}")
                    # 正常に配置できたら inner loop を抜け、次のオブジェクト配置に進む
                    break
        for ak in self.agent_keys:
            for _ in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2); rot = np.random.uniform(-np.pi, np.pi)
                if self._is_spawn_position_valid(p, self.R_AGENT, placed, margin=0.3):
                    jx = self.qpos_indices[ak]['x']
                    jy = self.qpos_indices[ak]['y']
                    jz = self.qpos_indices[ak]['z']
                    jr = self.qpos_indices[ak]['rot']
                    self.data.qpos[self.model.jnt_qposadr[jx]] = p[0]
                    self.data.qpos[self.model.jnt_qposadr[jy]] = p[1]
                    self.data.qpos[self.model.jnt_qposadr[jz]] = self.AGENT_Z_FLOAT
                    self.data.qpos[self.model.jnt_qposadr[jr]] = rot
                    placed.append((p, self.R_AGENT)); break
        mujoco.mj_forward(self.model, self.data)
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
            if (
                self.shared_team_policy
                and self.shared_policy_model is not None
                and ak.startswith(self.shared_team_prefix)
            ):
                self._prime_policy_history(ak, self.shared_policy_seq_len, norm_obs)

        idx_to_obs = self.learnable_agent_index

        # wall_distance を計算
        learnable_agent_body_id = self.body_ids[self.learnable_agent_key]
        learnable_agent_pos = self.data.xpos[learnable_agent_body_id]
        wall_dist = self.vis_engine.wall_distance(
            learnable_agent_pos[0], 
            learnable_agent_pos[1]
        )
        return self._normalize_obs(self._get_obs(idx_to_obs)), {"is_detected": False, "wall_distance": wall_dist}

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
        applied_forward_learnable = 0.0
        
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

            obs_raw = self._get_obs(i)
            boost = self._ramp_boost_gain_from_obs(ak, obs_raw)
            if boost > 0.0:
                f = float(np.clip(f + self.RAMP_BOOST_FWD * boost, -1.0, 1.0))
                boosted_agents += 1

            # record the final applied forward for the learnable agent
            if ak == self.learnable_agent_key:
                applied_forward_learnable = float(f)

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
            
            cv[self.actuator_ids[f"{ak}_fwd"]], cv[self.actuator_ids[f"{ak}_turn"]] = f, t
            self.last_debug_ctrl[ak] = (f, t)
            
        self.data.ctrl[:] = cv
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
            self._apply_object_constraints()
            self._stabilize_agent_vertical_motion()
        mujoco.mj_forward(self.model, self.data)

        box_speeds = [self._body_speed_xy(bid) for bid in self.box_ids]
        ramp_speeds = [self._body_speed_xy(rid) for rid in self.ramp_ids]
        moving_box_count = int(sum(1 for v in box_speeds if v > 0.06))
        moving_ramp_count = int(sum(1 for v in ramp_speeds if v > 0.06))
        blocked_ramp_count = int(
            sum(
                1
                for i, rid in enumerate(self.ramp_ids, start=1)
                if self._is_ramp_blocked_or_locked(f"ramp{i}", rid)
            )
        )
        
        rb, find, learnable_hider_seen = self._compute_team_reward()

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
            if (
                self.shared_team_policy
                and self.shared_policy_model is not None
                and ak.startswith(self.shared_team_prefix)
            ):
                self._update_policy_history(ak, self.shared_policy_seq_len, norm_obs_next)

        # 学習対象に合わせて観測を生成
        idx_to_obs = self.learnable_agent_index

        # 壁までの最短距離を計算
        learnable_agent_body_id = self.body_ids[self.learnable_agent_key]
        learnable_agent_pos = self.data.xpos[learnable_agent_body_id]
        wall_dist = self.vis_engine.wall_distance(
               learnable_agent_pos[0], 
               learnable_agent_pos[1]
        )
        
        # z方向速度をinfoに追加
        jz = self.qpos_indices[self.learnable_agent_key]['z']
        vz_idx = self.model.jnt_dofadr[jz]
        agent_vz = float(self.data.qvel[vz_idx])
        # xy 速度と直近コントロールも取得して info に含める
        learnable_agent_body_id = self.body_ids[self.learnable_agent_key]
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[learnable_agent_body_id]]
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
            
            "dbg_learnable_hider_seen": bool(learnable_hider_seen),
            "wall_distance": wall_dist,
            "agent_vz": agent_vz,
            "agent_vx": agent_vx,
            "agent_vy": agent_vy,
            "dbg_last_ctrl_f": last_ctrl_f,
            "dbg_last_ctrl_t": last_ctrl_t,
            "applied_forward": applied_forward_learnable,
        }
        self._debug_collect_stats(reward, info)
        
        return obs, reward, False, done, info

    def _compute_team_reward(self):
        if self.current_step <= self.prep_steps:
            return 0.0, False, False

        seen_count = 0
        min_seeker_dist = 13.0
        learnable_hider_seen = False

        for hk in self.hider_keys:
            hid = self.body_ids[hk]
            hpos = self.data.xpos[hid][:2]
            seen = False

            for sk in self.seeker_keys:
                sid = self.body_ids[sk]
                spos = self.data.xpos[sid][:2]
                srot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[sk]['rot']]]

                # ← 距離計算を追加
                dx = float(hpos[0] - spos[0])
                dy = float(hpos[1] - spos[1])
                dist = math.sqrt(dx * dx + dy * dy)
                min_seeker_dist = min(min_seeker_dist, dist)

                # gaze_cos 計算:
                # - 学習対象がハイダーの場合 (learnable_agent_key が h... ),
                #   そのハイダーに対する各シーカーの frontness を集計する。
                # - 学習対象がシーカーの場合 (learnable_agent_key が s... ),
                #   学習シーカーが各ハイダーを見ている度合いを集計する。
                # 両方のケースをサポートし、どちらか該当する場合に frontness を反映する。
                dist_with_margin = dist + 1e-8
                cos_align = (math.cos(srot) * (dx / dist_with_margin)) + (math.sin(srot) * (dy / dist_with_margin))
                frontness = max(float(cos_align), 0.0)
                # gaze metrics removed (not used in reward)
 
                if self._is_vis(spos, srot, hpos, sid, hid):
                    seen = True
                    # 学習対象がシーカーの場合、学習シーカーがこのハイダーを見ていればフラグを立てる
                    if sk == self.learnable_agent_key:
                        learnable_hider_seen = True
                    break
            if seen:
                seen_count += 1

            if hk == self.learnable_agent_key:
                learnable_hider_seen = bool(seen)

        # === 報酬計算（現行実装の説明） ===
        # 以前の単純実装（削除済み）は `team_reward = 1.0 if seen_count == 0 else -1.0` でしたが、
        # 現在は以下のように可視数と最短シーカー距離に基づく base/dist_bonus を組み合わせた
        # ロジックが採用されています（ゼロサム化はしているが、後段のカスタム報酬で非対称項が加わります）。
        
        # 新しいロジック（修正）:
        # ハイダーが1体でも見つかっていればハイダー側へ明確な負のペナルティを与える。
        # 具体的には:
        # - seen_count == 0 -> base = 1.0 (+ dist bonus)
        # - seen_count >= 1 -> base = -1.0 (明確なペナルティ)
        # これにより、学習対象（ハイダー）視点で「1体でも見られているときに中立(0)」
        # となる挙動を避けられます（n_hiders==2 の場合に見られた1体で base==0 になっていた問題を解消）。
        if seen_count == 0:
            base = 1.0
            # 敵から遠いほど+ボーナス
            dist_ratio = min(min_seeker_dist / 12.0, 1.0)
            dist_bonus = 0.2 * dist_ratio
        else:
            # seen_count が 1 のときも負の報酬を与えるが、段階的にする。
            # 例: n_hiders=2 の場合 seen_count=1 -> base=-0.5, seen_count=2 -> base=-1.0
            base = -float(seen_count) / float(len(self.hider_keys))
            # シーカーが見つけたハイダーに近づくほどハイダー側の報酬をさらに下げる
            # （これによりシーカー側は近づくインセンティブを得る）
            # min_seeker_dist が小さいほどペナルティが大きくなる。遠ければほぼ0。
            dist_far_ratio = min(min_seeker_dist / 12.0, 1.0)
            seeker_proximity_penalty = 0.2 * (1.0 - dist_far_ratio)
            dist_bonus = -seeker_proximity_penalty
        
        # team_reward = base + dist_bonus
        team_reward = 0.0
        
        return team_reward, bool(seen_count > 0), bool(learnable_hider_seen)

    def _is_vis(self, pos, rot, t_pos, my_id, t_id):
        rel = t_pos - pos; dist = math.sqrt(np.sum(rel**2)) + 1e-8
        if dist > L_SCALE or (math.cos(rot)*(rel[0]/dist) + math.sin(rot)*(rel[1]/dist)) < 0.38: return False
        return self.vis_engine.is_visible(pos, t_pos, body_exclude=my_id, target_body_id=t_id)

    def _normalize_obs(self, o):
        v = o.copy(); idx = self.idx
        v[idx.SELF.VEL_X] /= V_SCALE; v[idx.SELF.VEL_Y] /= V_SCALE; v[idx.SELF.ROT] /= R_SCALE; v[idx.LIDAR] /= L_SCALE
        for b in idx.B: v[b.REL_X] /= P_SCALE; v[b.REL_Y] /= P_SCALE; v[b.VEL_X] /= V_SCALE; v[b.VEL_Y] /= V_SCALE
        for r in idx.RAMP: v[r.REL_X] /= P_SCALE; v[r.REL_Y] /= P_SCALE; v[r.VEL_X] /= V_SCALE; v[r.VEL_Y] /= V_SCALE
        for en in idx.OTHERS: v[en.REL_X] /= P_SCALE; v[en.REL_Y] /= P_SCALE; v[en.VEL_X] /= V_SCALE; v[en.VEL_Y] /= V_SCALE
        return v

    def _get_obs(self, idx):
        o = np.zeros(self.idx.total_dim, dtype=np.float32); ak, m, d = self.agent_keys[idx], self.model, self.data
        ps, rv = d.xpos[self.body_ids[ak]], float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        vax, vay = m.jnt_dofadr[self.qpos_indices[ak]['x']], m.jnt_dofadr[self.qpos_indices[ak]['y']]
        cos_r, sin_r = math.cos(-rv), math.sin(-rv)
        si = self.idx.SELF
        o[si.VEL_X] = d.qvel[vax] * cos_r - d.qvel[vay] * sin_r
        o[si.VEL_Y] = d.qvel[vax] * sin_r + d.qvel[vay] * cos_r
        o[si.ROT] = rv; o[si.COS_ROT], o[si.SIN_ROT] = math.cos(rv), math.sin(rv)
        ignore_body_id = -1
        grabbed_key = self._current_grabbed_by(ak)
        if grabbed_key is not None:
            ignore_body_id = self.obj_body_map[grabbed_key]
        lidar_raw = self.vis_engine.cast_lidar(
            ps[:2], rv, 1, self.body_ids[ak], ignore_body_id
        )
        if grabbed_key is not None:
            carry_rel = self.data.xpos[ignore_body_id][:2] - ps[:2]
            h_cos = math.cos(rv)
            h_sin = math.sin(rv)
            for i in range(lidar_raw.shape[0]):
                vx = self.vis_engine.base_cos[i] * h_cos - self.vis_engine.base_sin[i] * h_sin
                vy = self.vis_engine.base_sin[i] * h_cos + self.vis_engine.base_cos[i] * h_sin
                proj = carry_rel[0] * vx + carry_rel[1] * vy
                if proj > 0.0:
                    lidar_raw[i] = max(0.02, float(lidar_raw[i] - proj))
        o[self.idx.LIDAR] = lidar_raw
        for i, tid in enumerate(self.box_ids):
            b_idx = self.idx.B[i]; d_w = d.xpos[tid][:2] - ps[:2]
            o[b_idx.REL_X] = d_w[0] * cos_r - d_w[1] * sin_r
            o[b_idx.REL_Y] = d_w[0] * sin_r + d_w[1] * cos_r
            b_vadr = m.jnt_dofadr[m.body_jntadr[tid]]
            b_speed = math.sqrt(d.qvel[b_vadr] ** 2 + d.qvel[b_vadr + 1] ** 2)
            o[b_idx.IS_MOVING] = 1.0 if b_speed > 0.05 else 0.0
            o[b_idx.IS_LOCKED] = 1.0 if self.object_state[f"b{i+1}"]["mode"] == "locked" else 0.0
            # 箱の向き情報（自分基準の相対yaw角）を観測にセット
            box_quat = d.xquat[tid]
            box_yaw = math.atan2(
                2.0 * (box_quat[0] * box_quat[3] + box_quat[1] * box_quat[2]),
                1.0 - 2.0 * (box_quat[2] ** 2 + box_quat[3] ** 2)
            )
            rel_box_yaw = box_yaw - rv
            o[b_idx.QUAT_0] = math.cos(rel_box_yaw)
            o[b_idx.QUAT_1] = math.sin(rel_box_yaw)
        for i, rid in enumerate(self.ramp_ids):
            r_idx = self.idx.RAMP[i]
            d_w_r = d.xpos[rid][:2] - ps[:2]
            o[r_idx.REL_X] = d_w_r[0] * cos_r - d_w_r[1] * sin_r
            o[r_idx.REL_Y] = d_w_r[0] * sin_r + d_w_r[1] * cos_r
            r_vadr = m.jnt_dofadr[m.body_jntadr[rid]]
            r_speed = math.sqrt(d.qvel[r_vadr] ** 2 + d.qvel[r_vadr + 1] ** 2)
            o[r_idx.IS_MOVING] = 1.0 if r_speed > 0.05 else 0.0
            o[r_idx.IS_LOCKED] = 1.0 if self.object_state[f"ramp{i+1}"]["mode"] == "locked" else 0.0
            # スロープの向き情報（自分基準の相対yaw角）を観測にセット
            ramp_quat = d.xquat[rid]
            ramp_yaw = math.atan2(
                2.0 * (ramp_quat[0] * ramp_quat[3] + ramp_quat[1] * ramp_quat[2]),
                1.0 - 2.0 * (ramp_quat[2] ** 2 + ramp_quat[3] ** 2)
            )
            rel_ramp_yaw = ramp_yaw - rv
            o[r_idx.QUAT_0] = math.cos(rel_ramp_yaw)
            o[r_idx.QUAT_1] = math.sin(rel_ramp_yaw)
        ens = [k for k in self.agent_keys if k != ak]
        if ak.startswith("s"):
            ens.sort(key=lambda k: (0 if k.startswith("h") else 1, k))
        else:
            ens.sort(key=lambda k: (0 if k.startswith("s") else 1, k))
        for i, enm in enumerate(ens[:len(self.idx.OTHERS)]):
            en_idx = self.idx.OTHERS[i]; eid = self.body_ids[enm]
            if self._is_vis(ps[:2], rv, d.xpos[eid][:2], self.body_ids[ak], eid):
                d_w = d.xpos[eid][:2] - ps[:2]
                o[en_idx.REL_X] = d_w[0] * cos_r - d_w[1] * sin_r
                o[en_idx.REL_Y] = d_w[0] * sin_r + d_w[1] * cos_r
                o[en_idx.VISIBLE] = 1.0
                # 他エージェントの向き情報（自分基準の相対yaw角）をcos,sinで観測にセット
                agent_rot = float(d.qpos[m.jnt_qposadr[self.qpos_indices[enm]['rot']]])
                rel_agent_rot = agent_rot - rv
                o[en_idx.QUAT_0] = math.cos(rel_agent_rot)
                o[en_idx.QUAT_1] = math.sin(rel_agent_rot)
            else:
                o[en_idx.REL_X], o[en_idx.REL_Y] = L_SCALE, L_SCALE
                o[en_idx.VISIBLE] = 0.0
                o[en_idx.QUAT_0] = 0.0
                o[en_idx.QUAT_1] = 0.0
        return o

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0, 0, 0.8]
                self.viewer.cam.distance = 18.0
                self.viewer.cam.elevation = -35.0
                self.viewer.cam.azimuth = 90.0
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            # 毎フレームエージェントの色をデフォルトに戻す
            try:
                for ak, gid in self.agent_geom_ids.items():
                    self.model.geom_rgba[gid] = self.agent_default_rgba[ak]
            except Exception:
                # 初期化順序が異なるケースを無視（安全策）
                pass
            for ak in self.agent_keys:
                sid = self.body_ids[ak]
                pos = self.data.xpos[sid]
                rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]['rot']]]
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
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 8.0 + abs(t_val)*25, p_start, p_end)
                        g.rgba[:] = color
                        self.viewer.user_scn.ngeom += 1
                targets = [
                    (self.body_ids[k], [1, 0, 0, 1] if ak.startswith("s") else [0, 0, 1, 1])
                    for k in self.agent_keys if k != ak
                ]
                for tid, color in targets:
                    if self._is_vis(pos[:2], rot, self.data.xpos[tid][:2], sid, tid):
                        # 可視なら（viewerがSeekerかつtargetがHiderの場合のみ）ターゲットのカプセルを黄色にする
                        tkey = self.body_to_agent_key.get(tid, None)
                        if tkey is not None and ak.startswith("s") and tkey.startswith("h"):
                            try:
                                self.model.geom_rgba[self.agent_geom_ids[tkey]] = np.array([1.0, 0.85, 0.1, 1.0])
                            except Exception:
                                pass
                        if self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                            g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, pos, self.data.xpos[tid])
                            g.rgba[:] = color
                            self.viewer.user_scn.ngeom += 1
        self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close()