# src/envs/hns28_environment.py
# hns28_environment.py 
import math
from collections import deque
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

    def print_throttled(self, key, message, current_step, force=False):
        # debug統計以外の出力は抑制
        return

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

# --- 環境クラス定義 ---
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

        # collect basic stats
        self._debug_reward_buffer.append(reward)
        self._debug_hide_buffer.append(0 if info.get('is_detected', False) else 1)
        wd = info.get('wall_distance', None)
        if wd is not None:
            if isinstance(wd, (list, tuple, np.ndarray)):
                self._debug_wall_distance_buffer.extend([float(v) for v in wd])
            else:
                self._debug_wall_distance_buffer.append(float(wd))

        self._debug_step_counter += 1

        # periodic logging (only print stats that are reliably collected)
        if self._debug_step_counter >= self._debug_log_interval:
            avg_reward = float(np.mean(self._debug_reward_buffer)) if self._debug_reward_buffer else 0.0
            hide_rate = float(np.mean(self._debug_hide_buffer)) if self._debug_hide_buffer else 0.0
            print(f"[DEBUG] Step={self.current_step} AvgR={avg_reward:.3f} HideRate={hide_rate:.2f}")
            self._debug_reward_buffer.clear()
            self._debug_hide_buffer.clear()
            self._debug_wall_distance_buffer.clear()
            self._debug_step_counter = 0

    """
    Hide and Seek 高度物理環境 (単一エージェント学習最適化版)
    """
    ARENA_HALF = 6.0
    SAFE_HALF = 5.0
    R_AGENT = 0.55
    R_BOX = 0.95
    R_RAMP = 1.30

    AGENT_MASS = 15.0 # 15.0
    AGENT_DAMPING_XY = 150.0
    AGENT_DAMPING_Z = 20.0
    AGENT_DAMPING_ROT = 50.0
    AGENT_ACTUATOR_FWD = 9000 # 1500
    AGENT_TURN_GAIN = 300
    AGENT_Z_POS = 0.5
    AGENT_Z_FLOAT = 0.02 # 0.02
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
    GRAB_STOP_EPS = 0.03
    GRAB_OWNER_VEL_EPS = 0.02
    GRAB_BREAK_DIST = 2.8
    GEAR_CAP = 0.4
    RAMP_BOOST_FWD = 0.8 # 0.35
    DIST_BONUS_SCALE = 0.48
    BASE_REWARD_SCALE = 1.0
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
                 shared_team_prefix=None,
                 dist_bonus_scale=None,
                 base_reward_scale=None):
        # print("dist_bonus_scale=", dist_bonus_scale, "base_reward_scale=", base_reward_scale)
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
        # 構築時に距離ボーナス係数を上書きできるようにする
        if dist_bonus_scale is not None:
            try:
                self.DIST_BONUS_SCALE = float(dist_bonus_scale)
            except Exception:
                pass
        # 基本報酬スケールの上書きを許可する（チーム報酬の `base` 項に重みをかける）
        if base_reward_scale is not None:
            try:
                self.BASE_REWARD_SCALE = float(base_reward_scale)
            except Exception:
                pass
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

        self._ens_orderings = {}
        for ak in self.agent_keys:
            ens = [k for k in self.agent_keys if k != ak]
            if ak.startswith("s"):
                ens.sort(key=lambda k: (0 if k.startswith("h") else 1, k))
            else:
                ens.sort(key=lambda k: (0 if k.startswith("s") else 1, k))
            self._ens_orderings[ak] = ens
        # keep a short history (last 3) of hider->seeker visible distances only.
        # used for a short grace period when visibility is temporarily lost.
        self._recent_min_seeker_dists = deque(maxlen=3)

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
        # 共有/チームポリシー（外部から読み込むことも可能）。モデルが
        # 指定された場合、shared_team_policy を自動的に有効にします。
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

        # envの外でポリシーの動作を一元化するアダプター
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
        # 可視性キャッシュをデフォルト（すべて False）にリセットします。キーセットは変更しません。
        try:
            self._cached_vis = {}
            # 内側ループ: 各シーカーがこのハイダーを見ているかを調べる
            # 注意: 学習対象がシーカーの場合（learnable_agent_key がシーカーである場合）は
            # 内側ループ内で `learnable_hider_seen` を立てる（学習シーカーがこのハイダーを見ているか）。
            # 学習対象がハイダーの場合は、内側ループを抜けた後に外側で `seen` を確認して
            # `learnable_hider_seen` を設定する（このハイダーは任意のシーカーに見られたか）。
            for sk in self.seeker_keys:
                sid = self._resolve_body_id(sk)
                for hk in self.hider_keys:
                    hid = self._resolve_body_id(hk)
                    self._cached_vis[(sid, hid)] = False
        except Exception:
            self._cached_vis = {}
        
        # 観測空間
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.idx.total_dim,), np.float32)
        
        # 【修正】アクションスペースを 4 次元に固定。
        # 外部（学習アルゴリズム）からは常に 1 体分の入力を受け取る。
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

        # visibility cache (initialized properly in `reset()`)
        self._cached_vis = {}

    # 共有チームポリシーの状態を設定するメソッド
    def set_shared_team_policy_state(self, state_dict, seq_len=8, hidden_dim=128):
        return self.policy_adapter.set_shared_team_policy_state(state_dict, seq_len=seq_len, hidden_dim=hidden_dim)

    # 個別ポリシーの状態を設定するメソッド
    def set_inference_policy_state(self, agent_keys, state_dict, seq_len=8, hidden_dim=128):
        return self.policy_adapter.set_inference_policy_state(agent_keys, state_dict, seq_len=seq_len, hidden_dim=hidden_dim)

    # 学習可能エージェントのポリシーを上書きするフラグを設定するメソッド
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

    # オブジェクトの平面上の位置と向きをキャッシュするヘルパー関数
    def _cache_planar_object_pose(self):
        for tk in self.obj_body_map:
            qadr, _ = self._obj_addr(tk)
            if tk.startswith("ramp"):
                self.object_state[tk]["planar_z"] = 0.0
            else:
                self.object_state[tk]["planar_z"] = 0.5
            self.object_state[tk]["planar_quat"] = self.data.qpos[qadr + 3:qadr + 7].copy()

    # オブジェクトのジョイント位置と速度のアドレスを取得するヘルパー関数
    def _obj_addr(self, obj_key):
        bid = self.obj_body_map[obj_key]
        jadr = self.model.body_jntadr[bid]
        return self.model.jnt_qposadr[jadr], self.model.jnt_dofadr[jadr]

    # 指定した体IDのXY平面での速度を計算するヘルパー関数
    def _body_speed_xy(self, bid):
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[bid]]
        qlen = self.data.qvel.shape[0]
        vx = self.data.qvel[vadr] if vadr < qlen else 0.0
        vy = self.data.qvel[vadr + 1] if (vadr + 1) < qlen else 0.0
        return math.sqrt(vx ** 2 + vy ** 2)

    # 指定した体IDのXY平面での位置を返すヘルパー関数（データへのビュー）
    def _body_pos(self, bid):
        """Return the (x,y) position of a body (view into `self.data.xpos`)."""
        return self.data.xpos[bid][:2]

    # 指定した体IDのXY平面での位置を返すヘルパー関数（コピーを返す）
    def _body_pos_copy(self, bid):
        """Return a copy of the (x,y) position of a body."""
        return self.data.xpos[bid][:2].copy()

    # 指定したエージェントキーのXY平面での位置を返すヘルパー関数
    def _agent_pos(self, ak):
        """Return the (x,y) position for agent key `ak`."""
        bid = self._resolve_body_id(ak)
        return self._body_pos(bid)

    # 指定したエージェントキーのZ軸位置を返すヘルパー関数
    def _agent_z_pos(self, ak):
        """Return the z-position for agent key `ak`."""
        bid = self._resolve_body_id(ak)
        return self.data.xpos[bid][2]

    # 指定したエージェントキーのZ軸回転を返すヘルパー関数
    def _agent_rot(self, ak):
        """Return the scalar yaw/rotation qpos for agent `ak`."""
        return float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]['rot']]])

    # 指定したエージェントキーのXY平面での速度を返すヘルパー関数
    def _resolve_body_id(self, ak_or_id):
        """Resolve an argument that may be an agent key (str) or a body id (int).
        Returns the body id (int) or None if lookup fails.
        """
        if isinstance(ak_or_id, str):
            return self.body_ids.get(ak_or_id)
        return ak_or_id

    # 指定したランプの上向き方向をXY平面で返すヘルパー関数
    def _ramp_uphill_dir(self, rid):
        quat = self.data.xquat[rid]
        yaw = math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)

    # ランプがブロックされているか、またはロックされているかを判定するヘルパー関数
    def _is_ramp_blocked_or_locked(self, ramp_key, rid):
        if self.object_state[ramp_key]["mode"] == "locked":
            return True
        rpos = self._body_pos(rid)
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
            if np.linalg.norm(self._body_pos(bx) - rpos) < 1.7:
                return True
        for wx, wy, sx, sy in self.maze_walls:
            if abs(rpos[0] - wx) <= sx + 0.9 and abs(rpos[1] - wy) <= sy + 0.9:
                return True
        return False

    # ランプブーストのゲインを観測に基づいて計算するヘルパー関数
    def _ramp_boost_gain_from_obs(self, ak, obs):
        """Observation-based計算でランプブーストを評価する（出力を現行実装と一致させる目的）。

        `obs` は `self._get_obs(i)` によって得られる未正規化の観測配列であることを想定する。
        """
        idx = self.idx
        si = idx.SELF
        # agent-frame velocity が観測に含まれる
        vx = float(obs[si.VEL_X])
        vy = float(obs[si.VEL_Y])
        vel_xy = np.array([vx, vy], dtype=np.float32)

        # 落下中はブースト無効（既存実装に合わせる）
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

    # エージェント間の相互作用が静的な壁によって遮られているかを判定するヘルパー関数
    def _blocked_by_walls(self, p1, p2):
        return bool(_blocked_walls_numba(
            float(p1[0]),
            float(p1[1]),
            float(p2[0]),
            float(p2[1]),
            self.static_wall_aabbs,
            float(self.INTERACT_OCCLUSION_MARGIN),
        ))

    # 指定したエージェントが相互作用の対象として選択しているオブジェクトのキーを返すヘルパー関数
    def _select_target(self, ak, for_grab=False):
        apos = self._agent_pos(ak)
        rot = self._agent_rot(ak)
        fwd = np.array([math.cos(rot), math.sin(rot)], dtype=np.float32)
        aid = self._resolve_body_id(ak)
        best_key, best_dist = None, 1e9
        for tk, bid in self.obj_body_map.items():
            st = self.object_state[tk]
            if for_grab and st["mode"] == "locked" and st["owner"] != ak:
                continue
            opos = self._body_pos(bid)
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

    # 指定したエージェントが現在掴んでいるオブジェクトのキーを返すヘルパー関数
    def _current_grabbed_by(self, ak):
        for tk, st in self.object_state.items():
            if st["mode"] == "grabbed" and st["owner"] == ak:
                return tk
        return None

    # オブジェクトのlock状態を切り替えるヘルパー関数
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

    # オブジェクトのgrab状態を切り替えるヘルパー関数
    def _toggle_grab(self, ak):
        cur = self._current_grabbed_by(ak)
        tk = self._select_target(ak, for_grab=True)

        if cur is not None:
            aid = self._resolve_body_id(ak)
            cid = self.obj_body_map[cur]
            apos = self._agent_pos(ak)
            cpos = self._body_pos(cid)
            if self._blocked_by_walls(apos, cpos):
                return False
            if not self.vis_engine.is_visible(
                apos, cpos, body_exclude=aid, target_body_id=cid
            ):
                return False

        if cur is not None and (tk is None or tk == cur):
            self.object_state[cur]["mode"] = "free"
            self.object_state[cur]["owner"] = None
            # パラメータを復元
            try:
                self._restore_object_contact(cur)
            except Exception:
                pass
            # 保存されているローカルグラブオフセットがある場合はクリア
            self.object_state[cur].pop("grab_local_offset", None)
            self.object_state[cur].pop("grab_local_angle", None)
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
            # ローカルオフセット (owner-local frameで表現されたobject位置)を保存
            try:
                owner_pos = self._agent_pos(ak)
                cid = self.obj_body_map[tk]
                obj_pos = self._body_pos(cid)
                rot = float(self._agent_rot(ak))
                c, s = math.cos(rot), math.sin(rot)
                rel = obj_pos - owner_pos
                local = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])
                st["grab_local_offset"] = local
            except Exception:
                st["grab_local_offset"] = None
            # 角度オフセットを保存する（親のヨーに対するオブジェクトのヨー）
            try:
                cid = self.obj_body_map[tk]
                quat = self.data.xquat[cid]
                w, x, y, z = quat[0], quat[1], quat[2], quat[3]
                obj_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                owner_rot = float(self._agent_rot(ak))
                ang_off = obj_yaw - owner_rot
                ang_off = (ang_off + math.pi) % (2.0 * math.pi) - math.pi
                st["grab_local_angle"] = ang_off
            except Exception:
                st["grab_local_angle"] = 0.0
            # 把持対象物の接触パラメータを緩和し、ソルバーの急激な反応を抑える
            try:
                self._soften_object_contact(tk)
            except Exception:
                pass
            return True
        return False

    # ボディに対して接触パラメータをソフト化するヘルパー関数
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

    # オブジェクトの接触パラメータをソフト化するヘルパー関数
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
            qadr, vadr = self._obj_addr(tk) # 位置と速度の開始アドレス
            if st["mode"] == "locked" and st["locked_pose"] is not None:
                self.data.qpos[qadr:qadr+POSE_SIZE] = st["locked_pose"]
                self.data.qvel[vadr:vadr+VEL_SIZE] = 0.0
            elif st["mode"] == "grabbed" and st["owner"] is not None:
                owner = st["owner"]
                opos = self._body_pos(self._resolve_body_id(owner))
                cur_xy = self.data.qpos[qadr:qadr+POSE_SIZE][XY_START:XY_STOP].copy()
                # --- 距離・壁判定によるgrab解除 ---
                grab_dist = float(np.linalg.norm(cur_xy - opos))
                if (
                    grab_dist > self.GRAB_BREAK_DIST or
                    self._blocked_by_walls(opos, cur_xy)
                ):
                    st["mode"] = "free"
                    st["owner"] = None
                    # 距離または壁によってグラブが解除された際に、コンタクトパラメータを復元する
                    try:
                        self._restore_object_contact(tk)
                    except Exception:
                        pass
                    self.data.qvel[vadr:vadr+VEL_SIZE][XY_VEL_START:XY_VEL_STOP] *= 0.5
                    continue
                # 所有者ローカル座標系での相対オフセットを維持する
                try:
                    rot = float(self._agent_rot(owner))
                    c, s = math.cos(rot), math.sin(rot)
                except Exception:
                    c, s = 1.0, 0.0

                if st.get("grab_local_offset") is None:
                    rel = cur_xy - opos
                    local = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]])
                    st["grab_local_offset"] = local
                else:
                    local = np.asarray(st["grab_local_offset"])

                # world offset = R * local
                world_off = np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])
                desired_pos = opos + world_off
                err_xy = desired_pos - cur_xy

                # 所有者の速度を計算して静止状態を検出する
                try:
                    owner_bid = self._resolve_body_id(owner)
                    owner_vadr = self.model.jnt_dofadr[self.model.body_jntadr[owner_bid]]
                    owner_lin_vel = np.array([float(self.data.qvel[owner_vadr]), float(self.data.qvel[owner_vadr + 1])])
                    owner_rot_vel_idx = self.model.jnt_dofadr[self.qpos_indices[owner]["rot"]]
                    owner_ang_z = float(self.data.qvel[owner_rot_vel_idx]) if owner_rot_vel_idx >= 0 else 0.0
                except Exception:
                    owner_lin_vel = np.array([0.0, 0.0])
                    owner_ang_z = 0.0

                # 力の積算を解消
                try:
                    self.data.qfrc_applied[vadr:vadr+VEL_SIZE] = 0.0
                except Exception:
                    pass

                # 位置誤差に比例した速度を目標とする（比例ゲイン制御）。マジックナンバーは定数化。
                try:
                    desired_vel = err_xy * float(self.GRAB_FOLLOW_GAIN)
                    speed = float(np.linalg.norm(desired_vel))
                    maxs = float(self.GRAB_MAX_SPEED)
                    if speed > maxs and speed > 0.0:
                        desired_vel = desired_vel * (maxs / speed)
                except Exception:
                    desired_vel = np.array([0.0, 0.0])

                # 所有者がほぼ静止しており、誤差が小さい場合は、対象物の動きを停止させる
                err_norm = float(np.linalg.norm(err_xy))
                if err_norm < float(self.GRAB_STOP_EPS) and np.linalg.norm(owner_lin_vel) < float(self.GRAB_OWNER_VEL_EPS) and abs(owner_ang_z) < float(self.GRAB_OWNER_VEL_EPS):
                    try:
                        self.data.qvel[vadr:vadr+VEL_SIZE][XY_VEL_START:XY_VEL_STOP] = np.array([0.0, 0.0])
                        self.data.qvel[vadr:vadr+VEL_SIZE][Z_VEL_IDX] = 0.0
                        self.data.qvel[vadr:vadr+VEL_SIZE][ANG_VEL_START:ANG_VEL_STOP] = np.array([0.0, 0.0, 0.0])
                    except Exception:
                        pass
                else:
                    # 所望の面内速度を適用し、1ステップごとの変化を制限する
                    try:
                        cur_vel = self.data.qvel[vadr:vadr+VEL_SIZE][XY_VEL_START:XY_VEL_STOP].copy()
                    except Exception:
                        cur_vel = np.array([0.0, 0.0])
                    delta = desired_vel - cur_vel
                    MAX_DELTA = 0.5
                    dnorm = float(np.linalg.norm(delta))
                    if dnorm > MAX_DELTA and dnorm > 0.0:
                        delta = delta * (MAX_DELTA / dnorm)
                    new_vel = cur_vel + delta
                    try:
                        # 平面上のXY方向の速度を記述し、Z方向の速度と角速度はゼロとする
                        self.data.qvel[vadr + XY_VEL_START : vadr + XY_VEL_STOP] = new_vel
                        self.data.qvel[vadr + Z_VEL_IDX] = 0.0
                        self.data.qvel[vadr + ANG_VEL_START : vadr + ANG_VEL_STOP] = np.array([0.0, 0.0, 0.0])
                    except Exception:
                        pass

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

    # --- ポリシー呼び出しのためのヘルパー関数群 ---
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
        # このエージェントには事前に計算済みの `ens` 順序を使用し、スクリプト化されたエージェントが
        # どの OTHERS エントリがどのエージェントキーに対応しているかを判別できるようにする。
        ens = self._ens_orderings.get(agent_key, [k for k in self.agent_keys if k != agent_key])
        # ObsIdx に、OTHERS エントリからエージェントキーへのマッピングを通知する
        try:
            self.idx.set_others_keys(ens)
        except Exception:
            pass
        arr = np.asarray(self.npcs[agent_key].get_action(norm_obs, self.idx, agent_key=agent_key, ens=ens)).reshape(-1)
        if arr.size >= 4:
            return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        return float(arr[0]), float(arr[1]), 0.0, 0.0

    #--- デバッグ用のポリシーソースロギング ---
    def _log_policy_source(self, agent_key, source):
        if not self.debug_mode:
            return
        self.debug_logger.log_policy_source(agent_key, source, self.current_step, force=True)

    #--- オブジェクトの接触パラメータを緩和 ---
    def _soften_object_contact(self, tk):
        """オブジェクトのジオメトリに対する接触パラメータを一時的に緩和し、元のパラメータを保存する。
        これにより、掴まれたオブジェクトは環境との相互作用を維持しつつ、
        所有者の動きにより容易に追従するようになります（接触の剛性が低下し、摩擦が減少します）。"""
        st = self.object_state.get(tk)
        if st is None:
            return
        if st.get("contact_backup") is not None:
            return
        gids = self.obj_geom_ids.get(tk, [])
        backup = {}
        for gid in gids:
            try:
                orig_fric = self.model.geom_friction[gid].copy()
            except Exception:
                orig_fric = self.model.geom_friction[gid]
            try:
                orig_solref = self.model.geom_solref[gid].copy()
            except Exception:
                orig_solref = self.model.geom_solref[gid]
            try:
                orig_solimp = self.model.geom_solimp[gid].copy()
            except Exception:
                orig_solimp = self.model.geom_solimp[gid]
            backup[gid] = (orig_fric, orig_solref, orig_solimp)
            # apply softer parameters
            try:
                self.model.geom_friction[gid] = np.array([0.2, 0.05, 0.001], dtype=np.float32)
            except Exception:
                pass
            try:
                self.model.geom_solref[gid] = np.array([0.06, 1.0], dtype=np.float32)
            except Exception:
                pass
            try:
                self.model.geom_solimp[gid] = np.array([0.9, 0.99, 0.001], dtype=np.float32)
            except Exception:
                pass
        st["contact_backup"] = backup

    # --- オブジェクトの接触パラメータを復元 ---
    def _restore_object_contact(self, tk):
        """Restore contact parameters saved by `_soften_object_contact`."""
        st = self.object_state.get(tk)
        if st is None:
            return
        backup = st.pop("contact_backup", None)
        if not backup:
            return
        for gid, (f, sr, si) in backup.items():
            try:
                self.model.geom_friction[gid] = f
            except Exception:
                pass
            try:
                self.model.geom_solref[gid] = sr
            except Exception:
                pass
            try:
                self.model.geom_solimp[gid] = si
            except Exception:
                pass

    # --- スポーン位置の妥当性チェック ---
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

    # --- OpenAI Gym API methods ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None:
            # APIの互換性を維持するため、optionsパラメータを残す
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

        # 可視性キャッシュを初期化する
        self._cached_vis = self._compute_visibility_cache()

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

        # 起動時の reset では検証用の wall_distance 情報を返さない（学習用ログは wandb に任せる）
        return self._normalize_obs(self._get_obs(idx_to_obs)), {"is_detected": False}

    # --- OpenAI Gym API methods ---
    def step(self, action):
        self.current_step += 1
        # 各エージェントの観測と状態を事前にキャッシュしておく
        pre_obs_by_agent = {}
        pre_state_by_agent = {}
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
            
            # 準備期間中のSeekerは無行動
            if is_seeker and self.current_step <= self.prep_steps:
                f, t, lck, grb = 0.0, 0.0, 0.0, 0.0
            # 学習エージェントで外部アクションを使う場合
            elif ak == self.learnable_agent_key and not self.override_learnable_policy:
                f = af[0] if len(af) > 0 else 0.0
                t = af[1] if len(af) > 1 else 0.0
                lck = af[2] if len(af) > 2 else 0.0
                grb = af[3] if len(af) > 3 else 0.0
            # それ以外は推論モデルを使用
            else:
                norm_obs = self._normalize_obs(self._get_obs(i))
                f, t, lck, grb = self._policy_action(ak, norm_obs)

            f = self.GEAR_CAP * f 
            # xpos/xquatへの直接的な依存を避けるため、観測データからブースト値を計算する
            obs_raw = self._get_obs(i)
            # ボディIDをキーとして、事前にキャッシュされた観測と状態を保存（タイミングの関係で直接アクセスできないため）
            try:
                bid = self._resolve_body_id(ak)
                pre_obs_by_agent[bid] = obs_raw.copy()
                pre_pos = self._body_pos_copy(bid)
                pre_rot = self._agent_rot(ak)
                pre_state_by_agent[bid] = {'pos': pre_pos, 'rot': pre_rot}
            except Exception:
                # ランプブーストの計算およびエージェントごとの関連する管理処理
                pre_obs_by_agent[i] = obs_raw.copy()
                pre_state_by_agent[i] = None
            # ランプブーストの計算およびエージェントごとの関連する管理処理
            boost = self._ramp_boost_gain_from_obs(ak, obs_raw)
            if boost > 0.0:
                f = float(np.clip(f + self.RAMP_BOOST_FWD * boost, -1.0, 1.0)) # 1.0
                boosted_agents += 1

            # 学習可能なエージェントに対して最終的に適用された前方伝播を記録する
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
        mujoco.mj_forward(self.model, self.data)

        # シーカーとハイダーのペアに対するポストステップ可視性キャッシュを構築する
        # ヘルパー関数は完全なキャッシュを返す（エラー時はfalseで埋められる）
        self._cached_vis = self._compute_visibility_cache()

        box_speeds = [self._body_speed_xy(bid) for bid in self.box_ids]
        moving_box_count = int(sum(1 for v in box_speeds if v > 0.06))
        
        rb, find, learnable_hider_seen = self._compute_team_reward_state(pre_state_by_agent=pre_state_by_agent)

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
        
        # z方向速度をinfoに追加
        jz = self.qpos_indices[self.learnable_agent_key]['z']
        vz_idx = self.model.jnt_dofadr[jz]
        agent_vz = float(self.data.qvel[vz_idx])
        # xy 速度と直近コントロールも取得して info に含める
        learnable_agent_body_id = self._resolve_body_id(self.learnable_agent_key)
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
            "dbg_max_box_speed": float(max(box_speeds) if box_speeds else 0.0),
            "dbg_boosted_agents": boosted_agents,
            "dbg_override_learnable_policy": bool(self.override_learnable_policy),
            "dbg_model_policy_deterministic": bool(self.model_policy_deterministic),
            
            "dbg_learnable_hider_seen": bool(learnable_hider_seen),
            "agent_vz": agent_vz,
            "agent_vx": agent_vx,
            "agent_vy": agent_vy,
            "dbg_last_ctrl_f": last_ctrl_f,
            "dbg_last_ctrl_t": last_ctrl_t,
            "applied_forward": applied_forward_learnable,
        }
        self._debug_collect_stats(reward, info)
        
        return obs, reward, False, done, info

    # --- 報酬計算の実装 ---
    def _finalize_team_reward(self, seen_count, min_seen_dist, min_world_dist, min_hider_view_dist, learnable_hider_seen):
        """Common tail for team reward computation.

        Returns (team_reward, any_seen, learnable_hider_seen)
        """
        if seen_count == 0:
            base = 1.0
            # hider->seeker 可視距離が記録されていない場合は直近のワールド距離を猶予として使う
            if min_hider_view_dist == float("inf"):
                recent = getattr(self, "_recent_min_seeker_dists", [])
                recent_has_finite = any((d != float("inf")) for d in recent)
                if recent_has_finite and min_world_dist != float("inf"):
                    dist_ratio = min(min_world_dist / 12.0, 1.0)
                else:
                    dist_ratio = 0.0
            else:
                dist_ratio = min(min_hider_view_dist / 12.0, 1.0)
            dist_bonus = dist_ratio
        else:
            base = -float(seen_count) / float(len(self.hider_keys))
            dist_far_ratio = min(min_seen_dist / 12.0, 1.0)
            seeker_proximity_penalty = (1.0 - dist_far_ratio)
            dist_bonus = -seeker_proximity_penalty

        BASE_R = float(self.BASE_REWARD_SCALE)
        BONUS_R = float(self.DIST_BONUS_SCALE) 

        team_reward = BASE_R * base + BONUS_R * dist_bonus

        # 最近の履歴を更新（deque maxlen=3）
        try:
            self._recent_min_seeker_dists.append(min_hider_view_dist)
        except Exception:
            self._recent_min_seeker_dists = deque(maxlen=3)
            self._recent_min_seeker_dists.append(min_hider_view_dist)

        return team_reward, bool(seen_count > 0), bool(learnable_hider_seen)

    # 互換性確認のため元実装を保持しつつ、状態ベースの報酬計算を行う新しい関数を定義
    def _compute_team_reward_state(self, pre_state_by_agent=None):
        """状態ベースの報酬計算（互換性確認のため元実装を保持）。

        新しい実装と同じ形式のタプルを返します:
        (team_reward, any_seen, learnable_hider_seen)
        """
        if self.current_step <= self.prep_steps:
            return 0.0, False, False

        # seeker->hider visibilityがあるペアの中での最小距離（距離ボーナスの算出に使用）
        # 全ペアの直線距離の最小値も計算しておく（visibilityが最近失われた場合のフォールバック用）
        min_world_dist = float("inf")
        try:
            for sk in self.seeker_keys:
                sid = self._resolve_body_id(sk)
                spos = self._body_pos(sid)
                for hk in self.hider_keys:
                    hid = self._resolve_body_id(hk)
                    hpos = self._body_pos(hid)
                    d = math.hypot(float(hpos[0] - spos[0]), float(hpos[1] - spos[1]))
                    if d < min_world_dist:
                        min_world_dist = d
        except Exception:
            min_world_dist = float("inf")

        # seeker->hider 可視性を直接計算して、見えているHiderの数と最小距離を求める
        seen_count = 0
        # seeker->hider visibilityがあるペアの中での最小距離（seenペナルティの算出に使用）
        min_seen_dist = float("inf")
        # hider->seeker visibilityがあるペアの最小距離（未検知時の距離ボーナスに使用）
        min_hider_view_dist = float("inf")
        # 学習対象のHiderがSeekerから見えているかどうか
        learnable_hider_seen = False 
        for hk in self.hider_keys:
            hid = self._resolve_body_id(hk)
            hpos = self._body_pos(hid)
            hrot = self._agent_rot(hk)
            seen = False

            for sk in self.seeker_keys:
                sid = self._resolve_body_id(sk)
                spos = self._body_pos(sid)
                srot = self._agent_rot(sk)
                dx = float(hpos[0] - spos[0])
                dy = float(hpos[1] - spos[1])
                dist = math.sqrt(dx * dx + dy * dy)

                # seeker->hider 可視性
                try:
                    vis_sk_h = bool(self._cached_vis.get((sid, hid), False))
                except Exception:
                    vis_sk_h = bool(self._is_vis(spos, srot, hpos, sid, hid))

                # hider->seeker 可視性: まず逆方向キャッシュを参照し、なければ _is_vis を使用する
                try:
                    cached_rev = self._cached_vis.get((hid, sid), None)
                except Exception:
                    cached_rev = None
                if cached_rev is not None:
                    vis_h_sk = bool(cached_rev)
                else:
                    try:
                        vis_h_sk = bool(self._is_vis(hpos, hrot, spos, hid, sid))
                    except Exception:
                        vis_h_sk = False

                # 各向きごとに最短距離を別々に計算する。
                # - Seeker->Hider の可視がある場合は seen側距離を更新
                # - Hider->Seeker の可視がある場合は hider視点距離を更新
                if vis_sk_h:
                    min_seen_dist = min(min_seen_dist, dist)
                if vis_h_sk:
                    min_hider_view_dist = min(min_hider_view_dist, dist)

                if vis_sk_h:
                    # このハイダーはこのシーカーに見えている
                    seen = True
                    if sk == self.learnable_agent_key:
                        learnable_hider_seen = True
                    # break をせず、他のシーカーもチェックし続ける。
                    # これにより、より近い（小さい）可視距離を持つシーカーが
                    # 存在する場合に `min_seeker_dist` を正しく最小化できる。

            if seen:
                seen_count += 1

            # 学習対象がハイダーであれば、外側ループ（このハイダー単位）の結果を反映する
            if hk == self.learnable_agent_key:
                learnable_hider_seen = bool(seen)

        return self._finalize_team_reward(
            seen_count,
            min_seen_dist,
            min_world_dist,
            min_hider_view_dist,
            learnable_hider_seen,
        )


    # --- 可視性計算の実装 ---
    def _is_vis(self, pos, rot, t_pos, my_id, t_id):
        rel = t_pos - pos; dist = math.sqrt(np.sum(rel**2)) + 1e-8
        if dist > L_SCALE or (math.cos(rot)*(rel[0]/dist) + math.sin(rot)*(rel[1]/dist)) < 0.38: return False
        return self.vis_engine.is_visible(pos, t_pos, body_exclude=my_id, target_body_id=t_id)

    # --- 可視性キャッシュの構築 ---
    def _compute_visibility_cache(self, pre_state_by_agent=None):
        """Compute visibility for all seeker->hider pairs.

        `pre_state_by_agent` が指定されている場合、利用可能な場合はそれらのポーズ/回転を使用します
        （ステップ前の可視性判定に使用されます）。そうでない場合は、現在のポーズを使用します
        （ステップ後の可視性判定に使用されます）。
        (seeker_body_id, hider_body_id) -> bool をキーとする辞書を返します。
        """
        try:
            cache = {}
            for sk in self.seeker_keys:
                sid = self._resolve_body_id(sk)
                # seeker pose: prefer provided pre-state if available
                spos = None; srot = None
                if pre_state_by_agent is not None:
                    s_pre = pre_state_by_agent.get(sid)
                    if s_pre is not None:
                        spos = s_pre.get('pos')
                        srot = s_pre.get('rot')
                if spos is None:
                    spos = self._body_pos(sid)
                if srot is None:
                    srot = self._agent_rot(sk)
                for hk in self.hider_keys:
                    hid = self._resolve_body_id(hk)
                    # hider pose: prefer provided pre-state if available
                    tpos = None
                    hrot = None
                    if pre_state_by_agent is not None:
                        t_pre = pre_state_by_agent.get(hid)
                        if t_pre is not None:
                            tpos = t_pre.get('pos')
                            hrot = t_pre.get('rot')
                    if tpos is None:
                        tpos = self._body_pos(hid)
                    if hrot is None:
                        try:
                            hrot = self._agent_rot(hk)
                        except Exception:
                            hrot = None
                    try:
                        vis_sk_h = bool(self._is_vis(spos, srot, tpos, sid, hid))
                    except Exception:
                        vis_sk_h = False
                    try:
                        vis_h_sk = bool(self._is_vis(tpos, hrot if hrot is not None else srot, spos, hid, sid))
                    except Exception:
                        vis_h_sk = False
                    # 両方向のデータをキャッシュに保存する
                    cache[(sid, hid)] = vis_sk_h
                    cache[(hid, sid)] = vis_h_sk
            return cache
        except Exception:
            # 保守的なフォールバック：すべてがfalseで埋められたキャッシュを返す
            cache = {}
            for sk in getattr(self, 'seeker_keys', []):
                sid = self._resolve_body_id(sk) if isinstance(self.body_ids, dict) else None
                for hk in getattr(self, 'hider_keys', []):
                    hid = self._resolve_body_id(hk) if isinstance(self.body_ids, dict) else None
                    if sid is None or hid is None:
                        continue
                    cache[(sid, hid)] = False
            return cache

    # -- ポーズ解決の実装（可視性キャッシュや報酬計算で使用） ---
    def _resolve_pose(self, sk, pre_state_by_agent=None):
        """指定されたシークター `sk` に対して (pos, rot) を算出する。

        `pre_state_by_agent` が指定され、かつ対応するボディ ID に対するエントリが含まれている場合は、
        そのポーズ/回転を優先する。そうでない場合は、
        現在の `self.data` の値を使用する。タプル (pos, rot) を返す。
        """
        sid = self._resolve_body_id(sk)
        # 事前状態が指定されていない場合は、明示的に現在のデータを使用する。
        if pre_state_by_agent is None:
            spos = self._body_pos(sid)
            srot = self._agent_rot(sk)
            return spos, srot

        # pre_state_by_agent が指定されている場合：そのエントリを使用するよう試みる。そうでない場合は、代替手段に切り替える
        pre = pre_state_by_agent.get(sid)
        if pre is None:
            spos = self._body_pos(sid)
            srot = self._agent_rot(sk)
        else:
            spos = pre.get('pos')
            srot = pre.get('rot')
            # いずれかのフィールドが欠落している場合は、現在の値を使用する
            if spos is None:
                spos = self._body_pos(sid)
            if srot is None:
                srot = self._agent_rot(sk)
        return spos, srot

    # --- 観測の正規化 ---
    def _normalize_obs(self, o):
        """観測値を正規化する"""
        v = o.copy(); idx = self.idx
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

    # -- 観測の取得 ---
    def _get_obs(self, idx):
        """観測値を取得する"""
        o = np.zeros(self.idx.total_dim, dtype=np.float32)
        ak, m, d = self.agent_keys[idx], self.model, self.data
        bid = self._resolve_body_id(ak)
        ps, rv = d.xpos[bid], float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        vax, vay = m.jnt_dofadr[self.qpos_indices[ak]['x']], m.jnt_dofadr[self.qpos_indices[ak]['y']]
        # 注意: ここではワールド座標系の速度をエージェント基準（body-frame）に回転している。
        # 具体的には速度ベクトルを -rv で回転させ、前方/左方成分を得る。
        # そのため o[si.VEL_X] はエージェント前方向成分（正=前）、o[si.VEL_Y] は左方向成分（正=左）を表す。
        # 一方で o[si.ROT] はワールド座標系のヨー角（rv）をそのまま格納する（ラジアン）。
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
            ps[:2], rv, 1, bid, ignore_body_id
        )
        if grabbed_key is not None:
            carry_rel = self._body_pos(ignore_body_id) - ps[:2]
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
        ens = self._ens_orderings.get(ak, [k for k in self.agent_keys if k != ak])
        for i, enm in enumerate(ens[:len(self.idx.OTHERS)]):
            en_idx = self.idx.OTHERS[i]; eid = self._resolve_body_id(enm)
            if self._is_vis(ps[:2], rv, self._body_pos(eid), bid, eid):
                d_w = self._body_pos(eid) - ps[:2]
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

    # -- 描画とクリーンアップ ---
    def render(self):
        # viewerが存在しない場合は新たに作成し、カメラを初期化する
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
                sid = self._resolve_body_id(ak)
                pos = self.data.xpos[sid]
                rot = self._agent_rot(ak)
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
                    (self._resolve_body_id(k), [1, 0, 0, 1] if ak.startswith("s") else [0, 0, 1, 1])
                    for k in self.agent_keys if k != ak
                ]
                for tid, color in targets:
                    if self._is_vis(pos[:2], rot, self._body_pos(tid), sid, tid):
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

    # 環境がクリーンアップされるときに viewer を閉じる
    def close(self):
        if self.viewer: 
            self.viewer.close()

