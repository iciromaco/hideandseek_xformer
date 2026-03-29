# src/envs/hns_environment.py
# flake8: noqa
# hns_environment.py v4.5９
import logging
import math
import os
from typing import Any, Dict, Optional

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch

# オプション: グローバルなキーボードリスナー
try:
    from pynput import keyboard as _pynput_keyboard  # type: ignore
except Exception:
    _pynput_keyboard = None
from gymnasium import spaces
from numba import njit

from agents.scripted_agents import RuleBasedHider, RuleBasedSeeker
from core.constants import L_SCALE, P_SCALE, R_SCALE, V_SCALE  # 観測の正規化に使用
from core.obs_indices import ObsIdx
from core.visibility_engine import VisibilityEngine
from models.ppo_transformer_v2 import AgentV2

# モジュールロガー
logger = logging.getLogger(__name__)
# 環境変数 `HNS_LOG_LEVEL` による中央ログレベル制御
# 例: export HNS_LOG_LEVEL=DEBUG
try:
    _hns_level = os.environ.get("HNS_LOG_LEVEL", "").strip().upper()
    if _hns_level:
        if hasattr(logging, _hns_level):
            logger.setLevel(getattr(logging, _hns_level))
except Exception:
    pass


# --- DebugLoggerクラス ---
class DebugLogger:
    def __init__(self, enabled=False, console=True, log_interval_steps=100):
        # `enabled` ロガーがデバッグイベントを記録・処理するかどうかを制御
        # `console` メッセージを標準出力（stdout）に表示するかどうかを制御
        self.enabled = enabled
        self.console = console
        self.log_interval_steps = log_interval_steps
        # no environment-specific initialization here; keep logger lightweight
        # internal buffers
        self._policy_src_log = []
        self._last_step_msgs = []

    def print(self, msg: str):
        """Emit a debug message. If `console` is True print to stdout,
        otherwise route to the module logger at debug level.
        """
        try:
            if self.console:
                builtins_print = __builtins__.get("print", print) if isinstance(__builtins__, dict) else print
                builtins_print(msg)
            else:
                if isinstance(msg, str) and msg.startswith("[DEBUG]"):
                    logger.debug(msg)
                else:
                    logger.info(msg)
        except Exception:
            try:
                logger.exception("DebugLogger.print failed")
            except Exception:
                pass

    def log_policy_src(self, agent_key, source, step, force=False):
        """Record policy source; safe no-op when disabled unless force=True."""
        try:
            if not (self.enabled or force):
                return
            self._policy_src_log.append((int(step), agent_key, source))
            if self.console:
                try:
                    builtins_print = __builtins__.get("print", print) if isinstance(__builtins__, dict) else print
                    builtins_print(f"[POLICY_SRC] step={step} agent={agent_key} src={source}")
                except Exception:
                    logger.debug(f"[POLICY_SRC] step={step} agent={agent_key} src={source}")
        except Exception:
            try:
                logger.exception("DebugLogger.log_policy_src failed")
            except Exception:
                pass

    def clear_last_step(self):
        try:
            self._last_step_msgs.clear()
        except Exception:
            pass

    def clear_policy_src_log(self):
        try:
            self._policy_src_log.clear()
        except Exception:
            pass


def _euler_z_to_quat(rot: float):
    """Return quaternion (w,x,y,z) for rotation around Z by `rot` radians."""
    return np.array([math.cos(rot * 0.5), 0.0, 0.0, math.sin(rot * 0.5)], dtype=np.float32)


@njit(cache=True)  # numbaで高速化された壁との遮蔽判定関数
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


# --- TeamCosEnvクラス ---
class TeamCosEnv(gym.Env):
    # --- 統計バッファ初期化 ---
    def _init_debug_stats(self):
        self._dbg_reward_buf = []
        self._dbg_hide_buf = []
        self._dbg_wall_dist_buf = []
        self._dbg_step_ctr = 0
        self._dbg_log_int = getattr(self, "debug_logger", None) and getattr(self.debug_logger, "log_interval_steps", 100) or 100
        # トグル検出のため、各エージェントの最後のランプブースト値を保持
        self._last_ramp_boost = {k: 0.0 for k in getattr(self, "agent_keys", [])}
        # 短時間の CLEAR トランジションを無視するためのエージェントごとのホールドカウンタ
        self._ramp_boost_hold = {k: 0 for k in getattr(self, "agent_keys", [])}
        # 上昇（登り）を検出するための前フレームのエージェントのワールド z
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

    # 物理パラメータ
    # エージェントの物理パラメータ
    AGENT_DAMPING_XY = 25  # エージェントの平面移動に対するダンピング係数（摩擦のような効果）。以前は 15.0 でしたが、スタックや振動を減らすために増加させました。
    AGENT_DAMPING_Z = 20  # エージェントの垂直移動に対するダンピング係数。これもスタックや振動を減らすために増加させました。
    AGENT_DAMPING_ROT = 25  # エージェントの回転に対するダンピング係数。これもスタックや振動を減らすために増加させました。
    AGENT_ACTUATOR_FWD = 1500  # エージェントの前進アクチュエータの力のスケーリング係数。以前は 1200 でしたが、より敏感な制御と壁反発の相互作用の改善のために増加させました。
    AGENT_ACTUATOR_TURN_GAIN = 150  # エージェントの回転アクチュエータのゲイン。これもより敏感な制御のために増加させました。
    AGENT_BOTTOM_MASS = 20.0  # エージェントの底部の質量。これを頭部より重くすることで、スタックや転倒からの回復を改善します。
    AGENT_HEAD_MASS = 10.0  # エージェントの頭部の質量。これを底部より軽くすることで、スタックや転倒からの回復を改善します。
    AGENT_Z_MIN = -0.05  # エージェントの最低 z 座標（地面にめり込むのを防止）
    AGENT_Z_MAX = 1.20  # エージェントの最大 z 座標（過度のジャンプや飛行を防止）
    AGENT_MAX_VZ = 2.2  # エージェントの最大垂直速度（ジャンプや飛行を防止）

    # オブジェクトの物理パラメータ
    RAMP_JOINT_DAMPING = 80.0  # ランプのジョイントダンピング。これを増やすと、ランプがより安定し、エージェントが登るのが容易になります。
    BOX_JOINT_DAMPING = 100.0  # 箱のジョイントダンピング。これを増やすと、箱がより安定し、エージェントが押すのが容易になります。
    RAMP_MASS = 60.0  # ランプの質量。これを増やすと、ランプがより安定し、エージェントが登るのが容易になります。
    RAMP_INNER_WEIGHT_MASS = 30.0  # ランプの内側の重りの質量。これを増やすと、ランプがより安定し、エージェントが登るのが容易になります。
    BOX_MASS = 100.0  # 箱の質量。これを増やすと、箱がより安定し、エージェントが押すのが容易になります。
    INTERACT_RANGE = 1.95  # エージェントがオブジェクトを掴む/離すための最大距離。これを増やすと、掴みやすくなりますが、過度に大きくすると不自然になります。
    BTN_ON = 0.1  # 掴む/離すボタンのオン閾値。これを増やすと、掴みやすくなりますが、過度に大きくすると不自然になります。
    BTN_COOLDOWN = 8  # 掴む/離すアクションのクールダウン期間（ステップ数）。これを増やすと、掴み/離しの切り替えがより意図的になりますが、過度に大きくすると操作性が低下します。
    GRAB_OFFSET = 1.45  # エージェントの中心から掴む位置までの距離。これを増やすと、掴む位置がオブジェクトの中心から遠くなりますが、過度に大きくすると不安定になります。
    GRAB_FOLLOW_GAIN = 8.0  # 掴んだオブジェクトをエージェントの動きに追従させるための力のゲイン。これを増やすと、掴んだオブジェクトがエージェントの動きにより忠実に追従しますが、過度に大きくすると振動や不安定さを引き起こす可能性があります。
    GRAB_MAX_SPEED = 2.6  # 掴んだオブジェクトの最大追従速度。これを増やすと、掴んだオブジェクトがエージェントの急な動きにより速く追従しますが、過度に大きくすると不安定になります。
    GRAB_BREAK_DIST = 2.4  # 掴んだオブジェクトがエージェントから離れると掴みが解除される距離。これを増やすと、掴みが切れにくくなりますが、過度に大きくすると不自然になります。
    RAMP_BOOST_FWD = 0.5  # ランプを登る際に一時的に前進力をブーストする量。これを増やすと、登りが容易になりますが、過度に大きくすると不自然になります。
    RAMP_BOOST_HOLD_STEPS = 8  # ランプブーストを保持するステップ数。これを増やすと、登りがより安定しますが、過度に大きくすると不自然になります。
    OBJECT_PLANAR_LOCK = True  # オブジェクトの平面ロックを有効にするかどうか。これを True にすると、オブジェクトが地面に対して垂直に立ち、より安定しますが、過度に制限される可能性があります。
    RAMP_LOCKED_SPEED_EPS = 0.035  # ランプが「ロックされている」とみなされる速度の閾値。これを増やすと、より多くの状態がロックとみなされますが、過度に大きくすると不安定になります。
    FREE_OBJ_LINEAR_DAMP = 0.90  # 自由なオブジェクトの線形ダンピング係数。これを増やすと、オブジェクトの動きがより減衰されますが、過度に大きくすると不自然になります。
    FREE_OBJ_STOP_EPS = 0.045  # 自由なオブジェクトが「停止している」とみなされる速度の閾値。これを増やすと、より多くの状態が停止とみなされますが、過度に大きくすると不自然になります。
    INTERACT_OCCLUSION_MARGIN = 0.03  # 掴む/離すの判定において、エージェントとオブジェクトの間の遮蔽を考慮する際のマージン（m）。これを増やすと、より多くの状態が遮蔽されているとみなされますが、過度に大きくすると不自然になります。
    INFERENCE_FORWARD_SIGN = 1.0  # 推論モデルの前進出力の符号（必要に応じて反転）

    # 壁の物理パラメータ
    WALL_REPULSION_RADIUS = 0.25  # 壁からの反発を開始する距離（m）。これを増やすと、壁からより早く反発が始まりますが、過度に大きくすると不自然になります。
    AGENT_EFFECTIVE_RADIUS = 0.4  # エージェントが壁からこの距離以内に入ると反発を開始する実効半径。これを増やすと、壁からより早く反発が始まりますが、過度に大きくすると不自然になります。
    WALL_REPULSION_CLEARANCE = 0.20  # エージェント表面からのクリアランス（m）。これを増やすと、壁からより早く反発が始まりますが、過度に大きくすると不自然になります。
    WALL_REPULSION_ALPHA = 0.3  # これを増やすと、推定される前方力に対する反発が強くなりますが、過度に大きくすると不安定になります。
    WALL_REPULSION_FMAX = 800.0  # 反発力の最大値（ニュートン）これを増やすと、壁からの反発が強くなりますが、過度に大きくすると不安定になります。
    # --- 実行時デフォルト：スムージング/スケーリング/クリッピング（安全な推奨値） ---
    WALL_REPULSION_CTRL_SCALE = 0.05  #  推定ベースの反発に掛ける全体倍率（0 で無効化）これを増やすと、推定される前方力に対する反発が強くなりますが、過度に大きくすると不安定になります。
    WALL_REPULSION_CTRL_LP = 0.2  # 反発推定に適用する EMA ローパス係数（0..1）
    WALL_REPULSION_CTRL_CLIP = 50.0  # 平滑化された制御値に対する明示的な上限クリップ（None で無効）
    WALL_REPULSION_ACCUM_CLAMP_MAX = 100.0  # `data.xfrc_applied` に書き込む前のステップ毎の累積力クリップ上限

    # --- その他の環境パラメータ ---
    # デバッグ: True の場合、各ステップでサンプリングした SDF と法線を常に表示（デバッグ用）
    FORCE_WALL_SDF_DEBUG = False
    WALL_DEBUG_VERBOSE = False  # True の場合、`debug_mode=True` のときでも壁関連の冗長なデバッグ出力を有効化します。
    # デフォルトは False（出力を最小限に保つ）
    WALL_REPULSION_TARGET_DIST = 0.9  # 壁フィードバック適用距離の拡張（m）
    # （法線との内積が WALL_REPULSION_FB_FWD_DOT 未満で適用）
    WALL_REPULSION_FB_FWD_DOT = -0.2  # フィードバック力適用の向き判定閾値 エージェントの前方が壁方向に概ね向いている
    WALL_REPULSION_PASSIVE_KP = 40.0  # 積極的に壁へ押し込んでいないときに適用する受動的な反発係数
    WALL_REPULSION_PASSIVE_FMAX = 30.0  # 積極的に壁へ押し込んでいないときの反発力の最大値（ニュートン）
    WALL_REPULSION_ATTENUATE_CLEARANCE = 0.25  # このクリアランス以下では、前進出力を弱める（m）
    WALL_REPULSION_ATTENUATE_FACTOR = 0.25  # クリアランスが WALL_REPULSION_ATTENUATE_CLEARANCE 以下のときに前進出力に掛ける倍率。これを増やすと、壁からの距離が近いときの前進がより強くなりますが、過度に大きくすると不安定になります。
    # ランプベースの反発抑制が考慮されるワールド z（この高さより上では抑制を検討）
    WALL_REPULSION_SUPPRESS_Z = 0.5 + 0.2  # z margin below which suppression is applied even if not currently climbing
    # 壁に接触した際、回転（スピン）を抑えるために角運動量ダンピング（トルク）を適用し、
    # 回転制御を抑えて壁に向かって回転して垂直になるのを防ぎます。以下は保守的なデフォルト値。
    WALL_CONTACT_ROT_DAMP_K = 30.0  # 壁接触時の回転ダンピング係数。これを増やすと、壁に接触しているときの回転がより強く抑制されますが、過度に大きくすると不自然になります。
    WALL_CONTACT_TURN_ATTENUATE_FACTOR = 0.0  # チョイス A: ターン入力を完全抑制する（壁接触時）
    WALL_CONTACT_MZ_CLIP = 50.0  # 反トルク Mz のクリップ上限（絶対値）。過大なトルクで振動するのを防ぐ。

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
        control_mode: str = "auto",
        seed: Optional[int] = None,
        USE_VIEWER: bool = False,
        human_key_bindings: Optional[Dict[str, Any]] = None,
        vis_facing_threshold: Optional[float] = None,
        prep_steps: int = 80,
        max_episode_steps: int = 500,
    ):
        # apply deterministic seed at instantiation when provided
        try:
            if seed is not None:
                import random as _py_random

                _py_random.seed(int(seed))
                np.random.seed(int(seed))
                try:
                    import torch as _torch

                    _torch.manual_seed(int(seed))
                    if _torch.cuda.is_available():
                        _torch.cuda.manual_seed_all(int(seed))
                except Exception:
                    pass
                try:
                    self._init_seed = int(seed)
                except Exception:
                    pass
        except Exception:
            try:
                logger.exception("failed to apply deterministic seed in __init__")
            except Exception:
                pass

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
        # runtime-configurable episode parameters
        self.current_step = 0
        self.prep_steps = int(prep_steps)
        self.max_episode_steps = int(max_episode_steps)
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
        self._prev_vis: Dict[Any, Any] = {}
        self._prev_being_hit: Dict[Any, Any] = {}
        # render disabled flag (set if passive viewer cannot be launched on macOS)
        self._render_disabled = False
        # counters for persisting BEING_HIT in observations
        self.being_hit_persist = 3
        self._being_hit_counters: Dict[Any, int] = {}
        # cached last observation returned for the learnable agent
        self._cached_obs = None
        # per-step cache to avoid recomputing observations/visibility repeatedly
        self._frame_cache_step: int = -1
        self._frame_obs_cache: Dict[int, Any] = {}
        # per-step visibility cache: keys are (viewer_body_id, target_body_id) -> bool
        self._frame_vis_cache: Dict[Any, Any] = {}
        self._init_debug_stats()
        # Visibility facing threshold: used by cheap prefilter in `_is_vis`.
        try:
            if vis_facing_threshold is not None:
                self.VIS_FACING_THRESHOLD = float(vis_facing_threshold)
            else:
                self.VIS_FACING_THRESHOLD = float(os.environ.get("HNS_VIS_FACING_THRESHOLD", 0.38))
        except Exception:
            self.VIS_FACING_THRESHOLD = 0.38
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
        # ランプ周りの静的ジオメトリ情報キャッシュ（準備は構造解析後に行う）
        self._ramp_static_cache: Dict[int, Dict[str, Any]] = {}
        self.viewer = None
        self.inference_policies = inference_policies or {}
        self._inference_models: Dict[str, Any] = {}
        self._inference_seq_lens: Dict[str, int] = {}
        self.shared_team_policy = False
        self.shared_policy_model = None
        self.shared_policy_seq_len = 8
        self.shared_policy_hdim = 128
        self.shared_team_prefix = "h" if self.learnable_agent_key.startswith("h") else "s"
        self._policy_histories: Dict[Any, Any] = {}
        self.override_learnable_policy = False
        self.model_policy_deterministic = True
        # デバッグ用ロガーに移譲

        self.last_debug_ctrl = dict.fromkeys(self.agent_keys, (0.0, 0.0))

        # 人間操作モードの設定
        # control_mode: 'auto'|'rule'|'model'|'learn'|'human'
        self.control_mode = str(control_mode)
        # True かつ control_mode=='human' の場合、環境はビューアを有効化し、
        # ターミナルから矢印キーや lock/grab 用のキー（デフォルトは o/p）等を読み取って手動制御を可能にします。
        self.USE_VIEWER = bool(USE_VIEWER)
        self._human_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._orig_term_settings = None
        # グローバルな pynput リスナーを使うか（ビューアがフォーカスされていてもキー入力を捕捉します）。
        # デフォルトは False（オプトイン）です。
        self.human_use_pynput = False
        self._pynput_listener = None
        # True の場合、pynput から受信したキーイベントをデバッグ表示します
        self.human_debug_keys = False
        # True の場合、lock/grab ボタンを押下でトグル動作にします（通常はモーメンタリ）
        self.human_toggle_buttons = False
        # 人間用キー割当: アクション名 -> キー（単一文字またはエスケープ列）
        # アクション一覧: forward, back, left, right, lock, grab
        default_bindings = {
            "forward": "\x1b[A",  # Up arrow
            "back": "\x1b[B",  # Down arrow
            "left": "\x1b[D",  # Left arrow
            "right": "\x1b[C",  # Right arrow
            "lock": "o",
            "grab": "p",
        }
        self.human_key_bindings = human_key_bindings or default_bindings
        # 内部逆引きマップ: key -> (index, value)
        # index の意味: 0=前進/後退, 1=回転左右, 2=lock, 3=grab
        self._human_key_map = {}
        try:
            # 矢印キー等の複数文字シーケンスをマッピングキーとして許可
            fwd = self.human_key_bindings.get("forward")
            back = self.human_key_bindings.get("back")
            left = self.human_key_bindings.get("left")
            right = self.human_key_bindings.get("right")
            lock = self.human_key_bindings.get("lock")
            grab = self.human_key_bindings.get("grab")

            if fwd is not None:
                self._human_key_map[fwd] = (0, 1.0)
            if back is not None:
                self._human_key_map[back] = (0, -1.0)
            if left is not None:
                self._human_key_map[left] = (1, 1.0)
            if right is not None:
                self._human_key_map[right] = (1, -1.0)
            if lock is not None:
                self._human_key_map[lock] = (2, 1.0)
            if grab is not None:
                self._human_key_map[grab] = (3, 1.0)
        except Exception:
            # 安全側の静的マッピングにフォールバック（単一文字のデフォルト）
            self._human_key_map = {"t": (0, 1.0), "g": (0, -1.0), "f": (1, 1.0), "h": (1, -1.0), "o": (2, 1.0), "p": (3, 1.0)}

        # 実行時に壁反発用定数のインスタンス属性が存在することを保証する
        # 実行時にクラス定数がインポートやオーバーライドで欠落することがあるため、
        # テスト時にも利用可能となるようここでインスタンス側のフォールバック値を設定する。
        try:
            self.AGENT_EFFECTIVE_RADIUS = float(getattr(self.__class__, "AGENT_EFFECTIVE_RADIUS", 0.4))
        except Exception:
            self.AGENT_EFFECTIVE_RADIUS = 0.4
        try:
            self.WALL_REPULSION_CLEARANCE = float(getattr(self.__class__, "WALL_REPULSION_CLEARANCE", 0.2))
        except Exception:
            self.WALL_REPULSION_CLEARANCE = 0.2

        self.body_ids: Dict[str, int] = {}
        self.qpos_indices: Dict[str, Any] = {}
        self.actuator_ids: Dict[str, int] = {}
        self.obj_body_map: Dict[Any, Any] = {}
        self.obj_geom_ids: Dict[Any, Any] = {}
        self.obj_default_rgba: Dict[Any, Any] = {}
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
        # ランプ静的キャッシュは構造解析後に準備する（ramp_ids 等が必要）
        try:
            self._prepare_ramp_static_cache()
        except Exception:
            try:
                logger.exception("failed to prepare ramp static cache")
            except Exception:
                pass
        self._init_agent_intelligence()
        self._init_interaction_state()

        # 観測空間
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.idx.total_dim,), np.float32)

        # アクションスペースを 4 次元に固定。
        # 外部（学習アルゴリズム）からは常に 1 体分の入力を受け取る。
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    # --- ポリシー管理 ---
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

    # 学習可能エージェントの推論モデルを設定するための API。state_dict が None の場合は無効化。
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

    # 学習可能エージェントの推論モデルを削除するための API。
    def set_override_learnable_policy(self, enabled):
        self.override_learnable_policy = bool(enabled)
        return True

    # 学習可能エージェントの推論モデルを決定論的にするかどうかを設定するための API。
    def set_model_policy_deterministic(self, enabled):
        self.model_policy_deterministic = bool(enabled)
        return True

    # 人間操作モードのための API: human action を注入するための簡易なインターフェース。
    def set_human_action(self, action):
        """外部ランナーが呼ぶための簡易 API: human action を注入する。
        action: 長さ4の iterable -> [forward, turn, lock, grab]
        """
        try:
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
        except Exception:
            raise ValueError("action must be iterable of 4 numeric values")
        if arr.shape[0] != 4:
            raise ValueError("human action must have length 4")
        self._human_action = arr.copy()
        return True

    # ポリシー履歴の管理: 各エージェントとシーケンス長の組み合わせに対して、観測の循環バッファを保持します。
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

    # ポリシー履歴の初期化: シーケンス全体を現在の観測で埋めることで、初期の推論が安定するようにします。
    def _prime_policy_history(self, agent_key, seq_len, norm_obs):
        hist = self._ensure_policy_history(agent_key, seq_len)
        obs_np = np.asarray(norm_obs, dtype=np.float32).reshape(-1)
        sl = int(seq_len)
        for i in range(sl):
            hist["buffer"][i] = obs_np
            hist["buffer"][i + sl] = obs_np
        hist["ptr"] = 0

    # ポリシー履歴の更新: 新しい観測を現在の位置に書き込み、ポインタを進めます。
    def _update_policy_history(self, agent_key, seq_len, norm_obs):
        hist = self._ensure_policy_history(agent_key, seq_len)
        obs_np = np.asarray(norm_obs, dtype=np.float32).reshape(-1)
        sl = int(seq_len)
        ptr = int(hist["ptr"])
        hist["buffer"][ptr] = obs_np
        hist["buffer"][ptr + sl] = obs_np
        hist["ptr"] = (ptr + 1) % sl

    # ポリシー履歴の取得: 現在のポインタ位置からシーケンス長分の観測を返します。履歴が空の場合は、現在の観測で初期化してから返します。
    def _get_policy_history_seq(self, agent_key, seq_len, norm_obs):
        hist = self._ensure_policy_history(agent_key, seq_len)
        if not np.any(hist["buffer"]):
            self._prime_policy_history(agent_key, seq_len, norm_obs)
            hist = self._ensure_policy_history(agent_key, seq_len)
        ptr = int(hist["ptr"])
        sl = int(seq_len)
        return hist["buffer"][ptr : ptr + sl]

    # --- XML シーン構築 ---
    # 動的な要素（エージェント、箱、ランプ）を含む XML を構築します。これらの要素はリセットごとに位置が変わるため、動的に生成されます。
    def _build_dynamic_xml(self):
        arena = self._xml_static_scene()
        ramps = "".join(self._xml_ramp(i, [0, 0], 0) for i in range(1, self.n_ramps + 1))
        # 箱は原点周辺の小さな半径上に配置する（リセット時にエージェントの真上に出現しないようにする）
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

    # 静的なシーン要素の XML を構築。床、壁、迷路の構造が含まれます。これらの要素はリセットごとに位置が変わらないため、静的に生成。
    def _xml_static_scene(self):
        s = self.ARENA_HALF
        # 壁/迷路の摩擦を一時的に下げて、壁沿いのスライドを容易にする
        attr = 'friction="0.01 0.01 0.0001" solref="0.01 1" solimp="0.95 0.99 0.001"'
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

    # ランプの XML を構築。位置と回転を引数として受け取り、ランプのボディ、ジョイント、ジオメトリを定義します。ランプはスライドジョイントとヒンジジョイントを持ち、物理的に動くことができます。
    def _xml_ramp(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        inner_weight = (
            f'<geom name="ramp{i}_inner_weight" type="box" size="0.3333 0.5 0.25" '
            f'pos="0.3333 0 0.25" rgba="0 1 0 0.3" mass="{self.RAMP_INNER_WEIGHT_MASS}" '
            'solimp="0.95 0.99 0.001" friction="1.35 0.22 0.01"/>'
        )
        return f"""
        <body name="ramp{i}_body" pos="{xy[0]} {xy[1]} 0" quat="{q[0]} {q[1]} {q[2]} {q[3]}">
            <inertial pos="0.3 0 0.25" mass="{self.RAMP_MASS}" diaginertia="10 10 20"/>
            <joint name="ramp{i}_joint_x" type="slide" axis="1 0 0" damping="{self.RAMP_JOINT_DAMPING}"/>
            <joint name="ramp{i}_joint_y" type="slide" axis="0 1 0" damping="{self.RAMP_JOINT_DAMPING}"/>
            <joint name="ramp{i}_joint_z" type="hinge" axis="0 0 1" damping="{self.RAMP_JOINT_DAMPING}"/>
            <geom name="ramp{i}_geom" type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0" rgba="0 1 0 1"/>
            <geom name="ramp{i}_slope_surface" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.3" friction="1.35 0.22 0.01"/>
            <geom name="ramp{i}_back_panel" type="box" size="0.02 0.5 0.5" pos="0.6666 0 0.5" rgba="0 1 0 0.3" friction="1.35 0.22 0.01"/>
            {inner_weight}
    </body>"""

    # 箱の XML を構築。位置と回転を引数として受け取り、箱のボディ、ジョイント、ジオメトリを定義します。箱はフリージョイントを持ち、物理的に動くことができます。
    def _xml_box(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        return f"""
    <body name="box{i}_body" pos="{xy[0]} {xy[1]} 0.5" quat="{q[0]} {q[1]} {q[2]} {q[3]}">
            <joint name="box{i}_joint" type="free" damping="{self.BOX_JOINT_DAMPING}"/>
            <geom name="box{i}_geom" type="box" size="0.6 0.6 0.5" mass="{self.BOX_MASS}" solref="0.02 1" condim="3" rgba="0.75 0.55 0.3 1" friction="1.2 0.08 0.003"/>
    </body>"""

    # 箱の初期位置を決定するためのヘルパー関数。原点から離れた円周上の決定論的な位置を返します。これにより、リセット時に箱がエージェントの真上にスポーンするのを防ぎます。
    def _default_box_positions(self):
        # 原点から離れた円周上の決定論的な位置
        # リセット時に箱がエージェントの真上にスポーンするのを防ぐ
        radius = min(max(2.5, self.ARENA_HALF - 2.0), self.ARENA_HALF - 1.0)
        angles = np.linspace(0.0, 2.0 * math.pi, num=max(1, self.n_boxes), endpoint=False)
        poses = [[float(math.cos(a) * radius), float(math.sin(a) * radius)] for a in angles]
        # if n_boxes==0 this returns [], but caller won't iterate
        return poses

    # エージェントの XML を構築。位置、回転、色を引数として受け取り、エージェントのボディ、ジョイント、ジオメトリを定義します。エージェントはスライドジョイントとヒンジジョイントを持ち、物理的に動くことができます。
    def _xml_agent(self, pre, xy, rot, color):
        q = _euler_z_to_quat(rot)
        r, g, b = color
        return f"""
    <body name="{pre}_anchor" pos="{xy[0]} {xy[1]} 0.5" quat="{q[0]} {q[1]} {q[2]} {q[3]}">
      <joint name="{pre}_x" type="slide" axis="1 0 0" damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{pre}_y" type="slide" axis="0 1 0" damping="{self.AGENT_DAMPING_XY}"/>
    <joint name="{pre}_z" type="slide" axis="0 0 1" damping="{self.AGENT_DAMPING_Z}" limited="true" range="{self.AGENT_Z_MIN} {self.AGENT_Z_MAX}"/>
    <joint name="{pre}_rot" type="hinge" axis="0 0 1" damping="{self.AGENT_DAMPING_ROT}" armature="3.0"/>
      <body name="{pre}_body">
        <site name="{pre}_thrust" pos="0 0 0"/>
        <geom name="{pre}_btm" type="sphere" size="0.4" pos="0 0 -0.05" mass="{self.AGENT_BOTTOM_MASS}" friction="0.2 0.06 0.001"/>
        <geom name="{pre}_capsule" type="capsule" size="0.3 0.2" rgba="{r} {g} {b} 1" mass="{self.AGENT_HEAD_MASS}" contype="0" conaffinity="0"/>
        <geom name="{pre}_nose" type="capsule" fromto="0 0 0.3 0.3 0 0.3" size="0.09" rgba="1 1 1 1" contype="0" conaffinity="0"/>
        <geom name="{pre}_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="{r} {g} {b} 1" contype="0" conaffinity="0"/>
      </body>
    </body>"""

    # エージェントのアクチュエータの XML を構築。前進/後退用のアクチュエータと回転用のアクチュエータを定義します。これらのアクチュエータは、エージェントの動きを制御するために使用されます。
    def _xml_actuators(self, pre):
        return f"""
    <general name="{pre}_fwd" site="{pre}_thrust" gear="1 0 0 0 0 0" gainprm="{self.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{pre}_turn" joint="{pre}_rot" gear="0.5" gainprm="{self.AGENT_ACTUATOR_TURN_GAIN}" ctrlrange="-1 1"/>
    """

    # アクチュエータの XML を構築するための固定バージョン。動的なアタッチメントコードパスとの互換性のために保持されます。デフォルトでは `_xml_actuators` に委譲します。
    def _xml_actuators_fixed(self, pre):
        """Fallback / mypy-friendly variant of actuator XML builder.

        Kept for compatibility with dynamic-attachment code paths; by
        default delegate to `_xml_actuators`.
        """
        return self._xml_actuators(pre)

    # --- シーン構造の分析とキャッシュ ---
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

    # --- エージェントのインテリジェンスと相互作用状態の初期化 ---
    def _init_agent_intelligence(self):
        self.npcs = {ak: (RuleBasedSeeker() if ak.startswith("s") else RuleBasedHider()) for ak in self.agent_keys}

    # --- エージェントの相互作用状態の初期化 ---
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

    # --- logger.enabled と同期を保つための debug_mode プロパティ ---
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

    # オブジェクトの qpos インデックスと xquat インデックスを返すヘルパー関数。オブジェクトの種類に応じて、適切なインデックスを返します。
    def _obj_addr(self, obj_key):
        bid = self.obj_body_map[obj_key]
        jadr = self.model.body_jntadr[bid]
        return self.model.jnt_qposadr[jadr], self.model.jnt_dofadr[jadr]

    # オブジェクトの XY 平面での速度を計算するヘルパー関数。オブジェクトのジョイント DOF アドレスから速度を取得し、XY 成分の大きさを返します。
    def _body_speed_xy(self, bid):
        vadr = self.model.jnt_dofadr[self.model.body_jntadr[bid]]
        qlen = self.data.qvel.shape[0]
        vx = self.data.qvel[vadr] if vadr < qlen else 0.0
        vy = self.data.qvel[vadr + 1] if (vadr + 1) < qlen else 0.0
        return math.sqrt(vx**2 + vy**2)

    # ランプの上り方向を XY ベクトルで返すヘルパー関数。ランプのクォータニオンから Yaw を計算し、その方向ベクトルを返します。
    def _ramp_uphill_dir(self, rid):
        quat = self.data.xquat[rid]
        yaw = math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
        )
        return np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)

    # ランプの lx（ランプローカルな前後座標）に対する閾値を計算するヘルパー関数。ランプのジオメトリサイズとクォータニオンから、ランプの長さと幅を推定し、アプローチマージンを考慮して lx の最小値と最大値を計算します。
    def _compute_ramp_lx_bounds(self, rid):
        """
        ランプ局所的な上り勾配座標における lx の範囲（最小値、最大値）を計算。
        利用可能な場合はジオメトリの位置を使用しようとしますが、利用できない場合は妥当なデフォルト値を使用します。
        (lx_min, lx_max) を返す。
        """
        # キャッシュにある静的ジオメトリ情報を利用して重いループを回避する。
        # キャッシュが無ければ従来のフォールバックロジックに寄せる。
        c = self._ramp_static_cache.get(rid)
        if c is not None:
            half_length = float(c.get("half_length", 0.5))
            half_width = float(c.get("half_width", 0.5))
        else:
            # 互換フォールバック（従来ロジックの簡略版）
            try:
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
                half_width = float(gsize[1]) if gsize.size > 1 else 0.5
            else:
                try:
                    L = float(self.R_RAMP)
                except Exception:
                    L = 0.833
                half_length = 0.5 * L
                half_width = 0.5

        # クォータニオンからピッチを算出（asin の引数を安全にクランプ）
        q = self.data.xquat[rid]
        sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
        if sinp >= 1.0:
            pitch = math.pi / 2
        elif sinp <= -1.0:
            pitch = -math.pi / 2
        else:
            pitch = math.asin(sinp)

        projected_half = half_length * float(math.cos(pitch))
        APPROACH_MARGIN = 0.4
        EPS_MARGIN = 0.0
        lx_min = -projected_half - APPROACH_MARGIN - EPS_MARGIN
        lx_max = projected_half + EPS_MARGIN
        ly_thresh = half_width + 0.05
        return lx_min, lx_max, ly_thresh

    # ランプの頂点 z を計算するヘルパー関数。
    def _compute_ramp_top_z(self, rid):
        """
        ジオメトリサイズとクォータニオンからランプの頂点 z（ワールド座標）を推定して返す。
        単一の float を返す（頂点 z）。計算に失敗した場合はボディの z をフォールバックとして返す。
        """
        try:
            # キャッシュから静的なジオメトリ情報を取り出す
            c = self._ramp_static_cache.get(rid)
            half_length = None
            geom_center_z = None
            geom_thickness = 0.0
            if c is not None:
                slope_gid = c.get("slope_gid")
                if slope_gid is not None:
                    try:
                        gsize = np.asarray(self.model.geom_size[slope_gid], dtype=np.float32)
                        if gsize.size >= 1:
                            half_length = float(gsize[0])
                        if gsize.size >= 3:
                            geom_thickness = float(gsize[2])
                        geom_center_z = float(self.data.geom_xpos[slope_gid][2])
                    except Exception:
                        half_length = None
                        geom_center_z = None
                else:
                    half_length = float(c.get("half_length", 0.5))
                    geom_center_z = float(self.data.xpos[rid][2])

            # フォールバック: キャッシュが使えない場合は従来ロジックを使う
            if half_length is None:
                try:
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

            # 体のクォータニオンからピッチを計算（上と同様）
            q = self.data.xquat[rid]
            sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
            if sinp >= 1.0:
                pitch = math.pi / 2
            elif sinp <= -1.0:
                pitch = -math.pi / 2
            else:
                pitch = math.asin(sinp)

            vertical_half = abs(half_length * float(math.sin(pitch)))
            top_z = float(geom_center_z) + vertical_half + float(geom_thickness)
            return top_z
        except Exception:
            try:
                return float(self.data.xpos[rid][2])
            except Exception:
                return 0.0

    # ランプが現在エージェントにホールドされている（grabbed）か、またはロックされている（locked）かを判定するヘルパー関数。ランプの状態と位置、速度から、ランプが実質的に動いていないかどうかを判断します。
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

    # ランプの静的ジオメトリ情報（geom_size 等）を一度だけ収集してキャッシュする。
    # これにより、毎ステップの model.geom_size の探索コストを削減する。
    def _prepare_ramp_static_cache(self):
        cache: Dict[int, Dict[str, Any]] = {}
        m = self.model
        for i, rid in enumerate(self.ramp_ids, start=1):
            rkey = f"ramp{i}"
            geom_ids = self.obj_geom_ids.get(rkey, []).copy() if hasattr(self, "obj_geom_ids") else []
            slope_gid = None
            gsize = None
            # まず slope surface ジオムを探す
            try:
                slope_name = f"{rkey}_slope_surface"
                gid = m.geom(slope_name).id
                slope_gid = gid
                gsize = np.asarray(m.geom_size[gid], dtype=np.float32)
            except Exception:
                slope_gid = None
                # 代表的なジオムから見つける
                for gid in geom_ids:
                    try:
                        gsz = np.asarray(m.geom_size[gid], dtype=np.float32)
                        if gsz.size > 0:
                            gsize = gsz
                            break
                    except Exception:
                        gsize = None
            if gsize is not None and gsize.size >= 1:
                half_length = float(gsize[0])
                half_width = float(gsize[1]) if gsize.size > 1 else 0.5
            else:
                try:
                    L = float(self.R_RAMP)
                except Exception:
                    L = 0.833
                half_length = 0.5 * L
                half_width = 0.5

            cache[rid] = {
                "rkey": rkey,
                "geom_ids": geom_ids,
                "gsize": gsize,
                "slope_gid": slope_gid,
                "half_length": half_length,
                "half_width": half_width,
            }
        self._ramp_static_cache = cache

    # ランプブーストのゲインを計算するヘルパー関数。エージェントとランプの位置関係、ランプの向き、エージェントの向きから、ブーストゲインを計算します。ブーストは、エージェントがランプの前方にいて、ランプに対して十分に正面を向いている場合に適用されます。
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
            # 診断用のランプ候補を記録
            # 動的境界を計算してランプ情報を保持する
            try:
                lx_min, lx_max, ly_thresh = self._compute_ramp_lx_bounds(rid)
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
            # ゲイン判定: ランプ開始側で完全ブースト、頂点付近で部分ブースト
            if lx_min <= lx <= full_thresh:
                gain = max(gain, 1.0)
            elif full_thresh < lx <= lx_max:
                gain = max(gain, 0.6)

        # ブーストのトグル検出と診断出力
        # ヒステリシスを適用: set_thresh でオン、clear_thresh でオフ
        SET_THRESH = 0.6
        CLEAR_THRESH = 0.4
        try:
            prev = float(self._last_ramp_boost.get(ak, 0.0))
        except Exception:
            prev = 0.0
        prev_on = prev >= SET_THRESH
        cand_on = gain >= SET_THRESH
        # ヒステリシスと短時間のホールドにより最終的なオン/オフを決定（誤クリア防止）
        try:
            hold = int(self._ramp_boost_hold.get(ak, 0))
        except Exception:
            hold = 0

        # 通常のヒステリシス判定
        if prev_on:
            would_clear = gain < CLEAR_THRESH
        else:
            would_clear = False

        # 以前オンで今クリア条件なら短いホールドを開始/継続
        if prev_on and would_clear:
            if hold <= 0:
                # ホールド期間を開始
                hold = int(self.RAMP_BOOST_HOLD_STEPS)
            # ホールド中はブーストを維持
            final_on = True
        elif hold > 0:
            # ホールドをデクリメントしつつブーストを強制オン
            hold = max(0, hold - 1)
            final_on = True
        else:
            # ホールド中でなければ通常のヒステリシス規則を適用
            final_on = cand_on

        # ゲインの選択: 現在の gain を優先し、現在 gain が 0 の場合は以前の値を残さない
        if final_on:
            final_gain = gain if gain > 0.0 else 0.0
        else:
            final_gain = 0.0

        # ホールド状態を永続化
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

    # エージェントの垂直方向の動きを安定させるためのヘルパー関数。エージェントの z 座標と z 速度を監視し、z 座標が許容範囲を超えた場合に位置と速度を修正します。これにより、エージェントが空中で不安定になるのを防ぎます。
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

    # エージェントの相互作用が静的な壁によって遮られているかどうかを判定するヘルパー関数。
    # エージェントの位置と対象物の位置を引数として受け取り、両者の間に静的な壁が存在するかどうかを判定。
    # 遮られている場合は True を返し、そうでない場合は False を返します。
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

    # エージェントがインタラクト可能な対象を選択するためのヘルパー関数。
    # エージェントの位置と向きを考慮して、インタラクト可能な対象の中から最も近いものを選択。
    # for_grab フラグが True の場合は、掴むことができる対象のみを考慮します。
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
            try:
                rot_view = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]])
            except Exception:
                rot_view = 0.0
            if not self._is_vis(apos, rot_view, opos, aid, bid):
                continue
            if dist < best_dist:
                best_key, best_dist = tk, dist
        return best_key

    # エージェントが現在掴んでいる対象を返すヘルパー関数。
    # エージェントが掴んでいる対象が存在する場合はそのキーを返し、存在しない場合は None を返します。
    def _current_grabbed_by(self, ak):
        for tk, st in self.object_state.items():
            if st["mode"] == "grabbed" and st["owner"] == ak:
                return tk
        return None

    # エージェントのロック状態をトグルするためのヘルパー関数。
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

    # エージェントの掴み状態をトグルするためのヘルパー関数。
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
            try:
                rot_view = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]])
            except Exception:
                rot_view = 0.0
            if not self._is_vis(apos, rot_view, cpos, aid, cid):
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

    # エージェントの入力ボタンを処理するためのヘルパー関数。エージェントのロックボタンとグラブボタンの状態を監視し、トグルイベントを検出します。クールダウン期間中はイベントを無視し、成功したイベントに対してのみクールダウンを適用します。ロックイベントとグラブイベントの両方を処理し、結果を返します。
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
            # 成功したイベント時のみクールダウンを適用（失敗時は再試行を許容する）
            if lock_evt:
                self.btn_cooldown[ak] = self.BTN_COOLDOWN
            else:
                # ランナー所有の human_action は変更しない（ランナー側で瞬間/切替制御するため）。
                # デバッグ: 候補理由を列挙
                if getattr(self, "human_debug_keys", False):
                    try:
                        apos = self.data.xpos[self.body_ids[ak]][:2]
                        out = []
                        for tk, bid in self.obj_body_map.items():
                            opos = self.data.xpos[bid][:2]
                            rel = opos - apos
                            dist = float(np.linalg.norm(rel))
                            wall_block = bool(self._interaction_blocked_by_static_walls(apos, opos))
                            try:
                                rot_view_dbg = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]])
                            except Exception:
                                rot_view_dbg = 0.0
                            vis = bool(self._is_vis(apos, rot_view_dbg, opos, self.body_ids[ak], bid))
                            out.append(f"{tk}:dist={dist:.3f},in_range={dist<=self.INTERACT_RANGE},vis={vis},wall_block={wall_block}")
                        print(f"[human_debug][LOCK_FAIL] ak={ak} candidates=[{';'.join(out)}]")
                    except Exception:
                        pass
            return lock_evt, False
        if grab_edge:
            grab_evt = self._toggle_grab(ak)
            if grab_evt:
                self.btn_cooldown[ak] = self.BTN_COOLDOWN
            else:
                # ランナー所有の human_action は変更しない（ランナー側で瞬間/切替制御するため）。
                # デバッグ出力にフォールスルー
                if getattr(self, "human_debug_keys", False):
                    try:
                        apos = self.data.xpos[self.body_ids[ak]][:2]
                        out = []
                        for tk, bid in self.obj_body_map.items():
                            opos = self.data.xpos[bid][:2]
                            rel = opos - apos
                            dist = float(np.linalg.norm(rel))
                            wall_block = bool(self._interaction_blocked_by_static_walls(apos, opos))
                            try:
                                rot_view_dbg = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]])
                            except Exception:
                                rot_view_dbg = 0.0
                            vis = bool(self._is_vis(apos, rot_view_dbg, opos, self.body_ids[ak], bid))
                            forward_dot = float(
                                np.dot(
                                    np.array(
                                        [math.cos(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]]), math.sin(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]])]
                                    ),
                                    rel,
                                )
                            )
                            out.append(f"{tk}:dist={dist:.3f},in_range={dist<=self.INTERACT_RANGE},vis={vis},wall_block={wall_block},fwd_dot={forward_dot:.3f}")
                        print(f"[human_debug][GRAB_FAIL] ak={ak} candidates=[{';'.join(out)}]")
                    except Exception:
                        pass
            return False, grab_evt
        return False, False

    # オブジェクトの Grab/Lock 状態を適用するためのヘルパー関数。オブジェクトの状態に応じて、位置と速度を更新します。Grab 状態の場合は、所有者の位置に線形追従させる（弱い溶接的な挙動）。Lock 状態の場合は、オブジェクトの姿勢を固定します。Free 状態の場合は、減衰のみを適用します。マジックナンバーは定数化して使用します。
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
            # オブジェクト毎の速度インデックス配置
            XY_VEL_START, XY_VEL_STOP = 0, 2
            if is_ramp:
                Z_VEL_IDX = 2
                # ランプはインデックス2以降に独立した角速度ブロックを持たない
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
                # 変更: オブジェクトをエージェント中心に移動させない。
                #       エージェント前方のホールド位置 (GRAB_OFFSET) へ追従させる。
                #       速度は比例ゲインで計算し、最大速度でクリップする。
                # 所有者の前方方向を計算
                try:
                    owner_rot = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[owner]["rot"]]])
                    afwd = np.array([math.cos(owner_rot), math.sin(owner_rot)], dtype=np.float32)
                except Exception:
                    afwd = np.array([1.0, 0.0], dtype=np.float32)

                # 希望保持位置は所有者の前方 GRAB_OFFSET 分離れた点
                hold_pos = opos + afwd * float(getattr(self, "GRAB_OFFSET", 1.45))
                err_xy = hold_pos - cur_xy

                # 一時的デバッグ: grab 状態での所有者入力・距離・保持位置を出力
                if getattr(self, "human_debug_grab", False):
                    try:
                        try:
                            owner_ctrl_fwd = float(self.data.ctrl[self.actuator_ids[f"{owner}_fwd"]])
                            owner_ctrl_turn = float(self.data.ctrl[self.actuator_ids[f"{owner}_turn"]])
                        except Exception:
                            owner_ctrl_fwd = None
                            owner_ctrl_turn = None
                        last_ctrl = self.last_debug_ctrl.get(owner)
                        if hasattr(self, "debug_logger") and getattr(self, "debug_mode", False):
                            self.debug_logger.print(
                                f"[DEBUG][grab] tk={tk} owner={owner} grab_dist={grab_dist:.3f} opos={opos.tolist()} hold_pos={hold_pos.tolist()} cur_xy={cur_xy.tolist()} err_xy={err_xy.tolist()} owner_ctrl_fwd={owner_ctrl_fwd} owner_ctrl_turn={owner_ctrl_turn} last_debug_ctrl={last_ctrl}"
                            )
                        else:
                            print(
                                f"[DEBUG][grab] tk={tk} owner={owner} grab_dist={grab_dist:.3f} opos={opos.tolist()} hold_pos={hold_pos.tolist()} cur_xy={cur_xy.tolist()} err_xy={err_xy.tolist()} owner_ctrl_fwd={owner_ctrl_fwd} owner_ctrl_turn={owner_ctrl_turn} last_debug_ctrl={last_ctrl}"
                            )
                    except Exception:
                        pass

                # proportional follow velocity with clip on speed
                v = float(getattr(self, "GRAB_FOLLOW_GAIN", 8.0)) * err_xy
                vnorm = float(np.linalg.norm(v))
                vmax = float(getattr(self, "GRAB_MAX_SPEED", 2.6))
                if vnorm > vmax and vnorm > 1e-9:
                    v = v * (vmax / vnorm)
                self.data.qvel[vadr : vadr + vel_size][XY_VEL_START:XY_VEL_STOP] = v
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

    # エージェントの行動を決定するためのヘルパー関数。エージェントごとに、モデル、カスタムポリシー、共有チームポリシー、ルールベースの順で行動を決定します。モデルが設定されている場合は必ずそのモデルを使用し、次にカスタムポリシー、共有チームポリシーを試し、最後にルールベースの行動を使用します。各ソースからの行動決定の過程で、デバッグログを記録します。
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

    # エージェントの行動決  定の履歴を管理するためのヘルパー関数。各エージェントごとに、過去の観測と行動を一定数保持し、モデルが必要とするシーケンス長に応じて履歴からシーケンスを生成します。新しい観測と行動を履歴に追加し、古い履歴は削除して常に最新の情報を保持します。
    # エージェントの行動決定のソースをログに記録。デバッグモードが有効な場合にのみログを記録し、エージェントキー、ソース、現在のステップを含む情報をログに出力します。
    def _log_policy_src(self, agent_key, source):
        if not self.debug_mode:
            return
        self.debug_logger.log_policy_src(agent_key, source, self.current_step, force=True)

    # メッセージをログに記録。force=True の場合は、デバッグモードに関係なく直接ログに出力します。通常は DebugLogger を使用してログを記録しますが、force=True の場合は、メッセージが "[DEBUG]" で始まる場合は logger.debug を使用し、それ以外の場合は logger.info を使用して直接ログに出力します。これにより、重要なデバッグ情報を確実にログに記録できます。
    def _dbg_print_throttled(self, message, force=False):
        # `print_throttled` removed. Preserve `force` semantics by logging directly
        # when `force=True`, otherwise route through DebugLogger.print().
        if force:
            try:
                if isinstance(message, str) and message.startswith("[DEBUG]"):
                    logger.debug(message)
                else:
                    logger.info(message)
            except Exception:
                try:
                    logger.exception("_dbg_print_throttled logging failed")
                except Exception:
                    pass
        else:
            self.debug_logger.print(message)

    # エージェントの行動履歴からモデルが必要とするシーケンスを生成するためのヘルパー関数。エージェントの過去の観測と行動を保持する履歴から、指定されたシーケンス長に応じて最新のシーケンスを生成します。シーケンスは、過去の観測と行動を連結して構成され、モデルが必要とする入力形式に合わせて整形されます。
    # エージェントのスポーン位置が有効かどうかを判定するためのヘルパー関数。指定された位置が安全な範囲内にあるかどうか、迷路の壁や静的な壁との重なりがないかどうか、既に配置されているオブジェクトとの距離が十分にあるかどうかをチェックします。これらの条件をすべて満たす場合に True を返し、そうでない場合は False を返します。
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
        # 既に配置済みのオブジェクトとの距離チェック
        for pp, pr in placed:
            if np.linalg.norm(pos_xy - pp) < (radius + pr + margin):
                return False
        return True

    def _randomly_place_bodies(self, specs, placed, trials=500, margin=0.2):
        """Place bodies (ramps/boxes) randomly using the existing placement logic.

        Args:
            specs: iterable of (body_id, radius, z)
            placed: list to append placed (pos, radius) tuples
            trials: number of sampling attempts per body
            margin: margin passed to `_is_spawn_position_valid`
        """
        for bid, rad, z in specs:
            for _ in range(trials):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                if not self._is_spawn_position_valid(p, rad, placed, margin=margin):
                    continue
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
                    if bid in getattr(self, "ramp_ids", []):
                        # slide x, slide y, hinge z
                        self.data.qpos[adr : adr + 3] = [p[0], p[1], 0.0]
                    else:
                        # boxes etc.: free joint (pos + quat)
                        self.data.qpos[adr : adr + 7] = [p[0], p[1], z, 1, 0, 0, 0]
                    placed.append((p, rad))
                except Exception:
                    if self.debug_mode:
                        self.debug_logger.print(f"[DEBUG][reset] exception placing body {bid}")
                    continue
                break

    def _randomly_place_agents(self, placed, trials=500, margin=0.3):
        """Place agents randomly using existing placement logic."""
        for ak in self.agent_keys:
            for _ in range(trials):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                rot = np.random.uniform(-np.pi, np.pi)
                if not self._is_spawn_position_valid(p, self.R_AGENT, placed, margin=margin):
                    continue
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

    def _populate_prev_vis_and_being_hit(self):
        """Populate `self._prev_vis` and `self._prev_being_hit` based on current positions.

        This extracts the duplicated logic used during `reset()` and `step()` so
        both can reuse the same code path.
        """
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
                logger.exception("failed to read viewer body/joint data while populating prev_vis; skipping viewer")
                continue
            for target in self.agent_keys:
                if target == viewer:
                    continue
                try:
                    t_bid = self.body_ids[target]
                    t_pos = self.data.xpos[t_bid][:2]
                except Exception:
                    logger.exception("failed to read target body data while populating prev_vis; marking not visible")
                    self._prev_vis[(viewer, target)] = False
                    continue
                vis = False
                try:
                    # Use direct engine call here to avoid polluting the
                    # per-frame `_frame_vis_cache` with pre-physics results.
                    p1 = np.array([v_pos[0], v_pos[1], 0.4])
                    p2 = np.array([t_pos[0], t_pos[1], 0.4])
                    vis = bool(self.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))
                except Exception:
                    logger.exception("vis_engine.is_visible failed while populating _prev_vis")
                    vis = False
                self._prev_vis[(viewer, target)] = vis
                # being_hit: viewer=hider, target=seeker -> store flag for (hider, seeker)
                if viewer.startswith("h") and target.startswith("s"):
                    is_hit = False
                    try:
                        s_idx_r = self.model.jnt_qposadr[self.qpos_indices[target]["rot"]]
                        s_rot = float(self.data.qpos[s_idx_r])
                        try:
                            p1_h = np.array([self.data.xpos[t_bid][0], self.data.xpos[t_bid][1], 0.4])
                            p2_h = np.array([v_pos[0], v_pos[1], 0.4])
                            is_hit = bool(self.vis_engine.is_visible(p1_h, p2_h, body_exclude=t_bid, target_body_id=v_bid))
                        except Exception:
                            logger.exception("vis_engine.is_visible failed when checking seeker->hider hit in prev_vis population")
                            is_hit = False
                    except Exception:
                        logger.exception("failed to read seeker rotation while computing being_hit in prev_vis population")
                        is_hit = False
                    self._prev_being_hit[(viewer, target)] = 1.0 if is_hit else 0.0
        # Invalidate per-frame observation cache to avoid serving
        # pre-physics computed observations later in this step.
        try:
            self._frame_obs_cache.clear()
        except Exception:
            self._frame_obs_cache = {}
        try:
            self._frame_vis_cache.clear()
        except Exception:
            self._frame_vis_cache = {}

    def _capture_prephysics_prev_vis(self):
        """Capture visibility / being-hit state before the physics step in `step()`.

        This mirrors the inline logic previously present in `step()` but is
        extracted for readability and reuse.
        """
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
                        p1 = np.array([v_pos[0], v_pos[1], 0.4])
                        p2 = np.array([t_pos[0], t_pos[1], 0.4])
                        vis = bool(self.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))
                    except Exception:
                        vis = False
                    self._prev_vis[(viewer, target)] = vis
                    # being_hit: viewer is seeker, target is hider -> store as (hider, seeker)
                    if viewer.startswith("s") and target.startswith("h"):
                        self._prev_being_hit[(target, viewer)] = 1.0 if vis else 0.0
        except Exception:
            # keep step robust to any visibility errors
            pass
        # Invalidate per-frame observation cache after capturing pre-physics visibility
        try:
            self._frame_obs_cache.clear()
        except Exception:
            self._frame_obs_cache = {}
        try:
            self._frame_vis_cache.clear()
        except Exception:
            self._frame_vis_cache = {}

    def _process_agent_actions(self, action_array):
        """Process `action_array` for all agents and return control vector and stats.

        Returns:
            cv: ndarray control vector sized `model.nu`
            stats: dict with keys used by `step()` (booleans, counts, applied_forward values, etc.)
        """
        af = np.ravel(action_array)
        cv = np.zeros(self.model.nu)
        any_lock_event = False
        any_grab_event = False
        any_lock_pressed = False
        any_grab_pressed = False
        any_lock_target = False
        any_grab_target = False
        max_lock_btn = 0.0
        max_grab_btn = 0.0
        boosted_agents = 0
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
                    f = af[0] if len(af) > 0 else 0.0
                    t = af[1] if len(af) > 1 else 0.0
                    lck = af[2] if len(af) > 2 else 0.0
                    grb = af[3] if len(af) > 3 else 0.0
            else:
                if is_seeker and self.current_step <= self.prep_steps:
                    f, t, lck, grb = 0.0, 0.0, 0.0, 0.0
                else:
                    norm_obs = self._normalize_obs(self._get_obs(i))
                    f, t, lck, grb = self._policy_action(ak, norm_obs)

            boost = self._ramp_boost_gain(ak)
            if boost > 0.0:
                f = float(np.clip(f + self.RAMP_BOOST_FWD * boost, -1.0, 1.0))
                boosted_agents += 1

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

            if getattr(self, "human_debug_keys", False):
                try:
                    if lock_evt:
                        print(f"[human_debug] lock event for {ak}: {lock_evt}")
                    if grab_evt:
                        print(f"[human_debug] grab event for {ak}: {grab_evt}")
                except Exception:
                    pass

            if ak == self.learnable_agent_key and self._inference_models.get(ak) is not None:
                f_env = float(self.INFERENCE_FORWARD_SIGN * f)
            else:
                f_env = float(f)

            applied_forward_env = f_env
            t_env = float(t)
            cv[self.actuator_ids[f"{ak}_fwd"]], cv[self.actuator_ids[f"{ak}_turn"]] = (f_env, t_env)
            self.last_debug_ctrl[ak] = (f_env, t_env)

        stats = dict(
            any_lock_event=any_lock_event,
            any_grab_event=any_grab_event,
            any_lock_pressed=any_lock_pressed,
            any_grab_pressed=any_grab_pressed,
            any_lock_target=any_lock_target,
            any_grab_target=any_grab_target,
            max_lock_btn=max_lock_btn,
            max_grab_btn=max_grab_btn,
            boosted_agents=boosted_agents,
            applied_forward_model=applied_forward_model,
            applied_forward_env=applied_forward_env,
        )
        return cv, stats

    # 環境をリセットするための関数。シードとオプションを受け取り、環境の状態を初期化します。エージェントとオブジェクトの位置をランダムに配置し、必要な初期化処理を行います。デバッグモードが有効な場合は、配置の過程で詳細なログを記録します。
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # invalidate per-step caches
        try:
            self._frame_cache_step = -1
            self._frame_obs_cache.clear()
        except Exception:
            pass
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
        # place static bodies (ramps/boxes) and agents using helpers
        placed = []
        ramp_specs = [(rid, self.R_RAMP, 0.0) for rid in self.ramp_ids]
        box_specs = [(b, self.R_BOX, 0.5) for b in self.box_ids]
        self._randomly_place_bodies(ramp_specs + box_specs, placed, margin=0.2)
        self._randomly_place_agents(placed)
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
            self._populate_prev_vis_and_being_hit()
        except Exception:
            logger.exception("failed while refreshing prev_vis/prev_being_hit caches after forward")
            raise

        # initialize previous-step visibility / being-hit caches so _get_obs
        # can use a well-defined 'previous' value immediately after reset
        self._prev_vis.clear()
        self._prev_being_hit.clear()
        try:
            self._populate_prev_vis_and_being_hit()
        except Exception:
            logger.exception("_populate_prev_vis_and_being_hit failed during reset initialization")
            pass

        self._cache_planar_object_pose()

        # 注意: ヒューマン入力の I/O (端末の raw モードや pynput リスナ) は
        # `step()` を純粋関数的に保つため環境外へ移動しています。
        # ランナー側でキー入力を収集し `env.set_human_action()` を呼んで
        # ヒューマン入力を注入してください。

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
        # 生のモデル出力（model forward）と、実際に環境へ適用した前進値（inference-only 変換後）の両方を保持する
        applied_forward_model = 0.0
        applied_forward_env = 0.0

        # 歴史的に各寄稿処理が `data.xfrc_applied` に累積してきたため。

        # process agent actions and build control vector
        cv, _stats = self._process_agent_actions(af)
        self.data.ctrl[:] = cv
        # --- capture pre-physics visibility / being-hit state (used as 'previous' freshness) ---
        try:
            self._capture_prephysics_prev_vis()
        except Exception:
            pass
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
        # ここでヒューマン固有の I/O や副作用は行わないでください。ランナーが一時的な
        # ボタンのクリア等を担当します。オブジェクト制約と安定化処理は決定論的に適用します。
        self._apply_object_constraints()
        self._stabilize_agent_vertical_motion()
        mujoco.mj_forward(self.model, self.data)
        # Refresh visibility cache to reflect post-physics positions so
        # observations computed below use current (post-step) visibility.
        try:
            self._populate_prev_vis_and_being_hit()
        except Exception:
            pass
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
        jx = self.qpos_indices[self.learnable_agent_key]["x"]
        vadr = self.model.jnt_dofadr[jx]
        qlen = self.data.qvel.shape[0]
        agent_vx = float(self.data.qvel[vadr]) if vadr < qlen else 0.0
        agent_vy = float(self.data.qvel[vadr + 1]) if (vadr + 1) < qlen else 0.0
        last_ctrl = self.last_debug_ctrl.get(self.learnable_agent_key, (0.0, 0.0))
        last_ctrl_f = float(last_ctrl[0])
        last_ctrl_t = float(last_ctrl[1])
        # 壁反発計算と適用をヘルパーに委譲
        try:
            # preserve the current applied_forward_env for the helper to read
            try:
                self._current_applied_forward_env = applied_forward_env
            except Exception:
                pass

            rep = self._compute_and_apply_wall_repulsion()

            # 必要最小限の値だけを取り出して利用する（冗長なローカル展開を避ける）
            bid = int(rep.get("bid", int(learnable_agent_body_id)))
            fx_tot = float(rep.get("fx_tot", 0.0))
            fy_tot = float(rep.get("fy_tot", 0.0))
            clearance = float(rep.get("clearance", 0.0))

            # data.xfrc_applied に反映（元の書き込み順序を再現）
            try:
                if hasattr(self.data, "xfrc_applied") and self.data.xfrc_applied.shape[1] >= 2:
                    self.data.xfrc_applied[bid, 0] = float(self.data.xfrc_applied[bid, 0]) + fx_tot
                    self.data.xfrc_applied[bid, 1] = float(self.data.xfrc_applied[bid, 1]) + fy_tot
            except Exception:
                pass

            # 回転抑制処理（Mz の加算等）を呼び出す
            try:
                self._apply_rotation_suppression(bid, clearance)
            except Exception:
                pass

            # verbose debug は rep を直接渡す
            try:
                self._debug_wall_repulsion(rep)
            except Exception:
                pass
        except Exception:
            # シミュレーションステップが中断されないように、反発ロジックのエラーを無視する
            bid = int(learnable_agent_body_id)
        finally:
            try:
                if hasattr(self, "_current_applied_forward_env"):
                    delattr(self, "_current_applied_forward_env")
            except Exception:
                try:
                    del self._current_applied_forward_env
                except Exception:
                    pass

        obs = self._normalize_obs(self._get_obs(idx_to_obs))
        reward = float(rb if self.target == "hider" else -rb)
        done = self.current_step >= self.max_episode_steps
        # ランプ関連のデバッグ指標をヘルパーで取得
        try:
            ramp_dbg = self._compute_ramp_metrics(learnable_agent_body_id, learnable_agent_pos, dbg_agent_z)
            dbg_ramp_progress = ramp_dbg.get("dbg_ramp_progress", None)
            dbg_ramp_lx = ramp_dbg.get("dbg_ramp_lx", None)
            dbg_ramp_ly = ramp_dbg.get("dbg_ramp_ly", None)
            dbg_ramp_facing = ramp_dbg.get("dbg_ramp_facing", None)
            dbg_ramp_rpos = ramp_dbg.get("dbg_ramp_rpos", None)
            dbg_ramp_climbing = ramp_dbg.get("dbg_ramp_climbing", False)
            dbg_ramp_reached_top = ramp_dbg.get("dbg_ramp_reached_top", False)
            dbg_agent_z = ramp_dbg.get("dbg_agent_z", dbg_agent_z)
        except Exception:
            dbg_ramp_progress = None
            dbg_ramp_lx = None
            dbg_ramp_ly = None
            dbg_ramp_facing = None
            dbg_ramp_rpos = None
            dbg_ramp_climbing = False
            dbg_ramp_reached_top = False
        # assemble reward and info via helper to improve readability
        reward, done, info = self._finalize_reward_and_info(
            rb=rb,
            find=find,
            stats=_stats,
            moving_box_count=moving_box_count,
            moving_ramp_count=moving_ramp_count,
            box_speeds=box_speeds,
            ramp_speeds=ramp_speeds,
            blocked_ramp_count=blocked_ramp_count,
            gaze_cos_front_max=gaze_cos_front_max,
            gaze_dist_max=gaze_dist_max,
            learnable_hider_seen=learnable_hider_seen,
            wall_dist=wall_dist,
            agent_vz=agent_vz,
            agent_vx=agent_vx,
            agent_vy=agent_vy,
            last_ctrl_f=last_ctrl_f,
            last_ctrl_t=last_ctrl_t,
            learnable_agent_body_id=learnable_agent_body_id,
            learnable_agent_pos=learnable_agent_pos,
            dbg_agent_z=dbg_agent_z,
        )
        # expose ramp boost under a neutral key (caller can decide to log it)
        try:
            info["ramp_boost"] = float(self._ramp_boost_gain(self.learnable_agent_key))
        except Exception:
            pass
        # expose ramp metrics under neutral names (for training/inference)
        if dbg_ramp_lx is not None:
            info["ramp_lx"] = dbg_ramp_lx
        if dbg_ramp_ly is not None:
            info["ramp_ly"] = dbg_ramp_ly
        if dbg_ramp_facing is not None:
            info["ramp_facing"] = dbg_ramp_facing
        if dbg_ramp_rpos is not None:
            info["ramp_rpos_x"] = float(dbg_ramp_rpos[0])
            info["ramp_rpos_y"] = float(dbg_ramp_rpos[1])

        # expose agent world z/vz under neutral names
        info["agent_world_z"] = dbg_agent_z
        info["agent_world_vz"] = dbg_agent_vz

        if dbg_ramp_progress is not None:
            info["ramp_progress"] = dbg_ramp_progress

        # climbing / reached-top diagnostics (neutral keys only)
        info["ramp_climbing"] = bool(dbg_ramp_climbing)
        info["ramp_reached_top"] = bool(dbg_ramp_reached_top)
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

        # (removed friction-override TTL restore logic)

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

    def _debug_wall_repulsion(self, rep: Optional[Dict[str, Any]]):
        """
        壁反発に関するデバッグ出力をまとめたヘルパー。
        既存のログ出力と低レベル診断をこのメソッドへ集約する。
        """
        if not (getattr(self, "debug_mode", False) and getattr(self, "WALL_DEBUG_VERBOSE", False)):
            return
        try:
            # rep に含まれる値を優先してデバッグ出力を構築する
            if rep is None:
                rep = {}
            dist = float(rep.get("dist", 0.0))
            clearance = float(rep.get("clearance", 0.0))
            applied_fwd = float(rep.get("applied_fwd", 0.0))
            fb_fx = float(rep.get("fb_fx", 0.0))
            fb_fy = float(rep.get("fb_fy", 0.0))
            ctrl_fx = float(rep.get("ctrl_fx", 0.0))
            ctrl_fy = float(rep.get("ctrl_fy", 0.0))
            fx_tot = float(rep.get("fx_tot", 0.0))
            fy_tot = float(rep.get("fy_tot", 0.0))
            bid = int(rep.get("bid", int(self.body_ids.get(self.learnable_agent_key, 0))))

            # facing_dot を再計算（エージェントの向きと法線が利用可能なら）
            try:
                nx = float(rep.get("nx", 0.0))
                ny = float(rep.get("ny", 0.0))
                rot_j = self.qpos_indices[self.learnable_agent_key]["rot"]
                rot_val = float(self.data.qpos[self.model.jnt_qposadr[rot_j]])
                afwd_x = float(math.cos(rot_val))
                afwd_y = float(math.sin(rot_val))
                facing_dot = float(np.dot([afwd_x, afwd_y], [nx, ny]))
            except Exception:
                facing_dot = float(rep.get("facing_dot", 0.0)) if rep.get("facing_dot") is not None else 0.0

            try:
                self.debug_logger.print(
                    f"[DEBUG][wall_repulsion] step={self.current_step} agent={self.learnable_agent_key} dist={dist:.3f} clearance={clearance:.3f} facing_dot={facing_dot:.3f} applied_fwd={applied_fwd:.3f} fb=({fb_fx:.1f},{fb_fy:.1f}) ctrl=({ctrl_fx:.1f},{ctrl_fy:.1f}) fx={fx_tot:.1f} fy={fy_tot:.1f}"
                )
            except Exception:
                pass

            try:
                aid = self.actuator_ids.get(f"{self.learnable_agent_key}_fwd")
                ctrl_val = float(self.data.ctrl[aid]) if (aid is not None and aid < self.data.ctrl.shape[0]) else None
                xfrc_written = tuple([float(self.data.xfrc_applied[bid, 0]), float(self.data.xfrc_applied[bid, 1])])
                ncon = int(getattr(self.data, "ncon", 0)) if hasattr(self.data, "ncon") else -1
                try:
                    vadr = self.model.jnt_dofadr[self.qpos_indices[self.learnable_agent_key]["x"]]
                    qlen = self.data.qvel.shape[0]
                    avx = float(self.data.qvel[vadr]) if vadr < qlen else 0.0
                    avy = float(self.data.qvel[vadr + 1]) if (vadr + 1) < qlen else 0.0
                except Exception:
                    avx = 0.0
                    avy = 0.0
                try:
                    self.debug_logger.print(
                        f"[DEBUG][wall_lowlevel] step={self.current_step} agent={self.learnable_agent_key} ctrl_fwd={ctrl_val} xfrc_applied={xfrc_written} vel=({avx:.3f},{avy:.3f}) ncon={ncon}"
                    )
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    def _compute_and_apply_wall_repulsion(self):
        """
        壁反発の中心ロジックをまとめて `data.xfrc_applied` に力を加え、
        必要なデバッグ値を辞書で返す。

        戻り値キー (すべて float または int):
            dist, nx, ny, clearance, applied_fwd,
            fb_fx, fb_fy, ctrl_fx, ctrl_fy, fx_tot, fy_tot, bid, ncon
        """
        # 必要な状態はインスタンスから取得する（メソッドはクラス内で呼ばれるため）
        try:
            bid = int(self.body_ids[self.learnable_agent_key])
        except Exception:
            bid = 0
        try:
            learnable_agent_pos = self.data.xpos[bid]
        except Exception:
            learnable_agent_pos = (0.0, 0.0, 0.0)
        # 初期値
        dist = 1.0
        nx = 0.0
        ny = 0.0
        clearance = 1.0
        # applied_forward_env と last_ctrl_f は last_debug_ctrl から取得する
        last_ctrl_f = float(self.last_debug_ctrl.get(self.learnable_agent_key, (0.0, 0.0))[0])
        applied_fwd = float(getattr(self, "_current_applied_forward_env", last_ctrl_f if last_ctrl_f is not None else 0.0))
        fb_fx = fb_fy = ctrl_fx = ctrl_fy = fx_tot = fy_tot = 0.0

        try:
            # SDF を使って壁距離・法線をサンプリング
            try:
                dist_w, nx_w, ny_w = self.vis_engine.sample_sdf_with_normal(float(learnable_agent_pos[0]), float(learnable_agent_pos[1]))
                dist = float(dist_w)
                nx = float(nx_w)
                ny = float(ny_w)
            except Exception:
                dist = 1.0
                nx = ny = 0.0

            # クリアランス (エージェント表面ベースの余裕距離)
            try:
                r_eff = float(getattr(self, "AGENT_EFFECTIVE_RADIUS", 0.4))
                clearance = float(dist - (r_eff - float(getattr(self, "WALL_REPULSION_CLEARANCE", 0.2))))
            except Exception:
                clearance = float(dist)

            # 制御ベースの反発推定 (EMA)
            try:
                ctrl_lp = float(getattr(self, "WALL_REPULSION_CTRL_LP", 0.2))
                scale = float(getattr(self, "WALL_REPULSION_CTRL_SCALE", 0.05))
                clip = getattr(self, "WALL_REPULSION_CTRL_CLIP", None)
                prev = float(getattr(self, "_wall_repulsion_ctrl_ema", 0.0))
                # 入力は実際に環境で適用された前進値を使う
                inp = float(getattr(self, "_current_applied_forward_env", last_ctrl_f if last_ctrl_f is not None else 0.0))
                ema = (1.0 - ctrl_lp) * prev + ctrl_lp * inp
                if clip is not None:
                    try:
                        ema = max(min(float(clip), float(ema)), -float(clip))
                    except Exception:
                        pass
                # persist for next step
                self._wall_repulsion_ctrl_ema = float(ema)
                ctrl_force = float(ema) * scale
                # forward vector (world)
                try:
                    rot_j = self.qpos_indices[self.learnable_agent_key]["rot"]
                    rot_val = float(self.data.qpos[self.model.jnt_qposadr[rot_j]])
                    afwd_x = float(math.cos(rot_val))
                    afwd_y = float(math.sin(rot_val))
                except Exception:
                    afwd_x, afwd_y = 1.0, 0.0
                ctrl_fx = ctrl_force * afwd_x
                ctrl_fy = ctrl_force * afwd_y
                # 推定ベースの反発は、エージェントが壁に向いていてかつ
                # 十分近い場合にのみ適用する（既存実装と整合するためのゲート）
                try:
                    facing_dot = float(np.dot([afwd_x, afwd_y], [nx, ny]))
                    target_dist = float(getattr(self, "WALL_REPULSION_TARGET_DIST", 0.9))
                    fb_dot_th = float(getattr(self, "WALL_REPULSION_FB_FWD_DOT", -0.2))
                    if not (facing_dot < fb_dot_th and float(dist) < target_dist):
                        ctrl_fx = 0.0
                        ctrl_fy = 0.0
                except Exception:
                    pass
            except Exception:
                ctrl_fx = ctrl_fy = 0.0

            # フィードバック（壁法線ベース）の反発
            try:
                # 距離が小さいほど強いフィードバック
                kp = float(getattr(self, "WALL_REPULSION_PASSIVE_KP", 40.0))
                fmax = float(getattr(self, "WALL_REPULSION_PASSIVE_FMAX", 30.0))
                r_start = float(getattr(self, "WALL_REPULSION_RADIUS", 0.25))
                depth = max(0.0, r_start - float(dist))
                fb_mag = kp * depth
                if fb_mag > fmax:
                    fb_mag = fmax
                fb_fx = fb_mag * float(nx)
                fb_fy = fb_mag * float(ny)
            except Exception:
                fb_fx = fb_fy = 0.0

            # 合計力（計算のみ、書き込みは呼び出し元で行う）
            try:
                fx_tot = float(fb_fx + ctrl_fx)
                fy_tot = float(fb_fy + ctrl_fy)
                # 1 ステップ内で書き込む増分をクリップ
                try:
                    clamp_v = float(getattr(self, "WALL_REPULSION_ACCUM_CLAMP_MAX", 100.0))
                    fx_tot = max(min(fx_tot, clamp_v), -clamp_v)
                    fy_tot = max(min(fy_tot, clamp_v), -clamp_v)
                except Exception:
                    pass
            except Exception:
                fx_tot = fy_tot = 0.0

        except Exception:
            pass

            # ncon (contacts) 取得
            try:
                ncon = int(getattr(self.data, "ncon", 0)) if hasattr(self.data, "ncon") else 0
            except Exception:
                ncon = 0

        return {
            "dist": float(dist),
            "nx": float(nx),
            "ny": float(ny),
            "clearance": float(clearance),
            "applied_fwd": float(applied_fwd),
            "fb_fx": float(fb_fx),
            "fb_fy": float(fb_fy),
            "ctrl_fx": float(ctrl_fx),
            "ctrl_fy": float(ctrl_fy),
            "fx_tot": float(fx_tot),
            "fy_tot": float(fy_tot),
            "bid": int(bid),
            "ncon": int(ncon),
        }

    def _assemble_step_info(
        self,
        *,
        find,
        any_lock_event,
        any_grab_event,
        any_lock_pressed,
        any_grab_pressed,
        any_lock_target,
        any_grab_target,
        max_lock_btn,
        max_grab_btn,
        moving_box_count,
        moving_ramp_count,
        box_speeds,
        ramp_speeds,
        blocked_ramp_count,
        boosted_agents,
        gaze_cos_front_max,
        gaze_dist_max,
        learnable_hider_seen,
        wall_dist,
        agent_vz,
        agent_vx,
        agent_vy,
        last_ctrl_f,
        last_ctrl_t,
        applied_forward_model,
        applied_forward_env,
    ):
        """
        step() 用の `info` 辞書を組み立てて返すヘルパー。
        値の正規化や安全な既定値の適用をここで行う。
        """
        return {
            "is_detected": find,
            "lock_event": any_lock_event,
            "grab_event": any_grab_event,
            "lock_pressed": any_lock_pressed,
            "grab_pressed": any_grab_pressed,
            "lock_target": any_lock_target,
            "grab_target": any_grab_target,
            "lock_btn_max": max_lock_btn,
            "grab_btn_max": max_grab_btn,
            "box_moving_count": moving_box_count,
            "ramp_moving_count": moving_ramp_count,
            "max_box_speed": float(max(box_speeds) if box_speeds else 0.0),
            "max_ramp_speed": float(max(ramp_speeds) if ramp_speeds else 0.0),
            "blocked_ramp_count": blocked_ramp_count,
            "boosted_agents": boosted_agents,
            "override_learnable_policy": bool(self.override_learnable_policy),
            "model_policy_deterministic": bool(self.model_policy_deterministic),
            "seek_gaze_cos_front_max": float(gaze_cos_front_max),
            "seek_gaze_cos_front_dist_max": float(gaze_dist_max),
            "learnable_hider_seen": bool(learnable_hider_seen),
            "wall_distance": wall_dist,
            "agent_vz": agent_vz,
            "agent_vx": agent_vx,
            "agent_vy": agent_vy,
            # whether current step is within prep/warmup
            "in_prep": bool(self.current_step <= self.prep_steps),
            # 'applied_forward_model' is the raw model output; 'applied_forward' is what was
            # actually applied to the actuator (may be transformed at runtime for compatibility)
            "applied_forward_model": applied_forward_model,
            "applied_forward": applied_forward_env,
        }

    def _finalize_reward_and_info(
        self,
        *,
        rb,
        find,
        stats: Optional[Dict[str, Any]] = None,
        moving_box_count,
        moving_ramp_count,
        box_speeds,
        ramp_speeds,
        blocked_ramp_count,
        gaze_cos_front_max,
        gaze_dist_max,
        learnable_hider_seen,
        wall_dist,
        agent_vz,
        agent_vx,
        agent_vy,
        last_ctrl_f,
        last_ctrl_t,
        learnable_agent_body_id,
        learnable_agent_pos,
        dbg_agent_z,
    ):
        """Compute final reward, done flag, and `info` dict for a step.

        This centralizes ramp-debug computation and `info` assembly so `step()`
        remains compact.
        """
        # unpack stats dict for local use (keeps step() simpler)
        try:
            s = stats or {}
            any_lock_event = bool(s.get("any_lock_event", False))
            any_grab_event = bool(s.get("any_grab_event", False))
            any_lock_pressed = bool(s.get("any_lock_pressed", False))
            any_grab_pressed = bool(s.get("any_grab_pressed", False))
            any_lock_target = bool(s.get("any_lock_target", False))
            any_grab_target = bool(s.get("any_grab_target", False))
            max_lock_btn = float(s.get("max_lock_btn", 0.0))
            max_grab_btn = float(s.get("max_grab_btn", 0.0))
            boosted_agents = int(s.get("boosted_agents", 0))
            applied_forward_model = float(s.get("applied_forward_model", 0.0))
            applied_forward_env = float(s.get("applied_forward_env", 0.0))
        except Exception:
            any_lock_event = False
            any_grab_event = False
            any_lock_pressed = False
            any_grab_pressed = False
            any_lock_target = False
            any_grab_target = False
            max_lock_btn = 0.0
            max_grab_btn = 0.0
            boosted_agents = 0
            applied_forward_model = 0.0
            applied_forward_env = 0.0

        # reward and done
        reward = float(rb if self.target == "hider" else -rb)
        done = self.current_step >= self.max_episode_steps

        # ramp/debug metrics
        try:
            ramp_dbg = self._compute_ramp_metrics(learnable_agent_body_id, learnable_agent_pos, dbg_agent_z)
            dbg_ramp_progress = ramp_dbg.get("dbg_ramp_progress", None)
            dbg_ramp_lx = ramp_dbg.get("dbg_ramp_lx", None)
            dbg_ramp_ly = ramp_dbg.get("dbg_ramp_ly", None)
            dbg_ramp_facing = ramp_dbg.get("dbg_ramp_facing", None)
            dbg_ramp_rpos = ramp_dbg.get("dbg_ramp_rpos", None)
            dbg_ramp_climbing = ramp_dbg.get("dbg_ramp_climbing", False)
            dbg_ramp_reached_top = ramp_dbg.get("dbg_ramp_reached_top", False)
            dbg_agent_z = ramp_dbg.get("dbg_agent_z", dbg_agent_z)
        except Exception:
            dbg_ramp_progress = None
            dbg_ramp_lx = None
            dbg_ramp_ly = None
            dbg_ramp_facing = None
            dbg_ramp_rpos = None
            dbg_ramp_climbing = False
            dbg_ramp_reached_top = False

        # assemble info
        info = self._assemble_step_info(
            find=find,
            any_lock_event=any_lock_event,
            any_grab_event=any_grab_event,
            any_lock_pressed=any_lock_pressed,
            any_grab_pressed=any_grab_pressed,
            any_lock_target=any_lock_target,
            any_grab_target=any_grab_target,
            max_lock_btn=max_lock_btn,
            max_grab_btn=max_grab_btn,
            moving_box_count=moving_box_count,
            moving_ramp_count=moving_ramp_count,
            box_speeds=box_speeds,
            ramp_speeds=ramp_speeds,
            blocked_ramp_count=blocked_ramp_count,
            boosted_agents=boosted_agents,
            gaze_cos_front_max=gaze_cos_front_max,
            gaze_dist_max=gaze_dist_max,
            learnable_hider_seen=learnable_hider_seen,
            wall_dist=wall_dist,
            agent_vz=agent_vz,
            agent_vx=agent_vx,
            agent_vy=agent_vy,
            last_ctrl_f=last_ctrl_f,
            last_ctrl_t=last_ctrl_t,
            applied_forward_model=applied_forward_model,
            applied_forward_env=applied_forward_env,
        )

        # add ramp boost and agent z/vz fields (neutral keys only)
        try:
            info["ramp_boost"] = float(self._ramp_boost_gain(self.learnable_agent_key))
        except Exception:
            pass

        if dbg_ramp_lx is not None:
            info["ramp_lx"] = dbg_ramp_lx
        if dbg_ramp_ly is not None:
            info["ramp_ly"] = dbg_ramp_ly
        if dbg_ramp_facing is not None:
            info["ramp_facing"] = dbg_ramp_facing
        if dbg_ramp_rpos is not None:
            info["ramp_rpos_x"] = float(dbg_ramp_rpos[0])
            info["ramp_rpos_y"] = float(dbg_ramp_rpos[1])
        info["agent_world_z"] = dbg_agent_z
        info["agent_world_vz"] = agent_vz
        if dbg_ramp_progress is not None:
            info["ramp_progress"] = dbg_ramp_progress
        info["ramp_climbing"] = bool(dbg_ramp_climbing)
        info["ramp_reached_top"] = bool(dbg_ramp_reached_top)

        # collect stats and return
        try:
            self._dbg_collect_stats(reward, info)
        except Exception:
            pass

        return reward, done, info

    def _apply_rotation_suppression(self, bid, clearance):
        """
        接触時に発生しがちな「壁方向へ回り込む」回転を抑制するための処理をまとめる。
        - ターン入力の減衰
        - 回転角速度に応じた反トルクを Mz に追加
        """
        try:
            ncon = int(getattr(self.data, "ncon", 0)) if hasattr(self.data, "ncon") else 0
            rot_atten_th = float(getattr(self, "WALL_REPULSION_ATTENUATE_CLEARANCE", 0.25))
            if ncon > 0 and float(clearance) < rot_atten_th:
                # ターン入力を減衰
                try:
                    turn_aid = self.actuator_ids.get(f"{self.learnable_agent_key}_turn")
                    if turn_aid is not None:
                        self.data.ctrl[turn_aid] = float(self.data.ctrl[turn_aid]) * float(getattr(self, "WALL_CONTACT_TURN_ATTENUATE_FACTOR", 0.2))
                except Exception:
                    pass
                # 回転速度に比例した反トルクを与える（Mz に追加）
                try:
                    rot_jid = self.qpos_indices[self.learnable_agent_key]["rot"]
                    rot_vadr = int(self.model.jnt_dofadr[rot_jid])
                    qlen = self.data.qvel.shape[0]
                    ang_vel_z = float(self.data.qvel[rot_vadr]) if rot_vadr < qlen else 0.0
                    k = float(getattr(self, "WALL_CONTACT_ROT_DAMP_K", 30.0))
                    mz = -k * ang_vel_z
                    # クリップして過大値を防ぐ
                    try:
                        mz_clip = float(getattr(self, "WALL_CONTACT_MZ_CLIP", 50.0))
                        if mz > mz_clip:
                            mz = mz_clip
                        elif mz < -mz_clip:
                            mz = -mz_clip
                    except Exception:
                        pass
                    # data.xfrc_applied の Mz (index 5) に加算
                    try:
                        if self.data.xfrc_applied.shape[1] > 5:
                            self.data.xfrc_applied[bid, 5] = float(self.data.xfrc_applied[bid, 5]) + mz
                    except Exception:
                        pass
                    if self.debug_mode and getattr(self, "WALL_DEBUG_VERBOSE", False):
                        try:
                            self.debug_logger.print(f"[DEBUG][wall_rot] step={self.current_step} agent={self.learnable_agent_key} ncon={ncon} ang_vel_z={ang_vel_z:.3f} mz={mz:.3f}")
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass

    def _dump_contacts(self, max_dump=32):
        """
        接触情報の詳細ダンプ（デバッグ専用）。contact 配列を走査して
        簡易的なテキストダンプと、該当ジオムの定義をログ出力する。
        """
        try:
            ncon = int(getattr(self.data, "ncon", 0)) if hasattr(self.data, "ncon") else 0
            if not (self.debug_mode and ncon > 0 and getattr(self, "WALL_DEBUG_VERBOSE", False)):
                return
            detail_lines = []
            max_dump = min(int(ncon), int(max_dump))
            for ci in range(max_dump):
                try:
                    c = self.data.contact[ci]
                except Exception:
                    break
                try:
                    g1 = getattr(c, "geom1", None)
                except Exception:
                    g1 = None
                try:
                    g2 = getattr(c, "geom2", None)
                except Exception:
                    g2 = None
                try:
                    g1_name = None if g1 is None else self.model.geom(g1).name
                except Exception:
                    try:
                        g1_name = None if g1 is None else self.model.geom(g1).id
                    except Exception:
                        g1_name = None
                try:
                    g2_name = None if g2 is None else self.model.geom(g2).name
                except Exception:
                    try:
                        g2_name = None if g2 is None else self.model.geom(g2).id
                    except Exception:
                        g2_name = None
                try:
                    dist_c = float(getattr(c, "dist", float("nan")))
                except Exception:
                    dist_c = None
                try:
                    pos_attr = getattr(c, "pos", None)
                    pos_c = tuple(float(x) for x in pos_attr) if pos_attr is not None else None
                except Exception:
                    pos_c = None
                try:
                    frame_attr = getattr(c, "frame", None)
                    frame0 = tuple(float(x) for x in frame_attr[:3]) if frame_attr is not None else None
                except Exception:
                    frame0 = None
                detail_lines.append(f"ci={ci} g1={g1}({g1_name}) g2={g2}({g2_name}) dist={dist_c} pos={pos_c} frame0={frame0}")
            if detail_lines:
                try:
                    self.debug_logger.print(f"[DEBUG][contacts] step={self.current_step} agent={self.learnable_agent_key} ncon={ncon} " + " | ".join(detail_lines))
                except Exception:
                    print(f"[DEBUG][contacts] agent={self.learnable_agent_key} ncon={ncon}")
            # 出現したジオムの定義をダンプして、どのパーツが接触しているか確認する
            try:
                geom_ids = set()
                for dl in detail_lines:
                    try:
                        parts = dl.split()
                        for p in parts:
                            if p.startswith("g1=") or p.startswith("g2="):
                                gid_part = p.split("=")[1]
                                if "(" in gid_part:
                                    gid = int(gid_part.split("(")[0])
                                else:
                                    gid = int(gid_part)
                                geom_ids.add(gid)
                    except Exception:
                        continue
                for gid in sorted(list(geom_ids)):
                    try:
                        g = self.model.geom(gid)
                        g_name = getattr(g, "name", None)
                        g_type = getattr(g, "type", None)
                        g_size = None
                        try:
                            g_size = tuple(float(x) for x in getattr(g, "size"))
                        except Exception:
                            g_size = None
                        g_pos = None
                        try:
                            g_pos = tuple(float(x) for x in getattr(g, "pos"))
                        except Exception:
                            g_pos = None
                        g_fromto = None
                        try:
                            g_fromto = tuple(float(x) for x in getattr(g, "fromto"))
                        except Exception:
                            g_fromto = None
                        self.debug_logger.print(f"[DEBUG][geom_info] step={self.current_step} gid={gid} name={g_name} type={g_type} size={g_size} pos={g_pos} fromto={g_fromto}")
                    except Exception:
                        continue
            except Exception:
                pass
        except Exception:
            pass

    def _compute_ramp_metrics(self, learnable_agent_body_id, learnable_agent_pos, dbg_agent_z):
        """
        Compute ramp-related metrics and return them as a dict.
        This function was previously named `_compute_ramp_debug`. It now
        returns the same diagnostic values but under a neutral helper name
        because several metrics are used during training/evaluation.
        """
        dbg_ramp_progress = None
        dbg_ramp_lx = None
        dbg_ramp_ly = None
        dbg_ramp_facing = None
        dbg_ramp_rpos = None
        dbg_ramp_rid = None
        dbg_ramp_climbing = False
        dbg_ramp_reached_top = False
        try:
            boost = float(self._ramp_boost_gain(self.learnable_agent_key))
            if boost > 0.0:
                agent_body_id = learnable_agent_body_id
                agent_z = float(self.data.xpos[agent_body_id][2])
                apos2 = learnable_agent_pos[:2]
                arot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[self.learnable_agent_key]["rot"]]]
                for _i, rid in enumerate(self.ramp_ids, start=1):
                    try:
                        rpos = self.data.xpos[rid][:2]
                        up = self._ramp_uphill_dir(rid)
                        side = np.array([-up[1], up[0]], dtype=np.float32)
                        rel = apos2 - rpos
                        lx = float(np.dot(rel, up))
                        ly = float(np.dot(rel, side))
                        if abs(ly) > 1.5:
                            continue
                        try:
                            lx_min, lx_max, _ = self._compute_ramp_lx_bounds(rid)
                        except Exception:
                            lx_min, lx_max = -1.15, 0.666
                        prog = (lx - lx_min) / max(1e-6, (lx_max - lx_min))
                        prog = max(0.0, min(1.0, prog))
                        if dbg_ramp_progress is None or prog > dbg_ramp_progress:
                            dbg_ramp_progress = float(prog)
                            dbg_agent_z = float(agent_z)
                            dbg_ramp_lx = float(lx)
                            dbg_ramp_ly = float(ly)
                            afwd = np.array([math.cos(arot), math.sin(arot)], dtype=np.float32)
                            dbg_ramp_facing = float(np.dot(afwd, up))
                            dbg_ramp_rpos = rpos.copy()
                            dbg_ramp_rid = rid
                    except Exception:
                        continue
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
                            details.append(f"{rkey2}(mode={st2.get('mode')},owner={st2.get('owner')},lx={lx2:.3f},ly={ly2:.3f},f={facing2:.3f},rpos=({rpos2[0]:.3f},{rpos2[1]:.3f}))")
                        s = ", ".join(details)
                        if getattr(self, "debug_mode", False):
                            print(f"[RAMP_DBG_REJECT] step={getattr(self,'current_step',-1)} ak={self.learnable_agent_key} " f"boost={boost:.3f} candidates=[{s}]")
                    except Exception:
                        pass

            prev_z = self._prev_agent_z.get(self.learnable_agent_key, None)
            if dbg_ramp_progress is not None:
                if prev_z is not None and (agent_z - prev_z) > 0.01 and dbg_ramp_progress > 0.05:
                    dbg_ramp_climbing = True
                try:
                    HEIGHT_TOL = 0.02
                    if dbg_ramp_rid is not None:
                        top_z = self._compute_ramp_top_z(dbg_ramp_rid)
                        dbg_ramp_reached_top = agent_z >= (top_z - HEIGHT_TOL)
                    else:
                        dbg_ramp_reached_top = False
                except Exception:
                    dbg_ramp_reached_top = False
            try:
                self._prev_agent_z[self.learnable_agent_key] = float(agent_z)
            except Exception:
                pass
        except Exception:
            dbg_ramp_progress = None
            dbg_ramp_climbing = False
            dbg_ramp_reached_top = False
        try:
            if dbg_ramp_progress is not None and dbg_agent_z is None:
                agent_body_id = learnable_agent_body_id
                dbg_agent_z = float(self.data.xpos[agent_body_id][2])
        except Exception:
            dbg_agent_z = None

        return {
            "dbg_ramp_progress": dbg_ramp_progress,
            "dbg_ramp_lx": dbg_ramp_lx,
            "dbg_ramp_ly": dbg_ramp_ly,
            "dbg_ramp_facing": dbg_ramp_facing,
            "dbg_ramp_rpos": dbg_ramp_rpos,
            "dbg_ramp_rid": dbg_ramp_rid,
            "dbg_ramp_climbing": dbg_ramp_climbing,
            "dbg_ramp_reached_top": dbg_ramp_reached_top,
            "dbg_agent_z": dbg_agent_z,
        }

    def _is_vis(self, pos, rot, t_pos, my_id, t_id):
        # Use per-frame cache to avoid repeated expensive visibility checks
        try:
            if int(self._frame_cache_step) != int(self.current_step):
                # invalidate cache for new step
                try:
                    self._frame_vis_cache.clear()
                except Exception:
                    self._frame_vis_cache = {}
                self._frame_cache_step = int(self.current_step)
        except Exception:
            pass

        key = (int(my_id) if my_id is not None else -1, int(t_id) if t_id is not None else -1)
        try:
            if key in self._frame_vis_cache:
                try:
                    if getattr(self, "debug_mode", False) or getattr(self, "VIS_DEBUG", False) or getattr(getattr(self, "debug_logger", None), "enabled", False):
                        print(f"[IS_VIS DEBUG] cache hit: key={key} val={self._frame_vis_cache[key]}")
                except Exception:
                    pass
                return bool(self._frame_vis_cache[key])
        except Exception:
            pass

        # Restore a cheap, semantic prefilter based on distance and facing.
        # Historically this prevented drawing/marking targets that are
        # clearly outside the seeker's forward cone. The repo's tests
        # and legacy code used a cosine threshold ~= 0.38. Apply the
        # same logic here before delegating to the (more expensive)
        # VisibilityEngine.is_visible. Keep this guarded so any error
        # falls back to the engine path.
        try:
            try:
                cs = int(getattr(self, "current_step", 0))
            except Exception:
                cs = 0
            if cs == 0:
                try:
                    print(
                        f"[RENDER DEBUG] _is_vis input(step=0): pos={list(pos) if hasattr(pos, '__iter__') else pos} t_pos={list(t_pos) if hasattr(t_pos, '__iter__') else t_pos} my_id={int(my_id) if my_id is not None else None} t_id={int(t_id) if t_id is not None else None} key={key}"
                    )
                except Exception:
                    pass

            # cheap prefilter: if target is far outside lidar range or behind
            # the seeker (dot < 0.38) then it's not considered visible.
            try:
                dx = float(t_pos[0]) - float(pos[0])
                dy = float(t_pos[1]) - float(pos[1])
                dist = math.hypot(dx, dy)
                dot = None
                if dist > 1e-8:
                    dot = math.cos(rot) * (dx / dist) + math.sin(rot) * (dy / dist)
                try:
                    if getattr(self, "debug_mode", False) or getattr(self, "VIS_DEBUG", False) or getattr(getattr(self, "debug_logger", None), "enabled", False):
                        print(f"[IS_VIS DEBUG] cs={cs} key={key} dist={dist:.4f} dot={dot}")
                except Exception:
                    pass
                if dist > L_SCALE:
                    vis = False
                    try:
                        self._frame_vis_cache[key] = bool(vis)
                    except Exception:
                        pass
                    try:
                        if getattr(self, "debug_mode", False) or getattr(self, "VIS_DEBUG", False) or getattr(getattr(self, "debug_logger", None), "enabled", False):
                            print(f"[IS_VIS DEBUG] prefilter: dist>{L_SCALE} -> vis=False key={key}")
                    except Exception:
                        pass
                    return vis
                if dist > 1e-8 and dot is not None:
                    if dot < float(getattr(self, "VIS_FACING_THRESHOLD", 0.38)):
                        vis = False
                        try:
                            self._frame_vis_cache[key] = bool(vis)
                        except Exception:
                            pass
                        try:
                            if getattr(self, "debug_mode", False) or getattr(self, "VIS_DEBUG", False) or getattr(getattr(self, "debug_logger", None), "enabled", False):
                                thr = float(getattr(self, "VIS_FACING_THRESHOLD", 0.38))
                                print(f"[IS_VIS DEBUG] prefilter: dot<{thr} -> vis=False key={key} dot={dot:.4f}")
                        except Exception:
                            pass
                        return vis
            except Exception:
                # if anything goes wrong in the cheap test, fall back to
                # calling the full visibility engine below
                pass

            vis = bool(self.vis_engine.is_visible(pos, t_pos, body_exclude=my_id, target_body_id=t_id))
            try:
                if getattr(self, "debug_mode", False) or getattr(self, "VIS_DEBUG", False) or getattr(getattr(self, "debug_logger", None), "enabled", False):
                    print(f"[IS_VIS DEBUG] vis_engine result for key={key}: {vis}")
            except Exception:
                pass
        except Exception:
            vis = False
        try:
            if cs == 0:
                try:
                    print(f"[RENDER DEBUG] _is_vis output(step=0): vis={bool(vis)} key={key}")
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._frame_vis_cache[key] = bool(vis)
        except Exception:
            pass
        return vis

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
        # Use per-frame cache to avoid recomputing the same observation multiple
        # times within a single environment step.
        try:
            if int(self._frame_cache_step) == int(self.current_step):
                cached = self._frame_obs_cache.get(int(idx), None)
                if cached is not None:
                    return cached.copy()
        except Exception:
            pass

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
            # For correctness, always use the visibility engine here rather
            # than relying on cached pre/post values which have caused
            # incorrect "through-wall" results. This is slightly slower
            # but ensures semantic parity with the engine.
            try:
                p1 = np.array([ps[0], ps[1], 0.4])
                p2 = np.array([e_pos[0], e_pos[1], 0.4])
                visible = bool(self.vis_engine.is_visible(p1, p2, body_exclude=self.body_ids[ak], target_body_id=eid))
            except Exception:
                visible = False

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
                        try:
                            p1_b = np.array([e_pos[0], e_pos[1], 0.4])
                            p2_b = np.array([ps[0], ps[1], 0.4])
                            if bool(self.vis_engine.is_visible(p1_b, p2_b, body_exclude=eid, target_body_id=self.body_ids[ak])):
                                being_hit_flag = 1.0
                        except Exception:
                            pass

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
        # store into per-frame cache
        try:
            self._frame_cache_step = int(self.current_step)
            self._frame_obs_cache[int(idx)] = o.copy()
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
                # macOSでは`launch_passive`はmjpython上で実行する必要があるため、
                # 繰り返し例外を投げないようこのプロセスではレンダリングを無効化する。
                logger.warning("mujoco.viewer.launch_passive failed: %s; disabling rendering for this process. Run under mjpython on macOS to enable passive viewer.", e)
                self._render_disabled = True
                return
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0, 0, 0.8]
                self.viewer.cam.distance = 18.0
                self.viewer.cam.elevation = -35.0
                self.viewer.cam.azimuth = 90.0
        with self.viewer.lock():
            step = int(getattr(self, "current_step", 0) or 0)
            # Only emit debug prints once per simulation step (viewer may render multiple frames)
            try:
                # Only enable render debug logs when debug_mode/debug_logger.enabled/VIS_DEBUG
                allowed = bool(getattr(self, "debug_mode", False) or getattr(getattr(self, "debug_logger", None), "enabled", False) or getattr(self, "VIS_DEBUG", False))
                do_log = (getattr(self, "_last_render_logged_step", None) != step) and allowed
            except Exception:
                do_log = False
            # Static detection removed — agents in this script are stationary.
            static = False
            try:
                if do_log and step == 0:
                    # Only print initial agents_pose once per run to avoid viewer frame flooding
                    try:
                        if not getattr(self, "_printed_step0", False):
                            poses = {}
                            for ak in self.agent_keys:
                                sid = self.body_ids[ak]
                                apos = self.data.xpos[sid][:2]
                                arot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]]
                                poses[ak] = [float(apos[0]), float(apos[1]), float(arot)]
                            print(f"[RENDER DEBUG] agents_pose: {poses} seed={getattr(self, '_init_seed', None)} step={step}")
                            try:
                                self._printed_step0 = True
                            except Exception:
                                pass
                    except Exception:
                        pass
                elif do_log and step > 0 and step % 20 == 0:
                    print(f"[RENDER DEBUG] agent_keys order: {self.agent_keys} seed={getattr(self, '_init_seed', None)} step={step}")
            except Exception:
                pass
            self.viewer.user_scn.ngeom = 0
            # render-time cache for direct engine visibility checks
            render_vis_cache = {}
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
                        try:
                            print(f"[RENDER DEBUG] turn-line draw: {ak} color={color} thickness={8.0 + abs(t_val) * 25}")
                        except Exception:
                            pass
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
                    # Use direct visibility engine for rendering consistency
                    try:
                        p1 = np.array([float(pos[0]), float(pos[1]), 0.4])
                        p2 = np.array([float(self.data.xpos[tid][0]), float(self.data.xpos[tid][1]), 0.4])
                        vkey = (int(sid), int(tid))
                        if vkey in render_vis_cache:
                            vis_direct = render_vis_cache[vkey]
                        else:
                            try:
                                vis_direct = bool(self.vis_engine.is_visible(p1, p2, body_exclude=sid, target_body_id=tid))
                            except Exception:
                                vis_direct = False
                            render_vis_cache[vkey] = vis_direct
                    except Exception:
                        vis_direct = False
                    if not vis_direct:
                        continue

                    # decide color
                    if ak.startswith("s") and k.startswith("h"):
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
                        try:
                            if do_log:
                                try:
                                    p1 = np.array([float(pos[0]), float(pos[1]), float(pos[2])])
                                    p2 = np.array([float(self.data.xpos[tid][0]), float(self.data.xpos[tid][1]), float(self.data.xpos[tid][2])])
                                    try:
                                        vis_direct = bool(self.vis_engine.is_visible(p1, p2, body_exclude=sid, target_body_id=tid))
                                    except Exception:
                                        vis_direct = False
                                    print(
                                        f"[RENDER DEBUG] gaze-line draw: {ak} -> {k} color={color} src_pos={list(p1)} rot={rot:.3f} tgt_pos={list(p2)} _is_vis=True vis_engine_direct={vis_direct} seed={getattr(self, '_init_seed', None)} step={step}"
                                    )
                                except Exception:
                                    print(
                                        f"[RENDER DEBUG] gaze-line draw: {ak} -> {k} color={color} src_pos={pos[:2]} rot={rot:.3f} tgt_pos={self.data.xpos[tid][:2]} seed={getattr(self, '_init_seed', None)} step={step}"
                                    )
                        except Exception:
                            pass
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, pos, self.data.xpos[tid])
                        g.rgba[:] = color
                        self.viewer.user_scn.ngeom += 1
                # ランタイム再着色: もし任意の Seeker が Hider を視認していれば、その Hider のジオムを黄色に再着色する
            try:
                # 各 hider キーごとの上書き設定を決定する
                overrides = {}
                for s in [k for k in self.agent_keys if k.startswith("s")]:
                    s_sid = self.body_ids[s]
                    s_pos = self.data.xpos[s_sid][:2]
                    s_rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[s]["rot"]]]
                    for h in [k for k in self.agent_keys if k.startswith("h")]:
                        h_tid = self.body_ids[h]
                        try:
                            vis = bool(self._is_vis(s_pos, s_rot, self.data.xpos[h_tid][:2], s_sid, h_tid))
                        except Exception:
                            vis = False
                        # Debug: 可視判定の入力/出力を出力（2Dベース）
                        try:
                            # Print override-check once per simulation step when logging is enabled.
                            if do_log:
                                try:
                                    t_rot = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[h]["rot"]]])
                                except Exception:
                                    t_rot = None
                                print(
                                    f"[RENDER DEBUG] override-check: seeker={s} s_pos={list(s_pos)} s_rot={s_rot:.3f} hider={h} h_pos={list(self.data.xpos[h_tid][:2])} h_rot={t_rot} vis={vis} seed={getattr(self, '_init_seed', None)} step={step}"
                                )
                        except Exception:
                            pass
                        if not vis:
                            continue
                        # Additionally compute direct visibility via vis_engine with explicit 3D points
                        try:
                            p1 = np.array([float(s_pos[0]), float(s_pos[1]), 0.4])
                            p2 = np.array([float(self.data.xpos[h_tid][0]), float(self.data.xpos[h_tid][1]), 0.4])
                            try:
                                vis_direct = bool(self.vis_engine.is_visible(p1, p2, body_exclude=s_sid, target_body_id=h_tid))
                            except Exception:
                                vis_direct = False
                            try:
                                if do_log:
                                    try:
                                        t_rot = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[h]["rot"]]])
                                    except Exception:
                                        t_rot = None
                                    print(
                                        f"[RENDER DEBUG] vis_compare: seeker={s} hider={h} _is_vis={vis} vis_engine_direct={vis_direct} p1={list(p1)} p2={list(p2)} s_rot={s_rot:.3f} h_rot={t_rot} body_exclude={s_sid} target_body_id={h_tid} step={step}"
                                    )
                            except Exception:
                                pass
                        except Exception:
                            pass
                        # hider のインデックスに応じて輝度を算出
                        try:
                            h_idx = self.hider_keys.index(h)
                            brightness = 1.0 - 0.4 * h_idx
                            if brightness < 0.2:
                                brightness = 0.2
                        except Exception:
                            brightness = 0.8
                        overrides[h] = [brightness, brightness, 0.0, 1.0]

                # Debug: 出力してどのhiderが上書き対象になったかを確認する
                try:
                    if do_log:
                        if overrides:
                            print(f"[RENDER DEBUG] overrides computed: {overrides} seed={getattr(self, '_init_seed', None)} step={step}")
                        else:
                            print(f"[RENDER DEBUG] no overrides computed seed={getattr(self, '_init_seed', None)} step={step}")
                except Exception:
                    pass

                # 上書きがあれば適用し、なければデフォルト色を復元する
                for ak in self.agent_keys:
                    geom_ids = self.agent_geom_ids.get(ak, [])
                    if ak in overrides:
                        col = overrides[ak]
                        try:
                            if do_log:
                                try:
                                    ak_rot = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]])
                                except Exception:
                                    ak_rot = None
                                print(f"[RENDER DEBUG] applying override for {ak}: {col} agent_rot={ak_rot} seed={getattr(self, '_init_seed', None)} step={step}")
                        except Exception:
                            pass
                        for gid in geom_ids:
                            try:
                                try:
                                    geom_pos = list(self.data.geom_xpos[gid])
                                except Exception:
                                    geom_pos = None
                                if do_log:
                                    try:
                                        ak_rot = float(self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]["rot"]]])
                                    except Exception:
                                        ak_rot = None
                                    print(f"[RENDER DEBUG] override-geom: agent={ak} gid={gid} geom_pos={geom_pos} agent_rot={ak_rot} applying_col={col} step={step}")
                                self.model.geom_rgba[gid][:] = col
                            except Exception:
                                pass
                    else:
                        # デフォルト色を復元
                        cols = self.agent_default_rgba.get(ak, [])
                        for gid, defcol in zip(geom_ids, cols, strict=True):
                            try:
                                self.model.geom_rgba[gid][:] = defcol
                            except Exception:
                                pass
            except Exception:
                # ビューアやモデルの問題に対して回復力を持たせる
                pass

        self.viewer.sync()
        # mark we've logged for this simulation step so repeated frame renders don't reprint
        try:
            if do_log:
                self._last_render_logged_step = step
        except Exception:
            pass

    def close(self):
        if self.viewer:
            self.viewer.close()
