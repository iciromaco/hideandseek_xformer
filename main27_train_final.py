import argparse
import math
import os
import sys
import time
import traceback
from functools import partial

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numba import njit, prange

from src.core.constants import L_SCALE, P_SCALE

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # pip install toml

try:
    import wandb
except Exception as exc:
    print(f"Exception in wandb import: {exc}")
    wandb = None

# sys.path拡張（src配下の独自モジュール用）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from agents.scripted_agents import RuleBasedHider, RuleBasedSeeker
from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2

# ============================================================================
# ポリシーモード定数
# ============================================================================
POLICY_MODEL_IF_AVAILABLE = "model_if_available"
POLICY_RULE = "rule"

CONFIG_PATH = "configs/hparams_main27.toml"

WORKER_SHUTDOWN_ERRORS = (EOFError, BrokenPipeError, ConnectionResetError)


# ============================================================================
# CLI/設定読み込み関数
# ============================================================================


def _cli_profile_override_from_argv(argv=None):
    """CLIから --profile 引数を解析"""
    args = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(args):
        if token.startswith("--profile="):
            return token.split("=", 1)[1]
        if token in {"--profile", "-p"}:
            if i + 1 < len(args):
                return args[i + 1]
    return None


def _cli_compare_profiles_from_argv(argv=None):
    """CLIから --compare-profiles 引数を解析"""
    args = list(sys.argv[1:] if argv is None else argv)
    raw = None
    for i, token in enumerate(args):
        if token.startswith("--compare-profiles="):
            raw = token.split("=", 1)[1]
            break
        if token == "--compare-profiles" and i + 1 < len(args):
            raw = args[i + 1]
            break
    if not raw:
        return []
    return [p.strip().lower() for p in str(raw).split(",") if p.strip()]


def _get_available_runtime_profiles(path):
    """TOMLから利用可能なプロファイルを取得"""
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        runtime_cfg = raw.get("runtime", {}) if isinstance(raw, dict) else {}
        names = {"train", "debug"}
        if isinstance(runtime_cfg, dict):
            for key, value in runtime_cfg.items():
                if key in {"active_profile", "common"}:
                    continue
                if isinstance(value, dict):
                    names.add(str(key).strip().lower())
        return sorted(names)
    except Exception as exc:
        print(f"Exception in _get_available_runtime_profiles: {exc}")
        return ["train", "debug"]


def _normalize_profile_name(value, available_profiles):
    """プロファイル名を正規化"""
    text = str(value).strip().lower()
    if text in available_profiles:
        return text
    print(f"Invalid runtime.active_profile={value}. " f"Fallback to train. available={available_profiles}")
    return "train"


def _load_runtime_config(path, profile_override=None, available_profiles=None):
    """ランタイム設定をロード"""
    if available_profiles is None:
        available_profiles = _get_available_runtime_profiles(path)

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        runtime_cfg = raw.get("runtime", {}) if isinstance(raw, dict) else {}
        if not isinstance(runtime_cfg, dict):
            runtime_cfg = {}
        profile_value = profile_override if profile_override is not None else runtime_cfg.get("active_profile", "train")
        profile_origin = "cli(--profile)" if profile_override is not None else "toml(runtime.active_profile)"
        profile_lower = _normalize_profile_name(profile_value, available_profiles)
        common = runtime_cfg.get("common", {}) if isinstance(runtime_cfg, dict) else {}
        # runtime_cfg のキーは TOML の記述通りの大文字小文字を保持している可能性があるため、
        # 小文字化した profile 名から実際のキー名を検索して一致させる。
        profile_key = None
        if isinstance(runtime_cfg, dict):
            for k in runtime_cfg.keys():
                if not isinstance(k, str):
                    continue
                if k.lower() == profile_lower:
                    profile_key = k
                    break
        if profile_key is None:
            profile_key = profile_lower
        profile_cfg = runtime_cfg.get(profile_key, {}) if isinstance(runtime_cfg, dict) else {}
        # 戻り値としては実際に選ばれたプロファイルキー（大文字小文字を含む実際のキー）を返す。
        profile = profile_key
        if not isinstance(common, dict):
            common = {}
        if not isinstance(profile_cfg, dict):
            profile_cfg = {}
        merged = dict(common)
        merged.update(profile_cfg)
        return merged, profile, profile_origin
    except Exception as exc:
        raise SystemExit(f"runtime config load failed: {path} ({exc}). " "Fix the TOML and retry.") from exc


def _parse_cli_args(argv=None):
    """CLIの引数をパース"""
    cli_examples = (
        "実行例:\n"
        "  1) 学習を実行する（TOMLで train を選択）\n"
        "     # configs/hparams_main27.toml の runtime.active_profile = 'train'\n"
        "     uv run mjpython main27_train_final.py\n\n"
        "  1-b) 実行時オプションで train を選択（TOMLを編集しない）\n"
        "     uv run mjpython main27_train_final.py --profile train\n\n"
        "  2) 学習結果を確認する（デバッグ再生）\n"
        "     # configs/hparams_main27.toml の runtime.active_profile = 'debug'\n"
        "     uv run mjpython main27_train_final.py\n\n"
        "  2-b) 実行時オプションで debug を選択（TOMLを編集しない）\n"
        "     uv run mjpython main27_train_final.py --profile debug\n\n"
        "  2-c) 複数プロファイルを連続実行（比較）\n"
        "     uv run mjpython main27_train_final.py --compare-profiles train_wall_only,train_no_wall_stick,train_base_reward\n\n"
        "  3) ヘルプと実行例を表示\n"
        "     uv run mjpython main27_train_final.py -h\n"
    )

    parser = argparse.ArgumentParser(
        description="HideAndSeek Transformer v27 runner",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=cli_examples,
    )
    parser.add_argument(
        "-p",
        "--profile",
        help="実行時だけ runtime profile を上書き（TOMLは変更しない）",
    )
    parser.add_argument(
        "--compare-profiles",
        help=("カンマ区切りで複数 profile を順次実行（例: " "train_wall_only,train_no_wall_stick,train_base_reward）"),
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="実行例のみ表示して終了",
    )
    args, _ = parser.parse_known_args(argv)
    if args.examples:
        print(cli_examples)
        raise SystemExit(0)


def _to_bool(value):
    """文字列をboolに変換"""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg(name, default, cast=None, overrides=None):
    """設定値を取得（オーバーライドを優先）"""
    if overrides is None:
        overrides = {}
    if name in overrides:
        raw = overrides[name]
    else:
        raw = default
    if cast is None:
        return raw
    return cast(raw)


# ============================================================================
# グローバル設定の初期化（一度だけ）
# ============================================================================

#
# CLI設定の取得
CLI_PROFILE_OVERRIDE = _cli_profile_override_from_argv()
CLI_COMPARE_PROFILES = _cli_compare_profiles_from_argv()

# 利用可能なプロファイルの取得
AVAILABLE_RUNTIME_PROFILES = _get_available_runtime_profiles(CONFIG_PATH)

# ランタイム設定のロード
RUNTIME_OVERRIDES, RUN_PROFILE, RUN_PROFILE_SOURCE = _load_runtime_config(CONFIG_PATH, CLI_PROFILE_OVERRIDE, AVAILABLE_RUNTIME_PROFILES)

# TOMLの読み込み
with open(CONFIG_PATH, "rb") as f:
    config = tomllib.load(f)

# ハイパーパラメータの構築
hp = dict(config.get("base", {}))
runtime_common = config.get("runtime", {}).get("common", {})
if runtime_common:
    hp.update(runtime_common)
runtime_profile = config.get("runtime", {}).get(RUN_PROFILE, {})
if runtime_profile:
    hp.update(runtime_profile)
if RUNTIME_OVERRIDES:
    hp.update(RUNTIME_OVERRIDES)

# フラグの初期化
# プロファイルで指定した値（hp）をデフォルトにし、さらに CLI/ランタイムオーバーライドで上書きする。
# debug flags may be set in TOML/runtime overrides
DEBUG_MODE = _cfg("debug_mode", hp.get("debug_mode", "0"), _to_bool, RUNTIME_OVERRIDES)
DEBUG_CONSOLE = _cfg("debug_console", hp.get("debug_console", "1"), _to_bool, RUNTIME_OVERRIDES)

TRAIN_MODE = _cfg("train_mode", hp.get("train_mode", "0"), _to_bool, RUNTIME_OVERRIDES)
USE_VIEWER = _cfg("use_viewer", hp.get("use_viewer", "1"), _to_bool, RUNTIME_OVERRIDES)
NPC_ONLY_DEBUG = _cfg("npc_only_debug", hp.get("npc_only_debug", "1"), _to_bool, RUNTIME_OVERRIDES)
SHOW_TURN_LINES = _cfg("show_turn_lines", hp.get("show_turn_lines", "0"), _to_bool, RUNTIME_OVERRIDES)
USE_CUSTOM_REWARD = _cfg("use_custom_reward", hp.get("use_custom_reward", "1"), _to_bool, RUNTIME_OVERRIDES)
AUTO_TUNE_HPARAMS = _cfg("auto_tune_hparams", hp.get("auto_tune_hparams", "1"), _to_bool, RUNTIME_OVERRIDES)
DEBUG_HIDER_POLICY = _cfg("debug_hider_policy", hp.get("debug_hider_policy", "rule"), str, RUNTIME_OVERRIDES)
DEBUG_SEEKER_POLICY = _cfg("debug_seeker_policy", hp.get("debug_seeker_policy", "rule"), str, RUNTIME_OVERRIDES)
MODEL_POLICY_DETERMINISTIC = _cfg("model_policy_deterministic", hp.get("model_policy_deterministic", "0"), _to_bool, RUNTIME_OVERRIDES)
DEBUG_SYMMETRIC_HIDER_POLICY = _cfg("debug_symmetric_hider_policy", hp.get("debug_symmetric_hider_policy", "0"), _to_bool, RUNTIME_OVERRIDES)
TRAIN_OTHER_HIDER_POLICY = _cfg("train_other_hider_policy", hp.get("train_other_hider_policy", "rule"), str, RUNTIME_OVERRIDES)
TRAIN_OTHER_SEEKER_POLICY = _cfg("train_other_seeker_policy", hp.get("train_other_seeker_policy", "rule"), str, RUNTIME_OVERRIDES)

# 以前の挙動では推論モード (TRAIN_MODE=False) のとき必ずビューアを有効化していたが、
# プロファイルで明示的に `use_viewer = false` とした場合でも上書きされてしまっていた。
# ここでは「デバッグプロファイル（RUN_PROFILE == 'debug'）から来た推論実行」の場合に
# 自動でビューアを有効化するのみとし、プロファイルで明示指定された `use_viewer` を尊重する。
if not TRAIN_MODE:
    if RUN_PROFILE == "debug":
        USE_VIEWER = True
    AUTO_TUNE_HPARAMS = False

MODE = _cfg("mode", hp.get("mode", "refinement"), str, RUNTIME_OVERRIDES)
TRAINING_TARGET = _cfg("training_target", hp.get("training_target", "seeker"), str, RUNTIME_OVERRIDES)


SEQ_LEN = _cfg("seq_len", hp.get("seq_len", "16"), int, RUNTIME_OVERRIDES)
HIDDEN_DIM = _cfg("hidden_dim", hp.get("hidden_dim", "256"), int, RUNTIME_OVERRIDES)
NUM_ENVS = _cfg("num_envs", hp.get("num_envs", "8"), int, RUNTIME_OVERRIDES)


# ENV_CONFIGはグローバル定数初期化後に定義し、依存関係を明確化
ENV_CONFIG = {
    "n_seekers": _cfg("n_seekers", "1", int, RUNTIME_OVERRIDES),
    "n_hiders": _cfg("n_hiders", "2", int, RUNTIME_OVERRIDES),
    "n_boxes": _cfg("n_boxes", "2", int, RUNTIME_OVERRIDES),
    "n_ramps": _cfg("n_ramps", "1", int, RUNTIME_OVERRIDES),
    "mode4_sdf_cell_size": _cfg("mode4_sdf_cell_size", "0.05", float, RUNTIME_OVERRIDES),
    "show_turn_lines": SHOW_TURN_LINES,
    "dbg_log_interval_steps": _cfg("dbg_log_interval_steps", "200", int, RUNTIME_OVERRIDES),
    "debug_mode": DEBUG_MODE,
    "debug_console": DEBUG_CONSOLE,
    "action_repeat": _cfg("action_repeat", "10", int, RUNTIME_OVERRIDES),
}

RAMP_CHECK_ENABLED = _cfg("ramp_check_enabled", "0", _to_bool, RUNTIME_OVERRIDES)
RAMP_CHECK_INTERVAL = _cfg("ramp_check_interval", str(hp.get("save_interval", 50)), int, RUNTIME_OVERRIDES)
RAMP_CHECK_EPISODES = _cfg("ramp_check_episodes", "3", int, RUNTIME_OVERRIDES)
RAMP_CHECK_STEPS = _cfg("ramp_check_steps", "220", int, RUNTIME_OVERRIDES)
RAMP_CHECK_VIEWER = _cfg("ramp_check_viewer", "0", _to_bool, RUNTIME_OVERRIDES)
RAMP_CHECK_FORCE_UPHILL = _cfg("ramp_check_force_uphill", "0", _to_bool, RUNTIME_OVERRIDES)
RAMP_SUCCESS_PROG = _cfg("ramp_success_prog", "0.90", float, RUNTIME_OVERRIDES)
RAMP_SUCCESS_TOP = _cfg("ramp_success_top", "0.05", float, RUNTIME_OVERRIDES)
RAMP_SUCCESS_LAT = _cfg("ramp_success_lat", "0.75", float, RUNTIME_OVERRIDES)
RAMP_SUCCESS_Z_RISE = _cfg("ramp_success_z_rise", "0.18", float, RUNTIME_OVERRIDES)

WANDB_ENABLED = _cfg("wandb_enabled", "1", _to_bool, RUNTIME_OVERRIDES)
WANDB_PROJECT = _cfg("wandb_project", "hideandseek-xformer", str, RUNTIME_OVERRIDES)
WANDB_ENTITY = _cfg("wandb_entity", "", str, RUNTIME_OVERRIDES)
WANDB_MODE = _cfg("wandb_mode", "online", str, RUNTIME_OVERRIDES)
WANDB_RUN_NAME = _cfg("wandb_run_name", "", str, RUNTIME_OVERRIDES)
WANDB_LOG_CODE = _cfg("wandb_log_code", "1", _to_bool, RUNTIME_OVERRIDES)
WANDB_CODE_ROOT = _cfg("wandb_code_root", os.path.dirname(__file__), str, RUNTIME_OVERRIDES)

RESUME_TRAINING = _cfg("resume_training", "0", _to_bool, RUNTIME_OVERRIDES)  # 学習再開フラグ（Trueの場合、checkpoint_pathからモデルをロードして学習再開を試みる）
RESET_MODEL_ON_TRAIN = _cfg(
    "reset_model_on_train", "0", _to_bool, RUNTIME_OVERRIDES
)  # 学習開始時にモデルを初期化するか（Trueの場合、checkpoint_pathからモデル構造はロードするが重みはランダム初期化する。RESUME_TRAINING=Trueと組み合わせて、構造は同じで重みだけリセットして学習再開したい場合などに使用）
HPARAMS_CONFIG_PATH = CONFIG_PATH  # ハイパーパラメータの設定ファイルパス（学習再開時に前回と同じハイパーパラメータで再開するために使用）

# 報酬形成係数
RW_SEEK_VISIBLE_BONUS = _cfg("rw_seek_visible_bonus", "0.05", float, RUNTIME_OVERRIDES)  # SeekerがHiderを視界に捉えている場合の報酬（Seeker専用、視界に入っているほど報酬大）
RW_STILL_SPEED_THRESHOLD = _cfg("rw_still_speed_threshold", "0.1", float, RUNTIME_OVERRIDES)  # 静止判定速度閾値（全エージェント共通）

RW_MOVE_SAT_PENALTY = _cfg("rw_move_sat_penalty", "0.02", float, RUNTIME_OVERRIDES)  # 移動行動の飽和ペナルティ（全エージェント共通、速度の絶対値が閾値を超えるとペナルティ発生）
RW_TURN_SAT_PENALTY = _cfg("rw_turn_sat_penalty", "0.02", float, RUNTIME_OVERRIDES)  # 回転行動の飽和ペナルティ（全エージェント共通、回転の絶対値が閾値を超えるとペナルティ発生）
RW_MOVE_CTRL_COST = _cfg("rw_move_ctrl_cost", "0.001", float, RUNTIME_OVERRIDES)  # 行動コスト（移動と回転の両方に適用、全エージェント共通）

RW_MOVE_INCENTIVE = _cfg("rw_move_incentive", "0.02", float, RUNTIME_OVERRIDES)  # 移動インセンティブ（速度が閾値に近づくほど増加、閾値以上で一定、閾値以下で減少。全エージェント共通）
RW_IDLE_PENALTY = _cfg("rw_idle_penalty", "0.03", float, RUNTIME_OVERRIDES)  # 動いていないほどペナルティ大（全エージェント共通）
RW_TURN_INCENTIVE = _cfg("rw_turn_incentive", "0.01", float, RUNTIME_OVERRIDES)  # 敵が見えていない時の回転インセンティブ（デフォルト0）
RW_WALL_AVOID_PENALTY = _cfg("rw_wall_avoid_penalty", "0.15", float, RUNTIME_OVERRIDES)  # 壁回避ペナルティ（全エージェント共通、壁に近いほどペナルティ大）

RW_WALL_STICK_PENALTY = _cfg("rw_wall_stick_penalty", "0.25", float, RUNTIME_OVERRIDES)  # 壁に張り付いていると判断された場合のペナルティ（全エージェント共通）
RW_HIDE_VISIBLE_NEAR_PENALTY = _cfg("rw_hide_visible_near_penalty", "0.05", float, RUNTIME_OVERRIDES)  # HiderがSeekerに近くて見えている場合のペナルティ（Hider専用、距離が近いほどペナルティ大）

RW_HIDE_SEEKER_GAZE_PENALTY = _cfg(
    "rw_hide_seeker_gaze_penalty", "0.03", float, RUNTIME_OVERRIDES
)  # HiderがSeekerの視界に入っている場合のペナルティ（Hider専用、Seekerの視界に入っているほどペナルティ大）
RW_SEEK_GAZE_REWARD = _cfg("c", "0.0", float, RUNTIME_OVERRIDES)  # Seeker gaze reward disabled (gaze_cos moved to environment base reward)

# BEING_HIT（痛覚）に基づく離脱報酬係数（Hider用）
RW_EVASION_BONUS = _cfg("rw_evasion_bonus_coeff", "0.05", float, RUNTIME_OVERRIDES)

# --- 速度・回転の閾値（TOMLで設定可） ---
IDLE_SPEED_THRESHOLD = _cfg("idle_speed_threshold", "0.1", float, RUNTIME_OVERRIDES)  # 静止判定速度閾値
MOVE_INCENTIVE_SPEED_THRESHOLD = _cfg("move_incentive_speed_threshold", "0.3", float, RUNTIME_OVERRIDES)  # インセンティブ速度閾値
MOVE_SAT_THRESHOLD = _cfg("move_sat_threshold", "1.0", float, RUNTIME_OVERRIDES)  # 行動飽和の閾値（速度）
TURN_SAT_THRESHOLD = _cfg("turn_sat_threshold", "1.0", float, RUNTIME_OVERRIDES)  # 行動飽和の閾値（回転）

# (removed local lowercase aliases; pass float(RW_*) inline at call sites)
WALL_SAFE = _cfg("wall_safe", "1.0", float, RUNTIME_OVERRIDES)

# エージェント半径を設定ファイルから取得（デフォルト 0.4）
AGENT_RADIUS = _cfg("agentl_radius", "0.4", float, RUNTIME_OVERRIDES)
# 壁近マージン（エージェント表面からの近接閾値; デフォルト 0.4）
WALL_NEAR_MARGIN = _cfg("wall_near_margin", "0.6", float, RUNTIME_OVERRIDES)
AGENT_HEAD_CLEARANCE = _cfg("agent_head_clearance", "0.2", float, RUNTIME_OVERRIDES)  # 壁近判定閾値（全エージェント共通）
WALL_REPULSION_RADIUS_DEFAULT = 0.25

# ============================================================================
# 観測履歴・ユーティリティクラス
# ============================================================================


class ObsHistory:
    def __init__(self, num_envs, seq_len, obs_dim, device):
        self.num_envs = int(num_envs)
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.device = device
        self.buffer = torch.zeros(
            (self.num_envs, seq_len * 2, obs_dim),
            device=device,
        )
        self.ptr = 0

    def reset(self):
        self.buffer.zero_()
        self.ptr = 0

    def prime(self, obs):
        self.reset()
        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        )
        obs_t = obs_t.reshape(self.num_envs, self.obs_dim)
        for i in range(self.seq_len):
            self.buffer[:, i] = obs_t
            self.buffer[:, i + self.seq_len] = obs_t

    def update(self, obs):
        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        )
        obs_t = obs_t.reshape(self.num_envs, self.obs_dim)
        self.buffer[:, self.ptr] = obs_t
        self.buffer[:, self.ptr + self.seq_len] = obs_t
        self.ptr = (self.ptr + 1) % self.seq_len

    def prime_single(self, obs):
        self.prime(np.asarray(obs, dtype=np.float32).reshape(1, -1))

    def update_single(self, obs):
        self.update(np.asarray(obs, dtype=np.float32).reshape(1, -1))

    def reset_env(self, env_idx, obs):
        obs_t = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).reshape(self.obs_dim)
        for i in range(self.seq_len):
            self.buffer[env_idx, i] = obs_t
            self.buffer[env_idx, i + self.seq_len] = obs_t

    def get(self):
        start = self.ptr
        end = self.ptr + self.seq_len
        return self.buffer[:, start:end]


# ============================================================================
# 報酬計算関数（インデックス・バッチ処理）
# ============================================================================


def _index_spec_to_array(index_spec):
    """インデックス仕様を配列に変換"""
    if isinstance(index_spec, slice):
        start = 0 if index_spec.start is None else int(index_spec.start)
        stop = int(index_spec.stop)
        step = 1 if index_spec.step is None else int(index_spec.step)
        return np.arange(start, stop, step, dtype=np.int32)
    return np.asarray(index_spec, dtype=np.int32)


def _build_reward_index_cache(idx):
    """報酬計算用のインデックスキャッシュを構築"""
    return {
        "self_vel_x": int(idx.SELF.VEL_X),
        "self_vel_y": int(idx.SELF.VEL_Y),
        "lidar": _index_spec_to_array(idx.LIDAR),
        "enemy_visible": np.asarray([int(en.VISIBLE) for en in idx.OTHERS], dtype=np.int32),
        "enemy_rel_x": np.asarray([int(en.REL_X) for en in idx.OTHERS], dtype=np.int32),
        "enemy_rel_y": np.asarray([int(en.REL_Y) for en in idx.OTHERS], dtype=np.int32),
        "enemy_quat_0": np.asarray([int(en.QUAT_0) for en in idx.OTHERS], dtype=np.int32),
        "enemy_quat_1": np.asarray([int(en.QUAT_1) for en in idx.OTHERS], dtype=np.int32),
        "enemy_being_hit": np.asarray([int(en.BEING_HIT) for en in idx.OTHERS], dtype=np.int32),
    }


def _min_lidar_from_obs(obs, lidar_indices):
    """観測からLiDAー最小値を取得"""
    # 前方3方向のみ（中央＋左右1つずつ）
    n = len(lidar_indices)
    if n < 3:
        # fallback: 全方向
        indices = lidar_indices
    else:
        center = n // 2
        indices = [lidar_indices[center - 1], lidar_indices[center], lidar_indices[center + 1]]
    lidar_min = 1.0
    for lidar_idx in indices:
        value = float(obs[int(lidar_idx)])
        if value < lidar_min:
            lidar_min = value
    return lidar_min


# 各エージェントごとに1つだけキャッシュ（idx, reward_idx_cacheのid）
_wall_stick_state_cache = {}


def _is_wall_stick_state(obs, idx, reward_idx_cache=None, speed=None, info=None, agent_idx=None):
    """壁張り付き状態かチェック（全エージェント共通, エージェントごとキャッシュ, speed外部渡し可, infoのwall_distance優先)"""
    global _wall_stick_state_cache
    cache_id = id(reward_idx_cache) if reward_idx_cache is not None else id(idx)
    key = (idx, cache_id, agent_idx if agent_idx is not None else 0)
    if key in _wall_stick_state_cache:
        # print(f"[DEBUG] wall_stick_state_cache hit: {key} -> {_wall_stick_state_cache[key]}")
        return _wall_stick_state_cache[key]
    # まず観測内の WALL_DIST を優先して使う（正規化距離: 0..1）。なければ info を参照し、それもなければ LiDAR で代替
    wall_near = None
    try:
        # obs は 1D 観測ベクトル。idx.WALL_DIST が定義されていれば利用
        if hasattr(idx, "WALL_DIST"):
            wall_dist_norm = float(obs[int(idx.WALL_DIST)])
            # 正規化距離 -> 実距離の近似（env の既定半径を使う）
            wall_distance = wall_dist_norm * WALL_REPULSION_RADIUS_DEFAULT
            wall_near = (wall_distance - AGENT_RADIUS) < WALL_NEAR_MARGIN
    except Exception:
        wall_near = None
    if wall_near is None:
        if info is not None and "wall_distance" in info:
            wall_distance = info["wall_distance"]
            if isinstance(wall_distance, (list, tuple, np.ndarray)):
                wall_distance = float(np.min(wall_distance))
            else:
                wall_distance = float(wall_distance)
            wall_near = (wall_distance - AGENT_RADIUS) < WALL_NEAR_MARGIN
    if wall_near is None:
        cache = reward_idx_cache if reward_idx_cache is not None else _build_reward_index_cache(idx)
        lidar_indices = cache["lidar"]
        # idx.LIDAR_FRONT_IDXをそのまま使う（範囲チェック不要）
        front_indices = np.array(idx.LIDAR_FRONT_IDX, dtype=np.int32)
        lidar_min = 1.0
        for lidar_idx in front_indices:
            value = float(obs[int(lidar_idx)])
            if value < lidar_min:
                lidar_min = value
        wall_near = (lidar_min - AGENT_RADIUS) < WALL_NEAR_MARGIN
    if not wall_near:
        # print(f"[DEBUG] wall_stick: wall_near=False (lidar_min={lidar_min:.3f}, threshold={AGENT_HEAD_CLEARANCE})")
        _wall_stick_state_cache[key] = False
        return False
    # 壁際なら速度判定
    if speed is None:
        cache = reward_idx_cache if reward_idx_cache is not None else _build_reward_index_cache(idx)
        speed = math.sqrt(float(obs[cache["self_vel_x"]]) ** 2 + float(obs[cache["self_vel_y"]]) ** 2)
    result = speed < RW_STILL_SPEED_THRESHOLD
    # print(f"[DEBUG] wall_stick: wall_near={wall_near}, speed={speed:.4f}, threshold={RW_STILL_SPEED_THRESHOLD}, result={result}, obs_vel=({obs[cache['self_vel_x']]:.4f},{obs[cache['self_vel_y']]:.4f})")
    _wall_stick_state_cache[key] = result
    return result


def _clear_wall_stick_state_cache():
    """_is_wall_stick_stateのキャッシュをクリア"""
    global _wall_stick_state_cache
    _wall_stick_state_cache.clear()


# デバッグ: Numbaコンパイル前に到達するか確認
# debug: removed compilation print


@njit(cache=True, parallel=True)
def _compute_custom_reward_batch_numba(
    obs_batch,
    next_obs_batch,
    action_batch,
    base_reward_batch,
    target_is_hider,
    self_vel_x_idx,
    self_vel_y_idx,
    enemy_visible_indices,
    enemy_rel_x_indices,
    enemy_rel_y_indices,
    enemy_being_hit_indices,
    move_ctrl_cost,
    move_incentive,
    idle_penalty,
    wall_avoid_penalty,
    wall_dist_idx,
    wall_repulsion_radius,
    idle_speed_threshold,
    move_incentive_speed_threshold,
    move_sat_threshold,
    move_sat_penalty,
    turn_sat_threshold,
    turn_sat_penalty,
    rw_turn_incentive,
    still_speed_threshold,
    wall_stick_penalty,
    agent_radius,
    evasion_bonus_coeff,
    agent_vz_batch,
):
    """バッチ報酬計算（Numba JIT最適化）"""
    n_envs = obs_batch.shape[0]
    rewards = np.empty(n_envs, dtype=np.float32)
    for i in prange(n_envs):
        move = np.float32(action_batch[i, 0])
        turn = np.float32(action_batch[i, 1])
        next_vel_x = np.float32(next_obs_batch[i, self_vel_x_idx])
        next_vel_y = np.float32(next_obs_batch[i, self_vel_y_idx])
        next_speed = np.float32(np.sqrt(next_vel_x * next_vel_x + next_vel_y * next_vel_y))

        bonus = np.float32(0.0)

        # --- 速度ペナルティ・インセンティブ（hider/seeker共通） ---
        if next_speed < idle_speed_threshold:
            bonus -= idle_penalty * (idle_speed_threshold - next_speed)
        elif next_speed < move_incentive_speed_threshold:
            bonus += move_incentive * ((next_speed - idle_speed_threshold) / (move_incentive_speed_threshold - idle_speed_threshold))
        elif next_speed <= move_sat_threshold:
            bonus += move_incentive
        else:
            bonus += move_incentive
            bonus -= move_sat_penalty * (next_speed - move_sat_threshold)

        # --- 回転ペナルティ・インセンティブ（hider/seeker共通） ---
        abs_turn = np.abs(turn)

        # 1. 飽和ペナルティ: 制御限界に近い回転にはブレーキをかける
        if abs_turn > turn_sat_threshold:
            bonus -= turn_sat_penalty * (abs_turn - turn_sat_threshold)
        else:
            # 2. 探索インセンティブ: 敵が見えていない時だけ、周囲を見渡す動きを促す
            visible = False
            for j in range(enemy_visible_indices.shape[0]):
                idx_vis = int(enemy_visible_indices[j])
                if obs_batch[i, idx_vis] > 0.5:
                    visible = True
                    break

            if not visible:
                # 独楽回りを防ぐため、ここでも「低速な回転」に限定してボーナスを出す設計にする
                # abs_turn が大きすぎない範囲で加点
                half_threshold = turn_sat_threshold / 2.0 if turn_sat_threshold > 0.0 else 0.5
                if abs_turn > half_threshold:  # abs_turn が閾値の半分を超えると、そこから先は減少させる（ただし turn_sat_threshold / 2.0 まで）
                    bonus += rw_turn_incentive * (1.0 - ((abs_turn - half_threshold) / half_threshold))
                else:
                    bonus += rw_turn_incentive * abs_turn / half_threshold
                    # 回転が大きいほどインセンティブ増加（ただし turn_sat_threshold / 2.0 まで）

        control_cost = -move_ctrl_cost * (move * move + turn * turn)
        # control_cost = 0.0 # 行動コストは一旦無効化（速度ペナルティと重複するため）

        # これにより、壁を乗り越えるための接近が「正解」として許容される
        is_on_ramp_climbing = agent_vz_batch[i] > 0.02

        if not is_on_ramp_climbing:
            # WALL_DIST が観測内にあれば正規化距離を使って実距離に復元
            wall_distance = np.nan
            if wall_dist_idx >= 0:
                try:
                    wall_distance = float(obs_batch[i, wall_dist_idx]) * float(wall_repulsion_radius)
                except Exception:
                    wall_distance = np.nan

            if not np.isnan(wall_distance):
                dist_from_surface = wall_distance - agent_radius
                wall_near_margin = 0.2  # ペナルティ開始境界

                if dist_from_surface < wall_near_margin:
                    # 0.0（境界）から 1.0（接触）に向かって線形に増加する比率
                    m = dist_from_surface
                    if m < 0.0:
                        m = 0.0
                    ratio = 1.0 - (m / wall_near_margin)

                    # A. 基本の接近ペナルティ
                    bonus -= wall_avoid_penalty * ratio

                    # B. 張り付き判定
                    if next_speed < still_speed_threshold:
                        bonus -= wall_stick_penalty * ratio

            # BEING_HIT ブロックはランプ登り中でも有効にするため、ここでは処理を行わない

        # --- BEING_HIT (pain) based evasion reward for Hider ---
        if target_is_hider:
            for j in range(enemy_visible_indices.shape[0]):
                idx_vis = int(enemy_visible_indices[j])
                visible = next_obs_batch[i, idx_vis]
                being_hit = next_obs_batch[i, int(enemy_being_hit_indices[j])]
                if visible < 0.5 and being_hit > 0.5:
                    idx_x = int(enemy_rel_x_indices[j])
                    idx_y = int(enemy_rel_y_indices[j])
                    cur_dx = float(next_obs_batch[i, idx_x])
                    cur_dy = float(next_obs_batch[i, idx_y])
                    current_dist = np.sqrt(cur_dx * cur_dx + cur_dy * cur_dy) * P_SCALE
                    prev_dx = float(obs_batch[i, idx_x])
                    prev_dy = float(obs_batch[i, idx_y])
                    prev_dist = np.sqrt(prev_dx * prev_dx + prev_dy * prev_dy) * P_SCALE
                    if prev_dist < (L_SCALE * 0.9):
                        dist_delta = current_dist - prev_dist
                        bonus += evasion_bonus_coeff * dist_delta

        rewards[i] = float(base_reward_batch[i]) + bonus + control_cost
    return rewards


def compute_custom_reward(obs, next_obs, action, base_reward, idx, target, info, reward_idx_cache=None):
    # obs, next_obs, action, base_reward はバッチまたは単一観測で渡される可能性がある。
    # NumPy配列に変換し、単一観測なら先頭にバッチ次元を追加して常にバッチ形状にする。
    obs_arr = np.asarray(obs, dtype=np.float32)
    if obs_arr.ndim == 1:
        obs_batch = obs_arr.reshape(1, -1)
    else:
        obs_batch = obs_arr

    next_obs_arr = np.asarray(next_obs, dtype=np.float32)
    if next_obs_arr.ndim == 1:
        next_obs_batch = next_obs_arr.reshape(1, -1)
    else:
        next_obs_batch = next_obs_arr

    action_arr = np.asarray(action, dtype=np.float32)
    if action_arr.ndim == 1:
        action_batch = action_arr.reshape(1, -1)
    else:
        action_batch = action_arr

    base_reward_arr = np.asarray(base_reward, dtype=np.float32)
    if base_reward_arr.ndim == 0:
        base_reward_batch = base_reward_arr.reshape(1)
    elif base_reward_arr.ndim == 1:
        base_reward_batch = base_reward_arr
    else:
        base_reward_batch = base_reward_arr

    target_is_hider = target
    cache = reward_idx_cache if reward_idx_cache is not None else _build_reward_index_cache(idx)
    self_vel_x_idx = int(idx.SELF.VEL_X)
    self_vel_y_idx = int(idx.SELF.VEL_Y)
    enemy_visible_indices = cache["enemy_visible"]
    enemy_rel_x_indices = cache["enemy_rel_x"]
    enemy_rel_y_indices = cache["enemy_rel_y"]
    enemy_being_hit_indices = cache["enemy_being_hit"]

    # 共通処理: 可視な敵の相対位置から距離・正規化近接量を計算して返す
    def _enemy_proximity(i, j):
        idx_vis = int(enemy_visible_indices[j])
        if obs_batch[i, idx_vis] <= 0.5:
            return None
        idx_x = int(enemy_rel_x_indices[j])
        idx_y = int(enemy_rel_y_indices[j])
        dx = float(next_obs_batch[i, idx_x])
        dy = float(next_obs_batch[i, idx_y])
        dist = np.sqrt(dx * dx + dy * dy) * P_SCALE
        amount = (2.0 - dist) / 2.0 if dist < 2.0 else 0.0
        return dx, dy, dist, amount, idx_x, idx_y, idx_vis

    # WALL_DIST が観測内にあればそのインデックスを numba に渡す（正規化距離 -> 実距離は numba 側で復元）
    if hasattr(idx, "WALL_DIST"):
        try:
            wall_dist_idx = int(idx.WALL_DIST)
        except Exception:
            wall_dist_idx = -1
    else:
        wall_dist_idx = -1
    wall_repulsion_radius_value = float(WALL_REPULSION_RADIUS_DEFAULT)
    # agent_vz_batchの生成（infoから取得、なければ0.0）
    if info is not None and "agent_vz" in info:
        vz = info["agent_vz"]
        if isinstance(vz, (list, tuple, np.ndarray)):
            agent_vz_batch = np.asarray(vz, dtype=np.float32)
        else:
            agent_vz_batch = np.full((obs_batch.shape[0],), float(vz), dtype=np.float32)
    else:
        agent_vz_batch = np.zeros((obs_batch.shape[0],), dtype=np.float32)
    # debug prints removed
    # Numba版呼び出し（安全に実行し、例外が発生したら基礎報酬を返す）
    try:
        reward_np = _compute_custom_reward_batch_numba(
            obs_batch,
            next_obs_batch,
            action_batch,
            base_reward_batch,
            target_is_hider,
            self_vel_x_idx,
            self_vel_y_idx,
            enemy_visible_indices,
            enemy_rel_x_indices,
            enemy_rel_y_indices,
            enemy_being_hit_indices,
            float(RW_MOVE_CTRL_COST),
            float(RW_MOVE_INCENTIVE),
            float(RW_IDLE_PENALTY),
            0.0,
            int(wall_dist_idx),
            wall_repulsion_radius_value,
            float(IDLE_SPEED_THRESHOLD),
            float(MOVE_INCENTIVE_SPEED_THRESHOLD),
            float(MOVE_SAT_THRESHOLD),
            float(RW_MOVE_SAT_PENALTY),
            float(TURN_SAT_THRESHOLD),
            float(RW_TURN_SAT_PENALTY),
            float(RW_TURN_INCENTIVE),
            float(RW_STILL_SPEED_THRESHOLD),
            float(RW_WALL_STICK_PENALTY),
            float(AGENT_RADIUS),  # AGENT_RADIUS
            float(RW_EVASION_BONUS),
            agent_vz_batch,
        )
    except Exception as exc:
        print(f"compute_custom_reward: Numba error: {exc}")
        try:
            # fall back to provided base reward for the first element
            return float(base_reward_batch[0])
        except Exception:
            return float(base_reward)
    # ensure numeric return
    try:
        return float(reward_np[0])
    except Exception:
        return float(base_reward_batch[0])


# ============================================================================
# 環境構築・ユーティリティ関数
# ============================================================================


def env_signature(config):
    """環境設定のシグネチャ文字列を生成（名やログ用）"""
    keys = [
        "n_seekers",
        "n_hiders",
        "n_boxes",
        "n_ramps",
        "mode4_sdf_cell_size",
    ]
    return ",".join(f"{k}={config.get(k)}" for k in keys if k in config)


def model_path_for_config(target, config):
    """モデルの保存・読み込み用パスを一意に決定する"""
    fname = f"HNS_V27_{target}_s{config.get('n_seekers', 1)}_h{config.get('n_hiders', 2)}_b{config.get('n_boxes', 2)}_r{config.get('n_ramps', 1)}.pt"
    return os.path.join("checkpoints", fname)


def build_env(mode, target, config, render_mode=None):
    """環境を構築"""
    config = dict(config)  # 破壊的変更を避ける
    config.pop("num_envs", None)
    return TeamCosEnv(
        mode=mode,
        target=target,
        render_mode=render_mode,
        **config,
    )


class SingleVecWrapper:
    """単一環境を1要素ベクトルとして扱う薄いラッパー。
    run_train_vector 側が期待する `reset()`, `step(action_np)`, `reset_at(i)`,
    `call(method, ...)` および `close()` を提供する。
    """

    def __init__(self, env):
        self._env = env
        self._last_obs = None

    def reset(self, options=None):
        obs, info = self._env.reset()
        self._last_obs = np.asarray(obs, dtype=np.float32)
        # 戻り値はベクトル形式: (obs_batch, info)
        return np.asarray([self._last_obs], dtype=np.float32), info

    def step(self, action_np):
        # action_np may be shape (1, act_dim) or (act_dim,)
        if hasattr(action_np, "ndim") and action_np.ndim == 2:
            a = action_np[0]
        else:
            a = action_np
        next_obs, base_r, term, trun, info = self._env.step(a)
        self._last_obs = np.asarray(next_obs, dtype=np.float32)
        return (
            np.asarray([self._last_obs], dtype=np.float32),
            np.asarray([base_r], dtype=np.float32),
            np.asarray([term], dtype=np.float32),
            np.asarray([trun], dtype=np.float32),
            info,
        )

    def reset_at(self, i):
        # single env: reset and return single observation (non-batched)
        obs, info = self._env.reset()
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return obs, info

    def render(self, *args, **kwargs):
        try:
            return self._env.render(*args, **kwargs)
        except Exception as exc:
            print(f"SingleVecWrapper.render: env.render raised: {exc}")
            import traceback

            traceback.print_exc()
            return None

    def call(self, method_name, *args, **kwargs):
        # emulate vector env .call() returning a list of per-env results
        fn = getattr(self._env, method_name, None)
        if fn is None:
            raise AttributeError(f"Underlying env has no method {method_name}")
        res = fn(*args, **kwargs)
        return [res]

    def close(self, terminate=True):
        try:
            self._env.close()
        except Exception:
            pass


# ============================================================================
# モデル・状態管理関数
# ============================================================================


def _snapshot_state_dict_cpu(agent):
    """エージェントの状態をCPUにコピー"""
    return {key: value.detach().cpu().clone() for key, value in agent.state_dict().items()}


def _normalize_policy_mode(value):
    """ポリシーモードを正規化"""
    text = str(value).strip().lower()
    if text in {"model", "model_if_available", "inference"}:
        return POLICY_MODEL_IF_AVAILABLE
    return POLICY_RULE


def _resolve_runtime_target():
    """ランタイムターゲットを決定"""
    # 以前は非TRAIN_MODE時に'hider'に固定していたが、
    # デバッグや再生で明示的に`training_target`を指定する用途があるため
    # 設定された `TRAINING_TARGET` を常に尊重するようにする。
    return TRAINING_TARGET


def _maybe_load_model_state(target):
    """モデル状態をロード（存在しない場合はNone）"""
    path = model_path_for_config(target, ENV_CONFIG)
    if not os.path.exists(path):
        return None, path
    try:
        return torch.load(path, map_location="cpu"), path
    except Exception as exc:
        print(f"Model load failed ({target}: {exc})")
        return None, path


def _apply_policy_state(env, vec_envs, agent_keys, state_dict, label):
    """ポリシー状態を適用"""
    if not agent_keys or state_dict is None:
        return False
    if vec_envs is not None:
        try:
            results = vec_envs.call(
                "set_inference_policy_state",
                list(agent_keys),
                state_dict,
                SEQ_LEN,
                HIDDEN_DIM,
            )
            enabled = sum(1 for x in results if bool(x))
            print(f"{label}: enabled on {enabled}/{len(results)} envs")
            return enabled > 0
        except Exception as exc:
            print(f"{label}: vector sync failed ({exc})")
            return False
    if env is not None:
        ok = bool(env.set_inference_policy_state(list(agent_keys), state_dict, SEQ_LEN, HIDDEN_DIM))
        if ok:
            print(f"{label}: enabled")
        return ok
    return False


def _set_override_learnable_policy(env, vec_envs, enabled):
    """学習可能ポリシーのオーバーライドを設定"""
    if vec_envs is not None:
        try:
            vec_envs.call("set_override_learnable_policy", bool(enabled))
            return True
        except Exception as exc:
            print(f"set_override_learnable_policy: vector sync failed ({exc})")
            return False
    if env is not None:
        try:
            return bool(env.set_override_learnable_policy(bool(enabled)))
        except Exception as exc:
            print(f"set_override_learnable_policy failed ({exc})")
            return False
    return False


def _set_model_policy_deterministic(env, vec_envs, enabled):
    """モデルポリシーを決定論的に設定"""
    if vec_envs is not None:
        try:
            vec_envs.call("set_model_policy_deterministic", bool(enabled))
            return True
        except Exception as exc:
            print(f"set_model_policy_deterministic: vector sync failed ({exc})")
            return False
    if env is not None:
        try:
            return bool(env.set_model_policy_deterministic(bool(enabled)))
        except Exception as exc:
            print(f"set_model_policy_deterministic failed ({exc})")
            return False
    return False


def configure_team_policy_modes(env, vec_envs, ref_env, runtime_target, primary_state_dict):
    """チームポリシーのモードを設定"""
    if ref_env is None:
        return

    if TRAIN_MODE:
        hider_mode = _normalize_policy_mode(TRAIN_OTHER_HIDER_POLICY)
        seeker_mode = _normalize_policy_mode(TRAIN_OTHER_SEEKER_POLICY)
    else:
        hider_mode = _normalize_policy_mode(DEBUG_HIDER_POLICY)
        seeker_mode = _normalize_policy_mode(DEBUG_SEEKER_POLICY)

    # Preserve explicit debug override; do not force deterministic behavior
    # during training unless explicitly requested. Use the boolean flag
    # Use the configured deterministic flag (MODEL_POLICY_DETERMINISTIC).
    model_policy_deterministic = MODEL_POLICY_DETERMINISTIC
    det_sync_ok = _set_model_policy_deterministic(env, vec_envs, model_policy_deterministic)

    hider_keys = [k for k in ref_env.hider_keys if k != ref_env.learnable_agent_key]
    seeker_keys = [k for k in ref_env.seeker_keys if k != ref_env.learnable_agent_key]

    hider_state = primary_state_dict if runtime_target == "hider" else None
    seeker_state = primary_state_dict if runtime_target == "seeker" else None

    if hider_mode == POLICY_MODEL_IF_AVAILABLE and hider_state is None:
        hider_state, hider_path = _maybe_load_model_state("hider")
        if hider_state is None:
            print(f"Hider policy: fallback to rule (missing model: {hider_path})")
            if hasattr(ref_env, "set_override_learnable_policy"):
                ref_env.set_override_learnable_policy(True)
        else:
            print(f"Hider model loaded from: {hider_path}")

    if seeker_mode == POLICY_MODEL_IF_AVAILABLE and seeker_state is None:
        seeker_state, seeker_path = _maybe_load_model_state("seeker")
        if seeker_state is None:
            print(f"Seeker policy: fallback to rule (missing model: {seeker_path})")
            if hasattr(ref_env, "set_override_learnable_policy"):
                ref_env.set_override_learnable_policy(True)
        else:
            print(f"Seeker model loaded from: {seeker_path}")

    if hider_mode == POLICY_MODEL_IF_AVAILABLE:
        ok = _apply_policy_state(env, vec_envs, hider_keys, hider_state, "Hider inference policy")
        print(f"Hider inference apply: {ok}")
    if seeker_mode == POLICY_MODEL_IF_AVAILABLE:
        ok2 = _apply_policy_state(env, vec_envs, seeker_keys, seeker_state, "Seeker inference policy")
        print(f"Seeker inference apply: {ok2}")

    symmetric_hider_debug = (not TRAIN_MODE) and DEBUG_SYMMETRIC_HIDER_POLICY and runtime_target == "hider" and hider_mode == POLICY_MODEL_IF_AVAILABLE and (hider_state is not None)
    override_enabled = False
    if symmetric_hider_debug:
        all_hider_keys = list(ref_env.hider_keys)
        ok = _apply_policy_state(
            env,
            vec_envs,
            all_hider_keys,
            hider_state,
            "Debug symmetric hider policy",
        )
        override_enabled = bool(ok)
        if ok:
            print("Debug mode: all hiders use same model inference path")

    print(
        "Policy wiring: "
        f"hider_mode={hider_mode}, "
        f"seeker_mode={seeker_mode}, "
        f"model_policy_deterministic={MODEL_POLICY_DETERMINISTIC}, "
        f"det_sync_ok={det_sync_ok}, "
        f"override_learnable_policy={override_enabled}"
    )


def safe_close_vector_env(vec_envs):
    """ベクター環境を安全にクローズ"""
    if vec_envs is None:
        return
    try:
        vec_envs.close(terminate=True)
    except WORKER_SHUTDOWN_ERRORS as exc:
        print(f"VectorEnv close skipped ({exc.__class__.__name__})")
    except Exception as exc:
        print(f"VectorEnv close suppressed ({exc.__class__.__name__}: {exc})")


# ============================================================================
# 評価・ビューワー関数
# ============================================================================


def evaluate_fixed_ramp_climb(mode, target, agent, device, episodes=3, max_steps=220):
    """
    固定ランプ登坂評価（ユーザー定義版）。
    mode: 環境モード
    target: "hider" or "seeker"
    agent: 学習済みエージェント
    device: torchデバイス
    episodes: 評価エピソード数
    max_steps: 1エピソードの最大ステップ数
    """
    env = build_env(mode, target, ENV_CONFIG, render_mode=None)
    obs_dim = env.observation_space.shape[0]
    history = ObsHistory(1, SEQ_LEN, obs_dim, device)
    success_count = 0
    peak_heights = []
    peak_progress = []
    for ep in range(episodes):
        obs, _ = env.reset()
        history.prime_single(obs)
        # deterministic placement: place learnable agent under first ramp facing uphill
        try:
            if getattr(env, "ramp_ids", None):
                rid = env.ramp_ids[0]
                rpos = env.data.xpos[rid][:2].copy()
                up = env._ramp_uphill_dir(rid)
                offset_along = -0.6
                place_xy = rpos + up * offset_along
                ak = env.learnable_agent_key
                jx = env.qpos_indices[ak]["x"]
                jy = env.qpos_indices[ak]["y"]
                jz = env.qpos_indices[ak]["z"]
                jr = env.qpos_indices[ak]["rot"]
                qx_adr = env.model.jnt_qposadr[jx]
                qy_adr = env.model.jnt_qposadr[jy]
                qz_adr = env.model.jnt_qposadr[jz]
                qr_adr = env.model.jnt_qposadr[jr]
                # place agent on top of ramp: ramp body z + estimated slope offset
                ramp_body_z = float(env.data.xpos[rid][2])
                # drop from slightly above the ramp so the agent settles naturally
                place_z = ramp_body_z + 1.0
                env.data.qpos[qx_adr] = float(place_xy[0])
                env.data.qpos[qy_adr] = float(place_xy[1])
                env.data.qpos[qz_adr] = float(place_z)
                rot = float(math.atan2(up[1], up[0]))
                env.data.qpos[qr_adr] = rot
                try:
                    import mujoco

                    mujoco.mj_forward(env.model, env.data)
                except Exception:
                    pass
        except Exception:
            pass
        done = False
        max_height = -float("inf")
        max_progress = -float("inf")
        for step in range(max_steps):
            with torch.no_grad():
                seq = history.get()
                out = agent.get_action_and_value(seq)
                action = out[0].cpu().numpy().reshape(-1)
            next_obs, _, term, trun, info = env.step(action)
            history.update(next_obs)
            obs = next_obs
            if "dbg_ramp_height" in info:
                h = float(info["dbg_ramp_height"])
                if h > max_height:
                    max_height = h
            if "dbg_ramp_progress" in info:
                p = float(info["dbg_ramp_progress"])
                if p > max_progress:
                    max_progress = p
            done = bool(term or trun)
            if done:
                break
        peak_heights.append(max_height)
        peak_progress.append(max_progress)
        if max_progress >= RAMP_SUCCESS_PROG or max_height >= RAMP_SUCCESS_Z_RISE:
            success_count += 1
    env.close()
    return {
        "success_rate": success_count / max(episodes, 1),
        "peak_height_mean": (float(np.mean(peak_heights)) if peak_heights else 0.0),
        "peak_progress_mean": (float(np.mean(peak_progress)) if peak_progress else 0.0),
    }


def run_ramp_check_viewer(mode, target, agent, device, episodes=1, max_steps=220):
    """ランプ登坂デバッグビューワ"""
    env = build_env(mode, target, ENV_CONFIG, render_mode="human")
    obs_dim = env.observation_space.shape[0]
    history = ObsHistory(1, SEQ_LEN, obs_dim, device)
    for ep in range(episodes):
        obs, _ = env.reset()
        history.prime_single(obs)
        done = False
        for step in range(max_steps):
            with torch.no_grad():
                seq = history.get()
                out = agent.get_action_and_value(seq)
                action = out[0].cpu().numpy().reshape(-1)
            next_obs, _, term, trun, info = env.step(action)
            history.update(next_obs)
            obs = next_obs
            env.render()
            time.sleep(0.025)
            done = bool(term or trun)
            if done:
                break
    env.close()


# ============================================================================
# info dict アクセスユーティリティ
# ============================================================================


def _info_at(info, key, env_idx, default=0):
    """info dictから環境別の値を取得"""
    if key not in info:
        return default
    value = info[key]
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) <= env_idx:
            return default
        return value[env_idx]
    return value


# ============================================================================
# run_train was removed. Use `run_train_vector` instead.
# For backward-compatibility, provide a short shim that fails fast with guidance.
# This helps catch any accidental external callers early.
# ============================================================================


def run_train(*args, **kwargs):
    raise RuntimeError("run_train() has been removed — wrap single env with SingleVecWrapper " "and call run_train_vector(envs, ...) instead.")


# ============================================================================
# 訓練関数（ベクター環境）
# ============================================================================


def run_train_vector(
    envs,
    ref_env,
    agent,
    optimizer,
    model_path,
    device,
    hp,
    training_target,
    num_envs,
    wandb_run=None,
):
    """ベクター環境での訓練"""

    def _atomic_save(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        torch.save(agent.state_dict(), tmp_path)
        os.replace(tmp_path, path)

    idx = ref_env.idx
    reward_idx_cache = _build_reward_index_cache(idx)
    obs_dim = ref_env.observation_space.shape[0]
    act_dim = ref_env.action_space.shape[0]
    history = ObsHistory(num_envs, SEQ_LEN, obs_dim, device)

    # 前方LiDARインデックスと初期 agent_vz バッチ（NumPy配列）を用意
    front_lidar_indices = np.array(idx.LIDAR_FRONT_IDX, dtype=np.int32)
    agent_vz_batch = np.zeros((num_envs,), dtype=np.float32)

    obs, _ = envs.reset()
    history.prime(obs)
    # ビューワが要求されている場合は初期フレームを描画してウィンドウを起動
    if USE_VIEWER:
        try:
            if hasattr(envs, "render"):
                envs.render()
        except Exception:
            pass

    rollout_obs = torch.zeros(
        (hp["rollout_steps"], num_envs, SEQ_LEN, obs_dim),
        device=device,
    )
    rollout_actions = torch.zeros(
        (hp["rollout_steps"], num_envs, act_dim),
        device=device,
    )
    rollout_logp = torch.zeros((hp["rollout_steps"], num_envs), device=device)
    rollout_rewards = torch.zeros((hp["rollout_steps"], num_envs), device=device)
    rollout_dones = torch.zeros((hp["rollout_steps"], num_envs), device=device)
    rollout_values = torch.zeros((hp["rollout_steps"], num_envs), device=device)

    num_updates = max(1, hp["total_timesteps"] // (hp["rollout_steps"] * num_envs))
    global_step = 0
    train_start_time = time.time()
    print(f"Vector training: envs={num_envs}, updates={num_updates}, " f"model={model_path}")
    interrupted = False
    # last_ramp_eval = None  # 未使用のため削除

    # --- スケジューリング用パラメータ取得 ---
    ent_coef_init = float(hp.get("ent_coef", 0.001))
    ent_coef_final = float(hp.get("ent_coef_final", ent_coef_init * 0.5))
    ent_coef_schedule = str(hp.get("ent_coef_schedule", "linear"))
    lr_init = float(hp.get("learning_rate", 3e-4))
    lr_final = float(hp.get("learning_rate_final", lr_init * 0.5))
    lr_schedule = str(hp.get("learning_rate_schedule", "linear"))
    # action penalty (policy-side) to discourage large turn outputs (default 0.0 = disabled)
    # NOTE: disabled by default to avoid changing policy loss definition; enable explicitly in hp if desired
    action_penalty_coef = float(hp.get("action_penalty_coef", 0.0))
    action_penalty_coef = 0.0

    try:
        wall_distance_buffer = []
        info_buffer = []
        viewer_started = False
        for update in range(1, num_updates + 1):
            # --- ent_coef/learning_rateスケジューリング ---
            frac = 1.0 - (update - 1) / max(num_updates - 1, 1)
            if ent_coef_schedule == "linear":
                hp["ent_coef"] = ent_coef_final + (ent_coef_init - ent_coef_final) * frac
            if lr_schedule == "linear":
                lr_now = lr_final + (lr_init - lr_final) * frac
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr_now

            lock_evt_sum = 0
            grab_evt_sum = 0
            max_box_speed = 0.0
            max_ramp_speed = 0.0
            blocked_ramp_sum = 0
            hidden_steps = 0
            learnable_seen_steps = 0
            wall_stick_steps = 0
            wall_stick_seen_steps = 0
            entropy_sum = 0.0
            entropy_count = 0
            reward_compute_sec_sum = 0.0
            env_step_sec_sum = 0.0

            done_last = np.zeros(num_envs, dtype=np.float32)
            rewards_sum = 0.0

            info_buffer.clear()
            for t in range(hp["rollout_steps"]):
                # 各ステップの先頭で wall_stick キャッシュをクリアして
                # 前ステップの判定が次ステップに影響しないようにする
                _clear_wall_stick_state_cache()
                seq = history.get()
                rollout_obs[t] = seq

                with torch.no_grad():
                    action, logp, _, value = agent.get_action_and_value(seq)
                action_np = action.cpu().numpy()
                step_t0 = time.perf_counter()
                next_obs, base_r, term, trun, info = envs.step(action_np)
                # Viewer 更新（可能なら）およびクローズ検出
                if USE_VIEWER:
                    try:
                        if hasattr(envs, "render"):
                            envs.render()
                            time.sleep(0.025)
                        # viewer 実行中か確認（SingleVecWrapper や env を透過的に扱う）
                        v = getattr(envs, "viewer", None)
                        if v is None and hasattr(envs, "_env"):
                            v = getattr(envs._env, "viewer", None)
                        running = False
                        if v is not None:
                            is_run = getattr(v, "is_running", None)
                            if callable(is_run):
                                running = bool(is_run())
                            else:
                                running = True
                        if not running and viewer_started:
                            print("Viewer closed by user — aborting training.")
                            raise KeyboardInterrupt()
                        if running:
                            viewer_started = True
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        pass
                info_buffer.append(info)
                env_step_sec_sum += time.perf_counter() - step_t0
                done_np = np.logical_or(term, trun).astype(np.float32)
                done_last = done_np

                reward_np = np.asarray(base_r, dtype=np.float32).copy()
                # WALL_DIST インデックスを numba に渡す（観測から正規化距離を参照）
                if hasattr(idx, "WALL_DIST"):
                    try:
                        wall_dist_idx = int(idx.WALL_DIST)
                    except Exception:
                        wall_dist_idx = -1
                else:
                    wall_dist_idx = -1
                wall_repulsion_radius_value = float(WALL_REPULSION_RADIUS_DEFAULT)
                reward_t0 = time.perf_counter()
                if USE_CUSTOM_REWARD:
                    if isinstance(info, (list, tuple)):
                        pass
                    reward_np = _compute_custom_reward_batch_numba(
                        np.asarray(obs, dtype=np.float32).reshape(action_np.shape[0], -1),
                        np.asarray(next_obs, dtype=np.float32).reshape(action_np.shape[0], -1),
                        np.asarray(action_np, dtype=np.float32).reshape(action_np.shape[0], -1),
                        np.asarray(reward_np, dtype=np.float32).reshape(action_np.shape[0]),
                        bool(ref_env.target == "hider"),
                        int(reward_idx_cache["self_vel_x"]),
                        int(reward_idx_cache["self_vel_y"]),
                        np.array(reward_idx_cache["enemy_visible"], dtype=np.int32),
                        np.array(reward_idx_cache["enemy_rel_x"], dtype=np.int32),
                        np.array(reward_idx_cache["enemy_rel_y"], dtype=np.int32),
                        np.array(reward_idx_cache["enemy_being_hit"], dtype=np.int32),
                        float(RW_MOVE_CTRL_COST),
                        float(RW_MOVE_INCENTIVE),
                        float(RW_IDLE_PENALTY),
                        float(RW_WALL_AVOID_PENALTY),
                        int(wall_dist_idx),
                        wall_repulsion_radius_value,
                        float(IDLE_SPEED_THRESHOLD),
                        float(MOVE_INCENTIVE_SPEED_THRESHOLD),
                        float(MOVE_SAT_THRESHOLD),
                        float(RW_MOVE_SAT_PENALTY),
                        float(TURN_SAT_THRESHOLD),
                        float(RW_TURN_SAT_PENALTY),
                        float(RW_TURN_INCENTIVE),
                        float(RW_STILL_SPEED_THRESHOLD),
                        float(RW_WALL_STICK_PENALTY),
                        float(AGENT_RADIUS),  # AGENT_RADIUS
                        float(RW_EVASION_BONUS),
                        agent_vz_batch,
                    )
                reward_compute_sec_sum += time.perf_counter() - reward_t0

                global_step += num_envs
                rollout_actions[t] = action
                rollout_logp[t] = logp
                rollout_rewards[t] = torch.as_tensor(reward_np, device=device)
                rollout_dones[t] = torch.as_tensor(done_np, device=device)
                rollout_values[t] = value.view(-1)
                # --- 詳細な統計情報の加算 ---
                # infoの型チェックはループ外で一度だけ
                info_is_list = isinstance(info, (list, tuple))
                info_is_dict = isinstance(info, dict)
                # infoはlist/tupleまたはdict
                for i in range(num_envs):
                    # info_iの取得
                    if info_is_list:
                        info_i = info[i] if len(info) > i else None
                    elif info_is_dict:
                        info_i = info
                    else:
                        info_i = None
                    if info_i is None:
                        continue
                    lock_evt_sum += int(_info_at(info, "lock_event", i, 0))
                    grab_evt_sum += int(_info_at(info, "grab_event", i, 0))
                    max_box_speed = max(max_box_speed, float(_info_at(info, "dbg_max_box_speed", i, 0.0)))
                    max_ramp_speed = max(max_ramp_speed, float(_info_at(info, "dbg_max_ramp_speed", i, 0.0)))
                    blocked_ramp_sum += int(_info_at(info, "dbg_blocked_ramp_count", i, 0))
                    is_detected = bool(_info_at(info, "is_detected", i, False))
                    if not is_detected:
                        hidden_steps += 1
                    seen_learnable = bool(_info_at(info, "dbg_learnable_hider_seen", i, is_detected))
                    if seen_learnable:
                        learnable_seen_steps += 1
                    # 壁張り付き判定: info や観測を優先的に参照するユーティリティを使う
                    next_speed = math.sqrt(float(next_obs[i, reward_idx_cache["self_vel_x"]]) ** 2 + float(next_obs[i, reward_idx_cache["self_vel_y"]]) ** 2)
                    is_wall_stick = _is_wall_stick_state(
                        next_obs[i],
                        idx,
                        reward_idx_cache=reward_idx_cache,
                        speed=next_speed,
                        info=info_i,
                        agent_idx=i,
                    )
                    if update % hp["log_interval"] == 0 and i == 0:
                        pass
                    if is_wall_stick:
                        wall_stick_steps += 1
                        if seen_learnable:
                            wall_stick_seen_steps += 1
                    rewards_sum += float(reward_np[i])

                for i in range(num_envs):
                    if done_np[i] > 0.5:
                        obs1d = np.asarray(next_obs[i]).flatten()
                        if obs1d.size == history.obs_dim:
                            # 個別reset: サブ環境iだけresetし、その観測だけをnext_obs[i]に代入
                            if hasattr(envs, "reset_at"):
                                next_obs_i, _ = envs.reset_at(i)
                            else:
                                # reset_maskで個別リセット
                                mask = np.zeros(num_envs, dtype=bool)
                                mask[i] = True
                                next_obses, _ = envs.reset(options={"reset_mask": mask})
                                next_obs_i = next_obses[i]
                            next_obs[i] = next_obs_i
                            history.reset_env(i, next_obs[i])
                        else:
                            print(f"[warn] next_obs[{i}] size {obs1d.size} != obs_dim {history.obs_dim}, skip reset")
                obs = next_obs

            with torch.no_grad():
                next_value = agent.get_value(history.get()).view(num_envs)

            advantages = torch.zeros((hp["rollout_steps"], num_envs), device=device)
            lastgaelam = torch.zeros(num_envs, device=device)
            done_last_t = torch.as_tensor(done_last, device=device)
            for t in reversed(range(hp["rollout_steps"])):
                if t == hp["rollout_steps"] - 1:
                    next_nonterminal = 1.0 - done_last_t
                    next_vals = next_value
                else:
                    next_nonterminal = 1.0 - rollout_dones[t + 1]
                    next_vals = rollout_values[t + 1]
                delta = rollout_rewards[t] + hp["gamma"] * next_vals * next_nonterminal - rollout_values[t]
                lastgaelam = delta + hp["gamma"] * hp["gae_lambda"] * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + rollout_values

            b_obs = rollout_obs.reshape(-1, SEQ_LEN, obs_dim)
            b_actions = rollout_actions.reshape(-1, act_dim)
            b_logp = rollout_logp.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_ret = returns.reshape(-1)

            batch_size = hp["rollout_steps"] * num_envs
            inds = np.arange(batch_size)
            for _ in range(hp["update_epochs"]):
                np.random.shuffle(inds)
                for start in range(0, batch_size, hp["minibatch_size"]):
                    mb_idx = inds[start : start + hp["minibatch_size"]]
                    _, new_logp, entropy, new_value = agent.get_action_and_value(
                        b_obs[mb_idx],
                        b_actions[mb_idx],
                    )
                    logratio = new_logp - b_logp[mb_idx]
                    ratio = logratio.exp()

                    mb_adv = b_adv[mb_idx]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(
                        ratio,
                        1.0 - hp["clip_coef"],
                        1.0 + hp["clip_coef"],
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    v_loss = 0.5 * ((new_value.view(-1) - b_ret[mb_idx]) ** 2).mean()
                    ent_loss = entropy.mean()
                    loss = pg_loss + hp["vf_coef"] * v_loss - hp["ent_coef"] * ent_loss
                    # policy-side penalty on predicted turn magnitude (deterministic mean)
                    if action_penalty_coef and action_penalty_coef > 0.0:
                        try:
                            action_mean, _, _, _ = agent.get_deterministic_action_and_value(b_obs[mb_idx])
                            # action_mean shape: (batch, act_dim)
                            turn_mean = action_mean[:, 1]
                            action_penalty = action_penalty_coef * (turn_mean * turn_mean).mean()
                            loss = loss + action_penalty
                        except Exception:
                            pass
                    entropy_sum += float(ent_loss.detach().cpu().item())
                    entropy_count += 1

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), hp["max_grad_norm"])
                    optimizer.step()

            elapsed = max(time.time() - train_start_time, 1e-6)
            sps = int(global_step / elapsed)
            hide_rate = hidden_steps / max(hp["rollout_steps"] * num_envs, 1)
            learnable_seen_rate = learnable_seen_steps / max(hp["rollout_steps"] * num_envs, 1)
            wall_stick_ratio = wall_stick_steps / max(hp["rollout_steps"] * num_envs, 1)
            wall_stick_seen_ratio = wall_stick_seen_steps / max(wall_stick_steps, 1)
            avg_reward = rewards_sum / (hp["rollout_steps"] * num_envs)
            avg_blocked_ramp = blocked_ramp_sum / (hp["rollout_steps"] * num_envs)
            avg_entropy = entropy_sum / max(entropy_count, 1)
            reward_compute_ms_per_step = 1000.0 * reward_compute_sec_sum / max(hp["rollout_steps"] * num_envs, 1)
            env_step_ms_per_step = 1000.0 * env_step_sec_sum / max(hp["rollout_steps"] * num_envs, 1)
            reward_time_ratio = reward_compute_sec_sum / max(env_step_sec_sum, 1e-9)

            ramp_eval = None
            if RAMP_CHECK_ENABLED and RAMP_CHECK_INTERVAL > 0 and update % RAMP_CHECK_INTERVAL == 0:
                ramp_eval = evaluate_fixed_ramp_climb(
                    MODE,
                    training_target,
                    agent,
                    device,
                    episodes=RAMP_CHECK_EPISODES,
                    max_steps=RAMP_CHECK_STEPS,
                )
                # last_ramp_eval = ramp_eval  # 未使用のため削除

            if wandb_run is not None:
                # diagnostic: average actor std and average turn magnitude over recent rollout
                try:
                    actor_std_mean = float(torch.exp(agent.actor_logstd).mean().detach().cpu().item())
                except Exception:
                    try:
                        actor_std_mean = float(torch.exp(agent.module.actor_logstd).mean().detach().cpu().item())
                    except Exception:
                        actor_std_mean = 0.0
                try:
                    avg_turn = float(torch.mean(torch.abs(rollout_actions[:, :, 1])).detach().cpu().item())
                except Exception:
                    try:
                        avg_turn = float(np.mean(np.abs(rollout_actions[:, :, 1].cpu().numpy())))
                    except Exception:
                        avg_turn = 0.0

                payload = {
                    "global_step": global_step,
                    "train/update": update,
                    "train/sps": sps,
                    "train/hide_rate": hide_rate,
                    "train/avg_reward": avg_reward,
                    "train/entropy": avg_entropy,
                    "train/actor_std_mean": actor_std_mean,
                    "train/avg_turn": avg_turn,
                    "train/learnable_seen_rate": learnable_seen_rate,
                    "train/wall_stick_ratio": wall_stick_ratio,
                    "train/wall_stick_seen_ratio": wall_stick_seen_ratio,
                    "perf/reward_compute_ms_per_step": reward_compute_ms_per_step,
                    "perf/env_step_ms_per_step": env_step_ms_per_step,
                    "perf/reward_to_env_time_ratio": reward_time_ratio,
                    "env/lock_events": lock_evt_sum,
                    "env/grab_events": grab_evt_sum,
                    "env/max_box_speed": max_box_speed,
                    "env/max_ramp_speed": max_ramp_speed,
                    "env/avg_blocked_ramp": avg_blocked_ramp,
                    "system/num_envs": num_envs,
                }
                if ramp_eval is not None:
                    payload.update(
                        {
                            "eval/ramp_climb_success_rate": ramp_eval["success_rate"],
                            "eval/ramp_peak_height_mean": ramp_eval["peak_height_mean"],
                            "eval/ramp_peak_progress_mean": ramp_eval["peak_progress_mean"],
                        }
                    )
                # aggregate gaze debug metrics from info_buffer (mean & max)
                gaze_count = 0
                gaze_sum = 0.0
                gaze_max = 0.0
                gaze_dist_sum = 0.0
                gaze_dist_max = 0.0
                for info in info_buffer:

                    def get_info_i(info_obj, j):
                        if isinstance(info_obj, (list, tuple)):
                            return info_obj[j] if len(info_obj) > j else None
                        if isinstance(info_obj, dict):
                            return info_obj
                        return None

                    # support both list-of-dicts and dict-of-arrays
                    for i in range(num_envs):
                        info_i = get_info_i(info, i)
                        if info_i is None:
                            continue

                        if isinstance(info, dict):
                            # dict-of-arrays: use None default to detect missing
                            raw_v = _info_at(info, "dbg_seek_gaze_cos_front_max", i, None)
                            raw_d = _info_at(info, "dbg_seek_gaze_cos_front_dist_max", i, None)
                            if raw_v is None and raw_d is None:
                                continue
                            val = float(raw_v) if raw_v is not None else 0.0
                            val2 = float(raw_d) if raw_d is not None else 0.0
                        else:
                            # list-of-dicts: check key presence per-env
                            if not (isinstance(info_i, dict) and ("dbg_seek_gaze_cos_front_max" in info_i or "dbg_seek_gaze_cos_front_dist_max" in info_i)):
                                continue
                            val = float(info_i.get("dbg_seek_gaze_cos_front_max", 0.0))
                            val2 = float(info_i.get("dbg_seek_gaze_cos_front_dist_max", 0.0))

                        # Count only non-zero measurements. Many envs produce
                        # dict-of-arrays with zeros for "no measurement",
                        # so treat strictly-zero entries as absent here to
                        # avoid counting every (env,step) pair.
                        if (val != 0.0) or (val2 != 0.0):
                            gaze_sum += val
                            gaze_dist_sum += val2
                            gaze_max = max(gaze_max, val)
                            gaze_dist_max = max(gaze_dist_max, val2)
                            gaze_count += 1
                # Always include gaze keys in payload; use 0.0 for means when no samples
                if gaze_count > 0:
                    gaze_mean = gaze_sum / float(gaze_count)
                    gaze_dist_mean = gaze_dist_sum / float(gaze_count)
                else:
                    gaze_mean = 0.0
                    gaze_dist_mean = 0.0
                payload.update(
                    {
                        "env/dbg_seek_gaze_cos_front_max_mean": gaze_mean,
                        "env/gaze_count": gaze_count,
                        "env/dbg_seek_gaze_cos_front_max": gaze_max,
                        "env/dbg_seek_gaze_cos_front_dist_max_mean": gaze_dist_mean,
                        "env/dbg_seek_gaze_cos_front_dist_max": gaze_dist_max,
                    }
                )
                # Debug: show gaze_count and which keys are being logged
                # try:
                #    print(f"[GazeDebug] Step={global_step} gaze_count={gaze_count} payload_keys={list(payload.keys())}")
                # except Exception:
                #    pass
                wandb_run.log(payload, step=global_step)

            if update % hp["log_interval"] == 0:
                print(f"Upd {update}/{num_updates} Step={global_step} SPS={sps} HideRate={hide_rate:.2f} WallStick={wall_stick_ratio:.2f} AvgR={avg_reward:.3f}")

            # wall_distance統計バッファ（ローカル変数で管理）
            for info in info_buffer:
                if isinstance(info, (list, tuple)):
                    for i in range(num_envs):
                        info_i = info[i] if len(info) > i else None
                        if info_i and "wall_distance" in info_i:
                            wd = info_i["wall_distance"]
                            if isinstance(wd, (list, tuple, np.ndarray)):
                                for v in wd:
                                    wall_distance_buffer.append(float(v))
                            else:
                                wall_distance_buffer.append(float(wd))
                elif isinstance(info, dict):
                    if "wall_distance" in info:
                        wd = info["wall_distance"]
                        if isinstance(wd, (list, tuple, np.ndarray)):
                            for v in wd:
                                wall_distance_buffer.append(float(v))
                        else:
                            wall_distance_buffer.append(float(wd))

            if update % hp["log_interval"] == 0 and wall_distance_buffer:
                arr = np.array(wall_distance_buffer, dtype=np.float32)
                print(f"[WallDist] step={global_step} mean={np.mean(arr):.3f} min={np.min(arr):.3f} max={np.max(arr):.3f}")
                wall_distance_buffer.clear()

            if update % hp["save_interval"] == 0:
                _atomic_save(model_path)

    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted. Saving checkpoint...")
        raise
    except WORKER_SHUTDOWN_ERRORS as exc:
        interrupted = True
        print(f"\nVector worker terminated ({exc.__class__.__name__}). " "Saving checkpoint and exiting cleanly...")
    finally:
        _atomic_save(model_path)
        if interrupted:
            print(f"Interrupted checkpoint saved: {model_path}")
        else:
            print(f"Saved model: {model_path}")


# ============================================================================
# デバッグ・再生機能
# ============================================================================


def run_debug_or_playback(env, agent, device, model_loaded):
    """デバッグモード・再生モード"""
    idx = env.idx
    reward_idx_cache = _build_reward_index_cache(idx)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    history = ObsHistory(1, SEQ_LEN, obs_dim, device)

    target_npc = None
    if NPC_ONLY_DEBUG:
        target_npc = RuleBasedSeeker() if env.target == "seeker" else RuleBasedHider()

    print("--- Start Simulation ---")
    print(f"Physics Step: {env.model.opt.timestep * 5}s")

    log_interval = hp["log_interval"]
    wall_dist_interval = 1000
    # `play_episodes` はデバッグ/再生用プロファイルで定義される想定。
    # 定義がないプロファイルから呼ばれた場合は安全なデフォルトを使う。
    if "play_episodes" in hp:
        num_episodes = int(hp["play_episodes"])
    else:
        num_episodes = int(hp.get("PLAY_EPISODES", 100))
        print(f"Warning: 'play_episodes' not found in profile; defaulting to {num_episodes} episodes.")
    episode_rewards = []
    hide_rates = []
    wall_distance_buffer = []
    lock_events_list = []
    wall_stick_list = []
    info_buffer = []
    global_step = 0
    debug_start_time = time.time()
    steps_per_episode = None
    num_updates = max(1, num_episodes // log_interval)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        # If deterministic policy replay is requested via TOML (model_policy_deterministic),
        # seed RNGs at episode start so
        # sampled actions become reproducible.
        if MODEL_POLICY_DETERMINISTIC:
            seed = 123456789
            torch.manual_seed(seed)
            np.random.seed(seed)
        history.prime_single(obs)
        done = False
        ep_reward = 0.0
        step_count = 0
        lock_events = 0
        wall_stick = 0.0
        wall_dist = []
        # per-step debug logging (first N steps)
        debug_log_n = int(hp.get("dbg_log_interval_steps", 200))
        step_logs = []

        viewer_started = False
        while not done:
            # ビューワの存在・稼働をチェック。閉じられている場合はプロセスを終了する。
            if USE_VIEWER:
                try:
                    viewer_running = bool(env.viewer and getattr(env.viewer, "is_running", lambda: False)())
                except Exception:
                    viewer_running = False
                if not viewer_running and not viewer_started:
                    # 未起動なら起動を試みる
                    try:
                        env.render()
                        viewer_started = True
                    except Exception:
                        viewer_started = False
                elif not viewer_running and viewer_started:
                    print("Viewer closed by user — exiting playback.")
                    return
            if NPC_ONLY_DEBUG and target_npc is not None:
                action = target_npc.get_action(obs, idx)
            elif model_loaded:
                with torch.no_grad():
                    seq = history.get()
                    if MODEL_POLICY_DETERMINISTIC and hasattr(agent, "get_deterministic_action_and_value"):
                        out = agent.get_deterministic_action_and_value(seq)
                    else:
                        out = agent.get_action_and_value(seq)
                    action = out[0].cpu().numpy().reshape(-1)
            else:
                if env.target == "hider" and getattr(env, "override_learnable_policy", False):
                    action = env.npcs[env.learnable_agent_key].get_action(obs, idx)
                else:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)

            next_obs, base_r, term, trun, info = env.step(action)
            reward = compute_custom_reward(obs, next_obs, action, base_r, idx, env.target, info, reward_idx_cache=reward_idx_cache) if USE_CUSTOM_REWARD else base_r
            history.update(next_obs)
            ep_reward += reward
            step_count += 1
            lock_events += int(info.get("lock_event", 0))
            wall_stick += float(info.get("wall_stick", 0.0))
            if "wall_distance" in info:
                wd = info["wall_distance"]
                if isinstance(wd, (list, tuple, np.ndarray)):
                    wall_dist.extend([float(v) for v in wd])
                else:
                    wall_dist.append(float(wd))
            info_buffer.append(info)

            # collect per-step debug info for the first debug_log_n steps
            if step_count <= debug_log_n:
                try:
                    vx = float(next_obs[reward_idx_cache["self_vel_x"]])
                    vy = float(next_obs[reward_idx_cache["self_vel_y"]])
                except Exception:
                    vx = 0.0
                    vy = 0.0
                speed = math.sqrt(vx * vx + vy * vy)
                # wall_distance (prefer scalar)
                wall_d = None
                if info is not None and "wall_distance" in info:
                    wd = info["wall_distance"]
                    if isinstance(wd, (list, tuple, np.ndarray)):
                        wall_d = float(wd[0])
                    else:
                        wall_d = float(wd)
                # enemy visibility and nearest distance
                enemy_visible = False
                nearest_dist = None
                ev_idxs = reward_idx_cache.get("enemy_visible", None)
                rel_x_idxs = reward_idx_cache.get("enemy_rel_x", None)
                rel_y_idxs = reward_idx_cache.get("enemy_rel_y", None)
                if ev_idxs is not None and rel_x_idxs is not None and rel_y_idxs is not None:
                    for j in range(len(ev_idxs)):
                        try:
                            idx_vis = int(ev_idxs[j])
                            if float(next_obs[idx_vis]) > 0.5:
                                enemy_visible = True
                                dx = float(next_obs[int(rel_x_idxs[j])])
                                dy = float(next_obs[int(rel_y_idxs[j])])
                                d = math.sqrt(dx * dx + dy * dy) * P_SCALE
                                if nearest_dist is None or d < nearest_dist:
                                    nearest_dist = d
                        except Exception:
                            continue
                # gaze value (if present in info)
                gaze_v = None
                if info is not None:
                    gaze_v = info.get("dbg_seek_gaze_cos_front_max", None)

                # also sample both deterministic and stochastic actions for comparison when model_loaded
                sampled_a = None
                det_a = None
                if model_loaded:
                    with torch.no_grad():
                        seq = history.get()
                        try:
                            sa_out = agent.get_action_and_value(seq)
                            sampled_a = sa_out[0].cpu().numpy().reshape(-1)
                        except Exception:
                            sampled_a = None
                        if hasattr(agent, "get_deterministic_action_and_value"):
                            try:
                                da_out = agent.get_deterministic_action_and_value(seq)
                                det_a = da_out[0].cpu().numpy().reshape(-1)
                            except Exception:
                                det_a = None

                step_logs.append(
                    {
                        "step": step_count,
                        "action": action.tolist() if hasattr(action, "tolist") else np.asarray(action).tolist(),
                        "obs": obs.tolist() if (hasattr(obs, "tolist") or isinstance(obs, (list, tuple, np.ndarray))) else obs,
                        "next_obs": next_obs.tolist() if (hasattr(next_obs, "tolist") or isinstance(next_obs, (list, tuple, np.ndarray))) else next_obs,
                        "sampled_action": sampled_a.tolist() if isinstance(sampled_a, np.ndarray) else sampled_a,
                        "det_action": det_a.tolist() if isinstance(det_a, np.ndarray) else det_a,
                        "vx": vx,
                        "vy": vy,
                        "speed": speed,
                        "wall_distance": wall_d,
                        "enemy_visible": enemy_visible,
                        "nearest_enemy_dist": nearest_dist,
                        "gaze": gaze_v,
                        "base_reward": float(base_r),
                        "reward": float(reward),
                        "custom_component": float(reward - base_r),
                    }
                )

            if USE_VIEWER:
                try:
                    env.render()
                    time.sleep(0.025)
                    viewer_started = True
                except Exception:
                    pass
            obs = next_obs
            done = bool(term or trun)

        episode_rewards.append(ep_reward)
        # Print per-step debug summary if we collected logs
        if step_logs:
            try:
                arr_forward = [s["action"][0] for s in step_logs if s.get("action")]
                arr_speed = [s["speed"] for s in step_logs]
                arr_custom = [s["custom_component"] for s in step_logs]
                arr_det_forward = [s["det_action"][0] for s in step_logs if s.get("det_action")]
                arr_samp_forward = [s["sampled_action"][0] for s in step_logs if s.get("sampled_action")]
                vis_count = sum(1 for s in step_logs if s.get("enemy_visible"))
                nearest_list = [s["nearest_enemy_dist"] for s in step_logs if s.get("nearest_enemy_dist") is not None]
                import numpy as _np

                forward_mean = float(_np.mean(arr_forward)) if arr_forward else 0.0
                det_forward_mean = float(_np.mean(arr_det_forward)) if arr_det_forward else float("nan")
                samp_forward_mean = float(_np.mean(arr_samp_forward)) if arr_samp_forward else float("nan")
                speed_mean = float(_np.mean(arr_speed)) if arr_speed else 0.0
                custom_mean = float(_np.mean(arr_custom)) if arr_custom else 0.0
                nearest_mean = float(_np.mean(nearest_list)) if nearest_list else float("nan")
                print(
                    f"[DebugLog] ep={episode} logged_steps={len(step_logs)} forward_mean={forward_mean:.3f} "
                    f"speed_mean={speed_mean:.3f} enemy_vis={vis_count}/{len(step_logs)} "
                    f"nearest_mean={nearest_mean:.3f} custom_mean={custom_mean:.4f} "
                    f"det_forward_mean={det_forward_mean:.3f} samp_forward_mean={samp_forward_mean:.3f}"
                )
                try:

                    def _to_primitive(v):
                        if isinstance(v, (np.generic,)):
                            return v.item()
                        if hasattr(v, "tolist"):
                            try:
                                return v.tolist()
                            except Exception:
                                pass
                        return v

                    fname = f"debug_step_logs_ep{episode}.jsonl"
                    with open(fname, "w") as jf:
                        for s in step_logs:
                            safe = {k: _to_primitive(v) for k, v in (s.items() if isinstance(s, dict) else {})}
                            jf.write(json.dumps(safe) + "\n")
                except Exception:
                    pass
            except Exception:
                pass
        hide_rates.append(float(info.get("hide_rate", 0.0)))
        lock_events_list.append(lock_events)
        wall_stick_list.append(wall_stick / max(step_count, 1))
        wall_distance_buffer.extend(wall_dist)
        global_step += step_count

        if steps_per_episode is None:
            steps_per_episode = step_count

        if (episode + 1) % log_interval == 0:
            elapsed = time.time() - debug_start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_hide = np.mean(hide_rates[-log_interval:])
            avg_wall_stick = np.mean(wall_stick_list[-log_interval:])
            print(f"Upd {(episode + 1) // log_interval}/{num_updates} Step={global_step} SPS={sps} HideRate={avg_hide:.2f} WallStick={avg_wall_stick:.2f} AvgR={avg_reward:.3f}")

        if global_step % wall_dist_interval == 0 and wall_distance_buffer:
            arr = np.array(wall_distance_buffer, dtype=np.float32)
            print(f"[WallDist] step={global_step} mean={np.mean(arr):.3f} min={np.min(arr):.3f} max={np.max(arr):.3f}")
            wall_distance_buffer.clear()


# ============================================================================
# wandb 初期化
# ============================================================================


def init_wandb_run(hp, model_path, runtime_target):
    """wandbを初期化"""
    if wandb is None or not WANDB_ENABLED:
        return None
    import socket

    run_name = WANDB_RUN_NAME or f"{runtime_target}_{os.path.basename(model_path)}_{socket.gethostname()}"
    config_dict = dict(hp)
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY or None,
            name=run_name,
            config=config_dict,
            mode=WANDB_MODE,
            dir=WANDB_CODE_ROOT,
            save_code=WANDB_LOG_CODE,
            reinit=True,
        )
        return run
    except Exception as exc:
        print(f"wandb.init failed: {exc}")
        return None


# ============================================================================
# プロファイル比較
# ============================================================================


def run_profile_comparison(profiles):
    """複数プロファイルを連続実行"""
    import subprocess

    names = [str(p).strip().lower() for p in profiles if str(p).strip()]
    if not names:
        print("No profiles specified for comparison.")
        return 2
    invalid = [p for p in names if p not in AVAILABLE_RUNTIME_PROFILES]
    if invalid:
        print(f"Invalid profiles: {invalid}")
        print(f"Available profiles: {AVAILABLE_RUNTIME_PROFILES}")
        return 2

    script_path = os.path.abspath(__file__)
    for i, profile in enumerate(names, start=1):
        print(f"[Compare {i}/{len(names)}] profile={profile}")
        cmd = ["uv", "run", "mjpython", script_path, "--profile", profile]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Profile '{profile}' failed with code {result.returncode}")
            return result.returncode
    print("Comparison run completed.")
    return 0


# ============================================================================
# メイン実行関数
# ============================================================================


def run():
    """メイン実行関数"""
    _parse_cli_args()
    runtime_target = _resolve_runtime_target()
    print("Initializing Environment...")
    print(f"Run profile: {RUN_PROFILE} (source: {RUN_PROFILE_SOURCE})")
    print(
        "Resolved flags: "
        f"TRAIN_MODE={TRAIN_MODE}, "
        f"USE_VIEWER={USE_VIEWER}, "
        f"NPC_ONLY_DEBUG={NPC_ONLY_DEBUG}, "
        f"RUNTIME_TARGET={runtime_target}, "
        f"DEBUG_HIDER_POLICY={DEBUG_HIDER_POLICY}, "
        f"DEBUG_SEEKER_POLICY={DEBUG_SEEKER_POLICY}, "
        f"MODEL_POLICY_DETERMINISTIC={MODEL_POLICY_DETERMINISTIC}, "
        f"DEBUG_SYMMETRIC_HIDER_POLICY={DEBUG_SYMMETRIC_HIDER_POLICY}, "
        f"USE_CUSTOM_REWARD={USE_CUSTOM_REWARD}, "
        f"HPARAMS_CONFIG_PATH={HPARAMS_CONFIG_PATH}, "
        f"RESUME_TRAINING={RESUME_TRAINING}, "
        f"RESET_MODEL_ON_TRAIN={RESET_MODEL_ON_TRAIN}, "
        f"TRAIN_OTHER_HIDER_POLICY={TRAIN_OTHER_HIDER_POLICY}, "
        f"TRAIN_OTHER_SEEKER_POLICY={TRAIN_OTHER_SEEKER_POLICY}, "
        f"NUM_ENVS={NUM_ENVS}, "
        f"RAMP_CHECK_VIEWER={RAMP_CHECK_VIEWER}, "
        f"RAMP_CHECK_FORCE_UPHILL={RAMP_CHECK_FORCE_UPHILL}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if USE_VIEWER:
        ENV_CONFIG["num_envs"] = 1

    def make_env(render_mode=None):
        return build_env(MODE, runtime_target, ENV_CONFIG, render_mode)

    env = None
    vec_envs = None
    ref_env = None
    try:
        if TRAIN_MODE and NUM_ENVS > 1 and not USE_VIEWER:
            env_config_snapshot = dict(ENV_CONFIG)
            vec_envs = gym.vector.AsyncVectorEnv(
                [
                    partial(
                        build_env,
                        MODE,
                        runtime_target,
                        env_config_snapshot,
                        None,
                    )
                    for _ in range(NUM_ENVS)
                ]
            )
            ref_env = make_env(render_mode=None)
        else:
            if TRAIN_MODE and NUM_ENVS > 1 and USE_VIEWER:
                print("Viewer有効時は並列実行を無効化し、" "単一環境でレンダリングします。")
            env = make_env(render_mode=("human" if USE_VIEWER else None))
            # ビューア使用時は env にデバッグ出力を有効化して
            # ポリシーソース等の情報を観察しやすくする
            env = env
            try:
                # Respect explicit debug flags from config; do not force debug on when viewer is enabled
                env.debug_mode = bool(DEBUG_MODE)
                env.debug_console = bool(DEBUG_CONSOLE)
            except Exception:
                pass
            ref_env = env
    except Exception:
        print("\nFailed to initialize environment:")
        traceback.print_exc()
        return

    obs_dim = ref_env.observation_space.shape[0]
    act_dim = ref_env.action_space.shape[0]
    print(f"Env signature: {env_signature(ENV_CONFIG)}")
    print(f"obs_dim={obs_dim}, act_dim={act_dim}, seq_len={SEQ_LEN}")

    model_path = model_path_for_config(runtime_target, ENV_CONFIG)
    print(
        f"HP[{runtime_target}]: T={hp['total_timesteps']} R={hp['rollout_steps']} "
        f"MB={hp['minibatch_size']} LR={hp['learning_rate']:.2e} "
        f"ENT={hp['ent_coef']:.2e} CLIP={hp['clip_coef']:.2f} "
        f"EPOCHS={hp['update_epochs']}"
    )

    agent = AgentV2(obs_dim, act_dim, HIDDEN_DIM, SEQ_LEN).to(device)
    wandb_run = init_wandb_run(hp, model_path, runtime_target) if TRAIN_MODE else None
    model_loaded = False
    should_load_model = True

    if TRAIN_MODE:
        should_load_model = bool(RESUME_TRAINING)
        if RESET_MODEL_ON_TRAIN and os.path.exists(model_path):
            try:
                os.remove(model_path)
                print(f"Removed existing checkpoint: {model_path}")
            except Exception as exc:
                print(f"Failed to remove checkpoint ({exc})")

    if should_load_model and os.path.exists(model_path):
        try:
            agent.load_state_dict(torch.load(model_path, map_location=device))
            model_loaded = True
            print(f"Loaded model: {model_path}")
        except Exception as exc:
            print(f"Model load skipped ({exc})")
    elif TRAIN_MODE and (not RESUME_TRAINING):
        print("Training from scratch (resume disabled).")

    primary_state_dict = _snapshot_state_dict_cpu(agent) if model_loaded else None
    configure_team_policy_modes(
        env,
        vec_envs,
        ref_env,
        runtime_target,
        primary_state_dict,
    )

    try:
        if RAMP_CHECK_VIEWER:
            run_ramp_check_viewer(
                MODE,
                runtime_target,
                agent,
                device,
                episodes=RAMP_CHECK_EPISODES,
                max_steps=RAMP_CHECK_STEPS,
            )
            return
        if TRAIN_MODE:
            optimizer = optim.Adam(
                agent.parameters(),
                lr=hp["learning_rate"],
                eps=1e-5,
            )
            if vec_envs is not None:
                run_train_vector(
                    vec_envs,
                    ref_env,
                    agent,
                    optimizer,
                    model_path,
                    device,
                    hp,
                    runtime_target,
                    NUM_ENVS,
                    wandb_run,
                )
            else:
                # 単一環境は SingleVecWrapper で 1 要素ベクトルに適合させて
                # 共通のベクタートレーナに委譲する（バグ温床を減らすため）
                wrapper = SingleVecWrapper(env)
                vec_envs = wrapper
                run_train_vector(
                    wrapper,
                    ref_env,
                    agent,
                    optimizer,
                    model_path,
                    device,
                    hp,
                    runtime_target,
                    num_envs=1,
                    wandb_run=wandb_run,
                )
        else:
            run_debug_or_playback(env, agent, device, model_loaded)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        print("\nUnexpected error:")
        traceback.print_exc()
        if USE_VIEWER:
            time.sleep(3)
    finally:
        if vec_envs is not None:
            safe_close_vector_env(vec_envs)
            vec_envs = None
        if env is not None:
            env.close()
        if ref_env is not None and ref_env is not env:
            ref_env.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    if CLI_COMPARE_PROFILES:
        raise SystemExit(run_profile_comparison(CLI_COMPARE_PROFILES))
    run()
