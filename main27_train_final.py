# main27_train_final.py
import sys
import os
import time
import argparse
import traceback
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from functools import partial
from numba import njit, prange

try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # pip install toml



# wandbはtry-import
try:
    import wandb
except Exception as exc:
    print(f"Exception in wandb import: {exc}")
    wandb = None


# sys.path拡張（src配下の独自モジュール用）
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "src")
    )
)


# --- 必要な独自モジュール ---
from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider


# --- ここから定数・関数・本体 ---

# --- ポリシーモード定数 ---
POLICY_MODEL_IF_AVAILABLE = "model_if_available"
POLICY_RULE = "rule"


# --- ユーティリティ関数 ---
def init_wandb_run(hp, model_path, runtime_target):
    """
    wandbのランを初期化して返す（暫定実装）。必要に応じて設定を拡張してください。
    """
    if wandb is None:
        return None
    run_name = WANDB_RUN_NAME or f"{runtime_target}_{os.path.basename(model_path)}"
    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY or None,
        name=run_name,
        config=hp,
        mode=WANDB_MODE,
        dir=WANDB_CODE_ROOT,
        save_code=WANDB_LOG_CODE,
        reinit=True,
    )


def select_hparams(env_config, runtime_target):
    """
    環境設定とターゲットに応じたハイパーパラメータ辞書を返す（暫定実装）
    必要に応じて詳細なロジックに拡張してください。
    """
    # ここではENV_CONFIGとTRAIN_MODE等から最低限のパラメータを組み立てる例
    return {
        "total_timesteps": TOTAL_TIMESTEPS,
        "rollout_steps": ROLLOUT_STEPS,
        "update_epochs": UPDATE_EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_coef": CLIP_COEF,
        "vf_coef": VF_COEF,
        "ent_coef": ENT_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "log_interval": LOG_INTERVAL,
        "save_interval": SAVE_INTERVAL,
    }


CONFIG_PATH = "configs/hparams_main27.toml"


CLI_EXAMPLES = (
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

# --- 未定義関数・変数のダミー実装（必要に応じて本実装に差し替えてください） ---
def compute_custom_reward(obs, action, base_reward, idx, target, info, reward_idx_cache=None):
    """
    カスタム報酬関数（ユーザー定義版）。
    obs: 観測（1次元np.array）
    action: 行動（1次元np.array）
    base_reward: 環境からの基本報酬
    idx: env.idx
    target: "hider" or "seeker"
    info: info dict
    reward_idx_cache: インデックスキャッシュ（省略可）
    """
    cache = (
        reward_idx_cache
        if reward_idx_cache is not None
        else _build_reward_index_cache(idx)
    )
    move = float(action[0]) if len(action) > 0 else 0.0
    turn = float(action[1]) if len(action) > 1 else 0.0
    vel_x = float(obs[cache["self_vel_x"]])
    vel_y = float(obs[cache["self_vel_y"]])
    speed = math.sqrt(vel_x * vel_x + vel_y * vel_y)
    control_cost = -RW_MOVE_CTRL_COST * (move * move + turn * turn)
    bonus = 0.0
    AGENT_RADIUS = 0.4
    WALL_NEAR = AGENT_RADIUS + 0.2  # 0.6
    WALL_SAFE = AGENT_RADIUS + 0.5  # 0.9

    if target == "hider":
        if speed < 0.1:
            bonus -= RW_IDLE_PENALTY
        elif speed > 0.3:
            bonus += RW_MOVE_INCENTIVE
    else:
        if speed < 0.1:
            bonus -= RW_IDLE_PENALTY
        for j in range(len(cache["enemy_visible"])):
            if obs[cache["enemy_visible"][j]] > 0.5:
                dx = float(obs[cache["enemy_rel_x"][j]])
                dy = float(obs[cache["enemy_rel_y"][j]])
                dist = math.sqrt(dx * dx + dy * dy)
                bonus += 0.05 / (dist + 1.0)

    # --- wall_distanceベースの壁ペナルティ（hider/seeker共通） ---
    if info is not None and 'wall_distance' in info:
        wall_distance = info['wall_distance']
        if isinstance(wall_distance, (list, tuple, np.ndarray)):
            wall_distance = float(np.min(wall_distance))
        else:
            wall_distance = float(wall_distance)
        if wall_distance < WALL_NEAR:
            bonus -= RW_WALL_AVOID_PENALTY * (
                1.0 - np.clip(wall_distance / WALL_NEAR, 0, 1)
            )
        elif wall_distance < WALL_SAFE:
            bonus -= (
                RW_WALL_AVOID_PENALTY
                * (WALL_SAFE - wall_distance)
                / (WALL_SAFE - WALL_NEAR)
                / 2.0
            )
    return float(base_reward) + bonus + control_cost

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
    from collections import deque
    env = build_env(mode, target, ENV_CONFIG, render_mode=None)
    idx = env.idx
    obs_dim = env.observation_space.shape[0]
    history = ObsHistory(1, SEQ_LEN, obs_dim, device)
    success_count = 0
    peak_heights = []
    peak_progress = []
    for ep in range(episodes):
        obs, _ = env.reset()
        history.prime_single(obs)
        done = False
        max_height = -float('inf')
        max_progress = -float('inf')
        for step in range(max_steps):
            with torch.no_grad():
                seq = history.get()
                out = agent.get_action_and_value(seq)
                action = out[0].cpu().numpy().reshape(-1)
            next_obs, _, term, trun, info = env.step(action)
            history.update(next_obs)
            obs = next_obs
            if 'dbg_ramp_height' in info:
                h = float(info['dbg_ramp_height'])
                if h > max_height:
                    max_height = h
            if 'dbg_ramp_progress' in info:
                p = float(info['dbg_ramp_progress'])
                if p > max_progress:
                    max_progress = p
            done = bool(term or trun)
            if done:
                break
        peak_heights.append(max_height)
        peak_progress.append(max_progress)
        # 成功判定: 進捗0.9以上 or 高さ0.18以上
        if (
            max_progress >= RAMP_SUCCESS_PROG
            or max_height >= RAMP_SUCCESS_Z_RISE
        ):
            success_count += 1
    env.close()
    return {
        "success_rate": success_count / max(episodes, 1),
        "peak_height_mean": (
            float(np.mean(peak_heights)) if peak_heights else 0.0
        ),
        "peak_progress_mean": (
            float(np.mean(peak_progress)) if peak_progress else 0.0
        ),
    }

def run_ramp_check_viewer(mode, target, agent, device, episodes=1, max_steps=220):
    """
    ランプ登坂デバッグビューワ（ユーザー定義版）。
    mode: 環境モード
    target: "hider" or "seeker"
    agent: 学習済みエージェント
    device: torchデバイス
    episodes: 評価エピソード数
    max_steps: 1エピソードの最大ステップ数
    """
    env = build_env(mode, target, ENV_CONFIG, render_mode="human")
    idx = env.idx
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


def _cli_profile_override_from_argv(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    for i, token in enumerate(args):
        if token.startswith("--profile="):
            return token.split("=", 1)[1]
        if token in {"--profile", "-p"}:
            if i + 1 < len(args):
                return args[i + 1]
    return None


def _cli_compare_profiles_from_argv(argv=None):
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


CLI_PROFILE_OVERRIDE = _cli_profile_override_from_argv()
CLI_COMPARE_PROFILES = _cli_compare_profiles_from_argv()


def _get_available_runtime_profiles(path):
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


AVAILABLE_RUNTIME_PROFILES = _get_available_runtime_profiles(CONFIG_PATH)


def _parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description="HideAndSeek Transformer v27 runner",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=CLI_EXAMPLES,
    )
    parser.add_argument(
        "-p",
        "--profile",
        help="実行時だけ runtime profile を上書き（TOMLは変更しない）",
    )
    parser.add_argument(
        "--compare-profiles",
        help=(
            "カンマ区切りで複数 profile を順次実行（例: "
            "train_wall_only,train_no_wall_stick,train_base_reward）"
        ),
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="実行例のみ表示して終了",
    )
    args, _ = parser.parse_known_args(argv)
    if args.examples:
        print(CLI_EXAMPLES)
        raise SystemExit(0)


def _normalize_profile_name(value):
    text = str(value).strip().lower()
    if text in AVAILABLE_RUNTIME_PROFILES:
        return text
    print(
        f"Invalid runtime.active_profile={value}. "
        f"Fallback to train. available={AVAILABLE_RUNTIME_PROFILES}"
    )
    return "train"


def _load_runtime_config(path, profile_override=None):
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        runtime_cfg = raw.get("runtime", {}) if isinstance(raw, dict) else {}
        if not isinstance(runtime_cfg, dict):
            runtime_cfg = {}
        profile_value = (
            profile_override
            if profile_override is not None
            else runtime_cfg.get("active_profile", "train")
        )
        profile_origin = (
            "cli(--profile)" if profile_override is not None else "toml(runtime.active_profile)"
        )
        profile = _normalize_profile_name(profile_value)
        common = (
            runtime_cfg.get("common", {}) if isinstance(runtime_cfg, dict) else {}
        )
        profile_cfg = (
            runtime_cfg.get(profile, {}) if isinstance(runtime_cfg, dict) else {}
        )
        if not isinstance(common, dict):
            common = {}
        if not isinstance(profile_cfg, dict):
            profile_cfg = {}
        merged = dict(common)
        merged.update(profile_cfg)
        return merged, profile, profile_origin
    except Exception as exc:
        raise SystemExit(
            f"runtime config load failed: {path} ({exc}). "
            "Fix the TOML and retry."
        ) from exc


RUNTIME_OVERRIDES, RUN_PROFILE, RUN_PROFILE_SOURCE = _load_runtime_config(
    CONFIG_PATH, CLI_PROFILE_OVERRIDE
)


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg(name, default, cast=None):
    if name in RUNTIME_OVERRIDES:
        raw = RUNTIME_OVERRIDES[name]
    else:
        raw = default
    if cast is None:
        return raw
    return cast(raw)


TRAIN_MODE = _cfg("TRAIN_MODE", "0", _to_bool)
USE_VIEWER = _cfg("USE_VIEWER", "1", _to_bool)
NPC_ONLY_DEBUG = _cfg("NPC_ONLY_DEBUG", "1", _to_bool)
SHOW_TURN_LINES = _cfg("SHOW_TURN_LINES", "0", _to_bool)
USE_CUSTOM_REWARD = _cfg("USE_CUSTOM_REWARD", "1", _to_bool)
AUTO_TUNE_HPARAMS = _cfg("AUTO_TUNE_HPARAMS", "1", _to_bool)
DEBUG_HIDER_POLICY = _cfg("DEBUG_HIDER_POLICY", "rule", str)
DEBUG_SEEKER_POLICY = _cfg("DEBUG_SEEKER_POLICY", "rule", str)
DEBUG_DETERMINISTIC_INFERENCE = _cfg("DEBUG_DETERMINISTIC_INFERENCE", "0", _to_bool)
DEBUG_SYMMETRIC_HIDER_POLICY = _cfg("DEBUG_SYMMETRIC_HIDER_POLICY", "0", _to_bool)
TRAIN_OTHER_HIDER_POLICY = _cfg("TRAIN_OTHER_HIDER_POLICY", "rule", str)
TRAIN_OTHER_SEEKER_POLICY = _cfg("TRAIN_OTHER_SEEKER_POLICY", "rule", str)

if not TRAIN_MODE:
    USE_VIEWER = True
    AUTO_TUNE_HPARAMS = False

MODE = _cfg("MODE", "refinement", str)
TRAINING_TARGET = _cfg("TRAINING_TARGET", "seeker", str)

ENV_CONFIG = {
    "n_seekers": _cfg("N_SEEKERS", "1", int),
    "n_hiders": _cfg("N_HIDERS", "2", int),
    "n_boxes": _cfg("N_BOXES", "2", int),
    "n_ramps": _cfg("N_RAMPS", "1", int),
    "mode4_sdf_cell_size": _cfg("MODE4_SDF_CELL_SIZE", "0.05", float),
    "show_turn_lines": SHOW_TURN_LINES,
    "policy_source_log": _cfg("POLICY_SOURCE_LOG", "0", _to_bool),
    "policy_source_log_each_reset": _cfg("POLICY_SOURCE_LOG_EACH_RESET", "0", _to_bool),
    "debug_log_interval_steps": _cfg("DEBUG_LOG_INTERVAL_STEPS", "200", int),
    "action_repeat": _cfg("ACTION_REPEAT", "16", int),
}

SEQ_LEN = _cfg("SEQ_LEN", "8", int)
HIDDEN_DIM = _cfg("HIDDEN_DIM", "128", int)
NUM_ENVS = _cfg("NUM_ENVS", "8", int)
ACTION_REPEAT = _cfg("ACTION_REPEAT", "10", int)

WORKER_SHUTDOWN_ERRORS = (EOFError, BrokenPipeError, ConnectionResetError)

TOTAL_TIMESTEPS = _cfg("TOTAL_TIMESTEPS", "300000", int)
ROLLOUT_STEPS = _cfg("ROLLOUT_STEPS", "128", int)
UPDATE_EPOCHS = _cfg("UPDATE_EPOCHS", "4", int)
MINIBATCH_SIZE = _cfg("MINIBATCH_SIZE", "64", int)
LEARNING_RATE = _cfg("LEARNING_RATE", "3e-4", float)
GAMMA = _cfg("GAMMA", "0.995", float)
GAE_LAMBDA = _cfg("GAE_LAMBDA", "0.95", float)
CLIP_COEF = _cfg("CLIP_COEF", "0.2", float)
VF_COEF = _cfg("VF_COEF", "0.5", float)
ENT_COEF = _cfg("ENT_COEF", "0.001", float)
MAX_GRAD_NORM = _cfg("MAX_GRAD_NORM", "0.5", float)

# 再生/デバッグ（TRAIN_MODE=False）のエピソード数
PLAY_EPISODES = _cfg("PLAY_EPISODES", "100", int)
LOG_INTERVAL = _cfg("LOG_INTERVAL", "10", int)
SAVE_INTERVAL = _cfg("SAVE_INTERVAL", "50", int)

RAMP_CHECK_ENABLED = _cfg("RAMP_CHECK_ENABLED", "1", _to_bool)
RAMP_CHECK_INTERVAL = _cfg("RAMP_CHECK_INTERVAL", str(SAVE_INTERVAL), int)
RAMP_CHECK_EPISODES = _cfg("RAMP_CHECK_EPISODES", "3", int)
RAMP_CHECK_STEPS = _cfg("RAMP_CHECK_STEPS", "220", int)
RAMP_CHECK_VIEWER = _cfg("RAMP_CHECK_VIEWER", "0", _to_bool)
RAMP_CHECK_FORCE_UPHILL = _cfg("RAMP_CHECK_FORCE_UPHILL", "1", _to_bool)
RAMP_SUCCESS_PROG = _cfg("RAMP_SUCCESS_PROG", "0.90", float)
RAMP_SUCCESS_TOP = _cfg("RAMP_SUCCESS_TOP", "0.05", float)
RAMP_SUCCESS_LAT = _cfg("RAMP_SUCCESS_LAT", "0.75", float)
RAMP_SUCCESS_Z_RISE = _cfg("RAMP_SUCCESS_Z_RISE", "0.18", float)

WANDB_ENABLED = _cfg("WANDB_ENABLED", "1", _to_bool)
WANDB_PROJECT = _cfg("WANDB_PROJECT", "hideandseek-xformer", str)
WANDB_ENTITY = _cfg("WANDB_ENTITY", "", str)
WANDB_MODE = _cfg("WANDB_MODE", "online", str)
WANDB_RUN_NAME = _cfg("WANDB_RUN_NAME", "", str)
WANDB_LOG_CODE = _cfg("WANDB_LOG_CODE", "1", _to_bool)
WANDB_CODE_ROOT = _cfg("WANDB_CODE_ROOT", os.path.dirname(__file__), str)

# Checkpoint handling
RESUME_TRAINING = _cfg("RESUME_TRAINING", "0", _to_bool)
RESET_MODEL_ON_TRAIN = _cfg("RESET_MODEL_ON_TRAIN", "0", _to_bool)
HPARAMS_CONFIG_PATH = CONFIG_PATH

# Reward shaping coefficients
RW_SEEK_VISIBLE_BONUS = _cfg("RW_SEEK_VISIBLE_BONUS", "0.03", float)
RW_SEEK_DIST_GAIN = _cfg("RW_SEEK_DIST_GAIN", "0.03", float)
RW_SEEK_IDLE_PENALTY = _cfg("RW_SEEK_IDLE_PENALTY", "0.01", float)
RW_HIDE_WALL_NEAR_THRESHOLD = _cfg("RW_HIDE_WALL_NEAR_THRESHOLD", "0.18", float)
RW_HIDE_STILL_SPEED_THRESHOLD = _cfg("RW_HIDE_STILL_SPEED_THRESHOLD", "0.05", float)

RW_MOVE_SAT_PENALTY = _cfg("RW_MOVE_SAT_PENALTY", "0.02", float)
RW_TURN_SAT_PENALTY = _cfg("RW_TURN_SAT_PENALTY", "0.01", float)
RW_MOVE_CTRL_COST = _cfg("RW_MOVE_CTRL_COST", "0.001", float)
RW_HAND_CTRL_COST = _cfg("RW_HAND_CTRL_COST", "0.0005", float)

# ✅ 追加する 3 個のパラメータ
RW_MOVE_INCENTIVE = _cfg("RW_MOVE_INCENTIVE", "0.02", float)
RW_IDLE_PENALTY = _cfg("RW_IDLE_PENALTY", "0.03", float)
RW_WALL_AVOID_PENALTY = _cfg("RW_WALL_AVOID_PENALTY", "0.05", float)


def _index_spec_to_array(index_spec):
    if isinstance(index_spec, slice):
        start = 0 if index_spec.start is None else int(index_spec.start)
        stop = int(index_spec.stop)
        step = 1 if index_spec.step is None else int(index_spec.step)
        return np.arange(start, stop, step, dtype=np.int32)
    return np.asarray(index_spec, dtype=np.int32)


def _build_reward_index_cache(idx):
    return {
        "self_vel_x": int(idx.SELF.VEL_X),
        "self_vel_y": int(idx.SELF.VEL_Y),
        "lidar": _index_spec_to_array(idx.LIDAR),
        "enemy_visible": np.asarray([int(en.VISIBLE) for en in idx.OTHERS], dtype=np.int32),
        "enemy_rel_x": np.asarray([int(en.REL_X) for en in idx.OTHERS], dtype=np.int32),
        "enemy_rel_y": np.asarray([int(en.REL_Y) for en in idx.OTHERS], dtype=np.int32),
    }


def _min_lidar_from_obs(obs, lidar_indices):
    lidar_min = 1.0
    for lidar_idx in lidar_indices:
        value = float(obs[int(lidar_idx)])
        if value < lidar_min:
            lidar_min = value
    return lidar_min


@njit(cache=True, parallel=True)
def _compute_custom_reward_batch_numba(
    obs_batch,
    action_batch,
    base_reward_batch,
    target_is_hider,
    self_vel_x_idx,
    self_vel_y_idx,
    lidar_indices,
    enemy_visible_indices,
    enemy_rel_x_indices,
    enemy_rel_y_indices,
    move_ctrl_cost,
    move_incentive,
    idle_penalty,
    wall_avoid_penalty,
):
    n_envs = obs_batch.shape[0]
    rewards = np.empty(n_envs, dtype=np.float32)
    for i in prange(n_envs):
        move = float(action_batch[i, 0])
        turn = float(action_batch[i, 1])
        vel_x = float(obs_batch[i, int(self_vel_x_idx)])
        vel_y = float(obs_batch[i, int(self_vel_y_idx)])
        speed = np.sqrt(vel_x * vel_x + vel_y * vel_y)

        control_cost = -move_ctrl_cost * (move * move + turn * turn)
        bonus = 0.0

        if target_is_hider:
            if speed < 0.1:
                bonus -= idle_penalty
            elif speed > 0.3:
                bonus += move_incentive

            lidar_min = 1.0
            for j in range(lidar_indices.shape[0]):
                idx = int(lidar_indices[j])
                lidar_val = float(obs_batch[i, idx])
                if lidar_val < lidar_min:
                    lidar_min = lidar_val
            if lidar_min < 0.3:
                ratio = lidar_min / 0.3
                if ratio < 0.0:
                    ratio = 0.0
                elif ratio > 1.0:
                    ratio = 1.0
                bonus -= wall_avoid_penalty * (1.0 - ratio)
        else:
            if speed < 0.1:
                bonus -= idle_penalty

            for j in range(enemy_visible_indices.shape[0]):
                idx_vis = int(enemy_visible_indices[j])
                if obs_batch[i, idx_vis] > 0.5:
                    idx_x = int(enemy_rel_x_indices[j])
                    idx_y = int(enemy_rel_y_indices[j])
                    dx = float(obs_batch[i, idx_x])
                    dy = float(obs_batch[i, idx_y])
                    dist = np.sqrt(dx * dx + dy * dy)
                    bonus += 0.05 / (dist + 1.0)

        rewards[i] = float(base_reward_batch[i]) + bonus + control_cost
    return rewards


def _is_wall_stick_state(obs, idx, reward_idx_cache=None):
    cache = reward_idx_cache if reward_idx_cache is not None else _build_reward_index_cache(idx)
    speed = math.sqrt(float(obs[cache["self_vel_x"]]) ** 2 + float(obs[cache["self_vel_y"]]) ** 2)
    lidar_min = _min_lidar_from_obs(obs, cache["lidar"])
    return bool(
        (lidar_min < RW_HIDE_WALL_NEAR_THRESHOLD)
        and (speed < RW_HIDE_STILL_SPEED_THRESHOLD)
    )


def env_signature(config):
    """
    環境設定のシグネチャ文字列を生成（wandb名やログ用）
    """
    keys = [
        "n_seekers",
        "n_hiders",
        "n_boxes",
        "n_ramps",
        "mode4_sdf_cell_size",
    ]
    return ",".join(f"{k}={config.get(k)}" for k in keys if k in config)

def model_path_for_config(target, config):
    """
    モデルの保存・読み込み用パスを一意に決定する
    """
    fname = f"HNS_V27_{target}_s{config.get('n_seekers', 1)}_h{config.get('n_hiders', 2)}_b{config.get('n_boxes', 2)}_r{config.get('n_ramps', 1)}.pt"
    return os.path.join("checkpoints", fname)


def build_env(mode, target, config, render_mode=None):
    return TeamCosEnv(
        mode=mode,
        target=target,
        render_mode=render_mode,
        **config,
    )


def _snapshot_state_dict_cpu(agent):
    return {
        key: value.detach().cpu().clone()
        for key, value in agent.state_dict().items()
    }

def _normalize_policy_mode(value):
    text = str(value).strip().lower()
    if text in {"model", "model_if_available", "inference"}:
        return POLICY_MODEL_IF_AVAILABLE
    return POLICY_RULE

def _resolve_runtime_target():
    if TRAIN_MODE:
        return TRAINING_TARGET
    return "hider"


def _maybe_load_model_state(target):
    path = model_path_for_config(target, ENV_CONFIG)
    if not os.path.exists(path):
        return None, path
    try:
        return torch.load(path, map_location="cpu"), path
    except Exception as exc:
        print(f"Model load failed ({target}: {exc})")
        return None, path


def _apply_policy_state(env, vec_envs, agent_keys, state_dict, label):
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
    if ref_env is None:
        return

    if TRAIN_MODE:
        hider_mode = _normalize_policy_mode(TRAIN_OTHER_HIDER_POLICY)
        seeker_mode = _normalize_policy_mode(TRAIN_OTHER_SEEKER_POLICY)
    else:
        hider_mode = _normalize_policy_mode(DEBUG_HIDER_POLICY)
        seeker_mode = _normalize_policy_mode(DEBUG_SEEKER_POLICY)

    model_policy_deterministic = True if TRAIN_MODE else bool(DEBUG_DETERMINISTIC_INFERENCE)
    det_sync_ok = _set_model_policy_deterministic(env, vec_envs, model_policy_deterministic)

    hider_keys = [k for k in ref_env.hider_keys if k != ref_env.learnable_agent_key]
    seeker_keys = [k for k in ref_env.seeker_keys if k != ref_env.learnable_agent_key]

    hider_state = primary_state_dict if runtime_target == "hider" else None
    seeker_state = primary_state_dict if runtime_target == "seeker" else None

    if hider_mode == POLICY_MODEL_IF_AVAILABLE and hider_state is None:
        hider_state, hider_path = _maybe_load_model_state("hider")
        if hider_state is None:
            print(f"Hider policy: fallback to rule (missing model: {hider_path})")

    if seeker_mode == POLICY_MODEL_IF_AVAILABLE and seeker_state is None:
        seeker_state, seeker_path = _maybe_load_model_state("seeker")
        if seeker_state is None:
            print(f"Seeker policy: fallback to rule (missing model: {seeker_path})")

    if hider_mode == POLICY_MODEL_IF_AVAILABLE:
        _apply_policy_state(env, vec_envs, hider_keys, hider_state, "Hider inference policy")
    if seeker_mode == POLICY_MODEL_IF_AVAILABLE:
        _apply_policy_state(env, vec_envs, seeker_keys, seeker_state, "Seeker inference policy")

    symmetric_hider_debug = (
        (not TRAIN_MODE)
        and DEBUG_SYMMETRIC_HIDER_POLICY
        and runtime_target == "hider"
        and hider_mode == POLICY_MODEL_IF_AVAILABLE
        and (hider_state is not None)
    )
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
        _set_override_learnable_policy(env, vec_envs, ok)
        override_enabled = bool(ok)
        if ok:
            print("Debug mode: all hiders use same model inference path")
    else:
        _set_override_learnable_policy(env, vec_envs, False)
        override_enabled = False

    print(
        "Policy wiring: "
        f"hider_mode={hider_mode}, "
        f"seeker_mode={seeker_mode}, "
        f"model_policy_deterministic={model_policy_deterministic}, "
        f"det_sync_ok={det_sync_ok}, "
        f"override_learnable_policy={override_enabled}"
    )


def run_debug_or_playback(env, agent, device, model_loaded):
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

    # --- trainと同じlog_interval統計出力に統一 ---
    log_interval = 1  # debug時は必ずUpd ...を出す
    wall_dist_interval = 1000
    num_episodes = PLAY_EPISODES
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
        history.prime_single(obs)
        done = False
        ep_reward = 0.0
        step_count = 0
        lock_events = 0
        wall_stick = 0.0
        wall_dist = []
        while not done:
            if USE_VIEWER and env.viewer and not env.viewer.is_running():
                return
            if NPC_ONLY_DEBUG and target_npc is not None:
                action = target_npc.get_action(obs, idx)
            elif model_loaded:
                with torch.no_grad():
                    seq = history.get()
                    if DEBUG_DETERMINISTIC_INFERENCE and hasattr(agent, "get_deterministic_action_and_value"):
                        out = agent.get_deterministic_action_and_value(seq)
                    else:
                        out = agent.get_action_and_value(seq)
                    action = out[0].cpu().numpy().reshape(-1)
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            next_obs, base_r, term, trun, info = env.step(action)
            reward = (
                compute_custom_reward(obs, action, base_r, idx, env.target, info, reward_idx_cache=reward_idx_cache)
                if USE_CUSTOM_REWARD else base_r
            )
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
            if USE_VIEWER:
                env.render()
                time.sleep(0.025)
            obs = next_obs
            done = bool(term or trun)
        episode_rewards.append(ep_reward)
        hide_rates.append(float(info.get("hide_rate", 0.0)))
        lock_events_list.append(lock_events)
        wall_stick_list.append(wall_stick / max(step_count, 1))
        wall_distance_buffer.extend(wall_dist)
        global_step += step_count
        if steps_per_episode is None:
            steps_per_episode = step_count
        # log_intervalごとに統計出力
        if (episode + 1) % log_interval == 0:
            elapsed = time.time() - debug_start_time
            sps = int(global_step / elapsed) if elapsed > 0 else 0
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_hide = np.mean(hide_rates[-log_interval:])
            avg_wall_stick = np.mean(wall_stick_list[-log_interval:])
            print(f"Upd {(episode + 1) // log_interval}/{num_updates} Step={global_step} SPS={sps} HideRate={avg_hide:.2f} WallStick={avg_wall_stick:.2f} AvgR={avg_reward:.3f}")
        # wall_distance統計出力
        if global_step % wall_dist_interval == 0 and wall_distance_buffer:
            arr = np.array(wall_distance_buffer, dtype=np.float32)
            print(f"[WallDist] step={global_step} mean={np.mean(arr):.3f} min={np.min(arr):.3f} max={np.max(arr):.3f}")
            wall_distance_buffer.clear()


def run_train(
    env,
    ref_env,
    agent,
    optimizer,
    model_path,
    device,
    hp,
    training_target,
    wandb_run=None,
):
    def _atomic_save(path):
        tmp_path = f"{path}.tmp"
        torch.save(agent.state_dict(), tmp_path)
        os.replace(tmp_path, path)

    idx = env.idx
    reward_idx_cache = _build_reward_index_cache(idx)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    history = ObsHistory(1, SEQ_LEN, obs_dim, device)

    obs, _ = env.reset()
    history.prime_single(obs)
    if USE_VIEWER:
        env.render()

    rollout_obs = torch.zeros((hp["rollout_steps"], SEQ_LEN, obs_dim), device=device)
    rollout_actions = torch.zeros((hp["rollout_steps"], act_dim), device=device)
    rollout_logp = torch.zeros(hp["rollout_steps"], device=device)
    rollout_rewards = torch.zeros(hp["rollout_steps"], device=device)
    rollout_dones = torch.zeros(hp["rollout_steps"], device=device)
    rollout_values = torch.zeros(hp["rollout_steps"], device=device)

    num_updates = max(1, hp["total_timesteps"] // hp["rollout_steps"])
    global_step = 0
    train_start_time = time.time()
    print(f"Training updates: {num_updates}, model: {model_path}")
    interrupted = False
    last_ramp_eval = None
    try:
        for update in range(1, num_updates + 1):
            lock_evt_sum = 0
            grab_evt_sum = 0
            max_box_speed = 0.0
            max_ramp_speed = 0.0
            blocked_ramp_sum = 0
            done_last = 0.0
            rewards_sum = 0.0
            hidden_steps = 0
            learnable_seen_steps = 0
            wall_stick_steps = 0
            wall_stick_seen_steps = 0
            entropy_sum = 0.0
            entropy_count = 0
            reward_compute_sec_sum = 0.0
            env_step_sec_sum = 0.0

            for t in range(hp["rollout_steps"]):
                if USE_VIEWER and env.viewer and not env.viewer.is_running():
                    print("Viewer closed. Stop training loop.")
                    return
                seq = history.get()
                rollout_obs[t] = seq

                with torch.no_grad():
                    action, logp, _, value = agent.get_action_and_value(seq)

                action_np = action.cpu().numpy()
                step_t0 = time.perf_counter()
                next_obs, base_r, term, trun, info = env.step(action_np)
                env_step_sec_sum += time.perf_counter() - step_t0
                done_np = np.logical_or(term, trun).astype(np.float32)
                done_last = done_np

                reward_np = np.asarray(base_r, dtype=np.float32).copy()
                reward_t0 = time.perf_counter()
                if USE_CUSTOM_REWARD:
                    reward_np = _compute_custom_reward_batch_numba(
                        np.asarray(obs, dtype=np.float32),
                        np.asarray(action_np, dtype=np.float32),
                        np.asarray(reward_np, dtype=np.float32),
                        bool(ref_env.target == "hider"),
                        reward_idx_cache["self_vel_x"],
                        reward_idx_cache["self_vel_y"],
                        reward_idx_cache["lidar"],
                        reward_idx_cache["enemy_visible"],
                        reward_idx_cache["enemy_rel_x"],
                        reward_idx_cache["enemy_rel_y"],
                        float(RW_MOVE_CTRL_COST),
                        float(RW_MOVE_INCENTIVE),
                        float(RW_IDLE_PENALTY),
                        float(RW_WALL_AVOID_PENALTY),
                    )
                reward_compute_sec_sum += time.perf_counter() - reward_t0

                global_step += num_envs
                rollout_actions[t] = action
                rollout_logp[t] = logp
                rollout_rewards[t] = torch.as_tensor(reward_np, device=device)
                rollout_dones[t] = torch.as_tensor(done_np, device=device)
                rollout_values[t] = value.view(-1)

                for i in range(num_envs):
                    lock_evt_sum += int(_info_at(info, "lock_event", i, 0))
                    grab_evt_sum += int(_info_at(info, "grab_event", i, 0))
                    max_box_speed = max(
                        max_box_speed,
                        float(_info_at(info, "dbg_max_box_speed", i, 0.0)),
                    )
                    max_ramp_speed = max(
                        max_ramp_speed,
                        float(_info_at(info, "dbg_max_ramp_speed", i, 0.0)),
                    )
                    blocked_ramp_sum += int(
                        _info_at(info, "dbg_blocked_ramp_count", i, 0)
                    )
                    if not bool(_info_at(info, "is_detected", i, False)):
                        hidden_steps += 1
                    seen_learnable = bool(_info_at(info, "dbg_learnable_hider_seen", i, _info_at(info, "is_detected", i, False)))
                    if seen_learnable:
                        learnable_seen_steps += 1
                    is_wall_stick = _is_wall_stick_state(next_obs[i], idx, reward_idx_cache=reward_idx_cache)
                    if is_wall_stick:
                        wall_stick_steps += 1
                        if seen_learnable:
                            wall_stick_seen_steps += 1

                rewards_sum += float(np.sum(reward_np))

                history.update(next_obs)
                for i in range(num_envs):
                    if done_np[i] > 0.5:
                        history.reset_env(i, next_obs[i])
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
                delta = (
                    rollout_rewards[t]
                    + hp["gamma"] * next_vals * next_nonterminal
                    - rollout_values[t]
                )
                lastgaelam = (
                    delta
                    + hp["gamma"]
                    * hp["gae_lambda"]
                    * next_nonterminal
                    * lastgaelam
                )
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
                    mb_idx = inds[start:start + hp["minibatch_size"]]
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
            if (
                RAMP_CHECK_ENABLED
                and RAMP_CHECK_INTERVAL > 0
                and update % RAMP_CHECK_INTERVAL == 0
            ):
                ramp_eval = evaluate_fixed_ramp_climb(
                    MODE,
                    training_target,
                    agent,
                    device,
                    episodes=RAMP_CHECK_EPISODES,
                    max_steps=RAMP_CHECK_STEPS,
                )
                last_ramp_eval = ramp_eval
            if wandb_run is not None:
                payload = {
                    "global_step": global_step,
                    "train/update": update,
                    "train/sps": sps,
                    "train/hide_rate": hide_rate,
                    "train/avg_reward": avg_reward,
                    "train/entropy": avg_entropy,
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
                wandb_run.log(payload, step=global_step)
            if update % hp["log_interval"] == 0:
                print(f"Upd {update}/{num_updates} Step={global_step} SPS={sps} HideRate={hide_rate:.2f} WallStick={wall_stick_ratio:.2f} AvgR={avg_reward:.3f}")
            # wall_distance統計バッファ
            if not hasattr(run_train_vector, '_wall_distance_buffer'):
                run_train_vector._wall_distance_buffer = []
            if not hasattr(run_train_vector, '_info_buffer'):
                run_train_vector._info_buffer = []
            # 1000ステップごとにinfoからwall_distance統計を集計・出力
            if global_step % 1000 == 0:
                wall_dist_list = []
                for t in range(len(run_train_vector._info_buffer)):
                    info = run_train_vector._info_buffer[t]
                    if isinstance(info, (list, tuple)):
                        for i in range(num_envs):
                            info_i = info[i] if len(info) > i else None
                            if info_i and 'wall_distance' in info_i:
                                wall_dist_list.append(float(info_i['wall_distance']))
                    elif isinstance(info, dict):
                        if 'wall_distance' in info:
                            wall_dist_list.append(float(info['wall_distance']))
                if wall_dist_list:
                    arr = np.array(wall_dist_list, dtype=np.float32)
                    print(f"[WallDist] step={global_step} mean={np.mean(arr):.3f} min={np.min(arr):.3f} max={np.max(arr):.3f}")
            # infoバッファをクリア
            run_train_vector._info_buffer.clear()

            if update % hp["save_interval"] == 0:
                _atomic_save(model_path)
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted. Saving checkpoint...")
        raise
    except WORKER_SHUTDOWN_ERRORS as exc:
        interrupted = True
        print(
            f"\nVector worker terminated ({exc.__class__.__name__}). "
            "Saving checkpoint and exiting cleanly..."
        )
    finally:
        _atomic_save(model_path)
        if interrupted:
            print(f"Interrupted checkpoint saved: {model_path}")
        else:
            print(f"Saved model: {model_path}")


def safe_close_vector_env(vec_envs):
    if vec_envs is None:
        return
    try:
        vec_envs.close(terminate=True)
    except WORKER_SHUTDOWN_ERRORS as exc:
        print(f"VectorEnv close skipped ({exc.__class__.__name__})")
    except Exception as exc:
        print(f"VectorEnv close suppressed ({exc.__class__.__name__}: {exc})")


def _info_at(info, key, env_idx, default=0):
    if key not in info:
        return default
    value = info[key]
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) <= env_idx:
            return default
        return value[env_idx]
    return value


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
    def _atomic_save(path):
        tmp_path = f"{path}.tmp"
        torch.save(agent.state_dict(), tmp_path)
        os.replace(tmp_path, path)

    idx = ref_env.idx
    reward_idx_cache = _build_reward_index_cache(idx)
    obs_dim = ref_env.observation_space.shape[0]
    act_dim = ref_env.action_space.shape[0]
    history = ObsHistory(num_envs, SEQ_LEN, obs_dim, device)

    obs, _ = envs.reset()
    history.prime(obs)

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
    print(
        f"Vector training: envs={num_envs}, updates={num_updates}, "
        f"model={model_path}"
    )
    interrupted = False
    last_ramp_eval = None

    try:
        # バッファをローカル変数で管理
        wall_distance_buffer = []
        info_buffer = []
        for update in range(1, num_updates + 1):
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
                seq = history.get()
                rollout_obs[t] = seq

                with torch.no_grad():
                    action, logp, _, value = agent.get_action_and_value(seq)

                action_np = action.cpu().numpy()
                step_t0 = time.perf_counter()
                next_obs, base_r, term, trun, info = envs.step(action_np)
                # infoをバッファに追加
                info_buffer.append(info)
                env_step_sec_sum += time.perf_counter() - step_t0
                done_np = np.logical_or(term, trun).astype(np.float32)
                done_last = done_np

                reward_np = np.asarray(base_r, dtype=np.float32).copy()
                reward_t0 = time.perf_counter()
                if USE_CUSTOM_REWARD:
                    reward_np = _compute_custom_reward_batch_numba(
                        np.asarray(obs, dtype=np.float32),
                        np.asarray(action_np, dtype=np.float32),
                        np.asarray(reward_np, dtype=np.float32),
                        bool(ref_env.target == "hider"),
                        reward_idx_cache["self_vel_x"],
                        reward_idx_cache["self_vel_y"],
                        reward_idx_cache["lidar"],
                        reward_idx_cache["enemy_visible"],
                        reward_idx_cache["enemy_rel_x"],
                        reward_idx_cache["enemy_rel_y"],
                        float(RW_MOVE_CTRL_COST),
                        float(RW_MOVE_INCENTIVE),
                        float(RW_IDLE_PENALTY),
                        float(RW_WALL_AVOID_PENALTY),
                    )
                reward_compute_sec_sum += time.perf_counter() - reward_t0

                global_step += num_envs
                rollout_actions[t] = action
                rollout_logp[t] = logp
                rollout_rewards[t] = torch.as_tensor(reward_np, device=device)
                rollout_dones[t] = torch.as_tensor(done_np, device=device)
                rollout_values[t] = value.view(-1)

                for i in range(num_envs):
                    lock_evt_sum += int(_info_at(info, "lock_event", i, 0))
                    grab_evt_sum += int(_info_at(info, "grab_event", i, 0))
                    max_box_speed = max(
                        max_box_speed,
                        float(_info_at(info, "dbg_max_box_speed", i, 0.0)),
                    )
                    max_ramp_speed = max(
                        max_ramp_speed,
                        float(_info_at(info, "dbg_max_ramp_speed", i, 0.0)),
                    )
                    blocked_ramp_sum += int(
                        _info_at(info, "dbg_blocked_ramp_count", i, 0)
                    )
                    if not bool(_info_at(info, "is_detected", i, False)):
                        hidden_steps += 1
                    seen_learnable = bool(_info_at(info, "dbg_learnable_hider_seen", i, _info_at(info, "is_detected", i, False)))
                    if seen_learnable:
                        learnable_seen_steps += 1
                    is_wall_stick = _is_wall_stick_state(next_obs[i], idx, reward_idx_cache=reward_idx_cache)
                    if is_wall_stick:
                        wall_stick_steps += 1
                        if seen_learnable:
                            wall_stick_seen_steps += 1

                rewards_sum += float(np.sum(reward_np))

                history.update(next_obs)
                for i in range(num_envs):
                    if done_np[i] > 0.5:
                        history.reset_env(i, next_obs[i])
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
                delta = (
                    rollout_rewards[t]
                    + hp["gamma"] * next_vals * next_nonterminal
                    - rollout_values[t]
                )
                lastgaelam = (
                    delta
                    + hp["gamma"]
                    * hp["gae_lambda"]
                    * next_nonterminal
                    * lastgaelam
                )
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
                    mb_idx = inds[start:start + hp["minibatch_size"]]
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
            if (
                RAMP_CHECK_ENABLED
                and RAMP_CHECK_INTERVAL > 0
                and update % RAMP_CHECK_INTERVAL == 0
            ):
                ramp_eval = evaluate_fixed_ramp_climb(
                    MODE,
                    training_target,
                    agent,
                    device,
                    episodes=RAMP_CHECK_EPISODES,
                    max_steps=RAMP_CHECK_STEPS,
                )
                last_ramp_eval = ramp_eval
            if wandb_run is not None:
                payload = {
                    "global_step": global_step,
                    "train/update": update,
                    "train/sps": sps,
                    "train/hide_rate": hide_rate,
                    "train/avg_reward": avg_reward,
                    "train/entropy": avg_entropy,
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
                wandb_run.log(payload, step=global_step)
            if update % hp["log_interval"] == 0:
                print(f"Upd {update}/{num_updates} Step={global_step} SPS={sps} HideRate={hide_rate:.2f} WallStick={wall_stick_ratio:.2f} AvgR={avg_reward:.3f}")

            # wall_distance統計バッファ（ローカル変数で管理）
            for info in info_buffer:
                if isinstance(info, (list, tuple)):
                    for i in range(num_envs):
                        info_i = info[i] if len(info) > i else None
                        if info_i and 'wall_distance' in info_i:
                            wd = info_i['wall_distance']
                            if isinstance(wd, (list, tuple, np.ndarray)):
                                for v in wd:
                                    wall_distance_buffer.append(float(v))
                            else:
                                wall_distance_buffer.append(float(wd))
                elif isinstance(info, dict):
                    if 'wall_distance' in info:
                        wd = info['wall_distance']
                        if isinstance(wd, (list, tuple, np.ndarray)):
                            for v in wd:
                                wall_distance_buffer.append(float(v))
                        else:
                            wall_distance_buffer.append(float(wd))
            if update % hp["log_interval"] == 0 and wall_distance_buffer:
                arr = np.array(wall_distance_buffer, dtype=np.float32)
                print(f"[WallDist] step={global_step} mean={np.mean(arr):.3f} min={np.min(arr):.3f} max={np.max(arr):.3f}")
                wall_distance_buffer.clear()
            if update % hp["log_interval"] == 0:
                info_buffer.clear()

            if update % hp["save_interval"] == 0:
                _atomic_save(model_path)
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted. Saving checkpoint...")
        raise
    finally:
        _atomic_save(model_path)
        if interrupted:
            print(f"Interrupted checkpoint saved: {model_path}")
        else:
            print(f"Saved model: {model_path}")


def run():
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
        f"DEBUG_DETERMINISTIC_INFERENCE={DEBUG_DETERMINISTIC_INFERENCE}, "
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
                print(
                    "Viewer有効時は並列実行を無効化し、"
                    "単一環境でレンダリングします。"
                )
            env = make_env(render_mode=("human" if USE_VIEWER else None))
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
    hp = select_hparams(ENV_CONFIG, runtime_target)
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
                run_train(
                    env,
                    ref_env,
                    agent,
                    optimizer,
                    model_path,
                    device,
                    hp,
                    runtime_target,
                    wandb_run,
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


def run_profile_comparison(profiles):
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


if __name__ == "__main__":
    if CLI_COMPARE_PROFILES:
        raise SystemExit(run_profile_comparison(CLI_COMPARE_PROFILES))
    run()