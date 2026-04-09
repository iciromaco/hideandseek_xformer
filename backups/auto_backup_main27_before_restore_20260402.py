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
from src.core.constants import P_SCALE
import gymnasium as gym
from functools import partial
from numba import njit, prange

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
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "src")
    )
)

from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider


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
    print(
        f"Invalid runtime.active_profile={value}. "
        f"Fallback to train. available={available_profiles}"
    )
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
        profile_value = (
            profile_override
            if profile_override is not None
            else runtime_cfg.get("active_profile", "train")
        )
        profile_origin = (
            "cli(--profile)" if profile_override is not None else "toml(runtime.active_profile)"
        )
        profile_lower = _normalize_profile_name(profile_value, available_profiles)
        common = (
            runtime_cfg.get("common", {}) if isinstance(runtime_cfg, dict) else {}
        )
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
        profile_cfg = (
            runtime_cfg.get(profile_key, {}) if isinstance(runtime_cfg, dict) else {}
        )
        profile = profile_key
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
        print(cli_examples)
        raise SystemExit(0)


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg(name, default, cast=None, overrides=None):
    if overrides is None:
        overrides = {}
    if name in overrides:
        raw = overrides[name]
    else:
        raw = default
    if cast is None:
        return raw
    return cast(raw)


# グローバル設定の初期化
CLI_PROFILE_OVERRIDE = _cli_profile_override_from_argv()
CLI_COMPARE_PROFILES = _cli_compare_profiles_from_argv()
AVAILABLE_RUNTIME_PROFILES = _get_available_runtime_profiles(CONFIG_PATH)
RUNTIME_OVERRIDES, RUN_PROFILE, RUN_PROFILE_SOURCE = _load_runtime_config(CONFIG_PATH, CLI_PROFILE_OVERRIDE, AVAILABLE_RUNTIME_PROFILES)
with open(CONFIG_PATH, "rb") as f:
    config = tomllib.load(f)
hp = dict(config.get("base", {}))
runtime_common = config.get("runtime", {}).get("common", {})
if runtime_common:
    hp.update(runtime_common)
runtime_profile = config.get("runtime", {}).get(RUN_PROFILE, {})
if runtime_profile:
    hp.update(runtime_profile)
if RUNTIME_OVERRIDES:
    hp.update(RUNTIME_OVERRIDES)

TRAIN_MODE = _cfg("train_mode", hp.get("train_mode", "0"), _to_bool, RUNTIME_OVERRIDES)
USE_VIEWER = _cfg("use_viewer", hp.get("use_viewer", "1"), _to_bool, RUNTIME_OVERRIDES)
NPC_ONLY_DEBUG = _cfg("npc_only_debug", hp.get("npc_only_debug", "1"), _to_bool, RUNTIME_OVERRIDES)
SHOW_TURN_LINES = _cfg("show_turn_lines", hp.get("show_turn_lines", "0"), _to_bool, RUNTIME_OVERRIDES)
USE_CUSTOM_REWARD = _cfg("use_custom_reward", hp.get("use_custom_reward", "1"), _to_bool, RUNTIME_OVERRIDES)
AUTO_TUNE_HPARAMS = _cfg("auto_tune_hparams", hp.get("auto_tune_hparams", "1"), _to_bool, RUNTIME_OVERRIDES)
DEBUG_HIDER_POLICY = _cfg("debug_hider_policy", hp.get("debug_hider_policy", "rule"), str, RUNTIME_OVERRIDES)
DEBUG_SEEKER_POLICY = _cfg("debug_seeker_policy", hp.get("debug_seeker_policy", "rule"), str, RUNTIME_OVERRIDES)
DEBUG_DETERMINISTIC_INFERENCE = _cfg("debug_deterministic_inference", hp.get("debug_deterministic_inference", "0"), _to_bool, RUNTIME_OVERRIDES)
DEBUG_SYMMETRIC_HIDER_POLICY = _cfg("debug_symmetric_hider_policy", hp.get("debug_symmetric_hider_policy", "0"), _to_bool, RUNTIME_OVERRIDES)
TRAIN_OTHER_HIDER_POLICY = _cfg("train_other_hider_policy", hp.get("train_other_hider_policy", "rule"), str, RUNTIME_OVERRIDES)
TRAIN_OTHER_SEEKER_POLICY = _cfg("train_other_seeker_policy", hp.get("train_other_seeker_policy", "rule"), str, RUNTIME_OVERRIDES)

# ビューアのフレーム間ディレイ（TOMLで `debug_viewer_step_delay` を設定可能）
DEBUG_VIEWER_STEP_DELAY = _cfg("debug_viewer_step_delay", hp.get("debug_viewer_step_delay", "0.03"), float, RUNTIME_OVERRIDES)

if not TRAIN_MODE:
    if RUN_PROFILE == "debug":
        USE_VIEWER = True
    AUTO_TUNE_HPARAMS = False

MODE = _cfg("mode", hp.get("mode", "refinement"), str, RUNTIME_OVERRIDES)
TRAINING_TARGET = _cfg("training_target", hp.get("training_target", "seeker"), str, RUNTIME_OVERRIDES)

SEQ_LEN = _cfg("seq_len", hp.get("seq_len", "16"), int, RUNTIME_OVERRIDES)
HIDDEN_DIM = _cfg("hidden_dim", hp.get("hidden_dim", "256"), int, RUNTIME_OVERRIDES)
NUM_ENVS = _cfg("num_envs", hp.get("num_envs", "8"), int, RUNTIME_OVERRIDES)

ENV_CONFIG = {
    "n_seekers": _cfg("n_seekers", "1", int, RUNTIME_OVERRIDES),
    "n_hiders": _cfg("n_hiders", "2", int, RUNTIME_OVERRIDES),
    "n_boxes": _cfg("n_boxes", "2", int, RUNTIME_OVERRIDES),
    "n_ramps": _cfg("n_ramps", "1", int, RUNTIME_OVERRIDES),
    "mode4_sdf_cell_size": _cfg("mode4_sdf_cell_size", "0.05", float, RUNTIME_OVERRIDES),
    "show_turn_lines": SHOW_TURN_LINES,
    "policy_source_log": _cfg("policy_source_log", "0", _to_bool, RUNTIME_OVERRIDES),
    "policy_source_log_each_reset": _cfg("policy_source_log_each_reset", "0", _to_bool, RUNTIME_OVERRIDES),
    "debug_log_interval_steps": _cfg("debug_log_interval_steps", "200", int, RUNTIME_OVERRIDES),
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

RESUME_TRAINING = _cfg("resume_training", "0", _to_bool, RUNTIME_OVERRIDES)
RESET_MODEL_ON_TRAIN = _cfg("reset_model_on_train", "0", _to_bool, RUNTIME_OVERRIDES)
HPARAMS_CONFIG_PATH = CONFIG_PATH

RW_SEEK_VISIBLE_BONUS = _cfg("rw_seek_visible_bonus", "0.03", float, RUNTIME_OVERRIDES)
RW_WALL_NEAR_THRESHOLD = _cfg("rw_wall_near_threshold", "0.18", float, RUNTIME_OVERRIDES)
RW_STILL_SPEED_THRESHOLD = _cfg("rw_still_speed_threshold", "0.05", float, RUNTIME_OVERRIDES)

RW_MOVE_SAT_PENALTY = _cfg("rw_move_sat_penalty", "0.02", float, RUNTIME_OVERRIDES)
RW_TURN_SAT_PENALTY = _cfg("rw_turn_sat_penalty", "0.01", float, RUNTIME_OVERRIDES)
RW_MOVE_CTRL_COST = _cfg("rw_move_ctrl_cost", "0.001", float, RUNTIME_OVERRIDES)

RW_MOVE_INCENTIVE = _cfg("rw_move_incentive", "0.02", float, RUNTIME_OVERRIDES)
RW_IDLE_PENALTY = _cfg("rw_idle_penalty", "0.03", float, RUNTIME_OVERRIDES)
RW_WALL_AVOID_PENALTY = _cfg("rw_wall_avoid_penalty", "0.05", float, RUNTIME_OVERRIDES)

RW_WALL_STICK_PENALTY = _cfg("rw_wall_stick_penalty", "0.5", float, RUNTIME_OVERRIDES)
RW_HIDE_VISIBLE_NEAR_PENALTY = _cfg("rw_hide_visible_near_penalty", "0.025", float, RUNTIME_OVERRIDES)

RW_HIDE_SEEKER_GAZE_PENALTY = _cfg("rw_hide_seeker_gaze_penalty", "0.03", float, RUNTIME_OVERRIDES)
RW_SEEK_GAZE_REWARD = _cfg("c", "0.03", float, RUNTIME_OVERRIDES)

IDLE_SPEED_THRESHOLD = _cfg("idle_speed_threshold", "0.1", float, RUNTIME_OVERRIDES)
MOVE_INCENTIVE_SPEED_THRESHOLD = _cfg("move_incentive_speed_threshold", "0.3", float, RUNTIME_OVERRIDES)
MOVE_SAT_THRESHOLD = _cfg("move_sat_threshold", "1.0", float, RUNTIME_OVERRIDES)
TURN_SAT_THRESHOLD = _cfg("turn_sat_threshold", "1.0", float, RUNTIME_OVERRIDES)

WALL_SAFE = _cfg("wall_safe", "1.0", float, RUNTIME_OVERRIDES)
AGENT_RADIUS = _cfg("agentl_radius", "0.4", float, RUNTIME_OVERRIDES)
WALL_NEAR_MARGIN = _cfg("wall_near_margin", "0.2", float, RUNTIME_OVERRIDES)

# 以下、バックアップ時の主要ロジックを復元（表示遅延は DEBUG_VIEWER_STEP_DELAY を使用）

class ObsHistory:
    def __init__(self, num_envs, seq_len, obs_dim, device):
        self.num_envs = int(num_envs)
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.device = device
        self.buffer = torch.zeros((self.num_envs, seq_len * 2, obs_dim), device=device)
        self.ptr = 0

    def reset(self):
        self.buffer.zero_()
        self.ptr = 0

    def prime(self, obs):
        self.reset()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = obs_t.reshape(self.num_envs, self.obs_dim)
        for i in range(self.seq_len):
            self.buffer[:, i] = obs_t
            self.buffer[:, i + self.seq_len] = obs_t

    def update(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs_t = obs_t.reshape(self.num_envs, self.obs_dim)
        self.buffer[:, self.ptr] = obs_t
        self.buffer[:, self.ptr + self.seq_len] = obs_t
        self.ptr = (self.ptr + 1) % self.seq_len

    def prime_single(self, obs):
        self.prime(np.asarray(obs, dtype=np.float32).reshape(1, -1))

    def update_single(self, obs):
        self.update(np.asarray(obs, dtype=np.float32).reshape(1, -1))

    def reset_env(self, env_idx, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(self.obs_dim)
        for i in range(self.seq_len):
            self.buffer[env_idx, i] = obs_t
            self.buffer[env_idx, i + self.seq_len] = obs_t

    def get(self):
        start = self.ptr
        end = self.ptr + self.seq_len
        return self.buffer[:, start:end]

# Fallback implementations to avoid syntax errors. Keep behavior minimal
# so the file can be imported and executed. Replace with optimized
# implementations when available.
def _compute_custom_reward_batch_numba(obs_batch, next_obs_batch, actions_batch, base_rewards, idx):
    """Minimal batch reward processor.

    Args:
        obs_batch: batch of observations (any sequence)
        next_obs_batch: batch of next observations
        actions_batch: batch of actions
        base_rewards: base rewards (sequence or scalar)
        idx: environment index or metadata

    Returns:
        numpy array of rewards corresponding to base_rewards.
    """
    try:
        return np.asarray(base_rewards, dtype=np.float32)
    except Exception:
        try:
            return np.zeros_like(np.asarray(base_rewards, dtype=np.float32))
        except Exception:
            return np.array([0.0], dtype=np.float32)


def compute_custom_reward(obs, next_obs, action, base_r, idx, target, info, reward_idx_cache=None):
    """Minimal single-step reward wrapper.

    For now returns the provided base reward. This preserves original
    control flow and avoids breaking debug/playback runs. Replace with
    real reward logic as needed.
    """
    return base_r

def env_signature(config):
    keys = ["n_seekers", "n_hiders", "n_boxes", "n_ramps", "mode4_sdf_cell_size"]
    return ",".join(f"{k}={config.get(k)}" for k in keys if k in config)

def model_path_for_config(target, config):
    fname = f"HNS_V27_{target}_s{config.get('n_seekers', 1)}_h{config.get('n_hiders', 2)}_b{config.get('n_boxes', 2)}_r{config.get('n_ramps', 1)}.pt"
    return os.path.join("checkpoints", fname)

def build_env(mode, target, config, render_mode=None):
    config = dict(config)
    config.pop('num_envs', None)
    return TeamCosEnv(mode=mode, target=target, render_mode=render_mode, **config)


def _build_reward_index_cache(idx):
    """Placeholder for reward index cache builder.

    Original implementation may build indices to accelerate reward computation.
    Provide a minimal stub so callers can pass it through.
    """
    return None


def run_ramp_check_viewer(mode, target, agent, device, episodes=1, max_steps=220):
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
            time.sleep(DEBUG_VIEWER_STEP_DELAY)
            done = bool(term or trun)
            if done:
                break
    env.close()

def run_debug_or_playback(env, agent, device, model_loaded):
    idx = env.idx
    reward_idx_cache = _build_reward_index_cache(idx)
    obs_dim = env.observation_space.shape[0]
    history = ObsHistory(1, SEQ_LEN, obs_dim, device)
    print("--- Start Simulation ---")
    print(f"Physics Step: {env.model.opt.timestep * 5}s")
    if "play_episodes" in hp:
        num_episodes = int(hp["play_episodes"])
    else:
        num_episodes = int(hp.get("PLAY_EPISODES", 100))
    episode_rewards = []
    for episode in range(num_episodes):
        obs, _ = env.reset()
        history.prime_single(obs)
        ep_reward = 0.0
        step_count = 0
        done = False
        while not done:
            if model_loaded:
                with torch.no_grad():
                    seq = history.get()
                    if DEBUG_DETERMINISTIC_INFERENCE and hasattr(agent, "get_deterministic_action_and_value"):
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
            if USE_VIEWER:
                try:
                    env.render()
                    time.sleep(DEBUG_VIEWER_STEP_DELAY)
                except Exception:
                    pass
            reward = compute_custom_reward(obs, next_obs, action, base_r, idx, env.target, info, reward_idx_cache=reward_idx_cache) if USE_CUSTOM_REWARD else base_r
            history.update(next_obs)
            ep_reward += reward
            step_count += 1
            obs = next_obs
            done = bool(term or trun)
        episode_rewards.append(ep_reward)
        print(f"[run_debug] episode={episode+1} steps={step_count} reward={ep_reward:.3f}")
    if USE_VIEWER:
        try:
            print("[runner] debug finished — keeping viewer open until closed by user")
            try:
                env.render()
            except Exception:
                pass
            v = getattr(env, 'viewer', None)
            while v is not None and getattr(v, 'is_running', lambda: False)():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


def run():
    """Minimal entrypoint to start debug/playback runs.

    This function restores the previous CLI-driven behavior in a minimal
    way: it builds the environment according to current hparams and
    launches the debug/playback loop. It intentionally avoids loading
    heavy models so it's safe to run in environments without checkpoints.
    """
    device = torch.device("cpu")
    try:
        mode = MODE
        target = TRAINING_TARGET
        render_mode = "human" if USE_VIEWER else None
        env = build_env(mode, target, ENV_CONFIG, render_mode=render_mode)
        model_loaded = False
        agent = None
        try:
            run_debug_or_playback(env, agent, device, model_loaded)
        finally:
            try:
                env.close()
            except Exception:
                pass
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    run()
