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

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "src")
    )
)

from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider
from experiments.online_update_untrained import set_hider_pos_fixed
# =====================
# main27_train_final.pyのグローバル設定・CLIパース・プロファイル切り替え
# =====================
POLICY_MODEL_IF_AVAILABLE = "model_if_available"
POLICY_RULE = "rule"

CONFIG_PATH = "configs/hparams_main27.toml"
WORKER_SHUTDOWN_ERRORS = (EOFError, BrokenPipeError, ConnectionResetError)

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

def _normalize_profile_name(value, available_profiles):
    text = str(value).strip().lower()
    if text in available_profiles:
        return text
    print(
        f"Invalid runtime.active_profile={value}. "
        f"Fallback to train. available={available_profiles}"
    )
    return "train"

def _load_runtime_config(path, profile_override=None, available_profiles=None):
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
    cli_examples = (
        "実行例:\n"
        "  1) 学習を実行する（TOMLで train を選択）\n"
        "     # configs/hparams_main27.toml の runtime.active_profile = 'train'\n"
        "     uv run mjpython main27_train_hiderfront.py\n\n"
        "  1-b) 実行時オプションで train を選択（TOMLを編集しない）\n"
        "     uv run mjpython main27_train_hiderfront.py --profile train\n\n"
        "  2) 学習結果を確認する（デバッグ再生）\n"
        "     # configs/hparams_main27.toml の runtime.active_profile = 'debug'\n"
        "     uv run mjpython main27_train_hiderfront.py\n\n"
        "  2-b) 実行時オプションで debug を選択（TOMLを編集しない）\n"
        "     uv run mjpython main27_train_hiderfront.py --profile debug\n\n"
        "  2-c) 複数プロファイルを連続実行（比較）\n"
        "     uv run mjpython main27_train_hiderfront.py --compare-profiles train_wall_only,train_no_wall_stick,train_base_reward\n\n"
        "  3) ヘルプと実行例を表示\n"
        "     uv run mjpython main27_train_hiderfront.py -h\n"
    )
    parser = argparse.ArgumentParser(
        description="HideAndSeek Transformer v27 runner (hider front)",
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

# =====================
# グローバル設定の初期化
# =====================
CLI_PROFILE_OVERRIDE = _cli_profile_override_from_argv()
CLI_COMPARE_PROFILES = _cli_compare_profiles_from_argv()
AVAILABLE_RUNTIME_PROFILES = _get_available_runtime_profiles(CONFIG_PATH)
RUNTIME_OVERRIDES, RUN_PROFILE, RUN_PROFILE_SOURCE = _load_runtime_config(
    CONFIG_PATH, CLI_PROFILE_OVERRIDE, AVAILABLE_RUNTIME_PROFILES
)
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
MODEL_POLICY_DETERMINISTIC = _cfg("model_policy_deterministic", hp.get("model_policy_deterministic", "0"), _to_bool, RUNTIME_OVERRIDES)
DEBUG_SYMMETRIC_HIDER_POLICY = _cfg("debug_symmetric_hider_policy", hp.get("debug_symmetric_hider_policy", "0"), _to_bool, RUNTIME_OVERRIDES)
TRAIN_OTHER_HIDER_POLICY = _cfg("train_other_hider_policy", hp.get("train_other_hider_policy", "rule"), str, RUNTIME_OVERRIDES)
TRAIN_OTHER_SEEKER_POLICY = _cfg("train_other_seeker_policy", hp.get("train_other_seeker_policy", "rule"), str, RUNTIME_OVERRIDES)
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
# ...（他の定数も必要に応じてmain27_train_final.pyからコピーしてください）...

# ...existing code (設定・定数・関数定義などはmain27_train_final.pyと同じ)...

def reset_env_with_hider_front(env):
    obs, info = env.reset()
    s_key = env.learnable_agent_key
    s_bid = env.body_ids[s_key]
    s_pos = env.data.xpos[s_bid][:2].copy()
    try:
        rot_jid = env.qpos_indices[s_key]['rot']
        qadr = env.model.jnt_qposadr[rot_jid]
        yaw = float(env.data.qpos[qadr])
    except Exception:
        try:
            q = env.data.xquat[s_bid]
            q0, q1, q2, q3 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            yaw = float(np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3)))
        except Exception:
            yaw = 0.0
    hx = float(s_pos[0] + 1.5 * math.cos(yaw))
    hy = float(s_pos[1] + 1.5 * math.sin(yaw))
    h_key = env.hider_keys[0]
    set_hider_pos_fixed(env, h_key, hx, hy, z=0.8)
    try:
        import mujoco
        mujoco.mj_forward(env.model, env.data)
    except Exception:
        pass
    return obs, info

############################################################
# main27_train_final.py からのコア関数・クラス・定数移植
############################################################

# --- 必要な定数（既に定義済みのものは重複しないよう注意） ---
RW_SEEK_VISIBLE_BONUS = _cfg("rw_seek_visible_bonus", "0.03", float, RUNTIME_OVERRIDES)
RW_STILL_SPEED_THRESHOLD = _cfg("rw_still_speed_threshold", "0.1", float, RUNTIME_OVERRIDES)
RW_MOVE_SAT_PENALTY = _cfg("rw_move_sat_penalty", "0.02", float, RUNTIME_OVERRIDES)
RW_TURN_SAT_PENALTY = _cfg("rw_turn_sat_penalty", "0.01", float, RUNTIME_OVERRIDES)
RW_MOVE_CTRL_COST = _cfg("rw_move_ctrl_cost", "0.001", float, RUNTIME_OVERRIDES)
RW_MOVE_INCENTIVE = _cfg("rw_move_incentive", "0.02", float, RUNTIME_OVERRIDES)
RW_IDLE_PENALTY = _cfg("rw_idle_penalty", "0.03", float, RUNTIME_OVERRIDES)
RW_WALL_AVOID_PENALTY = _cfg("rw_wall_avoid_penalty", "0.15", float, RUNTIME_OVERRIDES)
RW_WALL_STICK_PENALTY = _cfg("rw_wall_stick_penalty", "0.15", float, RUNTIME_OVERRIDES)
RW_HIDE_VISIBLE_NEAR_PENALTY = _cfg("rw_hide_visible_near_penalty", "0.05", float, RUNTIME_OVERRIDES)
RW_HIDE_SEEKER_GAZE_PENALTY = _cfg("rw_hide_seeker_gaze_penalty", "0.03", float, RUNTIME_OVERRIDES)
RW_SEEK_GAZE_REWARD = _cfg("c", "0.03", float, RUNTIME_OVERRIDES)
IDLE_SPEED_THRESHOLD = _cfg("idle_speed_threshold", "0.1", float, RUNTIME_OVERRIDES)
MOVE_INCENTIVE_SPEED_THRESHOLD = _cfg("move_incentive_speed_threshold", "0.3", float, RUNTIME_OVERRIDES)
MOVE_SAT_THRESHOLD = _cfg("move_sat_threshold", "1.0", float, RUNTIME_OVERRIDES)
TURN_SAT_THRESHOLD = _cfg("turn_sat_threshold", "1.0", float, RUNTIME_OVERRIDES)
AGENT_RADIUS = _cfg("agentl_radius", "0.4", float, RUNTIME_OVERRIDES)
WALL_NEAR_MARGIN = _cfg("wall_near_margin", "0.4", float, RUNTIME_OVERRIDES)
AGENT_HEAD_CLEARANCE = _cfg("agent_head_clearance", "0.2", float, RUNTIME_OVERRIDES)

# --- 必要なクラス・関数 ---

# ObsHistory, SingleVecWrapper, run_train_vector, run_debug_or_playback などを main27_train_final.py から全文コピー
# ここでは省略せず、main27_train_final.pyの該当部分を正確に貼り付けてください

# 例: 学習ループ内のenv.reset()をreset_env_with_hider_front(env)に置き換えてください
# for episode in range(num_episodes):
#     obs, info = reset_env_with_hider_front(env)
#     ...
