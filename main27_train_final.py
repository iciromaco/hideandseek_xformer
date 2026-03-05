# main27_train_final.py v6.0

import math
import os
import sys
import time
import traceback
from functools import partial

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
except Exception:
    wandb = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from agents.scripted_agents import RuleBasedHider, RuleBasedSeeker
from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2


LOCAL_OVERRIDES = {
    # ここを編集すれば環境変数なしで設定できます。
    # 例:
    #"TRAIN_MODE": True,
    #"USE_VIEWER": True,
    #"TRAINING_TARGET": "seeker",
    #"N_SEEKERS": 1,
    #"N_HIDERS": 2,
    #"N_BOXES": 2,
    #"N_RAMPS": 1,
}

# ここを "debug" / "train" で切り替えると主要設定を一括変更できます。
# 優先順位: LOCAL_OVERRIDES > 環境変数 > RUN_PROFILE > デフォルト
RUN_PROFILE = "train"  # "debug" / "train"

PROFILE_OVERRIDES = {
    "debug": {
        "TRAIN_MODE": False,
        # "USE_VIEWER": True,
        "DEBUG_HIDER_POLICY": "model_if_available",
        "DEBUG_SEEKER_POLICY": "model_if_available",
        # "NPC_ONLY_DEBUG": False,
        "PLAY_EPISODES": 100,
        # "AUTO_TUNE_HPARAMS": False,
        # "RAMP_CHECK_VIEWER": False
    },
    "train": {
        "TRAIN_MODE": True,
        "USE_VIEWER": False,
        "TRAINING_TARGET": "hider",
        "TRAIN_OTHER_HIDER_POLICY": "model_if_available",
        "TRAIN_OTHER_SEEKER_POLICY": "rule",
        "NPC_ONLY_DEBUG": False,
        "AUTO_TUNE_HPARAMS": True,
        "NUM_ENVS": "10",
        "TOTAL_TIMESTEPS":300000,
    },
}


def _to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg(name, default, cast=None):
    profile_map = PROFILE_OVERRIDES.get(RUN_PROFILE, {})
    if name in LOCAL_OVERRIDES:
        raw = LOCAL_OVERRIDES[name]
    elif name in os.environ:
        raw = os.environ[name]
    elif name in profile_map:
        raw = profile_map[name]
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
    "show_turn_lines": SHOW_TURN_LINES,
}

SEQ_LEN = _cfg("SEQ_LEN", "8", int)
HIDDEN_DIM = _cfg("HIDDEN_DIM", "128", int)
NUM_ENVS = _cfg("NUM_ENVS", "4", int)

WORKER_SHUTDOWN_ERRORS = (EOFError, BrokenPipeError, ConnectionResetError)

TOTAL_TIMESTEPS = _cfg("TOTAL_TIMESTEPS", "300000", int)
ROLLOUT_STEPS = _cfg("ROLLOUT_STEPS", "128", int)
UPDATE_EPOCHS = _cfg("UPDATE_EPOCHS", "4", int)
MINIBATCH_SIZE = _cfg("MINIBATCH_SIZE", "64", int)
LEARNING_RATE = _cfg("LEARNING_RATE", "3e-4", float)
GAMMA = _cfg("GAMMA", "0.99", float)
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


def env_signature(config):
    return (
        f"s{config['n_seekers']}_h{config['n_hiders']}_"
        f"b{config['n_boxes']}_r{config['n_ramps']}"
    )


def model_path_for_config(target, config):
    sig = env_signature(config)
    os.makedirs("checkpoints", exist_ok=True)
    return os.path.join("checkpoints", f"HNS_V27_{target}_{sig}.pt")


POLICY_RULE = "rule"
POLICY_MODEL_IF_AVAILABLE = "model_if_available"


def _wandb_include_code_file(path, root):
    rel = os.path.relpath(path, root).replace("\\", "/")
    if rel in {"pyproject.toml", "README.md"}:
        return True
    if not rel.endswith(".py"):
        return False
    if rel.startswith("src/"):
        return True
    name = os.path.basename(rel)
    return name.startswith("main")


def init_wandb_run(hp, model_path, training_target):
    if not TRAIN_MODE or not WANDB_ENABLED:
        return None
    if wandb is None:
        print("wandb not available. Continue without wandb logging.")
        return None

    env_sig = env_signature(ENV_CONFIG)
    run_name = WANDB_RUN_NAME or f"v27_{training_target}_{env_sig}"
    try:
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=(WANDB_ENTITY or None),
            name=run_name,
            mode=WANDB_MODE,
            config={
                "training_target": training_target,
                "mode": MODE,
                "env_signature": env_sig,
                "num_envs": NUM_ENVS,
                "seq_len": SEQ_LEN,
                "hidden_dim": HIDDEN_DIM,
                "use_custom_reward": USE_CUSTOM_REWARD,
                "model_path": model_path,
                **hp,
            },
            reinit=True,
        )
        if WANDB_LOG_CODE:
            print("wandb: uploading code snapshot...")
            run.log_code(
                root=WANDB_CODE_ROOT,
                include_fn=_wandb_include_code_file,
            )
            print("wandb: code snapshot uploaded")
        run.log({"run_initialized": 1}, step=0)
        run.define_metric("global_step")
        run.define_metric("train/*", step_metric="global_step")
        run.define_metric("env/*", step_metric="global_step")
        run.define_metric("system/*", step_metric="global_step")
        run.define_metric("eval/*", step_metric="global_step")
        if getattr(run, "url", None):
            print(f"wandb run url: {run.url}")
        return run
    except Exception as exc:
        print(f"wandb init failed ({exc}). Continue without wandb logging.")
        return None


def select_hparams(config, target):
    hp = {
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
    if not AUTO_TUNE_HPARAMS:
        return hp

    sig = env_signature(config)
    target_defaults = {
        "seeker": {
            "learning_rate": min(hp["learning_rate"], 2.2e-4),
            "clip_coef": 0.18,
            "ent_coef": max(hp["ent_coef"], 6.0e-4),
            "update_epochs": max(hp["update_epochs"], 4),
        },
        "hider": {
            "learning_rate": min(hp["learning_rate"], 2.0e-4),
            "clip_coef": 0.20,
            "ent_coef": max(hp["ent_coef"], 1.2e-3),
            "update_epochs": max(hp["update_epochs"], 5),
        },
    }
    hp.update(target_defaults.get(target, {}))

    target_presets = {
        "seeker": {
            "s1_h2_b2_r1": {
                "total_timesteps": 600_000,
                "rollout_steps": 192,
                "update_epochs": 5,
                "minibatch_size": 96,
                "learning_rate": 2.0e-4,
                "clip_coef": 0.17,
                "ent_coef": 6.0e-4,
                "log_interval": 5,
                "save_interval": 25,
            },
            "s2_h3_b2_r3": {
                "total_timesteps": 900_000,
                "rollout_steps": 256,
                "update_epochs": 4,
                "minibatch_size": 128,
                "learning_rate": 1.7e-4,
                "clip_coef": 0.18,
                "ent_coef": 7.0e-4,
                "log_interval": 5,
                "save_interval": 25,
            },
        },
        "hider": {
            "s1_h2_b2_r1": {
                "total_timesteps": 700_000,
                "rollout_steps": 192,
                "update_epochs": 6,
                "minibatch_size": 96,
                "learning_rate": 1.8e-4,
                "clip_coef": 0.20,
                "ent_coef": 1.4e-3,
                "log_interval": 5,
                "save_interval": 25,
            },
            "s2_h3_b2_r3": {
                "total_timesteps": 1_000_000,
                "rollout_steps": 256,
                "update_epochs": 5,
                "minibatch_size": 128,
                "learning_rate": 1.6e-4,
                "clip_coef": 0.20,
                "ent_coef": 1.6e-3,
                "log_interval": 5,
                "save_interval": 25,
            },
        },
    }
    presets = target_presets.get(target, {})
    if sig in presets:
        hp.update(presets[sig])
    else:
        complexity = (
            config["n_seekers"]
            + config["n_hiders"]
            + config["n_boxes"]
            + config["n_ramps"]
        )
        if complexity >= 8:
            hp["rollout_steps"] = max(hp["rollout_steps"], 192)
            hp["minibatch_size"] = max(hp["minibatch_size"], 96)
            hp["learning_rate"] = min(hp["learning_rate"], 2.0e-4)
            hp["ent_coef"] = max(hp["ent_coef"], 1.0e-3)
            hp["total_timesteps"] = max(hp["total_timesteps"], 700_000)
            if target == "hider":
                hp["clip_coef"] = max(hp["clip_coef"], 0.20)
                hp["ent_coef"] = max(hp["ent_coef"], 1.4e-3)
            else:
                hp["clip_coef"] = min(hp["clip_coef"], 0.19)

    if hp["rollout_steps"] % hp["minibatch_size"] != 0:
        hp["minibatch_size"] = max(32, min(hp["rollout_steps"], hp["minibatch_size"]))
        while hp["rollout_steps"] % hp["minibatch_size"] != 0 and hp["minibatch_size"] > 32:
            hp["minibatch_size"] //= 2

    return hp


def evaluate_fixed_ramp_climb(
    mode,
    target,
    agent,
    device,
    episodes=3,
    max_steps=220,
):
    eval_cfg = dict(ENV_CONFIG)
    eval_cfg["show_turn_lines"] = False
    eval_env = build_env(mode, target, eval_cfg, render_mode=None)
    eval_env.prep_steps = 0
    peak_heights = []
    peak_progress = []
    success_count = 0
    try:
        obs_dim = eval_env.observation_space.shape[0]
        expected_obs_dim = int(getattr(agent, "obs_dim", obs_dim))
        if obs_dim != expected_obs_dim:
            print(
                "Ramp check skipped: obs_dim mismatch "
                f"(env={obs_dim}, model={expected_obs_dim})"
            )
            return None
        history = ObsHistory(1, SEQ_LEN, obs_dim, device)
        learn_key = eval_env.learnable_agent_key
        body_id = eval_env.body_ids[learn_key]
        qidx = eval_env.qpos_indices[learn_key]

        for _ in range(max(episodes, 1)):
            obs, _ = eval_env.reset()

            rid = eval_env.ramp_ids[0]
            rkey = "ramp1"
            rpos = eval_env.data.xpos[rid][:2].copy()
            quat = eval_env.data.xquat[rid]
            yaw = math.atan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
            )
            up = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            side = np.array([-up[1], up[0]], dtype=np.float32)

            start_xy = rpos - 1.0 * up + np.random.uniform(-0.15, 0.15) * side
            start_prog = float(np.dot(start_xy - rpos, up))
            prog_sign = -1.0 if start_prog > 0.0 else 1.0
            qx = eval_env.model.jnt_qposadr[qidx["x"]]
            qy = eval_env.model.jnt_qposadr[qidx["y"]]
            qz = eval_env.model.jnt_qposadr[qidx["z"]]
            qr = eval_env.model.jnt_qposadr[qidx["rot"]]
            vx = eval_env.model.jnt_dofadr[qidx["x"]]
            vy = eval_env.model.jnt_dofadr[qidx["y"]]
            vz = eval_env.model.jnt_dofadr[qidx["z"]]
            vr = eval_env.model.jnt_dofadr[qidx["rot"]]
            eval_env.data.qpos[qx] = float(start_xy[0])
            eval_env.data.qpos[qy] = float(start_xy[1])
            eval_env.data.qpos[qz] = 0.55
            eval_env.data.qpos[qr] = float(yaw)
            eval_env.data.qvel[vx] = 0.0
            eval_env.data.qvel[vy] = 0.0
            eval_env.data.qvel[vz] = 0.0
            eval_env.data.qvel[vr] = 0.0

            qadr, vadr = eval_env._obj_addr(rkey)
            st = eval_env.object_state[rkey]
            st["mode"] = "locked"
            st["owner"] = "eval"
            st["locked_pose"] = eval_env.data.qpos[qadr:qadr + 7].copy()
            eval_env.data.qvel[vadr:vadr + 6] = 0.0
            mujoco.mj_forward(eval_env.model, eval_env.data)

            obs = eval_env._normalize_obs(eval_env._get_obs(eval_env.learnable_agent_index))
            history.prime_single(obs)

            ep_peak_h = float(eval_env.data.xpos[body_id][2])
            ep_start_h = ep_peak_h
            ep_peak_prog = 0.0
            ep_peak_top = -1e9
            ep_best_lat = 1e9
            ep_success = False

            for _ in range(max_steps):
                with torch.no_grad():
                    seq = history.get()
                    act = agent.get_action_and_value(seq)[0].cpu().numpy().reshape(-1)
                next_obs, _, term, trun, _ = eval_env.step(act)
                apos = eval_env.data.xpos[body_id]
                rel = apos[:2] - rpos
                prog = float(np.dot(rel, up))
                lat = float(np.dot(rel, side))
                directed_prog = prog_sign * (prog - start_prog)
                top_side = prog_sign * prog
                h = float(apos[2])
                rise = h - ep_start_h
                ep_peak_h = max(ep_peak_h, h)
                ep_peak_prog = max(ep_peak_prog, directed_prog)
                ep_peak_top = max(ep_peak_top, top_side)
                ep_best_lat = min(ep_best_lat, abs(lat))
                if (
                    directed_prog >= RAMP_SUCCESS_PROG
                    and top_side >= RAMP_SUCCESS_TOP
                    and abs(lat) <= RAMP_SUCCESS_LAT
                    and rise >= RAMP_SUCCESS_Z_RISE
                ):
                    ep_success = True
                history.update_single(next_obs)
                if term or trun:
                    break

            success_count += int(ep_success)
            peak_heights.append(ep_peak_h)
            peak_progress.append(ep_peak_prog)

        episodes_n = max(len(peak_heights), 1)
        return {
            "success_rate": success_count / episodes_n,
            "peak_height_mean": float(np.mean(peak_heights)) if peak_heights else 0.0,
            "peak_progress_mean": float(np.mean(peak_progress)) if peak_progress else 0.0,
            "episodes": episodes_n,
        }
    finally:
        eval_env.close()


def run_ramp_check_viewer(mode, target, agent, device, episodes=3, max_steps=220):
    eval_cfg = dict(ENV_CONFIG)
    eval_cfg["show_turn_lines"] = True
    eval_env = build_env(mode, target, eval_cfg, render_mode="human")
    eval_env.prep_steps = 0
    try:
        obs_dim = eval_env.observation_space.shape[0]
        expected_obs_dim = int(getattr(agent, "obs_dim", obs_dim))
        if obs_dim != expected_obs_dim:
            print(
                "Ramp viewer check skipped: obs_dim mismatch "
                f"(env={obs_dim}, model={expected_obs_dim})"
            )
            return

        history = ObsHistory(1, SEQ_LEN, obs_dim, device)
        learn_key = eval_env.learnable_agent_key
        body_id = eval_env.body_ids[learn_key]
        qidx = eval_env.qpos_indices[learn_key]

        print("--- Ramp Viewer Check Start ---")
        for ep in range(max(episodes, 1)):
            obs, _ = eval_env.reset()
            rid = eval_env.ramp_ids[0]
            rkey = "ramp1"
            rpos = eval_env.data.xpos[rid][:2].copy()
            quat = eval_env.data.xquat[rid]
            yaw = math.atan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
            )
            up = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            side = np.array([-up[1], up[0]], dtype=np.float32)

            start_xy = rpos - 1.75 * up + np.random.uniform(-0.08, 0.08) * side
            start_prog = float(np.dot(start_xy - rpos, up))
            prog_sign = -1.0 if start_prog > 0.0 else 1.0
            climb_dir = up * prog_sign
            qx = eval_env.model.jnt_qposadr[qidx["x"]]
            qy = eval_env.model.jnt_qposadr[qidx["y"]]
            qz = eval_env.model.jnt_qposadr[qidx["z"]]
            qr = eval_env.model.jnt_qposadr[qidx["rot"]]
            vx = eval_env.model.jnt_dofadr[qidx["x"]]
            vy = eval_env.model.jnt_dofadr[qidx["y"]]
            vz = eval_env.model.jnt_dofadr[qidx["z"]]
            vr = eval_env.model.jnt_dofadr[qidx["rot"]]
            eval_env.data.qpos[qx] = float(start_xy[0])
            eval_env.data.qpos[qy] = float(start_xy[1])
            eval_env.data.qpos[qz] = 0.55
            eval_env.data.qpos[qr] = float(yaw)
            eval_env.data.qvel[vx] = 0.0
            eval_env.data.qvel[vy] = 0.0
            eval_env.data.qvel[vz] = 0.0
            eval_env.data.qvel[vr] = 0.0

            qadr, vadr = eval_env._obj_addr(rkey)
            st = eval_env.object_state[rkey]
            st["mode"] = "locked"
            st["owner"] = "eval"
            st["locked_pose"] = eval_env.data.qpos[qadr:qadr + 7].copy()
            eval_env.data.qvel[vadr:vadr + 6] = 0.0
            mujoco.mj_forward(eval_env.model, eval_env.data)

            obs = eval_env._normalize_obs(eval_env._get_obs(eval_env.learnable_agent_index))
            history.prime_single(obs)
            eval_env.render()

            peak_prog = 0.0
            peak_top = -1e9
            best_lat = 1e9
            start_h = float(eval_env.data.xpos[body_id][2])
            peak_rise = 0.0
            success = False
            switched = False
            stall_count = 0
            unstick_ticks = 0
            for t in range(max_steps):
                if eval_env.viewer and not eval_env.viewer.is_running():
                    print("Viewer closed.")
                    return
                if RAMP_CHECK_FORCE_UPHILL:
                    if unstick_ticks > 0:
                        zig = 1.0 if (unstick_ticks // 4) % 2 == 0 else -1.0
                        action = np.array([1.0, 0.85 * zig, 0.0, 0.0], dtype=np.float32)
                        unstick_ticks -= 1
                    else:
                        rot = float(eval_env.data.qpos[qr])
                        apos_xy = eval_env.data.xpos[body_id][:2]
                        rel_xy = apos_xy - rpos
                        lat_now = float(np.dot(rel_xy, side))
                        prog_now = prog_sign * float(np.dot(rel_xy, up))
                        if prog_now < 0.15:
                            target_xy = rpos - 0.20 * climb_dir - 0.95 * lat_now * side
                        else:
                            target_xy = rpos + 1.35 * climb_dir - 0.75 * lat_now * side
                        to_target = target_xy - apos_xy
                        target_yaw = math.atan2(float(to_target[1]), float(to_target[0]))
                        yaw_err = target_yaw - rot
                        yaw_err = (yaw_err + math.pi) % (2.0 * math.pi) - math.pi
                        turn = float(np.clip(2.4 * yaw_err, -1.0, 1.0))
                        fwd = 1.0 if abs(yaw_err) < 1.15 else 0.35
                        action = np.array([fwd, turn, 0.0, 0.0], dtype=np.float32)
                else:
                    with torch.no_grad():
                        seq = history.get()
                        action = agent.get_action_and_value(seq)[0].cpu().numpy().reshape(-1)
                next_obs, _, term, trun, _ = eval_env.step(action)
                apos = eval_env.data.xpos[body_id]
                rel = apos[:2] - rpos
                prog = float(np.dot(rel, up))
                lat = float(np.dot(rel, side))
                directed_prog = prog_sign * (prog - start_prog)
                top_side = prog_sign * prog
                peak_prog = max(peak_prog, directed_prog)
                peak_top = max(peak_top, top_side)
                best_lat = min(best_lat, abs(lat))
                rise = float(apos[2] - start_h)
                peak_rise = max(peak_rise, rise)
                vel_xy = float(np.linalg.norm(eval_env.data.qvel[vx:vy + 1]))
                if RAMP_CHECK_FORCE_UPHILL:
                    if t > 24 and directed_prog < 0.35 and vel_xy < 0.08:
                        stall_count += 1
                    else:
                        stall_count = 0
                    if stall_count >= 16:
                        unstick_ticks = 22
                        stall_count = 0
                        print(f"RampViewer Ep {ep + 1}: edge-stuck assist")
                if RAMP_CHECK_FORCE_UPHILL and (not switched) and t >= 60 and peak_prog < 0.35:
                    climb_dir = -climb_dir
                    prog_sign = -prog_sign
                    switched = True
                    print(f"RampViewer Ep {ep + 1}: switching climb direction")
                if (
                    directed_prog >= RAMP_SUCCESS_PROG
                    and top_side >= RAMP_SUCCESS_TOP
                    and abs(lat) <= RAMP_SUCCESS_LAT
                    and rise >= RAMP_SUCCESS_Z_RISE
                ):
                    success = True
                history.update_single(next_obs)
                eval_env.render()
                time.sleep(0.02)
                if term or trun:
                    break

            print(
                f"RampViewer Ep {ep + 1}/{episodes}: "
                f"success={int(success)} peak_prog={peak_prog:.2f} "
                f"peak_top={peak_top:.2f} best_lat={best_lat:.2f} "
                f"peak_rise={peak_rise:.2f}"
            )
        print("--- Ramp Viewer Check End ---")
    finally:
        eval_env.close()


def compute_custom_reward(obs, action, base_reward, idx, target_team="seeker"):
    enemy_visible = any(obs[en.VISIBLE] > 0.5 for en in idx.OTHERS)
    min_enemy_dist = 999.0
    for en in idx.OTHERS:
        d = math.sqrt(obs[en.REL_X] ** 2 + obs[en.REL_Y] ** 2)
        if d < min_enemy_dist:
            min_enemy_dist = d
    speed = math.sqrt(obs[idx.SELF.VEL_X] ** 2 + obs[idx.SELF.VEL_Y] ** 2)

    bonus = 0.0
    if target_team == "seeker":
        if enemy_visible:
            bonus += 0.05
        bonus += 0.02 / (min_enemy_dist + 0.5)
        if speed < 0.01:
            bonus -= 0.02
    control_cost = -0.005 * np.sum(np.square(action))
    return base_reward + bonus + control_cost


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


def configure_team_policy_modes(env, vec_envs, ref_env, runtime_target, primary_state_dict):
    if ref_env is None:
        return

    if TRAIN_MODE:
        hider_mode = _normalize_policy_mode(TRAIN_OTHER_HIDER_POLICY)
        seeker_mode = _normalize_policy_mode(TRAIN_OTHER_SEEKER_POLICY)
    else:
        hider_mode = _normalize_policy_mode(DEBUG_HIDER_POLICY)
        seeker_mode = _normalize_policy_mode(DEBUG_SEEKER_POLICY)

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


def run_debug_or_playback(env, agent, device, model_loaded):
    idx = env.idx
    obs_dim = env.observation_space.shape[0]
    history = ObsHistory(1, SEQ_LEN, obs_dim, device)

    target_npc = None
    if NPC_ONLY_DEBUG:
        target_npc = RuleBasedSeeker() if env.target == "seeker" else RuleBasedHider()

    print("--- Start Simulation ---")
    print(f"Physics Step: {env.model.opt.timestep * 5}s")

    for episode in range(PLAY_EPISODES):
        obs, _ = env.reset()
        history.prime_single(obs)
        done = False
        ep_reward = 0.0
        step_count = 0
        lock_events = 0
        grab_events = 0
        lock_events_window = 0
        grab_events_window = 0
        sat_count = 0
        boosted_count = 0

        if USE_VIEWER:
            env.render()

        while not done:
            if USE_VIEWER and env.viewer and not env.viewer.is_running():
                return

            if NPC_ONLY_DEBUG and target_npc is not None:
                action = target_npc.get_action(obs, idx)
            elif model_loaded:
                with torch.no_grad():
                    seq = history.get()
                    out = agent.get_action_and_value(seq)
                    action = out[0].cpu().numpy().reshape(-1)
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)

            next_obs, base_r, term, trun, info = env.step(action)
            reward = (
                compute_custom_reward(obs, action, base_r, idx, env.target)
                if USE_CUSTOM_REWARD else base_r
            )

            history.update(next_obs)
            lock_events += int(info.get("lock_event", 0))
            grab_events += int(info.get("grab_event", 0))
            lock_events_window += int(info.get("lock_event", 0))
            grab_events_window += int(info.get("grab_event", 0))
            boosted_count += int(info.get("dbg_boosted_agents", 0) > 0)
            if float(np.max(np.abs(action[:2]))) >= 0.9:
                sat_count += 1

            if USE_VIEWER:
                env.render()
                time.sleep(0.025)

            obs = next_obs
            ep_reward += reward
            step_count += 1
            done = bool(term or trun)

            if step_count % 100 == 0:
                print(
                    "Ep "
                    f"{episode} - Step {step_count} | "
                    f"Steer={action[1]:.2f} "
                    f"LockBtnMax={info.get('dbg_lock_btn_max', 0.0):.2f} "
                    f"GrabBtnMax={info.get('dbg_grab_btn_max', 0.0):.2f} "
                    f"LockPressed={int(info.get('dbg_lock_pressed', 0))} "
                    f"GrabPressed={int(info.get('dbg_grab_pressed', 0))} "
                    f"LockTarget={int(info.get('dbg_lock_target', 0))} "
                    f"GrabTarget={int(info.get('dbg_grab_target', 0))} "
                    f"LockEvt={int(info.get('lock_event', 0))} "
                    f"GrabEvt={int(info.get('grab_event', 0))} "
                    f"WinLock={lock_events_window} "
                    f"WinGrab={grab_events_window} "
                    f"CumLock={lock_events} "
                    f"CumGrab={grab_events} "
                    f"BoxMove={int(info.get('dbg_box_moving_count', 0))} "
                    f"RampMove={int(info.get('dbg_ramp_moving_count', 0))} "
                    f"MaxBoxV={float(info.get('dbg_max_box_speed', 0.0)):.2f} "
                    f"MaxRampV={float(info.get('dbg_max_ramp_speed', 0.0)):.2f} "
                    f"BlockedRamp={int(info.get('dbg_blocked_ramp_count', 0))} "
                    f"Boosted={int(info.get('dbg_boosted_agents', 0))}"
                )
                lock_events_window = 0
                grab_events_window = 0

        print(
            f"Ep {episode} Finished. Steps={step_count}, Reward={ep_reward:.2f}, "
            f"LockEvents={lock_events}, GrabEvents={grab_events}, "
            f"ActSatRatio={sat_count / max(step_count, 1):.2f}, "
            f"BoostUseRatio={boosted_count / max(step_count, 1):.2f}"
        )


def run_train(
    env,
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
            entropy_sum = 0.0
            entropy_count = 0

            for t in range(hp["rollout_steps"]):
                if USE_VIEWER and env.viewer and not env.viewer.is_running():
                    print("Viewer closed. Stop training loop.")
                    return
                seq = history.get()
                rollout_obs[t] = seq[0]

                with torch.no_grad():
                    action, logp, _, value = agent.get_action_and_value(seq)

                action_np = action.cpu().numpy().reshape(-1)
                next_obs, base_r, term, trun, info = env.step(action_np)
                reward = (
                    compute_custom_reward(obs, action_np, base_r, idx, env.target)
                    if USE_CUSTOM_REWARD else base_r
                )

                done = bool(term or trun)
                done_last = float(done)
                global_step += 1

                rollout_actions[t] = action[0]
                rollout_logp[t] = logp[0]
                rollout_rewards[t] = float(reward)
                rollout_dones[t] = done_last
                rollout_values[t] = value.view(-1)[0]

                lock_evt_sum += int(info.get("lock_event", 0))
                grab_evt_sum += int(info.get("grab_event", 0))
                max_box_speed = max(max_box_speed, float(info.get("dbg_max_box_speed", 0.0)))
                max_ramp_speed = max(max_ramp_speed, float(info.get("dbg_max_ramp_speed", 0.0)))
                blocked_ramp_sum += int(info.get("dbg_blocked_ramp_count", 0))
                rewards_sum += float(reward)
                if not bool(info.get("is_detected", False)):
                    hidden_steps += 1

                if done:
                    obs, _ = env.reset()
                    history.prime_single(obs)
                    if USE_VIEWER:
                        env.render()
                else:
                    obs = next_obs
                    history.update_single(obs)

                if USE_VIEWER:
                    env.render()

            with torch.no_grad():
                next_value = agent.get_value(history.get()).view(-1)[0]

            advantages = torch.zeros(hp["rollout_steps"], device=device)
            lastgaelam = 0.0
            for t in reversed(range(hp["rollout_steps"])):
                if t == hp["rollout_steps"] - 1:
                    next_nonterminal = 1.0 - done_last
                    next_vals = next_value
                else:
                    next_nonterminal = 1.0 - rollout_dones[t + 1]
                    next_vals = rollout_values[t + 1]
                delta = rollout_rewards[t] + hp["gamma"] * next_vals * next_nonterminal - rollout_values[t]
                lastgaelam = delta + hp["gamma"] * hp["gae_lambda"] * next_nonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + rollout_values

            b_obs = rollout_obs
            b_actions = rollout_actions
            b_logp = rollout_logp
            b_adv = advantages
            b_ret = returns

            inds = np.arange(hp["rollout_steps"])
            for _ in range(hp["update_epochs"]):
                np.random.shuffle(inds)
                for start in range(0, hp["rollout_steps"], hp["minibatch_size"]):
                    mb_idx = inds[start:start + hp["minibatch_size"]]

                    _, new_logp, entropy, new_value = agent.get_action_and_value(
                        b_obs[mb_idx], b_actions[mb_idx]
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
            hide_rate = hidden_steps / max(hp["rollout_steps"], 1)
            avg_reward = rewards_sum / hp["rollout_steps"]
            avg_blocked_ramp = blocked_ramp_sum / hp["rollout_steps"]
            avg_entropy = entropy_sum / max(entropy_count, 1)
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
                    "env/lock_events": lock_evt_sum,
                    "env/grab_events": grab_evt_sum,
                    "env/max_box_speed": max_box_speed,
                    "env/max_ramp_speed": max_ramp_speed,
                    "env/avg_blocked_ramp": avg_blocked_ramp,
                    "system/num_envs": 1,
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
                ramp_text = ""
                if last_ramp_eval is not None:
                    ramp_text = (
                        f" RampOK={last_ramp_eval['success_rate']:.2f}"
                        f" RampProg={last_ramp_eval['peak_progress_mean']:.2f}"
                    )
                print(
                    f"Upd {update}/{num_updates} Step={global_step} "
                    f"SPS={sps} HideRate={hide_rate:.2f} "
                    f"AvgR={avg_reward:.3f} Ent={avg_entropy:.4f} "
                    f"LockEvt={lock_evt_sum} GrabEvt={grab_evt_sum} "
                    f"MaxBoxV={max_box_speed:.2f} MaxRampV={max_ramp_speed:.2f} "
                    f"AvgBlockedRamp={avg_blocked_ramp:.2f}"
                    f"{ramp_text}"
                )

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
        for update in range(1, num_updates + 1):
            lock_evt_sum = 0
            grab_evt_sum = 0
            max_box_speed = 0.0
            max_ramp_speed = 0.0
            blocked_ramp_sum = 0
            hidden_steps = 0
            entropy_sum = 0.0
            entropy_count = 0

            done_last = np.zeros(num_envs, dtype=np.float32)
            rewards_sum = 0.0

            for t in range(hp["rollout_steps"]):
                seq = history.get()
                rollout_obs[t] = seq

                with torch.no_grad():
                    action, logp, _, value = agent.get_action_and_value(seq)

                action_np = action.cpu().numpy()
                next_obs, base_r, term, trun, info = envs.step(action_np)
                done_np = np.logical_or(term, trun).astype(np.float32)
                done_last = done_np

                reward_np = np.asarray(base_r, dtype=np.float32).copy()
                if USE_CUSTOM_REWARD:
                    for i in range(num_envs):
                        reward_np[i] = compute_custom_reward(
                            obs[i],
                            action_np[i],
                            reward_np[i],
                            idx,
                            ref_env.target,
                        )

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
            avg_reward = rewards_sum / (hp["rollout_steps"] * num_envs)
            avg_blocked_ramp = blocked_ramp_sum / (hp["rollout_steps"] * num_envs)
            avg_entropy = entropy_sum / max(entropy_count, 1)
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
                ramp_text = ""
                if last_ramp_eval is not None:
                    ramp_text = (
                        f" RampOK={last_ramp_eval['success_rate']:.2f}"
                        f" RampProg={last_ramp_eval['peak_progress_mean']:.2f}"
                    )
                print(
                    f"Upd {update}/{num_updates} Step={global_step} "
                    f"SPS={sps} HideRate={hide_rate:.2f} "
                    f"AvgR={avg_reward:.3f} Ent={avg_entropy:.4f} "
                    f"LockEvt={lock_evt_sum} GrabEvt={grab_evt_sum} "
                    f"MaxBoxV={max_box_speed:.2f} MaxRampV={max_ramp_speed:.2f} "
                    f"AvgBlockedRamp={avg_blocked_ramp:.2f}"
                    f"{ramp_text}"
                )

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
    runtime_target = _resolve_runtime_target()
    print("Initializing Environment...")
    print(f"Run profile: {RUN_PROFILE}")
    print(
        "Resolved flags: "
        f"TRAIN_MODE={TRAIN_MODE}, "
        f"USE_VIEWER={USE_VIEWER}, "
        f"NPC_ONLY_DEBUG={NPC_ONLY_DEBUG}, "
        f"RUNTIME_TARGET={runtime_target}, "
        f"DEBUG_HIDER_POLICY={DEBUG_HIDER_POLICY}, "
        f"DEBUG_SEEKER_POLICY={DEBUG_SEEKER_POLICY}, "
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
    if os.path.exists(model_path):
        try:
            agent.load_state_dict(torch.load(model_path, map_location=device))
            model_loaded = True
            print(f"Loaded model: {model_path}")
        except Exception as exc:
            print(f"Model load skipped ({exc})")

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


if __name__ == "__main__":
    run()