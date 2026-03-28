# main21_layer0_runner.py
# 演習第21回：サブサンプション・アーキテクチャ Layer 0 (逃走)
#
# 【修正(v21.56)】
# - 報酬ロジックの厳格化 (視界依存距離報酬):
#   - SeekerがHiderの視界内(FOV+遮蔽なし)に居る時のみ、距離増分報酬を与えるように変更。
#   - 見えていない敵から離れても報酬は 0 となり、エージェントに「敵を確認する」動機を与える。
# - 最適化構造の改善:
#   - visible_cache をエージェントごとに分離し、SeekerとHiderの視界判定が混ざらないように修正。
# - 高速化資産 (RayCastキャッシュ, ダブルバッファ, スレッド制限) は維持。
#
# 【実行準備】
# main18_optimization.py が同じフォルダに必要です。

import os
import platform
import sys

# --- 環境変数設定 (数値計算ライブラリの初期化前) ---
if platform.processor() != "arm":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb

# --- インポートパスの解決 ---
search_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(4):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)
    search_path = os.path.dirname(search_path)

# ベース設定のインポート
try:
    import main18_optimization as base_config
    from main18_optimization import Agent, HideAndSeekEnv
except ImportError:
    print("Error: main18_optimization.py not found in sys.path.")
    exit(1)

# ==========================================
# 1. 設定のオーバーライド
# ==========================================
EXPERIMENT_NAME = "HideAndSeek_Layer0_Runner"

EXECUTION_MODE = "TRAIN"
LOAD_EXISTING_MODELS = True
SAVE_MODEL = True
TRACK_WANDB = True
FIXED_SEED = None

TOTAL_TIMESTEPS = 5000000
NUM_ENVS = 10
NUM_STEPS = 128
LEARNING_RATE = 0.0001
ENT_COEF = 1e-5

# ★報酬設定
REWARD_DISTANCE_DIFF_SCALE = 0.5  # 3.4  # 視界内での逃走成功報酬
REWARD_SURVIVAL_SCALE = 1.0  # 17.0     # 生存報酬 (共通)
PENALTY_CAPTURE = -300.0  # 捕獲ペナルティ
PENALTY_STAGNATION_FORCE = -2.5  # 停滞ペナルティ

# 環境定数
FOV_DEG = 135
RAYCAST_CACHE_POS_THRESH = 0.05

# 親クラスの報酬無効化
base_config.REWARD_SURVIVAL = 0.0
base_config.REWARD_DISTANCE_COEFF = 0.0
base_config.PENALTY_CAPTURE = 0.0
base_config.PENALTY_STAGNATION = 0.0
base_config.REWARD_CAPTURE_BONUS = 0.0
base_config.MODEL_PATH_HIDER = f"{EXPERIMENT_NAME}_HIDER.pt"

device = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")


# ==========================================
# 2. 高速化データ構造
# ==========================================
class ObsHistory:
    def __init__(self, num_envs, seq_len, obs_dim, device):
        self.buffer = torch.zeros((num_envs, seq_len * 2, obs_dim), device=device)
        self.device = device
        self.ptr = 0
        self.seq_len = seq_len

    def reset(self):
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        self.buffer[:, self.ptr] = obs_tensor
        self.buffer[:, self.ptr + self.seq_len] = obs_tensor
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        return self.buffer[:, self.ptr : self.ptr + self.seq_len]


# ==========================================
# 3. 環境クラスの拡張 (Layer0RunnerEnv)
# ==========================================
class Layer0RunnerEnv(HideAndSeekEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.prev_dist_to_seeker = {}

        worker_device = torch.device("cpu")
        self.npc_obs_history = {
            0: ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, 53, worker_device),
            1: ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, 53, worker_device),
            2: ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, 53, worker_device),
        }

        self.raycast_cache = {}
        self.raycast_stats = {"hits": 0, "misses": 0}
        # ★修正: 視界キャッシュをエージェントIDごとに保持するように変更
        self.visible_cache = {0: {}, 1: {}, 2: {}}

    def reset_raycast_stats(self):
        self.raycast_stats = {"hits": 0, "misses": 0}

    def get_raycast_stats(self):
        return self.raycast_stats

    def _get_obs(self, agent_id):
        if agent_id == 0:
            b_id, p_pref = self.s0_body, "s"
        elif agent_id == 1:
            b_id, p_pref = self.h1_body, "h1"
        else:
            b_id, p_pref = self.h2_body, "h2"

        hp, hra = (
            self.data.xpos[b_id][:2],
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(f"{p_pref}_rot").id]],
        )
        c, s = np.cos(-hra), np.sin(-hra)
        rot_mat = np.array([[c, -s], [s, c]])
        dof = self.model.jnt_dofadr[self.model.joint(f"{p_pref}_x").id]
        h_raw_vel = self.data.qvel[dof : dof + 2]
        h_local_vel = rot_mat @ h_raw_vel
        self_s = np.concatenate([h_local_vel / 12.0, [hra, np.cos(hra), np.sin(hra)]])

        lidar = np.zeros(len(self.lidar_angles), dtype=np.float32)
        geomid = np.zeros(1, dtype=np.int32)
        for i, angle_offset in enumerate(self.lidar_angles):
            beam_dir = angle_offset + hra
            direction = np.array([np.cos(beam_dir), np.sin(beam_dir), 0.0], dtype=np.float64)
            dist = mujoco.mj_ray(
                self.model,
                self.data,
                np.array([hp[0], hp[1], 0.5], dtype=np.float64),
                direction,
                None,
                1,
                b_id,
                geomid,
            )
            lidar[i] = min(dist, 2.5) / 2.5 if dist != -1 else 1.0

        # --- エージェント固有の視界判定 ---
        my_vis_cache = self.visible_cache[agent_id]
        my_vis_cache.clear()
        targets = [
            self.box1_body,
            self.box2_body,
            self.ramp_body,
            self.h1_body,
            self.h2_body,
            self.s0_body,
        ]
        targets = [t for t in targets if t != b_id]

        target_dists = []
        for t in targets:
            d = np.linalg.norm(self.data.xpos[t][:2] - hp)
            target_dists.append((d, t))
        target_dists.sort(key=lambda x: x[0], reverse=True)

        for _, tid in target_dists:
            if tid in my_vis_cache:
                continue
            tp_vec = self.data.xpos[tid]
            target_pos = tp_vec[:2]
            diff = target_pos - hp
            dist = np.linalg.norm(diff)
            if dist < 0.1:
                is_in_fov = True
            else:
                angle = (np.arctan2(diff[1], diff[0]) - hra + np.pi) % (2 * np.pi) - np.pi
                is_in_fov = abs(angle) <= np.deg2rad(FOV_DEG / 2.0)
            if not is_in_fov:
                my_vis_cache[tid] = False
                continue

            cache_key = (agent_id, tid)
            hit_id = None
            should_raycast = True
            if cache_key in self.raycast_cache:
                c_hp, c_tp, c_hid = self.raycast_cache[cache_key]
                if np.linalg.norm(hp - c_hp) < RAYCAST_CACHE_POS_THRESH and np.linalg.norm(target_pos - c_tp) < RAYCAST_CACHE_POS_THRESH:
                    hit_id = c_hid
                    should_raycast = False
                    self.raycast_stats["hits"] += 1
            if should_raycast:
                self.raycast_stats["misses"] += 1
                dr = np.array([diff[0] / dist, diff[1] / dist, 0.0], dtype=np.float64)
                gid = np.zeros(1, dtype=np.int32)
                res = mujoco.mj_ray(
                    self.model,
                    self.data,
                    np.array([hp[0], hp[1], 0.5], dtype=np.float64),
                    dr,
                    None,
                    1,
                    b_id,
                    gid,
                )
                if res != -1:
                    h_body = self.model.geom_bodyid[gid[0]]
                    if h_body == tid:
                        hit_id = tid
                    elif res < dist - 0.4:
                        hit_id = h_body
                    else:
                        hit_id = tid
                self.raycast_cache[cache_key] = (hp.copy(), target_pos.copy(), hit_id)

            is_vis = hit_id == tid
            my_vis_cache[tid] = is_vis
            if not is_vis and hit_id is not None and hit_id in targets:
                my_vis_cache[hit_id] = True

        def get_rel_info(target_id, lock=None):
            if my_vis_cache.get(target_id, False):
                tp = self.data.xpos[target_id]
                rel_p = rot_mat @ (tp[:2] - hp) / 12.0
                q = self.data.xquat[target_id]
                yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                tv = self.data.qvel[self.model.body_jntadr[target_id] : self.model.body_jntadr[target_id] + 2] if self.model.body_jntadr[target_id] != -1 else np.zeros(2)
                rel_v = rot_mat @ (tv - self.data.qvel[dof : dof + 2]) / 12.0
                info = [rel_p, rel_v, [np.cos(yaw - hra), np.sin(yaw - hra)]]
                if lock is not None:
                    info.append([1.0 if lock else 0.0])
                info.append([1.0])
                return np.concatenate(info)
            return np.zeros(8 if lock is not None else 7, dtype=np.float32)

        if agent_id == 0:
            h1 = get_rel_info(self.h1_body)[:5]
            h2 = get_rel_info(self.h2_body)[:5]
            objs = [
                get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]),
                get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]),
                get_rel_info(self.ramp_body),
            ]
            return np.concatenate([self_s, lidar, *objs, h1, h2, np.zeros(3, dtype=np.float32)]).astype(np.float32)
        partner = self.h2_body if agent_id == 1 else self.h1_body
        enemy = get_rel_info(self.s0_body)[:5]
        friend = get_rel_info(partner)
        st = np.array([1.0 if self.grasping[agent_id] else 0.0], dtype=np.float32)
        objs = [
            get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]),
            get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]),
            get_rel_info(self.ramp_body),
        ]
        return np.concatenate([self_s, lidar, *objs, enemy, friend, st]).astype(np.float32)

    def _get_current_dist(self):
        h1p = self.data.xpos[self.h1_body][:2]
        h2p = self.data.xpos[self.h2_body][:2]
        sp = self.data.xpos[self.s0_body][:2]
        return np.linalg.norm((h1p if self.learning_agent_id == 1 else h2p) - sp)

    def _update_seeker_state(self):
        sp, sr = self.data.xpos[self.s0_body][:2], self.data.qpos[self.srot_adr]
        _ = self._get_obs(0)
        v1 = self.visible_cache[0].get(self.h1_body, False)
        v2 = self.visible_cache[0].get(self.h2_body, False)
        if v1 or v2:
            self.seeker_target_pos = self.data.xpos[self.h1_body if v1 else self.h2_body][:2].copy()
            self.seeker_last_known_pos, self.seeker_mode = (
                self.seeker_target_pos.copy(),
                "CHASING",
            )
        elif self.seeker_last_known_pos is not None:
            if np.linalg.norm(sp - self.seeker_last_known_pos) > 0.5:
                self.seeker_target_pos, self.seeker_mode = (
                    self.seeker_last_known_pos,
                    "SEARCHING",
                )
            else:
                self.seeker_last_known_pos, self.seeker_search_timer = None, 50
        else:
            if self.seeker_search_timer <= 0:
                self.seeker_random_target, self.seeker_search_timer = (
                    self.np_random.uniform(-4, 4, 2),
                    80,
                )
            self.seeker_search_timer -= 1
            self.seeker_target_pos, self.seeker_mode = (
                self.seeker_random_target,
                "PATROLLING",
            )

    def _seeker_rule_based_policy(self):
        if self.current_step < base_config.PREP_STEPS:
            return 0.0, 0.0
        sp, sr = self.data.xpos[self.s0_body][:2], self.data.qpos[self.srot_adr]
        ad = (np.arctan2(self.seeker_target_pos[1] - sp[1], self.seeker_target_pos[0] - sp[0]) - sr + np.pi) % (2 * np.pi) - np.pi
        sx_dof = self.model.jnt_dofadr[self.model.joint("s_x").id]
        s_vel = np.linalg.norm(self.data.qvel[sx_dof : sx_dof + 2])
        sf = base_config.SEEKER_RB_THRUST
        sr_val = np.clip(ad * 6.0, -3.0, 3.0)
        if abs(ad) > base_config.SEEKER_RB_TURN_THRESH:
            sf *= 0.3
        if sf > 0.05 and s_vel < 0.05:
            self.s0_stuck_timer += 5
        else:
            self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
        if self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 15
            self.s0_stuck_timer = 0
        if self.s0_recovery_mode > 0:
            sf = -0.2
            sr_val = 1.5
            self.s0_recovery_mode -= 1
        return sf, sr_val

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.prev_dist_to_seeker[self.learning_agent_id] = self._get_current_dist()
        return obs, info

    def step(self, action):
        self.current_step += 1
        for i in [1, 2]:
            self.lock_cooldown[i] = max(0, self.lock_cooldown[i] - 1)
        self._update_seeker_state()
        prev_dist = self.prev_dist_to_seeker.get(self.learning_agent_id, 0.0)
        idx_main = 2 if self.learning_agent_id == 1 else 4
        self.data.ctrl[:] = 0.0
        self.data.ctrl[idx_main : idx_main + 2] = [
            float(action[0]) * base_config.HIDER_THRUST_LIMIT,
            float(action[1]),
        ]
        self._apply_action(self.learning_agent_id, action)
        partner_id = 2 if self.learning_agent_id == 1 else 1
        pidx = 4 if partner_id == 2 else 2
        act_npc = self._get_npc_action(partner_id, "HIDER")
        self.data.ctrl[pidx : pidx + 2] = [
            float(act_npc[0]) * base_config.HIDER_THRUST_LIMIT,
            float(act_npc[1]),
        ]
        self._apply_action(partner_id, act_npc)
        sf0, sr0 = self._seeker_rule_based_policy()
        self.data.ctrl[0:2] = [sf0, sr0]
        for _ in range(base_config.ACTION_REPEAT):
            for box, pose in self.locked_pose.items():
                if self.locked_boxes[box]:
                    bid = self.box1_joint_id if box == self.box1_body else self.box2_joint_id
                    self.data.qpos[self.model.jnt_qposadr[bid] : self.model.jnt_qposadr[bid] + 7] = pose
                    self.data.qvel[self.model.jnt_dofadr[bid] : self.model.jnt_dofadr[bid] + 6] = 0
            mujoco.mj_step(self.model, self.data)

        # --- 報酬計算 (視界依存距離報酬) ---
        new_obs = self._get_obs(self.learning_agent_id)
        # Seeker(0) が Hider(learning_agent_id) の視界に入っているかチェック
        is_seeker_visible = self.visible_cache[self.learning_agent_id].get(self.s0_body, False)

        current_dist = self._get_current_dist()
        self.prev_dist_to_seeker[self.learning_agent_id] = current_dist

        # ★修正: 敵が見えている時のみ距離の増分を報酬化
        reward = 0.0
        if is_seeker_visible:
            reward = (current_dist - prev_dist) * REWARD_DISTANCE_DIFF_SCALE

        captured_self = False
        captured_any = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1, n2 = self.model.geom(c.geom1).name, self.model.geom(c.geom2).name
            if ("seeker" in n1 and "hider" in n2) or ("seeker" in n2 and "hider" in n1):
                captured_any = True
                if ("hider1" in n1 or "hider1" in n2) and self.learning_agent_id == 1:
                    captured_self = True
                if ("hider2" in n1 or "hider2" in n2) and self.learning_agent_id == 2:
                    captured_self = True
                if captured_self:
                    break
        if max(abs(self.data.xpos[self.h1_body if self.learning_agent_id == 1 else self.h2_body][:2])) > 6.5:
            captured_self = True
            captured_any = True

        if captured_self:
            reward += PENALTY_CAPTURE
        else:
            reward += REWARD_SURVIVAL_SCALE

        dof_adr = self.model.jnt_dofadr[self.model.joint("h1_x" if self.learning_agent_id == 1 else "h2_x").id]
        if np.linalg.norm(self.data.qvel[dof_adr : dof_adr + 2]) < 0.1:
            reward += PENALTY_STAGNATION_FORCE

        terminated = captured_any or (self.current_step >= base_config.MAX_STEPS)
        return new_obs, reward, terminated, False, {}


# ==========================================
# 4. メイン処理
# ==========================================
def main():
    print(f"--- Layer 0 Training: {EXPERIMENT_NAME} ---")
    start_time = time.time()
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    if EXECUTION_MODE == "PLAY":
        from stable_baselines3.common.vec_env import DummyVecEnv

        env = DummyVecEnv([lambda: Layer0RunnerEnv(render_mode="human")])
        agent = Agent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        model_path = os.path.join(current_script_dir, f"{EXPERIMENT_NAME}_HIDER.pt")
        if os.path.exists(model_path):
            agent.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model: {model_path}")
        else:
            print("Model missing.")
            return
        agent.eval()
        obs_history = ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, env.observation_space.shape[0], device)
        for ep in range(10):
            obs = env.reset()
            obs_history.reset()
            obs_history.update(obs)
            done = [False]
            total_rew = 0
            while not any(done):
                step_start = time.time()
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_history.get())
                obs, reward, done, _ = env.step(action.cpu().numpy())
                total_rew += reward[0]
                obs_history.update(obs)
                env.envs[0].render(stats={"Ep": ep + 1, "Rew": f"{total_rew:.1f}"})
                wait = 0.08 - (time.time() - step_start)
                if wait > 0:
                    time.sleep(wait)
        env.close()
        return
    run_name = f"{EXPERIMENT_NAME}_{int(time.time())}"
    if TRACK_WANDB:
        wandb.init(
            project=base_config.WANDB_PROJECT_NAME,
            entity=base_config.WANDB_ENTITY,
            sync_tensorboard=True,
            config={"layer": 0, "mode": "runner_v21.56_vis_reward"},
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")

    def make_env():
        env = Layer0RunnerEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.get_raycast_stats = env.env.get_raycast_stats
        env.reset_raycast_stats = env.env.reset_raycast_stats
        return env

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    agent = Agent(envs.single_observation_space.shape[0], envs.single_action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    model_path = os.path.join(current_script_dir, f"{EXPERIMENT_NAME}_HIDER.pt")
    checkpoint_path = model_path.replace(".pt", "_checkpoint.json")
    global_step = 0
    start_global_step = 0
    if LOAD_EXISTING_MODELS and os.path.exists(model_path):
        print(f"★ Loading model: {model_path}")
        agent.load_state_dict(torch.load(model_path, map_location=device))
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                data = json.load(f)
                global_step = data.get("global_step", 0)
                start_global_step = global_step
    obs_history = ObsHistory(
        NUM_ENVS,
        base_config.TRANSFORMER_SEQ_LEN,
        envs.single_observation_space.shape[0],
        device,
    )
    obs = torch.zeros(
        (
            NUM_STEPS,
            NUM_ENVS,
            base_config.TRANSFORMER_SEQ_LEN,
            envs.single_observation_space.shape[0],
        ),
        device=device,
    )
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    next_obs, _ = envs.reset(seed=FIXED_SEED if FIXED_SEED else int(time.time()))
    next_done = torch.zeros(NUM_ENVS).to(device)
    obs_history.reset()
    obs_history.update(next_obs)
    num_updates = int(TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS))
    try:
        for update in tqdm(range(1, num_updates + 1), desc="Updates"):
            episodic_returns = []
            episodic_lengths = []
            for step in range(NUM_STEPS):
                global_step += NUM_ENVS
                obs[step] = obs_history.get()
                dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs_history.get())
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.tensor(next_done).to(device, dtype=torch.float32)
                if "episode" in infos:
                    mask = infos.get("_episode", [True] * len(infos["episode"]))
                    if isinstance(infos["episode"], dict) and "r" in infos["episode"]:
                        for i, is_done in enumerate(mask):
                            if is_done:
                                episodic_returns.append(infos["episode"]["r"][i])
                                episodic_lengths.append(infos["episode"]["l"][i])
                obs_history.update(next_obs)
            with torch.no_grad():
                next_v = agent.get_value(obs_history.get()).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_v
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + base_config.GAMMA * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + base_config.GAMMA * base_config.GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + values
            b_obs = obs.reshape(
                (
                    -1,
                    base_config.TRANSFORMER_SEQ_LEN,
                    envs.single_observation_space.shape[0],
                )
            )
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, envs.single_action_space.shape[0]))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_inds = np.arange(NUM_STEPS * NUM_ENVS)
            for epoch in range(base_config.UPDATE_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, NUM_STEPS * NUM_ENVS, base_config.MINIBATCH_SIZE):
                    end = start + base_config.MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - logratio).mean()
                    mb_adv = b_advantages[mb_inds]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - base_config.CLIP_COEF, 1 + base_config.CLIP_COEF)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                    loss = pg_loss - ENT_COEF * entropy.mean() + base_config.VF_COEF * v_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), base_config.MAX_GRAD_NORM)
                    optimizer.step()
            elapsed = time.time() - start_time
            sps = int((global_step - start_global_step) / elapsed) if elapsed > 0 else 0
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            if len(episodic_returns) > 0:
                writer.add_scalar(
                    "charts/mean_episodic_return",
                    np.mean(episodic_returns),
                    global_step,
                )
                writer.add_scalar(
                    "charts/mean_episodic_length",
                    np.mean(episodic_lengths),
                    global_step,
                )
            if update % 10 == 0:
                avg_ret = np.mean(episodic_returns) if episodic_returns else 0.0
                avg_len = np.mean(episodic_lengths) if episodic_lengths else 0.0
                print(f"Update {update}, Step {global_step}, SPS: {sps}, Loss: {loss.item():.3f}, EpRet: {avg_ret:.2f}, EpLen: {avg_len:.1f}")
                try:
                    stats_list = envs.call("get_raycast_stats")
                    total_h = sum(s["hits"] for s in stats_list)
                    total_m = sum(s["misses"] for s in stats_list)
                    if (total_h + total_m) > 0:
                        hr = 100 * total_h / (total_h + total_m)
                        writer.add_scalar("charts/raycast_cache_hit_rate", hr, global_step)
                        print(f"  -> RayCast Cache Hit Rate: {hr:.1f}%")
                    envs.call("reset_raycast_stats")
                except:
                    pass
    except KeyboardInterrupt:
        print("Training interrupted.")
    if SAVE_MODEL:
        torch.save(agent.state_dict(), model_path)
        try:
            with open(model_path.replace(".pt", "_checkpoint.json"), "w") as f:
                json.dump({"global_step": global_step}, f)
            print(f"Model saved to {model_path}")
        except:
            pass
    envs.close()
    writer.close()
    if TRACK_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
