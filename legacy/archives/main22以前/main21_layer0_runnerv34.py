# main21_layer0_runner.py
# 演習第21回：サブサンプション・アーキテクチャ Layer 0 (逃走)
#
# 【目的】
# 「敵から逃げる（距離を取る）」という原始的な生存能力（Layer 0）を学習させる。
#
# 【主な機能】
# 1. 特化型報酬: 距離の「増分」を報酬とし、能動的な逃走を促す。
# 2. 停留ペナルティ: 速度が遅い場合に強制的にペナルティを与え、棒立ちを防ぐ。
# 3. 実装の統一: SB3を使わず、main18と同じAgent/学習ループを使用。
# 4. 実行モード: 学習(TRAIN)と推論(PLAY)の切り替えが可能。
#
# 【修正(v21.20)】
# - NameError (EXECUTION_MODE) を修正。
#   - EXECUTION_MODE を main 関数の先頭で定義するように変更（またはグローバル参照を確実にする）。
# - ログ出力: EpLen をコンソールに追加。
#
# 【実行準備】
# main18_optimization.py が同じフォルダに必要です。

import json
import os
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

# ベースとなる環境と設定をインポート
try:
    import main18_optimization as base_config
    from main18_optimization import Agent, HideAndSeekEnv, ObsHistory
except ImportError:
    print("Error: main18_optimization.py not found.")
    exit(1)

# ==========================================
# 設定のオーバーライド
# ==========================================
EXPERIMENT_NAME = "HideAndSeek_Layer0_Runner"

# ★実行モード設定
# "TRAIN": 学習を実行 (GPU推奨)
# "PLAY" : 学習済みモデルで推論・描画を実行 (CPU/GPU)
# EXECUTION_MODE = "PLAY"
EXECUTION_MODE = "TRAIN"

TOTAL_TIMESTEPS = 1000000  # 基礎動作学習用
NUM_ENVS = 8
NUM_STEPS = 128
LEARNING_RATE = 3e-4

# ★特化型報酬設定
# 距離の増分に対する係数 (1m遠ざかると+10点)
REWARD_DISTANCE_DIFF_SCALE = 10.0
# 捕獲ペナルティ
PENALTY_CAPTURE = -100.0
# 強制的な停留ペナルティ (速度が0.1未満なら減点)
PENALTY_STAGNATION_FORCE = -1.5

# 親クラス(main18)の報酬設定を無効化 (super().step() で計算される報酬を0にするため)
base_config.REWARD_SURVIVAL = 0.0
base_config.REWARD_DISTANCE_COEFF = 0.0
base_config.PENALTY_CAPTURE = 0.0
base_config.PENALTY_STAGNATION = 0.0
base_config.REWARD_CAPTURE_BONUS = 0.0

# 親クラスが参照するモデルパスもこの実験用に書き換える
base_config.MODEL_PATH_HIDER = f"{EXPERIMENT_NAME}_HIDER.pt"
base_config.MODEL_PATH_SEEKER = f"{EXPERIMENT_NAME}_SEEKER.pt"

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")


# ==========================================
# 環境クラスの拡張
# ==========================================
class Layer0RunnerEnv(HideAndSeekEnv):
    """
    逃走(Runner)行動の獲得に特化した環境。
    親クラスの物理演算を利用しつつ、報酬ロジックのみを距離増分ベースに差し替える。
    """

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        # 前回の距離を保持するための変数
        self.prev_dist_to_seeker = {}

        # 接触判定用のIDセットを事前作成
        # 文字列比較よりも高速かつ確実に判定可能
        self.h1_geoms_set = set([i for i in range(self.model.ngeom) if "hider1" in self.model.geom(i).name])
        self.h2_geoms_set = set([i for i in range(self.model.ngeom) if "hider2" in self.model.geom(i).name])
        self.s0_geoms_set = set(self.s0_geoms)

        # 親クラスでCPU固定で初期化されているnpc_obs_historyを、現在のデバイスで再初期化
        self.npc_obs_history = {
            0: ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, 53, device),  # Seeker
            1: ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, 53, device),  # Hider1
            2: ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, 53, device),  # Hider2
        }

    def _get_current_dist(self):
        """現在のSeekerとの距離を計算"""
        h1p = self.data.xpos[self.h1_body][:2]
        h2p = self.data.xpos[self.h2_body][:2]
        sp = self.data.xpos[self.s0_body][:2]
        my_pos = h1p if self.learning_agent_id == 1 else h2p
        return np.linalg.norm(my_pos - sp)

    def _seeker_rule_based_policy(self):
        """親クラスのメソッドをオーバーライドしてロバスト化 (配列混入防止)"""
        target_pos = self.seeker_target_pos
        if self.current_step < base_config.PREP_STEPS or target_pos is None:
            return 0.0, 0.0

        if self.seeker_mode == "SCANNING":
            return 0.0, 1.0

        sp = self.data.xpos[self.s0_body][:2]
        sr = self.data.qpos[self.srot_adr]

        # 明示的なfloatキャストで計算
        dy = float(target_pos[1] - sp[1])
        dx = float(target_pos[0] - sp[0])
        curr_angle = float(sr)

        desired_angle = np.arctan2(dy, dx)
        angle_diff = (desired_angle - curr_angle + np.pi) % (2 * np.pi) - np.pi

        thrust = base_config.SEEKER_RB_THRUST
        steering = np.clip(angle_diff * 6.0, -3.0, 3.0)

        if abs(angle_diff) > base_config.SEEKER_RB_TURN_THRESH:
            thrust *= 0.3

        # スタック判定
        sx_dof = self.model.jnt_dofadr[self.model.joint("s_x").id]
        s_vel = np.linalg.norm(self.data.qvel[sx_dof : sx_dof + 2])

        if thrust > 0.05 and s_vel < 0.05:
            self.s0_stuck_timer += 1
        else:
            self.s0_stuck_timer = 0

        if self.s0_stuck_timer > 20:
            self.s0_recovery_mode = 15
            self.s0_stuck_timer = 0

        if self.s0_recovery_mode > 0:
            thrust = -0.2
            steering = 1.5
            self.s0_recovery_mode -= 1

        return float(thrust), float(steering)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # リセット時の距離を記録
        dist = self._get_current_dist()
        self.prev_dist_to_seeker[self.learning_agent_id] = dist

        return obs, info

    def step(self, action):
        # 状態更新
        self.current_step += 1
        for i in [1, 2]:
            if self.lock_cooldown[i] > 0:
                self.lock_cooldown[i] -= 1

        # これがないとSeekerがターゲットを更新できず、動かなくなる
        self._update_seeker_state()

        prev_dist = self.prev_dist_to_seeker.get(self.learning_agent_id, 0.0)

        # アクション適用 (親クラスのstepは呼ばずにここで実行)
        idx_main = 2 if self.learning_agent_id == 1 else 4
        self.data.ctrl[:] = 0.0

        # 学習エージェント
        self.data.ctrl[idx_main] = float(action[0]) * base_config.HIDER_THRUST_LIMIT
        self.data.ctrl[idx_main + 1] = float(action[1])
        self._apply_action(self.learning_agent_id, action)

        # パートナーNPC
        partner_id = 2 if self.learning_agent_id == 1 else 1
        pidx = 4 if partner_id == 2 else 2

        # 親クラスのメソッドを使ってNPC行動を決定（モデルがあれば使う）
        act_npc = self._get_npc_action(partner_id, "HIDER")
        self.data.ctrl[pidx] = float(act_npc[0]) * base_config.HIDER_THRUST_LIMIT
        self.data.ctrl[pidx + 1] = float(act_npc[1])
        self._apply_action(partner_id, act_npc)

        # Seeker
        sf0, sr0 = self._seeker_rule_based_policy()
        self.data.ctrl[0] = sf0
        self.data.ctrl[1] = sr0

        # 物理シミュレーション
        for _ in range(base_config.ACTION_REPEAT):
            for box, pose in self.locked_pose.items():
                if self.locked_boxes[box]:
                    bid = self.box1_joint_id if box == self.box1_body else self.box2_joint_id
                    q = self.model.jnt_qposadr[bid]
                    d = self.model.jnt_dofadr[bid]
                    self.data.qpos[q : q + 7] = pose
                    self.data.qvel[d : d + 6] = 0
            mujoco.mj_step(self.model, self.data)

        # --- 報酬計算 ---
        current_dist = self._get_current_dist()
        self.prev_dist_to_seeker[self.learning_agent_id] = current_dist

        dist_diff = current_dist - prev_dist
        reward = dist_diff * REWARD_DISTANCE_DIFF_SCALE

        # 捕獲判定 (IDセット使用)
        captured_self = False
        captured_any = False

        my_geoms_set = self.h1_geoms_set if self.learning_agent_id == 1 else self.h2_geoms_set

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2

            hider_touch_id = None
            if g1 in self.s0_geoms_set:
                if g2 in self.h1_geoms_set:
                    hider_touch_id = 1
                elif g2 in self.h2_geoms_set:
                    hider_touch_id = 2
            elif g2 in self.s0_geoms_set:
                if g1 in self.h1_geoms_set:
                    hider_touch_id = 1
                elif g1 in self.h2_geoms_set:
                    hider_touch_id = 2

            if hider_touch_id is not None:
                captured_any = True
                if hider_touch_id == self.learning_agent_id:
                    captured_self = True
                if captured_self:
                    break

        # 場外判定
        if self.learning_agent_id == 1:
            my_pos = self.data.xpos[self.h1_body][:2]
        else:
            my_pos = self.data.xpos[self.h2_body][:2]
        if max(abs(my_pos)) > 6.5:
            captured_self = True
            captured_any = True

        if captured_self:
            reward += PENALTY_CAPTURE

        # 生存報酬 (微小)
        if not captured_self:
            reward += 0.05

        # 停留ペナルティ
        if self.learning_agent_id == 1:
            dof_adr = self.model.jnt_dofadr[self.model.joint("h1_x").id]
            vel = self.data.qvel[dof_adr : dof_adr + 2]
        else:
            dof_adr = self.model.jnt_dofadr[self.model.joint("h2_x").id]
            vel = self.data.qvel[dof_adr : dof_adr + 2]

        speed = np.linalg.norm(vel)
        if speed < 0.1:
            reward += PENALTY_STAGNATION_FORCE

        # 終了判定: 誰かが捕まったら終了
        terminated = captured_any or (self.current_step >= base_config.MAX_STEPS)

        # 観測取得
        obs = self._get_obs(self.learning_agent_id)

        return obs, reward, terminated, False, {}


# ==========================================
# メイン処理
# ==========================================
def main():
    print(f"--- Layer 0 Training: {EXPERIMENT_NAME} ---")
    # ここでグローバル変数を参照
    print(f"Mode: {EXECUTION_MODE}")

    # グローバル変数をここで初期化（UnboundLocalError対策）
    start_time = time.time()

    # ----------------------------------------
    # 推論 (PLAY) モード
    # ----------------------------------------
    if EXECUTION_MODE == "PLAY":
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Viewer強制
        render_mode = "human"

        def make_env():
            env = Layer0RunnerEnv(render_mode=render_mode)
            # 必要なダミーメソッド
            env.get_raycast_stats = lambda: [{"hits": 0, "misses": 0}]
            env.reset_raycast_stats = lambda: None
            return env

        env = DummyVecEnv([make_env])
        agent = Agent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)

        # モデルロード
        model_path = f"{EXPERIMENT_NAME}_HIDER.pt"
        if os.path.exists(model_path):
            print(f"★ Loading model: {model_path}")
            try:
                agent.load_state_dict(torch.load(model_path, map_location=device))
                print("  -> Load successful.")
            except Exception as e:
                print(f"  -> Load failed: {e}")
                return
        else:
            print(f"  -> Model file not found: {model_path}")
            return

        agent.eval()

        # パートナー(NPC)にも同じ学習済みモデルをセット
        if hasattr(env.envs[0], "unwrapped"):
            real_env = env.envs[0].unwrapped
        else:
            real_env = env.envs[0]
        real_env.npc_hider_agent = agent
        print("  -> Set loaded model to NPC partner as well.")

        actual_num_envs = 1
        obs_history = ObsHistory(
            actual_num_envs,
            base_config.TRANSFORMER_SEQ_LEN,
            env.observation_space.shape[0],
            device,
        )

        try:
            for ep in range(10):
                obs = env.reset()
                obs_history.reset()
                obs_history.update(obs)

                done = [False]
                total_rew = 0

                while not any(done):
                    step_start_time = time.time()

                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(obs_history.get())

                    obs, reward, done, infos = env.step(action.cpu().numpy())
                    total_rew += reward[0]

                    obs_history.update(obs)

                    # 描画 (現在距離も表示)
                    real_env = env.envs[0].unwrapped
                    curr_dist = real_env._get_current_dist()

                    env.envs[0].render(
                        stats={
                            "Ep": ep + 1,
                            "Rew": f"{total_rew:.1f}",
                            "Dist": f"{curr_dist:.2f}m",
                        }
                    )

                    dt = time.time() - step_start_time
                    wait = 0.05 - dt
                    if wait > 0:
                        time.sleep(wait)

                print(f"Episode {ep+1} finished. Total Reward: {total_rew:.2f}")
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            env.close()
        return

    # ----------------------------------------
    # 学習 (TRAIN) モード
    # ----------------------------------------
    run_name = f"{EXPERIMENT_NAME}_{int(time.time())}"
    if base_config.TRACK_WANDB:
        wandb.init(
            project=base_config.WANDB_PROJECT_NAME,
            entity=base_config.WANDB_ENTITY,
            sync_tensorboard=True,
            config={"layer": 0, "mode": "runner_v2.2", "reward": "diff"},
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")

    def make_env(render_mode=None):
        def thunk():
            env = Layer0RunnerEnv(render_mode=render_mode)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.get_raycast_stats = lambda: [{"hits": 0, "misses": 0}]
            env.reset_raycast_stats = lambda: None
            return env

        return thunk

    if base_config.TRAIN_MODE:
        if not base_config.USE_VIEWER:
            envs = gym.vector.AsyncVectorEnv([make_env(render_mode=None) for i in range(NUM_ENVS)])
            actual_num_envs = NUM_ENVS
        else:
            envs = gym.vector.SyncVectorEnv([make_env(render_mode="human")])
            actual_num_envs = 1
    else:
        envs = gym.vector.SyncVectorEnv([make_env(render_mode="human")])
        actual_num_envs = 1

    agent = Agent(envs.single_observation_space.shape[0], envs.single_action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # 継続学習
    model_path = f"{EXPERIMENT_NAME}_HIDER.pt"
    checkpoint_path = model_path.replace(".pt", "_checkpoint.json")
    global_step = 0
    start_global_step = 0

    if os.path.exists(model_path):
        print(f"★ Loading existing model: {model_path}")
        try:
            agent.load_state_dict(torch.load(model_path, map_location=device))
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, "r") as f:
                    data = json.load(f)
                    global_step = data.get("global_step", 0)
                    start_global_step = global_step
            print(f"  -> Resumed from step {global_step}")
        except Exception as e:
            print(f"  -> Load failed: {e}. Starting from scratch.")

    obs_history = ObsHistory(
        actual_num_envs,
        base_config.TRANSFORMER_SEQ_LEN,
        envs.single_observation_space.shape[0],
        device,
    )

    obs = torch.zeros(
        (
            NUM_STEPS,
            actual_num_envs,
            base_config.TRANSFORMER_SEQ_LEN,
            envs.single_observation_space.shape[0],
        ),
        device=device,
    )
    actions = torch.zeros((NUM_STEPS, actual_num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((NUM_STEPS, actual_num_envs), device=device)
    rewards = torch.zeros((NUM_STEPS, actual_num_envs), device=device)
    dones = torch.zeros((NUM_STEPS, actual_num_envs), device=device)
    values = torch.zeros((NUM_STEPS, actual_num_envs), device=device)

    next_obs, _ = envs.reset(seed=base_config.SEED)
    next_done = torch.zeros(actual_num_envs).to(device)

    obs_history.reset()
    obs_history.update(next_obs)

    print(f"Start Training: {TOTAL_TIMESTEPS} steps")
    remaining_steps = TOTAL_TIMESTEPS - (global_step - start_global_step)
    if remaining_steps <= 0:
        remaining_steps = TOTAL_TIMESTEPS

    num_updates = int(remaining_steps // (actual_num_envs * NUM_STEPS))

    try:
        for update in tqdm(range(1, num_updates + 1), desc="Updates"):
            episodic_returns = []
            episodic_lengths = []

            for step in range(NUM_STEPS):
                global_step += actual_num_envs
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
                    ep_data = infos["episode"]
                    if isinstance(ep_data, dict) and "r" in ep_data:
                        for i, is_done in enumerate(mask):
                            if is_done:
                                episodic_returns.append(ep_data["r"][i])
                                episodic_lengths.append(ep_data["l"][i])

                obs_history.update(next_obs)
                if base_config.USE_VIEWER:
                    envs.envs[0].unwrapped.render()

            # GAE
            with torch.no_grad():
                next_value = agent.get_value(obs_history.get()).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + base_config.GAMMA * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + base_config.GAMMA * base_config.GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + values

            # Optimize
            b_obs = obs.reshape((-1, base_config.TRANSFORMER_SEQ_LEN, agent.obs_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, agent.action_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(NUM_STEPS * actual_num_envs)

            for epoch in range(base_config.UPDATE_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, NUM_STEPS * actual_num_envs, base_config.MINIBATCH_SIZE):
                    end = start + base_config.MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - base_config.CLIP_COEF, 1 + base_config.CLIP_COEF)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    loss = pg_loss - base_config.ENT_COEF * entropy.mean() + base_config.VF_COEF * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), base_config.MAX_GRAD_NORM)
                    optimizer.step()

            # Logging
            elapsed = time.time() - start_time
            sps = int((global_step - start_global_step) / elapsed) if elapsed > 0 else 0

            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)

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
                # ★追加: EpLen を出力
                avg_len = np.mean(episodic_lengths) if episodic_lengths else 0.0
                print(f"Update {update}, Step {global_step}, Loss: {loss.item():.3f}, EpRet: {avg_ret:.2f}, EpLen: {avg_len:.1f}")

    except KeyboardInterrupt:
        print("Training interrupted.")

    torch.save(agent.state_dict(), f"{EXPERIMENT_NAME}_HIDER.pt")
    # Checkpoint
    try:
        with open(checkpoint_path, "w") as f:
            json.dump({"global_step": global_step}, f)
        print(f"Checkpoint saved: global_step={global_step}")
    except:
        pass

    print(f"Model saved to {EXPERIMENT_NAME}_HIDER.pt")

    envs.close()
    writer.close()
    if base_config.TRACK_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
