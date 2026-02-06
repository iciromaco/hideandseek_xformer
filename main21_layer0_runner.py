# main21_layer0_runner.py
# 演習第21回：サブサンプション・アーキテクチャ Layer 0 (逃走)
#
# 【修正(v21.49)】
# - ログ密度の正常化:
#   - SPS, Entropy, 各種Lossを毎Update WandB(TensorBoard)に記録するように修正。
#   - エピソードデータ(Return, Length)も消失を防ぐため、データがある毎Update記録。
#   - コンソールへの print 出力のみ 10 Update ごとに制限し、視認性を確保。
# - インポートパス解決・モデル切り替えロジックは維持。
#
# 【実行準備】
# main18_optimization.py が同じフォルダに必要です。

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import mujoco
import wandb
from tqdm import tqdm

# --- インポートパスの解決 ---
search_path = os.path.dirname(os.path.abspath(__file__))
for _ in range(4):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)
    search_path = os.path.dirname(search_path)

# ベース設定のインポート
try:
    import main18_optimization as base_config
    from main18_optimization import HideAndSeekEnv, Agent, ObsHistory
except ImportError:
    print(f"Error: main18_optimization.py not found in sys.path.")
    exit(1)

# ==========================================
# 1. 設定のオーバーライド
# ==========================================
EXPERIMENT_NAME = "HideAndSeek_Layer0_Runner"

# ★実行制御設定 (Optunaから書き換え対象)
EXECUTION_MODE = "TRAIN" 
LOAD_EXISTING_MODELS = True
SAVE_MODEL = True
TRACK_WANDB = True           
FIXED_SEED = None

TOTAL_TIMESTEPS = 1000000 
NUM_ENVS = 8
NUM_STEPS = 128
LEARNING_RATE = 3e-4
ENT_COEF = 0.001

# ★報酬設定
REWARD_DISTANCE_DIFF_SCALE = 0.0 
REWARD_SURVIVAL_SCALE = 5.0      
PENALTY_CAPTURE = -100.0         
PENALTY_STAGNATION_FORCE = -0.1  

# 親クラスの報酬無効化
base_config.REWARD_SURVIVAL = 0.0
base_config.REWARD_DISTANCE_COEFF = 0.0
base_config.PENALTY_CAPTURE = 0.0
base_config.PENALTY_STAGNATION = 0.0
base_config.REWARD_CAPTURE_BONUS = 0.0
base_config.MODEL_PATH_HIDER = f"{EXPERIMENT_NAME}_HIDER.pt"

device = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")

# ==========================================
# 2. 環境クラスの拡張 (Layer0RunnerEnv)
# ==========================================
class Layer0RunnerEnv(HideAndSeekEnv):
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.prev_dist_to_seeker = {}
        self.h1_geoms_set = set([i for i in range(self.model.ngeom) if "hider1" in self.model.geom(i).name])
        self.h2_geoms_set = set([i for i in range(self.model.ngeom) if "hider2" in self.model.geom(i).name])
        self.s0_geoms_set = set(self.s0_geoms)
        self.npc_obs_history = {
            0: ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, 53, device),
            1: ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, 53, device),
            2: ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, 53, device),
        }
    def _get_current_dist(self):
        h1p = self.data.xpos[self.h1_body][:2]
        h2p = self.data.xpos[self.h2_body][:2]
        sp = self.data.xpos[self.s0_body][:2]
        return np.linalg.norm((h1p if self.learning_agent_id == 1 else h2p) - sp)

    def _seeker_rule_based_policy(self):
        target_pos = self.seeker_target_pos
        if self.current_step < PREP_STEPS or target_pos is None: return 0.0, 0.0
        if self.seeker_mode == "SCANNING": return 0.0, 1.0
        sp = self.data.xpos[self.s0_body][:2]; sr = self.data.qpos[self.srot_adr]
        dy = float(target_pos[1] - sp[1]); dx = float(target_pos[0] - sp[0])
        desired_angle = np.arctan2(dy, dx); angle_diff = (desired_angle - float(sr) + np.pi) % (2*np.pi) - np.pi
        thrust = SEEKER_RB_THRUST; steering = np.clip(angle_diff * 6.0, -3.0, 3.0)
        if abs(angle_diff) > SEEKER_RB_TURN_THRESH: thrust *= 0.3
        sx_dof = self.model.jnt_dofadr[self.model.joint('s_x').id]; s_vel = np.linalg.norm(self.data.qvel[sx_dof : sx_dof+2])
        if thrust > 0.05 and s_vel < 0.05: self.s0_stuck_timer += 1
        else: self.s0_stuck_timer = 0
        if self.s0_stuck_timer > 20: self.s0_recovery_mode = 15; self.s0_stuck_timer = 0
        if self.s0_recovery_mode > 0: thrust = -0.2; steering = 1.5; self.s0_recovery_mode -= 1
        return float(thrust), float(steering)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.prev_dist_to_seeker[self.learning_agent_id] = self._get_current_dist()
        return obs, info
        
    def step(self, action):
        self.current_step += 1
        for i in [1, 2]:
            if self.lock_cooldown[i] > 0: self.lock_cooldown[i] -= 1
        self._update_seeker_state()
        prev_dist = self.prev_dist_to_seeker.get(self.learning_agent_id, 0.0)
        idx_main = 2 if self.learning_agent_id == 1 else 4
        self.data.ctrl[:] = 0.0 
        self.data.ctrl[idx_main:idx_main+2] = [float(action[0]) * base_config.HIDER_THRUST_LIMIT, float(action[1])]
        self._apply_action(self.learning_agent_id, action)
        partner_id = 2 if self.learning_agent_id == 1 else 1
        pidx = 4 if partner_id == 2 else 2
        act_npc = self._get_npc_action(partner_id, "HIDER")
        self.data.ctrl[pidx:pidx+2] = [float(act_npc[0]) * base_config.HIDER_THRUST_LIMIT, float(act_npc[1])]
        self._apply_action(partner_id, act_npc)
        sf0, sr0 = self._seeker_rule_based_policy()
        self.data.ctrl[0:2] = [sf0, sr0]
        for _ in range(ACTION_REPEAT):
            for box, pose in self.locked_pose.items():
                if self.locked_boxes[box]:
                    bid = self.box1_joint_id if box==self.box1_body else self.box2_joint_id
                    self.data.qpos[self.model.jnt_qposadr[bid]:self.model.jnt_qposadr[bid]+7] = pose
                    self.data.qvel[self.model.jnt_dofadr[bid]:self.model.jnt_dofadr[bid]+6] = 0
            mujoco.mj_step(self.model, self.data)
        current_dist = self._get_current_dist(); self.prev_dist_to_seeker[self.learning_agent_id] = current_dist
        reward = (current_dist - prev_dist) * REWARD_DISTANCE_DIFF_SCALE
        captured_self = False; captured_any = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]; g1, g2 = c.geom1, c.geom2
            h_id = None
            if g1 in self.s0_geoms_set:
                if g2 in self.h1_geoms_set: h_id = 1
                elif g2 in self.h2_geoms_set: h_id = 2
            elif g2 in self.s0_geoms_set:
                if g1 in self.h1_geoms_set: h_id = 1
                elif g1 in self.h2_geoms_set: h_id = 2
            if h_id is not None:
                captured_any = True
                if h_id == self.learning_agent_id: captured_self = True
                if captured_self: break
        if max(abs(self.data.xpos[self.h1_body if self.learning_agent_id==1 else self.h2_body][:2])) > 6.5: captured_self = True; captured_any = True
        if captured_self: reward += PENALTY_CAPTURE
        else: reward += REWARD_SURVIVAL_SCALE
        dof_adr = self.model.jnt_dofadr[self.model.joint('h1_x' if self.learning_agent_id==1 else 'h2_x').id]
        if np.linalg.norm(self.data.qvel[dof_adr:dof_adr+2]) < 0.1: reward += PENALTY_STAGNATION_FORCE
        terminated = captured_any or (self.current_step >= MAX_STEPS)
        return self._get_obs(self.learning_agent_id), reward, terminated, False, {}

# ==========================================
# 3. メイン処理
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
        else: print("Model missing."); return
        agent.eval()
        obs_history = ObsHistory(1, base_config.TRANSFORMER_SEQ_LEN, env.observation_space.shape[0], device)
        for ep in range(10):
            obs = env.reset(); obs_history.reset(); obs_history.update(obs); done = [False]; total_rew = 0
            while not any(done):
                step_start = time.time()
                with torch.no_grad(): action, _, _, _ = agent.get_action_and_value(obs_history.get())
                obs, reward, done, _ = env.step(action.cpu().numpy()); total_rew += reward[0]; obs_history.update(obs)
                env.envs[0].render(stats={"Ep": ep+1, "Rew": f"{total_rew:.1f}"})
                wait = 0.05 - (time.time() - step_start)
                if wait > 0: time.sleep(wait)
        env.close(); return

    # --- TRAIN モード ---
    run_name = f"{EXPERIMENT_NAME}_{int(time.time())}"
    if TRACK_WANDB:
        wandb.init(
            project=base_config.WANDB_PROJECT_NAME,
            entity=base_config.WANDB_ENTITY,
            sync_tensorboard=True,
            config={"layer": 0, "mode": "runner_optimized_logging"},
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")

    def make_env():
        env = Layer0RunnerEnv(); env = gym.wrappers.RecordEpisodeStatistics(env); return env
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    agent = Agent(envs.single_observation_space.shape[0], envs.single_action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    model_path = os.path.join(current_script_dir, f"{EXPERIMENT_NAME}_HIDER.pt")
    checkpoint_path = model_path.replace('.pt', '_checkpoint.json')
    global_step = 0; start_global_step = 0

    if LOAD_EXISTING_MODELS and os.path.exists(model_path):
        print(f"★ Loading existing model: {model_path}")
        agent.load_state_dict(torch.load(model_path, map_location=device))
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                data = json.load(f); global_step = data.get('global_step', 0); start_global_step = global_step
        print(f"  -> Resumed from step {global_step}")
    
    obs_history = ObsHistory(NUM_ENVS, base_config.TRANSFORMER_SEQ_LEN, envs.single_observation_space.shape[0], device)
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, base_config.TRANSFORMER_SEQ_LEN, envs.single_observation_space.shape[0]), device=device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device); rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS), device=device); values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    
    next_obs, _ = envs.reset(seed=FIXED_SEED if FIXED_SEED else int(time.time()))
    next_done = torch.zeros(NUM_ENVS).to(device)
    obs_history.reset(); obs_history.update(next_obs)
    
    num_updates = int(TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS))
    try:
        for update in tqdm(range(1, num_updates + 1), desc="Updates"):
            episodic_returns = []; episodic_lengths = []
            for step in range(NUM_STEPS):
                global_step += NUM_ENVS; obs[step] = obs_history.get(); dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs_history.get())
                    values[step] = value.flatten()
                actions[step] = action; logprobs[step] = logprob
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.tensor(next_done).to(device, dtype=torch.float32)
                
                # エピソードデータの収集 (消失を防ぐため毎ステップチェック)
                if "episode" in infos:
                    mask = infos.get("_episode", [True] * len(infos["episode"]))
                    if isinstance(infos["episode"], dict) and "r" in infos["episode"]:
                        for i, is_done in enumerate(mask):
                            if is_done:
                                episodic_returns.append(infos["episode"]["r"][i])
                                episodic_lengths.append(infos["episode"]["l"][i])
                obs_history.update(next_obs)

            # GAE / PPO Update (計算部分は変更なし)
            with torch.no_grad():
                next_v = agent.get_value(obs_history.get()).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device); lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1: nextnonterminal = 1.0 - next_done; nextvalues = next_v
                    else: nextnonterminal = 1.0 - dones[t+1]; nextvalues = values[t+1]
                    delta = rewards[t] + base_config.GAMMA * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + base_config.GAMMA * base_config.GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1, base_config.TRANSFORMER_SEQ_LEN, envs.single_observation_space.shape[0]))
            b_logprobs = logprobs.reshape(-1); b_actions = actions.reshape((-1, envs.single_action_space.shape[0]))
            b_advantages = advantages.reshape(-1); b_returns = returns.reshape(-1); b_values = values.reshape(-1)
            b_inds = np.arange(NUM_STEPS * NUM_ENVS)
            for epoch in range(base_config.UPDATE_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, NUM_STEPS * NUM_ENVS, base_config.MINIBATCH_SIZE):
                    end = start + base_config.MINIBATCH_SIZE; mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]; ratio = logratio.exp()
                    with torch.no_grad(): approx_kl = ((ratio - 1) - logratio).mean()
                    mb_adv = b_advantages[mb_inds]; mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss1 = -mb_adv * ratio; pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - base_config.CLIP_COEF, 1 + base_config.CLIP_COEF)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean(); v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                    loss = pg_loss - ENT_COEF * entropy.mean() + base_config.VF_COEF * v_loss
                    optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), base_config.MAX_GRAD_NORM); optimizer.step()

            # ★修正: WandB(TensorBoard)への記録を毎Update実行するようにガードの外へ移動
            elapsed = time.time() - start_time
            sps = int((global_step - start_global_step) / elapsed) if elapsed > 0 else 0
            
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy.mean().item(), global_step) # エントロピーを毎Update記録
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            
            # エピソードデータも存在すれば毎Update記録 (消失防止)
            if len(episodic_returns) > 0:
                writer.add_scalar("charts/mean_episodic_return", np.mean(episodic_returns), global_step)
                writer.add_scalar("charts/mean_episodic_length", np.mean(episodic_lengths), global_step)

            # コンソールへの表示のみ 10回に 1回に制限
            if update % 10 == 0:
                avg_ret = np.mean(episodic_returns) if episodic_returns else 0.0
                avg_len = np.mean(episodic_lengths) if episodic_lengths else 0.0
                print(f"Update {update}, Step {global_step}, Loss: {loss.item():.3f}, Entropy: {entropy.mean().item():.4f}, EpRet: {avg_ret:.2f}, EpLen: {avg_len:.1f}")

    except KeyboardInterrupt: print("Training interrupted.")
    if SAVE_MODEL:
        torch.save(agent.state_dict(), model_path)
        try:
            with open(model_path.replace('.pt', '_checkpoint.json'), 'w') as f: json.dump({'global_step': global_step}, f)
            print(f"Model saved to {model_path}")
        except: pass
    else: print("★ Model saving skipped")
    envs.close(); writer.close(); 
    if TRACK_WANDB: wandb.finish()

if __name__ == "__main__":
    main()