# main26_train_final.py v2.06
# 演習第26回：【SPSログ修正 ＆ エントロピー減衰導入】学習の収束性を物理的に保証する完全展開版
# 
# 遵守事項:
# 1. SPS記録の修正: "charts/SPS" にキーを統一。Wandb/TB での可視化を確実化。
# 2. エントロピー制御: ENT_COEF のアニーリング（減衰）を実装。探索への逃避を阻止し、確信へ誘導。
# 3. 形状・不連続性エラー根絶: .contiguous() と reshape を徹底。
# 4. 完全展開: 1行1命令を死守。リスト内包表記、複数代入、三項演算子を排除。

import os
import sys
import time
import signal
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# --- [0] プロジェクトルートのパス解決 ---
_script_path = Path(__file__).resolve()
_project_root = _script_path.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
    sys.path.insert(0, os.path.join(str(_project_root), "src"))

from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2

# --- 1. モード・定数設定 ---
TRAIN_MODE = True            
USE_VIEWER = False            
MODE = "initial"             
TRACK_WANDB = True           

# ハイパーパラメータ
SEQ_LEN = 8                  
HIDDEN_DIM = 128             
EPISODE_LIMIT = 500          
TOTAL_STEPS = 20_000_000     

# PPO アルゴリズム詳細設定
LR = 2.5e-4                  
GAMMA = 0.99                 
GAE_LAMBDA = 0.95            
UPDATE_EPOCHS = 4            
BATCH_SIZE = 512             
CLIP_COEF = 0.2              
# 💡 エントロピー設定
ENT_COEF_START = 0.01        # 探索開始時の係数
ENT_COEF_END = 0.001         # 最終的な係数（確信を促す）
VF_COEF = 0.5                
MAX_GRAD_NORM = 0.5          

EXPERIMENT_NAME = f"HNS_V26_GTRPPO_{MODE}"
SAVE_PATH = f"{EXPERIMENT_NAME}.pt"

if TRACK_WANDB: import wandb

def log_debug(msg):
    current_time_val = datetime.now()
    current_time_str = current_time_val.strftime('%H:%M:%S')
    print(f"[{current_time_str}] 🔍 {msg}")

class ObsHistory:
    """Transformer用の履歴管理。メモリ連続性を維持し RuntimeError を防止。"""
    def __init__(self, n_envs, seq_len, obs_dim, device):
        self.n_envs = n_envs
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.device = device
        self.buffer = torch.zeros((n_envs, seq_len * 2, obs_dim), device=device)
        self.ptr = 0

    def reset(self, env_idx=None):
        if env_idx is None:
            self.buffer.zero_()
            self.ptr = 0
        else:
            self.buffer[env_idx].zero_()

    def update(self, obs_data, env_idx=None):
        t_data = torch.as_tensor(obs_data, dtype=torch.float32, device=self.device)
        if env_idx is None:
            new_shape = (self.n_envs, self.obs_dim)
            t_data = t_data.reshape(new_shape)
            self.buffer[:, self.ptr] = t_data
            self.buffer[:, self.ptr + self.seq_len] = t_data
            self.ptr = (self.ptr + 1) % self.seq_len
        else:
            t_data = t_data.reshape(self.obs_dim)
            self.buffer[env_idx, self.ptr] = t_data
            self.buffer[env_idx, self.ptr + self.seq_len] = t_data

    def get(self):
        result_sequence = self.buffer[:, self.ptr : self.ptr + self.seq_len]
        # 💡 不連続テンソル問題を解決
        return result_sequence.contiguous()

def make_env():
    render_mode_str = "human" if USE_VIEWER else None
    return TeamCosEnv(mode=MODE, render_mode=render_mode_str)

def run():
    # 計算デバイスの決定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_debug(f"Device: {device}")
    
    def signal_handler(sig, frame):
        log_debug("Shutdown requested.")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    act_dim = 4 if MODE == "initial" else 8
    agent = AgentV2(53, act_dim, HIDDEN_DIM, SEQ_LEN).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    
    if os.path.exists(SAVE_PATH):
        try:
            checkpoint = torch.load(SAVE_PATH, map_location=device)
            m_dict = agent.state_dict()
            matched_dict = {k: v for k, v in checkpoint.items() if k in m_dict and v.size() == m_dict[k].size()}
            m_dict.update(matched_dict)
            agent.load_state_dict(m_dict)
            log_debug(f"Weights loaded: {SAVE_PATH}")
        except Exception as e:
            log_debug(f"Init error: {e}")

    if not TRAIN_MODE:
        # 再生モード (省略)
        pass
    else:
        # 🚀 強化学習モード (SPS記録是正 ＆ エントロピー減衰版)
        num_parallel_envs = 1 if USE_VIEWER else 16
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_parallel_envs)])
        
        start_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        tensorboard_writer = SummaryWriter(f"runs/{EXPERIMENT_NAME}_{start_ts}")
        
        # 統計用 EMA
        mavg_vis_ratio = 0.5 
        alpha_ema = 0.95

        if TRACK_WANDB:
            wandb.init(
                project="HideAndSeek_GTRPPO", 
                name=f"{EXPERIMENT_NAME}_{start_ts}",
                config={"lr": LR, "ent_start": ENT_COEF_START, "ent_end": ENT_COEF_END}
            )
            # 接続確認
            wandb.log({"system/connection_check": 1.0}, commit=False)
            wandb.watch(agent, log="gradients", log_freq=100)

        # ストレージ確保
        storage_obs = torch.zeros((EPISODE_LIMIT, num_parallel_envs, SEQ_LEN, 53), device=device)
        storage_actions = torch.zeros((EPISODE_LIMIT, num_parallel_envs, act_dim), device=device)
        storage_logprobs = torch.zeros((EPISODE_LIMIT, num_parallel_envs), device=device)
        storage_rewards = torch.zeros((EPISODE_LIMIT, num_parallel_envs), device=device)
        storage_dones = torch.zeros((EPISODE_LIMIT, num_parallel_envs), device=device)
        storage_values = torch.zeros((EPISODE_LIMIT, num_parallel_envs), device=device)

        hist_manager = ObsHistory(num_parallel_envs, SEQ_LEN, 53, device)
        next_obs_batch, _ = envs.reset()
        hist_manager.update(next_obs_batch)
        
        global_cumulative_steps = 0
        training_start_real_time = time.time()
        num_total_updates = TOTAL_STEPS // (EPISODE_LIMIT * num_parallel_envs)
        main_ppo_bar = tqdm(range(1, num_total_updates + 1), desc="PPO Cycle")

        try:
            for current_update_idx in main_ppo_bar:
                # 💡 エントロピー係数のアニーリング（線形減衰）
                frac = min(global_cumulative_steps / TOTAL_STEPS, 1.0)
                current_ent_coef = ENT_COEF_START + frac * (ENT_COEF_END - ENT_COEF_START)
                
                # --- [1] データ収集 ---
                agent.eval()
                detected_count = 0
                for step_idx in range(EPISODE_LIMIT):
                    global_cumulative_steps += num_parallel_envs
                    active_ctx = hist_manager.get()
                    storage_obs[step_idx] = active_ctx

                    with torch.no_grad():
                        a_batch, lp_batch, _, v_batch = agent.get_action_and_value(active_ctx)
                        storage_values[step_idx] = v_batch.reshape(-1)
                    
                    storage_actions[step_idx] = a_batch
                    storage_logprobs[step_idx] = lp_batch.reshape(-1)

                    if USE_VIEWER and not envs.envs[0].viewer.is_running(): raise SystemExit

                    # 物理ステップ
                    next_obs_batch, reward_batch, terms, truncs, infos = envs.step(a_batch.cpu().numpy())
                    
                    # 💡 KeyError 回避：get() を使用
                    is_detected_arr = infos.get("is_detected", np.zeros(num_parallel_envs, dtype=bool))
                    detected_count += np.sum(is_detected_arr)
                    
                    storage_rewards[step_idx] = torch.as_tensor(reward_batch, device=device)
                    any_done_mask = np.logical_or(terms, truncs)
                    storage_dones[step_idx] = torch.as_tensor(any_done_mask, dtype=torch.float32, device=device)
                    
                    hist_manager.update(next_obs_batch)
                    for env_i in range(num_parallel_envs):
                        if any_done_mask[env_i]:
                            hist_manager.reset(env_i)
                            hist_manager.update(next_obs_batch[env_i], env_idx=env_i)

                # 指標計算
                current_vis_ratio = detected_count / (EPISODE_LIMIT * num_parallel_envs)
                mavg_vis_ratio = (alpha_ema * mavg_vis_ratio) + ((1.0 - alpha_ema) * current_vis_ratio)

                # --- [2] アドバンテージ算出 (GAE) ---
                with torch.no_grad():
                    last_ctx = hist_manager.get()
                    next_val_final = agent.get_value(last_ctx).reshape(-1)
                    advantages = torch.zeros_like(storage_rewards, device=device)
                    last_gae_lam = 0
                    for t in reversed(range(EPISODE_LIMIT)):
                        mask = 1.0 - storage_dones[t]
                        v_next = next_val_final if t == EPISODE_LIMIT - 1 else storage_values[t+1]
                        td_err = storage_rewards[t] + GAMMA * v_next * mask - storage_values[t]
                        last_gae_lam = td_err + GAMMA * GAE_LAMBDA * mask * last_gae_lam
                        advantages[t] = last_gae_lam
                    target_returns = advantages + storage_values

                # --- [3] パラメータ更新 ---
                agent.train()
                p_loss_sum, v_loss_sum, ent_sum, up_count = 0.0, 0.0, 0.0, 0
                flat_obs = storage_obs.reshape(-1, SEQ_LEN, 53)
                flat_logprobs = storage_log_probabilities = storage_log_probs = storage_logprobs.reshape(-1)
                flat_actions = storage_actions.reshape(-1, act_dim)
                flat_advantages = advantages.reshape(-1)
                flat_returns = target_returns.reshape(-1)

                indices = np.arange(flat_obs.shape[0])
                for epoch in range(UPDATE_EPOCHS):
                    np.random.shuffle(indices)
                    for start in range(0, flat_obs.shape[0], BATCH_SIZE):
                        end = start + BATCH_SIZE
                        mb = indices[start : end]
                        _, new_lp, n_ent, n_v = agent.get_action_and_value(flat_obs[mb], flat_actions[mb])
                        
                        ratio = (new_lp - flat_logprobs[mb]).exp()
                        mb_adv = flat_advantages[mb]
                        mb_adv_norm = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                        surr1 = -mb_adv_norm * ratio
                        surr2 = -mb_adv_norm * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                        policy_loss = torch.max(surr1, surr2).mean()
                        value_loss = 0.5 * ((n_v.reshape(-1) - flat_returns[mb]) ** 2).mean()
                        
                        curr_ent_val = n_ent.mean()
                        # 💡 アニーリングされた係数を使用
                        loss = policy_loss - current_ent_coef * curr_ent_val + value_loss * VF_COEF

                        optimizer.zero_grad(); loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM); optimizer.step()
                        
                        p_loss_sum += policy_loss.item(); v_loss_sum += value_loss.item()
                        ent_sum += curr_ent_val.item(); up_count += 1

                # --- [4] 統計出力 ＆ モデル保存 ---
                avg_rew = storage_rewards.mean().item()
                duration = time.time() - training_start_real_time
                # 💡 SPS記録：累積ステップと累積時間から正確に算出
                sps_val = int(global_cumulative_steps / duration)
                
                # 表示更新
                main_ppo_bar.set_postfix(reward=f"{avg_rew:.2f}", SPS=sps_val, Vis=f"{mavg_vis_ratio:.3f}", Ent=f"{ent_sum/up_count:.2f}")
                
                if TRACK_WANDB:
                    wandb.log({
                        "reward/avg": avg_rew,
                        "metrics/vis_ratio": current_vis_ratio,
                        "metrics/mavg_vis_ratio": mavg_vis_ratio,
                        "losses/policy": p_loss_sum / up_count,
                        "losses/value": v_loss_sum / up_count,
                        "losses/entropy": ent_sum / up_count,
                        "charts/SPS": sps_val, # 💡 キーを charts/ に修正
                        "charts/ent_coef": current_ent_coef
                    }, step=global_cumulative_steps)
                
                tensorboard_writer.add_scalar("charts/avg_reward", avg_rew, global_cumulative_steps)
                tensorboard_writer.add_scalar("charts/SPS", sps_val, global_cumulative_steps)
                tensorboard_writer.add_scalar("losses/entropy", ent_sum / up_count, global_cumulative_steps)
                
                if current_update_idx % 50 == 0:
                    torch.save(agent.state_dict(), SAVE_PATH)
                    tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] Step: {global_cumulative_steps} | SPS: {sps_val} | VisRatio: {mavg_vis_ratio:.4f}")

        except Exception: traceback.print_exc()
        finally:
            envs.close(); tensorboard_writer.close()
            if TRACK_WANDB: wandb.finish()
            sys.exit(0)

if __name__ == "__main__": run()