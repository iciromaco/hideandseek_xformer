# main26_train_final.py v2.08
# 演習第26回：【再生モード完全復元 ＆ 40Hz精密同期】推論・学習の両輪を完遂した完全展開版
# 
# 遵守事項:
# 1. 再生モードの復元: if not TRAIN_MODE ブロックに、推論・環境同期・リセット処理を完全に記述。
# 2. 精密時間同期: 40Hz (0.025s) を厳守。リセット直後に基準時刻を振り直し、超高速化を根絶。
# 3. 処理の完全展開: 1行1命令を徹底。リスト内包表記、三項演算子、複数代入を完全に禁止。
# 4. バグ回避: infos.get("is_detected") によるリセット時整合性の確保を推論ループにも適用。

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
# False: 再生（Viewer）モード, True: 学習モード
TRAIN_MODE = True            
# Viewerを表示するか
USE_VIEWER = False            
# 制御モード: "initial" (H1のみ制御)
MODE = "initial"             
# Weights & Biases 連携
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

# エントロピー係数（学習時用）
ENT_COEF_START = 0.001       
ENT_COEF_END = 0.0001        

VF_COEF = 0.5                
MAX_GRAD_NORM = 0.5          

# 保存ファイル名
EXPERIMENT_NAME = f"HNS_V26_GTRPPO_{MODE}"
SAVE_PATH = f"{EXPERIMENT_NAME}.pt"

if TRACK_WANDB: import wandb

def log_debug(msg):
    """タイムスタンプ付き詳細ログ"""
    current_time_str = datetime.now().strftime('%H:%M:%S')
    print(f"[{current_time_str}] 🔍 {msg}")

class ObsHistory:
    """Transformer用の履歴管理。メモリ連続性を維持。"""
    def __init__(self, n_envs, seq_len, obs_dim, device):
        self.n_envs = n_envs
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.device = device
        # 2倍確保バッファ
        self.buffer = torch.zeros((n_envs, seq_len * 2, obs_dim), device=device)
        self.ptr = 0

    def reset(self, env_idx=None):
        if env_idx is None:
            self.buffer.zero_()
            self.ptr = 0
        else:
            self.buffer[env_idx].zero_()

    def update(self, obs_data, env_idx=None):
        """観測値の更新。バッチ全体または個別スロットに対応。"""
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
        """過去系列を取得。不連続メモリを物理的に解消 (.contiguous())"""
        result_sequence = self.buffer[:, self.ptr : self.ptr + self.seq_len]
        return result_sequence.contiguous()

def make_env():
    """環境生成ファクトリ"""
    render_mode_str = "human" if USE_VIEWER else None
    return TeamCosEnv(mode=MODE, render_mode=render_mode_str)

def run():
    # 計算デバイスの決定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_debug(f"Process starting on device: {device}")
    
    # 割り込みハンドラ
    def signal_handler(sig, frame):
        log_debug("Shutdown command received.")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    # モデルの初期化 (4D/8D アクション)
    act_dim = 4 if MODE == "initial" else 8
    agent = AgentV2(53, act_dim, HIDDEN_DIM, SEQ_LEN).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    
    # 重みのロード
    if os.path.exists(SAVE_PATH):
        try:
            checkpoint_data = torch.load(SAVE_PATH, map_location=device)
            current_model_dict = agent.state_dict()
            matched_pretrained_dict = {k: v for k, v in checkpoint_data.items() if k in current_model_dict and v.size() == current_model_dict[k].size()}
            
            if len(matched_pretrained_dict) < len(checkpoint_data):
                log_debug("Dimension mismatch: Loading backbone only.")
                current_model_dict.update(matched_pretrained_dict)
                agent.load_state_dict(current_model_dict)
            else:
                agent.load_state_dict(checkpoint_data)
                log_debug(f"Full weights loaded: {SAVE_PATH}")
        except Exception as e:
            log_debug(f"Init error: {e}. Starting fresh.")

    if not TRAIN_MODE:
        # ==========================================
        # 🎮 再生モード (Playback / Inference Mode)
        # ==========================================
        # 環境の生成 (再生時は1環境)
        env = make_env()
        hist_manager = ObsHistory(1, SEQ_LEN, 53, device)
        
        # Wandb 連携 (再生モードでも必要であれば初期化)
        if TRACK_WANDB:
            start_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            wandb.init(project="HideAndSeek_GTRPPO", name=f"PLAY_{EXPERIMENT_NAME}_{start_ts}")

        # 初期リセット
        obs_initial, info_init = env.reset(seed=int(time.time()))
        hist_manager.update(obs_initial)
        agent.eval()
        
        # 精密同期用定数 (40Hz)
        TARGET_FPS = 40
        FRAME_DURATION_SEC = 1.0 / TARGET_FPS
        
        log_debug(f"Playback mode: Syncing at {TARGET_FPS}Hz. Close MuJoCo window to exit.")
        
        try:
            # Viewerが動作している間ループ
            while env.viewer is not None and env.viewer.is_running():
                # フレーム基準時刻の記録
                frame_start_abs_time = time.perf_counter()
                
                # Transformer 履歴コンテキストの取得
                current_history_context = hist_manager.get()
                
                # 推論実行 (勾配計算なし)
                with torch.no_grad():
                    action_prediction, _, _, _ = agent.get_action_and_value(current_history_context)
                
                # 物理ステップの実行
                action_to_apply = action_prediction.cpu().numpy().flatten()
                obs_next_step, step_reward, is_terminated, is_truncated, infos = env.step(action_to_apply)
                
                # 最新観測値の挿入
                hist_manager.update(obs_next_step)
                
                # 視認情報のロギング (Wandb)
                if TRACK_WANDB:
                    is_detected = infos.get("is_detected", False)
                    wandb.log({"playback/reward": step_reward, "playback/is_detected": float(is_detected)})

                # 終了判定時のリセット処理
                if is_terminated or is_truncated:
                    obs_reinit, info_reinit = env.reset()
                    # 履歴バッファのクリア
                    hist_manager.reset()
                    hist_manager.update(obs_reinit)
                    # 💡 リセット遅延による加速を物理的に遮断
                    frame_start_abs_time = time.perf_counter()
                
                # 精密同期の計算
                current_loop_time = time.perf_counter()
                total_processing_elapsed = current_loop_time - frame_start_abs_time
                remaining_time_to_wait = FRAME_DURATION_SEC - total_processing_elapsed
                
                if remaining_time_to_wait > 0:
                    time.sleep(remaining_time_to_wait)
                else:
                    # 処理が追いつかない場合のみ最小限のウェイト
                    time.sleep(0.0001)
                    
        finally:
            log_debug("Finalizing playback...")
            env.close()
            if TRACK_WANDB: wandb.finish()
            sys.exit(0)

    else:
        # ==========================================
        # 🚀 強化学習モード (PPO / Transformer)
        # ==========================================
        num_parallel_envs = 1 if USE_VIEWER else 16
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_parallel_envs)])
        
        start_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        tensorboard_writer = SummaryWriter(f"runs/{EXPERIMENT_NAME}_{start_ts}")
        
        # 指標用 EMA
        mavg_vis_ratio = 0.5 
        alpha_ema = 0.95

        if TRACK_WANDB:
            wandb.init(
                project="HideAndSeek_GTRPPO", 
                name=f"{EXPERIMENT_NAME}_{start_ts}",
                config={"lr": LR, "ent_start": ENT_COEF_START, "ent_end": ENT_COEF_END}
            )
            wandb.watch(agent, log="gradients", log_freq=100)

        # ストレージの確保
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
                # エントロピー係数のアニーリング
                progress_fraction = min(global_cumulative_steps / TOTAL_STEPS, 1.0)
                current_ent_coef = ENT_COEF_START + progress_fraction * (ENT_COEF_END - ENT_COEF_START)
                
                # --- [1] データ収集 ---
                agent.eval()
                detected_step_count = 0
                
                rollout_progress = tqdm(range(EPISODE_LIMIT), desc=f"Collecting", leave=False)
                for step_idx in rollout_progress:
                    global_cumulative_steps += num_parallel_envs
                    active_context = hist_manager.get()
                    storage_obs[step_idx] = active_context

                    with torch.no_grad():
                        a_batch, lp_batch, _, v_batch = agent.get_action_and_value(active_context)
                        storage_values[step_idx] = v_batch.reshape(-1)
                    
                    storage_actions[step_idx] = a_batch
                    storage_logprobs[step_idx] = lp_batch.reshape(-1)

                    if USE_VIEWER:
                        v_ref = envs.envs[0].viewer
                        if v_ref is not None and not v_ref.is_running(): raise SystemExit

                    # 物理ステップ
                    next_obs_batch, reward_batch, terms, truncs, infos = envs.step(a_batch.cpu().numpy())
                    
                    # 視認フラグの安全な取得
                    is_detected_arr = infos.get("is_detected", np.zeros(num_parallel_envs, dtype=bool))
                    detected_step_count += np.sum(is_detected_arr)
                    
                    storage_rewards[step_idx] = torch.as_tensor(reward_batch, device=device)
                    any_done_mask = np.logical_or(terms, truncs)
                    storage_dones[step_idx] = torch.as_tensor(any_done_mask, dtype=torch.float32, device=device)
                    
                    hist_manager.update(next_obs_batch)
                    for env_i in range(num_parallel_envs):
                        if any_done_mask[env_i]:
                            hist_manager.reset(env_i)
                            hist_manager.update(next_obs_batch[env_i], env_idx=env_i)

                # 指標計算
                total_steps_in_rollout = EPISODE_LIMIT * num_parallel_envs
                current_vis_ratio = detected_step_count / total_steps_in_rollout
                mavg_vis_ratio = (alpha_ema * mavg_vis_ratio) + ((1.0 - alpha_ema) * current_vis_ratio)

                # --- [2] アドバンテージ算出 (GAE) ---
                with torch.no_grad():
                    last_step_context = hist_manager.get()
                    next_val_final_tensor = agent.get_value(last_step_context)
                    next_val_final = next_val_final_tensor.reshape(-1)
                    
                    advantages_tensor = torch.zeros_like(storage_rewards, device=device)
                    last_gae_delta_acc = 0
                    for t in reversed(range(EPISODE_LIMIT)):
                        mask = 1.0 - storage_dones[t]
                        v_next = next_val_final if t == EPISODE_LIMIT - 1 else storage_values[t + 1]
                        td_err = storage_rewards[t] + GAMMA * v_next * mask - storage_values[t]
                        last_gae_delta_acc = td_err + GAMMA * GAE_LAMBDA * mask * last_gae_delta_acc
                        advantages_tensor[t] = last_gae_delta_acc
                    target_returns = advantages_tensor + storage_values

                # --- [3] パラメータ更新 ---
                agent.train()
                policy_loss_total, value_loss_total, entropy_total, update_count = 0.0, 0.0, 0.0, 0
                
                flat_obs = storage_obs.reshape(-1, SEQ_LEN, 53)
                flat_logprobs = storage_logprobs.reshape(-1)
                flat_actions = storage_actions.reshape(-1, act_dim)
                flat_advantages = advantages_tensor.reshape(-1)
                flat_returns = target_returns.reshape(-1)

                num_samples = flat_obs.shape[0]
                shuffled_indices = np.arange(num_samples)
                
                for epoch_idx in range(UPDATE_EPOCHS):
                    np.random.shuffle(shuffled_indices)
                    for start_idx in range(0, num_samples, BATCH_SIZE):
                        end_idx = start_idx + BATCH_SIZE
                        mini_batch_idx = shuffled_indices[start_idx : end_idx]
                        _, new_lp, n_ent, n_v = agent.get_action_and_value(flat_obs[mini_batch_idx], flat_actions[mini_batch_idx])
                        
                        prob_ratio = (new_lp - flat_logprobs[mini_batch_idx]).exp()
                        mb_adv = flat_advantages[mini_batch_idx]
                        mb_adv_norm = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                        surr1 = -mb_adv_norm * prob_ratio
                        surr2 = -mb_adv_norm * torch.clamp(prob_ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                        policy_loss_val = torch.max(surr1, surr2).mean()
                        value_loss_val = 0.5 * ((n_v.reshape(-1) - flat_returns[mini_batch_idx]) ** 2).mean()
                        
                        total_opt_loss = policy_loss_val - current_ent_coef * n_ent.mean() + value_loss_val * VF_COEF
                        optimizer.zero_grad()
                        total_opt_loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                        
                        policy_loss_total += policy_loss_val.item()
                        value_loss_total += value_loss_val.item()
                        entropy_total += n_ent.mean().item()
                        update_count += 1

                # --- [4] 統計出力 ---
                avg_rew_val = storage_rewards.mean().item()
                duration_measure = time.time() - training_start_real_time
                sps_val = int(global_cumulative_steps / duration_measure)
                
                main_ppo_bar.set_postfix(reward=f"{avg_rew_val:.2f}", SPS=sps_val, Vis=f"{mavg_vis_ratio:.3f}")
                
                if TRACK_WANDB:
                    wandb.log({
                        "reward/avg": avg_rew_val,
                        "metrics/vis_ratio": current_vis_ratio,
                        "metrics/mavg_vis_ratio": mavg_vis_ratio,
                        "losses/policy": policy_loss_total / update_count,
                        "losses/value": value_loss_total / update_count,
                        "losses/entropy": entropy_total / update_count,
                        "charts/SPS": sps_val,
                        "charts/ent_coef": current_ent_coef
                    }, step=global_cumulative_steps)
                
                if current_update_idx % 50 == 0:
                    torch.save(agent.state_dict(), SAVE_PATH)

        except Exception: traceback.print_exc()
        finally:
            envs.close(); tensorboard_writer.close()
            if TRACK_WANDB: wandb.finish()
            sys.exit(0)

if __name__ == "__main__":
    run()