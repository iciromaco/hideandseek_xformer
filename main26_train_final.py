# main26_train_final.py v2.30
# 演習第26回：【全ロジック極限展開・復元 ＆ Windows/NDR/WANDB完全統合版】
# 
# 修正内容:
# 1. 演算ロジックの再展開 (行数と可読性の完全復旧):
#    - 前回の更新で凝縮された GAE、PPO損失計算、パラメータ更新、評価ループをすべて1行1命令に再展開。
#    - 中間変数を明示的に使用し、テンソル演算の各ステップを独立した行で記述。
# 2. Windows 最適化設定の維持:
#    - 非ARM環境におけるスレッド数制限 (OMP/MKL等) を冒頭に配置。
# 3. 重要指標 NDR (No-Detected Ratio) の完全計算:
#    - find_buf を用いた「チーム生存率」の算出プロセスを詳細に記述。
# 4. WANDB 統合の完全記述:
#    - init, log, finish の各フェーズを省略なく実装。
# 5. 堅牢なリソース管理: try-finally ブロックによる確実な終了処理。

import os
import sys
import time
import random
import signal
import platform
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# 自作モジュール
from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2


STOP_REQUESTED = False


def _handle_termination_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    raise KeyboardInterrupt

# ==========================================
# 0. 環境変数設定 (Windows/並列実行最適化)
# ==========================================
processor_type = platform.processor()
if processor_type != 'arm':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ==========================================
# 1. ハイパーパラメータ ＆ グローバル設定
# ==========================================
TRAIN_MODE = True            # 学習時は True、再生時は False
USE_VIEWER = True             # 再生・学習時に Viewer を使用するか
MODE = "initial"              # initial (4次元) or refinement (8次元)
TRACK_WANDB = False           # wandb ログの使用フラグ

# Transformer 設定
SEQ_LEN = 8                   # 過去の観測を参照する長さ
HIDDEN_DIM = 128              # 隠れ層のユニット数

# エピソード ＆ 学習設定
EPISODE_LIMIT = 500           # 1エピソードの最大ステップ
TOTAL_STEPS = 20_000_000      # 総学習ステップ数
NUM_ENVS = 16                 # 並列実行する環境数

# PPO 最適化設定
LR_START = 2.5e-4             # 学習率（初期値）
LR_END = 5e-6                 # 学習率（最終値）
GAMMA = 0.99                  # 割引報酬率
GAE_LAMBDA = 0.95             # GAE (Generalized Advantage Estimation)
UPDATE_EPOCHS = 4             # 1データセットあたりの更新回数
BATCH_SIZE = 512              # ミニバッチサイズ
CLIP_COEF = 0.2               # PPO クリップ範囲
MAX_GRAD_NORM = 0.5           # 勾配クリッピング閾値

# 報酬・探索・評価設定
ENT_COEF_START = 0.01         # エントロピー係数（初期値）
ENT_COEF_END = 0.001          # エントロピー係数（最終値）
VF_COEF = 0.5                 # 価値関数の損失係数
TARGET_KL = 0.02              # KL ダイバージェンスによる早期停止閾値
EVAL_INTERVAL = 20            # N 更新ごとに評価を実施
EVAL_EPISODES = 3             # 評価時のエピソード数

# 保存設定
SAVE_PATH = f"HNS_V26_GTRPPO_{MODE}.pt"
BEST_SAVE_PATH = f"HNS_V26_GTRPPO_{MODE}_best.pt"

# WANDB インポート（フラグ有効時のみ）
if TRACK_WANDB:
    import wandb

# ==========================================
# 2. 履歴管理 (ObsHistory)
# ==========================================
class ObsHistory:
    """Transformer への入力となる時系列バッファを管理"""
    def __init__(self, n_envs, seq_len, obs_dim, device):
        self.n_envs = n_envs
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.device = device
        self.buffer_size = seq_len * 2
        self.buffer_shape = (n_envs, self.buffer_size, obs_dim)
        self.buffer = torch.zeros(self.buffer_shape, device=device)
        self.ptr = 0

    def reset(self, env_idx=None):
        if env_idx is None:
            self.buffer.zero_()
            self.ptr = 0
        else:
            self.buffer[env_idx].zero_()

    def update(self, obs_data, env_idx=None):
        obs_tensor = torch.as_tensor(obs_data, dtype=torch.float32, device=self.device)
        if env_idx is None:
            # 全環境一括更新
            obs_reshaped = obs_tensor.reshape(self.n_envs, self.obs_dim)
            write_pos = self.ptr
            self.buffer[:, write_pos] = obs_reshaped
            mirror_pos = write_pos + self.seq_len
            self.buffer[:, mirror_pos] = obs_reshaped
            self.ptr = (self.ptr + 1) % self.seq_len
        else:
            # 特定環境のみ更新（リセット時）
            obs_reshaped = obs_tensor.reshape(self.obs_dim)
            write_pos = self.ptr
            self.buffer[env_idx, write_pos] = obs_reshaped
            mirror_pos = write_pos + self.seq_len
            self.buffer[env_idx, mirror_pos] = obs_reshaped

    def get(self):
        """現在の ptr から過去 SEQ_LEN 分のシーケンスを取得"""
        s_idx = self.ptr
        e_idx = self.ptr + self.seq_len
        sequence_data = self.buffer[:, s_idx:e_idx]
        contiguous_data = sequence_data.contiguous()
        return contiguous_data

def make_env():
    render_mode = None
    if USE_VIEWER:
        if not TRAIN_MODE:
            render_mode = "human"
    env_instance = TeamCosEnv(mode=MODE, render_mode=render_mode)
    return env_instance

# ==========================================
# 3. メイン実行ロジック (Run)
# ==========================================
def run():
    global STOP_REQUESTED
    STOP_REQUESTED = False
    try:
        signal.signal(signal.SIGINT, _handle_termination_signal)
        signal.signal(signal.SIGTERM, _handle_termination_signal)
    except Exception:
        pass

    # ハードウェア・シード設定
    cuda_available = torch.cuda.is_available()
    device_name = "cuda" if cuda_available else "cpu"
    device = torch.device(device_name)
    
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    
    # ネットワーク設定
    obs_probe_env = make_env()
    obs_dimension = int(obs_probe_env.observation_space.shape[0])
    obs_probe_env.close()
    action_dimension = 8
    if MODE == "initial":
        action_dimension = 4
        
    # エージェント初期化
    agent = AgentV2(obs_dimension, action_dimension, HIDDEN_DIM, SEQ_LEN)
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR_START, eps=1e-5)
    
    # モデルのロード処理
    if os.path.exists(SAVE_PATH):
        try:
            checkpoint = torch.load(SAVE_PATH, map_location=device)
            agent.load_state_dict(checkpoint)
            print(f"✅ Successfully loaded model: {SAVE_PATH}")
        except Exception as load_err:
            print(f"⚠️ Load failed: {load_err}. Starting from scratch.")

    # ------------------------------------------
    # [A] 再生モード (PLAYBACK)
    # ------------------------------------------
    if not TRAIN_MODE:
        env = make_env()
        history = ObsHistory(1, SEQ_LEN, obs_dimension, device)
        agent.eval()
        
        obs_current, _ = env.reset()
        history.update(obs_current)
        render_enabled = USE_VIEWER
        render_warned = False
        
        if USE_VIEWER:
            env.render()
            time.sleep(0.5)
            
        target_fps = 40
        frame_duration = 1.0 / target_fps
        print(f"🎮 Playback Mode Active.")
        
        try:
            while True:
                if STOP_REQUESTED:
                    raise KeyboardInterrupt
                if render_enabled:
                    if env.viewer is None:
                        break
                    if not env.viewer.is_running():
                        break
                if render_enabled:
                    try:
                        env.render()
                        time.sleep(0.5)
                    except RuntimeError as render_err:
                        if not render_warned:
                            print(f"⚠️ Viewer disabled: {render_err}")
                            render_warned = True
                        render_enabled = False
                        
                start_tick = time.perf_counter()
                
                with torch.no_grad():
                    sequence_in = history.get()
                    action_out, _, _, _ = agent.get_action_and_value(sequence_in)
                    
                action_np = action_out.cpu().numpy().flatten()
                obs_next, reward_step, terminated, truncated, info_step = env.step(action_np)
                
                history.update(obs_next)
                
                if terminated or truncated:
                    obs_re, _ = env.reset()
                    history.reset()
                    history.update(obs_re)
                    start_tick = time.perf_counter()
                    
                elapsed_tick = time.perf_counter() - start_tick
                wait_time = frame_duration - elapsed_tick
                if wait_time > 0:
                    time.sleep(wait_time)
                else:
                    time.sleep(0.001)
        finally:
            env.close()
            print("🏁 Playback terminated.")
            sys.exit(0)

    # ------------------------------------------
    # [B] 学習モード (TRAINING)
    # ------------------------------------------
    else:
        print(f"🚀 Training Mode Started: {MODE}")
        
        # 1. W&B 初期化
        if TRACK_WANDB:
            time_str = datetime.now().strftime("%m%d_%H%M")
            run_name = f"ppo_transformer_{MODE}_{time_str}"
            wandb.init(
                project="HNS_V26_TeamCos",
                name=run_name,
                config={
                    "mode": MODE,
                    "learning_rate": LR_START,
                    "batch_size": BATCH_SIZE,
                    "total_steps": TOTAL_STEPS,
                    "num_envs": NUM_ENVS,
                    "sequence_length": SEQ_LEN,
                    "hidden_dim": HIDDEN_DIM,
                    "gamma": GAMMA,
                    "gae_lambda": GAE_LAMBDA,
                    "clip_coef": CLIP_COEF,
                    "update_epochs": UPDATE_EPOCHS
                }
            )

        # 2. 環境 ＆ 履歴バッファ ＆ TensorBoard
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(NUM_ENVS)])
        history = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dimension, device)
        log_timestamp = datetime.now().strftime("%m%d_%H%M")
        tb_log_dir = f"runs/HNS_{MODE}_{log_timestamp}"
        writer = SummaryWriter(tb_log_dir)
        render_enabled = USE_VIEWER
        render_warned = False
        
        # 3. 学習用バッファの確保
        obs_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS, SEQ_LEN, obs_dimension), device=device)
        act_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS, action_dimension), device=device)
        prob_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        rew_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        done_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        val_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        # NDR (No-Detected Ratio) 計測用
        find_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        
        total_global_step = 0
        update_iteration = 0
        max_eval_reward = -float('inf')
        training_start_time = time.time()
        
        # 初期観測の取得
        obs_initial, _ = envs.reset()
        history.update(obs_initial)
        
        if render_enabled:
            try:
                envs.envs[0].render()
                time.sleep(0.5)
            except RuntimeError as render_err:
                if not render_warned:
                    print(f"⚠️ Viewer disabled: {render_err}")
                    render_warned = True
                render_enabled = False
        
        try:
            while total_global_step < TOTAL_STEPS:
                if STOP_REQUESTED:
                    raise KeyboardInterrupt
                
                # --- 4. データ収集フェーズ (Rollout) ---
                agent.eval()
                for step_idx in range(EPISODE_LIMIT):
                    if STOP_REQUESTED:
                        raise KeyboardInterrupt
                    total_global_step = total_global_step + NUM_ENVS
                    
                    rollout_sequence = history.get()
                    obs_buffer[step_idx] = rollout_sequence

                    if render_enabled:
                        try:
                            envs.envs[0].render()
                        except RuntimeError as render_err:
                            if not render_warned:
                                print(f"⚠️ Viewer disabled: {render_err}")
                                render_warned = True
                            render_enabled = False
                        
                    with torch.no_grad():
                        action_vec, logprob_vec, _, value_vec = agent.get_action_and_value(rollout_sequence)
                        val_buffer[step_idx] = value_vec.flatten()
                        
                    act_buffer[step_idx] = action_vec
                    prob_buffer[step_idx] = logprob_vec
                    
                    next_obs_raw, reward_vec, done_vec, trunc_vec, info_dict = envs.step(action_vec.cpu().numpy())
                    
                    rew_buffer[step_idx] = torch.as_tensor(reward_vec, device=device)
                    done_buffer[step_idx] = torch.as_tensor(done_vec, device=device)
                    
                    # NDRデータの抽出
                    is_detected_flags = info_dict.get("is_detected", np.zeros(NUM_ENVS))
                    find_buffer[step_idx] = torch.as_tensor(is_detected_flags, dtype=torch.float32, device=device)
                    
                    history.update(next_obs_raw)
                    
                    # 個別環境のリセット対応
                    for env_idx in range(NUM_ENVS):
                        if done_vec[env_idx] or trunc_vec[env_idx]:
                            history.reset(env_idx=env_idx)
                            history.update(next_obs_raw[env_idx], env_idx=env_idx)

                # --- 5. アドバンテージ計算フェーズ (GAE) ---
                agent.eval()
                with torch.no_grad():
                    final_sequence = history.get()
                    v_next_raw = agent.get_value(final_sequence)
                    v_next_val = v_next_raw.reshape(1, -1)
                    
                    advantage_buffer = torch.zeros_like(rew_buffer, device=device)
                    last_gae_value = 0
                    
                    for t_step in reversed(range(EPISODE_LIMIT)):
                        if t_step == EPISODE_LIMIT - 1:
                            terminal_mask = 1.0 - done_buffer[t_step]
                            v_target_step = v_next_val
                        else:
                            terminal_mask = 1.0 - done_buffer[t_step]
                            v_target_step = val_buffer[t_step + 1]
                            
                        delta_step = rew_buffer[t_step] + GAMMA * v_target_step * terminal_mask - val_buffer[t_step]
                        last_gae_value = delta_step + GAMMA * GAE_LAMBDA * terminal_mask * last_gae_value
                        advantage_buffer[t_step] = last_gae_value
                        
                    return_buffer = advantage_buffer + val_buffer

                # --- 6. 最適化フェーズ (PPO Optimization) ---
                agent.train()
                # バッファのフラット化
                flat_obs = obs_buffer.reshape(-1, SEQ_LEN, obs_dimension)
                flat_act = act_buffer.reshape(-1, action_dimension)
                flat_prob = prob_buffer.reshape(-1)
                flat_adv = advantage_buffer.reshape(-1)
                flat_ret = return_buffer.reshape(-1)
                flat_val = val_buffer.reshape(-1)
                
                total_samples = flat_obs.shape[0]
                sample_indices = np.arange(total_samples)
                clip_rate_history = []
                current_kl_div = torch.tensor(0.0, device=device)
                
                for epoch_idx in range(UPDATE_EPOCHS):
                    np.random.shuffle(sample_indices)
                    for start_pos in range(0, total_samples, BATCH_SIZE):
                        end_pos = start_pos + BATCH_SIZE
                        minibatch_idx = sample_indices[start_pos:end_pos]
                        
                        _, new_logprob, entropy_vec, new_value = agent.get_action_and_value(
                            flat_obs[minibatch_idx], 
                            flat_act[minibatch_idx]
                        )
                        
                        log_ratio = new_logprob - flat_prob[minibatch_idx]
                        approx_ratio = log_ratio.exp()
                        
                        with torch.no_grad():
                            # KL ダイバージェンス計測
                            current_kl_div = ((approx_ratio - 1.0) - log_ratio).mean()
                            clip_rate = (approx_ratio - 1.0).abs().gt(CLIP_COEF).float().mean().item()
                            clip_rate_history.append(clip_rate)
                        
                        # アドバンテージの正規化 (ミニバッチ単位)
                        mb_advantage = flat_adv[minibatch_idx]
                        mb_adv_mean = mb_advantage.mean()
                        mb_adv_std = mb_advantage.std()
                        mb_adv_norm = (mb_advantage - mb_adv_mean) / (mb_adv_std + 1e-8)
                        
                        # PPO Surrogate Loss
                        pg_loss_unclipped = -mb_adv_norm * approx_ratio
                        pg_ratio_clipped = torch.clamp(approx_ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                        pg_loss_clipped = -mb_adv_norm * pg_ratio_clipped
                        policy_loss_step = torch.max(pg_loss_unclipped, pg_loss_clipped).mean()
                        
                        # Value Function Loss
                        v_diff = new_value.flatten() - flat_ret[minibatch_idx]
                        value_loss_step = 0.5 * (v_diff ** 2).mean()
                        
                        # ハイパーパラメータのアニーリング (線形減衰)
                        learning_progress = 1.0 - (total_global_step / TOTAL_STEPS)
                        current_lr = LR_END + (LR_START - LR_END) * learning_progress
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = current_lr
                            
                        current_ent_coef = ENT_COEF_END + (ENT_COEF_START - ENT_COEF_END) * learning_progress
                        entropy_loss_step = entropy_vec.mean()
                        
                        # Total Loss 合算
                        total_loss = policy_loss_step - current_ent_coef * entropy_loss_step + value_loss_step * VF_COEF
                        
                        optimizer.zero_grad()
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                        
                    # KL による早期停止チェック
                    if current_kl_div > TARGET_KL:
                        break
                
                update_iteration = update_iteration + 1

                # --- 7. 評価フェーズ (Evaluation) ---
                if update_iteration % EVAL_INTERVAL == 0:
                    agent.eval()
                    evaluation_episode_rewards = []
                    
                    for _ in range(EVAL_EPISODES):
                        eval_obs_raw, _ = envs.reset()
                        eval_history = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dimension, device)
                        eval_history.update(eval_obs_raw)
                        eval_accumulated_reward = np.zeros(NUM_ENVS)
                        
                        for _ in range(EPISODE_LIMIT):
                            with torch.no_grad():
                                eval_sequence = eval_history.get()
                                eval_action, _, _, _ = agent.get_action_and_value(eval_sequence)
                                
                            eval_next_obs, eval_reward, eval_done, eval_trunc, _ = envs.step(eval_action.cpu().numpy())
                            eval_accumulated_reward = eval_accumulated_reward + eval_reward
                            eval_history.update(eval_next_obs)
                            
                            if any(eval_done) or any(eval_trunc):
                                break
                                
                        avg_ep_reward = np.mean(eval_accumulated_reward)
                        evaluation_episode_rewards.append(avg_ep_reward)
                        
                    final_avg_eval_reward = np.mean(evaluation_episode_rewards)
                    writer.add_scalar("eval/avg_reward", final_avg_eval_reward, total_global_step)
                    
                    if TRACK_WANDB:
                        wandb.log({"eval/avg_reward": final_avg_eval_reward}, step=total_global_step)
                        
                    # ベストモデルの保存
                    if final_avg_eval_reward > max_eval_reward:
                        max_eval_reward = final_avg_eval_reward
                        torch.save(agent.state_dict(), BEST_SAVE_PATH)

                # --- 8. 統計ロギングフェーズ (Logging) ---
                y_prediction = flat_val.cpu().numpy()
                y_true_value = flat_ret.cpu().numpy()
                variance_y = np.var(y_true_value)
                if variance_y > 0:
                    residual_variance = np.var(y_true_value - y_prediction)
                    explained_variance_value = 1.0 - residual_variance / variance_y
                else:
                    explained_variance_value = np.nan
                    
                current_duration = time.time() - training_start_time
                steps_per_second = int(total_global_step / current_duration)
                mean_rollout_reward = rew_buffer.mean().item()
                
                # 重要指標：非検知割合 (NDR) の算出
                mean_detection_flag = find_buffer.mean().item()
                no_detected_ratio_value = 1.0 - mean_detection_flag
                
                # 標準出力
                print(f"Step: {total_global_step:8d} | SPS: {steps_per_second:4d} | Rew: {mean_rollout_reward:7.4f} | NDR: {no_detected_ratio_value:6.2%} | KL: {current_kl_div:6.4f}")
                
                # TensorBoard 出力
                writer.add_scalar("params/learning_rate", current_lr, total_global_step)
                writer.add_scalar("params/entropy_coef", current_ent_coef, total_global_step)
                writer.add_scalar("losses/value_loss", value_loss_step.item(), total_global_step)
                writer.add_scalar("losses/policy_loss", policy_loss_step.item(), total_global_step)
                writer.add_scalar("losses/entropy", entropy_loss_step.item(), total_global_step)
                writer.add_scalar("losses/approx_kl", current_kl_div.item(), total_global_step)
                writer.add_scalar("losses/clip_fraction", np.mean(clip_rate_history), total_global_step)
                writer.add_scalar("losses/explained_variance", explained_variance_value, total_global_step)
                writer.add_scalar("charts/SPS", steps_per_second, total_global_step)
                writer.add_scalar("charts/avg_reward", mean_rollout_reward, total_global_step)
                writer.add_scalar("charts/no_detected_ratio", no_detected_ratio_value, total_global_step)
                
                # WANDB 出力
                if TRACK_WANDB:
                    wandb.log({
                        "params/learning_rate": current_lr,
                        "params/entropy_coef": current_ent_coef,
                        "losses/value_loss": value_loss_step.item(),
                        "losses/policy_loss": policy_loss_step.item(),
                        "losses/entropy": entropy_loss_step.item(),
                        "losses/approx_kl": current_kl_div.item(),
                        "losses/clip_fraction": np.mean(clip_rate_history),
                        "losses/explained_variance": explained_variance_value,
                        "charts/SPS": steps_per_second,
                        "charts/avg_reward": mean_rollout_reward,
                        "charts/no_detected_ratio": no_detected_ratio_value
                    }, step=total_global_step)
                
                # 定期的なモデル保存
                torch.save(agent.state_dict(), SAVE_PATH)

        except KeyboardInterrupt:
            print("✋ Training interrupted by user.")
        except Exception as e:
            traceback.print_exc()
        finally:
            # 終了処理
            local_context = locals()
            if 'envs' in local_context:
                envs.close()
            if 'writer' in local_context:
                writer.close()
            if TRACK_WANDB:
                wandb.finish()
            print("💾 Training session closed and resources released.")

if __name__ == "__main__":
    run()