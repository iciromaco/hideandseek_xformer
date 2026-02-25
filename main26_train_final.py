# main26_train_final.py v2.22
# 演習第26回：【55次元観測空間同期 ＆ 1行1文死守版】
# 
# 修正内容:
# 1. 次元数不一致の解消: 観測次元を 53 から 55 (55 * 16 = 880) へ修正。
#    - 環境側 (v2.20以降) で追加された敵・味方の相対回転情報に対応。
# 2. 1行1命令の厳格遵守: テンソル生成、代入、計算ステップをすべて独立した行で記述。
# 3. 堅牢なリソース管理: locals() を用いた終了処理を維持し、再生モードでの NameError を防止。
# 4. ロジックの完全復元: アドバンテージ計算、アニーリング、評価フェーズを省略なしで展開。

import os
import sys
import time
import random
import signal
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

# ==========================================
# 1. ハイパーパラメータ ＆ グローバル設定
# ==========================================
TRAIN_MODE = False            # 学習時は True、再生時は False
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
        
        # リングバッファを効率化するため2倍の長さを確保
        self.buffer_size = seq_len * 2
        self.buffer_shape = (n_envs, self.buffer_size, obs_dim)
        self.buffer = torch.zeros(self.buffer_shape, device=device)
        self.ptr = 0

    def reset(self, env_idx=None):
        """履歴の初期化"""
        if env_idx is None:
            # 全環境を一括クリア
            self.buffer.zero_()
            self.ptr = 0
        else:
            # 特定の環境のみクリア
            self.buffer[env_idx].zero_()

    def update(self, obs_data, env_idx=None):
        """最新の観測値でバッファを更新（1行1文展開）"""
        # テンソル変換
        t_obs = torch.as_tensor(obs_data, dtype=torch.float32, device=self.device)
        
        if env_idx is None:
            # ベクトル環境全体を一括更新
            # 💡 修正: ここで self.obs_dim (55) を用いて再形成
            t_obs = t_obs.reshape(self.n_envs, self.obs_dim)
            # 現在の書き込み位置
            write_idx = self.ptr
            self.buffer[:, write_idx] = t_obs
            # 連続性を保つためのミラー書き込み
            mirror_idx = write_idx + self.seq_len
            self.buffer[:, mirror_idx] = t_obs
            # ポインタの進捗
            self.ptr = (self.ptr + 1) % self.seq_len
        else:
            # 特定の単一環境のみ更新
            t_obs = t_obs.reshape(self.obs_dim)
            write_idx = self.ptr
            self.buffer[env_idx, write_idx] = t_obs
            mirror_idx = write_idx + self.seq_len
            self.buffer[env_idx, mirror_idx] = t_obs

    def get(self):
        """直近 seq_len 分の連続したシーケンスを抽出"""
        start_pos = self.ptr
        end_pos = self.ptr + self.seq_len
        sequence = self.buffer[:, start_pos:end_pos]
        # メモリ配置を連続化
        contiguous_seq = sequence.contiguous()
        return contiguous_seq

def make_env():
    """環境インスタンスの生成"""
    render_mode = None
    if USE_VIEWER:
        if not TRAIN_MODE:
            render_mode = "human"
    
    # 環境クラスの初期化
    env_inst = TeamCosEnv(mode=MODE, render_mode=render_mode)
    return env_inst

# ==========================================
# 3. メイン実行ロジック (Run)
# ==========================================
def run():
    # 実行デバイスの自動選択
    cuda_available = torch.cuda.is_available()
    device_type = "cpu"
    if cuda_available:
        device_type = "cuda"
    device = torch.device(device_type)
    
    # 決定論的実行のためのシード固定
    seed_value = 42
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    
    # 観測次元の定義 (環境 v2.20 に同期)
    # 💡 修正: 53 -> 55
    obs_dim = 55
    
    # アクション次元の決定
    action_dim = 8
    if MODE == "initial":
        action_dim = 4
        
    # エージェントモデルの構築
    agent = AgentV2(obs_dim, action_dim, HIDDEN_DIM, SEQ_LEN)
    agent = agent.to(device)
    
    # オプティマイザ
    optimizer = optim.Adam(agent.parameters(), lr=LR_START, eps=1e-5)
    
    # 既存の重みがある場合はロード
    if os.path.exists(SAVE_PATH):
        try:
            state_dict = torch.load(SAVE_PATH, map_location=device)
            agent.load_state_dict(state_dict)
            print(f"✅ Pre-trained model loaded: {SAVE_PATH}")
        except Exception as e:
            print(f"⚠️ Load failed: {e}. Starting from scratch.")

    # ------------------------------------------
    # [A] 再生モード (PLAYBACK)
    # ------------------------------------------
    if not TRAIN_MODE:
        env = make_env()
        # 💡 修正: 53 -> 55
        hist = ObsHistory(1, SEQ_LEN, obs_dim, device)
        agent.eval()
        
        # 初期状態の設定
        obs_raw, _ = env.reset()
        hist.update(obs_raw)
        
        # Viewer 起動の確実化
        if USE_VIEWER:
            env.render()
            time.sleep(0.5)
        
        fps_limit = 40
        duration = 1.0 / fps_limit
        print(f"🎮 Playback Mode Active. Frequency: {fps_limit}Hz.")
        
        try:
            while True:
                if USE_VIEWER:
                    if env.viewer is None:
                        break
                    if not env.viewer.is_running():
                        break
                
                t_loop_start = time.perf_counter()
                
                with torch.no_grad():
                    current_hist_data = hist.get()
                    act, _, _, _ = agent.get_action_and_value(current_hist_data)
                
                # 物理ステップ
                act_numpy = act.cpu().numpy().flatten()
                next_obs, rew, term, trunc, info = env.step(act_numpy)
                
                hist.update(next_obs)
                
                if term or trunc:
                    obs_reset, _ = env.reset()
                    hist.reset()
                    hist.update(obs_reset)
                    t_loop_start = time.perf_counter()
                
                t_now_pb = time.perf_counter()
                t_elapsed_pb = t_now_pb - t_loop_start
                wait_val_pb = duration - t_elapsed_pb
                if wait_val_pb > 0:
                    time.sleep(wait_val_pb)
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
        print(f"🚀 Training Mode Started: {MODE} | Device: {device}")
        
        # 並列環境の生成
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(NUM_ENVS)])
        # 💡 修正: 53 -> 55
        hist = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim, device)
        
        # ログ構成
        timestamp = datetime.now().strftime("%m%d_%H%M")
        log_dir = f"runs/HNS_{MODE}_{timestamp}"
        writer = SummaryWriter(log_dir)
        
        # ロールアウトバッファ
        # 💡 修正: 53 -> 55
        obs_buf = torch.zeros((EPISODE_LIMIT, NUM_ENVS, SEQ_LEN, obs_dim), device=device)
        act_buf = torch.zeros((EPISODE_LIMIT, NUM_ENVS, action_dim), device=device)
        prob_buf = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        rew_buf = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        done_buf = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        val_buf = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        
        # 状態変数の初期化
        global_step = 0
        update_count = 0
        best_avg_reward = -float('inf')
        start_time = time.time()
        
        # 最初の観測の取得
        next_obs, _ = envs.reset()
        hist.update(next_obs)
        
        # 学習モードでの Viewer 初期化
        if USE_VIEWER:
            envs.envs[0].render()
            time.sleep(0.5)
        
        try:
            while global_step < TOTAL_STEPS:
                # --- [1] データ収集 (Rollout) ---
                agent.eval()
                for step in range(EPISODE_LIMIT):
                    # 総ステップ更新
                    global_step = global_step + NUM_ENVS
                    
                    # 現在のスタックを記録
                    current_seq_rollout = hist.get()
                    obs_buf[step] = current_seq_rollout
                    
                    if USE_VIEWER:
                        envs.envs[0].render()
                    
                    with torch.no_grad():
                        # 推論
                        action, logprob, _, value = agent.get_action_and_value(current_seq_rollout)
                        val_buf[step] = value.flatten()
                    
                    act_buf[step] = action
                    prob_buf[step] = logprob
                    
                    # 環境ステップ
                    next_obs_vec, reward, done, trunc, _ = envs.step(action.cpu().numpy())
                    
                    rew_buf[step] = torch.as_tensor(reward, device=device)
                    done_buf[step] = torch.as_tensor(done, device=device)
                    
                    # 履歴を更新
                    hist.update(next_obs_vec)
                    
                    # 完了した環境のバッファ・リセット
                    for env_i in range(NUM_ENVS):
                        if done[env_i]:
                            hist.reset(env_idx=env_i)
                            hist.update(next_obs_vec[env_i], env_idx=env_i)
                        elif trunc[env_i]:
                            hist.reset(env_idx=env_i)
                            hist.update(next_obs_vec[env_i], env_idx=env_i)

                # --- [2] 利益推定 (GAE Calculation) ---
                agent.eval()
                with torch.no_grad():
                    # 未来の状態価値
                    next_hist_gae = hist.get()
                    v_next_raw = agent.get_value(next_hist_gae)
                    v_next = v_next_raw.reshape(1, -1)
                    
                    adv_buf = torch.zeros_like(rew_buf, device=device)
                    last_gae_lam = 0
                    
                    for t_idx in reversed(range(EPISODE_LIMIT)):
                        if t_idx == EPISODE_LIMIT - 1:
                            non_term = 1.0 - done_buf[t_idx]
                            v_target_gae = v_next
                        else:
                            non_term = 1.0 - done_buf[t_idx]
                            v_target_gae = val_buf[t_idx + 1]
                        
                        # TDエラー
                        delta_gae = rew_buf[t_idx] + GAMMA * v_target_gae * non_term - val_buf[t_idx]
                        # 再帰計算
                        last_gae_lam = delta_gae + GAMMA * GAE_LAMBDA * non_term * last_gae_lam
                        adv_buf[t_idx] = last_gae_lam
                    
                    # 目標収益
                    ret_buf = adv_buf + val_buf

                # --- [3] モデルの最適化 (PPO Optimization) ---
                agent.train()
                # データの平坦化
                # 💡 修正: 53 -> obs_dim (55)
                f_obs = obs_buf.reshape(-1, SEQ_LEN, obs_dim)
                f_act = act_buf.reshape(-1, action_dim)
                f_prob = prob_buf.reshape(-1)
                f_adv = adv_buf.reshape(-1)
                f_ret = ret_buf.reshape(-1)
                f_val = val_buf.reshape(-1)
                
                dataset_size = f_obs.shape[0]
                indices_set = np.arange(dataset_size)
                
                # 最適化統計用
                clip_fractions = []
                final_approx_kl = torch.tensor(0.0, device=device)
                
                for epoch_idx in range(UPDATE_EPOCHS):
                    np.random.shuffle(indices_set)
                    for start_idx in range(0, dataset_size, BATCH_SIZE):
                        end_idx = start_idx + BATCH_SIZE
                        m_idx = indices_set[start_idx:end_idx]
                        
                        # 方策評価
                        _, n_prob, entropy, n_val = agent.get_action_and_value(f_obs[m_idx], f_act[m_idx])
                        
                        # 確率比
                        log_ratio_step = n_prob - f_prob[m_idx]
                        ratio_step = log_ratio_step.exp()
                        
                        # 診断指標
                        with torch.no_grad():
                            final_approx_kl = ((ratio_step - 1.0) - log_ratio_step).mean()
                            diff_ratio_step = ratio_step - 1.0
                            abs_diff_step = diff_ratio_step.abs()
                            clip_bool_step = abs_diff_step > CLIP_COEF
                            clip_rate_step = clip_bool_step.float().mean().item()
                            clip_fractions.append(clip_rate_step)
                        
                        # アドバンテージ正規化
                        mb_adv = f_adv[m_idx]
                        mb_adv_mean = mb_adv.mean()
                        mb_adv_std = mb_adv.std()
                        mb_adv = (mb_adv - mb_adv_mean) / (mb_adv_std + 1e-8)
                        
                        # 方策損失
                        pg_l1_step = -mb_adv * ratio_step
                        ratio_clamped_step = torch.clamp(ratio_step, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                        pg_l2_step = -mb_adv * ratio_clamped_step
                        pg_loss_step = torch.max(pg_l1_step, pg_l2_step).mean()
                        
                        # 価値損失
                        v_pred_step = n_val.flatten()
                        v_target_mb_step = f_ret[m_idx]
                        v_diff_step = v_pred_step - v_target_mb_step
                        v_loss_step = 0.5 * (v_diff_step ** 2).mean()
                        
                        # アニーリング
                        progress_val = 1.0 - (global_step / TOTAL_STEPS)
                        # 学習率
                        current_lr_val = LR_END + (LR_START - LR_END) * progress_val
                        for pg_grp in optimizer.param_groups:
                            pg_grp["lr"] = current_lr_val
                        # エントロピー係数
                        current_ent_val = ENT_COEF_END + (ENT_COEF_START - ENT_COEF_END) * progress_val
                        
                        # 最終損失
                        e_loss_step = entropy.mean()
                        total_loss_step = pg_loss_step - current_ent_val * e_loss_step + v_loss_step * VF_COEF
                        
                        # 勾配更新
                        optimizer.zero_grad()
                        total_loss_step.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                    
                    if final_approx_kl > TARGET_KL:
                        break
                
                update_count = update_count + 1

                # --- [4] 評価フェーズ (Evaluation) ---
                if update_count % EVAL_INTERVAL == 0:
                    agent.eval()
                    eval_rewards = []
                    print(f"🔍 Starting Evaluation...")
                    for _ in range(EVAL_EPISODES):
                        e_obs_raw, _ = envs.reset()
                        # 💡 修正: 53 -> 55
                        e_hist_eval = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim, device)
                        e_hist_eval.update(e_obs_raw)
                        ep_rew_accum = np.zeros(NUM_ENVS)
                        for _ in range(EPISODE_LIMIT):
                            with torch.no_grad():
                                e_act_eval, _, _, _ = agent.get_action_and_value(e_hist_eval.get())
                            e_next_raw, e_r_raw, e_d_raw, e_t_raw, _ = envs.step(e_act_eval.cpu().numpy())
                            ep_rew_accum = ep_rew_accum + e_r_raw
                            e_hist_eval.update(e_next_raw)
                            if any(e_d_raw):
                                break
                            if any(e_t_raw):
                                break
                        eval_rewards.append(np.mean(ep_rew_accum))
                    
                    avg_eval_reward_val = np.mean(eval_rewards)
                    writer.add_scalar("eval/avg_reward", avg_eval_reward_val, global_step)
                    print(f"📊 Eval Reward: {avg_eval_reward_val:.4f}")
                    
                    if avg_eval_reward_val > best_avg_reward:
                        best_avg_reward = avg_eval_reward_val
                        torch.save(agent.state_dict(), BEST_SAVE_PATH)
                        print(f"⭐ New Best Model Saved!")

                # --- [5] ログ ＆ 保存 ---
                # 説明分散
                y_pred_final = f_val.cpu().numpy()
                y_true_final = f_ret.cpu().numpy()
                var_y_final = np.var(y_true_final)
                explained_var_val = np.nan
                if var_y_final > 0:
                    err_var_final = np.var(y_true_final - y_pred_final)
                    explained_var_val = 1.0 - err_var_final / var_y_final
                
                cur_t_final = time.time()
                elapsed_final = cur_t_final - start_time
                sps_val_final = int(global_step / elapsed_final)
                mean_rollout_reward_val = rew_buf.mean().item()
                
                # コンソール
                msg_final = f"Step: {global_step:8d} | SPS: {sps_val_final:4d} | Rew: {mean_rollout_reward_val:7.4f} | KL: {final_approx_kl:6.4f}"
                print(msg_final)
                
                # TensorBoard
                writer.add_scalar("params/learning_rate", current_lr_val, global_step)
                writer.add_scalar("params/entropy_coef", current_ent_val, global_step)
                writer.add_scalar("losses/value_loss", v_loss_step.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss_step.item(), global_step)
                writer.add_scalar("losses/entropy", e_loss_step.item(), global_step)
                writer.add_scalar("losses/approx_kl", final_approx_kl.item(), global_step)
                writer.add_scalar("losses/clip_fraction", np.mean(clip_fractions), global_step)
                writer.add_scalar("losses/explained_variance", explained_var_val, global_step)
                writer.add_scalar("charts/SPS", sps_val_final, global_step)
                writer.add_scalar("charts/avg_reward", mean_rollout_reward_val, global_step)
                
                torch.save(agent.state_dict(), SAVE_PATH)

        except KeyboardInterrupt:
            print("✋ Training interrupted.")
        except Exception:
            print("❌ Unexpected error occurred:")
            traceback.print_exc()
        finally:
            current_locals_final = locals()
            if 'envs' in current_locals_final:
                envs.close()
            if 'writer' in current_locals_final:
                writer.close()
            print("💾 Training session closed.")

if __name__ == "__main__":
    run()