# main26_train_final.py v2.34
# 演習第26回：【論理整合性・極限展開・完全復旧 ＆ 55次元カテゴリカル観測統合 ＆ 決定版】
# 
# 修正内容:
# 1. 論理の完全復元:
#    - PPOの安定性を支える GAE 計算、KL早期停止、アドバンテージ正規化の各ステップを中間変数を介して詳細化。
#    - explained_variance 等の統計計算において、数値的安定性を確保するための例外処理を明示的に記述。
# 2. 55次元カテゴリカル観測 (interaction_state) の整合性確保:
#    - 環境側の v2.71 (0:なし, 1:Lock, 2:Me, 3:Others) に完全に同期した入力次元を設定。
# 3. Windows/非ARM環境最適化の完全定着:
#    - 並列計算ライブラリのスレッド制限を冒頭に配置し、実行速度を担保。
# 4. NDR (No-Detected Ratio) の厳密な算出:
#    - チーム全員が非検知の状態をロールアウトごとに集計し、Optunaの目的関数として利用可能な精度で出力。
# 5. 1行1命令（1-line-1-command）の徹底遵守:
#    - 演算、条件分岐、代入、ロギングをすべて独立した行で記述し、ロジックの省略を物理的に不可能にする。

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

# ==========================================
# 0. 環境変数設定 (Windows/並列実行最適化)
# ==========================================
proc_name = platform.processor()
if proc_name != 'arm':
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
USE_VIEWER_RUNTIME = USE_VIEWER

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

# WANDB インポート
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
            t_shape = (self.n_envs, self.obs_dim)
            obs_reshaped = obs_tensor.reshape(t_shape)
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
        start_idx = self.ptr
        end_idx = self.ptr + self.seq_len
        sequence_data = self.buffer[:, start_idx:end_idx]
        contiguous_seq = sequence_data.contiguous()
        return contiguous_seq

def make_env():
    render_mode_val = None
    if USE_VIEWER_RUNTIME:
        if not TRAIN_MODE:
            render_mode_val = "human"
    env_inst = TeamCosEnv(mode=MODE, render_mode=render_mode_val)
    return env_inst

# ==========================================
# 3. メイン実行ロジック (Run)
# ==========================================
def run():
    # ハードウェア設定
    c_avail = torch.cuda.is_available()
    d_type_name = "cpu"
    if c_avail:
        d_type_name = "cuda"
    device = torch.device(d_type_name)

    # viewer は事前判定せず、実際の render 失敗時に自動無効化する
    global USE_VIEWER_RUNTIME
    USE_VIEWER_RUNTIME = USE_VIEWER
    viewer_warned = False

    def _safe_render(target_env):
        nonlocal viewer_warned
        global USE_VIEWER_RUNTIME
        if not USE_VIEWER_RUNTIME:
            return False
        try:
            target_env.render()
            return True
        except RuntimeError as render_err:
            if not viewer_warned:
                print(f"⚠️ Viewer disabled: {render_err}")
                viewer_warned = True
            USE_VIEWER_RUNTIME = False
            return False
    
    # 乱数シード
    s_val = 42
    random.seed(s_val)
    np.random.seed(s_val)
    torch.manual_seed(s_val)
    torch.backends.cudnn.deterministic = True
    
    # ネットワーク次元設定
    obs_dim_actual = 55
    act_dim_actual = 8
    if MODE == "initial":
        act_dim_actual = 4
        
    # エージェントとオプティマイザ
    agent = AgentV2(obs_dim_actual, act_dim_actual, HIDDEN_DIM, SEQ_LEN)
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR_START, eps=1e-5)
    
    # モデルロード
    if os.path.exists(SAVE_PATH):
        try:
            ckpt = torch.load(SAVE_PATH, map_location=device)
            agent.load_state_dict(ckpt)
            print(f"✅ Loaded checkpoint: {SAVE_PATH}")
        except Exception as e_load:
            print(f"⚠️ Load failed: {e_load}")

    # ------------------------------------------
    # [A] 再生モード (PLAYBACK)
    # ------------------------------------------
    if not TRAIN_MODE:
        env_p = make_env()
        hist_p = ObsHistory(1, SEQ_LEN, obs_dim_actual, device)
        agent.eval()
        
        obs_raw_p, _ = env_p.reset()
        hist_p.update(obs_raw_p)
        
        if USE_VIEWER_RUNTIME:
            rendered = _safe_render(env_p)
            if rendered:
                time.sleep(0.5)
            
        fps_p = 40
        dur_p = 1.0 / fps_p
        print(f"🎮 Playback Mode Active.")
        
        try:
            while True:
                if USE_VIEWER_RUNTIME:
                    if env_p.viewer is None:
                        break
                    if not env_p.viewer.is_running():
                        break
                        
                tick_p = time.perf_counter()
                
                with torch.no_grad():
                    seq_p = hist_p.get()
                    act_p, _, _, _ = agent.get_action_and_value(seq_p)
                    
                act_np_p = act_p.cpu().numpy().flatten()
                obs_n_p, r_p, term_p, trunc_p, info_p = env_p.step(act_np_p)
                
                hist_p.update(obs_n_p)
                
                if term_p or trunc_p:
                    obs_re_p, _ = env_p.reset()
                    hist_p.reset()
                    hist_p.update(obs_re_p)
                    tick_p = time.perf_counter()
                    
                t_elapsed_p = time.perf_counter() - tick_p
                t_wait_p = dur_p - t_elapsed_p
                if t_wait_p > 0:
                    time.sleep(t_wait_p)
                else:
                    time.sleep(0.001)
        finally:
            env_p.close()
            print("🏁 Playback terminated.")
            sys.exit(0)

    # ------------------------------------------
    # [B] 学習モード (TRAINING)
    # ------------------------------------------
    else:
        print(f"🚀 Training Mode Started: {MODE}")
        
        # 1. W&B 初期化
        if TRACK_WANDB:
            now_dt = datetime.now().strftime("%m%d_%H%M")
            run_tag = f"ppo_transformer_{MODE}_{now_dt}"
            wandb.init(
                project="HNS_V26_TeamCos",
                name=run_tag,
                config={
                    "mode": MODE,
                    "lr_start": LR_START,
                    "batch_size": BATCH_SIZE,
                    "total_steps": TOTAL_STEPS,
                    "num_envs": NUM_ENVS,
                    "seq_len": SEQ_LEN,
                    "hidden_dim": HIDDEN_DIM,
                    "gamma": GAMMA,
                    "gae_lambda": GAE_LAMBDA,
                    "clip_coef": CLIP_COEF,
                    "update_epochs": UPDATE_EPOCHS
                }
            )

        # 2. 環境 ＆ 履歴バッファ
        envs_train = gym.vector.SyncVectorEnv([make_env for _ in range(NUM_ENVS)])
        hist_train = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim_actual, device)
        log_time = datetime.now().strftime("%m%d_%H%M")
        tb_writer = SummaryWriter(f"runs/HNS_{MODE}_{log_time}")
        
        # 3. テンソルバッファの極限展開
        obs_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS, SEQ_LEN, obs_dim_actual), device=device)
        act_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS, act_dim_actual), device=device)
        prob_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        rew_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        done_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        val_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        # 重要：非検知割合計測用
        find_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        
        global_step_count = 0
        update_iter = 0
        best_reward_eval = -float('inf')
        t_start_training = time.time()
        
        # 初回リセット
        obs_v_start, _ = envs_train.reset()
        hist_train.update(obs_v_start)
        
        if USE_VIEWER_RUNTIME:
            rendered = _safe_render(envs_train.envs[0])
            if rendered:
                time.sleep(0.5)
        
        try:
            while global_step_count < TOTAL_STEPS:
                
                # --- 4. データ収集 (Rollout) ---
                agent.eval()
                for rollout_step in range(EPISODE_LIMIT):
                    global_step_count = global_step_count + NUM_ENVS
                    
                    rollout_seq = hist_train.get()
                    obs_buffer[rollout_step] = rollout_seq
                    
                    if USE_VIEWER_RUNTIME:
                        _safe_render(envs_train.envs[0])
                        
                    with torch.no_grad():
                        action_roll, logprob_roll, _, value_roll = agent.get_action_and_value(rollout_seq)
                        val_buffer[rollout_step] = value_roll.flatten()
                        
                    act_buffer[rollout_step] = action_roll
                    prob_buffer[rollout_step] = logprob_roll
                    
                    obs_v_next, rew_v_step, done_v_step, trunc_v_step, info_v_step = envs_train.step(action_roll.cpu().numpy())
                    
                    rew_buffer[rollout_step] = torch.as_tensor(rew_v_step, device=device)
                    done_buffer[rollout_step] = torch.as_tensor(done_v_step, device=device)
                    
                    # NDRデータの抽出と保存
                    is_detected_flags = info_v_step.get("is_detected", np.zeros(NUM_ENVS))
                    find_buffer[rollout_step] = torch.as_tensor(is_detected_flags, dtype=torch.float32, device=device)
                    
                    hist_train.update(obs_v_next)
                    
                    for env_idx_res in range(NUM_ENVS):
                        if done_v_step[env_idx_res] or trunc_v_step[env_idx_res]:
                            hist_train.reset(env_idx=env_idx_res)
                            hist_train.update(obs_v_next[env_idx_res], env_idx=env_idx_res)

                # --- 5. アドバンテージ計算 (GAE) ---
                agent.eval()
                with torch.no_grad():
                    final_seq_gae = hist_train.get()
                    v_next_raw_gae = agent.get_value(final_seq_gae)
                    v_next_val_gae = v_next_raw_gae.reshape(1, -1)
                    
                    adv_buffer = torch.zeros_like(rew_buffer, device=device)
                    last_gae_pointer = 0
                    
                    for t_idx_gae in reversed(range(EPISODE_LIMIT)):
                        if t_idx_gae == EPISODE_LIMIT - 1:
                            non_term_gae = 1.0 - done_buffer[t_idx_gae]
                            v_target_gae = v_next_val_gae
                        else:
                            non_term_gae = 1.0 - done_buffer[t_idx_gae]
                            v_target_gae = val_buffer[t_idx_gae + 1]
                            
                        delta_gae = rew_buffer[t_idx_gae] + GAMMA * v_target_gae * non_term_gae - val_buffer[t_idx_gae]
                        last_gae_pointer = delta_gae + GAMMA * GAE_LAMBDA * non_term_gae * last_gae_pointer
                        adv_buffer[t_idx_gae] = last_gae_pointer
                        
                    ret_buffer = adv_buffer + val_buffer

                # --- 6. PPO 最適化 (Optimization) ---
                agent.train()
                # 平坦化
                f_obs = obs_buffer.reshape(-1, SEQ_LEN, obs_dim_actual)
                f_act = act_buffer.reshape(-1, act_dim_actual)
                f_prob = prob_buffer.reshape(-1)
                f_adv = adv_buffer.reshape(-1)
                f_ret = ret_buffer.reshape(-1)
                f_val = val_buffer.reshape(-1)
                
                size_ds = f_obs.shape[0]
                idx_ppo = np.arange(size_ds)
                clip_rates = []
                kl_div_final = torch.tensor(0.0, device=device)
                
                for epoch_ppo in range(UPDATE_EPOCHS):
                    np.random.shuffle(idx_ppo)
                    for start_ppo in range(0, size_ds, BATCH_SIZE):
                        end_ppo = start_ppo + BATCH_SIZE
                        m_idx_ppo = idx_ppo[start_ppo:end_ppo]
                        
                        _, n_logp, n_ent, n_val_ppo = agent.get_action_and_value(f_obs[m_idx_ppo], f_act[m_idx_ppo])
                        
                        l_ratio_ppo = n_logp - f_prob[m_idx_ppo]
                        ratio_ppo = l_ratio_ppo.exp()
                        
                        with torch.no_grad():
                            # KL ダイバージェンス
                            kl_div_final = ((ratio_ppo - 1.0) - l_ratio_ppo).mean()
                            c_rate_ppo = (ratio_ppo - 1.0).abs().gt(CLIP_COEF).float().mean().item()
                            clip_rates.append(c_rate_ppo)
                        
                        # アドバンテージ正規化
                        mb_adv_ppo = f_adv[m_idx_ppo]
                        mb_adv_mean = mb_adv_ppo.mean()
                        mb_adv_std = mb_adv_ppo.std()
                        mb_adv_norm = (mb_adv_ppo - mb_adv_mean) / (mb_adv_std + 1e-8)
                        
                        # Policy Loss
                        pg_l_un = -mb_adv_norm * ratio_ppo
                        pg_r_cl = torch.clamp(ratio_ppo, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                        pg_l_cl = -mb_adv_norm * pg_r_cl
                        pg_loss_ppo = torch.max(pg_l_un, pg_l_cl).mean()
                        
                        # Value Loss
                        v_err_ppo = n_val_ppo.flatten() - f_ret[m_idx_ppo]
                        v_loss_ppo = 0.5 * (v_err_ppo ** 2).mean()
                        
                        # アニーリング
                        prog_ratio = 1.0 - (global_step_count / TOTAL_STEPS)
                        c_lr_ppo = LR_END + (LR_START - LR_END) * prog_ratio
                        for p_grp in optimizer.param_groups:
                            p_grp["lr"] = c_lr_ppo
                            
                        c_ent_ppo = ENT_COEF_END + (ENT_COEF_START - ENT_COEF_END) * prog_ratio
                        ent_loss_ppo = n_ent.mean()
                        
                        # Total Loss
                        l_total = pg_loss_ppo - c_ent_ppo * ent_loss_ppo + v_loss_ppo * VF_COEF
                        
                        optimizer.zero_grad()
                        l_total.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                        
                    if kl_div_final > TARGET_KL:
                        break
                
                update_iter = update_iter + 1

                # --- 7. 評価フェーズ (Evaluation) ---
                if update_iter % EVAL_INTERVAL == 0:
                    agent.eval()
                    eval_ep_accum_list = []
                    
                    for _ in range(EVAL_EPISODES):
                        obs_ev_raw, _ = envs_train.reset()
                        hist_ev = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim_actual, device)
                        hist_ev.update(obs_ev_raw)
                        rew_ev_sum = np.zeros(NUM_ENVS)
                        
                        for _ in range(EPISODE_LIMIT):
                            with torch.no_grad():
                                seq_ev = hist_ev.get()
                                act_ev, _, _, _ = agent.get_action_and_value(seq_ev)
                                
                            obs_ev_next, r_ev, d_ev, t_ev, _ = envs_train.step(act_ev.cpu().numpy())
                            rew_ev_sum = rew_ev_sum + r_ev
                            hist_ev.update(obs_ev_next)
                            
                            if any(d_ev) or any(t_ev):
                                break
                                
                        eval_ep_accum_list.append(np.mean(rew_ev_sum))
                        
                    avg_reward_eval_final = np.mean(eval_ep_accum_list)
                    tb_writer.add_scalar("eval/avg_reward", avg_reward_eval_final, global_step_count)
                    
                    if TRACK_WANDB:
                        wandb.log({"eval/avg_reward": avg_reward_eval_final}, step=global_step_count)
                        
                    if avg_reward_eval_final > best_reward_eval:
                        best_reward_eval = avg_reward_eval_final
                        torch.save(agent.state_dict(), BEST_SAVE_PATH)

                # --- 8. 統計ロギング (Logging) ---
                y_pred_np = f_val.cpu().numpy()
                y_true_np = f_ret.cpu().numpy()
                var_y_stats = np.var(y_true_np)
                if var_y_stats > 1e-6:
                    residual_var = np.var(y_true_np - y_pred_np)
                    exp_var_val = 1.0 - residual_var / var_y_stats
                else:
                    exp_var_val = 0.0
                    
                dur_train = time.time() - t_start_training
                sps_train = int(global_step_count / (dur_train + 1e-8))
                m_rollout_r = rew_buffer.mean().item()
                
                # NDR (No-Detected Ratio) 厳密計算
                found_mean_step = find_buffer.mean().item()
                ndr_val_final = 1.0 - found_mean_step
                
                print(f"Step: {global_step_count:8d} | SPS: {sps_train:4d} | Rew: {m_rollout_r:7.4f} | NDR: {ndr_val_final:6.2%} | KL: {kl_div_final:6.4f}")
                
                # TB 出力
                tb_writer.add_scalar("params/learning_rate", c_lr_ppo, global_step_count)
                tb_writer.add_scalar("params/entropy_coef", c_ent_ppo, global_step_count)
                tb_writer.add_scalar("losses/value_loss", v_loss_ppo.item(), global_step_count)
                tb_writer.add_scalar("losses/policy_loss", pg_loss_ppo.item(), global_step_count)
                tb_writer.add_scalar("losses/entropy", ent_loss_ppo.item(), global_step_count)
                tb_writer.add_scalar("losses/approx_kl", kl_div_final.item(), global_step_count)
                tb_writer.add_scalar("losses/clip_fraction", np.mean(clip_rates), global_step_count)
                tb_writer.add_scalar("losses/explained_variance", exp_var_val, global_step_count)
                tb_writer.add_scalar("charts/SPS", sps_train, global_step_count)
                tb_writer.add_scalar("charts/avg_reward", m_rollout_r, global_step_count)
                tb_writer.add_scalar("charts/no_detected_ratio", ndr_val_final, global_step_count)
                
                # WANDB 出力
                if TRACK_WANDB:
                    wandb.log({
                        "params/learning_rate": c_lr_ppo,
                        "params/entropy_coef": c_ent_ppo,
                        "losses/value_loss": v_loss_ppo.item(),
                        "losses/policy_loss": pg_loss_ppo.item(),
                        "losses/entropy": ent_loss_ppo.item(),
                        "losses/approx_kl": kl_div_final.item(),
                        "losses/clip_fraction": np.mean(clip_rates),
                        "losses/explained_variance": exp_var_val,
                        "charts/SPS": sps_train,
                        "charts/avg_reward": m_rollout_r,
                        "charts/no_detected_ratio": ndr_val_final
                    }, step=global_step_count)
                
                torch.save(agent.state_dict(), SAVE_PATH)

        except KeyboardInterrupt:
            print("✋ Training interrupted.")
        except Exception:
            traceback.print_exc()
        finally:
            ctx_locals = locals()
            if 'envs_train' in ctx_locals:
                envs_train.close()
            if 'tb_writer' in ctx_locals:
                tb_writer.close()
            if TRACK_WANDB:
                wandb.finish()
            print("💾 Training session closed.")

if __name__ == "__main__":
    run()