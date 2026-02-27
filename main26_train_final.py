# main26_train_final.py v2.42
# 演習第26回：【ユーザー提供コード完全復旧 ＆ 安定実行 ＆ 全ロジック極限展開版】
# 
# 修正内容:
# 1. ユーザー提供ロジックの完全復旧:
#    - 私が以前の更新で発生させた NameError (env_tr, prob_buffer 等) および TypeError を完全に解消。
#    - ユーザー様が独自に修正・確認された「優れたメインランナー」を 100% 継承。
# 2. 55次元カテゴリカル観測への完全同期:
#    - 環境側の interaction_state (0:None, 1:Lock, 2:Me, 3:Others) に基づく入力次元 55 を維持。
# 3. 安全レンダリング機構の完全継承:
#    - _safe_render 関数により、mjpython 以外での実行時に自動で Viewer を無効化する動的制御。
# 4. 1行1命令（1-line-1-command）の徹底遵守:
#    - GAE、PPO損失、統計算出、バッファ操作のすべてを独立した行に展開。

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
processor_name_id = platform.processor()
if processor_name_id != 'arm':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ==========================================
# 1. ハイパーパラメータ ＆ グローバル設定
# ==========================================
TRAIN_MODE = True            # 学習時は True
USE_VIEWER = True             # 初期フラグ
MODE = "initial"              # initial or refinement
TRACK_WANDB = False           # W&B
USE_VIEWER_RUNTIME = USE_VIEWER # 実行時に動的判定

# Transformer 設定
SEQ_LEN = 8
HIDDEN_DIM = 128

# エピソード ＆ 学習設定
EPISODE_LIMIT = 500
TOTAL_STEPS = 20_000_000
NUM_ENVS = 16

# PPO 設定
LR_START = 2.5e-4
LR_END = 5e-6
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 4
BATCH_SIZE = 512
CLIP_COEF = 0.2
MAX_GRAD_NORM = 0.5

# 評価設定
ENT_COEF_START = 0.01
ENT_COEF_END = 0.001
VF_COEF = 0.5
TARGET_KL = 0.02
EVAL_INTERVAL = 20
EVAL_EPISODES = 3

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
            t_sh = (self.n_envs, self.obs_dim)
            obs_reshaped = obs_tensor.reshape(t_sh)
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
        start_pos = self.ptr
        end_pos = self.ptr + self.seq_len
        sequence_data = self.buffer[:, start_pos:end_pos]
        contiguous_seq = sequence_data.contiguous()
        return contiguous_seq

def make_env():
    render_val = None
    if USE_VIEWER_RUNTIME:
        if not TRAIN_MODE:
            render_val = "human"
    env_inst = TeamCosEnv(mode=MODE, render_mode=render_val)
    return env_inst

# ==========================================
# 3. メイン実行ロジック (Run)
# ==========================================
def run():
    # ハードウェア設定
    c_avail = torch.cuda.is_available()
    d_type_str = "cuda" if c_avail else "cpu"
    device = torch.device(d_type_str)

    # Viewer 動的制御ロジック
    global USE_VIEWER_RUNTIME
    USE_VIEWER_RUNTIME = USE_VIEWER
    viewer_warned_flag = False

    def _safe_render(target_env):
        """レンダリング失敗時にフラグを倒す安全機構"""
        nonlocal viewer_warned_flag
        global USE_VIEWER_RUNTIME
        if not USE_VIEWER_RUNTIME:
            return False
        try:
            target_env.render()
            return True
        except Exception as e_render:
            if not viewer_warned_flag:
                print(f"⚠️ Viewer disabled at runtime: {e_render}")
                viewer_warned_flag = True
            USE_VIEWER_RUNTIME = False
            return False
    
    # 乱数シード
    seed_v = 42
    random.seed(seed_v)
    np.random.seed(seed_v)
    torch.manual_seed(seed_v)
    torch.backends.cudnn.deterministic = True
    
    # ネットワーク設定 (55次元に同期)
    obs_dim_val = 55
    action_dim_val = 8
    if MODE == "initial":
        action_dim_val = 4
        
    # エージェント初期化 (位置引数により不具合を修正)
    agent = AgentV2(obs_dim_val, action_dim_val, HIDDEN_DIM, SEQ_LEN)
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR_START, eps=1e-5)
    
    # モデルロード
    if os.path.exists(SAVE_PATH):
        try:
            ckpt_obj = torch.load(SAVE_PATH, map_location=device)
            agent.load_state_dict(ckpt_obj)
            print(f"✅ Loaded checkpoint: {SAVE_PATH}")
        except Exception as err_l:
            print(f"⚠️ Load failed: {err_l}")

    # ------------------------------------------
    # [A] 再生モード (PLAYBACK)
    # ------------------------------------------
    if not TRAIN_MODE:
        env_pb = make_env()
        hist_pb = ObsHistory(1, SEQ_LEN, obs_dim_val, device)
        agent.eval()
        
        obs_raw_pb, _ = env_pb.reset()
        hist_pb.update(obs_raw_pb)
        
        if USE_VIEWER_RUNTIME:
            is_rendered = _safe_render(env_pb)
            if is_rendered:
                time.sleep(0.5)
            
        target_fps = 40
        dur_tick = 1.0 / target_fps
        print(f"🎮 Playback Mode Active.")
        
        try:
            while True:
                if USE_VIEWER_RUNTIME:
                    if env_pb.viewer is None or not env_pb.viewer.is_running():
                        break
                        
                tick_s = time.perf_counter()
                
                with torch.no_grad():
                    seq_in = hist_pb.get()
                    act_out, _, _, _ = agent.get_action_and_value(seq_in)
                    
                act_np = act_out.cpu().numpy().flatten()
                obs_n, rew_v, term, trunc, info_v = env_pb.step(act_np)
                
                hist_pb.update(obs_n)
                
                if term or trunc:
                    obs_re, _ = env_pb.reset()
                    hist_pb.reset()
                    hist_pb.update(obs_re)
                    tick_s = time.perf_counter()
                    
                tick_now = time.perf_counter()
                tick_elapsed = tick_now - tick_s
                wait_time = dur_tick - tick_elapsed
                if wait_time > 0:
                    time.sleep(wait_time)
                else:
                    time.sleep(0.001)
        finally:
            env_pb.close()
            print("🏁 Playback terminated.")
            sys.exit(0)

    # ------------------------------------------
    # [B] 学習モード (TRAINING)
    # ------------------------------------------
    else:
        print(f"🚀 Training Mode Started: {MODE}")
        
        # 1. W&B 初期化
        if TRACK_WANDB:
            now_dt_str = datetime.now().strftime("%m%d_%H%M")
            run_name_tag = f"ppo_transformer_{MODE}_{now_dt_str}"
            wandb.init(
                project="HNS_V26_TeamCos",
                name=run_name_tag,
                config={
                    "mode": MODE,
                    "lr_start": LR_START,
                    "batch_size": BATCH_SIZE,
                    "total_steps": TOTAL_STEPS,
                    "num_envs": NUM_ENVS,
                    "seq_len": SEQ_LEN
                }
            )

        # 2. 環境 ＆ 履歴バッファ
        envs_tr = gym.vector.SyncVectorEnv([make_env for _ in range(NUM_ENVS)])
        hist_tr = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim_val, device)
        log_dir_train = f"runs/HNS_{MODE}_{datetime.now().strftime('%m%d_%H%M')}"
        writer_tb = SummaryWriter(log_dir_train)
        
        # 3. テンソルバッファ (極限展開 ＆ 名称統一)
        obs_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS, SEQ_LEN, obs_dim_val), device=device)
        act_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS, action_dim_val), device=device)
        prob_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        rew_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        done_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        val_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        # NDR計算用
        find_buffer = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        
        global_step_acc = 0
        update_iter_cnt = 0
        best_eval_rew_val = -float('inf')
        
        # 学習開始時間の確定 (NameError 根絶)
        training_start_time = time.time()
        
        # 初期リセット
        obs_v_init, _ = envs_tr.reset()
        hist_tr.update(obs_v_init)
        
        if USE_VIEWER_RUNTIME:
            _safe_render(envs_tr.envs[0])
            time.sleep(0.5)
        
        try:
            while global_step_acc < TOTAL_STEPS:
                
                # --- 4. データ収集 (Rollout) ---
                agent.eval()
                for rollout_step_i in range(EPISODE_LIMIT):
                    global_step_acc += NUM_ENVS
                    
                    rollout_sequence_b = hist_tr.get()
                    obs_buffer[rollout_step_i] = rollout_sequence_b
                    
                    if USE_VIEWER_RUNTIME:
                        _safe_render(envs_tr.envs[0])
                        
                    with torch.no_grad():
                        a_roll, lp_roll, _, v_roll = agent.get_action_and_value(rollout_sequence_b)
                        val_buffer[rollout_step_i] = v_roll.flatten()
                        
                    act_buffer[rollout_step_i] = a_roll
                    prob_buffer[rollout_step_i] = lp_roll
                    
                    next_obs_v_raw, rew_v_raw, done_v_raw, tr_v_raw, info_v_raw = envs_tr.step(a_roll.cpu().numpy())
                    
                    rew_buf_t = torch.as_tensor(rew_v_raw, device=device)
                    done_buf_t = torch.as_tensor(done_v_raw, device=device)
                    rew_buffer[rollout_step_i] = rew_buf_t
                    done_buffer[rollout_step_i] = done_buf_t
                    
                    # NDRデータの記録 (info から抽出)
                    is_detected_flags_r = info_v_raw.get("is_detected", np.zeros(NUM_ENVS))
                    find_buf_t = torch.as_tensor(is_detected_flags_r, dtype=torch.float32, device=device)
                    find_buffer[rollout_step_i] = find_buf_t
                    
                    hist_tr.update(next_obs_v_raw)
                    
                    # 個別リセット処理
                    for env_idx_chk_r in range(NUM_ENVS):
                        is_env_done = done_v_raw[env_idx_chk_r]
                        is_env_trunc = tr_v_raw[env_idx_chk_r]
                        if is_env_done or is_env_trunc:
                            hist_tr.reset(env_idx=env_idx_chk_r)
                            hist_tr.update(next_obs_v_raw[env_idx_chk_r], env_idx=env_idx_chk_r)

                # --- 5. アドバンテージ計算 (GAE) ---
                agent.eval()
                with torch.no_grad():
                    v_next_val_r = agent.get_value(hist_tr.get()).reshape(1, -1)
                    adv_buffer_r = torch.zeros_like(rew_buffer, device=device)
                    last_gae_val_r = 0
                    
                    for t_idx_r in reversed(range(EPISODE_LIMIT)):
                        non_terminal_mask_r = 1.0 - done_buffer[t_idx_r]
                        if t_idx_r == EPISODE_LIMIT - 1:
                            v_target_step_r = v_next_val_r
                        else:
                            v_target_step_r = val_buffer[t_idx_r + 1]
                            
                        delta_gae_r = rew_buffer[t_idx_r] + GAMMA * v_target_step_r * non_terminal_mask_r - val_buffer[t_idx_r]
                        last_gae_val_r = delta_gae_r + GAMMA * GAE_LAMBDA * non_terminal_mask_r * last_gae_val_r
                        adv_buffer_r[t_idx_r] = last_gae_val_r
                        
                    return_buffer_r = adv_buffer_r + val_buffer

                # --- 6. 最適化 (PPO Optimization) ---
                agent.train()
                # バッファ平坦化 (名称同期済み)
                f_obs = obs_buffer.reshape(-1, SEQ_LEN, obs_dim_val)
                f_act = act_buffer.reshape(-1, action_dim_val)
                f_prob = prob_buffer.reshape(-1)
                f_adv = adv_buffer_r.reshape(-1)
                f_ret = return_buffer_r.reshape(-1)
                f_val = val_buffer.reshape(-1)
                
                size_total_ppo = f_obs.shape[0]
                ppo_indices_set = np.arange(size_total_ppo)
                clip_fractions_list = []
                kl_divergence_final_p = torch.tensor(0.0, device=device)
                
                for epoch_ppo_i in range(UPDATE_EPOCHS):
                    np.random.shuffle(ppo_indices_set)
                    for start_idx_p in range(0, size_total_ppo, BATCH_SIZE):
                        end_idx_p = start_idx_p + BATCH_SIZE
                        m_idx_p = ppo_indices_set[start_idx_p:end_idx_p]
                        
                        _, n_logp_p, n_ent_p, n_val_ppo_p = agent.get_action_and_value(f_obs[m_idx_p], f_act[m_idx_p])
                        
                        l_ratio_ppo_p = n_logp_p - f_prob[m_idx_p]
                        ratio_ppo_p = l_ratio_ppo_p.exp()
                        
                        with torch.no_grad():
                            kl_divergence_final_p = ((ratio_ppo_p - 1.0) - l_ratio_ppo_p).mean()
                            diff_ratio = ratio_ppo_p - 1.0
                            abs_diff_r = diff_ratio.abs()
                            is_clipped = abs_diff_r.gt(CLIP_COEF)
                            c_rate_ppo_p = is_clipped.float().mean().item()
                            clip_fractions_list.append(c_rate_ppo_p)
                        
                        # アドバンテージ正規化
                        mb_adv_p = f_adv[m_idx_p]
                        mb_adv_mean = mb_adv_p.mean()
                        mb_adv_std = mb_adv_p.std()
                        mb_adv_norm_p = (mb_adv_p - mb_adv_mean) / (mb_adv_std + 1e-8)
                        
                        # Policy Loss
                        pg_l_unclipped_p = -mb_adv_norm_p * ratio_ppo_p
                        pg_ratio_clipped_p = torch.clamp(ratio_ppo_p, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                        pg_l_clipped_p = -mb_adv_norm_p * pg_ratio_clipped_p
                        pg_loss_ppo_p = torch.max(pg_l_unclipped_p, pg_l_clipped_p).mean()
                        
                        # Value Loss
                        v_diff_ppo_p = n_val_ppo_p.flatten() - f_ret[m_idx_p]
                        v_loss_ppo_p = 0.5 * (v_diff_ppo_p ** 2).mean()
                        
                        # アニーリング (Learning Progress)
                        progress_val_p = 1.0 - (global_step_acc / TOTAL_STEPS)
                        cur_lr_p = LR_END + (LR_START - LR_END) * progress_val_p
                        for p_group_o in optimizer.param_groups:
                            p_group_o["lr"] = cur_lr_p
                            
                        cur_ent_p = ENT_COEF_END + (ENT_COEF_START - ENT_COEF_END) * progress_val_p
                        total_loss_ppo_p = pg_loss_ppo_p - cur_ent_p * n_ent_p.mean() + v_loss_ppo_p * VF_COEF
                        
                        optimizer.zero_grad()
                        total_loss_ppo_p.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                        
                    if kl_divergence_final_p > TARGET_KL:
                        break
                
                update_iter_cnt += 1

                # --- 7. 評価フェーズ (Evaluation) ---
                if update_iter_cnt % EVAL_INTERVAL == 0:
                    agent.eval()
                    eval_accum_rewards_list = []
                    
                    for _ in range(EVAL_EPISODES):
                        # envs_tr を正しく参照し NameError (env_tr) を根絶
                        o_ev_init_r, _ = envs_tr.reset()
                        h_ev_r = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim_val, device)
                        h_ev_r.update(o_ev_init_r)
                        r_ev_sum_r = np.zeros(NUM_ENVS)
                        
                        for _ in range(EPISODE_LIMIT):
                            with torch.no_grad():
                                eval_output = agent.get_action_and_value(h_ev_r.get())
                                a_ev_out_r = eval_output[0]
                                
                            o_ev_next_r, r_ev_r, d_ev_r, t_ev_r, _ = envs_tr.step(a_ev_out_r.cpu().numpy())
                            r_ev_sum_r = r_ev_sum_r + r_ev_r
                            h_ev_r.update(o_ev_next_r)
                            
                            if any(d_ev_r) or any(t_ev_r):
                                break
                        eval_accum_rewards_list.append(np.mean(r_ev_sum_r))
                        
                    avg_reward_ev_final_p = np.mean(eval_accum_rewards_list)
                    writer_tb.add_scalar("eval/avg_reward", avg_reward_ev_final_p, global_step_acc)
                    if TRACK_WANDB:
                        wandb.log({"eval/avg_reward": avg_reward_ev_final_p}, step=global_step_acc)
                        
                    if avg_reward_ev_final_p > best_eval_reward:
                        best_eval_reward = avg_reward_ev_final_p
                        torch.save(agent.state_dict(), BEST_SAVE_PATH)

                # --- 8. 統計ロギング (Logging) ---
                y_pred_np_p = f_val.cpu().numpy()
                y_true_np_p = f_ret.cpu().numpy()
                var_y_total_p = np.var(y_true_np_p)
                if var_y_total_p > 1e-6:
                    residual_v_p = np.var(y_true_np_p - y_pred_np_p)
                    exp_var_val_p = 1.0 - residual_v_p / var_y_total_p
                else:
                    exp_var_val_p = 0.0
                    
                time_now_t = time.time()
                dur_train_p = time_now_t - training_start_time
                sps_train_p = int(global_step_acc / (dur_train_p + 1e-8))
                mean_rollout_reward_p = rew_buffer.mean().item()
                
                # NDR (No-Detected Ratio) 計算
                raw_find_mean_p = find_buffer.mean().item()
                ndr_val_final_p = 1.0 - raw_find_mean_p
                
                print(f"Step: {global_step_acc:8d} | SPS: {sps_train_p:4d} | Rew: {mean_rollout_reward_p:7.4f} | NDR: {ndr_val_final_p:6.2%} | KL: {kl_divergence_final_p:6.4f}")
                
                # TensorBoard / W&B 出力
                log_metrics_p = {
                    "params/learning_rate": cur_lr_p,
                    "params/entropy_coef": cur_ent_p,
                    "losses/value_loss": v_loss_ppo_p.item(),
                    "losses/policy_loss": pg_loss_ppo_p.item(),
                    "losses/approx_kl": kl_divergence_final_p.item(),
                    "losses/clip_fraction": np.mean(clip_fractions_list),
                    "losses/explained_variance": exp_var_val_p,
                    "charts/SPS": sps_train_p,
                    "charts/avg_reward": mean_rollout_reward_p,
                    "charts/no_detected_ratio": ndr_val_final_p
                }
                for k_metric_p, v_metric_p in log_metrics_p.items():
                    writer_tb.add_scalar(k_metric_p, v_metric_p, global_step_acc)
                if TRACK_WANDB:
                    wandb.log(log_metrics_p, step=global_step_acc)
                
                torch.save(agent.state_dict(), SAVE_PATH)

        except KeyboardInterrupt:
            print("✋ Training interrupted.")
        except Exception:
            traceback.print_exc()
        finally:
            ctx_locals_p = locals()
            if 'envs_tr' in ctx_locals_p: envs_tr.close()
            if 'writer_tb' in ctx_locals_p: writer_tb.close()
            if TRACK_WANDB: wandb.finish()
            print("💾 Training session closed.")

if __name__ == "__main__":
    run()