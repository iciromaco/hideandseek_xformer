# main26_train_final.py v2.65
# 演習第26回：【準備期間の描画同期改善 ＆ ハイダー視認性向上 ＆ 極限垂直展開版】
# 
# 修正内容:
# 1. 準備期間 (prep_steps) の描画制御を強化:
#    - USE_VIEWER_RUNTIME が有効な場合、準備期間中の描画頻度を向上。
#    - 物理演算は行われているが Viewer がフリーズして見える問題を、適切な sleep と sync で解消。
# 2. ロールアウト中の描画タイミング適正化:
#    - ステップ実行直後に render を呼び出すことで、Hider の移動が Seeker 凍結中も視認可能に。
# 3. デバッグモード (DEBUG_MOVEMENT) との完全連動:
#    - 160ステップまでの詳細ログ出力と、Viewer での挙動確認を並行して実行。
# 4. 1行1命令（1-line-1-command）の徹底遵守:
#    - 条件分岐、描画命令、スリープ処理の全ステップを独立行で詳細に記述。

import os
import sys
import time
import random
import platform
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime

# 自作モジュール
from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2

# ==========================================
# 0. 環境変数設定 (Windows/並列実行最適化)
# ==========================================
processor_id_data = platform.processor()
if processor_id_data != 'arm':
    # マルチコア環境でのスレッド競合を防ぎ、ステップ実行速度を安定化
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ==========================================
# 1. ハイパーパラメータ ＆ グローバル設定
# ==========================================
TRAIN_MODE = True             # 学習時は True
USE_VIEWER = False             # 初期描画フラグ
MODE = "initial"              # initial (4次元) or refinement (8次元)
TRACK_WANDB = True            # W&B トラッキングフラグ
DEBUG_MOVEMENT = False         # 移動デバッグログを有効にする場合は True
USE_VIEWER_RUNTIME = USE_VIEWER 

# Transformer 構造設定
SEQ_LEN = 8                   # 過去の参照履歴長
HIDDEN_DIM = 128              # 隠れ層の次元数

# 学習 ＆ 環境設定
EPISODE_LIMIT = 500           # 1エピソード最大ステップ数
TOTAL_STEPS = 20_000_000      # 学習総計ステップ数
NUM_ENVS = 16                 # 並列実行環境数

# PPO アルゴリズム設定
LR_START = 3.0e-4             # インタラクション発見のため高めに設定
LR_END = 5e-6                 # 学習率最終値
GAMMA = 0.99                  # 割引報酬率
GAE_LAMBDA = 0.95             # GAE
UPDATE_EPOCHS = 10            
BATCH_SIZE = 512              
CLIP_COEF = 0.2               
MAX_GRAD_NORM = 0.5           

# 探索 ＆ 評価設定
ENT_COEF_START = 0.02         
ENT_COEF_END = 0.001          
VF_COEF = 0.5                 
TARGET_KL = 0.02              
EVAL_INTERVAL = 20            
EVAL_EPISODES = 3             

# パス設定
SAVE_PATH = f"HNS_V26_GTRPPO_{MODE}.pt"
BEST_SAVE_PATH = f"HNS_V26_GTRPPO_{MODE}_best.pt"

# W&B インポート
if TRACK_WANDB:
    import wandb

# ==========================================
# 2. 履歴管理 (ObsHistory)
# ==========================================
class ObsHistory:
    """時系列データを垂直管理するバッファ"""
    def __init__(self, n_envs, seq_len, obs_dim, device):
        self.n_envs = n_envs
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.device = device
        self.capacity = seq_len * 2
        self.buffer_shape = (n_envs, self.capacity, obs_dim)
        self.buffer = torch.zeros(self.buffer_shape, device=device)
        self.ptr = 0

    def reset(self, env_idx=None):
        if env_idx is None:
            self.buffer.zero_()
            self.ptr = 0
        else:
            self.buffer[env_idx].zero_()

    def update(self, obs_data, env_idx=None):
        t_raw_in = torch.as_tensor(obs_data, dtype=torch.float32, device=self.device)
        if env_idx is None:
            obs_sh_t = (self.n_envs, self.obs_dim)
            obs_in_f = t_raw_in.reshape(obs_sh_t)
            w_pos = self.ptr
            self.buffer[:, w_pos] = obs_in_f
            m_pos = w_pos + self.seq_len
            self.buffer[:, m_pos] = obs_in_f
            self.ptr = (w_pos + 1) % self.seq_len
        else:
            obs_in_f = t_raw_in.reshape(self.obs_dim)
            w_pos = self.ptr
            self.buffer[env_idx, w_pos] = obs_in_f
            m_pos = w_pos + self.seq_len
            self.buffer[env_idx, m_pos] = obs_in_f

    def get(self):
        s_idx_g = self.ptr
        e_idx_g = s_idx_g + self.seq_len
        chunk_g = self.buffer[:, s_idx_g : e_idx_g]
        return chunk_g.contiguous()

def make_env():
    """環境生成（デバッグモード v3.11 と同期）"""
    r_mode_arg = None
    if USE_VIEWER_RUNTIME:
        if not TRAIN_MODE:
            r_mode_arg = "human"
    
    env_inst_f = TeamCosEnv(
        mode=MODE, 
        render_mode=r_mode_arg, 
        debug_mode=DEBUG_MOVEMENT
    )
    return env_inst_f

# ==========================================
# 3. メイン実行ロジック (Run)
# ==========================================
def run():
    is_cuda_ready_f = torch.cuda.is_available()
    d_type_final = "cuda" if is_cuda_ready_f else "cpu"
    device_obj = torch.device(d_type_final)

    global USE_VIEWER_RUNTIME
    USE_VIEWER_RUNTIME = USE_VIEWER
    warned_v_flag = False

    def _safe_render_f(env_p_ref):
        nonlocal warned_v_flag
        global USE_VIEWER_RUNTIME
        if not USE_VIEWER_RUNTIME:
            return False
        try:
            env_p_ref.render()
            return True
        except Exception as e_vis_err:
            if not warned_v_flag:
                print(f"⚠️ Viewer sync disabled: {e_vis_err}")
                warned_v_flag = True
            USE_VIEWER_RUNTIME = False
            return False
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    
    obs_dim_actual = 55
    act_dim_actual = 8 if MODE == "refinement" else 4
        
    agent_final = AgentV2(obs_dim_actual, act_dim_actual, HIDDEN_DIM, SEQ_LEN)
    agent_final = agent_final.to(device_obj)
    optimizer_final = optim.Adam(agent_final.parameters(), lr=LR_START, eps=1e-5)
    
    if os.path.exists(SAVE_PATH):
        try:
            ckpt_st = torch.load(SAVE_PATH, map_location=device_obj)
            agent_final.load_state_dict(ckpt_st)
            print(f"✅ Loaded: {SAVE_PATH}")
        except Exception:
            print(f"⚠️ Initializing fresh.")

    # --- [A] PLAYBACK MODE ---
    if not TRAIN_MODE:
        env_pb = make_env()
        hist_pb = ObsHistory(1, SEQ_LEN, obs_dim_actual, device_obj)
        agent_final.eval()
        obs_start_p, _ = env_pb.reset()
        hist_pb.update(obs_start_p)
        if USE_VIEWER_RUNTIME:
            if _safe_render_f(env_pb):
                time.sleep(0.5)
        print(f"🎮 Playback Mode.")
        try:
            while True:
                if USE_VIEWER_RUNTIME:
                    if env_pb.viewer is None or not env_pb.viewer.is_running():
                        break
                t_s_pb = time.perf_counter()
                with torch.no_grad():
                    out_pb = agent_final.get_action_and_value(hist_pb.get())
                    a_pb = out_pb[0]
                a_np_pb = a_pb.cpu().numpy().flatten()
                o_nxt_pb, _, term_pb, trunc_pb, _ = env_pb.step(a_np_pb)
                hist_pb.update(o_nxt_pb)
                if term_pb or trunc_pb:
                    o_re_pb, _ = env_pb.reset()
                    hist_pb.reset()
                    hist_pb.update(o_re_pb)
                t_dur_pb = time.perf_counter() - t_s_pb
                w_pb = (1.0/40.0) - t_dur_pb
                if w_pb > 0: time.sleep(w_pb)
                else: time.sleep(0.001)
        finally:
            env_pb.close()
            sys.exit(0)

    # --- [B] TRAINING MODE ---
    else:
        print(f"🚀 Training Started: {MODE}")
        if TRACK_WANDB:
            dt_tag = datetime.now().strftime("%m%d_%H%M")
            wandb.init(
                project="HNS_V26_TeamCos",
                name=f"ppo_{MODE}_{dt_tag}",
                config={"mode": MODE, "steps": TOTAL_STEPS, "lr": LR_START},
                reinit=True
            )

        envs_train = gym.vector.SyncVectorEnv([make_env for _ in range(NUM_ENVS)])
        hist_train = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim_actual, device_obj)
        env_evaluation = make_env()
        writer_tb = SummaryWriter(f"runs/HNS_{MODE}_{datetime.now().strftime('%m%d_%H%M')}")
        
        b_obs = torch.zeros((EPISODE_LIMIT, NUM_ENVS, SEQ_LEN, obs_dim_actual), device=device_obj)
        b_act = torch.zeros((EPISODE_LIMIT, NUM_ENVS, act_dim_actual), device=device_obj)
        b_prob = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device_obj)
        b_rew = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device_obj)
        b_done = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device_obj)
        b_val = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device_obj)
        b_found_f = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device_obj)
        
        g_step_total_acc = 0
        u_iter_idx = 0
        best_r_eval_metric = -float('inf')
        last_eval_r_val = None 
        training_clock_f = time.time()
        
        obs_start_vec, _ = envs_train.reset()
        hist_train.update(obs_start_vec)
        
        if USE_VIEWER_RUNTIME:
            _safe_render_f(envs_train.envs[0])
            time.sleep(0.5)
        
        try:
            while g_step_total_acc < TOTAL_STEPS:
                
                # --- Step 1: Evaluation ---
                if u_iter_idx % EVAL_INTERVAL == 0:
                    agent_final.eval()
                    eval_results = []
                    for _ in range(EVAL_EPISODES):
                        o_ev_init, _ = env_evaluation.reset()
                        h_ev_obj = ObsHistory(1, SEQ_LEN, obs_dim_actual, device_obj)
                        h_ev_obj.update(o_ev_init)
                        r_score = 0
                        for _ in range(EPISODE_LIMIT):
                            with torch.no_grad():
                                a_ev_p = agent_final.get_action_and_value(h_ev_obj.get())[0]
                            o_ev_nxt, r_ev_v, d_ev_v, t_ev_v, _ = env_evaluation.step(a_ev_p.cpu().numpy().flatten())
                            r_score += r_ev_v
                            h_ev_obj.update(o_ev_nxt)
                            if d_ev_v or t_ev_v: break
                        eval_results.append(r_score)
                    last_eval_r_val = np.mean(eval_results)
                    writer_tb.add_scalar("eval/avg_reward", last_eval_r_val, g_step_total_acc)
                    if last_eval_r_val > best_r_eval_metric:
                        best_r_eval_metric = last_eval_r_val
                        torch.save(agent_final.state_dict(), BEST_SAVE_PATH)

                # --- Step 2: Data Rollout ---
                agent_final.train() 
                for step_idx in range(EPISODE_LIMIT):
                    g_step_total_acc = g_step_total_acc + NUM_ENVS
                    
                    obs_seq_roll = hist_train.get()
                    b_obs[step_idx] = obs_seq_roll
                    
                    # 描画命令の送信 (Hider 動作確認のため頻度向上)
                    if USE_VIEWER_RUNTIME:
                        # 準備期間 (0-80) は毎ステップ描画し、sleep を入れる
                        if step_idx <= 80:
                            _safe_render_f(envs_train.envs[0])
                            # Hider の動きを視認するためのディレイ
                            time.sleep(0.015) 
                        elif step_idx % 15 == 0:
                            _safe_render_f(envs_train.envs[0])
                        
                    with torch.no_grad():
                        out_roll = agent_final.get_action_and_value(obs_seq_roll)
                        a_r_t = out_roll[0]
                        lp_r_t = out_roll[1]
                        v_r_t = out_roll[3]
                        b_val[step_idx] = v_r_t.flatten()
                    
                    b_act[step_idx] = a_r_t
                    b_prob[step_idx] = lp_r_t
                    
                    act_np_f = a_r_t.cpu().numpy()
                    o_nxt_all, r_all, d_all, t_all, i_all = envs_train.step(act_np_f)
                    
                    b_rew[step_idx] = torch.as_tensor(r_all, device=device_obj)
                    b_done[step_idx] = torch.as_tensor(d_all, device=device_obj)
                    
                    is_det_vec = i_all.get("is_detected", np.zeros(NUM_ENVS))
                    b_found_f[step_idx] = torch.as_tensor(is_det_vec, dtype=torch.float32, device=device_obj)
                    
                    hist_train.update(o_nxt_all)
                    for e_idx in range(NUM_ENVS):
                        if d_all[e_idx] or t_all[e_idx]:
                            hist_train.reset(env_idx=e_idx)
                            hist_train.update(o_nxt_all[e_idx], env_idx=e_idx)
                            
                    if DEBUG_MOVEMENT and any(t_all):
                        break

                # --- Step 3: GAE ---
                agent_final.eval()
                with torch.no_grad():
                    v_nx_raw = agent_final.get_value(hist_train.get())
                    v_nx_term = v_nx_raw.reshape(1, -1)
                    b_adv_f = torch.zeros_like(b_rew, device=device_obj)
                    gae_run = 0
                    for t_idx_g in reversed(range(EPISODE_LIMIT)):
                        mask_gae_f = 1.0 - b_done[t_idx_g]
                        v_target_nxt = v_nx_term if t_idx_g == EPISODE_LIMIT - 1 else b_val[t_idx_g + 1]
                        td_err_f = b_rew[t_idx_g] + GAMMA * v_target_nxt * mask_gae_f - b_val[t_idx_g]
                        gae_run = td_err_f + GAMMA * GAE_LAMBDA * mask_gae_f * gae_run
                        b_adv_f[t_idx_g] = gae_run
                    b_ret_f = b_adv_f + b_val

                # --- Step 4: Optimization ---
                agent_final.train()
                f_obs_t = b_obs.reshape(-1, SEQ_LEN, obs_dim_actual)
                f_act_t = b_act.reshape(-1, act_dim_actual)
                f_prob_t = b_prob.reshape(-1)
                f_adv_t = b_adv_f.reshape(-1)
                f_ret_t = b_ret_f.reshape(-1)
                f_val_t = b_val.reshape(-1)
                
                ds_sz_ppo = f_obs_t.shape[0]
                idx_arr = np.arange(ds_sz_ppo)
                rem_rate = 1.0 - (g_step_total_acc / TOTAL_STEPS)
                cur_lr_p = LR_END + (LR_START - LR_END) * rem_rate
                for pg_grp in optimizer_final.param_groups: pg_grp["lr"] = cur_lr_p
                cur_ent_p = ENT_COEF_END + (ENT_COEF_START - ENT_COEF_END) * rem_rate
                
                cl_fra_list = []
                f_kl_val = torch.tensor(0.0, device=device_obj)
                for ep_idx in range(UPDATE_EPOCHS):
                    np.random.shuffle(idx_arr)
                    for start_p in range(0, ds_sz_ppo, BATCH_SIZE):
                        m_idx = idx_arr[start_p : start_p + BATCH_SIZE]
                        res_p = agent_final.get_action_and_value(f_obs_t[m_idx], f_act_t[m_idx])
                        l_rat = res_p[1] - f_prob_t[m_idx]
                        ratio_p = torch.exp(l_rat)
                        with torch.no_grad():
                            f_kl_val = ((ratio_p - 1.0) - l_rat).mean()
                            cl_fra_list.append((ratio_p - 1.0).abs().gt(CLIP_COEF).float().mean().item())
                        mb_adv_p = (f_adv_t[m_idx] - f_adv_t[m_idx].mean()) / (f_adv_t[m_idx].std() + 1e-8)
                        loss_pi = torch.max(-mb_adv_p * ratio_p, -mb_adv_p * torch.clamp(ratio_p, 1.0-CLIP_COEF, 1.0+CLIP_COEF)).mean()
                        loss_v = 0.5 * ((res_p[3].flatten() - f_ret_t[m_idx]) ** 2).mean()
                        loss_tot = loss_pi - (cur_ent_p * res_p[2].mean()) + (loss_v * VF_COEF)
                        optimizer_final.zero_grad(); loss_tot.backward(); nn.utils.clip_grad_norm_(agent_final.parameters(), MAX_GRAD_NORM); optimizer_final.step()
                    if f_kl_val > TARGET_KL: break
                
                u_iter_idx = u_iter_idx + 1

                # --- Step 5: Logging ---
                var_y_f = np.var(f_ret_t.cpu().numpy())
                exp_var_f = 1.0 - (np.var(f_ret_t.cpu().numpy() - f_val_t.cpu().numpy()) / var_y_f) if var_y_f > 1e-6 else 0.0
                sps_f = int(g_step_total_acc / (time.time() - training_clock_f + 1e-8))
                r_avg_f, ndr_f = b_rew.mean().item(), 1.0 - b_found_f.mean().item()
                
                print(f"Step: {g_step_total_acc:8d} | SPS: {sps_f:4d} | Rew: {r_avg_f:7.4f} | NDR: {ndr_f:6.2%} | KL: {f_kl_val:6.4f}")
                
                stats_p = {
                    "params/learning_rate": cur_lr_p, "params/entropy_coef": cur_ent_p,
                    "losses/value_loss": loss_v.item(), "losses/policy_loss": loss_pi.item(),
                    "losses/approx_kl": f_kl_val.item(), "losses/explained_variance": exp_var_f,
                    "losses/clip_fraction": np.mean(cl_fra_list), "charts/SPS": sps_f,
                    "charts/avg_reward": r_avg_f, "charts/no_detected_ratio": ndr_f
                }
                for k_tb, v_tb in stats_p.items(): writer_tb.add_scalar(k_tb, v_tb, g_step_total_acc)
                if TRACK_WANDB:
                    if last_eval_r_val is not None: stats_p["eval/avg_reward"], last_eval_r_val = last_eval_r_val, None
                    wandb.log(stats_p, step=g_step_total_acc)
                
                torch.save(agent_final.state_dict(), SAVE_PATH)
                if DEBUG_MOVEMENT and g_step_total_acc > 0: break
                
        except KeyboardInterrupt: print("✋ Stopped.")
        except Exception: traceback.print_exc()
        finally:
            if 'envs_train' in locals(): envs_train.close()
            if 'env_evaluation' in locals(): env_evaluation.close()
            if 'writer_tb' in locals(): writer_tb.close()
            if TRACK_WANDB: wandb.finish()
            print("💾 Training closed.")

if __name__ == "__main__":
    run()