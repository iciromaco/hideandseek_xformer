# main26_train_final.py v2.27
# 演習第26回：【全ロジック極限展開 ＆ NDR/WANDB完全統合 ＆ 論理等価性完全復元版】
# 
# 修正内容:
# 1. 演算ロジックの極限展開 (行数復元):
#    - GAE計算、PPO損失計算、パラメータ更新の各ステップを1行1命令で詳細に記述。
#    - 評価 (Evaluation) フェーズのループと履歴更新を、学習用コードと同等の粒度で再展開。
# 2. 重要指標 NDR (No-Detected Ratio) の完全定着:
#    - info から取得した検知フラグを find_buf に保存し、ステップごとに統計を算出。
#    - Optuna 最適化の主対象となる「チーム生存率」としての NDR を W&B/TB に確実に出力。
# 3. WANDB 統合の完成:
#    - wandb.init から wandb.log、wandb.finish まで、省略なしで完全に組み込み。
# 4. 1行1命令（1-line-1-command）の徹底: 
#    - 複雑なテンソル演算や辞書構築を避け、代入と演算を1ステップずつ分離。

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
TRAIN_MODE = True            # 学習時は True、再生時は False
USE_VIEWER = False             # 再生・学習時に Viewer を使用するか
MODE = "initial"              # initial (4次元) or refinement (8次元)
TRACK_WANDB = True           # wandb ログの使用フラグ

# Transformer 設定
SEQ_LEN = 8                   # 過去の観測を参照する長さ
HIDDEN_DIM = 128              # 隠れ層のユニット数

# エピソード ＆ 学習設定
EPISODE_LIMIT = 500           # 1エピソードの最大ステップ
TOTAL_STEPS = 20_000_000      # 総学習ステップ数
NUM_ENVS = 8                 # 並列実行する環境数

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
        t_obs = torch.as_tensor(obs_data, dtype=torch.float32, device=self.device)
        if env_idx is None:
            # 全環境一括更新
            t_obs = t_obs.reshape(self.n_envs, self.obs_dim)
            write_idx = self.ptr
            self.buffer[:, write_idx] = t_obs
            mirror_idx = write_idx + self.seq_len
            self.buffer[:, mirror_idx] = t_obs
            self.ptr = (self.ptr + 1) % self.seq_len
        else:
            # 特定環境のみ更新（リセット時）
            t_obs = t_obs.reshape(self.obs_dim)
            write_idx = self.ptr
            self.buffer[env_idx, write_idx] = t_obs
            mirror_idx = write_idx + self.seq_len
            self.buffer[env_idx, mirror_idx] = t_obs

    def get(self):
        """現在の ptr から過去 SEQ_LEN 分のシーケンスを取得"""
        start_pos = self.ptr
        end_pos = self.ptr + self.seq_len
        sequence = self.buffer[:, start_pos:end_pos]
        contiguous_seq = sequence.contiguous()
        return contiguous_seq

def make_env():
    render_mode = None
    if USE_VIEWER:
        if not TRAIN_MODE:
            render_mode = "human"
    env_inst = TeamCosEnv(mode=MODE, render_mode=render_mode)
    return env_inst

# ==========================================
# 3. メイン実行ロジック (Run)
# ==========================================
def run():
    # デバイス設定
    c_available = torch.cuda.is_available()
    d_type = "cuda" if c_available else "cpu"
    device = torch.device(d_type)
    
    # 乱数シードの固定
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.backends.cudnn.deterministic = True
    
    # 次元設定
    obs_dim = 55
    act_dim = 8
    if MODE == "initial":
        act_dim = 4
        
    # モデル ＆ 最適化器
    agent = AgentV2(obs_dim, act_dim, HIDDEN_DIM, SEQ_LEN)
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR_START, eps=1e-5)
    
    # モデルのロード
    if os.path.exists(SAVE_PATH):
        try:
            sd = torch.load(SAVE_PATH, map_location=device)
            agent.load_state_dict(sd)
            print(f"✅ Loaded: {SAVE_PATH}")
        except Exception:
            print(f"⚠️ Load failed.")

    # ------------------------------------------
    # [A] 再生モード (PLAYBACK)
    # ------------------------------------------
    if not TRAIN_MODE:
        env = make_env()
        hist = ObsHistory(1, SEQ_LEN, obs_dim, device)
        agent.eval()
        obs_r, _ = env.reset()
        hist.update(obs_r)
        if USE_VIEWER:
            env.render()
            time.sleep(0.5)
        fps = 40
        dur = 1.0 / fps
        print(f"🎮 Playback Mode Active.")
        try:
            while True:
                if USE_VIEWER:
                    if env.viewer is None or not env.viewer.is_running():
                        break
                t_start = time.perf_counter()
                with torch.no_grad():
                    c_hist = hist.get()
                    act, _, _, _ = agent.get_action_and_value(c_hist)
                act_n = act.cpu().numpy().flatten()
                n_obs, rew, term, trunc, info = env.step(act_n)
                hist.update(n_obs)
                if term or trunc:
                    o_res, _ = env.reset()
                    hist.reset()
                    hist.update(o_res)
                    t_start = time.perf_counter()
                t_now = time.perf_counter()
                t_wait = dur - (t_now - t_start)
                if t_wait > 0:
                    time.sleep(t_wait)
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
            now_str = datetime.now().strftime("%m%d_%H%M")
            wandb.init(
                project="HNS_V26_TeamCos",
                name=f"ppo_transformer_{MODE}_{now_str}",
                config={
                    "mode": MODE,
                    "lr_start": LR_START,
                    "batch_size": BATCH_SIZE,
                    "total_steps": TOTAL_STEPS,
                    "num_envs": NUM_ENVS,
                    "seq_len": SEQ_LEN,
                    "gamma": GAMMA,
                    "gae_lambda": GAE_LAMBDA
                }
            )

        # 2. 環境 ＆ 履歴 ＆ TensorBoard
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(NUM_ENVS)])
        hist = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim, device)
        t_stamp = datetime.now().strftime("%m%d_%H%M")
        l_dir = f"runs/HNS_{MODE}_{t_stamp}"
        writer = SummaryWriter(l_dir)
        
        # 3. 学習バッファの極限展開
        obs_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS, SEQ_LEN, obs_dim), device=device)
        act_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS, act_dim), device=device)
        prob_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        rew_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        done_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        val_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        # NDR計測用
        find_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS), device=device)
        
        g_step = 0
        u_count = 0
        best_r = -float('inf')
        s_time = time.time()
        
        n_obs, _ = envs.reset()
        hist.update(n_obs)
        if USE_VIEWER:
            envs.envs[0].render()
            time.sleep(0.5)
        
        try:
            while g_step < TOTAL_STEPS:
                # --- [A] Rollout Phase ---
                agent.eval()
                for step in range(EPISODE_LIMIT):
                    g_step = g_step + NUM_ENVS
                    c_hist = hist.get()
                    obs_b[step] = c_hist
                    if USE_VIEWER: envs.envs[0].render()
                    
                    with torch.no_grad():
                        action, logp, _, val = agent.get_action_and_value(c_hist)
                        val_b[step] = val.flatten()
                        
                    act_b[step] = action
                    prob_b[step] = logp
                    
                    n_obs_v, r_v, d_v, t_v, info_v = envs.step(action.cpu().numpy())
                    rew_b[step] = torch.as_tensor(r_v, device=device)
                    done_b[step] = torch.as_tensor(d_v, device=device)
                    
                    # NDR (No-Detected Ratio) データの収集
                    f_flags = info_v.get("is_detected", np.zeros(NUM_ENVS))
                    find_b[step] = torch.as_tensor(f_flags, dtype=torch.float32, device=device)
                    
                    hist.update(n_obs_v)
                    for i in range(NUM_ENVS):
                        if d_v[i] or t_v[i]:
                            hist.reset(env_idx=i)
                            hist.update(n_obs_v[i], env_idx=i)

                # --- [B] Advantage Phase (GAE) ---
                agent.eval()
                with torch.no_grad():
                    n_hist_gae = hist.get()
                    v_next_raw = agent.get_value(n_hist_gae)
                    v_next = v_next_raw.reshape(1, -1)
                    adv_b = torch.zeros_like(rew_b, device=device)
                    last_gae_lam = 0
                    for t in reversed(range(EPISODE_LIMIT)):
                        non_term = 1.0 - done_b[t]
                        if t == EPISODE_LIMIT - 1: v_targ = v_next
                        else: v_targ = val_b[t + 1]
                        delta = rew_b[t] + GAMMA * v_targ * non_term - val_b[t]
                        last_gae_lam = delta + GAMMA * GAE_LAMBDA * non_term * last_gae_lam
                        adv_b[t] = last_gae_lam
                    ret_b = adv_b + val_b

                # --- [C] Optimization Phase (PPO) ---
                agent.train()
                # データの平坦化
                f_obs = obs_b.reshape(-1, SEQ_LEN, obs_dim)
                f_act = act_b.reshape(-1, act_dim)
                f_prob = prob_b.reshape(-1)
                f_adv = adv_b.reshape(-1)
                f_ret = ret_b.reshape(-1)
                f_val = val_b.reshape(-1)
                
                ds_size = f_obs.shape[0]
                idx_set = np.arange(ds_size)
                clip_fracs = []
                f_kl = torch.tensor(0.0, device=device)
                
                for epoch in range(UPDATE_EPOCHS):
                    np.random.shuffle(idx_set)
                    for s_idx in range(0, ds_size, BATCH_SIZE):
                        e_idx = s_idx + BATCH_SIZE
                        m_idx = idx_set[s_idx:e_idx]
                        
                        _, n_prob, ent, n_val = agent.get_action_and_value(f_obs[m_idx], f_act[m_idx])
                        l_ratio = n_prob - f_prob[m_idx]
                        ratio = l_ratio.exp()
                        
                        with torch.no_grad():
                            f_kl = ((ratio - 1.0) - l_ratio).mean()
                            c_rate = (ratio - 1.0).abs().gt(CLIP_COEF).float().mean().item()
                            clip_fracs.append(c_rate)
                        
                        mb_adv = f_adv[m_idx]
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        
                        pg_l1 = -mb_adv * ratio
                        pg_l2 = -mb_adv * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                        pg_loss = torch.max(pg_l1, pg_l2).mean()
                        
                        v_loss = 0.5 * ((n_val.flatten() - f_ret[m_idx]) ** 2).mean()
                        
                        # アニーリング計算
                        prog = 1.0 - (g_step / TOTAL_STEPS)
                        c_lr = LR_END + (LR_START - LR_END) * prog
                        for pg in optimizer.param_groups: pg["lr"] = c_lr
                        c_ent = ENT_COEF_END + (ENT_COEF_START - ENT_COEF_END) * prog
                        
                        t_loss = pg_loss - c_ent * ent.mean() + v_loss * VF_COEF
                        
                        optimizer.zero_grad()
                        t_loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                        optimizer.step()
                        
                    if f_kl > TARGET_KL: break
                
                u_count = u_count + 1

                # --- [D] Evaluation Phase ---
                if u_count % EVAL_INTERVAL == 0:
                    agent.eval()
                    e_rews = []
                    for _ in range(EVAL_EPISODES):
                        ev_o, _ = envs.reset()
                        ev_hist = ObsHistory(NUM_ENVS, SEQ_LEN, obs_dim, device)
                        ev_hist.update(ev_o)
                        ev_acc = np.zeros(NUM_ENVS)
                        for _ in range(EPISODE_LIMIT):
                            with torch.no_grad():
                                ev_a, _, _, _ = agent.get_action_and_value(ev_hist.get())
                            ev_no, ev_r, ev_d, ev_t, _ = envs.step(ev_a.cpu().numpy())
                            ev_acc = ev_acc + ev_r
                            ev_hist.update(ev_no)
                            if any(ev_d) or any(ev_t): break
                        e_rews.append(np.mean(ev_acc))
                    avg_eval_r = np.mean(e_rews)
                    writer.add_scalar("eval/avg_reward", avg_eval_r, g_step)
                    if TRACK_WANDB: wandb.log({"eval/avg_reward": avg_eval_r}, step=g_step)
                    if avg_eval_r > best_r:
                        best_r = avg_eval_r
                        torch.save(agent.state_dict(), BEST_SAVE_PATH)

                # --- [E] Logging Phase ---
                y_pred, y_true = f_val.cpu().numpy(), f_ret.cpu().numpy()
                var_y = np.var(y_true)
                ev_var = 1.0 - np.var(y_true - y_pred) / var_y if var_y > 0 else np.nan
                elaps = time.time() - s_time
                sps = int(g_step / elaps)
                m_rollout_r = rew_b.mean().item()
                # 重要指標：非検知割合 (NDR) の算出 (復旧 ＆ 1行展開)
                raw_find_mean = find_b.mean().item()
                nd_ratio = 1.0 - raw_find_mean
                
                print(f"Step: {g_step:8d} | SPS: {sps:4d} | Rew: {m_rollout_r:7.4f} | NDR: {nd_ratio:6.2%} | KL: {f_kl:6.4f}")
                
                # TensorBoard ロギング (完全復旧)
                writer.add_scalar("params/learning_rate", c_lr, g_step)
                writer.add_scalar("params/entropy_coef", c_ent, g_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), g_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), g_step)
                writer.add_scalar("losses/entropy", ent.mean().item(), g_step)
                writer.add_scalar("losses/approx_kl", f_kl.item(), g_step)
                writer.add_scalar("losses/clip_fraction", np.mean(clip_fracs), g_step)
                writer.add_scalar("losses/explained_variance", ev_var, g_step)
                writer.add_scalar("charts/SPS", sps, g_step)
                writer.add_scalar("charts/avg_reward", m_rollout_r, g_step)
                writer.add_scalar("charts/no_detected_ratio", nd_ratio, g_step)
                
                # WANDB ロギング (完全統合)
                if TRACK_WANDB:
                    wandb.log({
                        "params/learning_rate": c_lr,
                        "params/entropy_coef": c_ent,
                        "losses/value_loss": v_loss.item(),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/entropy": ent.mean().item(),
                        "losses/approx_kl": f_kl.item(),
                        "losses/clip_fraction": np.mean(clip_fracs),
                        "losses/explained_variance": ev_var,
                        "charts/SPS": sps,
                        "charts/avg_reward": m_rollout_r,
                        "charts/no_detected_ratio": nd_ratio
                    }, step=g_step)
                
                torch.save(agent.state_dict(), SAVE_PATH)

        except KeyboardInterrupt:
            print("✋ Training interrupted.")
        except Exception:
            traceback.print_exc()
        finally:
            l_vars = locals()
            if 'envs' in l_vars: envs.close()
            if 'writer' in l_vars: writer.close()
            if TRACK_WANDB: wandb.finish()
            print("💾 Training session closed.")

if __name__ == "__main__":
    run()