# main26_train_final.py v1.68
# 演習第26回：完全展開・ロジック省略一切禁止・40Hz精密同期 ＆ 推力・Rampブースト復旧版
#
# 遵守事項:
# 1. 処理の完全展開: 1行1命令を徹底。圧縮や ";" による複数代入を禁止。
# 2. ロジックの完遂: PPO学習ループ、GAE、Wandb(init/watch/log)、tqdm、Transformer履歴、40Hz同期。
# 3. 物理的整合性: 40Hz同期(0.025s)により高推力下でのオーバーシュートを抑制。
# 4. 資源管理: Viewerクローズ検知と sys.exit(0) によるクリーンな終了。

import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# --- [0] プロジェクトルートのパス解決 ---
# 実行スクリプトの場所に依存せず、src/ パッケージを読み込めるように設定
_script_path = Path(__file__).resolve()
_project_root = _script_path.parent

if str(_project_root) not in sys.path:
    # プロジェクトルート自体を検索パスに追加
    sys.path.insert(0, str(_project_root))
    # src ディレクトリを検索パスの最優先に追加（パッケージインポート用）
    sys.path.insert(0, os.path.join(str(_project_root), "src"))

from envs.hns_environment import TeamCosEnv
from models.ppo_transformer_v2 import AgentV2

# --- 1. モード・定数設定 ---
TRAIN_MODE = False  # True: 学習実行, False: 実時間再生(Viewer)
USE_VIEWER = True  # Viewerを表示するか
MODE = "initial"  # "initial" (H1のみ) or "refinement" (H1+H2学習)
TRACK_WANDB = True  # Weights & Biases 連携を使用するか

# Transformer / PPO ハイパーパラメータ
SEQ_LEN = 8  # Transformerの参照ステップ数（過去8歩のコンテキスト）
HIDDEN_DIM = 128  # Transformer 内部層の次元数
EPISODE_LIMIT = 500  # 1エピソードあたりの最大ステップ数
TOTAL_STEPS = 20_000_000  # 総学習環境ステップ数

# PPO アルゴリズム詳細設定
LR = 2.5e-4  # Adam 学習率
GAMMA = 0.99  # 未来報酬の割引率
GAE_LAMBDA = 0.95  # GAE パラメータ
UPDATE_EPOCHS = 4  # データ収集1回あたりの学習エポック数
BATCH_SIZE = 512  # 学習に使用するミニバッチサイズ
CLIP_COEF = 0.2  # PPO の方策変化制限
ENT_COEF = 0.01  # エントロピー係数
VF_COEF = 0.5  # 価値関数損失の重み
MAX_GRAD_NORM = 0.5  # 勾配クリッピング

# 管理情報の定義
EXPERIMENT_NAME = f"HNS_V26_GTRPPO_{MODE}"
SAVE_PATH = f"{EXPERIMENT_NAME}.pt"

# Wandb インポートの試行
if TRACK_WANDB:
    import wandb


def log_debug(msg):
    """詳細なタイムスタンプ付きデバッグログ出力"""
    now_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{now_str}] 🔍 {msg}")


class ObsHistory:
    """
    Transformer用の履歴管理クラス。
    ダブルバッファ構造を採用し、過去系列の連続取得を高速化。
    """

    def __init__(self, n_envs, seq_len, obs_dim, device):
        self.n_envs = n_envs
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.device = device
        # 連続スライスのため、2倍の長さのバッファを確保
        self.buffer = torch.zeros((n_envs, seq_len * 2, obs_dim), device=device)
        self.ptr = 0

    def reset(self, env_idx=None):
        """特定環境、または全環境の履歴をゼロリセット"""
        if env_idx is None:
            self.buffer.zero_()
            self.ptr = 0
        else:
            self.buffer[env_idx].zero_()

    def update(self, obs_batch):
        """最新の観測値で履歴を更新"""
        t_obs = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        t_obs = t_obs.view(self.n_envs, self.obs_dim)

        # 巡回ダブルバッファへの書き込み
        self.buffer[:, self.ptr] = t_obs
        self.buffer[:, self.ptr + self.seq_len] = t_obs

        # インデックスの更新
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        """過去 SEQ_LEN 分の系列データを取得。形状: (batch, seq, obs)"""
        return self.buffer[:, self.ptr : self.ptr + self.seq_len]


def make_env():
    """環境生成のファクトリ関数。描画設定を物理環境に伝播させる。"""
    render = "human" if USE_VIEWER else None
    return TeamCosEnv(mode=MODE, render_mode=render)


def run():
    # 計算デバイスの決定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_debug(f"Device Initialized: {device}")

    # 割り込み信号（Ctrl+C）のハンドラ
    def signal_handler(sig, frame):
        log_debug("Shutdown command received.")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # モデルの初期化
    act_dim = 2 if MODE == "initial" else 4
    agent = AgentV2(53, act_dim, HIDDEN_DIM, SEQ_LEN).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # 重みのロード
    if os.path.exists(SAVE_PATH):
        try:
            agent.load_state_dict(torch.load(SAVE_PATH, map_location=device))
            log_debug(f"Weight Loaded: {SAVE_PATH}")
        except Exception as e:
            log_debug(f"Load Error: {e}. Re-initializing weights.")

    if not TRAIN_MODE:
        # ==========================================
        # 🎮 実時間再生モード (Playback Mode)
        # ==========================================
        env = make_env()
        hist = ObsHistory(1, SEQ_LEN, 53, device)

        obs, _ = env.reset()
        hist.update(obs)
        agent.eval()

        log_debug("Playback running. Synchronized at 40Hz (0.025s/step).")
        try:
            # Viewerが開いている間ループ
            while env.viewer is not None and env.viewer.is_running():
                t_loop_start = time.perf_counter()

                # 推論実行
                with torch.no_grad():
                    context = hist.get()
                    action, _, _, _ = agent.get_action_and_value(context)

                # 環境ステップ実行 (0.025s)
                obs_next, reward, term, trunc, _ = env.step(action.cpu().numpy().flatten())
                hist.update(obs_next)

                # エピソード終了判定
                if term or trunc:
                    obs_next, _ = env.reset()
                    hist.reset()
                    hist.update(obs_next)

                # 💡 重要：物理5ステップ（0.025秒）と制御を同期
                # これにより 9000N ギア環境下でのフィードバック遅延による暴走を防ぐ
                elapsed = time.perf_counter() - t_loop_start
                sleep_t = 0.025 - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    time.sleep(0.001)

        finally:
            log_debug("Releasing simulation resources...")
            env.close()
            sys.exit(0)

    else:
        # ==========================================
        # 🚀 強化学習モード (PPO Training Mode)
        # ==========================================
        num_envs = 1 if USE_VIEWER else 16
        envs = gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])

        # TensorBoard ログ
        t_stamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(f"runs/{EXPERIMENT_NAME}_{t_stamp_str}")

        # Wandb 連携
        if TRACK_WANDB:
            wandb.init(
                project="HideAndSeek_GTRPPO",
                name=f"{EXPERIMENT_NAME}_{t_stamp_str}",
                config={
                    "lr": LR,
                    "batch_size": BATCH_SIZE,
                    "num_envs": num_envs,
                    "seq_len": SEQ_LEN,
                    "hidden_dim": HIDDEN_DIM,
                    "mode": MODE,
                },
            )
            wandb.watch(agent, log="gradients", log_freq=100)

        # ロールアウトバッファ
        storage_obs = torch.zeros((EPISODE_LIMIT, num_envs, SEQ_LEN, 53), device=device)
        storage_actions = torch.zeros((EPISODE_LIMIT, num_envs, act_dim), device=device)
        storage_logprobs = torch.zeros((EPISODE_LIMIT, num_envs), device=device)
        storage_rewards = torch.zeros((EPISODE_LIMIT, num_envs), device=device)
        storage_dones = torch.zeros((EPISODE_LIMIT, num_envs), device=device)
        storage_values = torch.zeros((EPISODE_LIMIT, num_envs), device=device)

        hist = ObsHistory(num_envs, SEQ_LEN, 53, device)

        # 最初の観測
        next_obs_init, _ = envs.reset()
        hist.update(next_obs_init)

        global_step_count = 0
        train_start_t = time.time()
        num_ppo_updates = TOTAL_STEPS // (EPISODE_LIMIT * num_envs)

        update_bar = tqdm(range(1, num_ppo_updates + 1), desc="PPO Cycle")

        try:
            for update in update_bar:
                # --- [1] データ収集 (Rollout) ---
                agent.eval()
                rollout_bar = tqdm(range(EPISODE_LIMIT), desc=f"Update {update} Rollout", leave=False)

                for step in rollout_bar:
                    global_step_count += num_envs

                    # 履歴の保存
                    current_ctx = hist.get()
                    storage_obs[step] = current_ctx

                    # 意思決定
                    with torch.no_grad():
                        action, logp, entropy, value = agent.get_action_and_value(current_ctx)
                        storage_values[step] = value.flatten()

                    storage_actions[step] = action
                    storage_logprobs[step] = logp

                    # Viewer終了検知
                    if USE_VIEWER:
                        if envs.envs[0].viewer is not None and not envs.envs[0].viewer.is_running():
                            log_debug("Viewer closed. Terminating.")
                            raise SystemExit

                    # 物理エンジン駆動
                    next_obs_batch, reward_batch, terms, truncs, _ = envs.step(action.cpu().numpy())

                    storage_rewards[step] = torch.as_tensor(reward_batch, device=device)
                    done_m = np.logical_or(terms, truncs)
                    storage_dones[step] = torch.as_tensor(done_m, dtype=torch.float32, device=device)

                    # 履歴更新
                    hist.update(next_obs_batch)

                    # 終了環境のリセット
                    for i, d in enumerate(done_m):
                        if d:
                            hist.reset(i)
                            hist.update(next_obs_batch[i])

                # --- [2] アドバンテージ計算 (GAE) ---
                with torch.no_grad():
                    # ブートストラップ用
                    next_val_pred = agent.get_value(hist.get()).reshape(1, -1)
                    advantages = torch.zeros_like(storage_rewards, device=device)
                    last_gae_lam = 0

                    for t in reversed(range(EPISODE_LIMIT)):
                        if t == EPISODE_LIMIT - 1:
                            next_non_term = 1.0 - storage_dones[t]
                            next_val_p = next_val_pred
                        else:
                            next_non_term = 1.0 - storage_dones[t]
                            next_val_p = storage_values[t + 1]

                        # TD誤差
                        delta = storage_rewards[t] + GAMMA * next_val_p * next_non_term - storage_values[t]
                        # GAE再帰更新
                        last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_term * last_gae_lam
                        advantages[t] = last_gae_lam

                    returns = advantages + storage_values

                # --- [3] 学習更新 (Learning) ---
                agent.train()
                # データの平滑化
                b_obs = storage_obs.reshape(-1, SEQ_LEN, 53)
                b_logp = storage_logprobs.reshape(-1)
                b_acts = storage_actions.reshape(-1, act_dim)
                b_advs = advantages.reshape(-1)
                b_rets = returns.reshape(-1)
                b_vals = storage_values.reshape(-1)

                indices = np.arange(len(b_obs))
                for epoch in range(UPDATE_EPOCHS):
                    np.random.shuffle(indices)
                    for start in range(0, len(b_obs), BATCH_SIZE):
                        mb_idx = indices[start : start + BATCH_SIZE]

                        # 最新方策で再評価
                        _, new_lp, ent, new_val = agent.get_action_and_value(b_obs[mb_idx], b_acts[mb_idx])

                        # 比率算出
                        ratio = (new_lp - b_logp[mb_idx]).exp()

                        # アドバンテージ正規化
                        mb_adv = b_advs[mb_idx]
                        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                        # PPO Surrogate Loss
                        pg_loss1 = -mb_adv * ratio
                        pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                        policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                        # Value Loss
                        v_loss = 0.5 * ((new_val.flatten() - b_rets[mb_idx]) ** 2).mean()

                        # 合計損失
                        loss_total = policy_loss - ENT_COEF * ent.mean() + v_loss * VF_COEF

                        # 勾配更新
                        optimizer.zero_grad()
                        loss_total.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                        optimizer.step()

                # --- [4] 出力と保存 ---
                avg_rew_val = storage_rewards.mean().item()
                curr_t_val = time.time()
                total_elap = curr_t_val - train_start_t
                sps_curr = int(global_step_count / total_elap)

                update_bar.set_postfix(reward=f"{avg_rew_val:.3f}", SPS=sps_curr)

                writer.add_scalar("charts/avg_reward", avg_rew_val, global_step_count)
                writer.add_scalar("charts/SPS", sps_curr, global_step_count)
                writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step_count)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step_count)

                if TRACK_WANDB:
                    wandb.log(
                        {
                            "reward": avg_rew_val,
                            "sps": sps_curr,
                            "policy_loss": policy_loss.item(),
                            "value_loss": v_loss.item(),
                            "entropy": ent.mean().item(),
                        },
                        step=global_step_count,
                    )

                if update % 50 == 0:
                    torch.save(agent.state_dict(), SAVE_PATH)
                    tqdm.write(f"[{datetime.now().strftime('%H:%M:%S')}] 💾 Model Saved: {SAVE_PATH}")

        except SystemExit:
            log_debug("Simulation Interrupted.")
        except KeyboardInterrupt:
            log_debug("Manual Stop.")
        finally:
            log_debug("Finalizing processes...")
            envs.close()
            writer.close()
            if TRACK_WANDB:
                wandb.finish()
            sys.exit(0)


if __name__ == "__main__":
    run()
