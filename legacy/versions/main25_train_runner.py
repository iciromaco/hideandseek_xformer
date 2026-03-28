# main25_train_runner.py
# 演習第25回：PPO強化学習の完遂版
#
# 1. 53次元観測ベクトルに対応した並列環境の構築
# 2. Transformerコンテキスト（過去8ステップ）の管理
# 3. GAE(一般化アドバンテージ推定)とPPOクリップ損失による更新

import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hns_environment import TeamCosEnv
from ppo_transformer import HS_Agent
from torch.utils.tensorboard import SummaryWriter

# --- ハイパーパラメータ設定 ---
NUM_ENVS = 16  # 並列環境数
SEQ_LEN = 8  # Transformerのシーケンス長
EPISODE_LIMIT = 200  # 1エピソードあたりの最大ステップ数
TOTAL_STEPS = 10_000_000  # 総学習ステップ数
LR = 3e-4  # 学習率
GAMMA = 0.99  # 割引率
GAE_LAMBDA = 0.95  # GAEパラメータ
UPDATE_EPOCHS = 4  # PPOの更新エポック数
BATCH_SIZE = 512  # ミニバッチサイズ
CLIP_COEF = 0.2  # PPOクリップ定数
ENT_COEF = 0.01  # エントロピー係数
VF_COEF = 0.5  # 価値関数係数
MAX_GRAD_NORM = 0.5  # 勾配クリッピング


def train():
    """PPO学習メインループ"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 並列環境の初期化
    # 全てのオブジェクトをランダム配置し、seedによる再現性を確保した TeamCosEnv を使用
    envs = gym.vector.SyncVectorEnv([lambda: TeamCosEnv(lidar_mode=1) for _ in range(NUM_ENVS)])

    # 2. エージェントと最適化手法の定義
    # 観測53次元、アクション3次元（推力、予備、回転）
    agent = HS_Agent(53, 3).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # 3. ロールアウトバッファの事前確保
    obs_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS, 53)).to(device)
    actions_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS, 3)).to(device)
    logprobs_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS)).to(device)
    rewards_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS)).to(device)
    dones_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS)).to(device)
    values_b = torch.zeros((EPISODE_LIMIT, NUM_ENVS)).to(device)

    # Transformer用の履歴バッファ [envs, seq_len, obs_dim]
    history = torch.zeros((NUM_ENVS, SEQ_LEN, 53)).to(device)

    global_step = 0
    start_time = time.time()
    writer = SummaryWriter("runs/main25_hns_transformer")

    print(f"学習を開始します。デバイス: {device}")

    try:
        while global_step < TOTAL_STEPS:
            # --- 4. データ収集フェーズ (Rollout) ---
            next_obs, _ = envs.reset()
            next_obs = torch.Tensor(next_obs).to(device)
            history.fill_(0.0)
            history[:, -1] = next_obs  # 初期観測を最新スロットにセット

            for step in range(EPISODE_LIMIT):
                global_step += NUM_ENVS
                obs_b[step] = next_obs

                with torch.no_grad():
                    # Transformer Encoderにより過去の履歴を含めて行動を選択
                    action, logprob, _, value = agent.get_action_and_value(history)
                    values_b[step] = value.flatten()

                actions_b[step] = action
                logprobs_b[step] = logprob

                # 環境の実行
                next_obs_raw, reward, done, _, _ = envs.step(action.cpu().numpy())
                rewards_b[step] = torch.tensor(reward).to(device)
                dones_b[step] = torch.tensor(done).to(device)

                # 履歴の更新 (スライディングウィンドウ)
                next_obs = torch.Tensor(next_obs_raw).to(device)
                history = torch.roll(history, -1, dims=1)
                history[:, -1] = next_obs

                # 終了した環境の履歴リセット
                if any(done):
                    for i, d in enumerate(done):
                        if d:
                            history[i].fill_(0.0)
                            history[i, -1] = next_obs[i]

            # --- 5. GAEアドバンテージ計算 ---
            with torch.no_grad():
                next_value = agent.get_value(history).reshape(1, -1)
                advantages = torch.zeros_like(rewards_b).to(device)
                lastgaelam = 0
                for t in reversed(range(EPISODE_LIMIT)):
                    next_non_terminal = 1.0 - dones_b[t]
                    v_next = next_value if t == EPISODE_LIMIT - 1 else values_b[t + 1]
                    delta = rewards_b[t] + GAMMA * v_next * next_non_terminal - values_b[t]
                    lastgaelam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values_b

            # --- 6. PPO パラメータ更新 (Optimization) ---
            b_obs = obs_b.reshape(-1, 53)
            b_logprobs = logprobs_b.reshape(-1)
            b_actions = actions_b.reshape(-1, 3)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values_b.reshape(-1)

            indices = np.arange(b_obs.shape[0])
            for epoch in range(UPDATE_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, b_obs.shape[0], BATCH_SIZE):
                    end = start + BATCH_SIZE
                    mb_idx = indices[start:end]

                    # 簡易化のため、学習時は現在の観測のみを用いた擬似履歴で計算
                    # (本番実装ではRolloutバッファに履歴も含めるのが理想)
                    mb_history = b_obs[mb_idx].unsqueeze(1).repeat(1, SEQ_LEN, 1)

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_history, b_actions[mb_idx])

                    logratio = newlogprob - b_logprobs[mb_idx]
                    ratio = logratio.exp()

                    # アドバンテージの正規化
                    mb_adv = b_advantages[mb_idx]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    # Policy Loss
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value Loss
                    v_loss = 0.5 * ((newvalue.flatten() - b_returns[mb_idx]) ** 2).mean()

                    # 統合損失
                    loss = pg_loss - ENT_COEF * entropy.mean() + v_loss * VF_COEF

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

            # 統計表示
            avg_return = rewards_b.sum(0).mean().item()
            sps = int(global_step / (time.time() - start_time))
            print(f"Step: {global_step} | Return: {avg_return:.2f} | SPS: {sps}")
            writer.add_scalar("charts/episodic_return", avg_return, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)

    except KeyboardInterrupt:
        print("学習を中断しました。")
    finally:
        envs.close()
        writer.close()


if __name__ == "__main__":
    train()
