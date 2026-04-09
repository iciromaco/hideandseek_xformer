import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

# === 環境の定義 ===
from hidenadseek import HideAndSeekEnv  # ← 実際のモジュール名に置き換えてね！

def make_env(rank, seed=0):
    def _init():
        env = HideAndSeekEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + rank)
        return env
    return _init

# === Transformerベースのポリシー ===
class TransformerPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, embed_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.embed = nn.Linear(obs_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.mean = nn.Linear(embed_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # 学習可能な分散

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x.unsqueeze(1)).squeeze(1)
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

from gymnasium.vector import SyncVectorEnv
# === メイン ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    envs = AsyncVectorEnv([make_env(i, seed=args.seed) for i in range(args.num_envs)])
    # envs = SyncVectorEnv([make_env(i, seed=args.seed) for i in range(args.num_envs)])
    obs_shape = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape[0]

    policy = TransformerPolicy(obs_dim=obs_shape[0], act_dim=act_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2.5e-4)
    writer = SummaryWriter()

    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    global_step = 0

    for step in range(args.total_timesteps):
        with torch.no_grad():
            mean, std = policy(obs)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action_clipped = torch.clamp(action, -1.0, 1.0)  # 必要なら範囲を制限

        next_obs, reward, terminated, truncated, info = envs.step(action_clipped.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # ログ出力（任意）
        if "episode" in info:
            for r in info["episode"]["r"]:
                if r != 0:
                    writer.add_scalar("charts/episodic_return", float(r), global_step)
        global_step += 1

    envs.close()
    writer.close()

if __name__ == "__main__":
    main()
'''
def main():
    print(f"--- Multi-Agent Training Stage: Target = {TRAIN_TARGET} ---")
    
    if TRAIN_MODE:
        # ★修正(v79): USE_VIEWER=True時はDummyVecEnv(直列・描画可)を使用するよう分岐
        if USE_VIEWER:
            env = DummyVecEnv([make_env(i, render_mode="human") for i in range(N_ENVS)])
        else:
            env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
        
        if os.path.exists(SAVE_PATH + ".zip"):
            print(f"Loading existing model: {SAVE_PATH}")
            model = RecurrentPPO.load(SAVE_PATH, env=env, device="cpu")
        else:
            print("Creating new model...")
            model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, 
                                 n_steps=2048, batch_size=128, ent_coef=ENT_COEF, 
                                 policy_kwargs=dict(lstm_hidden_size=LSTM_HIDDEN_SIZE))

 def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---


    # ---
    envs = AsyncVectorEnv([make_env(i, seed=args.seed) for i in range(args.num_envs)])
    # envs = SyncVectorEnv([make_env(i, seed=args.seed) for i in range(args.num_envs)])
    obs_shape = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape[0]

    policy = TransformerPolicy(obs_dim=obs_shape[0], act_dim=act_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2.5e-4)
    writer = SummaryWriter()

    obs, _ = envs.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    global_step = 0

    for step in range(args.total_timesteps):
        with torch.no_grad():
            mean, std = policy(obs)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            action_clipped = torch.clamp(action, -1.0, 1.0)  # 必要なら範囲を制限

        next_obs, reward, terminated, truncated, info = envs.step(action_clipped.cpu().numpy())
        done = np.logical_or(terminated, truncated)
        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # ログ出力（任意）
        if "episode" in info:
            for r in info["episode"]["r"]:
                if r != 0:
                    writer.add_scalar("charts/episodic_return", float(r), global_step)
        global_step += 1

    envs.close()
    writer.close()       


main(SAVE_PATH=SAVE_PATH, ...)'''