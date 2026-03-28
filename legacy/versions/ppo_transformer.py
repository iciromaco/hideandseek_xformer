# ppo_transformer.py
# Transformerを用いた Actor-Critic ネットワーク
#
# 過去の履歴（シーケンス）から敵の動きや自分の位置関係の文脈を読み取ります。

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """重みの初期化。直交初期化は勾配消失を防ぎ、学習を安定させます。"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HS_Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.obs_to_hidden = nn.Linear(obs_dim, hidden_dim)

        # Transformer Encoder: 各ステップ間の関係性を抽出する
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=2,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Actor: 行動（平均値）を出力。最後は tanh で [-1, 1] に制限。
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        # 行動の分散（logスケール）。学習可能なパラメータ。
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic: 現在の状態の価値（スカラー）を推定。
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def get_value(self, x):
        """シーケンスの最後（最新）の価値を推定します。"""
        h = torch.relu(self.obs_to_hidden(x))
        h = self.transformer(h)
        return self.critic(h[:, -1, :])

    def get_action_and_value(self, x, action=None):
        """
        行動のサンプリング、対数確率、エントロピー、および価値を一度に計算します。
        """
        h = torch.relu(self.obs_to_hidden(x))
        h = self.transformer(h)
        h_last = h[:, -1, :]  # 最新のタイムステップのみを使用

        mean = self.actor_mean(h_last)
        std = torch.exp(self.actor_logstd)
        probs = Normal(mean, std)

        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action).sum(-1),
            probs.entropy().sum(-1),
            self.critic(h_last),
        )
