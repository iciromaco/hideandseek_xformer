"""ppo_transformer_v2.py - legacy AgentV2 (restored)

This module provides the legacy AgentV2 implementation expected by main27/main28
codepaths. It uses a single continuous actor head (mean + learnable logstd)
with Tanh squashing, and a critic head for value estimation.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


LAYER_INIT_STD_DEFAULT = np.sqrt(2.0)


def layer_init(layer, std=LAYER_INIT_STD_DEFAULT, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentV2(nn.Module):
    """Legacy AgentV2: single continuous actor head with Tanh squashing and a learnable logstd.
    Actions: continuous vector of size `action_dim`, squashed with Tanh.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, seq_len):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # encoder
        self.obs_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

        # simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # actor mean head
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
            nn.Tanh(),
        )

        # a single learnable logstd per action dim (legacy style)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        batch_size = x.shape[0]
        flat_x = x.reshape(-1, self.obs_dim)
        encoded = self.obs_encoder(flat_x)
        encoded_sequence = encoded.reshape(batch_size, self.seq_len, self.hidden_dim)
        context = self.transformer(encoded_sequence)
        last_step_context = context[:, -1, :]
        value = self.critic(last_step_context)
        return value

    def get_action_and_value(self, x, action=None):
        batch_size = x.shape[0]
        flat_x_eval = x.reshape(-1, self.obs_dim)
        encoded_eval = self.obs_encoder(flat_x_eval)
        encoded_eval_seq = encoded_eval.reshape(batch_size, self.seq_len, self.hidden_dim)
        context_eval = self.transformer(encoded_eval_seq)
        latent_eval = context_eval[:, -1, :]

        mean = self.actor_mean(latent_eval)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)

        if action is None:
            try:
                sampled = dist.rsample()
            except Exception:
                sampled = dist.sample()
            action = torch.tanh(sampled)

        # For log_prob we need to map the action back through atanh (inverse tanh)
        # Clip to avoid numerical issues
        atanh_input = torch.clamp(action, -0.999999, 0.999999)
        pre_tanh = 0.5 * torch.log((1 + atanh_input) / (1 - atanh_input))
        log_prob = dist.log_prob(pre_tanh).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        value = self.critic(latent_eval)
        return action, log_prob, entropy, value

    def get_deterministic_action_and_value(self, x):
        batch_size = x.shape[0]
        flat_x_eval = x.reshape(-1, self.obs_dim)
        encoded_eval = self.obs_encoder(flat_x_eval)
        encoded_eval_seq = encoded_eval.reshape(batch_size, self.seq_len, self.hidden_dim)
        context_eval = self.transformer(encoded_eval_seq)
        latent_eval = context_eval[:, -1, :]

        mean = self.actor_mean(latent_eval)
        action = mean
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(mean).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        value = self.critic(latent_eval)
        return action, log_prob, entropy, value