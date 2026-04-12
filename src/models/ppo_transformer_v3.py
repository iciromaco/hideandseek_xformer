# ppo_transformer_v3.py
# Continuous/Discrete separated actor (Version 3)
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Bernoulli, TransformedDistribution
from torch.distributions.transforms import TanhTransform

LAYER_INIT_STD_DEFAULT = np.sqrt(2.0)


def layer_init(layer, std=LAYER_INIT_STD_DEFAULT, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentV3(nn.Module):
    """Transformer backbone PPO agent with separated continuous and discrete heads.
    Actions: [forward (cont), steer (cont), grab (bin), lock (bin)].
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

        # transformer
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

        # continuous actor head: outputs 2 means + 2 logstd
        self.actor_continuous = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 4), std=0.01),
        )

        # discrete actor head: outputs 2 logits for Bernoulli
        self.actor_discrete = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 2), std=0.01),
        )

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

        cont_out = self.actor_continuous(latent_eval)
        cont_mean = cont_out[:, 0:2]
        cont_logstd = torch.clamp(cont_out[:, 2:4], -5.0, 2.0)
        cont_std = torch.exp(cont_logstd)

        base_cont = Normal(cont_mean, cont_std)
        cont_dist = TransformedDistribution(base_cont, [TanhTransform(cache_size=1)])

        disc_logits = self.actor_discrete(latent_eval)
        disc_dist = Bernoulli(logits=disc_logits)

        if action is None:
            try:
                action_cont = cont_dist.rsample()
            except Exception:
                action_cont = cont_dist.sample()
            action_disc = disc_dist.sample()
            action = torch.cat([action_cont, action_disc], dim=1)

        action_cont = action[:, 0:2]
        action_disc = action[:, 2:4]

        logp_cont = cont_dist.log_prob(action_cont).sum(dim=1)
        logp_disc = disc_dist.log_prob(action_disc).sum(dim=1)
        log_prob_sum = logp_cont + logp_disc

        entropy_cont = base_cont.entropy().sum(dim=1)
        entropy_disc = disc_dist.entropy().sum(dim=1)
        entropy_sum = entropy_cont + entropy_disc

        value = self.critic(latent_eval)
        return action, log_prob_sum, entropy_sum, value

    def get_deterministic_action_and_value(self, x):
        batch_size = x.shape[0]
        flat_x_eval = x.reshape(-1, self.obs_dim)
        encoded_eval = self.obs_encoder(flat_x_eval)
        encoded_eval_seq = encoded_eval.reshape(batch_size, self.seq_len, self.hidden_dim)
        context_eval = self.transformer(encoded_eval_seq)
        latent_eval = context_eval[:, -1, :]

        cont_out = self.actor_continuous(latent_eval)
        cont_mean = cont_out[:, 0:2]
        action_cont = torch.tanh(cont_mean)

        disc_logits = self.actor_discrete(latent_eval)
        disc_probs = torch.sigmoid(disc_logits)
        action_disc = (disc_probs >= 0.5).float()

        action = torch.cat([action_cont, action_disc], dim=1)

        cont_logstd = torch.clamp(cont_out[:, 2:4], -5.0, 2.0)
        cont_std = torch.exp(cont_logstd)
        base_cont = Normal(cont_mean, cont_std)
        cont_dist = TransformedDistribution(base_cont, [TanhTransform(cache_size=1)])
        disc_dist = Bernoulli(logits=disc_logits)

        logp_cont = cont_dist.log_prob(action_cont).sum(dim=1)
        logp_disc = disc_dist.log_prob(action_disc).sum(dim=1)
        log_prob = logp_cont + logp_disc

        entropy_cont = base_cont.entropy().sum(dim=1)
        entropy_disc = disc_dist.entropy().sum(dim=1)
        entropy = entropy_cont + entropy_disc
        value = self.critic(latent_eval)
        return action, log_prob, entropy, value
