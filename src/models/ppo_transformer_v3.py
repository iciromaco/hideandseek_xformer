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


# stable atanh for tensors
def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


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
        if flat_x.shape[1] != self.obs_dim:
            raise RuntimeError(f"[SHAPE_ERR] expected obs dim {self.obs_dim}, got {flat_x.shape}")
        encoded = self.obs_encoder(flat_x)
        if encoded.shape[1] != self.hidden_dim:
            raise RuntimeError(f"[SHAPE_ERR] expected encoded dim {self.hidden_dim}, got {encoded.shape}")
        encoded_sequence = encoded.reshape(batch_size, self.seq_len, self.hidden_dim)
        context = self.transformer(encoded_sequence)
        last_step_context = context[:, -1, :]
        # check for NaN/Inf in value path
        if torch.isnan(last_step_context).any() or torch.isinf(last_step_context).any():
            print(f"[NAN_DBG_VALUE] last_step_context contains NaN/Inf; shape={last_step_context.shape}")
            raise RuntimeError("NaN/Inf detected in critic latent")
        value = self.critic(last_step_context)
        return value

    def get_action_and_value(self, x, action=None):
        batch_size = x.shape[0]
        flat_x_eval = x.reshape(-1, self.obs_dim)
        if flat_x_eval.shape[1] != self.obs_dim:
            raise RuntimeError(f"[SHAPE_ERR] expected obs dim {self.obs_dim}, got {flat_x_eval.shape}")
        encoded_eval = self.obs_encoder(flat_x_eval)
        # layer-wise checks
        if torch.isnan(encoded_eval).any() or torch.isinf(encoded_eval).any():
            print(f"[NAN_DBG] encoded_eval contains NaN/Inf; shape={encoded_eval.shape}")
            try:
                print("flat_x_eval[0,:10] =", flat_x_eval[0, :10].detach().cpu().numpy())
            except Exception:
                pass
            raise RuntimeError("NaN/Inf detected in encoder output")
        encoded_eval_seq = encoded_eval.reshape(batch_size, self.seq_len, self.hidden_dim)
        context_eval = self.transformer(encoded_eval_seq)
        if torch.isnan(context_eval).any() or torch.isinf(context_eval).any():
            print(f"[NAN_DBG] context_eval contains NaN/Inf; shape={context_eval.shape}")
            try:
                print("encoded_eval[0,:10] =", encoded_eval[0, :10].detach().cpu().numpy())
            except Exception:
                pass
            raise RuntimeError("NaN/Inf detected in transformer output")
        latent_eval = context_eval[:, -1, :]
        if torch.isnan(latent_eval).any() or torch.isinf(latent_eval).any():
            print(f"[NAN_DBG] latent_eval contains NaN/Inf; shape={latent_eval.shape}")
            try:
                print("context_eval[0, -1, :10] =", context_eval[0, -1, :10].detach().cpu().numpy())
            except Exception:
                pass
            raise RuntimeError("NaN/Inf detected in latent representation")

        cont_out = self.actor_continuous(latent_eval)
        if torch.isnan(cont_out).any() or torch.isinf(cont_out).any():
            print(f"[NAN_DBG] cont_out contains NaN/Inf; shape={cont_out.shape}")
            try:
                print("latent_eval[0,:10] =", latent_eval[0, :10].detach().cpu().numpy())
            except Exception:
                pass
            raise RuntimeError("NaN/Inf detected in continuous actor output")
        cont_mean = cont_out[:, 0:2]
        cont_logstd = torch.clamp(cont_out[:, 2:4], -5.0, 2.0)
        cont_std = torch.exp(cont_logstd)
        if torch.isnan(cont_mean).any() or torch.isnan(cont_std).any() or torch.isinf(cont_mean).any() or torch.isinf(cont_std).any():
            print(
                f"[NAN_DBG] cont_mean_nan={torch.isnan(cont_mean).any().item()} cont_std_nan={torch.isnan(cont_std).any().item()}",
                f"cont_mean_inf={torch.isinf(cont_mean).any().item()} cont_std_inf={torch.isinf(cont_std).any().item()}",
            )
            try:
                print("cont_out[0,:] =", cont_out[0, :].detach().cpu().numpy())
            except Exception:
                pass
            raise RuntimeError("NaN/Inf detected in continuous distribution parameters")

        # Debugging: detect NaNs early to locate source (obs / encoder / transformer)
        if torch.isnan(cont_mean).any() or torch.isnan(cont_std).any():
            try:
                has_nan_flat = torch.isnan(flat_x_eval).any().item()
            except Exception:
                has_nan_flat = None
            try:
                has_nan_encoded = torch.isnan(encoded_eval).any().item()
            except Exception:
                has_nan_encoded = None
            try:
                has_nan_latent = torch.isnan(latent_eval).any().item()
            except Exception:
                has_nan_latent = None
            print(
                f"[NAN_DBG] cont_mean_nan={torch.isnan(cont_mean).any().item()} cont_std_nan={torch.isnan(cont_std).any().item()}",
                f"flat_nan={has_nan_flat} encoded_nan={has_nan_encoded} latent_nan={has_nan_latent}",
            )
            # print small diagnostics (first element only to reduce log)
            try:
                print("flat_x_eval[0,:10] =", flat_x_eval[0, :10].detach().cpu().numpy())
                print("encoded_eval[0,:10] =", encoded_eval[0, :10].detach().cpu().numpy())
                print("latent_eval[0,:10] =", latent_eval[0, :10].detach().cpu().numpy())
            except Exception:
                pass
            raise RuntimeError("NaN detected in AgentV3 policy tensors; see [NAN_DBG] logs")

        base_cont = Normal(cont_mean, cont_std)
        cont_dist = TransformedDistribution(base_cont, [TanhTransform(cache_size=1)])

        disc_logits = self.actor_discrete(latent_eval)
        if torch.isnan(disc_logits).any() or torch.isinf(disc_logits).any():
            print(f"[NAN_DBG] disc_logits contains NaN/Inf; shape={disc_logits.shape}")
            try:
                print("latent_eval[0,:10] =", latent_eval[0, :10].detach().cpu().numpy())
            except Exception:
                pass
            raise RuntimeError("NaN/Inf detected in discrete actor logits")
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

        # compute log-prob for Tanh-transformed Normal in a numerically stable way
        eps = 1e-6
        ac_clamped = action_cont.clamp(-1.0 + eps, 1.0 - eps)
        atanh_act = atanh(ac_clamped)
        log_base = base_cont.log_prob(atanh_act).sum(dim=1)
        log_det = -torch.log(1.0 - ac_clamped * ac_clamped + eps).sum(dim=1)
        logp_cont = log_base + log_det
        logp_disc = disc_dist.log_prob(action_disc).sum(dim=1)
        log_prob_sum = logp_cont + logp_disc

        entropy_cont = base_cont.entropy().sum(dim=1)
        entropy_disc = disc_dist.entropy().sum(dim=1)
        entropy_sum = entropy_cont + entropy_disc

        value = self.critic(latent_eval)
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"[NAN_DBG] critic value contains NaN/Inf; shape={value.shape}")
            raise RuntimeError("NaN/Inf detected in critic value output")
        return action, log_prob_sum, entropy_sum, value

    def get_deterministic_action_and_value(self, x):
        batch_size = x.shape[0]
        flat_x_eval = x.reshape(-1, self.obs_dim)
        if flat_x_eval.shape[1] != self.obs_dim:
            raise RuntimeError(f"[SHAPE_ERR] expected obs dim {self.obs_dim}, got {flat_x_eval.shape}")
        encoded_eval = self.obs_encoder(flat_x_eval)
        if torch.isnan(encoded_eval).any() or torch.isinf(encoded_eval).any():
            print(f"[NAN_DBG] encoded_eval (det) contains NaN/Inf; shape={encoded_eval.shape}")
            raise RuntimeError("NaN/Inf detected in encoder output (deterministic)")
        encoded_eval_seq = encoded_eval.reshape(batch_size, self.seq_len, self.hidden_dim)
        context_eval = self.transformer(encoded_eval_seq)
        if torch.isnan(context_eval).any() or torch.isinf(context_eval).any():
            print(f"[NAN_DBG] context_eval (det) contains NaN/Inf; shape={context_eval.shape}")
            raise RuntimeError("NaN/Inf detected in transformer output (deterministic)")
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

        # deterministic path: stable log_prob for tanh-inverse with clamping
        eps = 1e-6
        ac_clamped = action_cont.clamp(-1.0 + eps, 1.0 - eps)
        atanh_act = atanh(ac_clamped)
        log_base = Normal(cont_mean, cont_std).log_prob(atanh_act).sum(dim=1)
        log_det = -torch.log(1.0 - ac_clamped * ac_clamped + eps).sum(dim=1)
        logp_cont = log_base + log_det
        logp_disc = disc_dist.log_prob(action_disc).sum(dim=1)
        log_prob = logp_cont + logp_disc

        entropy_cont = base_cont.entropy().sum(dim=1)
        entropy_disc = disc_dist.entropy().sum(dim=1)
        entropy = entropy_cont + entropy_disc
        value = self.critic(latent_eval)
        if torch.isnan(value).any() or torch.isinf(value).any():
            print(f"[NAN_DBG] critic value (det) contains NaN/Inf; shape={value.shape}")
            raise RuntimeError("NaN/Inf detected in critic value output (deterministic)")
        return action, log_prob, entropy, value
