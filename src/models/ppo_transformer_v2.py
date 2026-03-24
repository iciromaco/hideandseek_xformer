# ppo_transformer_v2.py v2.16.1
# 演習第26回：【構文エラー是正版】言語タグを修正し、行動飽和制御（Tanh）を維持
#
# 修正内容:
# 1. 構文エラーの解消: 前回のブロック形式によるコンパイルエラーを修正するため、python形式で再生成。
# 2. 処理の完全展開: 1行1命令の原則を遵守し、可読性を確保。
# 3. 飽和活性化（Tanh）: actor_mean の終端に nn.Tanh() を適用し、[-1, 1] の出力を保証。
# 4. Transformer構成: PyTorch標準の TransformerEncoderLayer を使用し、時系列コンテキストを処理。

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn as nn

LAYER_INIT_STD_DEFAULT = np.sqrt(2.0)


def layer_init(layer, std=LAYER_INIT_STD_DEFAULT, bias_const=0.0):
    """
    層の重みを直交初期化し、バイアスを定数で初期化する補助関数。
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AgentV2(nn.Module):
    """
    Transformerをバックボーンに持つPPOエージェント。
    [移動, 旋回, ロック, 掴み] の多次元アクションを制御。
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, seq_len):
        super(AgentV2, self).__init__()

        # ハイパーパラメータの保持
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # --- [1] 入力エンコーディング層 ---
        self.obs_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

        # --- [2] Transformer バックボーン ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- [3] 状態価値（Critic）ヘッド ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # --- [4] 方策（Actor）ヘッド ---
        # 出力を [-1, 1] に制限することで、環境側の 0.5 閾値判定との整合性を高める
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
            nn.Tanh(),
        )

        # 学習可能な分散パラメータ
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """
        状態価値 V(s) を算出。GAE計算に使用。
        """
        batch_size = x.shape[0]
        # 1. 観測値の形状変換（フラット化）
        flat_x = x.reshape(-1, self.obs_dim)
        # 2. 特徴量抽出
        encoded = self.obs_encoder(flat_x)
        # 3. シーケンス形状へ復元 (batch, seq, hidden)
        encoded_sequence = encoded.reshape(batch_size, self.seq_len, self.hidden_dim)
        # 4. Transformerによるコンテキスト解析
        context = self.transformer(encoded_sequence)
        # 5. 最終ステップ（最新状態）のコンテキストを抽出
        last_step_context = context[:, -1, :]
        # 6. 状態価値の算出
        value = self.critic(last_step_context)
        return value

    def get_action_and_value(self, x, action=None):
        """
        行動サンプリング、対数確率、エントロピー、および状態価値を算出。
        """
        batch_size = x.shape[0]

        # 1. 共通バックボーン演算
        flat_x_eval = x.reshape(-1, self.obs_dim)
        encoded_eval = self.obs_encoder(flat_x_eval)
        encoded_eval_seq = encoded_eval.reshape(batch_size, self.seq_len, self.hidden_dim)
        context_eval = self.transformer(encoded_eval_seq)
        latent_eval = context_eval[:, -1, :]

        # 2. 行動平均値の算出 (Tanh活性化済み)
        action_mean = self.actor_mean(latent_eval)

        # 3. 行動分散の算出
        # 数値的安定性のために logstd をクランプ
        logstd_clamped = torch.clamp(self.actor_logstd, -3.0, 2.0)
        action_logstd_eval = logstd_clamped.expand_as(action_mean)
        action_std_eval = torch.exp(action_logstd_eval)

        # 4. 正規分布に基づく確率分布の構築
        probs = distributions.Normal(action_mean, action_std_eval)

        # 5. 行動の決定
        if action is None:
            # 学習時はサンプリング、推論時は mean を使うなどの使い分けが可能だが、
            # PPOの標準に従いここではサンプリングを行う
            action = probs.sample()

        # 6. 各種統計量の計算
        log_prob = probs.log_prob(action)
        # 全アクション次元の対数確率を合計
        log_prob_sum = log_prob.sum(dim=1)

        # エントロピー（探索の多様性指標）
        entropy = probs.entropy()
        entropy_sum = entropy.sum(dim=1)

        # 状態価値
        value = self.critic(latent_eval)

        return action, log_prob_sum, entropy_sum, value

    def get_deterministic_action_and_value(self, x):
        """
        推論用: 行動分布の平均値（deterministic action）を返す。
        """
        batch_size = x.shape[0]

        flat_x_eval = x.reshape(-1, self.obs_dim)
        encoded_eval = self.obs_encoder(flat_x_eval)
        encoded_eval_seq = encoded_eval.reshape(batch_size, self.seq_len, self.hidden_dim)
        context_eval = self.transformer(encoded_eval_seq)
        latent_eval = context_eval[:, -1, :]

        action_mean = self.actor_mean(latent_eval)

        logstd_clamped = torch.clamp(self.actor_logstd, -3.0, 2.0)
        action_logstd_eval = logstd_clamped.expand_as(action_mean)
        action_std_eval = torch.exp(action_logstd_eval)
        probs = distributions.Normal(action_mean, action_std_eval)

        log_prob = probs.log_prob(action_mean).sum(dim=1)
        entropy = probs.entropy().sum(dim=1)
        value = self.critic(latent_eval)
        return action_mean, log_prob, entropy, value
