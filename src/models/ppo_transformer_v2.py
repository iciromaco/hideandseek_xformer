# ppo_transformer_v2.py v2.14
# 演習第26回：【完全展開・省略一切禁止】エントロピー爆発防止クランプ ＆ 形状整合性完遂版
# 
# 遵守事項:
# 1. 処理の完全展開: 1行1命令を徹底。計算プロセスを一行に詰め込む圧縮や省略を完全に禁止。
# 2. 形状変形エラー修正: すべての .view() を .reshape() に全置換。不連続テンソルでも安全。
# 3. エントロピー制御: actor_logstd を [-5, 2] の範囲にクランプし、ランダム化への逃避を物理的に遮断。
# 4. 詳細コメント: 各ネットワーク層の役割とテンソル形状の変遷を詳細に記述。

import torch
import torch.nn as nn
import torch.distributions as distributions
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    ニューラルネットワーク層の重みを直交初期化する補助関数。
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AgentV2(nn.Module):
    """
    Transformerをバックボーンに持つPPOエージェント。
    [移動, 旋回, ロック, 掴み] の多次元アクションを制御可能。
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, seq_len):
        super(AgentV2, self).__init__()
        
        # ハイパーパラメータの保存
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

        # --- [1] 入力エンコーディング層 ---
        # 53次元の観測ベクトルを内部表現へ投影
        self.obs_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )

        # --- [2] Transformer バックボーン ---
        # 過去のコンテキストを考慮するためのマルチヘッドアテンション
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- [3] 状態価値（Critic）ヘッド ---
        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        )

        # --- [4] 方策（Actor）ヘッド ---
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)
        )
        
        # 各アクション次元ごとの学習可能な分散（探索範囲）
        # 初期値 0 は exp(0) = 1.0 の標準偏差を意味する
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """
        GAE計算に使用するための状態価値のみを取得。
        """
        # 特徴抽出
        batch_size = x.shape[0]
        
        # 入力をフラット化してエンコード
        # 💡 view ではなく reshape を使用して不連続テンソルに対応
        flat_x = x.reshape(-1, self.obs_dim)
        
        # エンコーダの適用
        encoded = self.obs_encoder(flat_x)
        
        # 時系列形状 (batch, seq, hidden) へ復元
        encoded_sequence = encoded.reshape(batch_size, self.seq_len, self.hidden_dim)
        
        # Transformerによる時間相関の解析
        context = self.transformer(encoded_sequence)
        
        # 最新ステップ（インデックス -1）の情報を抽出
        last_step_context = context[:, -1, :]
        
        # 状態価値の算出
        value = self.critic(last_step_context)
        
        return value

    def get_action_and_value(self, x, action=None):
        """
        行動のサンプリング、対数確率、エントロピー、状態価値を計算。
        """
        # バッチサイズの取得
        batch_size = x.shape[0]
        
        # 1. 特徴抽出（バックボーン）
        # 💡 view ではなく reshape を使用
        flat_x_eval = x.reshape(-1, self.obs_dim)
        
        # エンコーダによる特徴投影
        encoded_eval = self.obs_encoder(flat_x_eval)
        
        # 時系列形状へ復元
        encoded_eval_seq = encoded_eval.reshape(batch_size, self.seq_len, self.hidden_dim)
        
        # 2. Transformer による文脈解析
        context_eval = self.transformer(encoded_eval_seq)
        
        # 3. 最新の文脈（隠れ状態）を抽出
        latent_eval = context_eval[:, -1, :]
        
        # 4. 分布パラメータの算出
        # 平均値(Mean)の計算
        action_mean = self.actor_mean(latent_eval)
        
        # 💡 【核心修正】探索範囲（エントロピー）の暴走を物理的にクランプ
        # -5.0 (std ≒ 0.006) から 2.0 (std ≒ 7.38) の範囲に制限
        # これにより、エージェントが学習を諦めて「ランダム化」に向かうのを防ぐ
        logstd_clamped = torch.clamp(self.actor_logstd, -5.0, 2.0)
        
        # 対数標準偏差を全バッチに拡張
        action_logstd_eval = logstd_clamped.expand_as(action_mean)
        
        # exp による標準偏差の算出
        action_std_eval = torch.exp(action_logstd_eval)
        
        # 5. 多変量正規分布の構築
        action_distribution = distributions.Normal(action_mean, action_std_eval)
        
        # 6. 行動の決定
        if action is None:
            # 推論時：再パラメータ化を行わないサンプリング
            action = action_distribution.sample()
        
        # 7. 各種統計量の計算
        # 各アクション軸の対数確率を算出し、全次元で合算
        log_prob_eval = action_distribution.log_prob(action)
        log_prob_sum = log_prob_eval.sum(1)
        
        # 探索を維持するためのエントロピー
        entropy_eval = action_distribution.entropy()
        entropy_sum = entropy_eval.sum(1)
        
        # 状態価値（Critic出力）
        value_eval = self.critic(latent_eval)
        
        return action, log_prob_sum, entropy_sum, value_eval