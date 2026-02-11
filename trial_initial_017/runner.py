# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【修正内容 (v25.24)】
# 1. PEP 8 完全準拠の独立行展開:
#    - すべてのセミコロン (;) を削除し、1行につき1つの動作のみを記述することを徹底。
#    - if文、for文、代入処理をすべて複数行に展開し、可読性と文法的な安定性を確保。
# 2. 詳細な日本語解説コメントの完全復元:
#    - 報酬設計の数理的背景、物理シミュレーションの進展制御、観測ベクトルの整形など、
#      すべての論理ブロックに詳細な技術解説を付加。
# 3. 停滞ペナルティ (PENALTY_STAGNATION) の実装:
#    - エージェントが「棒立ち」で生存報酬を稼ぐのを防ぐため、移動距離が 1cm 未満の場合に
#      マイナス報酬を課すロジックを独立した行で記述。
# 4. 観測外れ値の適正化 (OUTLIER_VALUE = 2.0):
#    - ネットワークの飽和を防ぎつつ、不可視情報を明示するための最適な外れ値を設定。

import os
import sys
import platform
import json
import time
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- 実行環境の最適化 ---
# 並列実行時に数値計算ライブラリが過剰なスレッドを作成し、CPUリソースを奪い合うのを防ぎます。
if platform.processor() != 'arm':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- プロジェクトパスの解決 ---
# 共通資産である main18_optimization.py 等にアクセスするため、ディレクトリを遡ります。
current_file_path = os.path.abspath(__file__)
search_path = os.path.dirname(current_file_path)
for _ in range(5):
    if os.path.exists(os.path.join(search_path, "main18_optimization.py")):
        if search_path not in sys.path:
            sys.path.insert(0, search_path)
        break
    search_path = os.path.dirname(search_path)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# ==========================================
# 1. 実験設定 (定数定義)
# ==========================================
# 探索モード: "initial" (新規学習) / "refinement" (既存モデルをベースに微調整)
MODE = "initial" 
EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"
# 学習対象: 2体の Hider をチームとして育成します
TRAIN_TARGET = "HIDER" 

EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"
# refinement モードのときのみ、既存のウェイトファイルをロードして再開します
LOAD_EXISTING_MODELS = False

# 実行モード: "TRAIN" (学習) または "PLAY" ( Viewer で挙動を観察)
EXECUTION_MODE = "TRAIN" 

SAVE_MODEL = False
TRACK_WANDB = True
FIXED_SEED = None
TRIAL_MODE = True

# PPO (強化学習アルゴリズム) のハイパーパラメータ
TOTAL_TIMESTEPS = 150000
NUM_ENVS = 8
NUM_STEPS = 128            # 1回の更新で収集するデータ長
LEARNING_RATE = 4.4913274566167296e-05
ENT_COEF = 0.00010572245490200716
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4

# Transformer (記憶メカニズム) の設定
TRANSFORMER_SEQ_LEN = 8    # 過去何ステップの記憶を同時に処理するか
HIDDEN_DIM = 64            # 内部の隠れ層の次元数
NUM_LAYERS = 2             # Transformer ブロックの積層数
NUM_HEADS = 2              # Attention 機構のヘッド数

# 環境・物理シミュレーションの設定
ACTION_REPEAT = 16         # 1アクションにつき MuJoCo を何回進めるか
PREP_STEPS = 80            # 序盤の数秒間、Seeker は待機（Hider が陣形を整える時間）
MAX_STEPS = 300            # 固定長エピソード。途中で捕まっても 300歩まで継続
FOV_DEG = 135              # 視野角の設定（135度）

# 高速化のためのキャッシュ制御閾値
LIDAR_CACHE_POS_THRESH = 0.05
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)
RAYCAST_CACHE_POS_THRESH = 0.05

# 観測情報のマスク用外れ値
# 相対距離 0.0 (自分自身や重なり) と「見えない状態」を区別するため、
# フィールドの対角線長を考慮し、ネットワークが処理しやすい 2.0 を採用。
OUTLIER_VALUE = 2.0

# 報酬設計
REWARD_HIDDEN_BONUS = 1.4599247818575418
COS_PENALTY_SCALE = 4.018424914811919
REWARD_DISTANCE_DIFF_SCALE = 1.0 # 敵から遠ざかる動きに対する補助報酬
PENALTY_SAFEGUARD = -20.0        # 場外脱走に対するペナルティ
PENALTY_STAGNATION = -0.5        # その場に留まった（棒立ち）場合へのペナルティ

# エージェントの移動能力制限
HIDER_THRUST_LIMIT = 0.40  
SEEKER_THRUST_LIMIT = 0.35 
SEEKER_RB_THRUST = 0.38          # NPCシーカーの基準推力
SEEKER_RB_TURN_THRESH = np.pi/6    # 急旋回時に減速を行う角度閾値

# 保存パス
SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# ==========================================
# 2. モデル・バッファ定義 (Agent / ObsHistory)
# ==========================================
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """重みの初期化を行い、学習の初期段階での出力の偏りを抑えます。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """Transformer Actor-Critic。過去の系列をコンテキストとして行動を決定。"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 観測ベクトルの埋め込み
        self.embedding = nn.Linear(obs_dim, HIDDEN_DIM)
        # 過去の順序を認識させるための学習可能な位置エンコーディング
        self.pos_encoder = nn.Parameter(torch.zeros(1, TRANSFORMER_SEQ_LEN, HIDDEN_DIM))
        
        # Transformer エンコーダの定義
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, 
            nhead=NUM_HEADS, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=NUM_LAYERS, 
            enable_nested_tensor=False
        )
        
        # 行動の平均値を出力するアクター層
        self.actor_mean = layer_init(nn.Linear(HIDDEN_DIM, action_dim), std=0.01)
        # 行動の分散を調整するパラメータ
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        # 状態価値を予測するクリティック層
        self.critic = layer_init(nn.Linear(HIDDEN_DIM, 1), std=1)

    def get_value(self, x):
        """現在の観測系列から状態価値 V(s) を返します。"""
        # 入力を埋め込み空間へ射影
        x = self.embedding(x)
        # 位置情報を加算
        x = x + self.pos_encoder
        # Transformer で系列情報を処理
        x = self.transformer(x)
        # 最後の時間ステップの表現をコンテキストとして使用
        h_last = x[:, -1, :]
        val = self.critic(h_last)
        return val

    def get_action_and_value(self, x, action=None):
        """行動、対数確率、エントロピー、状態価値を一括取得します。"""
        # 共通の特徴抽出処理
        x = self.embedding(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        h_context = x[:, -1, :]
        
        # 確率密度関数（正規分布）の構築
        mean = self.actor_mean(h_context)
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        probs = Normal(mean, std)
        
        if action is None:
            # 行動が未指定の場合はサンプリング
            action = probs.sample()
            
        # 確率計算とエントロピーの取得
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic(h_context)
        
        return action, log_prob, entropy, value

class ObsHistory:
    """
    ミラーリングバッファを用いたゼロコピー履歴管理クラス。
    シーケンス長を 2倍確保することで、スライスのみで連続系列を取得可能にします。
    """
    def __init__(self, num_envs, seq_len, obs_dim, device):
        # 連続したメモリ領域として取得できるよう、seq_len の2倍の長さを確保
        buffer_size = seq_len * 2
        self.buffer = torch.zeros((num_envs, buffer_size, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.ptr = 0

    def reset(self):
        """バッファのゼロクリアとポインタのリセット。"""
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        """最新の観測値で履歴を更新します。ミラーリングにより 2箇所に書き込みます。"""
        # テンソルへの変換
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # バッチ次元の有無を補正
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        # 現在のポインタ位置とその seq_len 先の 2箇所に書き込む
        self.buffer[:, self.ptr] = obs_tensor
        self.buffer[:, self.ptr + self.seq_len] = obs_tensor
        
        # ポインタを循環させる
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        """最新の seq_len ステップ分の履歴を View (コピーなし) として返します。"""
        # ミラーリング書き込みにより、ptr から seq_len 分をスライスするだけで常に正しい順序になります
        return self.buffer[:, self.ptr : self.ptr + self.seq_len]

# ==========================================
# 3. ヘルパー関数
# ==========================================
def load_model_safely(model_obj, base_name, target_type):
    """
    指定されたターゲットに対応する最新のモデルファイルを安全にロードします。
    """
    import torch
    # 探索対象のファイルリスト (優先順位順)
    candidates = [
        f"{base_name}_refinement_{target_type}.pt",
        f"{base_name}_initial_{target_type}.pt",
        f"{base_name}_{target_type}.pt"
    ]
    
    for path in candidates:
        if os.path.exists(path):
            try:
                # CPU マッピングで読み込み、どの環境でも動作を保証
                state_dict = torch.load(path, map_location="cpu")
                model_obj.load_state_dict(state_dict)
                model_obj.eval()
                return path
            except Exception:
                # ロード失敗時は次の候補へ
                continue
    return None

# ==========================================
# 4. 環境作成用ファクトリ
# ==========================================
def create_env(render_mode=None):
    """
    Linux 環境等でのマルチプロセス競合（CUDA デッドロック）を避けるため、
    MuJoCo のロードとクラス定義をプロセス生成時まで遅らせます。
    """
    import torch
    import mujoco
    import gymnasium as gym
    import main18_optimization as base_config

    class TeamCosEnv(base_config.HideAndSeekEnv):
        """
        視界勾配、外れ値マスク、停滞ペナルティを統合した環境クラス。
        """
        def __init__(self, render_mode=None):
            # 基底クラス HideAndSeekEnv の初期化
            super().__init__(render_mode=render_mode)
            
            cpu_device = torch.device("cpu")
            # 各エージェント（Seeker=0, Hider1=1, Hider2=2）の観測履歴バッファ
            self.npc_obs_history = {
                0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
                1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
                2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device)
            }
            
            # 高速化のためのキャッシュ機構
            self.visible_cache = {0: {}, 1: {}, 2: {}}
            self.lidar_array_cache = {} 
            self.raycast_cache = {} 
            self.raycast_stats = {"hits": 0, "misses": 0}
            
            # 統計用カウンタ
            self.hidden_steps_count = 0
            self.caught_steps_count = 0 
            # 報酬計算用の「前ステップの距離」
            self.prev_dist = {1: 0.0, 2: 0.0}
            
            # ★追加: 停滞ペナルティ判定用の「前ステップの絶対座標」
            self.prev_hider_xy = {1: None, 2: None}
            
            # フレーム内重複計算回避用のメモ化
            self._obs_memo = {}
            # スタック時のランダム旋回方向フラグ
            self.s0_recovery_turn_dir = 1.0

            # NPC 用のエージェントモデルを作成
            self.npc_hider_agent = Agent(53, 4).to("cpu")
            self.npc_seeker_agent = Agent(53, 4).to("cpu")
            
            # ログ表示の制御（最初のワーカープロセスのみ出力）
            logged_flag = os.environ.get("NPC_MODELS_LOGGED")
            should_print = (logged_flag != "TRUE")

            # HIDER NPC モデルのロード
            h_path = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
            if h_path:
                if should_print:
                    print(f"Loaded NPC Hider from {h_path}", flush=True)
            else:
                self.npc_hider_agent = None

            # SEEKER NPC モデルのロード
            s_path = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
            if s_path:
                if should_print:
                    print(f"Loaded NPC Seeker from {s_path}", flush=True)
            else:
                self.npc_seeker_agent = None

            # ロード完了フラグをセット
            if should_print:
                os.environ["NPC_MODELS_LOGGED"] = "TRUE"

        def _is_visible(self, origin_pos, origin_rot, target_pos, target_body_id, exclude_body_id):
            """視界角と遮蔽を考慮して対象が可視か判定します。"""
            diff_vec = target_pos[:2] - origin_pos[:2]
            dist_val = np.linalg.norm(diff_vec)
            
            # 極至近距離（10cm以内）は無条件で可視
            if dist_val < 0.1:
                return True, target_body_id
            
            # 1. 視野角（FOV）判定
            angle_to_target = np.arctan2(diff_vec[1], diff_vec[0])
            rel_angle = (angle_to_target - origin_rot + np.pi) % (2 * np.pi) - np.pi
            if abs(rel_angle) > np.deg2rad(FOV_DEG / 2.0):
                return False, -1
            
            # 2. 物理的な遮蔽（RayCast）判定
            direction = np.array([diff_vec[0]/dist_val, diff_vec[1]/dist_val, 0.0], dtype=np.float64)
            # レイの始点を腰の高さ (0.5m) に固定
            ray_from = np.array([origin_pos[0], origin_pos[1], 0.5], dtype=np.float64)
            geom_out = np.zeros(1, dtype=np.int32)
            
            # MuJoCo の mj_ray を実行
            dist_res = mujoco.mj_ray(
                self.model, 
                self.data, 
                ray_from, 
                direction, 
                None, 
                1, 
                exclude_body_id, 
                geom_out
            )
            
            if dist_res != -1:
                hit_bid = self.model.geom_bodyid[geom_out[0]]
                # ターゲット自体に当たった場合
                if hit_bid == target_body_id:
                    return True, target_body_id
                # ターゲットより手前（40cm以上の差）で別の物体に当たった場合
                if dist_res < dist_val - 0.4:
                    return False, hit_bid
            
            # 何にも遮られなかった場合
            return True, target_body_id

        def _check_collision_all(self, pos, threshold):
            """指定座標が障害物の内部にあるか判定します。"""
            # 外壁との衝突
            for wall_pos, wall_size in self.wall_data:
                dx = abs(pos[0] - wall_pos[0]) - wall_size[0]
                dy = abs(pos[1] - wall_pos[1]) - wall_size[1]
                # 矩形境界からの距離の二乗を算出
                dist_sq = max(dx, 0.0)**2 + max(dy, 0.0)**2
                if dist_sq < threshold**2:
                    return True
            
            # フィールド内オブジェクトとの衝突
            for gid in self.box_geoms + self.ramp_all_geoms:
                o_pos = self.data.geom_xpos[gid][:2]
                o_size = self.model.geom(gid).size[:2]
                dx_o = abs(pos[0] - o_pos[0]) - o_size[0]
                dy_o = abs(pos[1] - o_pos[1]) - o_size[1]
                dist_sq_o = max(dx_o, 0.0)**2 + max(dy_o, 0.0)**2
                if dist_sq_o < threshold**2:
                    return True
            
            return False

        def _get_cached_ray(self, agent_id, origin_p, direction, beam_id):
            """mj_ray の呼び出しを座標ベースでキャッシュし、物理演算負荷を下げます。"""
            angle = np.arctan2(direction[1], direction[0])
            cache_key = (agent_id, beam_id)
            
            # キャッシュヒット判定
            if cache_key in self.raycast_cache:
                cached_pos, cached_ang, cached_res, cached_gid = self.raycast_cache[cache_key]
                p_moved = np.linalg.norm(origin_p - cached_pos)
                if p_moved < RAYCAST_CACHE_POS_THRESH:
                    a_err = (angle - cached_ang + np.pi) % (2 * np.pi) - np.pi
                    if abs(a_err) < 0.05:
                        self.raycast_stats["hits"] += 1
                        return cached_res, cached_gid
            
            # キャッシュミス時
            self.raycast_stats["misses"] += 1
            hit_res = np.zeros(1, dtype=np.int32)
            ray_fr = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
            ray_dr = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
            
            # 自己遮蔽の除外設定
            if agent_id == 0:
                exclude = self.s0_body
            elif agent_id == 1:
                exclude = self.h1_body
            else:
                exclude = self.h2_body
                
            dist_val = mujoco.mj_ray(self.model, self.data, ray_fr, ray_dr, None, 1, exclude, hit_res)
            
            # キャッシュに保存
            self.raycast_cache[cache_key] = (origin_p.copy(), angle, dist_val, hit_res[0])
            return dist_val, hit_res[0]

        def _get_obs(self, agent_id):
            """
            53次元観測を生成します。不可視時は外れ値 2.0 でマスクします。
            すべて独立した行で記述し、ロジックを透明化します。
            """
            # 同一ステップ内での重複呼び出しであれば即座に回答
            if agent_id in self._obs_memo:
                return self._obs_memo[agent_id]
                
            # エージェント別の ID と名称プレフィックスを決定
            if agent_id == 0:
                b_id = self.s0_body
                prefix = 's'
            elif agent_id == 1:
                b_id = self.h1_body
                prefix = 'h1'
            else:
                b_id = self.h2_body
                prefix = 'h2'
            
            # 現在の位置と角度の取得
            pos_xy = self.data.xpos[b_id][:2]
            q_idx = self.model.jnt_qposadr[self.model.joint(f'{prefix}_rot').id]
            angle_rad = self.data.qpos[q_idx]
            
            # ローカル座標変換用の回転行列
            cos_a = np.cos(-angle_rad)
            sin_a = np.sin(-angle_rad)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # 自己速度の取得と正規化
            jx_idx = self.model.jnt_dofadr[self.model.joint(f'{prefix}_x').id]
            v_raw = self.data.qvel[jx_idx : jx_idx + 2]
            v_local = rot_matrix @ v_raw
            v_normalized = v_local / 12.0
            
            # 自己状態配列 [local_vx, local_vy, rad, cos, sin]
            self_state = np.concatenate([
                v_normalized, 
                [angle_rad, np.cos(angle_rad), np.sin(angle_rad)]
            ])
            
            # Lidar 情報の一括キャッシュ処理
            lidar_out = None
            cached_l = self.lidar_array_cache.get(agent_id)
            if cached_l is not None:
                c_p, c_r, c_lidar = cached_l
                p_err = np.linalg.norm(pos_xy - c_p)
                r_err = abs((angle_rad - c_r + np.pi) % (2 * np.pi) - np.pi)
                if p_err < LIDAR_CACHE_POS_THRESH and r_err < LIDAR_CACHE_ANG_THRESH:
                    lidar_out = c_lidar
            
            # キャッシュミス時は全方向再計算
            if lidar_out is None:
                lidar_out = np.zeros(len(self.lidar_angles), dtype=np.float32)
                for i, offset in enumerate(self.lidar_angles):
                    bd = offset + angle_rad
                    beam_dir = np.array([np.cos(bd), np.sin(bd)])
                    ray_dist, _ = self._get_cached_ray(agent_id, pos_xy, beam_dir, i + 100)
                    # 最大距離 2.5m で正規化
                    lidar_out[i] = min(ray_dist, 2.5) / 2.5 if ray_dist != -1 else 1.0
                # キャッシュ保存
                self.lidar_array_cache[agent_id] = (pos_xy.copy(), angle_rad, lidar_out.copy())

            # 視界判定キャッシュの更新
            my_vis = self.visible_cache[agent_id]
            my_vis.clear()
            candidates = [self.box1_body, self.box2_body, self.ramp_body, self.h1_body, self.h2_body, self.s0_body]
            # 自分以外を対象にループ
            for target_id in candidates:
                if target_id != b_id:
                    v_flag, _ = self._is_visible(self.data.xpos[b_id], angle_rad, self.data.xpos[target_id], target_id, b_id)
                    my_vis[target_id] = v_flag

            def get_rel_info(target_id, lock_state=None):
                """相対情報を整形します。不可視時は外れ値 2.0 を採用。"""
                is_seen = my_vis.get(target_id, False)
                # Lock状態の有無で次元数を変更
                sz = 8 if lock_state is not None else 7
                
                if is_seen:
                    t_xyz = self.data.xpos[target_id]
                    # 相対座標
                    rel_p = rot_matrix @ (t_xyz[:2] - pos_xy) / 12.0
                    
                    # 相対回転
                    q_val = self.data.xquat[target_id]
                    t_yaw = np.arctan2(
                        2 * (q_val[0] * q_val[3] + q_val[1] * q_val[2]), 
                        1 - 2 * (q_val[2]**2 + q_val[3]**2)
                    )
                    
                    # 相対速度
                    j_adr = self.model.body_jntadr[target_id]
                    t_v_g = self.data.qvel[j_adr : j_adr + 2] if j_adr != -1 else np.zeros(2)
                    rel_v = rot_matrix @ (t_v_g - v_raw) / 12.0
                    
                    info = [rel_p, rel_v, [np.cos(t_yaw - angle_rad), np.sin(t_yaw - angle_rad)]]
                    if lock_state is not None:
                        val = 1.0 if lock_state else 0.0
                        info.append([val])
                    
                    # 可視フラグ
                    info.append([1.0])
                    return np.concatenate(info)
                else:
                    # 不可視な場合：数値計算の 0.0 (原点) と区別するため外れ値 2.0 で埋める
                    masked_vec = np.full(sz, OUTLIER_VALUE, dtype=np.float32)
                    # ただし、最後の要素（可視フラグ）は論理的に 0.0 に固定
                    masked_vec[-1] = 0.0
                    return masked_vec

            # エージェント種別ごとのデータ構築
            if agent_id == 0:
                # シーカー視点の観測
                h1_rel = get_rel_info(self.h1_body)[:5]
                h2_rel = get_rel_info(self.h2_body)[:5]
                obj_list = [
                    get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    get_rel_info(self.ramp_body)
                ]
                obs_final = np.concatenate([
                    self_state, lidar_out, *obj_list, h1_rel, h2_rel, np.zeros(3, dtype=np.float32)
                ])
            else:
                # ハイダー視点の観測
                partner_id = self.h2_body if agent_id == 1 else self.h1_body
                enemy_rel = get_rel_info(self.s0_body)[:5]
                friend_rel = get_rel_info(partner_id)
                grasp_val = 1.0 if self.grasping[agent_id] else 0.0
                obj_list = [
                    get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    get_rel_info(self.ramp_body)
                ]
                obs_final = np.concatenate([
                    self_state, lidar_out, *obj_list, enemy_rel, friend_rel, [grasp_val]
                ])

            res_final = obs_final.astype(np.float32)
            self._obs_memo[agent_id] = res_final
            return res_final

        def _update_seeker_state(self):
            """NPCシーカーの思考ルーチン。追跡、捜索、巡回の状態遷移を管理します。"""
            sp_xy = self.data.xpos[self.s0_body][:2]
            
            # 視覚情報の更新
            self._get_obs(0)
            seen_h1 = self.visible_cache[0].get(self.h1_body, False)
            seen_h2 = self.visible_cache[0].get(self.h2_body, False)
            
            if seen_h1 or seen_h2:
                # いずれかを発見した瞬間
                target_bid = self.h1_body if seen_h1 else self.h2_body
                self.seeker_target_pos = self.data.xpos[target_bid][:2].copy()
                self.seeker_last_known_pos = self.seeker_target_pos.copy()
                self.seeker_mode = "CHASING"
            elif self.seeker_last_known_pos is not None:
                # 見失ったが、最後に見た座標へ向かう状態
                dist_to_mem = np.linalg.norm(sp_xy - self.seeker_last_known_pos)
                if dist_to_mem > 0.5:
                    self.seeker_target_pos = self.seeker_last_known_pos.copy()
                    self.seeker_mode = "SEARCHING"
                else:
                    # 捜索完了（見つからなかった）
                    self.seeker_last_known_pos = None
                    self.seeker_search_timer = 50
            else:
                # 手がかりがない状態：ランダムに巡回
                if self.seeker_search_timer <= 0:
                    self.seeker_random_target = self.np_random.uniform(-4, 4, 2)
                    self.seeker_search_timer = 80
                
                self.seeker_search_timer = self.seeker_search_timer - 1
                self.seeker_target_pos = self.seeker_random_target.copy()
                self.seeker_mode = "PATROLLING"

        def _seeker_rule_based_policy(self):
            if self.current_step < PREP_STEPS:
                return 0.0, 0.0
            
            s_p = self.data.xpos[self.s0_body][:2]
            s_r = self.data.qpos[self.srot_adr]
            t_p = self.seeker_target_pos
            
            # ターゲット方向への方位角
            dx = t_p[0] - s_p[0]
            dy = t_p[1] - s_p[1]
            t_a = np.arctan2(dy, dx)
            
            # 角度偏差 (-pi 〜 pi)
            a_d = (t_a - s_r + np.pi) % (2 * np.pi) - np.pi
            
            thrust_val = SEEKER_RB_THRUST
            turn_val = np.clip(a_d * 6.0, -3.0, 3.0)
            
            # 角度が大きくずれている場合は前進速度を落とす
            if abs(a_d) > SEEKER_RB_TURN_THRESH:
                thrust_val = thrust_val * 0.3
            
            # スタック判定
            sx_id = self.model.joint('s_x').id
            s_dof = self.model.jnt_dofadr[sx_id]
            v_now = np.linalg.norm(self.data.qvel[s_dof : s_dof + 2])
            
            if thrust_val > 0.05 and v_now < 0.05:
                self.s0_stuck_timer = self.s0_stuck_timer + 5
            else:
                self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            
            # リカバリーモード（後退と旋回）
            if self.s0_stuck_timer > 15:
                self.s0_recovery_mode = 15
                self.s0_stuck_timer = 0
                self.s0_recovery_turn_dir = self.np_random.choice([-1.0, 1.0])
            
            if self.s0_recovery_mode > 0:
                thrust_val = -0.2 # 後退
                turn_val = 1.5 * self.s0_recovery_turn_dir # ランダム転回
                self.s0_recovery_mode = self.s0_recovery_mode - 1
                
            return float(thrust_val), float(turn_val)

        def _get_npc_action(self, agent_id, agent_type):
            """NPCの意思決定。モデル（AI）を最優先し、なければルール/ランダムに切り替えます。"""
            import torch
            # 現在の観測を取得してバッファ更新
            obs_raw = self._get_obs(agent_id)
            self.npc_obs_history[agent_id].update(obs_raw)
            
            # ロード済みモデルの選択
            if agent_type == "HIDER":
                model_to_use = self.npc_hider_agent
            else:
                model_to_use = self.npc_seeker_agent
                
            # AI モデル推論
            if model_to_use is not None:
                with torch.no_grad():
                    context = self.npc_obs_history[agent_id].get()
                    # 出力は正規化済みアクション [-1.0, 1.0]
                    act_t, _, _, _ = model_to_use.get_action_and_value(context)
                return act_t.cpu().numpy()[0]
            
            # Seeker ルールベース（モデルがない場合）
            if agent_type == "SEEKER":
                raw_f, raw_r = self._seeker_rule_based_policy()
                # step メソッド側の期待（LIMITを掛ける前）に合わせて正規化
                norm_f = raw_f / SEEKER_THRUST_LIMIT
                return np.array([norm_f, raw_r, 0.0, 0.0], dtype=np.float32)
            
            # Hider ランダム（モデルがない場合）
            return self.action_space.sample() * 0.5

        def reset(self, seed=None, options=None):
            """物理シミュレーションと各バッファを初期化します。"""
            obs, info = super().reset(seed=seed, options=options)
            
            self.hidden_steps_count = 0
            self.caught_steps_count = 0
            self._obs_memo.clear()
            self.lidar_array_cache.clear()
            self.s0_recovery_turn_dir = 1.0
            
            # 報酬計算用の初期化
            s_xy = self.data.xpos[self.s0_body][:2]
            for i in [1, 2]:
                bid = self.h1_body if i == 1 else self.h2_body
                self.prev_dist[i] = np.linalg.norm(self.data.xpos[bid][:2] - s_xy)
                # ★追加: 停滞判定用座標の保存
                self.prev_hider_xy[i] = self.data.xpos[bid][:2].copy()
            
            return obs, info

        def step(self, action):
            """アクション適用 -> 物理演算 -> 報酬算出を1サイクル行います。"""
            # 新しいステップのためメモ化バッファをクリア
            self._obs_memo.clear()
            # ステップカウント更新
            self.current_step = self.current_step + 1
            
            # ロック機能のクールダウン更新
            for i in [1, 2]:
                val = self.lock_cooldown[i]
                self.lock_cooldown[i] = max(0, val - 1)
            
            # Seeker NPC の状態更新
            self._update_seeker_state()
            # 制御入力の初期化
            self.data.ctrl[:] = 0.0 
            
            if TRAIN_TARGET == "HIDER":
                # 学習エージェント (Hider)
                m_idx = self._apply_action(self.learning_agent_id, action)
                self.data.ctrl[m_idx] = float(action[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[m_idx + 1] = float(action[1])
                
                # 相棒 NPC (Hider)
                p_id = 2 if self.learning_agent_id == 1 else 1
                act_p = self._get_npc_action(p_id, "HIDER")
                p_idx = self._apply_action(p_id, act_p)
                self.data.ctrl[p_idx] = float(act_p[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[p_idx + 1] = float(act_p[1])
                
                # 敵 NPC (Seeker - AI優先)
                act_s = self._get_npc_action(0, "SEEKER")
                self.data.ctrl[0] = float(act_s[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(act_s[1])
            else:
                # 学習エージェント (Seeker)
                self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(action[1])
                
                # 敵 NPC (Hiders x2)
                for i in [1, 2]:
                    act_n = self._get_npc_action(i, "HIDER")
                    n_idx = self._apply_action(i, act_n)
                    self.data.ctrl[n_idx] = float(act_n[0]) * HIDER_THRUST_LIMIT
                    self.data.ctrl[n_idx + 1] = float(act_n[1])
            
            # --- 物理進展ループ (Action Repeat) ---
            for _ in range(ACTION_REPEAT):
                # ロック状態の物理強制固定 (Velocity Freeze)
                for b_bid, pose_val in self.locked_pose.items():
                    if self.locked_boxes[b_bid]:
                        # 対象 Box の Joint 特定
                        t_jid = self.box1_joint_id if b_bid == self.box1_body else self.box2_joint_id
                        q_a = self.model.jnt_qposadr[t_jid]
                        d_a = self.model.jnt_dofadr[t_jid]
                        # 座標の強制上書き
                        self.data.qpos[q_a : q_a + 7] = pose_val
                        # 速度を完全にリセット
                        self.data.qvel[d_a : d_a + 6] = 0
                
                # 物理計算
                mujoco.mj_step(self.model, self.data)
            
            # 演算後の観測と報酬の確定
            self._obs_memo.clear()
            obs_learner = self._get_obs(self.learning_agent_id)
            # 統計用にシーカー視界を更新
            self._get_obs(0) 
            
            team_reward_sum = 0.0
            l_body = self.h1_body if self.learning_agent_id == 1 else self.h2_body
            
            # 統計情報の更新
            is_l_vis = self.visible_cache[0].get(l_body, False)
            is_a_vis = any(self.visible_cache[0].get(bid, False) for bid in [self.h1_body, self.h2_body])
            
            if not is_l_vis:
                self.hidden_steps_count = self.hidden_steps_count + 1
            if is_a_vis:
                self.caught_steps_count = self.caught_steps_count + 1
                
            # チーム全体（Hider1+2）の評価
            for h_idx, body_id in [(1, self.h1_body), (2, self.h2_body)]:
                f_vis = self.visible_cache[0].get(body_id, False)
                seeker_xy = self.data.xpos[self.s0_body][:2]
                curr_xy = self.data.xpos[body_id][:2]
                curr_d = np.linalg.norm(curr_xy - seeker_xy)
                
                if f_vis:
                    # 視界内ペナルティ（真正面ほど大きなマイナス）
                    sr_pos = self.data.qpos[self.srot_adr]
                    to_vec = curr_xy - seeker_xy
                    norm_v = to_vec / (np.linalg.norm(to_vec) + 1e-8)
                    cos_theta = np.dot(norm_v, np.array([np.cos(sr_pos), np.sin(sr_pos)]))
                    h_rew = -cos_theta * COS_PENALTY_SCALE
                    # 距離増分による逃走成功ボーナス
                    h_rew = h_rew + (curr_d - self.prev_dist[h_idx]) * REWARD_DISTANCE_DIFF_SCALE
                else:
                    # 隠蔽ボーナス
                    h_rew = REWARD_HIDDEN_BONUS
                
                # ★復活: 停滞ペナルティ（学習対象の Hider に適用）
                if h_idx == self.learning_agent_id:
                    if self.prev_hider_xy[h_idx] is not None:
                        dist_moved = np.linalg.norm(curr_xy - self.prev_hider_xy[h_idx])
                        # 1環境ステップ（16物理進展後）で 1cm 未満ならマイナス
                        if dist_moved < 0.01:
                            h_rew = h_rew + PENALTY_STAGNATION
                    # 座標の更新
                    self.prev_hider_xy[h_idx] = curr_xy.copy()
                
                # 場外への脱走に対する保険
                if max(abs(curr_xy)) > 6.5:
                    h_rew = h_rew + PENALTY_SAFEGUARD
                
                team_reward_sum = team_reward_sum + h_rew
                self.prev_dist[h_idx] = curr_d
                
            final_reward = team_reward_sum if TRAIN_TARGET == "HIDER" else -team_reward_sum
            truncated_flag = (self.current_step >= MAX_STEPS)
            
            # デバッグ情報
            info_out = {
                "hidden_steps": float(self.hidden_steps_count), 
                "caught_steps": float(self.caught_steps_count)
            }
            return obs_learner, float(final_reward), False, truncated_flag, info_out

    return TeamCosEnv(render_mode=render_mode)

# ==========================================
# 5. メイン処理 (学習ループ)
# ==========================================
def main():
    # --- 1. プロセス起動設定 (Linux デッドロック回避) ---
    if platform.system() == "Linux":
        try:
            # 1行1動作：spawn 方式を強制
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # 並列ワーカー生成用ファクトリ
    def env_factory():
        # create_env 内部で MuJoCo/Torch を遅延ロード
        env_instance = create_env()
        import gymnasium as gym
        return gym.wrappers.RecordEpisodeStatistics(env_instance)
    
    import gymnasium as gym
    print(f"--- [Parent] 1. Initializing {NUM_ENVS} parallel workers ---", flush=True)
    
    # ワーカーが立ち上がるまで親プロセスは CUDA に触れません。
    try:
        vec_envs = gym.vector.AsyncVectorEnv([env_factory for _ in range(NUM_ENVS)])
        print("--- [Parent] 2. Workers initialized successfully ---", flush=True)
    except Exception as e_start:
        print(f"--- [Parent] [CRITICAL] Parallel startup failed: {e_start} ---", flush=True)
        sys.exit(1)

    # --- 3. 親プロセスでのライブラリ初期化 (ワーカー起動後に解禁) ---
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb
    import main18_optimization as base_config

    # デバイス確定
    device_final = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")
    run_ts = int(time.time())
    run_name = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{run_ts}"
    
    # PLAY モードの処理
    if EXECUTION_MODE == "PLAY":
        print(f"--- Inference Mode (PLAY) ---")
        env_p = create_env(render_mode="human")
        agent_p = Agent(env_p.observation_space.shape[0], env_p.action_space.shape[0]).to(device_final)
        lp_p = load_model_safely(agent_p, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if lp_p:
            print(f"Loaded successfully: {lp_p}")
        
        agent_p.eval()
        hist_p = ObsHistory(1, TRANSFORMER_SEQ_LEN, env_p.observation_space.shape[0], device_final)
        # MuJoCo の 1 ステップあたりの実時間
        dt_p = 0.005 * ACTION_REPEAT
        
        try:
            while True:
                obs_v, _ = env_p.reset()
                hist_p.reset()
                hist_p.update(obs_v)
                done_v = False
                total_rew = 0.0
                while not done_v:
                    t_start = time.time()
                    with torch.no_grad():
                        series = hist_p.get()
                        act_v, _, _, _ = agent_p.get_action_and_value(series)
                    
                    obs_next, reward, term, trunc, info = env_p.step(act_v.cpu().numpy()[0])
                    done_v = term or trunc
                    total_rew = total_rew + reward
                    hist_p.update(obs_next)
                    env_p.render()
                    
                    # 再生速度の調整
                    proc_time = time.time() - t_start
                    wait_time = dt_p - proc_time
                    if wait_time > 0:
                        time.sleep(wait_time)
                    
                    if env_p.viewer is not None and not env_p.viewer.is_running():
                        return
                
                print(f"Return: {total_rew:.1f}, Hidden Steps: {info['hidden_steps']:.0f}")
        except KeyboardInterrupt:
            pass
        finally:
            env_p.close()
        return

    # TRAIN モードの初期化
    if TRACK_WANDB:
        run_obj = wandb.init(
            project=base_config.WANDB_PROJECT_NAME, 
            config={"Target": TRAIN_TARGET, "MODE": MODE, "v": "25.24_fully_unrolled"}, 
            name=run_name, 
            sync_tensorboard=False,
            save_code=True
        )
        run_obj.define_metric("global_step")
        run_obj.define_metric("*", step_metric="global_step")

    t_writer = SummaryWriter(f"runs/{run_name}")
    agent_t = Agent(vec_envs.single_observation_space.shape[0], vec_envs.single_action_space.shape[0]).to(device_final)
    optimizer_t = optim.Adam(agent_t.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    global_s_idx = 0
    start_s_val = 0
    if LOAD_EXISTING_MODELS:
        lp_t = load_model_safely(agent_t, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if lp_t:
            print(f"★ Resumed from: {lp_t}")
            cp_f = lp_t.replace('.pt', '_checkpoint.json')
            if os.path.exists(cp_f):
                try:
                    with open(cp_f, 'r') as f_c:
                        data_c = json.load(f_c)
                        global_s_idx = data_c.get('global_step', 0)
                        start_s_val = global_s_idx
                except Exception:
                    pass

    # 訓練用バッファの確保
    hist_train = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device_final)
    S, E, O, A = NUM_STEPS, NUM_ENVS, 53, 4
    batch_obs = torch.zeros((S, E, TRANSFORMER_SEQ_LEN, O), device=device_final)
    batch_act = torch.zeros((S, E, A), device=device_final)
    batch_log = torch.zeros((S, E), device=device_final)
    batch_rew = torch.zeros((S, E), device=device_final)
    batch_don = torch.zeros((S, E), device=device_final)
    batch_val = torch.zeros((S, E), device=device_final)
    
    # 初回リセット
    sd_val = FIXED_SEED if FIXED_SEED else int(time.time())
    next_obs_v, _ = vec_envs.reset(seed=sd_val)
    next_done_v = torch.zeros(E).to(device_final)
    hist_train.reset()
    hist_train.update(next_obs_v)
    
    num_u_total = int(max(1, (TOTAL_TIMESTEPS - global_s_idx) // (E * S)))
    hist_returns, hist_hidden, hist_caught = [], [], []
    start_wall_time = time.time()
    l_loss, l_ent = 0.0, 0.0

    print(f"--- Training loop starting ---")
    try:
        for u_idx in tqdm(range(1, num_u_total + 1), desc="Updates"):
            # データ収集フェーズ (Rollout)
            for step_idx in range(S):
                global_s_idx = global_s_idx + E
                batch_obs[step_idx] = hist_train.get()
                batch_don[step_idx] = next_done_v
                
                with torch.no_grad():
                    a_out, lp_out, _, v_out = agent_t.get_action_and_value(hist_train.get())
                    batch_val[step_idx] = v_out.flatten()
                
                batch_act[step_idx] = a_out
                batch_log[step_idx] = lp_out
                
                # 環境の進展
                nxt_o_v, reward_v, term_v, trunc_v, info_v = vec_envs.step(a_out.cpu().numpy())
                done_mask = np.logical_or(term_v, trunc_v)
                
                # 統計収集
                if "final_info" in info_v:
                    for e_idx in range(E):
                        item = info_v["final_info"][e_idx]
                        if done_mask[e_idx] and item is not None:
                            if "episode" in item:
                                hist_returns.append(float(item["episode"]["r"]))
                            if "hidden_steps" in item:
                                hist_hidden.append(float(item["hidden_steps"]))
                            if "caught_steps" in info_v:
                                hist_caught.append(float(item["caught_steps"]))
                elif "episode" in info_v:
                    e_mask = info_v.get("_episode", [True] * E)
                    for e_idx in range(E):
                        if e_mask[e_idx] and done_mask[e_idx]:
                            hist_returns.append(float(info_v["episode"]["r"][e_idx]))
                            try:
                                if "hidden_steps" in info_v:
                                    hist_hidden.append(float(info_v["hidden_steps"][e_idx]))
                                if "caught_steps" in info_v:
                                    hist_caught.append(float(info_v["caught_steps"][e_idx]))
                            except Exception:
                                pass
                
                batch_rew[step_idx] = torch.tensor(reward_v).to(device_final).view(-1)
                next_done_v = torch.tensor(done_mask).to(device_final, dtype=torch.float32)
                hist_train.update(nxt_o_v)

            # アドバンテージ計算 (GAE)
            with torch.no_grad():
                v_next_pred = agent_t.get_value(hist_train.get()).reshape(1, -1)
                batch_adv = torch.zeros_like(batch_rew).to(device_final)
                gae_ptr = 0
                for t in reversed(range(S)):
                    if t == S - 1:
                        mask_non_term = 1.0 - next_done_v
                        v_post = v_next_pred
                    else:
                        mask_non_term = 1.0 - batch_don[t + 1]
                        v_post = batch_val[t + 1]
                    
                    # 誤差計算
                    delta_t = batch_rew[t] + 0.99 * v_post * mask_non_term - batch_val[t]
                    # 利得加重
                    gae_ptr = delta_t + 0.99 * 0.95 * mask_non_term * gae_ptr
                    batch_adv[t] = gae_ptr
                
                batch_ret = batch_adv + batch_val

            # PPO パラメータ最適化
            f_obs = batch_obs.reshape((-1, TRANSFORMER_SEQ_LEN, O))
            f_log = batch_log.reshape(-1)
            f_act = batch_act.reshape((-1, A))
            f_adv = batch_adv.reshape(-1)
            f_ret = batch_ret.reshape(-1)
            
            for ep_inner in range(UPDATE_EPOCHS):
                idx_shuffle = np.arange(S * E)
                np.random.shuffle(idx_shuffle)
                for ptr in range(0, S * E, MINIBATCH_SIZE):
                    mb_idx = idx_shuffle[ptr : ptr + MINIBATCH_SIZE]
                    
                    _, n_lp, e_v, n_v_v = agent_t.get_action_and_value(f_obs[mb_idx], f_act[mb_idx])
                    
                    # 方策比率
                    ratio = (n_lp - f_log[mb_idx]).exp()
                    mb_adv_raw = f_adv[mb_idx]
                    # アドバンテージ正規化
                    mb_adv_n = (mb_adv_raw - mb_adv_raw.mean()) / (mb_adv_raw.std() + 1e-8)
                    
                    # PPO クリップ損失
                    l_p1 = -mb_adv_n * ratio
                    l_p2 = -mb_adv_n * torch.clamp(ratio, 0.8, 1.2)
                    l_pol = torch.max(l_p1, l_p2).mean()
                    
                    # 状態価値 MSE 損失
                    l_val = 0.5 * ((n_v_v.view(-1) - f_ret[mb_idx]) ** 2).mean()
                    
                    # トータル損失
                    total_loss = l_pol - ENT_COEF * e_v.mean() + 0.5 * l_val
                    
                    optimizer_t.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(agent_t.parameters(), 0.5)
                    optimizer_t.step()
                    
                    l_loss = total_loss.item()
                    l_ent = e_v.mean().item()

            # 表示とログ
            if (TRIAL_MODE) or (u_idx % 10 == 0):
                d_real = time.time() - start_wall_time
                sps_now = int((global_s_idx - start_s_val) / d_real) if d_real > 0 else 0
                
                log_map = {
                    "charts/SPS": sps_now, "losses/total_loss": l_loss, 
                    "losses/entropy": l_ent, "global_step": global_s_idx
                }
                
                if hist_returns:
                    avg_h = np.mean(hist_hidden)
                    avg_c = np.mean(hist_caught)
                    avg_rt = np.mean(hist_returns)
                    log_map.update({
                        "charts/episodic_return": avg_rt, 
                        "charts/steps_hidden": avg_h, "charts/steps_caught": avg_c
                    })
                    print(f"Update {u_idx}, Step {global_s_idx}, SPS: {sps_now}, EpRet: {avg_rt:.1f}, Hidden: {avg_h:.1f}, Caught: {avg_c:.1f}", flush=True)
                    # 次のログに向けてバッファをクリア
                    hist_returns, hist_hidden, hist_caught = [], [], []
                elif not TRIAL_MODE:
                    print(f"Update {u_idx}, Step {global_s_idx}, SPS: {sps_now} (Collecting statistics...)", flush=True)
                
                if TRACK_WANDB:
                    wandb.log(log_map)
                t_writer.add_scalar("charts/SPS", sps_now, global_s_idx)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # 保存処理
    if SAVE_MODEL:
        torch.save(agent_t.state_dict(), SAVE_MODEL_PATH)
        chk_out = SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json')
        with open(chk_out, 'w') as f_o:
            json.dump({'global_step': global_s_idx}, f_o)
        print(f"Model saved to {SAVE_MODEL_PATH}")
    
    vec_envs.close()
    t_writer.close()
    if TRACK_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()