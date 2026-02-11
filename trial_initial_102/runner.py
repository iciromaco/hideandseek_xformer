# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【修正内容 (v25.29)】
# 1. 致命的なバグ (NameError) の修正:
#    - 訓練ループで使用されていた変数名 total_updates_total を、
#      定義名である total_updates_needed に統一しました。
# 2. 変数名の英語維持 (Maintain Continuity):
#    - 過去コードとの連続性を保つため、標準的な英語の変数名を継続。
# 3. PEP 8 完全準拠の絶対独立行展開:
#    - すべてのセミコロン (;) を排除し、1行1ステートメントを徹底。
#    - 代入、条件分岐、ループをすべて複数行に展開。
# 4. チーム統計指標の厳密な反映:
#    - hidden_steps: チーム全員 (Hider1 & Hider2) が同時に隠れている時間。
#    - caught_steps: 少なくとも一人が発見されている時間。
#    - 合計が必ずエピソード長 (300) に一致するロジックを独立行で記述。
# 5. 日本語詳細コメントの完全維持:
#    - ロジックの物理的・数学的意味を省略せず、すべて注釈として復元。

import os
import sys
import platform
import json
import time
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- 実行環境の最適化 ---
# 並列実行時に各プロセスがスレッドを奪い合わないよう、ライブラリ側でシングルスレッド化します。
if platform.processor() != 'arm':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- プロジェクトパスの解決 ---
# 共通モジュール main18_optimization.py 等にアクセスするため、ディレクトリパスを特定します。
current_file_path = os.path.abspath(__file__)
search_path = os.path.dirname(current_file_path)
for _ in range(5):
    # ファイルの存在を確認
    if os.path.exists(os.path.join(search_path, "main18_optimization.py")):
        if search_path not in sys.path:
            sys.path.insert(0, search_path)
        break
    # 一つ上の階層へ遡る
    search_path = os.path.dirname(search_path)

# カレントディレクトリも実行パスに追加
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# ==========================================
# 1. 実験設定 (定数定義)
# ==========================================
# 探索モード: "initial" (新規学習) / "refinement" (既存モデルの微調整)
MODE = "initial" 
EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"
# 学習ターゲット: HIDER チーム (2体)
TRAIN_TARGET = "HIDER" 

EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"
# refinement モード時のみ既存のウェイトファイルを読み込みます
LOAD_EXISTING_MODELS = False

# 実行モード: "TRAIN" (学習) または "PLAY" ( Viewer で挙動を鑑賞)
EXECUTION_MODE = "TRAIN" 

SAVE_MODEL = False
TRACK_WANDB = True
FIXED_SEED = None
TRIAL_MODE = True

# PPO アルゴリズムのハイパーパラメータ
TOTAL_TIMESTEPS = 150000
NUM_ENVS = 8
NUM_STEPS = 128            # 更新1回あたりのデータ収集長さ
LEARNING_RATE = 1.6527918912821367e-05
ENT_COEF = 0.00019532430787861334
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4

# Transformer 設定
TRANSFORMER_SEQ_LEN = 8    # 過去何ステップの記憶を文脈として見るか
HIDDEN_DIM = 64            # 特徴量次元
NUM_LAYERS = 2             # スタック層数
NUM_HEADS = 2              # 注意ヘッド数

# 環境・物理定数
ACTION_REPEAT = 16         # 1アクションあたりの物理エンジン進展回数
PREP_STEPS = 80            # 準備期間 (Seeker待機時間)
MAX_STEPS = 300            # エピソード長
FOV_DEG = 135              # 視野角の設定

# 高速化キャッシュの閾値
LIDAR_CACHE_POS_THRESH = 0.05
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)
RAYCAST_CACHE_POS_THRESH = 0.05

# 観測情報の外れ値設定
# 0.0（原点重なり）と区別するため、フィールドの枠外を意味する 2.0 を採用。
OUTLIER_VALUE = 2.0

# 報酬設計
REWARD_HIDDEN_BONUS = 2.6042974660462024
COS_PENALTY_SCALE = 2.0
REWARD_DISTANCE_DIFF_SCALE = 2.6698493868363062
PENALTY_SAFEGUARD = -20.0        # フィールド外脱走に対する罰
PENALTY_STAGNATION = -0.5        # 停滞（棒立ち）に対する罰

# 移動能力の制限
HIDER_THRUST_LIMIT = 0.40  
SEEKER_THRUST_LIMIT = 0.35 
SEEKER_RB_THRUST = 0.38          # NPC時の推力
SEEKER_RB_TURN_THRESH = np.pi/6    # 旋回時の減速角閾値

# 保存パス
SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# ==========================================
# 2. モデル・バッファ定義 (Agent / ObsHistory)
# ==========================================
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """ネットワーク重みの直交初期化。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """Transformer ベースの Actor-Critic ネットワーク。"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 観測ベクトルの埋め込み
        self.embedding = nn.Linear(obs_dim, HIDDEN_DIM)
        # 時間順序情報を埋め込む位置エンコーディング
        self.pos_encoder = nn.Parameter(torch.zeros(1, TRANSFORMER_SEQ_LEN, HIDDEN_DIM))
        
        # Transformer エンコーダ
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
        
        # 行動平均を出力する Actor 層
        self.actor_mean = layer_init(nn.Linear(HIDDEN_DIM, action_dim), std=0.01)
        # 分散パラメータ
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        # 状態価値を予測する Critic 層
        self.critic = layer_init(nn.Linear(HIDDEN_DIM, 1), std=1)

    def get_value(self, x):
        """状態系列 x から価値 V(s) を返します。"""
        # 特徴抽出プロセス
        x_in = self.embedding(x)
        x_pe = x_in + self.pos_encoder
        x_out = self.transformer(x_pe)
        # 最終トークンを要約として使用
        h_last = x_out[:, -1, :]
        v_val = self.critic(h_last)
        return v_val

    def get_action_and_value(self, x, action=None):
        """行動サンプリングと価値の算出を一括で行います。"""
        # 系列情報を特徴量へ変換
        x_in = self.embedding(x)
        x_pe = x_in + self.pos_encoder
        x_out = self.transformer(x_pe)
        h_context = x_out[:, -1, :]
        
        # 正規分布に基づく確率的ポリシー
        mu = self.actor_mean(h_context)
        logstd = self.actor_logstd.expand_as(mu)
        std = torch.exp(logstd)
        distribution = Normal(mu, std)
        
        if action is None:
            # 訓練時はサンプリング
            action = distribution.sample()
            
        log_prob = distribution.log_prob(action).sum(1)
        entropy = distribution.entropy().sum(1)
        value = self.critic(h_context)
        
        return action, log_prob, entropy, value

class ObsHistory:
    """
    ミラーリングバッファを用いたゼロコピー履歴管理。
    """
    def __init__(self, num_envs, seq_len, obs_dim, device):
        # メモリの連続性を保つために 2倍のバッファを確保
        total_len = seq_len * 2
        self.buffer = torch.zeros((num_envs, total_len, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.ptr = 0

    def reset(self):
        """履歴バッファのゼロ初期化。"""
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        """観測値 obs を 2箇所にミラーリングして書き込みます。"""
        # テンソルへの変換
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # 1次元ベクトルの場合に次元を追加
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        # リング位置とその seq_len 先に書き込む
        self.buffer[:, self.ptr] = obs_tensor
        self.buffer[:, self.ptr + self.seq_len] = obs_tensor
        
        # ポインタのインクリメント
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        """最新 seq_len ステップ分の系列を View として抽出します。"""
        # ミラーリングにより、このスライスは常に時間順に並んでいます
        view = self.buffer[:, self.ptr : self.ptr + self.seq_len]
        return view

# ==========================================
# 3. ヘルパー関数
# ==========================================
def load_model_safely(model_obj, base_name, target_type):
    """候補リストから最新の学習済みモデルファイルを安全にロードします。"""
    import torch
    # 優先順位: 微調整版 > 初期版 > 一般版
    list_to_search = [
        f"{base_name}_refinement_{target_type}.pt",
        f"{base_name}_initial_{target_type}.pt",
        f"{base_name}_{target_type}.pt"
    ]
    
    for file_path in list_to_search:
        if os.path.exists(file_path):
            try:
                # CPU マッピングで安全にロード
                weights = torch.load(file_path, map_location="cpu")
                model_obj.load_state_dict(weights)
                # 推論モードへの移行
                model_obj.eval()
                return file_path
            except Exception:
                continue
    return None

# ==========================================
# 4. 環境作成用ファクトリ
# ==========================================
def create_env(render_mode=None):
    """
    MuJoCo の初期化を並列ワーカー起動時まで遅らせることで、
    Linux 等での CUDA デッドロックを回避します。
    """
    import torch
    import mujoco
    import gymnasium as gym
    import main18_optimization as base_config

    class TeamCosEnv(base_config.HideAndSeekEnv):
        """
        視界勾配、チーム評価、外れ値マスク、停滞罰を実装した決定版環境クラス。
        """
        def __init__(self, render_mode=None):
            # 親クラスの初期化
            super().__init__(render_mode=render_mode)
            
            cpu_device = torch.device("cpu")
            # 各エージェント個別の履歴バッファ
            self.npc_obs_history = {
                0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
                1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
                2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device)
            }
            
            # 高速化キャッシュ類
            self.visible_cache = {0: {}, 1: {}, 2: {}}
            self.lidar_array_cache = {} 
            self.raycast_cache = {} 
            self.raycast_stats = {"hits": 0, "misses": 0}
            
            # 統計用変数
            self.hidden_steps_count = 0
            self.caught_steps_count = 0 
            # 前ステップのシーカーとの距離（報酬計算用）
            self.prev_dist = {1: 0.0, 2: 0.0}
            
            # ★追加: 停滞ペナルティ用の座標バッファ
            self.prev_hider_xy = {1: None, 2: None}
            
            # ステップ内重複計算回避用のメモ化
            self._obs_memo = {}
            # スタック復帰時の転回方向
            self.s0_recovery_turn_dir = 1.0

            # NPC 用のモデルを作成
            self.npc_hider_agent = Agent(53, 4).to("cpu")
            self.npc_seeker_agent = Agent(53, 4).to("cpu")
            
            # 最初のワーカープロセスのみモデル読み込みログを表示
            is_logged_now = os.environ.get("NPC_MODELS_LOGGED")
            should_print_log = (is_logged_now != "TRUE")

            # HIDER ロード
            path_h = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
            if path_h:
                if should_print_log:
                    print(f"Loaded NPC Hider from {path_h}", flush=True)
            else:
                self.npc_hider_agent = None

            # SEEKER ロード
            path_s = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
            if path_s:
                if should_print_log:
                    print(f"Loaded NPC Seeker from {path_s}", flush=True)
            else:
                self.npc_seeker_agent = None

            # ログ表示済みフラグのセット
            if should_print_log:
                os.environ["NPC_MODELS_LOGGED"] = "TRUE"

        def _is_visible(self, origin_pos, origin_rot, target_pos, target_body_id, exclude_body_id):
            """視野角と物理的な遮蔽を計算し、対象が見えているか判定します。"""
            diff_vec = target_pos[:2] - origin_pos[:2]
            distance_val = np.linalg.norm(diff_vec)
            
            # 至近距離であれば常に可視
            if distance_val < 0.1:
                return True, target_body_id
            
            # 1. 視野角 (FOV) チェック
            angle_to_target = np.arctan2(diff_vec[1], diff_vec[0])
            angle_relative = (angle_to_target - origin_rot + np.pi) % (2 * np.pi) - np.pi
            if abs(angle_relative) > np.deg2rad(FOV_DEG / 2.0):
                return False, -1
            
            # 2. RayCast による物理遮蔽チェック
            direction_vec = np.array([diff_vec[0]/distance_val, diff_vec[1]/distance_val, 0.0], dtype=np.float64)
            # レイの始点を腰の高さ (0.5m) に配置
            origin_pt = np.array([origin_pos[0], origin_pos[1], 0.5], dtype=np.float64)
            geom_out_arr = np.zeros(1, dtype=np.int32)
            
            # mujoco.mj_ray 実行
            hit_dist = mujoco.mj_ray(
                self.model, 
                self.data, 
                origin_pt, 
                direction_vec, 
                None, 
                1, 
                exclude_body_id, 
                geom_out_arr
            )
            
            if hit_dist != -1:
                hit_body_id = self.model.geom_bodyid[geom_out_arr[0]]
                # ターゲット自体にヒットした場合
                if hit_body_id == target_body_id:
                    return True, target_body_id
                # ターゲットより手前（40cm以上の余裕）で何かにヒットした場合
                if hit_dist < distance_val - 0.4:
                    return False, hit_body_id
            
            # 遮蔽なし
            return True, target_body_id

        def _check_collision_all(self, pos, threshold):
            """指定座標が障害物内部にあるか判定します (AttributeError対策)。"""
            # 外壁
            for wall_p, wall_s in self.wall_data:
                dx_w = abs(pos[0] - wall_p[0]) - wall_s[0]
                dy_w = abs(pos[1] - wall_p[1]) - wall_s[1]
                # 矩形境界からの距離の二乗を算出
                dist_sq = max(dx_w, 0.0)**2 + max(dy_w, 0.0)**2
                if dist_sq < threshold**2:
                    return True
            
            # フィールド内オブジェクト (Box/Ramp)
            all_obj_geoms = self.box_geoms + self.ramp_all_geoms
            for g_id in all_obj_geoms:
                o_center = self.data.geom_xpos[g_id][:2]
                o_size = self.model.geom(g_id).size[:2]
                dx_o = abs(pos[0] - o_center[0]) - o_size[0]
                dy_o = abs(pos[1] - o_center[1]) - o_size[1]
                dist_obj_sq = max(dx_o, 0.0)**2 + max(dy_o, 0.0)**2
                if dist_obj_sq < threshold**2:
                    return True
            
            return False

        def _get_cached_ray(self, agent_id, origin_p, direction, beam_id):
            """mj_ray の呼び出しを座標ベースでキャッシュし、物理演算負荷を下げます。"""
            angle_val = np.arctan2(direction[1], direction[0])
            cache_key = (agent_id, beam_id)
            
            # 前回からの座標と角度の変動が少なければキャッシュを使用
            if cache_key in self.raycast_cache:
                c_p, c_a, c_res, c_gid = self.raycast_cache[cache_key]
                dist_moved = np.linalg.norm(origin_p - c_p)
                if dist_moved < RAYCAST_CACHE_POS_THRESH:
                    angle_err = (angle_val - c_a + np.pi) % (2 * np.pi) - np.pi
                    if abs(angle_err) < 0.05:
                        self.raycast_stats["hits"] = self.raycast_stats["hits"] + 1
                        return c_res, c_gid
            
            # キャッシュミス時
            self.raycast_stats["misses"] = self.raycast_stats["misses"] + 1
            hit_res_arr = np.zeros(1, dtype=np.int32)
            ray_fr_vec = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
            ray_dr_vec = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
            
            # 自己遮蔽の除外ボディ
            if agent_id == 0:
                exclude_bid = self.s0_body
            elif agent_id == 1:
                exclude_bid = self.h1_body
            else:
                exclude_bid = self.h2_body
                
            dist_found = mujoco.mj_ray(
                self.model, 
                self.data, 
                ray_fr_vec, 
                ray_dr_vec, 
                None, 
                1, 
                exclude_bid, 
                hit_res_arr
            )
            
            # キャッシュへ保存
            self.raycast_cache[cache_key] = (origin_p.copy(), angle_val, dist_found, hit_res_arr[0])
            return dist_found, hit_res_arr[0]

        def _get_obs(self, agent_id):
            """
            53次元の観測ベクトルを、外れ値 2.0 マスクを適用して生成します。
            PEP 8 形式で代入を分解して記述します。
            """
            # メモ化チェック
            if agent_id in self._obs_memo:
                return self._obs_memo[agent_id]
                
            # エージェント種別の解決
            if agent_id == 0:
                body_id = self.s0_body
                prefix = 's'
            elif agent_id == 1:
                body_id = self.h1_body
                prefix = 'h1'
            else:
                body_id = self.h2_body
                prefix = 'h2'
            
            # 物理状態の取得
            pos_xy = self.data.xpos[body_id][:2]
            q_joint_id = self.model.joint(f'{prefix}_rot').id
            q_adr_val = self.model.jnt_qposadr[q_joint_id]
            angle_rad = self.data.qpos[q_adr_val]
            
            # ローカル座標系への回転行列
            cos_r = np.cos(-angle_rad)
            sin_r = np.sin(-angle_rad)
            rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            
            # 速度正規化
            jx_id = self.model.joint(f'{prefix}_x').id
            dof_ptr = self.model.jnt_dofadr[jx_id]
            v_global = self.data.qvel[dof_ptr : dof_ptr + 2]
            v_local = rot_matrix @ v_global
            v_obs = v_local / 12.0
            
            # 自身の状態ベクトル [vx, vy, rad, cos, sin]
            self_state = np.concatenate([
                v_obs, 
                [angle_rad, np.cos(angle_rad), np.sin(angle_rad)]
            ])
            
            # Lidar 情報の生成 (キャッシュ適用)
            lidar_out = None
            cached_l = self.lidar_array_cache.get(agent_id)
            if cached_l is not None:
                c_p, c_r, c_lidar = cached_l
                dist_moved = np.linalg.norm(pos_xy - c_p)
                angle_moved = abs((angle_rad - c_r + np.pi) % (2 * np.pi) - np.pi)
                if dist_moved < LIDAR_CACHE_POS_THRESH:
                    if angle_moved < LIDAR_CACHE_ANG_THRESH:
                        lidar_out = c_lidar
            
            if lidar_out is None:
                lidar_out = np.zeros(len(self.lidar_angles), dtype=np.float32)
                for i_beam, angle_off in enumerate(self.lidar_angles):
                    beam_angle = angle_off + angle_rad
                    beam_dir = np.array([np.cos(beam_angle), np.sin(beam_angle)])
                    res_dist, _ = self._get_cached_ray(agent_id, pos_xy, beam_dir, i_beam + 100)
                    # 2.5m を 1.0 に正規化
                    lidar_out[i_beam] = min(res_dist, 2.5) / 2.5 if res_dist != -1 else 1.0
                # キャッシュ更新
                self.lidar_array_cache[agent_id] = (pos_xy.copy(), angle_rad, lidar_out.copy())

            # 視界判定キャッシュの構築
            current_vis = self.visible_cache[agent_id]
            current_vis.clear()
            obj_candidates = [self.box1_body, self.box2_body, self.ramp_body, self.h1_body, self.h2_body, self.s0_body]
            for target_id in obj_candidates:
                if target_id != body_id:
                    v_flag, _ = self._is_visible(self.data.xpos[body_id], angle_rad, self.data.xpos[target_id], target_id, body_id)
                    current_vis[target_id] = v_flag

            def get_rel_info(target_id, lock_state=None):
                """相対的位置・速度・回転を生成。不可視時は外れ値 2.0 マスクを採用。"""
                is_seen = current_vis.get(target_id, False)
                vector_size = 8 if lock_state is not None else 7
                
                if is_seen:
                    t_xyz = self.data.xpos[target_id]
                    rel_p = rot_matrix @ (t_xyz[:2] - pos_xy) / 12.0
                    
                    q_val = self.data.xquat[target_id]
                    t_yaw = np.arctan2(
                        2 * (q_val[0] * q_val[3] + q_val[1] * q_val[2]), 
                        1 - 2 * (q_val[2]**2 + q_val[3]**2)
                    )
                    
                    j_adr = self.model.body_jntadr[target_id]
                    t_v_global = self.data.qvel[j_adr : j_adr + 2] if j_adr != -1 else np.zeros(2)
                    rel_v = rot_matrix @ (t_v_global - v_global) / 12.0
                    
                    parts = [rel_p, rel_v, [np.cos(t_yaw - angle_rad), np.sin(t_yaw - angle_rad)]]
                    if lock_state is not None:
                        f_lock = 1.0 if lock_state else 0.0
                        parts.append([f_lock])
                    
                    # 可視フラグ
                    parts.append([1.0])
                    return np.concatenate(parts)
                else:
                    # 不可視な場合：数値計算の原点重なりと区別するため外れ値 2.0 で埋める
                    masked_arr = np.full(vector_size, OUTLIER_VALUE, dtype=np.float32)
                    # 最後の要素（可視フラグ）のみ、情報の不在を示す 0.0 をセット
                    masked_arr[-1] = 0.0
                    return masked_arr

            # 役割に応じた結合処理
            if agent_id == 0:
                # シーカー視点
                H1_rel = get_rel_info(self.h1_body)[:5]
                H2_rel = get_rel_info(self.h2_body)[:5]
                objs = [
                    get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    get_rel_info(self.ramp_body)
                ]
                obs_final = np.concatenate([self_state, lidar_out, *objs, H1_rel, H2_rel, np.zeros(3, dtype=np.float32)])
            else:
                # ハイダー視点
                p_id = self.h2_body if agent_id == 1 else self.h1_body
                enemy_rel = get_rel_info(self.s0_body)[:5]
                friend_rel = get_rel_info(p_id)
                grasp_f = 1.0 if self.grasping[agent_id] is not None else 0.0
                objs = [
                    get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    get_rel_info(self.ramp_body)
                ]
                obs_final = np.concatenate([self_state, lidar_out, *objs, enemy_rel, friend_rel, [grasp_f]])

            obs_output = obs_final.astype(np.float32)
            self._obs_memo[agent_id] = obs_output
            return obs_output

        def _update_seeker_state(self):
            """NPCシーカーのターゲット地点と探索モードの状態遷移を管理します。"""
            s_pos_xy = self.data.xpos[self.s0_body][:2]
            
            # 視覚情報の更新
            self._get_obs(0)
            vis_h1 = self.visible_cache[0].get(self.h1_body, False)
            vis_h2 = self.visible_cache[0].get(self.h2_body, False)
            
            if vis_h1 or vis_h2:
                # 目撃中：発見した座標を追跡
                target_bid = self.h1_body if vis_h1 else self.h2_body
                target_pos = self.data.xpos[target_bid][:2].copy()
                self.seeker_target_pos = target_pos
                self.seeker_last_known_pos = target_pos.copy()
                self.seeker_mode = "CHASING"
            elif self.seeker_last_known_pos is not None:
                # 見失った直後：手がかりのある場所へ急行
                dist_to_mem = np.linalg.norm(s_pos_xy - self.seeker_last_known_pos)
                if dist_to_mem > 0.5:
                    self.seeker_target_pos = self.seeker_last_known_pos.copy()
                    self.seeker_mode = "SEARCHING"
                else:
                    # 捜索地点に到着したが居なかった場合
                    self.seeker_last_known_pos = None
                    self.seeker_search_timer = 50
            else:
                # 手がかりなし：ランダムに巡回
                if self.seeker_search_timer <= 0:
                    self.seeker_random_target = self.np_random.uniform(-4, 4, 2)
                    self.seeker_search_timer = 80
                
                self.seeker_search_timer = self.seeker_search_timer - 1
                self.seeker_target_pos = self.seeker_random_target.copy()
                self.seeker_mode = "PATROLLING"

        def _seeker_rule_based_policy(self):
            """最短方位角へ向かうシンプルなルールベース制御。"""
            if self.current_step < PREP_STEPS:
                return 0.0, 0.0
            
            s_xy = self.data.xpos[self.s0_body][:2]
            s_rot_val = self.data.qpos[self.srot_adr]
            target_xy = self.seeker_target_pos
            
            # 目的地へのベクトルと最短角
            dx = target_xy[0] - s_xy[0]
            dy = target_xy[1] - s_xy[1]
            target_angle = np.arctan2(dy, dx)
            
            # 偏差を -pi 〜 pi の範囲に正規化
            angle_diff = (target_angle - s_rot_val + np.pi) % (2 * np.pi) - np.pi
            
            thrust_out = SEEKER_RB_THRUST
            turn_out = np.clip(angle_diff * 6.0, -3.0, 3.0)
            
            # 向きが大きくずれている場合は前進速度を落とす
            if abs(angle_diff) > SEEKER_RB_TURN_THRESH:
                thrust_out = thrust_out * 0.3
            
            # 物理的なスタックの検知 (速度監視)
            sx_joint_id = self.model.joint('s_x').id
            sx_dof_ptr = self.model.jnt_dofadr[sx_joint_id]
            current_v = np.linalg.norm(self.data.qvel[sx_dof_ptr : sx_dof_ptr + 2])
            
            if thrust_out > 0.05 and current_v < 0.05:
                self.s0_stuck_timer = self.s0_stuck_timer + 5
            else:
                self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            
            # スタック復帰モードの開始
            if self.s0_stuck_timer > 15:
                self.s0_recovery_mode = 15
                self.s0_stuck_timer = 0
                self.s0_recovery_turn_dir = self.np_random.choice([-1.0, 1.0])
            
            if self.s0_recovery_mode > 0:
                thrust_out = -0.2 # 後退
                turn_out = 1.5 * self.s0_recovery_turn_dir # 回避旋回
                self.s0_recovery_mode = self.s0_recovery_mode - 1
                
            return float(thrust_out), float(turn_out)

        def _get_npc_action(self, agent_id, agent_type):
            """NPCエージェントのアクションを生成。モデル優先、なければルール/ランダム。"""
            import torch
            # 現在の観測取得と履歴バッファの更新
            obs_raw = self._get_obs(agent_id)
            self.npc_obs_history[agent_id].update(obs_raw)
            
            # ロード済みモデルの選択
            if agent_type == "HIDER":
                target_model = self.npc_hider_agent
            else:
                target_model = self.npc_seeker_agent
                
            # モデル推論が可能な場合
            if target_model is not None:
                with torch.no_grad():
                    context_series = self.npc_obs_history[agent_id].get()
                    # 出力は正規化済みアクション [-1.0, 1.0]
                    act_tensor, _, _, _ = target_model.get_action_and_value(context_series)
                return act_tensor.cpu().numpy()[0]
            
            # Seeker モデルなし時のルールベース
            if agent_type == "SEEKER":
                f_rb, r_rb = self._seeker_rule_based_policy()
                # step関数側の期待（LIMIT適用前）に合わせて正規化
                norm_f = f_rb / SEEKER_THRUST_LIMIT
                return np.array([norm_f, r_rb, 0.0, 0.0], dtype=np.float32)
            
            # Hider モデルなし時のランダム行動
            return self.action_space.sample() * 0.5

        def reset(self, seed=None, options=None):
            """物理環境、統計カウンタ、座標履歴を初期化します。"""
            obs, info = super().reset(seed=seed, options=options)
            
            self.hidden_steps_count = 0
            self.caught_steps_count = 0
            self._obs_memo.clear()
            self.lidar_array_cache.clear()
            self.s0_recovery_turn_dir = 1.0
            
            # チーム評価計算用の初期データを保存
            seeker_xy_init = self.data.xpos[self.s0_body][:2]
            for i_h in [1, 2]:
                body_id = self.h1_body if i_h == 1 else self.h2_body
                pos_xy_init = self.data.xpos[body_id][:2].copy()
                self.prev_dist[i_h] = np.linalg.norm(pos_xy_init - seeker_xy_init)
                # 停滞判定用
                self.prev_hider_xy[i_h] = pos_xy_init
            
            return obs, info

        def step(self, action):
            """1ステップの進行 (アクション適用 -> 物理演算 -> チーム統計更新)。"""
            self._obs_memo.clear()
            self.current_step = self.current_step + 1
            
            # ロッククールダウンの更新
            for i_h in [1, 2]:
                curr_cd = self.lock_cooldown[i_h]
                self.lock_cooldown[i_h] = max(0, curr_cd - 1)
            
            # Seeker NPC の思考
            self._update_seeker_state()
            # 入力バッファのクリア
            self.data.ctrl[:] = 0.0 
            
            if TRAIN_TARGET == "HIDER":
                # 学習エージェント (Hider)
                m_ctrl_idx = self._apply_action(self.learning_agent_id, action)
                self.data.ctrl[m_ctrl_idx] = float(action[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[m_ctrl_idx + 1] = float(action[1])
                
                # 相棒 NPC (Hider)
                partner_id = 2 if self.learning_agent_id == 1 else 1
                act_partner = self._get_npc_action(partner_id, "HIDER")
                p_ctrl_idx = self._apply_action(partner_id, act_partner)
                self.data.ctrl[p_ctrl_idx] = float(act_partner[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[p_ctrl_idx + 1] = float(act_partner[1])
                
                # 敵 NPC (Seeker - AI優先)
                act_seeker_npc = self._get_npc_action(0, "SEEKER")
                self.data.ctrl[0] = float(act_seeker_npc[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(act_seeker_npc[1])
            else:
                # 学習エージェント (Seeker)
                self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(action[1])
                
                # 敵 NPC (Hiders x2)
                for i_h in [1, 2]:
                    act_hider_npc = self._get_npc_action(i_h, "HIDER")
                    h_ctrl_idx = self._apply_action(i_h, act_hider_npc)
                    self.data.ctrl[h_ctrl_idx] = float(act_hider_npc[0]) * HIDER_THRUST_LIMIT
                    self.data.ctrl[h_ctrl_idx + 1] = float(act_hider_npc[1])
            
            # --- 物理進展ループ (Action Repeat) ---
            for _ in range(ACTION_REPEAT):
                # ロック状態の物理強制固定 (Velocity Freeze)
                for body_target_id, pose_fixed in self.locked_pose.items():
                    if self.locked_boxes[body_target_id]:
                        target_jid = self.box1_joint_id if body_target_id == self.box1_body else self.box2_joint_id
                        q_idx_val = self.model.jnt_qposadr[target_jid]
                        dof_idx_val = self.model.jnt_dofadr[target_jid]
                        # 座標の上書き
                        self.data.qpos[q_idx_val : q_idx_val + 7] = pose_fixed
                        # 速度を完全に抹消して反発を根絶
                        self.data.qvel[dof_idx_val : dof_idx_val + 6] = 0
                # 物理エンジンの進展
                mujoco.mj_step(self.model, self.data)
            
            # --- 統計および報酬の確定 ---
            self._obs_memo.clear()
            obs_learner = self._get_obs(self.learning_agent_id)
            # 統計収集のためにシーカー視界を更新
            _ = self._get_obs(0) 
            
            total_team_reward = 0.0
            
            # ★チーム生存指標の計算 (Team concealment check)
            visible_h1 = self.visible_cache[0].get(self.h1_body, False)
            visible_h2 = self.visible_cache[0].get(self.h2_body, False)
            is_any_visible = visible_h1 or visible_h2
            
            if is_any_visible:
                # チームの誰かが発見されている状態をカウント
                self.caught_steps_count = self.caught_steps_count + 1
            else:
                # チーム全員が完全に隠蔽されている状態をカウント
                self.hidden_steps_count = self.hidden_steps_count + 1
                
            # 各個体への報酬集計
            for hider_id_val, body_target_id in [(1, self.h1_body), (2, self.h2_body)]:
                is_currently_seen = self.visible_cache[0].get(body_target_id, False)
                seeker_xy_now = self.data.xpos[self.s0_body][:2]
                hider_xy_now = self.data.xpos[body_target_id][:2]
                distance_now = np.linalg.norm(hider_xy_now - seeker_xy_now)
                
                if is_currently_seen:
                    # 真正面にいるほど大きなマイナス報酬 (-cos)
                    s_rot_now = self.data.qpos[self.srot_adr]
                    vec_rel = hider_xy_now - seeker_xy_now
                    vec_unit = vec_rel / (np.linalg.norm(vec_rel) + 1e-8)
                    vec_fwd = np.array([np.cos(s_rot_now), np.sin(s_rot_now)])
                    cos_theta = np.dot(vec_unit, vec_fwd)
                    current_reward = -cos_theta * COS_PENALTY_SCALE
                    # 逃走（距離拡大）に対するわずかな加算
                    dist_diff_val = distance_now - self.prev_dist[hider_id_val]
                    current_reward = current_reward + dist_diff_val * REWARD_DISTANCE_DIFF_SCALE
                else:
                    # 全く見えていない時の隠蔽報酬
                    current_reward = REWARD_HIDDEN_BONUS
                
                # ★停滞ペナルティ (学習対象個体にのみ課して積極性を促す)
                if hider_id_val == self.learning_agent_id:
                    if self.prev_hider_xy[hider_id_val] is not None:
                        dist_moved_now = np.linalg.norm(hider_xy_now - self.prev_hider_xy[hider_id_val])
                        if dist_moved_now < 0.01: # 1cm未満の動き
                            current_reward = current_reward + PENALTY_STAGNATION
                    # 座標の保存更新
                    self.prev_hider_xy[hider_id_val] = hider_xy_now.copy()
                
                # 場外への脱走に対する保険ペナルティ
                if max(abs(hider_xy_now)) > 6.5:
                    current_reward = current_reward + PENALTY_SAFEGUARD
                
                total_team_reward = total_team_reward + current_reward
                self.prev_dist[hider_id_val] = distance_now
                
            # 最終報酬の確定
            final_rew_val = total_team_reward if TRAIN_TARGET == "HIDER" else -total_team_reward
            is_episode_truncated = (self.current_step >= MAX_STEPS)
            
            # 情報パケットの作成
            step_info_packet = {
                "hidden_steps": float(self.hidden_steps_count), 
                "caught_steps": float(self.caught_steps_count)
            }
            return obs_learner, float(final_rew_val), False, is_episode_truncated, step_info_packet

    return TeamCosEnv(render_mode=render_mode)

# ==========================================
# 5. メイン処理 (学習ループ)
# ==========================================
def main():
    # --- 1. プロセス起動設定 (Linux デッドロック回避の核心) ---
    if platform.system() == "Linux":
        try:
            # 絶対独立行：spawn 方式を強制指定
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # 並列ワーカーの生成用ファクトリ
    def create_parallel_worker():
        # create_env 内部で MuJoCo/Torch を遅延ロード
        env_ins = create_env()
        import gymnasium as gym
        monitor_wrapped_env = gym.wrappers.RecordEpisodeStatistics(env_ins)
        return monitor_wrapped_env
    
    import gymnasium as gym
    print(f"--- [Parent] 1. Initializing {NUM_ENVS} parallel workers ---", flush=True)
    
    # ワーカーが立ち上がるまで親プロセスは CUDA コンテキストに触れない
    try:
        parallel_envs = gym.vector.AsyncVectorEnv([create_parallel_worker for _ in range(NUM_ENVS)])
        print("--- [Parent] 2. Parallel environments initialized ---", flush=True)
    except Exception as startup_err:
        print(f"--- [Parent] [CRITICAL] Parallel startup failed: {startup_err} ---", flush=True)
        sys.exit(1)

    # --- 3. 親プロセスでのライブラリ初期化 (ワーカー起動後に解禁) ---
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb
    import main18_optimization as base_config

    device_actual = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")
    current_ts = int(time.time())
    exp_display_name = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{current_ts}"
    
    # 鑑賞モード (PLAY) の実行
    if EXECUTION_MODE == "PLAY":
        print(f"--- Inference Mode (PLAY) ---")
        play_env = create_env(render_mode="human")
        play_agent = Agent(play_env.observation_space.shape[0], play_env.action_space.shape[0]).to(device_actual)
        file_loaded = load_model_safely(play_agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if file_loaded:
            print(f"Successfully loaded: {file_loaded}")
        
        play_agent.eval()
        play_history = ObsHistory(1, TRANSFORMER_SEQ_LEN, play_env.observation_space.shape[0], device_actual)
        dt_step_real = 0.005 * ACTION_REPEAT
        
        try:
            while True:
                obs_arr, _ = play_env.reset()
                play_history.reset()
                play_history.update(obs_arr)
                is_done_play = False
                total_rew_sum = 0.0
                while not is_done_play:
                    t_loop_start = time.time()
                    with torch.no_grad():
                        series_data = play_history.get()
                        act_arr, _, _, _ = play_agent.get_action_and_value(series_data)
                    
                    obs_next, r_val, term, trunc, info_val = play_env.step(act_arr.cpu().numpy()[0])
                    is_done_play = term or trunc
                    total_rew_sum = total_rew_sum + r_val
                    play_history.update(obs_next)
                    play_env.render()
                    
                    # 再生速度の調整
                    t_elapsed = time.time() - t_loop_start
                    t_wait = dt_step_real - t_elapsed
                    if t_wait > 0:
                        time.sleep(t_wait)
                    
                    if play_env.viewer is not None:
                        if not play_env.viewer.is_running():
                            return
                
                print(f"Return: {total_rew_sum:.1f}, Team Hidden Steps: {info_val['hidden_steps']:.0f}")
        except KeyboardInterrupt:
            pass
        finally:
            play_env.close()
        return

    # 学習モード (TRAIN) の初期化
    if TRACK_WANDB:
        wandb_run = wandb.init(
            project=base_config.WANDB_PROJECT_NAME, 
            config={"Target": TRAIN_TARGET, "MODE": MODE, "v": "25.29_fully_unrolled_team_fixed"}, 
            name=exp_display_name, 
            sync_tensorboard=False,
            save_code=True
        )
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("*", step_metric="global_step")

    tb_writer = SummaryWriter(f"runs/{exp_display_name}")
    train_target_agent = Agent(parallel_envs.single_observation_space.shape[0], parallel_envs.single_action_space.shape[0]).to(device_actual)
    optimizer_obj = optim.Adam(train_target_agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    global_step_counter = 0
    start_step_offset_val = 0
    if LOAD_EXISTING_MODELS:
        latest_file_path = load_model_safely(train_target_agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if latest_file_path:
            print(f"★ Resumed from: {latest_file_path}")
            checkpoint_file = latest_file_path.replace('.pt', '_checkpoint.json')
            if os.path.exists(checkpoint_file):
                try:
                    with open(checkpoint_file, 'r') as f_checkpoint:
                        json_data = json.load(f_checkpoint)
                        global_step_counter = json_data.get('global_step', 0)
                        start_step_offset_val = global_step_counter
                except Exception:
                    pass

    # 訓練用バッファの確保
    train_history_buffer = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device_actual)
    S_len = NUM_STEPS
    E_num = NUM_ENVS
    O_dim = 53
    A_dim = 4
    
    batch_obs_storage = torch.zeros((S_len, E_num, TRANSFORMER_SEQ_LEN, O_dim), device=device_actual)
    batch_act_storage = torch.zeros((S_len, E_num, A_dim), device=device_actual)
    batch_log_storage = torch.zeros((S_len, E_num), device=device_actual)
    batch_rew_storage = torch.zeros((S_len, E_num), device=device_actual)
    batch_done_storage = torch.zeros((S_len, E_num), device=device_actual)
    batch_val_storage = torch.zeros((S_len, E_num), device=device_actual)
    
    # 環境の初期リセット
    seed_actual = FIXED_SEED if FIXED_SEED else int(time.time())
    obs_next_arr, _ = parallel_envs.reset(seed=seed_actual)
    done_next_tensor = torch.zeros(E_num).to(device_actual)
    train_history_buffer.reset()
    train_history_buffer.update(obs_next_arr)
    
    total_updates_needed = int(max(1, (TOTAL_TIMESTEPS - global_step_counter) // (E_num * S_len)))
    list_returns = []
    list_team_hidden = []
    list_team_caught = []
    start_real_clock = time.time()
    last_known_loss = 0.0
    last_known_entropy = 0.0

    print(f"--- Training loop starting (v25.29) ---")
    try:
        for update_idx in tqdm(range(1, total_updates_needed + 1), desc="Updates"):
            # データ収集フェーズ (Rollout)
            for step_ptr in range(S_len):
                global_step_counter = global_step_counter + E_num
                batch_obs_storage[step_ptr] = train_history_buffer.get()
                batch_done_storage[step_ptr] = done_next_tensor
                
                with torch.no_grad():
                    a_out, lp_out, _, v_out = train_target_agent.get_action_and_value(train_history_buffer.get())
                    batch_val_storage[step_ptr] = v_out.flatten()
                
                batch_act_storage[step_ptr] = a_out
                batch_log_storage[step_ptr] = lp_out
                
                # 並列環境を進める
                obs_next_arr, rewards_arr, term_arr, trunc_arr, info_dict = parallel_envs.step(a_out.cpu().numpy())
                done_mask_arr = np.logical_or(term_arr, trunc_arr)
                
                # 統計の抽出処理
                if "final_info" in info_dict:
                    for env_idx in range(E_num):
                        item_info = info_dict["final_info"][env_idx]
                        if done_mask_arr[env_idx]:
                            if item_info is not None:
                                if "episode" in item_info:
                                    list_returns.append(float(item_info["episode"]["r"]))
                                if "hidden_steps" in item_info:
                                    list_team_hidden.append(float(item_info["hidden_steps"]))
                                if "caught_steps" in item_info:
                                    list_team_caught.append(float(item_info["caught_steps"]))
                elif "episode" in info_dict:
                    episode_mask = info_dict.get("_episode", [True] * E_num)
                    for env_idx in range(E_num):
                        if episode_mask[env_idx]:
                            if done_mask_arr[env_idx]:
                                list_returns.append(float(info_dict["episode"]["r"][env_idx]))
                                try:
                                    if "hidden_steps" in info_dict:
                                        list_team_hidden.append(float(info_dict["hidden_steps"][env_idx]))
                                    if "caught_steps" in info_dict:
                                        list_team_caught.append(float(info_dict["caught_steps"][env_idx]))
                                except Exception:
                                    pass
                
                batch_rew_storage[step_ptr] = torch.tensor(rewards_arr).to(device_actual).view(-1)
                done_next_tensor = torch.tensor(done_mask_arr).to(device_actual, dtype=torch.float32)
                train_history_buffer.update(obs_next_arr)

            # 利得の計算 (GAE)
            with torch.no_grad():
                v_next_pred_val = train_target_agent.get_value(train_history_buffer.get()).reshape(1, -1)
                batch_adv_storage = torch.zeros_like(batch_rew_storage).to(device_actual)
                ptr_gae = 0
                for t in reversed(range(S_len)):
                    if t == S_len - 1:
                        non_term_flag = 1.0 - done_next_tensor
                        next_value_pred = v_next_pred_val
                    else:
                        non_term_flag = 1.0 - batch_done_storage[t + 1]
                        next_value_pred = batch_val_storage[t + 1]
                    
                    # delta = 報酬 + 割引率 * 未来価値 - 現在価値
                    delta_val = batch_rew_storage[t] + 0.99 * next_value_pred * non_term_flag - batch_val_storage[t]
                    # 再帰的なアドバンテージ加重
                    ptr_gae = delta_val + 0.99 * 0.95 * non_term_flag * ptr_gae
                    batch_adv_storage[t] = ptr_gae
                
                batch_ret_storage = batch_adv_storage + batch_val_storage

            # バッファの整形と平坦化
            flat_obs_batch = batch_obs_storage.reshape((-1, TRANSFORMER_SEQ_LEN, O_dim))
            flat_log_batch = batch_log_storage.reshape(-1)
            flat_act_batch = batch_act_storage.reshape((-1, A_dim))
            flat_adv_batch = batch_adv_storage.reshape(-1)
            flat_ret_batch = batch_ret_storage.reshape(-1)
            
            # PPO パラメータ更新ループ
            for epoch_ptr in range(UPDATE_EPOCHS):
                idx_pool = np.arange(S_len * E_num)
                np.random.shuffle(idx_pool)
                for start_ptr in range(0, S_len * E_num, MINIBATCH_SIZE):
                    batch_indices_val = idx_pool[start_ptr : start_ptr + MINIBATCH_SIZE]
                    
                    _, new_log_p, ent_batch, new_v_batch = train_target_agent.get_action_and_value(
                        flat_obs_batch[batch_indices_val], 
                        flat_act_batch[batch_indices_val]
                    )
                    
                    # 変化率の計算
                    prob_ratio_val = (new_log_p - flat_log_batch[batch_indices_val]).exp()
                    adv_batch_val = flat_adv_batch[batch_indices_val]
                    # アドバンテージの標準化
                    adv_norm_val = (adv_batch_val - adv_batch_val.mean()) / (adv_batch_val.std() + 1e-8)
                    
                    # PPO クリップ損失関数
                    loss_part1 = -adv_norm_val * prob_ratio_val
                    loss_part2 = -adv_norm_val * torch.clamp(prob_ratio_val, 0.8, 1.2)
                    policy_loss_val = torch.max(loss_part1, loss_part2).mean()
                    
                    # 状態価値予測の MSE 誤差
                    value_loss_val = 0.5 * ((new_v_batch.view(-1) - flat_ret_batch[batch_indices_val]) ** 2).mean()
                    
                    # 合計損失の統合
                    total_loss_val = policy_loss_val - ENT_COEF * ent_batch.mean() + 0.5 * value_loss_val
                    
                    optimizer_obj.zero_grad()
                    total_loss_val.backward()
                    nn.utils.clip_grad_norm_(train_target_agent.parameters(), 0.5)
                    optimizer_obj.step()
                    
                    last_known_loss = total_loss_val.item()
                    last_known_entropy = ent_batch.mean().item()

            # ログ表示および WandB 送信
            if (TRIAL_MODE) or (update_idx % 10 == 0):
                time_now_val = time.time()
                elapsed_seconds = time_now_val - start_real_clock
                current_sps_val = int((global_step_counter - start_step_offset_val) / elapsed_seconds) if elapsed_seconds > 0 else 0
                
                log_metrics_map = {
                    "charts/SPS": current_sps_val, 
                    "losses/total_loss": last_known_loss, 
                    "losses/entropy": last_known_entropy, 
                    "global_step": global_step_counter
                }
                
                if list_returns:
                    avg_team_hidden_val = np.mean(list_team_hidden)
                    avg_team_caught_val = np.mean(list_team_caught)
                    avg_returns_val = np.mean(list_returns)
                    log_metrics_map.update({
                        "charts/episodic_return": avg_returns_val, 
                        "charts/steps_hidden": avg_team_hidden_val, 
                        "charts/steps_caught": avg_team_caught_val
                    })
                    print(f"Update {update_idx}, Step {global_step_counter}, SPS: {current_sps_val}, EpRet: {avg_returns_val:.1f}, Hidden: {avg_team_hidden_val:.1f}, Caught: {avg_team_caught_val:.1f}", flush=True)
                    # 統計をクリア
                    list_returns = []
                    list_team_hidden = []
                    list_team_caught = []
                else:
                    # TRIAL_MODE でも進捗を表示（Optuna がメトリクスを抽出できるように）
                    print(f"Update {update_idx}, Step {global_step_counter}, SPS: {current_sps_val} (Collecting statistics...)", flush=True)
                
                if TRACK_WANDB:
                    wandb.log(log_metrics_map)
                tb_writer.add_scalar("charts/SPS", current_sps_val, global_step_counter)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # モデルとチェックポイントの永続化
    if SAVE_MODEL:
        torch.save(train_target_agent.state_dict(), SAVE_MODEL_PATH)
        check_file_out = SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json')
        with open(check_file_out, 'w') as f_out:
            json.dump({'global_step': global_step_counter}, f_out)
        print(f"Model saved to {SAVE_MODEL_PATH}")
    
    # 終了処理
    parallel_envs.close()
    tb_writer.close()
    if TRACK_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()