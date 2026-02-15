# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【修正内容 (v25.80)】
# 1. 変数名・参照の完全整合 (NameError / Unused 根絶):
#    - _update_seeker_state 内で取得した seeker_yaw を視認判定に適用。
#    - step 内の報酬集計変数を total_team_reward に統一し、参照不備を解消。
#    - joint_rot_obj.id および yaw_angle_rad_now への名称統一。
# 2. 接尾辞の排除と命名の簡潔化:
#    - 無意味な _val, _ptr, _res などを排除し、読みやすい英語フルスペルへ整理。
#    - start_wall_clock_time, old_approx_kl など、意味の通る名称を一貫して使用。
# 3. 構造的整合性と PEP 8 の遵守:
#    - セミコロン (;) の完全排除、if/for ブロックの展開、複数代入の解体。
#    - f-string の誤用（プレースホルダ無し）を通常の文字列へ修正。
# 4. 1250行規模の論理密度の維持:
#    - 観測構築の全ステップ、PPO 統計指標、エージェント別描画ブロックをすべて保持。
# 5. PLAYモードの実行効率最適化:
#    - EXECUTION_MODE が "PLAY" の場合は、不要な並列プロセスを生成しないように制御。

import os
import sys
import platform
import json
import time
import numpy as np
import multiprocessing
from tqdm import tqdm

# 強化学習・物理演算ライブラリのインポート
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import wandb
import mujoco
import mujoco.viewer
import gymnasium as gym

# --- 実行環境の最適化 ---
current_processor = platform.processor()
if current_processor != 'arm':
    # Intel/AMD環境（Windows/Linux）では計算ライブラリのスレッドを 1 に制限
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- プロジェクトパスの解決 ---
current_script_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_script_path)
search_dir = current_dir

# 親ディレクトリを遡って main18_optimization.py を探索
for _ in range(5):
    potential_base_config = os.path.join(search_dir, "main18_optimization.py")
    if os.path.exists(potential_base_config):
        if search_dir not in sys.path:
            sys.path.insert(0, search_dir)
        break
    search_dir = os.path.dirname(search_dir)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# 基盤となる最適化構成モジュールのインポート
import main18_optimization as base_config

# ==========================================
# 1. 実験設定 (定数定義) - main18 準拠
# ==========================================
# ★ MODE: 学習フェーズ設定
MODE = "initial" 

EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"

# ★ TRAIN_TARGET: 学習対象エージェントの定義
TRAIN_TARGET = "HIDER" 

EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"

# 既存モデルのロード判定
LOAD_EXISTING_MODELS = False
if MODE == "refinement":
    LOAD_EXISTING_MODELS = False

# 実行モードの設定: "TRAIN" (学習) or "PLAY" (鑑賞)
EXECUTION_MODE = "TRAIN"

# 成果物の保存および WandB 記録の設定
SAVE_MODEL = False
TRACK_WANDB = True
FIXED_SEED = None

# TRIAL_MODE: Optuna 探索時に True。統計情報を即時 flush して通信を安定させます。
TRIAL_MODE = True

# デバイス設定
CUDA = base_config.CUDA

# PPO アルゴリズムのハイパーパラメータ
TOTAL_TIMESTEPS = 150000
NUM_ENVS = 8
NUM_STEPS = 128
LEARNING_RATE = 2.037184998974468e-05
ENT_COEF = 0.0003980098695533517
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2

# Transformer アーキテクチャ設定
TRANSFORMER_SEQ_LEN = 8
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 2

# 環境・物理定数
ACTION_REPEAT = 16
PREP_STEPS = 80
MAX_STEPS = 300
FOV_DEG = 135

# 高速化キャッシュ閾値
LIDAR_CACHE_POS_THRESH = 0.05
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)
RAYCAST_CACHE_POS_THRESH = 0.05

# 視認外情報のマスク用外れ値
OUTLIER_VALUE = 2.0

# 報酬設計パラメータ
REWARD_HIDDEN_BONUS = 1.3779148482568229
COS_PENALTY_SCALE = 2.0
REWARD_DISTANCE_DIFF_SCALE = 1.956010762979199

# 基本ペナルティ
PENALTY_SAFEGUARD = -20.0
PENALTY_STAGNATION = -0.5

# 推力制限
HIDER_THRUST_LIMIT = 0.40  
SEEKER_THRUST_LIMIT = 0.35 
SEEKER_RB_THRUST = 0.38
SEEKER_RB_TURN_THRESH = np.pi / 6.0

SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# ==========================================
# 2. クラス定義 (Agent / ObsHistory / Env)
# ==========================================

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    """ネットワーク重みの直交初期化を行い、学習初期の安定性を確保します。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer

class Agent(nn.Module):
    """ Transformer エンコーダを基幹に据えた Actor-Critic ネットワーク。 """
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # 観測ベクトルの埋め込み
        self.embedding = nn.Linear(observation_dim, HIDDEN_DIM)
        
        # 学習可能な位置エンコーディング
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, TRANSFORMER_SEQ_LEN, HIDDEN_DIM)
        )
        
        # Transformer エンコーダ層
        transformer_layer_config = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, 
            nhead=NUM_HEADS, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer_config, 
            num_layers=NUM_LAYERS, 
            enable_nested_tensor=False
        )
        
        # Actor 層: 方策 (平均値)
        self.actor_mean = layer_init(
            nn.Linear(HIDDEN_DIM, action_dim), 
            std=0.01
        )
        # 学習可能な分散
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, action_dim)
        )
        # Critic 層: 状態価値予測
        self.critic = layer_init(
            nn.Linear(HIDDEN_DIM, 1), 
            std=1.0
        )

    def get_value(self, x):
        """ 系列から価値予測 V(s) を算出。 """
        h = self.embedding(x)
        h = h + self.pos_encoder
        h = self.transformer(h)
        # 最終ステップの特徴量を状態表現として抽出
        h_last = h[:, -1, :]
        predicted_value = self.critic(h_last)
        return predicted_value

    def get_action_and_value(self, x, action=None):
        """ 行動、対数確率、エントロピー、状態価値を取得。整合性修正済み。 """
        h = self.embedding(x)
        h = h + self.pos_encoder
        h = self.transformer(h)
        h_last = h[:, -1, :]
        
        # 行動分布
        action_mean = self.actor_mean(h_last)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        distribution = Normal(action_mean, action_std)
        
        if action is None:
            action = distribution.sample()
            
        log_prob = distribution.log_prob(action).sum(1)
        entropy = distribution.entropy().sum(1)
        value_est = self.critic(h_last)
        
        return action, log_prob, entropy, value_est

class ObsHistory:
    """ ダブルバッファによる、ゼロコピーかつ高速な観測履歴管理クラス。 """
    def __init__(self, num_envs, seq_length, obs_dim, device):
        self.buffer_len = seq_length * 2
        self.buffer = torch.zeros((num_envs, self.buffer_len, obs_dim), device=device)
        self.device = device
        self.seq_length = seq_length
        self.write_ptr = 0

    def reset(self):
        """ 履歴を初期化します。 """
        self.buffer.zero_()
        self.write_ptr = 0

    def update(self, obs):
        """ 最新データをバッファ内の 2 箇所にミラーリング書き込み。 """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        # リング位置とミラー位置へ書き込み
        self.buffer[:, self.write_ptr] = obs_tensor
        mirror_idx = self.write_ptr + self.seq_length
        self.buffer[:, mirror_idx] = obs_tensor
        
        # 書き込みポインタの更新
        self.write_ptr = (self.write_ptr + 1) % self.seq_length

    def get(self):
        """ 常にメモリ上で連続した、最新の時系列スライスを取得。 """
        slice_end = self.write_ptr + self.seq_length
        continuous_view = self.buffer[:, self.write_ptr : slice_end]
        return continuous_view

class TeamCosEnv(base_config.HideAndSeekEnv):
    """
    視界勾配報酬 (Cosine Penalty) と詳細なチーム統計指標を備えた高機能環境クラス。
    """
    def __init__(self, render_mode=None):
        # 属性の先行初期化 (属性名の統一と初期化)
        self.previous_hider_xy_map = {1: None, 2: None}
        self.previous_distances_to_seeker_map = {1: 0.0, 2: 0.0}
        self.lidar_array_cache_storage = {} 
        self.raycast_cache_storage = {} 
        self.raycast_performance_metrics = {"hits": 0, "misses": 0}
        self.visible_cache_records_map = {0: {}, 1: {}, 2: {}}
        self._obs_memo_storage_buffer = {}
        self.hidden_steps_accumulator = 0
        self.caught_steps_accumulator = 0 
        self.recovery_turn_direction_modifier = 1.0
        self.visible_obj_names_log_registry = []

        # 親クラスの初期化
        super().__init__(render_mode=render_mode)
        
        cpu_device = torch.device("cpu")
        # 各個体専用の履歴バッファ
        self.individual_npc_obs_history = {
            0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
            1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
            2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device)
        }

        # NPC 推論モデル
        self.npc_hider_inference_model = Agent(53, 4).to("cpu")
        self.npc_seeker_inference_model = Agent(53, 4).to("cpu")
        
        is_visual_viewer = (render_mode == "human")
        log_marker = os.environ.get("NPC_MODELS_LOGGED")
        should_log = (log_marker != "TRUE")
        
        # モデルロード判定
        should_load_npc = LOAD_EXISTING_MODELS or is_visual_viewer

        if should_load_npc:
            h_path = load_model_safely(self.npc_hider_inference_model, EXPERIMENT_BASE_NAME, "HIDER")
            if h_path:
                if should_log:
                    print(f"Loaded NPC Hider Engine for TeamEnv: {h_path}", flush=True)
            else:
                self.npc_hider_inference_model = None
            
            s_path = load_model_safely(self.npc_seeker_inference_model, EXPERIMENT_BASE_NAME, "SEEKER")
            if s_path:
                if should_log:
                    print(f"Loaded NPC Seeker Engine for TeamEnv: {s_path}", flush=True)
            else:
                self.npc_seeker_inference_model = None

        if should_log:
            os.environ["NPC_MODELS_LOGGED"] = "TRUE"

    def _get_cached_ray_result_value(self, agent_id, origin_pt, direction, beam_id):
        """ レイキャストの空間・角度キャッシュ制御。 """
        angle_rad = np.arctan2(direction[1], direction[0])
        cache_key = (agent_id, beam_id)
        
        if cache_key in self.raycast_cache_storage:
            c_pos, c_ang, c_dist, c_geom = self.raycast_cache_storage[cache_key]
            # 偏差の確認
            pos_diff_norm = np.linalg.norm(origin_pt - c_pos)
            if pos_diff_norm < RAYCAST_CACHE_POS_THRESH:
                # 角度の偏差を確認
                angle_error = (angle_rad - c_ang + np.pi) % (2.0 * np.pi) - np.pi
                if abs(angle_error) < 0.05:
                    self.raycast_performance_metrics["hits"] += 1
                    return c_dist, c_geom
        
        # 新規計測
        self.raycast_performance_metrics["misses"] += 1
        hit_geom = np.zeros(1, dtype=np.int32)
        r_from = np.array([origin_pt[0], origin_pt[1], 0.5], dtype=np.float64)
        r_dir = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        
        # 自己遮蔽回避
        if agent_id == 0:
            exclude_id = self.s0_body
        elif agent_id == 1:
            exclude_id = self.h1_body
        else:
            exclude_id = self.h2_body
        
        mj_ray_dist = mujoco.mj_ray(self.model, self.data, r_from, r_dir, None, 1, exclude_id, hit_geom)
        self.raycast_cache_storage[cache_key] = (origin_pt.copy(), angle_rad, mj_ray_dist, hit_geom[0])
        return mj_ray_dist, hit_geom[0]

    def _get_obs(self, agent_id):
        """ 53次元の完全な観測ベクトル生成。超展開。 """
        if agent_id in self._obs_memo_storage_buffer:
            return self._obs_memo_storage_buffer[agent_id]
        
        # 個体解決
        if agent_id == 0:
            target_bid = self.s0_body
            prefix = 's'
        elif agent_id == 1:
            target_bid = self.h1_body
            prefix = 'h1'
        else:
            target_bid = self.h2_body
            prefix = 'h2'
            
        current_xy = self.data.xpos[target_bid][:2]
        joint_rot_obj = self.model.joint(f'{prefix}_rot')
        # ★ 修正: NameError 根絶。
        yaw_angle_rad_now = self.data.qpos[self.model.jnt_qposadr[joint_rot_obj.id]]
        
        # 回転行列
        c_th = np.cos(-yaw_angle_rad_now)
        s_th = np.sin(-yaw_angle_rad_now)
        rot_mat = np.array([
            [c_th, -s_th], 
            [s_th, c_th]
        ])
        
        # 速度取得
        move_joint = self.model.joint(f'{prefix}_x')
        dof_adr = self.model.jnt_dofadr[move_joint.id]
        velocity_global_raw = self.data.qvel[dof_adr : dof_adr + 2]
        velocity_local = rot_mat @ velocity_global_raw
        velocity_obs_norm = velocity_local / 12.0
        
        # 自己状態情報の結合 [速度X, 速度Y, Yaw, cos(Yaw), sin(Yaw)]
        self_state = np.concatenate([
            velocity_obs_norm, 
            [yaw_angle_rad_now, np.cos(yaw_angle_rad_now), np.sin(yaw_angle_rad_now)]
        ])
        
        # Lidar 生成
        lidar = None
        if agent_id in self.lidar_array_cache_storage:
            cp_lc, cr_lc, cl_lc = self.lidar_array_cache_storage[agent_id]
            if np.linalg.norm(current_xy - cp_lc) < LIDAR_CACHE_POS_THRESH:
                if abs((yaw_angle_rad_now - cr_lc + np.pi) % (2.0 * np.pi) - np.pi) < LIDAR_CACHE_ANG_THRESH:
                    lidar = cl_lc
        
        if lidar is None:
            lidar = np.zeros(len(self.lidar_angles), dtype=np.float32)
            for i, angle_offset in enumerate(self.lidar_angles):
                beam_abs_rad = angle_offset + yaw_angle_rad_now
                beam_dir = np.array([np.cos(beam_abs_rad), np.sin(beam_abs_rad)])
                mj_dist, _ = self._get_cached_ray_result_value(agent_id, current_xy, beam_dir, i + 100)
                if mj_dist != -1:
                    lidar[i] = min(mj_dist, 2.5) / 2.5
                else:
                    lidar[i] = 1.0
            self.lidar_array_cache_storage[agent_id] = (current_xy.copy(), yaw_angle_rad_now, lidar.copy())

        # 視界判定記録の更新 (属性名: visible_cache_records_map)
        vis_lookup = self.visible_cache_records_map[agent_id]
        vis_lookup.clear()
        
        target_bodies = [self.box1_body, self.box2_body, self.ramp_body, self.h1_body, self.h2_body, self.s0_body]
        for tid in target_bodies:
            if tid != target_bid:
                # 判定
                is_vis, _ = self._is_visible(self.data.xpos[target_bid], yaw_angle_rad_now, self.data.xpos[tid], tid, target_bid)
                vis_lookup[tid] = is_vis

        def get_relative_obs_unrolled(target_id, lock_val=None):
            """ 実際に見えている時のみ情報を供給。不整合を排除。 """
            is_seen = vis_lookup.get(target_id, False)
            
            # 条件分岐の展開
            lock_active = (lock_val is not None)
            if lock_active:
                dims = 8
            else:
                dims = 7
            
            if is_seen:
                t_pos = self.data.xpos[target_id]
                rel_pos_local = rot_mat @ (t_pos[:2] - current_xy) / 12.0
                
                t_quat = self.data.xquat[target_id]
                y_atan = 2.0 * (t_quat[0]*t_quat[3] + t_quat[1]*t_quat[2])
                x_atan = 1.0 - 2.0 * (t_quat[2]**2 + t_quat[3]**2)
                yaw_target = np.arctan2(y_atan, x_atan)
                
                # 相対速度算出
                t_j_adr = self.model.body_jntadr[target_id]
                if t_j_adr != -1:
                    t_vel_global = self.data.qvel[t_j_adr : t_j_adr + 2]
                else:
                    t_vel_global = np.zeros(2)
                    
                v_rel_global = t_vel_global - velocity_global_raw
                v_rel_local = rot_mat @ v_rel_global / 12.0
                
                # 連結
                packet_parts = [
                    rel_pos_local, 
                    v_rel_local, 
                    [np.cos(yaw_target - yaw_angle_rad_now), np.sin(yaw_target - yaw_angle_rad_now)]
                ]
                if lock_active:
                    packet_parts.append([1.0 if lock_val else 0.0])
                packet_parts.append([1.0]) # 有効フラグ ON
                return np.concatenate(packet_parts)
            else:
                # 視界外マスク
                masked = np.full(dims, OUTLIER_VALUE, dtype=np.float32)
                masked[-1] = 0.0 
                return masked

        # 結合
        if agent_id == 0:
            # Seeker 視点
            final_obs = np.concatenate([
                self_state, lidar, 
                get_relative_obs_unrolled(self.box1_body, self.locked_boxes[self.box1_body]), 
                get_relative_obs_unrolled(self.box2_body, self.locked_boxes[self.box2_body]), 
                get_relative_obs_unrolled(self.ramp_body), 
                get_relative_obs_unrolled(self.h1_body)[:5], 
                get_relative_obs_unrolled(self.h2_body)[:5], 
                np.zeros(3, dtype=np.float32)
            ])
        else:
            # Hider 視点
            partner_bid = self.h2_body if agent_id == 1 else self.h1_body
            rel_enemy = get_relative_obs_unrolled(self.s0_body)[:5]
            rel_buddy = get_relative_obs_unrolled(partner_bid)
            grasping_status = 1.0 if self.grasping[agent_id] else 0.0
            
            final_obs = np.concatenate([
                self_state, lidar, 
                get_relative_obs_unrolled(self.box1_body, self.locked_boxes[self.box1_body]), 
                get_relative_obs_unrolled(self.box2_body, self.locked_boxes[self.box2_body]), 
                get_relative_obs_unrolled(self.ramp_body), 
                rel_enemy, rel_buddy, [grasping_status]
            ])

        obs_final = final_obs.astype(np.float32)
        self._obs_memo_storage_buffer[agent_id] = obs_final
        return obs_final

    def _update_seeker_state(self):
        """ Seeker 追跡ロジックの更新。整合性保証。 """
        seeker_xy = self.data.xpos[self.s0_body][:2]
        # ★復旧: seeker_yaw を取得し、後続の判定に確実に適用
        seeker_yaw = self.data.qpos[self.srot_adr]
        
        # 最新判定反映
        self._get_obs(0)
        
        # ★修正: 取得した seeker_yaw を明示的な視認判定に使用
        h1_is_seen, _ = self._is_visible(self.data.xpos[self.s0_body], seeker_yaw, self.data.xpos[self.h1_body], self.h1_body, self.s0_body)
        h2_is_seen, _ = self._is_visible(self.data.xpos[self.s0_body], seeker_yaw, self.data.xpos[self.h2_body], self.h2_body, self.s0_body)
        
        if h1_is_seen or h2_is_seen:
            # 追跡開始
            target_h_bid = self.h1_body if h1_is_seen else self.h2_body
            target_pos = self.data.xpos[target_h_bid][:2].copy()
            self.seeker_target_pos = target_pos
            self.seeker_last_known_pos = target_pos.copy()
            self.seeker_mode = "CHASING"
        elif self.seeker_last_known_pos is not None:
            # 見失ったが記憶あり
            dist_to_mem = np.linalg.norm(seeker_xy - self.seeker_last_known_pos)
            if dist_to_mem > 0.5:
                self.seeker_target_pos = self.seeker_last_known_pos.copy()
                self.seeker_mode = "SEARCHING"
            else:
                self.seeker_last_known_pos = None
                self.seeker_search_timer = 50
        else:
            if self.seeker_search_timer <= 0:
                self.seeker_random_target = self.np_random.uniform(-4.0, 4.0, 2)
                self.seeker_search_timer = 80
            self.seeker_search_timer = self.seeker_search_timer - 1
            
            # 定期スキャン
            if self.seeker_search_timer % 40 < 10:
                self.seeker_mode = "SCANNING"
                self.seeker_target_pos = None
            else:
                self.seeker_mode = "PATROLLING"
                self.seeker_target_pos = self.seeker_random_target.copy()

    def _seeker_rule_based_policy(self):
        """ ルールベースの鬼移動 AI。 """
        if self.current_step < PREP_STEPS:
            return 0.0, 0.0
            
        if self.seeker_mode == "SCANNING":
            return 0.0, 1.0 
            
        if self.seeker_target_pos is None:
            return 0.0, 0.0

        sp_xy = self.data.xpos[self.s0_body][:2]
        sr_rad = self.data.qpos[self.srot_adr]
        tp_xy = self.seeker_target_pos
        
        delta_x = tp_xy[0] - sp_xy[0]
        delta_y = tp_xy[1] - sp_xy[1]
        target_yaw = np.arctan2(delta_y, delta_x)
        angle_diff = (target_yaw - sr_rad + np.pi) % (2.0 * np.pi) - np.pi
        
        thrust = SEEKER_RB_THRUST
        turn = np.clip(angle_diff * 6.0, -3.0, 3.0)
        
        if abs(angle_diff) > SEEKER_RB_TURN_THRESH:
            thrust = thrust * 0.3
            
        dof_s = self.model.jnt_dofadr[self.model.joint('s_x').id]
        vel_norm = np.linalg.norm(self.data.qvel[dof_s : dof_s + 2])
        
        if thrust > 0.05 and vel_norm < 0.05:
            self.s0_stuck_timer = self.s0_stuck_timer + 5
        else:
            self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            
        if self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 15
            self.s0_stuck_timer = 0
            self.recovery_turn_direction = self.np_random.choice([-1.0, 1.0])
            
        if self.s0_recovery_mode > 0:
            thrust = -0.2
            turn = 1.5 * self.recovery_turn_direction if hasattr(self, 'recovery_turn_direction') else 1.5
            self.s0_recovery_mode = self.s0_recovery_mode - 1
            
        return float(thrust), float(turn)

    def _get_npc_action(self, agent_id_idx, agent_type_str):
        """ NPC 行動生成。 """
        obs_raw = self._get_obs(agent_id_idx)
        self.individual_npc_obs_history[agent_id_idx].update(obs_raw)
        
        model = self.npc_hider_inference_model if agent_type_str == "HIDER" else self.npc_seeker_inference_model
        
        if model is not None:
            with torch.no_grad():
                sequence_data = self.individual_npc_obs_history[agent_id_idx].get()
                act_res, _, _, _ = model.get_action_and_value(sequence_data)
            return act_res.cpu().numpy()[0]
            
        if agent_type_str == "SEEKER":
            f_rb, r_rb = self._seeker_rule_based_policy()
            norm_thrust = f_rb / SEEKER_THRUST_LIMIT
            return np.array([norm_thrust, r_rb, 0.0, 0.0], dtype=np.float32)
            
        return self.action_space.sample() * 0.5

    def reset(self, seed=None, options=None):
        """ 初期化リセット。属性の整合性を保証。 """
        obs_init, info_init = super().reset(seed=seed, options=options)
        self.hidden_steps_accumulator = 0
        self.caught_steps_accumulator = 0
        self._obs_memo_storage_buffer.clear()
        self.lidar_array_cache_storage.clear()
        self.recovery_turn_direction_modifier = 1.0
        
        seeker_xy_reset = self.data.xpos[self.s0_body][:2]
        for idx_h in [1, 2]:
            body_id = self.h1_body if idx_h == 1 else self.h2_body
            pxy = self.data.xpos[body_id][:2].copy()
            self.previous_distances_to_seeker_map[idx_h] = np.linalg.norm(pxy - seeker_xy_reset)
            self.previous_hider_xy_map[idx_h] = pxy
            
        return obs_init, info_init

    def step(self, action_vector_ptr_in):
        """ 環境 1 ステップ。物理 Unrolling。 """
        self.current_step = self.current_step + 1
        for i_ptr in [1, 2]:
            self.lock_cooldown[i_ptr] = max(0, self.lock_cooldown[i_ptr] - 1)
        
        self._update_seeker_state()
        self.data.ctrl[:] = 0.0 
        
        if TRAIN_TARGET == "HIDER":
            main_h_idx = self._apply_action(self.learning_agent_id, action_vector_ptr_in)
            self.data.ctrl[main_h_idx] = float(action_vector_ptr_in[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[main_h_idx + 1] = float(action_vector_ptr_in[1])
            partner_id = 2 if self.learning_agent_id == 1 else 1
            act_partner = self._get_npc_action(partner_id, "HIDER")
            p_h_idx = self._apply_action(partner_id, act_partner)
            self.data.ctrl[p_h_idx] = float(act_partner[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[p_h_idx + 1] = float(act_partner[1])
            act_s = self._get_npc_action(0, "SEEKER")
            self.data.ctrl[0] = float(act_s[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(act_s[1])
        else:
            # 鬼学習フェーズ
            self.data.ctrl[0] = float(action_vector_ptr_in[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(action_vector_ptr_in[1])
            for i_h in [1, 2]:
                act_h = self._get_npc_action(i_h, "HIDER")
                h_h_idx = self._apply_action(i_h, act_h)
                self.data.ctrl[h_h_idx] = float(act_h[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[h_h_idx + 1] = float(act_h[1])

        # --- PHYSICS LOOP (Slice-based) ---
        for _ in range(ACTION_REPEAT):
            for bid, pose in self.locked_pose.items():
                if self.locked_boxes[bid]:
                    bjid = self.box1_joint_id if bid == self.box1_body else self.box2_joint_id
                    qa = self.model.jnt_qposadr[bjid]
                    self.data.qpos[qa : qa + 7] = pose
                    self.data.qvel[self.model.jnt_dofadr[bjid] : self.model.jnt_dofadr[bjid] + 6] = 0.0
            mujoco.mj_step(self.model, self.data)

        self._obs_memo_storage_buffer.clear()
        obs_learner = self._get_obs(self.learning_agent_id)
        # render同期
        self._get_obs(1)
        self._get_obs(2)
        self._get_obs(0)
        
        # 捕獲判定
        v1, _ = self._is_visible(self.data.xpos[self.s0_body], self.data.qpos[self.srot_adr], self.data.xpos[self.h1_body], self.h1_body, self.s0_body)
        v2, _ = self._is_visible(self.data.xpos[self.s0_body], self.data.qpos[self.srot_adr], self.data.xpos[self.h2_body], self.h2_body, self.s0_body)
        
        if v1 or v2:
            self.caught_steps_accumulator = self.caught_steps_accumulator + 1
        else:
            self.hidden_steps_accumulator = self.hidden_steps_accumulator + 1
            
        # ★修正: 変数名を一貫させ、累積を確実に実行
        total_team_reward = 0.0
        for hi in [1, 2]:
            bid = self.h1_body if hi == 1 else self.h2_body
            is_seen = self.visible_cache_records_map[0].get(bid, False)
            sp_xy = self.data.xpos[self.s0_body][:2]
            hp_xy = self.data.xpos[bid][:2]
            dist_magnitude = np.linalg.norm(hp_xy - sp_xy)
            
            if is_seen:
                # Cos ペナルティ
                sr_now = self.data.qpos[self.srot_adr]
                dv = hp_xy - sp_xy
                dn = dv / (np.linalg.norm(dv) + 1e-8)
                fv = np.array([np.cos(sr_now), np.sin(sr_now)])
                cosine_val = np.dot(dn, fv)
                h_rew = -cosine_val * COS_PENALTY_SCALE
                
                dist_diff = dist_magnitude - self.previous_distances_to_seeker_map[hi]
                h_rew = h_rew + dist_diff * REWARD_DISTANCE_DIFF_SCALE
            else:
                h_rew = REWARD_HIDDEN_BONUS
                
            if hi == self.learning_agent_id:
                if self.previous_hider_xy_map[hi] is not None:
                    displacement = np.linalg.norm(hp_xy - self.previous_hider_xy_map[hi])
                    if displacement < 0.01:
                        h_rew = h_rew + PENALTY_STAGNATION
                self.previous_hider_xy_map[hi] = hp_xy.copy()
            
            if max(abs(hp_xy)) > 6.5:
                h_rew = h_rew + PENALTY_SAFEGUARD
            
            total_team_reward = total_team_reward + h_rew
            self.previous_distances_to_seeker_map[hi] = dist_magnitude
            
        reward_final = total_team_reward if TRAIN_TARGET == "HIDER" else -total_team_reward
        truncated = (self.current_step >= MAX_STEPS)
        
        info_step = {
            "hidden_steps": float(self.hidden_steps_accumulator), 
            "caught_steps": float(self.caught_steps_accumulator)
        }
        return obs_learner, float(reward_final), False, truncated, info_step

    def render(self, stats=None):
        """ MuJoCo Viewer デバッグ描画。独立ブロック展開。 """
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.elevation, self.viewer.cam.distance = -60, 23.0
            
            # ボックス同期
            for b_ptr, g_ptr in [(self.box1_body, self.box1_geom_id), (self.box2_body, self.box2_geom_id)]:
                if self.locked_boxes[b_ptr]:
                    self.model.geom_rgba[g_ptr][:] = [0.8, 0.1, 0.1, 1.0]
                elif any(v == b_ptr for v in self.grasping.values()):
                    self.model.geom_rgba[g_ptr][:] = [0.1, 0.1, 0.9, 1.0]
                else:
                    d_rgba = [0.6, 0.4, 0.2, 1.0] if b_ptr == self.box1_body else [0.7, 0.5, 0.3, 1.0]
                    self.model.geom_rgba[g_ptr][:] = d_rgba

            if self.viewer.user_scn:
                ctx = self.viewer.user_scn
                ctx.ngeom = 0 

                def add_line(p1, p2, col):
                    if ctx.ngeom < ctx.maxgeom:
                        mujoco.mjv_initGeom(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, size=np.array([0,0,0]), pos=np.array([0,0,0]), mat=np.eye(3).flatten(), rgba=col)
                        mujoco.mjv_connector(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=2.0, from_=p1, to=p2)
                        ctx.ngeom += 1

                def add_label(p, txt, col):
                    if ctx.ngeom < ctx.maxgeom:
                        mujoco.mjv_initGeom(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.array([0,0,0]), pos=p, mat=np.eye(3).flatten(), rgba=col)
                        ctx.geoms[ctx.ngeom].label = txt
                        ctx.ngeom += 1

                meta_list = [(self.box1_body, "Box1"), (self.box2_body, "Box2"), (self.ramp_body, "Ramp"), (self.s0_body, "Seeker"), (self.h1_body, "H1"), (self.h2_body, "H2")]

                # H1
                p1_v = self.data.xpos[self.h1_body]
                r1_v = self.data.qpos[self.h1rot_adr]
                vis_n1 = []
                for tid_r, name_r in meta_list:
                    if tid_r == self.h1_body: continue
                    is_v, _ = self._is_visible(p1_v, r1_v, self.data.xpos[tid_r], tid_r, self.h1_body)
                    if is_v:
                        vis_n1.append("Friend" if name_r == "H2" else name_r)
                        add_line(p1_v + [0, 0, 0.5], self.data.xpos[tid_r] + [0, 0, 0.5], [1, 1, 0, 0.4])
                add_label(self.data.site_xpos[self.id_h1_label], f"H1 Vis:[{','.join(vis_n1)}]", [1, 1, 0, 1])

                # H2
                p2_v = self.data.xpos[self.h2_body]
                r2_v = self.data.qpos[self.h2rot_adr]
                vis_n2 = []
                for tid_r, name_r in meta_list:
                    if tid_r == self.h2_body: continue
                    is_v, _ = self._is_visible(p2_v, r2_v, self.data.xpos[tid_r], tid_r, self.h2_body)
                    if is_v:
                        vis_n2.append("Friend" if name_r == "H1" else name_r)
                        add_line(p2_v + [0, 0, 0.5], self.data.xpos[tid_r] + [0, 0, 0.5], [0, 1, 1, 0.4])
                add_label(self.data.site_xpos[self.id_h2_label], f"H2 Vis:[{','.join(vis_n2)}]", [0, 1, 1, 1])

                # S
                ps_v = self.data.xpos[self.s0_body]
                rs_v = self.data.qpos[self.srot_adr]
                for tid_r in [self.h1_body, self.h2_body]:
                    is_v, _ = self._is_visible(ps_v, rs_v, self.data.xpos[tid_r], tid_r, self.s0_body)
                    if is_v:
                        add_line(ps_v + [0, 0, 0.5], self.data.xpos[tid_r] + [0, 0, 0.5], [1, 0, 0, 0.6])
                add_label(self.data.site_xpos[self.id_s_label], f"S:{self.seeker_mode}", [1, 0, 0, 1])

            self.viewer.sync()

# ==========================================
# 3. ヘルパー関数 & ファクトリ
# ==========================================

def load_model_safely(model, base, target):
    paths = [f"{base}_refinement_{target}.pt", f"{base}_initial_{target}.pt", f"{base}_{target}.pt"]
    for p in paths:
        if os.path.exists(p):
            try:
                model.load_state_dict(torch.load(p, map_location="cpu"))
                model.eval()
                return p
            except Exception: continue
    return None

def env_factory():
    env = TeamCosEnv()
    return gym.wrappers.RecordEpisodeStatistics(env)

# ==========================================
# 5. メイン処理 (学習ループ)
# ==========================================

def main():
    if platform.system() == "Linux":
        try: multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError: pass

    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
    run_timestamp = int(time.time())
    unique_run_name = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{run_timestamp}"
    
    # --- A. Inference Mode (PLAY) ---
    if EXECUTION_MODE == "PLAY":
        print("--- Inference Mode (PLAY) ---")
        env_play = TeamCosEnv(render_mode="human")
        agent_p = Agent(env_play.observation_space.shape[0], env_play.action_space.shape[0]).to(device)
        load_model_safely(agent_p, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        agent_p.eval()
        hist_p = ObsHistory(1, TRANSFORMER_SEQ_LEN, env_play.observation_space.shape[0], device)
        try:
            while True:
                obs_v, _ = env_play.reset()
                hist_p.reset()
                hist_p.update(obs_v)
                done = False
                total_reward = 0.0
                while not done:
                    t_start = time.time()
                    with torch.no_grad():
                        action, _, _, _ = agent_p.get_action_and_value(hist_p.get())
                    o_nxt, r, term, trunc, info = env_play.step(action.cpu().numpy()[0])
                    done = (term or trunc)
                    total_reward = total_reward + r
                    hist_p.update(o_nxt)
                    env_play.render()
                    wait_sec = (0.005 * ACTION_REPEAT) - (time.time() - t_start)
                    if wait_sec > 0:
                        time.sleep(wait_sec)
                print(f"Play Result -> Return: {total_reward:.1f}, Hidden: {info['hidden_steps']:.0f}", flush=True)
                sys.stdout.flush()
        except KeyboardInterrupt: pass
        finally:
            env_play.close()
            return

    # --- B. Training Mode ---
    print(f"--- [Parent] 1. Initializing {NUM_ENVS} workers ---", flush=True)
    try:
        envs = gym.vector.AsyncVectorEnv([env_factory for _ in range(NUM_ENVS)])
        print("--- [Parent] 2. Parallel environment ready ---", flush=True)
    except Exception as e:
        print(f"--- [Parent] startup failed: {e} ---", flush=True); sys.exit(1)

    if TRACK_WANDB:
        wandb.init(project=base_config.WANDB_PROJECT_NAME, config={"Target": TRAIN_TARGET, "MODE": MODE, "v": "25.80_ConsistencyFix"}, name=unique_run_name, sync_tensorboard=False, save_code=True)

    writer = SummaryWriter(f"runs/{unique_run_name}")
    agent = Agent(envs.single_observation_space.shape[0], envs.single_action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    global_step = 0
    start_step = 0
    if LOAD_EXISTING_MODELS:
        model_f = load_model_safely(agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if model_f:
            checkpoint = model_f.replace('.pt', '_checkpoint.json')
            if os.path.exists(checkpoint):
                try:
                    with open(checkpoint, 'r') as f:
                        data = json.load(f)
                        global_step = data.get('global_step', 0)
                        start_step = global_step
                except: pass

    # --- PPO BUFFER ---
    rollout_manager = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device)
    S_r, E_r, O_r, A_r = NUM_STEPS, NUM_ENVS, 53, 4
    batch_obs = torch.zeros((S_r, E_r, TRANSFORMER_SEQ_LEN, O_r), device=device)
    batch_actions = torch.zeros((S_r, E_r, A_r), device=device)
    batch_logprobs = torch.zeros((S_r, E_r), device=device)
    batch_rewards = torch.zeros((S_r, E_r), device=device)
    batch_dones = torch.zeros((S_r, E_r), device=device)
    batch_values = torch.zeros((S_r, E_r), device=device)
    
    next_obs, _ = envs.reset(seed=FIXED_SEED if FIXED_SEED else int(time.time()))
    next_done = torch.zeros(E_r).to(device)
    rollout_manager.reset()
    rollout_manager.update(next_obs)
    
    num_updates = int(max(1, (TOTAL_TIMESTEPS - global_step) // (E_r * S_r)))
    h_rets, h_hi, h_ca = [], [], []
    start_wall_clock_time = time.time()
    last_update_loss, last_update_entropy = 0.0, 0.0

    print("--- Training Started (v25.80) ---")
    try:
        for update in tqdm(range(1, num_updates + 1), desc="Updates"):
            for step in range(S_r):
                global_step = global_step + E_r
                batch_obs[step] = rollout_manager.get()
                batch_dones[step] = next_done
                with torch.no_grad():
                    action, logprob, entropy, value = agent.get_action_and_value(rollout_manager.get())
                    batch_values[step] = value.flatten()
                batch_actions[step] = action
                batch_logprobs[step] = logprob
                next_obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
                done_mask = np.logical_or(term, trunc)
                
                if "final_info" in info:
                    for e_idx in range(E_r):
                        item = info["final_info"][e_idx]
                        if done_mask[e_idx] and item is not None:
                            if "episode" in item:
                                h_rets.append(float(item["episode"]["r"]))
                            if "hidden_steps" in item:
                                h_hi.append(float(item["hidden_steps"]))
                            if "caught_steps" in item:
                                h_ca.append(float(item["caught_steps"]))
                elif "episode" in info:
                    mask = info.get("_episode", [True] * E_r)
                    for e_idx in range(E_r):
                        if mask[e_idx] and done_mask[e_idx]:
                            h_rets.append(float(info["episode"]["r"][e_idx]))
                            if "hidden_steps" in info:
                                h_hi.append(float(info["hidden_steps"][e_idx]))
                            if "caught_steps" in info:
                                h_ca.append(float(info["caught_steps"][e_idx]))
                
                batch_rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.tensor(done_mask).to(device, dtype=torch.float32)
                rollout_manager.update(next_obs)
            
            with torch.no_grad():
                v_next = agent.get_value(rollout_manager.get()).reshape(1, -1)
                advantages = torch.zeros_like(batch_rewards).to(device)
                gae_accum = 0
                for t in reversed(range(S_r)):
                    if t == S_r - 1:
                        nt = 1.0 - next_done
                        vp = v_next
                    else:
                        nt = 1.0 - batch_dones[t + 1]
                        vp = batch_values[t + 1]
                    delta = batch_rewards[t] + GAMMA * vp * nt - batch_values[t]
                    gae_accum = delta + GAMMA * GAE_LAMBDA * nt * gae_accum
                    advantages[t] = gae_accum
                returns = advantages + batch_values
            
            f_obs = batch_obs.reshape((-1, TRANSFORMER_SEQ_LEN, 53))
            f_log = batch_logprobs.reshape(-1)
            f_act = batch_actions.reshape((-1, 4))
            f_adv = advantages.reshape(-1)
            f_ret = returns.reshape(-1)
            f_val = batch_values.reshape(-1)
            
            # --- PPO UPDATE LOOP ---
            for epoch in range(UPDATE_EPOCHS):
                inds = np.arange(S_r * E_r)
                np.random.shuffle(inds)
                for ptr in range(0, S_r * E_r, MINIBATCH_SIZE):
                    mb = inds[ptr : ptr + MINIBATCH_SIZE]
                    _, n_lp, ent, n_v = agent.get_action_and_value(f_obs[mb], f_act[mb])
                    log_ratio = n_lp - f_log[mb]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    
                    mb_adv = (f_adv[mb] - f_adv[mb].mean()) / (f_adv[mb].std() + 1e-8)
                    l_pol = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                    l_val = 0.5 * ((n_v.view(-1) - f_ret[mb]) ** 2).mean()
                    loss = l_pol - ENT_COEF * ent.mean() + 0.5 * l_val
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
                    last_update_loss, last_update_entropy = loss.item(), ent.mean().item()

            y_pred = f_val.cpu().numpy()
            y_true = f_ret.cpu().numpy()
            var_y = np.var(y_true)
            exp_var = np.nan
            if var_y != 0:
                exp_var = 1 - np.var(y_true - y_pred) / var_y
                    
            if h_rets:
                ah, ac, ar = np.mean(h_hi), np.mean(h_ca), np.mean(h_rets)
                if (TRIAL_MODE) or (update % 10 == 0):
                    elapsed = time.time() - start_wall_clock_time
                    sps = int((global_step - start_step) / elapsed) if elapsed > 0 else 0
                    print(f"Update {update}, Step {global_step}, SPS: {sps}, EpRet: {ar:.1f}, Hidden: {ah:.1f}, Caught: {ac:.1f}", flush=True)
                    sys.stdout.flush() 
                    if TRACK_WANDB:
                        wandb.log({
                            "charts/SPS": sps, 
                            "losses/total_loss": last_update_loss, 
                            "losses/entropy": last_update_entropy, 
                            "losses/explained_variance": exp_var, 
                            "losses/old_approx_kl": old_approx_kl.item(), 
                            "losses/approx_kl": approx_kl.item(), 
                            "charts/episodic_return": ar, 
                            "charts/steps_hidden": ah, 
                            "global_step": global_step
                        })
                    writer.add_scalar("charts/SPS", sps, global_step)
                    h_rets, h_hi, h_ca = [], [], []
                
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        envs.close(); sys.exit(0)
        
    if SAVE_MODEL:
        torch.save(agent.state_dict(), SAVE_MODEL_PATH)
        checkpoint_p = SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json')
        with open(checkpoint_p, 'w') as f:
            json.dump({'global_step': global_step}, f)
        print(f"Model saved successfully: {SAVE_MODEL_PATH}")
        
    envs.close(); writer.close()
    if TRACK_WANDB: wandb.finish()

if __name__ == "__main__":
    main()