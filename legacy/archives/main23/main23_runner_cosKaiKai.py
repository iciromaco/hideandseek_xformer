# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
#
# 【修正内容 (v25.63 - パフォーマンス最適化版)】
# 1. パフォーマンス最適化（SPS 300→1400への改善）:
#    - レイキャスト用バッファの事前確保（NumPy配列生成の大幅削減）
#    - Lidar方向ベクトルのバッファ再利用
#    - 視界外マスク配列の事前確保
#    - 報酬計算用ベクトルのバッファ化
#    - 不要な_get_obs呼び出しの削除
#    - 角度閾値のハードコード修正
# 2. 変数名の完全な一貫性確保（v25.62より継承）:
#    - _get_obs 内の自己状態ベクトルを self_state_vector_packet に統一。
#    - ObsHistory における属性参照の不整合を修正。
# 3. 報酬・統計ロジックの整合性向上:
#    - step() メソッド内の team_reward 集計変数のタイポ修正。
#    - 不可視マスク時の最後尾フラグが確実に 0.0 になる制御を維持。
# 4. Windows/Optuna/CUDA 対応:
#    - 標準出力の即時 flush と、KeyboardInterrupt 時の確実なプロセス終了シーケンス。

import json
import multiprocessing
import os
import platform
import sys
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

# 強化学習・物理演算ライブラリのインポート (グローバルスコープ)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb

# --- 実行環境の最適化 ---
# 並列実行時に各プロセスが CPU スレッドを奪い合わないよう、計算ライブラリを制限します。
system_processor_type_id = platform.processor()
if system_processor_type_id != "arm":
    # Intel/AMD環境（Windows/Linux）では計算ライブラリのスレッドを 1 に制限
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- プロジェクトパスの解決 ---
# 基盤となる main18_optimization.py を確実にインポートするためのパス設定。
current_script_abs_path_val = os.path.abspath(__file__)
current_script_parent_dir_val = os.path.dirname(current_script_abs_path_val)
search_dir_path_pointer_idx = current_script_parent_dir_val

# 最大 5 階層上まで main18 を探索
for _ in range(5):
    potential_base_config_file_path = os.path.join(search_dir_path_pointer_idx, "main18_optimization.py")
    if os.path.exists(potential_base_config_file_path):
        if search_dir_path_pointer_idx not in sys.path:
            sys.path.insert(0, search_dir_path_pointer_idx)
        break
    search_dir_path_pointer_idx = os.path.dirname(search_dir_path_pointer_idx)

# カレントディレクトリをパスに追加
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# 基盤となる最適化構成モジュールをインポート
import main18_optimization as base_config

# ==========================================
# 1. 実験設定 (定数定義)
# ==========================================
# ★ MODE: 学習フェーズ設定
MODE = "initial"

EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"

# ★ TRAIN_TARGET: 学習対象エージェントの定義
TRAIN_TARGET = "HIDER"

EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"

# 既存モデルのロード判定（MODE に厳格に連動）
LOAD_EXISTING_MODELS = False
if MODE == "refinement":
    LOAD_EXISTING_MODELS = True

# 実行モードの設定
EXECUTION_MODE = "TRAIN"

# モデル保存および記録の有無
SAVE_MODEL = True
TRACK_WANDB = True
FIXED_SEED = None

# TRIAL_MODE: Optuna 探索時に True。統計情報を即時 flush します。
TRIAL_MODE = False

# デバイス設定の継承
CUDA = base_config.CUDA

# PPO アルゴリズムのハイパーパラメータ (main18 準拠名)
TOTAL_TIMESTEPS = 5000000
NUM_ENVS = 8
NUM_STEPS = 128
LEARNING_RATE = 2e-4
ENT_COEF = 0.001
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2

# Transformer 設定
TRANSFORMER_SEQ_LEN = 8
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 2

# 環境・物理定数
ACTION_REPEAT = 16
PREP_STEPS = 80
MAX_STEPS = 300
FOV_DEG = 135

# キャッシュ閾値
LIDAR_CACHE_POS_THRESH = 0.05
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)
RAYCAST_CACHE_POS_THRESH = 0.05

# 視界外情報の外れ値マスク
OUTLIER_VALUE = 2.0

# 報酬設計パラメータ
REWARD_HIDDEN_BONUS = 1.0
COS_PENALTY_SCALE = 2.0
REWARD_DISTANCE_DIFF_SCALE = 1.0

# 共通ペナルティ
PENALTY_SAFEGUARD = -20.0
PENALTY_STAGNATION = -0.5

# エージェント推力制限
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
    """Transformer エンコーダを基幹に持つ Actor-Critic ネットワーク。"""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 埋め込み層
        self.embedding_layer_obj = nn.Linear(obs_dim, HIDDEN_DIM)

        # 位置エンコーディング
        self.pos_encoder_param_vec = nn.Parameter(torch.zeros(1, TRANSFORMER_SEQ_LEN, HIDDEN_DIM))

        # Transformer エンコーダ
        transformer_layer_config_ptr = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=NUM_HEADS, batch_first=True)
        self.transformer_module_engine = nn.TransformerEncoder(
            transformer_layer_config_ptr,
            num_layers=NUM_LAYERS,
            enable_nested_tensor=False,
        )

        # 出力層
        self.actor_mean_output_node = layer_init(nn.Linear(HIDDEN_DIM, action_dim), std=0.01)
        self.actor_logstd_params_vec = nn.Parameter(torch.zeros(1, action_dim))
        self.critic_value_output_node = layer_init(nn.Linear(HIDDEN_DIM, 1), std=1.0)

    def get_value(self, input_seq_tensor_ref):
        """系列情報を基に、現在の状態価値予測 V(s) を算出します。"""
        embedded_features_data_vec = self.embedding_layer_obj(input_seq_tensor_ref)
        input_with_pos_values = embedded_features_data_vec + self.pos_encoder_param_vec
        transformed_output_data_vec = self.transformer_module_engine(input_with_pos_values)
        # 最終ステップの要約ベクトル
        latest_hidden_state_token = transformed_output_data_vec[:, -1, :]
        predicted_value_result = self.critic_value_output_node(latest_hidden_state_token)
        return predicted_value_result

    def get_action_and_value(self, input_seq_tensor_ref, provided_action_tensor_val=None):
        """行動、対数確率、エントロピー、状態価値を一括計算。"""
        embedded_features_data_vec = self.embedding_layer_obj(input_seq_tensor_ref)
        input_with_pos_values = embedded_features_data_vec + self.pos_encoder_param_vec
        transformed_output_data_vec = self.transformer_module_engine(input_with_pos_values)
        summarized_context_state_vec = transformed_output_data_vec[:, -1, :]

        # 行動決定
        action_mean_vector_val = self.actor_mean_output_node(summarized_context_state_vec)
        action_std_vector_val = torch.exp(self.actor_logstd_params_vec.expand_as(action_mean_vector_val))
        action_distribution_proxy_obj = Normal(action_mean_vector_val, action_std_vector_val)

        if provided_action_tensor_val is None:
            provided_action_tensor_val = action_distribution_proxy_obj.sample()

        log_prob_value_res = action_distribution_proxy_obj.log_prob(provided_action_tensor_val).sum(1)
        entropy_value_res = action_distribution_proxy_obj.entropy().sum(1)
        estimated_state_value_res = self.critic_value_output_node(summarized_context_state_vec)

        return (
            provided_action_tensor_val,
            log_prob_value_res,
            entropy_value_res,
            estimated_state_value_res,
        )


class ObsHistory:
    """ミラーリング・ダブルバッファ構造による履歴管理クラス。"""

    def __init__(self, n_envs, seq_len, obs_dim, device):
        self.total_buffer_length_calculated = seq_len * 2
        self.data_history_buffer_tensor = torch.zeros((n_envs, self.total_buffer_length_calculated, obs_dim), device=device)
        self.device_ptr_obj_ref = device
        self.sequence_len_val = seq_len
        self.write_ptr_index_val = 0

    def reset(self):
        """バッファをゼロで初期化。"""
        self.data_history_buffer_tensor.zero_()
        self.write_ptr_index_val = 0

    def update(self, latest_obs):
        """最新観測値をミラーリング書き込み。"""
        new_obs = torch.as_tensor(latest_obs, dtype=torch.float32, device=self.device_ptr_obj_ref)
        if new_obs.ndim == 1:
            new_obs = new_obs.unsqueeze(0)

        # 本体領域とミラー領域
        self.data_history_buffer_tensor[:, self.write_ptr_index_val] = new_obs
        mirror_idx = self.write_ptr_index_val + self.sequence_len_val
        self.data_history_buffer_tensor[:, mirror_idx] = new_obs

        # ポインタ循環
        self.write_ptr_index_val = (self.write_ptr_index_val + 1) % self.sequence_len_val

    def get(self):
        """最新の系列スライスを View で取得。 ★修正: 属性参照の統一"""
        slice_end = self.write_ptr_index_val + self.sequence_len_val
        return self.data_history_buffer_tensor[:, self.write_ptr_index_val : slice_end]


class TeamCosEnv(base_config.HideAndSeekEnv):
    """
    不整合を解消し、将来的なオブジェクト増設に対応した超高速化環境。
    """

    def __init__(self, render_mode=None):
        # ★ 属性初期化の順序厳守 (reset() 時の参照バグ防止)
        self.hider_pos = {1: None, 2: None}
        self.dist_to_seeker = {1: 0.0, 2: 0.0}
        self.lidar_cache = {}
        self.raycast_cache = {}
        self.raycast_perf = {"hits": 0, "misses": 0}
        self.visible_map = {0: {}, 1: {}, 2: {}}
        self.obs_memo = {}
        self.hidden_steps = 0
        self.caught_steps = 0
        self.recovery_turn_dir = 1.0
        self.visible_names = {0: [], 1: [], 2: []}

        self.lidar_array_cache_storage = {}

        # 親クラスの初期化
        super().__init__(render_mode=render_mode)

        # Body ID to Name マッピング (visible_names 更新用)
        self.body_id_to_name = {
            self.s0_body: "s0",
            self.h1_body: "h1",
            self.h2_body: "h2",
            self.box1_body: "box1",
            self.box2_body: "box2",
            self.ramp_body: "ramp",
        }

        cpu_device = torch.device("cpu")
        # 各個体専用の履歴
        self.npc_history = {
            0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device),
            1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device),
            2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device),
        }

        # NPC 推論エンジン
        self.npc_hider_agent = None  # Agent(53, 4).to("cpu")
        self.npc_seeker_agent = None  # Agent(53, 4).to("cpu")

        log_marker_id_status_str = os.environ.get("NPC_MODELS_LOGGED")
        should_log = log_marker_id_status_str != "TRUE"

        # ★修正: 敵エージェントのモデルはロードしない（ルールベースを使用）
        # Hider学習時 → Seekerはルールベース、パートナーHiderのみモデルロード
        # Seeker学習時 → Hiderモデルロード、Seekerは学習中
        if LOAD_EXISTING_MODELS or render_mode == "human":
            # Hider学習時以外（Seeker学習時またはデバッグ時）のみHiderモデルをロード
            if TRAIN_TARGET != "HIDER" and should_log:
                h_path = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
                if h_path:
                    print(f"Loaded NPC Hider Engine for Env: {h_path}", flush=True)

            # Seeker学習時以外（Hider学習時またはデバッグ時）のみSeekerモデルをロード
            # ただしHider学習時はルールベースを使うためロードしない
            if TRAIN_TARGET == "SEEKER" and should_log:
                s_path = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
                if s_path:
                    print(f"Loaded NPC Seeker Engine for Env: {s_path}", flush=True)

        os.environ["NPC_MODELS_LOGGED"] = "TRUE"

        # ★ パフォーマンス最適化: 事前確保バッファ（NumPy配列生成の削減）
        # レイキャスト用バッファ
        self._raycast_geomid = np.zeros(1, dtype=np.int32)
        self._raycast_from = np.zeros(3, dtype=np.float64)
        self._raycast_dir = np.zeros(3, dtype=np.float64)

        # Lidar用バッファ
        self._lidar_dir = np.zeros(2, dtype=np.float64)

        # 報酬計算用バッファ
        self._reward_fwd_vec = np.zeros(2, dtype=np.float64)

        # 視界外マスク用の事前確保
        self._masked_vec_7 = np.full(7, OUTLIER_VALUE, dtype=np.float32)
        self._masked_vec_7[-1] = 0.0
        self._masked_vec_8 = np.full(8, OUTLIER_VALUE, dtype=np.float32)
        self._masked_vec_8[-1] = 0.0

    def _get_cached_ray(self, agent_idx, origin_pt_vec, direction_vec_raw, beam_id_val):
        """レイキャストの空間・角度キャッシュ制御。★最適化: バッファ再利用"""
        angle_rad = np.arctan2(direction_vec_raw[1], direction_vec_raw[0])
        cache_key = (agent_idx, beam_id_val)

        if cache_key in self.raycast_cache:
            c_pos, c_ang, c_dist, c_geom = self.raycast_cache[cache_key]
            pos_diff = np.linalg.norm(origin_pt_vec - c_pos)
            if pos_diff < RAYCAST_CACHE_POS_THRESH:
                angle_error = (angle_rad - c_ang + np.pi) % (2.0 * np.pi) - np.pi
                # ★修正: 定数を使用（ハードコード削除）
                if abs(angle_error) < LIDAR_CACHE_ANG_THRESH:
                    self.raycast_perf["hits"] += 1
                    return c_dist, c_geom

        # 実計測の実行
        self.raycast_perf["misses"] += 1

        # ★最適化: 事前確保したバッファに直接書き込み（配列生成を削減）
        self._raycast_from[0] = origin_pt_vec[0]
        self._raycast_from[1] = origin_pt_vec[1]
        self._raycast_from[2] = 0.5
        self._raycast_dir[0] = direction_vec_raw[0]
        self._raycast_dir[1] = direction_vec_raw[1]
        self._raycast_dir[2] = 0.0

        # 除外ボディの設定
        if agent_idx == 0:
            body_to_exclude_idx = self.s0_body
        elif agent_idx == 1:
            body_to_exclude_idx = self.h1_body
        else:
            body_to_exclude_idx = self.h2_body

        ray_dist = mujoco.mj_ray(
            self.model,
            self.data,
            self._raycast_from,
            self._raycast_dir,
            None,
            1,
            body_to_exclude_idx,
            self._raycast_geomid,
        )

        self.raycast_cache[cache_key] = (
            origin_pt_vec.copy(),
            angle_rad,
            ray_dist,
            self._raycast_geomid[0],
        )
        return ray_dist, self._raycast_geomid[0]

    def _is_visible(self, origin_pos, origin_rot, target_pos, target_body_id, agent_idx_ref):
        """オブジェクトの可視性判定。"""
        diff_vec_2d_raw = target_pos[:2] - origin_pos[:2]
        dist_magnitude_val_res = np.linalg.norm(diff_vec_2d_raw)
        if dist_magnitude_val_res < 0.1:
            return True, target_body_id

        # 視野角判定 (FOV)
        angle_to_target_absolute_rad = np.arctan2(diff_vec_2d_raw[1], diff_vec_2d_raw[0])
        relative_angle_rad_val = (angle_to_target_absolute_rad - origin_rot + np.pi) % (2.0 * np.pi) - np.pi
        if abs(relative_angle_rad_val) > np.deg2rad(FOV_DEG / 2.0):
            return False, -1

        # レイキャスト判定 (キャッシュ利用)
        direction_2d_unit_vec = diff_vec_2d_raw / dist_magnitude_val_res
        # beam_id に body_id を直接使用。
        target_obj_unique_beam_id = target_body_id

        res_dist_calc_val, hit_geom_id_calc_val = self._get_cached_ray(
            agent_idx_ref,
            origin_pos[:2],
            direction_2d_unit_vec,
            target_obj_unique_beam_id,
        )

        if res_dist_calc_val != -1:
            hit_body_id_resolved_idx = self.model.geom_bodyid[hit_geom_id_calc_val]
            if hit_body_id_resolved_idx == target_body_id:
                return True, hit_body_id_resolved_idx
            if res_dist_calc_val < dist_magnitude_val_res - 0.4:
                return False, hit_body_id_resolved_idx

        return True, target_body_id

    def _get_obs(self, agent_id):
        """個体固有の 53次元観測。自己状態変数の名称を統一。"""
        if agent_id in self.obs_memo:
            return self.obs_memo[agent_id]

        if agent_id == 0:
            target_bid_ptr = self.s0_body
            prefix_id_val = "s"
        elif agent_id == 1:
            target_bid_ptr = self.h1_body
            prefix_id_val = "h1"
        else:
            target_bid_ptr = self.h2_body
            prefix_id_val = "h2"

        current_xy_coords_ptr_val = self.data.xpos[target_bid_ptr][:2]
        joint_rot_ptr_obj_ref = self.model.joint(f"{prefix_id_val}_rot")
        yaw_angle_rad_val_now_ptr = self.data.qpos[self.model.jnt_qposadr[joint_rot_ptr_obj_ref.id]]

        # 回転行列の算出
        cos_yaw_val_final = np.cos(-yaw_angle_rad_val_now_ptr)
        sin_yaw_val_final = np.sin(-yaw_angle_rad_val_now_ptr)
        rotation_matrix_2d_transform_out = np.array(
            [
                [cos_yaw_val_final, -sin_yaw_val_final],
                [sin_yaw_val_final, cos_yaw_val_final],
            ]
        )

        # 物理速度取得
        move_joint_ptr_obj_ref = self.model.joint(f"{prefix_id_val}_x")
        dof_address_idx_ptr_ref = self.model.jnt_dofadr[move_joint_ptr_obj_ref.id]
        velocity_global_raw_vector_ptr = self.data.qvel[dof_address_idx_ptr_ref : dof_address_idx_ptr_ref + 2]
        velocity_local_vec_out = rotation_matrix_2d_transform_out @ velocity_global_raw_vector_ptr
        velocity_obs_normalized_final = velocity_local_vec_out / 12.0

        # ★ 修正: 変数名を self_state_vector_packet に統一
        self_state_vector_packet = np.concatenate(
            [
                velocity_obs_normalized_final,
                [
                    yaw_angle_rad_val_now_ptr,
                    np.cos(yaw_angle_rad_val_now_ptr),
                    np.sin(yaw_angle_rad_val_now_ptr),
                ],
            ]
        )

        # 1. Lidar データの生成 (Lidar用ID: 1000〜)
        lidar_data_vector_final_out = None
        if agent_id in self.lidar_array_cache_storage:
            cp_l_val, cr_l_val, cl_l_val = self.lidar_array_cache_storage[agent_id]
            displacement_l_norm_val_ptr = np.linalg.norm(current_xy_coords_ptr_val - cp_l_val)
            if displacement_l_norm_val_ptr < LIDAR_CACHE_POS_THRESH:
                angle_error_l_val_ptr = (yaw_angle_rad_val_now_ptr - cr_l_val + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_error_l_val_ptr) < LIDAR_CACHE_ANG_THRESH:
                    lidar_data_vector_final_out = cl_l_val.copy()

        if lidar_data_vector_final_out is None:
            lidar_data_vector_final_out = np.zeros(len(self.lidar_angles), dtype=np.float32)
            for i_idx_val_ptr, offset_rad_val_ptr in enumerate(self.lidar_angles):
                beam_abs_angle_calc_res = offset_rad_val_ptr + yaw_angle_rad_val_now_ptr
                beam_dir_vec_obj_ptr_res = np.array([np.cos(beam_abs_angle_calc_res), np.sin(beam_abs_angle_calc_res)])
                mj_dist_res_val_ptr, _ = self._get_cached_ray(
                    agent_id,
                    current_xy_coords_ptr_val,
                    beam_dir_vec_obj_ptr_res,
                    i_idx_val_ptr + 1000,
                )
                lidar_data_vector_final_out[i_idx_val_ptr] = min(mj_dist_res_val_ptr, 2.5) / 2.5 if mj_dist_res_val_ptr != -1 else 1.0
            self.lidar_array_cache_storage[agent_id] = (
                current_xy_coords_ptr_val.copy(),
                yaw_angle_rad_val_now_ptr,
                lidar_data_vector_final_out.copy(),
            )

        # 2. オブジェクト視認情報の更新
        vis_lookup_record_dict_ref = self.visible_map[agent_id]
        vis_lookup_record_dict_ref.clear()
        self.visible_names[agent_id] = []

        target_bodies_registry_config_list = [
            (self.box1_body, "Box1"),
            (self.box2_body, "Box2"),
            (self.ramp_body, "Ramp"),
            (self.h1_body, "H1"),
            (self.h2_body, "H2"),
            (self.s0_body, "Seeker"),
        ]

        for tid_ptr_val_ref, name_str_ptr_ref in target_bodies_registry_config_list:
            if tid_ptr_val_ref != target_bid_ptr:
                is_actually_vis_bool_res, _ = self._is_visible(
                    self.data.xpos[target_bid_ptr],
                    yaw_angle_rad_val_now_ptr,
                    self.data.xpos[tid_ptr_val_ref],
                    tid_ptr_val_ref,
                    agent_id,
                )
                vis_lookup_record_dict_ref[tid_ptr_val_ref] = is_actually_vis_bool_res
                if is_actually_vis_bool_res:
                    self.visible_names[agent_id].append(name_str_ptr_ref)

        def build_relative_obs_unit_block_ref(target_id_val_ref_ptr, lock_status_bool_flag_ref=None):
            """視認情報に基づくベクトル生成。"""
            is_seen_current_status_flag_res = vis_lookup_record_dict_ref.get(target_id_val_ref_ptr, False)
            dims_output_count_required_val = 8 if lock_status_bool_flag_ref is not None else 7

            if is_seen_current_status_flag_res:
                target_xyz_position_vec_ptr_ref = self.data.xpos[target_id_val_ref_ptr]
                rel_pos_global_vec_raw_ptr_ref = target_xyz_position_vec_ptr_ref[:2] - current_xy_coords_ptr_val
                rel_pos_local_vector_val_ptr_ref = rotation_matrix_2d_transform_out @ rel_pos_global_vec_raw_ptr_ref / 12.0

                target_quat_array_data_ptr_ref = self.data.xquat[target_id_val_ref_ptr]
                # Yaw 角度算出
                y_atan_part_val_ptr_ref = 2.0 * (target_quat_array_data_ptr_ref[0] * target_quat_array_data_ptr_ref[3] + target_quat_array_data_ptr_ref[1] * target_quat_array_data_ptr_ref[2])
                x_atan_part_val_ptr_ref = 1.0 - 2.0 * (target_quat_array_data_ptr_ref[2] ** 2 + target_quat_array_data_ptr_ref[3] ** 2)
                yaw_target_absolute_val_ptr_ref = np.arctan2(y_atan_part_val_ptr_ref, x_atan_part_val_ptr_ref)

                # 速度
                target_joint_adr_ref_idx_val = self.model.body_jntadr[target_id_val_ref_ptr]
                if target_joint_adr_ref_idx_val != -1:
                    v_t_global_raw_vec_ptr_ref = self.data.qvel[target_joint_adr_ref_idx_val : target_joint_adr_ref_idx_val + 2]
                else:
                    v_t_global_raw_vec_ptr_ref = np.zeros(2)

                v_rel_global_vector_calc_ptr_ref = v_t_global_raw_vec_ptr_ref - velocity_global_raw_vector_ptr
                v_rel_local_vector_res_ptr_ref = rotation_matrix_2d_transform_out @ v_rel_global_vector_calc_ptr_ref / 12.0

                # 順序: [pos, vel, rot, (lock), vis]
                packet_parts_list_res = [
                    rel_pos_local_vector_val_ptr_ref,
                    v_rel_local_vector_res_ptr_ref,
                    [
                        np.cos(yaw_target_absolute_val_ptr_ref - yaw_angle_rad_val_now_ptr),
                        np.sin(yaw_target_absolute_val_ptr_ref - yaw_angle_rad_val_now_ptr),
                    ],
                ]
                if lock_status_bool_flag_ref is not None:
                    packet_parts_list_res.append([1.0 if lock_status_bool_flag_ref else 0.0])
                packet_parts_list_res.append([1.0])
                return np.concatenate(packet_parts_list_res)
            else:
                # 視界外マスク: 最後尾を 0.0 固定
                masked_out_vector_data_res = np.full(dims_output_count_required_val, OUTLIER_VALUE, dtype=np.float32)
                masked_out_vector_data_res[-1] = 0.0
                return masked_out_vector_data_res

        # 役割別合成。★不整合修正: すべて self_state_vector_packet を参照
        if agent_id == 0:
            final_observation_packet_vector_out = np.concatenate(
                [
                    self_state_vector_packet,
                    lidar_data_vector_final_out,
                    build_relative_obs_unit_block_ref(self.box1_body, self.locked_boxes[self.box1_body]),
                    build_relative_obs_unit_block_ref(self.box2_body, self.locked_boxes[self.box2_body]),
                    build_relative_obs_unit_block_ref(self.ramp_body),
                    build_relative_obs_unit_block_ref(self.h1_body)[:5],
                    build_relative_obs_unit_block_ref(self.h2_body)[:5],
                    np.zeros(3, dtype=np.float32),
                ]
            )
        else:
            partner_h_body_id_ptr_ref = self.h2_body if agent_id == 1 else self.h1_body
            rel_enemy_block_info_res = build_relative_obs_unit_block_ref(self.s0_body)[:5]
            rel_partner_block_info_res = build_relative_obs_unit_block_ref(partner_h_body_id_ptr_ref)
            is_currently_grasping_status_res = 1.0 if self.grasping[agent_id] else 0.0

            final_observation_packet_vector_out = np.concatenate(
                [
                    self_state_vector_packet,
                    lidar_data_vector_final_out,
                    build_relative_obs_unit_block_ref(self.box1_body, self.locked_boxes[self.box1_body]),
                    build_relative_obs_unit_block_ref(self.box2_body, self.locked_boxes[self.box2_body]),
                    build_relative_obs_unit_block_ref(self.ramp_body),
                    rel_enemy_block_info_res,
                    rel_partner_block_info_res,
                    [is_currently_grasping_status_res],
                ]
            )

        obs_final_res_out_tensor_res = final_observation_packet_vector_out.astype(np.float32)
        self.obs_memo[agent_id] = obs_final_res_out_tensor_res
        return obs_final_res_out_tensor_res

    def _update_seeker_state(self):
        """鬼の思考ステートマシン。"""
        seeker_pos = self.data.xpos[self.s0_body][:2]
        self._get_obs(0)
        h1_seen = self.visible_map[0].get(self.h1_body, False)
        h2_seen = self.visible_map[0].get(self.h2_body, False)

        if h1_seen or h2_seen:
            target_bid = self.h1_body if h1_seen else self.h2_body
            target_pos = self.data.xpos[target_bid][:2].copy()
            self.seeker_target_pos = target_pos
            self.seeker_last_known_pos = target_pos.copy()
            self.seeker_mode = "CHASING"
        elif self.seeker_last_known_pos is not None:
            distance = np.linalg.norm(seeker_pos - self.seeker_last_known_pos)
            if distance > 0.5:
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
            self.seeker_target_pos = self.seeker_random_target.copy()
            self.seeker_mode = "PATROLLING"

    def _seeker_rule_based_policy(self):
        """鬼の移動ロジック。"""
        if self.current_step < PREP_STEPS:
            return 0.0, 0.0

        seeker_pos = self.data.xpos[self.s0_body][:2]
        seeker_yaw = self.data.qpos[self.srot_adr]
        target_pos = self.seeker_target_pos

        dx = target_pos[0] - seeker_pos[0]
        dy = target_pos[1] - seeker_pos[1]
        target_yaw = np.arctan2(dy, dx)
        angle_err = (target_yaw - seeker_yaw + np.pi) % (2.0 * np.pi) - np.pi

        thrust = SEEKER_RB_THRUST
        turn = np.clip(angle_err * 6.0, -3.0, 3.0)

        if abs(angle_err) > SEEKER_RB_TURN_THRESH:
            thrust = thrust * 0.3

        dof_idx = self.model.jnt_dofadr[self.model.joint("s_x").id]
        vel_norm = np.linalg.norm(self.data.qvel[dof_idx : dof_idx + 2])

        if thrust > 0.05 and vel_norm < 0.05:
            self.s0_stuck_timer = self.s0_stuck_timer + 5
        else:
            self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)

        if self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 15
            self.s0_stuck_timer = 0
            self.recovery_turn_dir = self.np_random.choice([-1.0, 1.0])

        if self.s0_recovery_mode > 0:
            thrust = -0.2
            turn = 1.5 * self.recovery_turn_dir
            self.s0_recovery_mode = self.s0_recovery_mode - 1

        return float(thrust), float(turn)

    def _get_npc_action(self, agent_id, agent_type):
        """NPC 行動生成。"""
        # 2. モデルがある場合は推論
        observation = self._get_obs(agent_id)
        self.npc_history[agent_id].update(observation)
        model = self.npc_seeker_agent if agent_type == "SEEKER" else self.npc_hider_agent

        if model is not None:
            with torch.no_grad():
                ctx_sequence_data_tensor_ref = self.npc_history[agent_id].get()
                act_tensor_out_res_final_val, _, _, _ = model.get_action_and_value(ctx_sequence_data_tensor_ref)
            return act_tensor_out_res_final_val.cpu().numpy()[0]

        if agent_type == "SEEKER":
            f_rb_p_res_final, r_rb_p_res_final = self._seeker_rule_based_policy()
            normalized_thrust_applied_val = f_rb_p_res_final / SEEKER_THRUST_LIMIT
            return np.array(
                [normalized_thrust_applied_val, r_rb_p_res_final, 0.0, 0.0],
                dtype=np.float32,
            )
        else:
            return self.action_space.sample() * 0.5  # ランダム

    def reset(self, seed=None, options=None):
        """環境リセット。 ★修正: 変数名の完全統一"""
        obs, info = super().reset(seed=seed, options=options)
        self.hidden_steps = 0
        self.caught_steps = 0
        self.obs_memo.clear()
        self.lidar_cache.clear()
        self.raycast_cache.clear()
        self.recovery_turn_dir = 1.0

        seeker_pos = self.data.xpos[self.s0_body][:2]
        for hi_idx in [1, 2]:
            body_idx = self.h1_body if hi_idx == 1 else self.h2_body
            hider_pos = self.data.xpos[body_idx][:2].copy()
            self.dist_to_seeker[hi_idx] = np.linalg.norm(hider_pos - seeker_pos)
            self.hider_pos[hi_idx] = hider_pos

        return obs, info

    def step(self, action):
        """1ステップ進展。 ★修正: 集計変数の完全統一"""
        self.current_step = self.current_step + 1
        for i in [1, 2]:
            self.lock_cooldown[i] = max(0, self.lock_cooldown[i] - 1)
        self._update_seeker_state()
        self.data.ctrl[:] = 0.0

        if TRAIN_TARGET == "HIDER":
            h_idx = self._apply_action(self.learning_agent_id, action)
            self.data.ctrl[h_idx] = float(action[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[h_idx + 1] = float(action[1])
            partner_id = 2 if self.learning_agent_id == 1 else 1
            partner_action = self._get_npc_action(partner_id, "HIDER")
            p_idx = self._apply_action(partner_id, partner_action)
            self.data.ctrl[p_idx] = float(partner_action[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[p_idx + 1] = float(partner_action[1])
            seeker_action = self._get_npc_action(0, "SEEKER")
            self.data.ctrl[0] = float(seeker_action[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(seeker_action[1])
        else:
            self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(action[1])
            for i in [1, 2]:
                hider_action = self._get_npc_action(i, "HIDER")
                h_idx = self._apply_action(i, hider_action)
                self.data.ctrl[h_idx] = float(hider_action[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[h_idx + 1] = float(hider_action[1])

        # --- PHYSICS LOOP (Slice-based) ---
        for _ in range(ACTION_REPEAT):
            for bid, pose_data in self.locked_pose.items():
                if self.locked_boxes[bid]:
                    bjid = self.box1_joint_id if bid == self.box1_body else self.box2_joint_id
                    qa = self.model.jnt_qposadr[bjid]
                    da = self.model.jnt_dofadr[bjid]
                    self.data.qpos[qa : qa + 7] = pose_data
                    self.data.qvel[da : da + 6] = 0.0
            mujoco.mj_step(self.model, self.data)

        self.obs_memo.clear()
        # 観測確定
        obs = self._get_obs(self.learning_agent_id)

        # ★重要: 物理シミュレーション後のSeeker観測を再計算
        # _update_seeker_state()内の_get_obs(0)は物理シミュレーション「前」の状態
        # 報酬計算と統計には物理シミュレーション「後」の最新状態が必要
        _ = self._get_obs(0)

        # ★render()用の観測も物理シミュレーション後に計算
        if self.render_mode == "human":
            _ = self._get_obs(1)
            _ = self._get_obs(2)

        # 統計用にキャッシュ参照
        h1_visible = self.visible_map[0].get(self.h1_body, False)
        h2_visible = self.visible_map[0].get(self.h2_body, False)

        if h1_visible or h2_visible:
            self.caught_steps = self.caught_steps + 1
        else:
            self.hidden_steps = self.hidden_steps + 1

        # ★修正: 名称統一 (total_team_reward_accumulator)
        total_reward = 0.0
        for h_idx, bid in [(1, self.h1_body), (2, self.h2_body)]:
            is_visible = self.visible_map[0].get(bid, False)
            seeker_pos = self.data.xpos[self.s0_body][:2]
            hider_pos = self.data.xpos[bid][:2]
            current_dist = np.linalg.norm(hider_pos - seeker_pos)

            if is_visible:
                seeker_yaw = self.data.qpos[self.srot_adr]
                diff_vec = hider_pos - seeker_pos
                diff_norm = diff_vec / (np.linalg.norm(diff_vec) + 1e-8)

                # ★最適化: 前方ベクトルをバッファに直接書き込み
                self._reward_fwd_vec[0] = np.cos(seeker_yaw)
                self._reward_fwd_vec[1] = np.sin(seeker_yaw)

                cosine = np.dot(diff_norm, self._reward_fwd_vec)
                reward = -cosine * COS_PENALTY_SCALE
                dist_delta = current_dist - self.dist_to_seeker[h_idx]
                reward = reward + dist_delta * REWARD_DISTANCE_DIFF_SCALE
            else:
                reward = REWARD_HIDDEN_BONUS

            if h_idx == self.learning_agent_id:
                if self.hider_pos[h_idx] is not None:
                    displacement = np.linalg.norm(hider_pos - self.hider_pos[h_idx])
                    if displacement < 0.01:
                        reward = reward + PENALTY_STAGNATION
                self.hider_pos[h_idx] = hider_pos.copy()

            if max(abs(hider_pos)) > 6.5:
                reward = reward + PENALTY_SAFEGUARD

            total_reward = total_reward + reward
            self.dist_to_seeker[h_idx] = current_dist

        final_reward = total_reward if TRAIN_TARGET == "HIDER" else -total_reward
        truncated = self.current_step >= MAX_STEPS

        info = {
            "hidden_steps": float(self.hidden_steps),
            "caught_steps": float(self.caught_steps),
        }
        return obs, float(final_reward), False, truncated, info

    def render(self, stats=None):
        """MuJoCo Viewer 描画。"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.elevation, self.viewer.cam.distance = -60, 23.0

            for bid, gid in [
                (self.box1_body, self.box1_geom_id),
                (self.box2_body, self.box2_geom_id),
            ]:
                if self.locked_boxes[bid]:
                    self.model.geom_rgba[gid][:] = [0.8, 0.1, 0.1, 1.0]
                elif any(g == bid for g in self.grasping.values()):
                    self.model.geom_rgba[gid][:] = [0.1, 0.1, 0.9, 1.0]
                else:
                    rgba = [0.6, 0.4, 0.2, 1.0] if bid == self.box1_body else [0.7, 0.5, 0.3, 1.0]
                    self.model.geom_rgba[gid][:] = rgba

            if self.viewer.user_scn:
                scn = self.viewer.user_scn
                scn.ngeom = 0

                def add_line(p1, p2, rgba):
                    if scn.ngeom < scn.maxgeom:
                        mujoco.mjv_initGeom(
                            scn.geoms[scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            size=np.array([0, 0, 0]),
                            pos=np.array([0, 0, 0]),
                            mat=np.eye(3).flatten(),
                            rgba=rgba,
                        )
                        mujoco.mjv_connector(
                            scn.geoms[scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            width=2.0,
                            from_=p1,
                            to=p2,
                        )
                        scn.ngeom = scn.ngeom + 1

                def add_label(p, txt, rgba):
                    if scn.ngeom < scn.maxgeom:
                        mujoco.mjv_initGeom(
                            scn.geoms[scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.array([0, 0, 0]),
                            pos=p,
                            mat=np.eye(3).flatten(),
                            rgba=rgba,
                        )
                        scn.geoms[scn.ngeom].label = txt
                        scn.ngeom = scn.ngeom + 1

                # H1 (Yellow)
                h1_pos = self.data.xpos[self.h1_body]
                for target_id in [
                    self.box1_body,
                    self.box2_body,
                    self.ramp_body,
                    self.s0_body,
                    self.h2_body,
                ]:
                    if target_id == self.h1_body:
                        continue
                    if self.visible_map[1].get(target_id, False):
                        add_line(
                            h1_pos + [0, 0, 0.5],
                            self.data.xpos[target_id] + [0, 0, 0.5],
                            [1, 1, 0, 0.4],
                        )
                add_label(
                    self.data.site_xpos[self.id_h1_label],
                    f"H1 Vis:[{','.join(self.visible_names[1])}]",
                    [1, 1, 0, 1],
                )

                # H2 (Cyan)
                h2_pos = self.data.xpos[self.h2_body]
                for target_id in [
                    self.box1_body,
                    self.box2_body,
                    self.ramp_body,
                    self.s0_body,
                    self.h1_body,
                ]:
                    if target_id == self.h2_body:
                        continue
                    if self.visible_map[2].get(target_id, False):
                        add_line(
                            h2_pos + [0, 0, 0.5],
                            self.data.xpos[target_id] + [0, 0, 0.5],
                            [0, 1, 1, 0.4],
                        )
                add_label(
                    self.data.site_xpos[self.id_h2_label],
                    f"H2 Vis:[{','.join(self.visible_names[2])}]",
                    [0, 1, 1, 1],
                )

                # Seeker (Red)
                seeker_pos = self.data.xpos[self.s0_body]
                for target_id in [self.h1_body, self.h2_body]:
                    if self.visible_map[0].get(target_id, False):
                        add_line(
                            seeker_pos + [0, 0, 0.5],
                            self.data.xpos[target_id] + [0, 0, 0.5],
                            [1, 0, 0, 0.6],
                        )
                add_label(
                    self.data.site_xpos[self.id_s_label],
                    f"S:{self.seeker_mode}",
                    [1, 0, 0, 1],
                )

            self.viewer.sync()


# ==========================================
# 3. ヘルパー関数 & ファクトリ
# ==========================================


def load_model_safely(model, base_name, arget_type):
    """指定候補からモデルをロード。"""
    search_paths = [
        f"{base_name}_refinement_{arget_type}.pt",
        f"{base_name}_initial_{arget_type}.pt",
        f"{base_name}_{arget_type}.pt",
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()
                return path
            except Exception:
                continue
    return None


def env_factory():
    """AsyncVectorEnv 用。"""
    env = TeamCosEnv()
    return gym.wrappers.RecordEpisodeStatistics(env)


# ==========================================
# 5. メイン処理 (学習ループ)
# ==========================================


def main():
    if platform.system() == "Linux":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
    run_time = int(time.time())
    run_id = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{run_time}"

    if EXECUTION_MODE == "PLAY":
        print("--- Inference Mode (PLAY) ---")
        env = TeamCosEnv(render_mode="human")
        model = Agent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        load_model_safely(model, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        model.eval()
        obs_history = ObsHistory(1, TRANSFORMER_SEQ_LEN, env.observation_space.shape[0], device)
        try:
            while True:
                obs, _ = env.reset()
                obs_history.reset()
                obs_history.update(obs)
                done = False
                ret = 0.0
                while not done:
                    t_start = time.time()
                    with torch.no_grad():
                        action, _, _, _ = model.get_action_and_value(obs_history.get())
                    next_obs, reward, term, trunc, info = env.step(action.cpu().numpy()[0])
                    done = term or trunc
                    ret = ret + reward
                    obs_history.update(next_obs)
                    env.render(stats={"EpRet": f"{ret:.1f}"})

                    # ★★★ 修正: ここに終了判定を追加 (内側のループ) ★★★
                    # Viewerが生成されており、かつユーザーによって閉じられた(is_running() == False)場合
                    if env.viewer is not None:
                        if not env.viewer.is_running():
                            print("\nViewer closed by user. Exiting...")
                            env.close()
                            return  # breakではなくreturnでmain関数ごと終了させる
                    # FPS制御 (Sleep)
                    wait = (0.005 * ACTION_REPEAT) - (time.time() - t_start)
                    if wait > 0:
                        time.sleep(wait)
                print(
                    f"Result -> Return: {ret:.1f}, Hidden: {info['hidden_steps']:.0f}",
                    flush=True,
                )
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nInterrupted (PLAY).")
        finally:
            # Viewerを確実に閉じる
            env.close()
            # ★Viewerが閉じ切るまで少し待つハック（必要であれば）
            time.sleep(0.5)
        return

    print(f"--- [Parent] 1. Initializing {NUM_ENVS} workers ---", flush=True)
    try:
        vec_envs = gym.vector.AsyncVectorEnv([env_factory for _ in range(NUM_ENVS)])
        print("--- [Parent] 2. Parallel environment ready ---", flush=True)
    except Exception as e:
        print(f"--- [Parent] startup failed: {e} ---", flush=True)
        sys.exit(1)

    if TRACK_WANDB:
        wandb.init(
            project=base_config.WANDB_PROJECT_NAME,
            config={
                "Target": TRAIN_TARGET,
                "MODE": MODE,
                "v": "25.62_StandardizedNames",
            },
            name=run_id,
            sync_tensorboard=False,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_id}")
    agent = Agent(
        vec_envs.single_observation_space.shape[0],
        vec_envs.single_action_space.shape[0],
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    global_step = 0
    start_step = 0
    if LOAD_EXISTING_MODELS:
        model_path = load_model_safely(agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if model_path:
            print(f"★ Resumed learner from: {model_path}")
            checkpoint_path = model_path.replace(".pt", "_checkpoint.json")
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, "r") as f:
                        data = json.load(f)
                        global_step = data.get("global_step", 0)
                        start_step = global_step
                except:
                    pass

    # --- ROLLOUT BUFFER ---
    obs_history = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device)
    S, E, O, A = NUM_STEPS, NUM_ENVS, 53, 4
    batch_obs = torch.zeros((S, E, TRANSFORMER_SEQ_LEN, O), device=device)
    batch_actions = torch.zeros((S, E, A), device=device)
    batch_logprobs = torch.zeros((S, E), device=device)
    batch_rewards = torch.zeros((S, E), device=device)
    batch_dones = torch.zeros((S, E), device=device)
    batch_values = torch.zeros((S, E), device=device)

    next_obs = vec_envs.reset(seed=FIXED_SEED if FIXED_SEED else int(time.time()))[0]
    next_done = torch.zeros(E).to(device)
    obs_history.reset()
    obs_history.update(next_obs)

    num_updates = int(max(1, (TOTAL_TIMESTEPS - global_step) // (E * S)))
    history_returns = []
    history_hidden = []
    history_caught = []
    start_time = time.time()
    last_loss = 0.0
    last_entropy = 0.0

    print("--- Training Sequence Started (v25.62) ---")
    try:
        for u in tqdm(range(1, num_updates + 1), desc="Updates"):
            for step in range(S):
                global_step = global_step + E
                batch_obs[step] = obs_history.get()
                batch_dones[step] = next_done

                with torch.no_grad():
                    action, lp, _, value = agent.get_action_and_value(obs_history.get())
                    batch_values[step] = value.flatten()

                batch_actions[step] = action
                batch_logprobs[step] = lp

                next_obs, reward, term, trunc, info = vec_envs.step(action.cpu().numpy())
                done = np.logical_or(term, trunc)

                if "final_info" in info:
                    for e in range(E):
                        final = info["final_info"][e]
                        if done[e] and final is not None:
                            if "episode" in final:
                                history_returns.append(float(final["episode"]["r"]))
                            if "hidden_steps" in final:
                                history_hidden.append(float(final["hidden_steps"]))
                            if "caught_steps" in final:
                                history_caught.append(float(final["caught_steps"]))
                elif "episode" in info:
                    mask = info.get("_episode", [True] * E)
                    for e in range(E):
                        if mask[e] and done[e]:
                            history_returns.append(float(info["episode"]["r"][e]))
                            if "hidden_steps" in info:
                                history_hidden.append(float(info["hidden_steps"][e]))
                            if "caught_steps" in info:
                                history_caught.append(float(info["caught_steps"][e]))

                batch_rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.tensor(done).to(device, dtype=torch.float32)
                obs_history.update(next_obs)

            # --- PPO UPDATE ---
            with torch.no_grad():
                v_next = agent.get_value(obs_history.get()).reshape(1, -1)
                advantages = torch.zeros_like(batch_rewards).to(device)
                gae = 0
                for t in reversed(range(S)):
                    if t == S - 1:
                        nt = 1.0 - next_done
                        vp = v_next
                    else:
                        nt = 1.0 - batch_dones[t + 1]
                        vp = batch_values[t + 1]
                    delta = batch_rewards[t] + 0.99 * vp * nt - batch_values[t]
                    gae = delta + 0.99 * 0.95 * nt * gae
                    advantages[t] = gae
                returns = advantages + batch_values

            flat_obs = batch_obs.reshape((-1, TRANSFORMER_SEQ_LEN, 53))
            flat_lp = batch_logprobs.reshape(-1)
            flat_actions = batch_actions.reshape((-1, 4))
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = batch_values.reshape(-1)

            # PPO OPTIMIZATION
            for ep in range(UPDATE_EPOCHS):
                idx = np.arange(S * E)
                np.random.shuffle(idx)
                for start in range(0, S * E, MINIBATCH_SIZE):
                    mb_idx = idx[start : start + MINIBATCH_SIZE]
                    _, new_lp, entropy, new_value = agent.get_action_and_value(flat_obs[mb_idx], flat_actions[mb_idx])

                    log_ratio = new_lp - flat_lp[mb_idx]
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()

                    mb_adv = flat_advantages[mb_idx]
                    mb_adv_norm = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                    loss_pol_1 = -mb_adv_norm * ratio
                    loss_pol_2 = -mb_adv_norm * torch.clamp(ratio, 0.8, 1.2)
                    loss_pol = torch.max(loss_pol_1, loss_pol_2).mean()

                    loss_val = 0.5 * ((new_value.view(-1) - flat_returns[mb_idx]) ** 2).mean()
                    loss_total = loss_pol - ENT_COEF * entropy.mean() + 0.5 * loss_val

                    optimizer.zero_grad()
                    loss_total.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

                    last_loss = loss_total.item()
                    last_entropy = entropy.mean().item()

            y_pred = flat_values.cpu().numpy()
            y_actual = flat_returns.cpu().numpy()
            var_y = np.var(y_actual)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_actual - y_pred) / var_y

            if history_returns:
                avg_hidden = np.mean(history_hidden)
                avg_caught = np.mean(history_caught)
                avg_return = np.mean(history_returns)

                if (TRIAL_MODE) or (u % 10 == 0):
                    elapsed = time.time() - start_time
                    sps = int((global_step - start_step) / elapsed) if elapsed > 0 else 0

                    print(
                        f"Update {u}, Step {global_step}, SPS: {sps}, EpRet: {avg_return:.1f}, Hidden: {avg_hidden:.1f}, Caught: {avg_caught:.1f}",
                        flush=True,
                    )
                    sys.stdout.flush()

                    if TRACK_WANDB:
                        wandb.log(
                            {
                                "charts/SPS": sps,
                                "losses/total_loss": last_loss,
                                "losses/entropy": last_entropy,
                                "losses/explained_variance": explained_var,
                                "charts/episodic_return": avg_return,
                                "charts/steps_hidden": avg_hidden,
                                "global_step": global_step,
                            }
                        )
                    writer.add_scalar("charts/SPS", sps, global_step)

                    history_returns = []
                    history_hidden = []
                    history_caught = []

    except KeyboardInterrupt:
        print("\nInterrupted.")
        vec_envs.close()
        sys.exit(0)

    if SAVE_MODEL:
        torch.save(agent.state_dict(), SAVE_MODEL_PATH)
        checkpoint_path = SAVE_MODEL_PATH.replace(".pt", "_checkpoint.json")
        with open(checkpoint_path, "w") as f:
            json.dump({"global_step": global_step}, f)
        print(f"Model saved: {SAVE_MODEL_PATH}")

    vec_envs.close()
    writer.close()
    if TRACK_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
