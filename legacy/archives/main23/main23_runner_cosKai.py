# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
#
# 【修正内容 (v25.62)】
# 1. 変数名の完全な一貫性確保:
#    - _get_obs 内の自己状態ベクトルを self_state_vector_packet に統一。
#    - 以前の版で混入していた不自然な locals() による別名参照や回避策を完全に削除しました。
#    - ObsHistory における属性参照の不整合を修正し、hasattr による冗長な分岐を排除。
# 2. 報酬・統計ロジックの整合性向上:
#    - step() メソッド内の team_reward 集計変数と、reset() 内のループ変数のタイポを修正。
#    - 不可視マスク時の最後尾フラグ (Visible Flag) が確実に 0.0 になる制御を維持。
# 3. 物理・キャッシュ・高速化の維持:
#    - スライスによる物理状態代入により可読性を確保しつつ、他は 1 行 1 動作を徹底。
#    - オブジェクト ID を beam_id に流用する衝突回避設計を継続採用。
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
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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
EXECUTION_MODE = "PLAY"

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


def layer_init(layer_object_ptr_val, std_val_init_ref=np.sqrt(2), bias_val_init_ref=0.0):
    """ネットワーク重みの直交初期化を行い、学習初期の安定性を確保します。"""
    nn.init.orthogonal_(layer_object_ptr_val.weight, std_val_init_ref)
    nn.init.constant_(layer_object_ptr_val.bias, bias_val_init_ref)
    return layer_object_ptr_val


class Agent(nn.Module):
    """Transformer エンコーダを基幹に持つ Actor-Critic ネットワーク。"""

    def __init__(self, obs_dim_size_val, action_dim_size_val):
        super().__init__()
        self.obs_dim_size_val = obs_dim_size_val
        self.action_dim_size_val = action_dim_size_val

        # 埋め込み層
        self.embedding_layer_obj = nn.Linear(obs_dim_size_val, HIDDEN_DIM)

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
        self.actor_mean_output_node = layer_init(nn.Linear(HIDDEN_DIM, action_dim_size_val), std_val_init_ref=0.01)
        self.actor_logstd_params_vec = nn.Parameter(torch.zeros(1, action_dim_size_val))
        self.critic_value_output_node = layer_init(nn.Linear(HIDDEN_DIM, 1), std_val_init_ref=1.0)

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

    def __init__(self, envs_count_val, sequence_len_val, observation_dim_val, device_ptr_obj_ref):
        self.total_buffer_length_calculated = sequence_len_val * 2
        self.data_history_buffer_tensor = torch.zeros(
            (envs_count_val, self.total_buffer_length_calculated, observation_dim_val),
            device=device_ptr_obj_ref,
        )
        self.device_ptr_obj_ref = device_ptr_obj_ref
        self.sequence_len_val = sequence_len_val
        self.write_ptr_index_val = 0

    def reset(self):
        """バッファをゼロで初期化。"""
        self.data_history_buffer_tensor.zero_()
        self.write_ptr_index_val = 0

    def update(self, latest_obs_array_input):
        """最新観測値をミラーリング書き込み。"""
        new_obs_tensor_obj_ptr = torch.as_tensor(latest_obs_array_input, dtype=torch.float32, device=self.device_ptr_obj_ref)
        if new_obs_tensor_obj_ptr.ndim == 1:
            new_obs_tensor_obj_ptr = new_obs_tensor_obj_ptr.unsqueeze(0)

        # 本体領域とミラー領域
        self.data_history_buffer_tensor[:, self.write_ptr_index_val] = new_obs_tensor_obj_ptr
        mirror_index_location_idx = self.write_ptr_index_val + self.sequence_len_val
        self.data_history_buffer_tensor[:, mirror_index_location_idx] = new_obs_tensor_obj_ptr

        # ポインタ循環
        self.write_ptr_index_val = (self.write_ptr_index_val + 1) % self.sequence_len_val

    def get(self):
        """最新の系列スライスを View で取得。 ★修正: 属性参照の統一"""
        view_slice_end_ptr_idx = self.write_ptr_index_val + self.sequence_len_val
        result_view_slice_tensor_out = self.data_history_buffer_tensor[:, self.write_ptr_index_val : view_slice_end_ptr_idx]
        return result_view_slice_tensor_out


class TeamCosEnv(base_config.HideAndSeekEnv):
    """
    不整合を解消し、将来的なオブジェクト増設に対応した超高速化環境。
    """

    def __init__(self, render_mode=None):
        # ★ 属性初期化の順序厳守 (reset() 時の参照バグ防止)
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
        self.visible_obj_names_log_registry = {0: [], 1: [], 2: []}

        # 親クラスの初期化
        super().__init__(render_mode=render_mode)

        cpu_device = torch.device("cpu")
        # 各個体専用の履歴
        self.individual_npc_obs_history = {
            0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device),
            1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device),
            2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device),
        }

        # NPC 推論エンジン
        self.npc_hider_inference_model = Agent(53, 4).to("cpu")
        self.npc_seeker_inference_model = Agent(53, 4).to("cpu")

        is_manual_render_view_active = render_mode == "human"
        log_marker_id_status_str = os.environ.get("NPC_MODELS_LOGGED")
        should_display_initial_log_info = log_marker_id_status_str != "TRUE"

        if LOAD_EXISTING_MODELS or is_manual_render_view_active:
            h_path_str_val = load_model_safely(self.npc_hider_inference_model, EXPERIMENT_BASE_NAME, "HIDER")
            if h_path_str_val and should_display_initial_log_info:
                print(f"Loaded NPC Hider Engine for Env: {h_path_str_val}", flush=True)

            s_path_str_val = load_model_safely(self.npc_seeker_inference_model, EXPERIMENT_BASE_NAME, "SEEKER")
            if s_path_str_val and should_display_initial_log_info:
                print(f"Loaded NPC Seeker Engine for Env: {s_path_str_val}", flush=True)

        if should_display_initial_log_info:
            os.environ["NPC_MODELS_LOGGED"] = "TRUE"

    def _get_cached_ray(self, agent_idx, origin_pt_vec, direction_vec_raw, beam_id_val):
        """レイキャストの空間・角度キャッシュ制御。"""
        angle_rad_val = np.arctan2(direction_vec_raw[1], direction_vec_raw[0])
        cache_key_tuple_identifier = (agent_idx, beam_id_val)

        if cache_key_tuple_identifier in self.raycast_cache_storage:
            c_pos, c_ang, c_dist, c_geom = self.raycast_cache_storage[cache_key_tuple_identifier]
            pos_diff_dist_magnitude_val = np.linalg.norm(origin_pt_vec - c_pos)
            if pos_diff_dist_magnitude_val < RAYCAST_CACHE_POS_THRESH:
                angle_error_res_rad_val_ptr = (angle_rad_val - c_ang + np.pi) % (2.0 * np.pi) - np.pi
                if abs(angle_error_res_rad_val_ptr) < 0.05:
                    self.raycast_performance_metrics["hits"] += 1
                    return c_dist, c_geom

        # 実計測の実行
        self.raycast_performance_metrics["misses"] += 1
        hit_geom_id_out_ptr_buffer = np.zeros(1, dtype=np.int32)
        ray_from_3d_coordinates_vec = np.array([origin_pt_vec[0], origin_pt_vec[1], 0.5], dtype=np.float64)
        ray_dir_3d_coordinates_vec = np.array([direction_vec_raw[0], direction_vec_raw[1], 0.0], dtype=np.float64)

        # 除外ボディの設定
        if agent_idx == 0:
            body_to_exclude_idx = self.s0_body
        elif agent_idx == 1:
            body_to_exclude_idx = self.h1_body
        else:
            body_to_exclude_idx = self.h2_body

        mj_ray_actual_computed_dist_res = mujoco.mj_ray(
            self.model,
            self.data,
            ray_from_3d_coordinates_vec,
            ray_dir_3d_coordinates_vec,
            None,
            1,
            body_to_exclude_idx,
            hit_geom_id_out_ptr_buffer,
        )
        self.raycast_cache_storage[cache_key_tuple_identifier] = (
            origin_pt_vec.copy(),
            angle_rad_val,
            mj_ray_actual_computed_dist_res,
            hit_geom_id_out_ptr_buffer[0],
        )
        return mj_ray_actual_computed_dist_res, hit_geom_id_out_ptr_buffer[0]

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
        if agent_id in self._obs_memo_storage_buffer:
            return self._obs_memo_storage_buffer[agent_id]

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
        vis_lookup_record_dict_ref = self.visible_cache_records_map[agent_id]
        vis_lookup_record_dict_ref.clear()
        self.visible_obj_names_log_registry[agent_id] = []

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
                    self.visible_obj_names_log_registry[agent_id].append(name_str_ptr_ref)

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
        self._obs_memo_storage_buffer[agent_id] = obs_final_res_out_tensor_res
        return obs_final_res_out_tensor_res

    def _update_seeker_state(self):
        """鬼の思考ステートマシン。"""
        seeker_xy_now_coords_val_res = self.data.xpos[self.s0_body][:2]
        self._get_obs(0)
        h1_is_seen_v_bool_res = self.visible_cache_records_map[0].get(self.h1_body, False)
        h2_is_seen_v_bool_res = self.visible_cache_records_map[0].get(self.h2_body, False)

        if h1_is_seen_v_bool_res or h2_is_seen_v_bool_res:
            detected_h_target_bid_ptr = self.h1_body if h1_is_seen_v_bool_res else self.h2_body
            found_target_coords_vector_res = self.data.xpos[detected_h_target_bid_ptr][:2].copy()
            self.seeker_target_pos = found_target_coords_vector_res
            self.seeker_last_known_pos = found_target_coords_vector_res.copy()
            self.seeker_mode = "CHASING"
        elif self.seeker_last_known_pos is not None:
            displacement_mem_pos_magnitude = np.linalg.norm(seeker_xy_now_coords_val_res - self.seeker_last_known_pos)
            if displacement_mem_pos_magnitude > 0.5:
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

        sp_xy_vec_now = self.data.xpos[self.s0_body][:2]
        sr_rad_val_now = self.data.qpos[self.srot_adr]
        tp_xy_vec_target_pos = self.seeker_target_pos

        delta_x_val_res = tp_xy_vec_target_pos[0] - sp_xy_vec_now[0]
        delta_y_val_res = tp_xy_vec_target_pos[1] - sp_xy_vec_now[1]
        target_yaw_rad_computed_val_ptr = np.arctan2(delta_y_val_res, delta_x_val_res)
        angle_err_rad_computed_final = (target_yaw_rad_computed_val_ptr - sr_rad_val_now + np.pi) % (2.0 * np.pi) - np.pi

        final_thrust_applied_out = SEEKER_RB_THRUST
        final_turn_applied_out = np.clip(angle_err_rad_computed_final * 6.0, -3.0, 3.0)

        if abs(angle_err_rad_computed_final) > SEEKER_RB_TURN_THRESH:
            final_thrust_applied_out = final_thrust_applied_out * 0.3

        dof_adr_sx_joint_ptr_idx = self.model.jnt_dofadr[self.model.joint("s_x").id]
        velocity_current_magnitude_norm = np.linalg.norm(self.data.qvel[dof_adr_sx_joint_ptr_idx : dof_adr_sx_joint_ptr_idx + 2])

        if final_thrust_applied_out > 0.05 and velocity_current_magnitude_norm < 0.05:
            self.s0_stuck_timer = self.s0_stuck_timer + 5
        else:
            self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)

        if self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 15
            self.s0_stuck_timer = 0
            self.recovery_turn_direction_modifier = self.np_random.choice([-1.0, 1.0])

        if self.s0_recovery_mode > 0:
            final_thrust_applied_out = -0.2
            final_turn_applied_out = 1.5 * self.recovery_turn_direction_modifier
            self.s0_recovery_mode = self.s0_recovery_mode - 1

        return float(final_thrust_applied_out), float(final_turn_applied_out)

    def _get_npc_action(self, agent_id_idx_val_ref, agent_type_label_str_ref):
        """NPC 行動生成。"""
        observation_raw_array_data_vec = self._get_obs(agent_id_idx_val_ref)
        self.individual_npc_obs_history[agent_id_idx_val_ref].update(observation_raw_array_data_vec)

        model_ptr_obj_reference_val = self.npc_hider_inference_model if agent_type_label_str_ref == "HIDER" else self.npc_seeker_inference_model

        if model_ptr_obj_reference_val is not None:
            with torch.no_grad():
                ctx_sequence_data_tensor_ref = self.individual_npc_obs_history[agent_id_idx_val_ref].get()
                act_tensor_out_res_final_val, _, _, _ = model_ptr_obj_reference_val.get_action_and_value(ctx_sequence_data_tensor_ref)
            return act_tensor_out_res_final_val.cpu().numpy()[0]

        if agent_type_label_str_ref == "SEEKER":
            f_rb_p_res_final, r_rb_p_res_final = self._seeker_rule_based_policy()
            normalized_thrust_applied_val = f_rb_p_res_final / SEEKER_THRUST_LIMIT
            return np.array(
                [normalized_thrust_applied_val, r_rb_p_res_final, 0.0, 0.0],
                dtype=np.float32,
            )

        return self.action_space.sample() * 0.5

    def reset(self, seed=None, options=None):
        """環境リセット。 ★修正: 変数名の完全統一"""
        obs_initial_val_ptr, info_initial_val_ptr = super().reset(seed=seed, options=options)
        self.hidden_steps_accumulator = 0
        self.caught_steps_accumulator = 0
        self._obs_memo_storage_buffer.clear()
        self.lidar_array_cache_storage.clear()
        self.raycast_cache_storage.clear()
        self.recovery_turn_direction_modifier = 1.0

        seeker_xy_init_position_vector_val = self.data.xpos[self.s0_body][:2]
        for hi_idx_ptr_val_iter in [1, 2]:
            body_pointer_idx_walker_ref = self.h1_body if hi_idx_ptr_val_iter == 1 else self.h2_body
            xy_pos_pointer_walker_ref_ptr = self.data.xpos[body_pointer_idx_walker_ref][:2].copy()
            # 名称統一
            self.previous_distances_to_seeker_map[hi_idx_ptr_val_iter] = np.linalg.norm(xy_pos_pointer_walker_ref_ptr - seeker_xy_init_position_vector_val)
            self.previous_hider_xy_map[hi_idx_ptr_val_iter] = xy_pos_pointer_walker_ref_ptr

        return obs_initial_val_ptr, info_initial_val_ptr

    def step(self, action_vector_ptr_input_val):
        """1ステップ進展。 ★修正: 集計変数の完全統一"""
        self.current_step = self.current_step + 1
        for i_cooldown_ptr_idx_val in [1, 2]:
            self.lock_cooldown[i_cooldown_ptr_idx_val] = max(0, self.lock_cooldown[i_cooldown_ptr_idx_val] - 1)
        self._update_seeker_state()
        self.data.ctrl[:] = 0.0

        if TRAIN_TARGET == "HIDER":
            main_h_idx_val_ptr_res = self._apply_action(self.learning_agent_id, action_vector_ptr_input_val)
            self.data.ctrl[main_h_idx_val_ptr_res] = float(action_vector_ptr_input_val[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[main_h_idx_val_ptr_res + 1] = float(action_vector_ptr_input_val[1])
            partner_h_id_val_ref_ptr = 2 if self.learning_agent_id == 1 else 1
            partner_h_npc_action_result_out = self._get_npc_action(partner_h_id_val_ref_ptr, "HIDER")
            p_h_idx_val_ptr_res = self._apply_action(partner_h_id_val_ref_ptr, partner_h_npc_action_result_out)
            self.data.ctrl[p_h_idx_val_ptr_res] = float(partner_h_npc_action_result_out[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[p_h_idx_val_ptr_res + 1] = float(partner_h_npc_action_result_out[1])
            act_seeker_npc_final_agent_val = self._get_npc_action(0, "SEEKER")
            self.data.ctrl[0] = float(act_seeker_npc_final_agent_val[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(act_seeker_npc_final_agent_val[1])
        else:
            self.data.ctrl[0] = float(action_vector_ptr_input_val[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(action_vector_ptr_input_val[1])
            for i_h_npc_idx_ptr_ref_idx in [1, 2]:
                act_hider_npc_individual_val_res = self._get_npc_action(i_h_npc_idx_ptr_ref_idx, "HIDER")
                h_h_idx_ptr_walker_ref_ptr = self._apply_action(i_h_npc_idx_ptr_ref_idx, act_hider_npc_individual_val_res)
                self.data.ctrl[h_h_idx_ptr_walker_ref_ptr] = float(act_hider_npc_individual_val_res[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[h_h_idx_ptr_walker_ref_ptr + 1] = float(act_hider_npc_individual_val_res[1])

        # --- PHYSICS LOOP (Slice-based) ---
        for _ in range(ACTION_REPEAT):
            for bid_loop_id_ptr_val, pose_loop_data_ptr_val in self.locked_pose.items():
                if self.locked_boxes[bid_loop_id_ptr_val]:
                    bjid_v_loop_ptr_walker_ref_idx = self.box1_joint_id if bid_loop_id_ptr_val == self.box1_body else self.box2_joint_id
                    qa_ptr_loop_idx_val_ptr = self.model.jnt_qposadr[bjid_v_loop_ptr_walker_ref_idx]
                    da_ptr_loop_idx_val_ptr = self.model.jnt_dofadr[bjid_v_loop_ptr_walker_ref_idx]
                    self.data.qpos[qa_ptr_loop_idx_val_ptr : qa_ptr_loop_idx_val_ptr + 7] = pose_loop_data_ptr_val
                    self.data.qvel[da_ptr_loop_idx_val_ptr : da_ptr_loop_idx_val_ptr + 6] = 0.0
            mujoco.mj_step(self.model, self.data)

        self._obs_memo_storage_buffer.clear()
        # 観測確定
        obs_learner_step_final_ptr_val = self._get_obs(self.learning_agent_id)
        _ = self._get_obs(0)  # 統計用

        if self.render_mode == "human":
            _ = self._get_obs(1)
            _ = self._get_obs(2)

        # 統計用にキャッシュ参照
        vis_h1_bool_flag_res = self.visible_cache_records_map[0].get(self.h1_body, False)
        vis_h2_bool_flag_res = self.visible_cache_records_map[0].get(self.h2_body, False)

        if vis_h1_bool_flag_res or vis_h2_bool_flag_res:
            self.caught_steps_accumulator = self.caught_steps_accumulator + 1
        else:
            self.hidden_steps_accumulator = self.hidden_steps_accumulator + 1

        # ★修正: 名称統一 (total_team_reward_accumulator)
        total_team_reward_accumulator = 0.0
        for h_idx_ptr_iter_idx_val_ptr, bid_ptr_val_iter_ptr_ref_ptr in [
            (1, self.h1_body),
            (2, self.h2_body),
        ]:
            is_seen_v_bool_val_now_ptr = self.visible_cache_records_map[0].get(bid_ptr_val_iter_ptr_ref_ptr, False)
            sp_xy_iter_coords_vec_val_out = self.data.xpos[self.s0_body][:2]
            hp_xy_iter_coords_vec_val_out = self.data.xpos[bid_ptr_val_iter_ptr_ref_ptr][:2]
            current_dist_norm_magnitude_val_out = np.linalg.norm(hp_xy_iter_coords_vec_val_out - sp_xy_iter_coords_vec_val_out)

            if is_seen_v_bool_val_now_ptr:
                sr_rad_iter_val_now_ptr_out = self.data.qpos[self.srot_adr]
                dv_iter_vector_raw_ptr_out = hp_xy_iter_coords_vec_val_out - sp_xy_iter_coords_vec_val_out
                dn_iter_vector_norm_ptr_out = dv_iter_vector_raw_ptr_out / (np.linalg.norm(dv_iter_vector_raw_ptr_out) + 1e-8)
                fv_iter_forward_vector_ptr_out = np.array(
                    [
                        np.cos(sr_rad_iter_val_now_ptr_out),
                        np.sin(sr_rad_iter_val_now_ptr_out),
                    ]
                )
                cosine_value_computed_ptr_out = np.dot(dn_iter_vector_norm_ptr_out, fv_iter_forward_vector_ptr_out)
                individual_reward_piece_val_out = -cosine_value_computed_ptr_out * COS_PENALTY_SCALE
                dist_delta_val_ptr_res_final = current_dist_norm_magnitude_val_out - self.previous_distances_to_seeker_map[h_idx_ptr_iter_idx_val_ptr]
                individual_reward_piece_val_out = individual_reward_piece_val_out + dist_delta_val_ptr_res_final * REWARD_DISTANCE_DIFF_SCALE
            else:
                individual_reward_piece_val_out = REWARD_HIDDEN_BONUS

            if h_idx_ptr_iter_idx_val_ptr == self.learning_agent_id:
                if self.previous_hider_xy_map[h_idx_ptr_iter_idx_val_ptr] is not None:
                    displacement_val_norm_val_res_final = np.linalg.norm(hp_xy_iter_coords_vec_val_out - self.previous_hider_xy_map[h_idx_ptr_iter_idx_val_ptr])
                    if displacement_val_norm_val_res_final < 0.01:
                        individual_reward_piece_val_out = individual_reward_piece_val_out + PENALTY_STAGNATION
                self.previous_hider_xy_map[h_idx_ptr_iter_idx_val_ptr] = hp_xy_iter_coords_vec_val_out.copy()

            if max(abs(hp_xy_iter_coords_vec_val_out)) > 6.5:
                individual_reward_piece_val_out = individual_reward_piece_val_out + PENALTY_SAFEGUARD

            total_team_reward_accumulator = total_team_reward_accumulator + individual_reward_piece_val_out
            self.previous_distances_to_seeker_map[h_idx_ptr_iter_idx_val_ptr] = current_dist_norm_magnitude_val_out

        final_reward_step_result = total_team_reward_accumulator if TRAIN_TARGET == "HIDER" else -total_team_reward_accumulator
        truncated_flag_status_ptr_out = self.current_step >= MAX_STEPS

        info_step_stats_payload_vector_res_final = {
            "hidden_steps": float(self.hidden_steps_accumulator),
            "caught_steps": float(self.caught_steps_accumulator),
        }
        return (
            obs_learner_step_final_ptr_val,
            float(final_reward_step_result),
            False,
            truncated_flag_status_ptr_out,
            info_step_stats_payload_vector_res_final,
        )

    def render(self, stats=None):
        """MuJoCo Viewer 描画。"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.elevation, self.viewer.cam.distance = -60, 23.0

            for b_ptr_r_walker_ref_ptr, g_ptr_r_walker_ref_ptr in [
                (self.box1_body, self.box1_geom_id),
                (self.box2_body, self.box2_geom_id),
            ]:
                if self.locked_boxes[b_ptr_r_walker_ref_ptr]:
                    self.model.geom_rgba[g_ptr_r_walker_ref_ptr][:] = [
                        0.8,
                        0.1,
                        0.1,
                        1.0,
                    ]
                elif any(g_ptr_v_ptr_ref_ptr == b_ptr_r_walker_ref_ptr for g_ptr_v_ptr_ref_ptr in self.grasping.values()):
                    self.model.geom_rgba[g_ptr_r_walker_ref_ptr][:] = [
                        0.1,
                        0.1,
                        0.9,
                        1.0,
                    ]
                else:
                    orig_rgba_vec_val_res_final = [0.6, 0.4, 0.2, 1.0] if b_ptr_r_walker_ref_ptr == self.box1_body else [0.7, 0.5, 0.3, 1.0]
                    self.model.geom_rgba[g_ptr_r_walker_ref_ptr][:] = orig_rgba_vec_val_res_final

            if self.viewer.user_scn:
                ctx_scn_ptr_walker_ref_ptr = self.viewer.user_scn
                ctx_scn_ptr_walker_ref_ptr.ngeom = 0

                def add_line_internal_func_final(p1_vec_v_in_ref_ptr, p2_vec_v_in_ref_ptr, rgba_vec_v_in_ref_ptr):
                    if ctx_scn_ptr_walker_ref_ptr.ngeom < ctx_scn_ptr_walker_ref_ptr.maxgeom:
                        mujoco.mjv_initGeom(
                            ctx_scn_ptr_walker_ref_ptr.geoms[ctx_scn_ptr_walker_ref_ptr.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            size=np.array([0, 0, 0]),
                            pos=np.array([0, 0, 0]),
                            mat=np.eye(3).flatten(),
                            rgba=rgba_vec_v_in_ref_ptr,
                        )
                        mujoco.mjv_connector(
                            ctx_scn_ptr_walker_ref_ptr.geoms[ctx_scn_ptr_walker_ref_ptr.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            width=2.0,
                            from_=p1_vec_v_in_ref_ptr,
                            to=p2_vec_v_in_ref_ptr,
                        )
                        ctx_scn_ptr_walker_ref_ptr.ngeom = ctx_scn_ptr_walker_ref_ptr.ngeom + 1

                def add_label_internal_func_final(p_vec_v_in_ref_ptr, txt_str_v_in_ref_ptr, rgba_vec_v_in_ref_ptr):
                    if ctx_scn_ptr_walker_ref_ptr.ngeom < ctx_scn_ptr_walker_ref_ptr.maxgeom:
                        mujoco.mjv_initGeom(
                            ctx_scn_ptr_walker_ref_ptr.geoms[ctx_scn_ptr_walker_ref_ptr.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LABEL,
                            size=np.array([0, 0, 0]),
                            pos=p_vec_v_in_ref_ptr,
                            mat=np.eye(3).flatten(),
                            rgba=rgba_vec_v_in_ref_ptr,
                        )
                        ctx_scn_ptr_walker_ref_ptr.geoms[ctx_scn_ptr_walker_ref_ptr.ngeom].label = txt_str_v_in_ref_ptr
                        ctx_scn_ptr_walker_ref_ptr.ngeom = ctx_scn_ptr_walker_ref_ptr.ngeom + 1

                # H1 (Yellow)
                p1_c_v_final = self.data.xpos[self.h1_body]
                for tid_r_v_iter_ptr_ref in [
                    self.box1_body,
                    self.box2_body,
                    self.ramp_body,
                    self.s0_body,
                    self.h2_body,
                ]:
                    if tid_r_v_iter_ptr_ref == self.h1_body:
                        continue
                    if self.visible_cache_records_map[1].get(tid_r_v_iter_ptr_ref, False):
                        add_line_internal_func_final(
                            p1_c_v_final + [0, 0, 0.5],
                            self.data.xpos[tid_r_v_iter_ptr_ref] + [0, 0, 0.5],
                            [1, 1, 0, 0.4],
                        )
                add_label_internal_func_final(
                    self.data.site_xpos[self.id_h1_label],
                    f"H1 Vis:[{','.join(self.visible_obj_names_log_registry[1])}]",
                    [1, 1, 0, 1],
                )

                # H2 (Cyan)
                p2_c_v_final = self.data.xpos[self.h2_body]
                for tid_r_v_iter_ptr_ref in [
                    self.box1_body,
                    self.box2_body,
                    self.ramp_body,
                    self.s0_body,
                    self.h1_body,
                ]:
                    if tid_r_v_iter_ptr_ref == self.h2_body:
                        continue
                    if self.visible_cache_records_map[2].get(tid_r_v_iter_ptr_ref, False):
                        add_line_internal_func_final(
                            p2_c_v_final + [0, 0, 0.5],
                            self.data.xpos[tid_r_v_iter_ptr_ref] + [0, 0, 0.5],
                            [0, 1, 1, 0.4],
                        )
                add_label_internal_func_final(
                    self.data.site_xpos[self.id_h2_label],
                    f"H2 Vis:[{','.join(self.visible_obj_names_log_registry[2])}]",
                    [0, 1, 1, 1],
                )

                # Seeker (Red)
                ps_c_v_final = self.data.xpos[self.s0_body]
                for tid_r_v_iter_ptr_ref in [self.h1_body, self.h2_body]:
                    if self.visible_cache_records_map[0].get(tid_r_v_iter_ptr_ref, False):
                        add_line_internal_func_final(
                            ps_c_v_final + [0, 0, 0.5],
                            self.data.xpos[tid_r_v_iter_ptr_ref] + [0, 0, 0.5],
                            [1, 0, 0, 0.6],
                        )
                add_label_internal_func_final(
                    self.data.site_xpos[self.id_s_label],
                    f"S:{self.seeker_mode}",
                    [1, 0, 0, 1],
                )

            self.viewer.sync()


# ==========================================
# 3. ヘルパー関数 & ファクトリ
# ==========================================


def load_model_safely(model_obj_ptr_val_ref_ptr, base_name_val_str_ref_ptr, target_type_val_str_ref_ptr):
    """指定候補からモデルをロード。"""
    search_paths_array_list_ptr_ref_ptr = [
        f"{base_name_val_str_ref_ptr}_refinement_{target_type_val_str_ref_ptr}.pt",
        f"{base_name_val_str_ref_ptr}_initial_{target_type_val_str_ref_ptr}.pt",
        f"{base_name_val_str_ref_ptr}_{target_type_val_str_ref_ptr}.pt",
    ]
    for path_str_val_ptr_ref_ptr in search_paths_array_list_ptr_ref_ptr:
        if os.path.exists(path_str_val_ptr_ref_ptr):
            try:
                state_dict_loaded_v_ref_ptr = torch.load(path_str_val_ptr_ref_ptr, map_location="cpu")
                model_obj_ptr_val_ref_ptr.load_state_dict(state_dict_loaded_v_ref_ptr)
                model_obj_ptr_val_ref_ptr.eval()
                return path_str_val_ptr_ref_ptr
            except Exception:
                continue
    return None


def env_factory_parallel_executor_ptr_ref_ptr():
    """AsyncVectorEnv 用。"""
    env_instance_created_ptr_ref_ptr = TeamCosEnv()
    return gym.wrappers.RecordEpisodeStatistics(env_instance_created_ptr_ref_ptr)


# ==========================================
# 5. メイン処理 (学習ループ)
# ==========================================


def main():
    if platform.system() == "Linux":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    import gymnasium as gym

    print(f"--- [Parent] 1. Initializing {NUM_ENVS} workers ---", flush=True)
    try:
        vec_envs_ptr_obj_ref_final_ptr = gym.vector.AsyncVectorEnv([env_factory_parallel_executor_ptr_ref_ptr for _ in range(NUM_ENVS)])
        print("--- [Parent] 2. Parallel environment ready ---", flush=True)
    except Exception as e_parent_startup_ptr_ref_ptr:
        print(
            f"--- [Parent] startup failed: {e_parent_startup_ptr_ref_ptr} ---",
            flush=True,
        )
        sys.exit(1)

    import torch.optim as optim

    import wandb

    device_ptr_final_hardware_ref_final_ptr = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
    run_epoch_timestamp_int_val_ptr = int(time.time())
    unique_exp_run_unique_id_str_final_ptr = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{run_epoch_timestamp_int_val_ptr}"

    if EXECUTION_MODE == "PLAY":
        print("--- Inference Mode (PLAY) ---")
        env_play_v_ptr_obj_ref_ptr = TeamCosEnv(render_mode="human")
        agent_p_net_obj_res_ref_ptr = Agent(
            env_play_v_ptr_obj_ref_ptr.observation_space.shape[0],
            env_play_v_ptr_obj_ref_ptr.action_space.shape[0],
        ).to(device_ptr_final_hardware_ref_final_ptr)
        load_model_safely(agent_p_net_obj_res_ref_ptr, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        agent_p_net_obj_res_ref_ptr.eval()
        hist_p_manager_ptr_obj_ref_ptr = ObsHistory(
            1,
            TRANSFORMER_SEQ_LEN,
            env_play_v_ptr_obj_ref_ptr.observation_space.shape[0],
            device_ptr_final_hardware_ref_final_ptr,
        )
        try:
            while True:
                obs_play_v_iter_res_ref_ptr, _ = env_play_v_ptr_obj_ref_ptr.reset()
                hist_p_manager_ptr_obj_ref_ptr.reset()
                hist_p_manager_ptr_obj_ref_ptr.update(obs_play_v_iter_res_ref_ptr)
                done_play_v_iter_res_ref_ptr = False
                ret_play_v_accum_res_ref_ptr = 0.0
                while not done_play_v_iter_res_ref_ptr:
                    t_loop_start_val_res_ref_ptr = time.time()
                    with torch.no_grad():
                        action_play_v_result_ref_ptr, _, _, _ = agent_p_net_obj_res_ref_ptr.get_action_and_value(hist_p_manager_ptr_obj_ref_ptr.get())
                    (
                        o_next_p_step_ptr_res_ref_ptr,
                        r_p_step_ptr_res_ref_ptr,
                        term_p_step_ptr_res_ref_ptr,
                        trunc_p_step_ptr_res_ref_ptr,
                        i_p_step_ptr_res_ref_ptr,
                    ) = env_play_v_ptr_obj_ref_ptr.step(action_play_v_result_ref_ptr.cpu().numpy()[0])
                    done_play_v_iter_res_ref_ptr = term_p_step_ptr_res_ref_ptr or trunc_p_step_ptr_res_ref_ptr
                    ret_play_v_accum_res_ref_ptr = ret_play_v_accum_res_ref_ptr + r_p_step_ptr_res_ref_ptr
                    hist_p_manager_ptr_obj_ref_ptr.update(o_next_p_step_ptr_res_ref_ptr)
                    env_play_v_ptr_obj_ref_ptr.render(stats={"EpRet": f"{ret_play_v_accum_res_ref_ptr:.1f}"})
                    wait_sec_val_res_ref_ptr = (0.005 * ACTION_REPEAT) - (time.time() - t_loop_start_val_res_ref_ptr)
                    if wait_sec_val_res_ref_ptr > 0:
                        time.sleep(wait_sec_val_res_ref_ptr)
                print(
                    f"Result -> Return: {ret_play_v_accum_res_ref_ptr:.1f}, Hidden: {i_p_step_ptr_res_ref_ptr['hidden_steps']:.0f}",
                    flush=True,
                )
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass
        finally:
            env_play_v_ptr_obj_ref_ptr.close()
        return

    if TRACK_WANDB:
        wandb.init(
            project=base_config.WANDB_PROJECT_NAME,
            config={
                "Target": TRAIN_TARGET,
                "MODE": MODE,
                "v": "25.62_StandardizedNames",
            },
            name=unique_exp_run_unique_id_str_final_ptr,
            sync_tensorboard=False,
            save_code=True,
        )

    tensorboard_writer_ptr_instance_ref_ptr = SummaryWriter(f"runs/{unique_exp_run_unique_id_str_final_ptr}")
    agent_train_net_ptr_instance_ref_ptr = Agent(
        vec_envs_ptr_obj_ref_final_ptr.single_observation_space.shape[0],
        vec_envs_ptr_obj_ref_final_ptr.single_action_space.shape[0],
    ).to(device_ptr_final_hardware_ref_final_ptr)
    optimizer_train_ptr_executor_ref_ptr = optim.Adam(agent_train_net_ptr_instance_ref_ptr.parameters(), lr=LEARNING_RATE, eps=1e-5)

    global_s_index_total_val_ref_ptr = 0
    start_s_index_val_ptr_res_ref_ptr = 0
    if LOAD_EXISTING_MODELS:
        model_file_path_ptr_res_ref_ptr = load_model_safely(agent_train_net_ptr_instance_ref_ptr, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if model_file_path_ptr_res_ref_ptr:
            print(f"★ Resumed learner from: {model_file_path_ptr_res_ref_ptr}")
            checkpoint_json_ptr_path_res_ref_ptr = model_file_path_ptr_res_ref_ptr.replace(".pt", "_checkpoint.json")
            if os.path.exists(checkpoint_json_ptr_path_res_ref_ptr):
                try:
                    with open(checkpoint_json_ptr_path_res_ref_ptr, "r") as f_ptr_json_file_res_ref_ptr:
                        checkpoint_payload_data_ref_ptr = json.load(f_ptr_json_file_res_ref_ptr)
                        global_s_index_total_val_ref_ptr = checkpoint_payload_data_ref_ptr.get("global_step", 0)
                        start_s_index_val_ptr_res_ref_ptr = global_s_index_total_val_ref_ptr
                except:
                    pass

    # --- ROLLOUT BUFFER ---
    rollout_history_buffer_logic_ref_ptr = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device_ptr_final_hardware_ref_final_ptr)
    S_r_ref_ptr, E_r_ref_ptr, O_r_ref_ptr, A_r_ref_ptr = NUM_STEPS, NUM_ENVS, 53, 4
    batch_obs_storage_final_tensor_ref_ptr = torch.zeros(
        (S_r_ref_ptr, E_r_ref_ptr, TRANSFORMER_SEQ_LEN, O_r_ref_ptr),
        device=device_ptr_final_hardware_ref_final_ptr,
    )
    batch_actions_storage_final_tensor_ref_ptr = torch.zeros(
        (S_r_ref_ptr, E_r_ref_ptr, A_r_ref_ptr),
        device=device_ptr_final_hardware_ref_final_ptr,
    )
    batch_logprobs_storage_final_tensor_ref_ptr = torch.zeros((S_r_ref_ptr, E_r_ref_ptr), device=device_ptr_final_hardware_ref_final_ptr)
    batch_rewards_storage_final_tensor_ref_ptr = torch.zeros((S_r_ref_ptr, E_r_ref_ptr), device=device_ptr_final_hardware_ref_final_ptr)
    batch_dones_storage_final_tensor_ref_ptr = torch.zeros((S_r_ref_ptr, E_r_ref_ptr), device=device_ptr_final_hardware_ref_final_ptr)
    batch_values_storage_final_tensor_ref_ptr = torch.zeros((S_r_ref_ptr, E_r_ref_ptr), device=device_ptr_final_hardware_ref_final_ptr)

    next_obs_array_start_ptr_ref_ptr = vec_envs_ptr_obj_ref_final_ptr.reset(seed=FIXED_SEED if FIXED_SEED else int(time.time()))[0]
    next_done_tensor_flag_ptr_res_ref_ptr = torch.zeros(E_r_ref_ptr).to(device_ptr_final_hardware_ref_final_ptr)
    rollout_history_buffer_logic_ref_ptr.reset()
    rollout_history_buffer_logic_ref_ptr.update(next_obs_array_start_ptr_ref_ptr)

    updates_total_needed_count_res_ref_ptr = int(
        max(
            1,
            (TOTAL_TIMESTEPS - global_s_index_total_val_ref_ptr) // (E_r_ref_ptr * S_r_ref_ptr),
        )
    )
    history_returns_list_accumulator_ref_ptr = []
    history_hidden_list_accumulator_ref_ptr = []
    history_caught_list_accumulator_ref_ptr = []
    start_wall_clock_time_val_res_ref_ptr = time.time()
    last_update_loss_result_val_ref_ptr = 0.0
    last_update_entropy_result_val_ref_ptr = 0.0

    print("--- Training Sequence Started (v25.62) ---")
    try:
        for u_idx_iter_val_ref_ptr in tqdm(range(1, updates_total_needed_count_res_ref_ptr + 1), desc="Updates"):
            for step_idx_iter_rollout_ref_ptr in range(S_r_ref_ptr):
                global_s_index_total_val_ref_ptr = global_s_index_total_val_ref_ptr + E_r_ref_ptr
                batch_obs_storage_final_tensor_ref_ptr[step_idx_iter_rollout_ref_ptr] = rollout_history_buffer_logic_ref_ptr.get()
                batch_dones_storage_final_tensor_ref_ptr[step_idx_iter_rollout_ref_ptr] = next_done_tensor_flag_ptr_res_ref_ptr

                with torch.no_grad():
                    (
                        action_out_v_ptr_res_ref_ptr,
                        lp_out_v_ptr_res_ref_ptr,
                        _,
                        value_out_v_ptr_res_ref_ptr,
                    ) = agent_train_net_ptr_instance_ref_ptr.get_action_and_value(rollout_history_buffer_logic_ref_ptr.get())
                    batch_values_storage_final_tensor_ref_ptr[step_idx_iter_rollout_ref_ptr] = value_out_v_ptr_res_ref_ptr.flatten()

                batch_actions_storage_final_tensor_ref_ptr[step_idx_iter_rollout_ref_ptr] = action_out_v_ptr_res_ref_ptr
                batch_logprobs_storage_final_tensor_ref_ptr[step_idx_iter_rollout_ref_ptr] = lp_out_v_ptr_res_ref_ptr

                (
                    next_obs_v_step_array_res_ref_ptr,
                    reward_v_step_array_res_ref_ptr,
                    term_v_step_array_res_ref_ptr,
                    trunc_v_step_array_res_ref_ptr,
                    info_v_step_dict_res_ref_ptr,
                ) = vec_envs_ptr_obj_ref_final_ptr.step(action_out_v_ptr_res_ref_ptr.cpu().numpy())
                done_mask_v_step_final_ref_ptr = np.logical_or(term_v_step_array_res_ref_ptr, trunc_v_step_array_res_ref_ptr)

                if "final_info" in info_v_step_dict_res_ref_ptr:
                    for e_idx_iter_ptr_final_ref_ptr in range(E_r_ref_ptr):
                        individual_info_ptr_final_ref_ptr = info_v_step_dict_res_ref_ptr["final_info"][e_idx_iter_ptr_final_ref_ptr]
                        if done_mask_v_step_final_ref_ptr[e_idx_iter_ptr_final_ref_ptr] and individual_info_ptr_final_ref_ptr is not None:
                            if "episode" in individual_info_ptr_final_ref_ptr:
                                history_returns_list_accumulator_ref_ptr.append(float(individual_info_ptr_final_ref_ptr["episode"]["r"]))
                            if "hidden_steps" in individual_info_ptr_final_ref_ptr:
                                history_hidden_list_accumulator_ref_ptr.append(float(individual_info_ptr_final_ref_ptr["hidden_steps"]))
                            if "caught_steps" in individual_info_ptr_final_ref_ptr:
                                history_caught_list_accumulator_ref_ptr.append(float(individual_info_ptr_final_ref_ptr["caught_steps"]))
                elif "episode" in info_v_step_dict_res_ref_ptr:
                    e_mask_array_val_final_ref_ptr = info_v_step_dict_res_ref_ptr.get("_episode", [True] * E_r_ref_ptr)
                    for e_idx_iter_ptr_final_ref_ptr in range(E_r_ref_ptr):
                        if e_mask_array_val_final_ref_ptr[e_idx_iter_ptr_final_ref_ptr] and done_mask_v_step_final_ref_ptr[e_idx_iter_ptr_final_ref_ptr]:
                            history_returns_list_accumulator_ref_ptr.append(float(info_v_step_dict_res_ref_ptr["episode"]["r"][e_idx_iter_ptr_final_ref_ptr]))
                            if "hidden_steps" in info_v_step_dict_res_ref_ptr:
                                history_hidden_list_accumulator_ref_ptr.append(float(info_v_step_dict_res_ref_ptr["hidden_steps"][e_idx_iter_ptr_final_ref_ptr]))
                            if "caught_steps" in info_v_step_dict_res_ref_ptr:
                                history_caught_list_accumulator_ref_ptr.append(float(info_v_step_dict_res_ref_ptr["caught_steps"][e_idx_iter_ptr_final_ref_ptr]))

                batch_rewards_storage_final_tensor_ref_ptr[step_idx_iter_rollout_ref_ptr] = torch.tensor(reward_v_step_array_res_ref_ptr).to(device_ptr_final_hardware_ref_final_ptr).view(-1)
                next_done_tensor_flag_ptr_res_ref_ptr = torch.tensor(done_mask_v_step_final_ref_ptr).to(device_ptr_final_hardware_ref_final_ptr, dtype=torch.float32)
                rollout_history_buffer_logic_ref_ptr.update(next_obs_v_step_array_res_ref_ptr)

            # --- PPO UPDATE ---
            with torch.no_grad():
                v_next_p_val_ptr_ref_final_res_ptr = agent_train_net_ptr_instance_ref_ptr.get_value(rollout_history_buffer_logic_ref_ptr.get()).reshape(1, -1)
                advantages_tensor_step_final_ptr_res_ptr = torch.zeros_like(batch_rewards_storage_final_tensor_ref_ptr).to(device_ptr_final_hardware_ref_final_ptr)
                gae_accum_val_ptr_res_ptr = 0
                for t_idx_ptr_v_rollout_ref_ptr in reversed(range(S_r_ref_ptr)):
                    if t_idx_ptr_v_rollout_ref_ptr == S_r_ref_ptr - 1:
                        nt_v_val_ref_final_res_ptr = 1.0 - next_done_tensor_flag_ptr_res_ref_ptr
                        vp_v_val_ref_final_res_ptr = v_next_p_val_ptr_ref_final_res_ptr
                    else:
                        nt_v_val_ref_final_res_ptr = 1.0 - batch_dones_storage_final_tensor_ref_ptr[t_idx_ptr_v_rollout_ref_ptr + 1]
                        vp_v_val_ref_final_res_ptr = batch_values_storage_final_tensor_ref_ptr[t_idx_ptr_v_rollout_ref_ptr + 1]
                    delta_v_calc_res_final_res_ptr = (
                        batch_rewards_storage_final_tensor_ref_ptr[t_idx_ptr_v_rollout_ref_ptr]
                        + 0.99 * vp_v_val_ref_final_res_ptr * nt_v_val_ref_final_res_ptr
                        - batch_values_storage_final_tensor_ref_ptr[t_idx_ptr_v_rollout_ref_ptr]
                    )
                    gae_accum_val_ptr_res_ptr = delta_v_calc_res_final_res_ptr + 0.99 * 0.95 * nt_v_val_ref_final_res_ptr * gae_accum_val_ptr_res_ptr
                    advantages_tensor_step_final_ptr_res_ptr[t_idx_ptr_v_rollout_ref_ptr] = gae_accum_val_ptr_res_ptr
                returns_tensor_step_final_ptr_res_ptr = advantages_tensor_step_final_ptr_res_ptr + batch_values_storage_final_tensor_ref_ptr

            f_obs_unrolled_res_ref_ptr = batch_obs_storage_final_tensor_ref_ptr.reshape((-1, TRANSFORMER_SEQ_LEN, 53))
            f_log_unrolled_res_ref_ptr = batch_logprobs_storage_final_tensor_ref_ptr.reshape(-1)
            f_act_unrolled_res_ref_ptr = batch_actions_storage_final_tensor_ref_ptr.reshape((-1, 4))
            f_adv_unrolled_res_ref_ptr = advantages_tensor_step_final_ptr_res_ptr.reshape(-1)
            f_ret_unrolled_res_ref_ptr = returns_tensor_step_final_ptr_res_ptr.reshape(-1)
            f_val_unrolled_res_ref_ptr = batch_values_storage_final_tensor_ref_ptr.reshape(-1)

            # PPO OPTIMIZATION
            for ep_inner_iter_res_ref_ptr in range(UPDATE_EPOCHS):
                idx_shuffle_ptr_v_step_final_res_ptr = np.arange(S_r_ref_ptr * E_r_ref_ptr)
                np.random.shuffle(idx_shuffle_ptr_v_step_final_res_ptr)
                for start_ptr_idx_rollout_ref_ptr in range(0, S_r_ref_ptr * E_r_ref_ptr, MINIBATCH_SIZE):
                    mb_idx_ptr_unrolled_res_ref_ptr = idx_shuffle_ptr_v_step_final_res_ptr[start_ptr_idx_rollout_ref_ptr : start_ptr_idx_rollout_ref_ptr + MINIBATCH_SIZE]
                    (
                        _,
                        n_lp_v_unroll_res_ref_ptr,
                        e_v_v_unroll_res_ref_ptr,
                        n_v_val_v_unroll_res_ref_ptr,
                    ) = agent_train_net_ptr_instance_ref_ptr.get_action_and_value(
                        f_obs_unrolled_res_ref_ptr[mb_idx_ptr_unrolled_res_ref_ptr],
                        f_act_unrolled_res_ref_ptr[mb_idx_ptr_unrolled_res_ref_ptr],
                    )

                    log_ratio_v_unroll_res_ref_ptr = n_lp_v_unroll_res_ref_ptr - f_log_unrolled_res_ref_ptr[mb_idx_ptr_unrolled_res_ref_ptr]
                    ratio_ptr_v_unroll_res_ref_ptr = log_ratio_v_unroll_res_ref_ptr.exp()

                    with torch.no_grad():
                        approx_kl_v_ptr_ref_ptr = ((ratio_ptr_v_unroll_res_ref_ptr - 1.0) - log_ratio_v_unroll_res_ref_ptr).mean()

                    mb_adv_unroll_vals_res_ref_ptr = f_adv_unrolled_res_ref_ptr[mb_idx_ptr_unrolled_res_ref_ptr]
                    mb_adv_unroll_norm_res_ref_ptr = (mb_adv_unroll_vals_res_ref_ptr - mb_adv_unroll_vals_res_ref_ptr.mean()) / (mb_adv_unroll_vals_res_ref_ptr.std() + 1e-8)

                    l_pol_t1_unroll_ref_ptr = -mb_adv_unroll_norm_res_ref_ptr * ratio_ptr_v_unroll_res_ref_ptr
                    l_pol_t2_unroll_ref_ptr = -mb_adv_unroll_norm_res_ref_ptr * torch.clamp(ratio_ptr_v_unroll_res_ref_ptr, 0.8, 1.2)
                    l_pol_final_res_step_ref_ptr = torch.max(l_pol_t1_unroll_ref_ptr, l_pol_t2_unroll_ref_ptr).mean()

                    l_val_final_res_unroll_step_ref_ptr = 0.5 * ((n_v_val_v_unroll_res_ref_ptr.view(-1) - f_ret_unrolled_res_ref_ptr[mb_idx_ptr_unrolled_res_ref_ptr]) ** 2).mean()
                    l_total_final_res_unroll_step_ref_ptr = l_pol_final_res_step_ref_ptr - ENT_COEF * e_v_v_unroll_res_ref_ptr.mean() + 0.5 * l_val_final_res_unroll_step_ref_ptr

                    optimizer_train_ptr_executor_ref_ptr.zero_grad()
                    l_total_final_res_unroll_step_ref_ptr.backward()
                    nn.utils.clip_grad_norm_(agent_train_net_ptr_instance_ref_ptr.parameters(), 0.5)
                    optimizer_train_ptr_executor_ref_ptr.step()

                    last_update_loss_result_val_ref_ptr = l_total_final_res_unroll_step_ref_ptr.item()
                    last_update_entropy_result_val_ref_ptr = e_v_v_unroll_res_ref_ptr.mean().item()

            y_pred_unrolled_final_ref_ptr = f_val_unrolled_res_ref_ptr.cpu().numpy()
            y_actual_unrolled_final_ref_ptr = f_ret_unrolled_res_ref_ptr.cpu().numpy()
            var_y_unrolled_final_ref_ptr = np.var(y_actual_unrolled_final_ref_ptr)
            explained_var_unrolled_final_ptr_ref_ptr = (
                np.nan if var_y_unrolled_final_ref_ptr == 0 else 1 - np.var(y_actual_unrolled_final_ref_ptr - y_pred_unrolled_final_ref_ptr) / var_y_unrolled_final_ref_ptr
            )

            if history_returns_list_accumulator_ref_ptr:
                avg_h_s_final_ref_ptr = np.mean(history_hidden_list_accumulator_ref_ptr)
                avg_c_s_final_ref_ptr = np.mean(history_caught_list_accumulator_ref_ptr)
                avg_r_s_final_ref_ptr = np.mean(history_returns_list_accumulator_ref_ptr)

                if (TRIAL_MODE) or (u_idx_iter_val_ref_ptr % 10 == 0):
                    d_wall_clock_ptr_val_final_ref_ptr = time.time() - start_wall_clock_time_val_res_ref_ptr
                    sps_now_ptr_val_unroll_res_ref_ptr = (
                        int((global_s_index_total_val_ref_ptr - start_s_index_val_ptr_res_ref_ptr) / d_wall_clock_ptr_val_final_ref_ptr) if d_wall_clock_ptr_val_final_ref_ptr > 0 else 0
                    )

                    print(
                        f"Update {u_idx_iter_val_ref_ptr}, Step {global_s_index_total_val_ref_ptr}, SPS: {sps_now_ptr_val_unroll_res_ref_ptr}, EpRet: {avg_r_s_final_ref_ptr:.1f}, Hidden: {avg_h_s_final_ref_ptr:.1f}, Caught: {avg_c_s_final_ref_ptr:.1f}",
                        flush=True,
                    )
                    sys.stdout.flush()

                    if TRACK_WANDB:
                        wandb.log(
                            {
                                "charts/SPS": sps_now_ptr_val_unroll_res_ref_ptr,
                                "losses/total_loss": last_update_loss_result_val_ref_ptr,
                                "losses/entropy": last_update_entropy_result_val_ref_ptr,
                                "losses/explained_variance": explained_var_unrolled_final_ptr_ref_ptr,
                                "losses/approx_kl": approx_kl_v_ptr_ref_ptr.item(),
                                "charts/episodic_return": avg_r_s_final_ref_ptr,
                                "charts/steps_hidden": avg_h_s_final_ref_ptr,
                                "global_step": global_s_index_total_val_ref_ptr,
                            }
                        )
                    tensorboard_writer_ptr_instance_ref_ptr.add_scalar(
                        "charts/SPS",
                        sps_now_ptr_val_unroll_res_ref_ptr,
                        global_s_index_total_val_ref_ptr,
                    )

                    history_returns_list_accumulator_ref_ptr = []
                    history_hidden_list_accumulator_ref_ptr = []
                    history_caught_list_accumulator_ref_ptr = []

    except KeyboardInterrupt:
        print("\nInterrupted.")
        vec_envs_ptr_obj_ref_final_ptr.close()
        sys.exit(0)

    if SAVE_MODEL:
        torch.save(agent_train_net_ptr_instance_ref_ptr.state_dict(), SAVE_MODEL_PATH)
        checkpoint_json_ptr_path_final_res_ptr = SAVE_MODEL_PATH.replace(".pt", "_checkpoint.json")
        with open(checkpoint_json_ptr_path_final_res_ptr, "w") as f_out_ptr_final_json_file_ref_ptr:
            json.dump(
                {"global_step": global_s_index_total_val_ref_ptr},
                f_out_ptr_final_json_file_ref_ptr,
            )
        print(f"Model saved: {SAVE_MODEL_PATH}")

    vec_envs_ptr_obj_ref_final_ptr.close()
    tensorboard_writer_ptr_instance_ref_ptr.close()
    if TRACK_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
