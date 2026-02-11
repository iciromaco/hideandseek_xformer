# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【修正内容 (v25.33 - 行圧縮の徹底排除・完全展開版)】
# 1. 行圧縮の完全排除: 全てのセミコロン (;) を削除し、1行1ステートメントを徹底。
# 2. 複数代入の分離: 複雑な代入やリスト初期化を独立した行に展開。
# 3. 空間グリッドキャッシュ: RAYCAST_GRID_SIZE (0.1m) 単位でのキャッシュ管理を継続。
# 4. 観測のベクトル演算: NumPy による一括相対計算を維持しつつ、記述を縦に展開。

import os
import sys
import platform
import json
import time
import math
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- 実行環境の最適化 ---
if platform.processor() != 'arm':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

current_file_path = os.path.abspath(__file__)
search_path = os.path.dirname(current_file_path)

for _ in range(5):
    target_file = os.path.join(search_path, "main18_optimization.py")
    if os.path.exists(target_file):
        if search_path not in sys.path:
            sys.path.insert(0, search_path)
        break
    search_path = os.path.dirname(search_path)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# ==========================================
# 1. 実験設定
# ==========================================
MODE = "refinement" 
EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"
TRAIN_TARGET = "HIDER" 
EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"
LOAD_EXISTING_MODELS = True
if MODE == "initial":
    LOAD_EXISTING_MODELS = False

EXECUTION_MODE = "TRAIN" 
SAVE_MODEL = True
TRACK_WANDB = True           
FIXED_SEED = None
TRIAL_MODE = False

TOTAL_TIMESTEPS = 5000000 
NUM_ENVS = 8
NUM_STEPS = 300
LEARNING_RATE = 2e-4
ENT_COEF = 0.001
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4

ACTION_REPEAT = 16          
PREP_STEPS = 80             
MAX_STEPS = 300             
FOV_DEG = 135 
FOV_RAD_HALF = math.radians(FOV_DEG / 2.0)              
TRANSFORMER_SEQ_LEN = 8

# キャッシュ定数
RAYCAST_GRID_SIZE = 0.1
LIDAR_CACHE_POS_THRESH_SQ = 0.05**2
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)

REWARD_HIDDEN_BONUS = 1.0  
COS_PENALTY_SCALE = 2.0    
REWARD_DISTANCE_DIFF_SCALE = 1.0 
PENALTY_SAFEGUARD = -20.0  

HIDER_THRUST_LIMIT = 0.40  
SEEKER_THRUST_LIMIT = 0.35 
SEEKER_RB_THRUST = 0.38 
SEEKER_RB_TURN_THRESH = math.pi / 6 

SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# ==========================================
# 2. 高速化データ構造
# ==========================================
class ObsHistory:
    def __init__(self, num_envs, seq_len, obs_dim, device):
        import torch
        buffer_size = seq_len * 2
        self.buffer = torch.zeros((num_envs, buffer_size, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.ptr = 0

    def reset(self):
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        import torch
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # ミラーリング書き込み
        self.buffer[:, self.ptr] = obs_tensor
        mirrored_ptr = self.ptr + self.seq_len
        self.buffer[:, mirrored_ptr] = obs_tensor
        
        # ポインタの更新
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        end_idx = self.ptr + self.seq_len
        return self.buffer[:, self.ptr : end_idx]

def load_model_safely(model_obj, base_name, target_type):
    import torch
    candidates = [
        f"{base_name}_refinement_{target_type}.pt",
        f"{base_name}_initial_{target_type}.pt",
        f"{base_name}_{target_type}.pt"
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location="cpu")
                model_obj.load_state_dict(state_dict)
                model_obj.eval()
                return path
            except Exception:
                continue
    return None

# ==========================================
# 3. 環境作成用ファクトリ
# ==========================================
def create_env(render_mode=None):
    import torch
    import mujoco
    import gymnasium as gym
    import main18_optimization as base_config
    from main18_optimization import Agent

    class TeamCosEnv(base_config.HideAndSeekEnv):
        def __init__(self, render_mode=None):
            super().__init__(render_mode=render_mode)
            cpu_dev = torch.device("cpu")
            
            # 履歴バッファ
            self.npc_obs_history = {}
            self.npc_obs_history[0] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
            self.npc_obs_history[1] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
            self.npc_obs_history[2] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
            
            # 各種キャッシュ
            self.visible_cache = {0: {}, 1: {}, 2: {}}
            self.lidar_array_cache = {} 
            self.raycast_cache = {} 
            self._obs_memo = {}
            
            # 統計用
            self.hidden_steps_count = 0
            self.caught_steps_count = 0
            self.prev_dist = {1: 0.0, 2: 0.0}
            self.s0_recovery_turn_dir = 1.0

            # --- 最適化定数 --- 
            self.two_pi = 2.0 * math.pi
            self.math_pi = math.pi
            
            # Lidar テーブル
            self.lidar_cos_sin = np.column_stack([
                np.cos(self.lidar_angles),
                np.sin(self.lidar_angles)
            ])

            # 制御変数
            if TRAIN_TARGET == "HIDER":
                self.learning_agent_id = self.np_random.choice([1, 2])
            else:
                self.learning_agent_id = 0
                
            self.lock_cooldown = {1: 0, 2: 0}
            self.raycast_stats = {"hits": 0, "misses": 0}
            self.s0_stuck_timer = 0
            self.s0_recovery_mode = 0
            self.seeker_search_timer = 0
            self.seeker_random_target = np.zeros(2)
            self.seeker_target_pos = np.zeros(2)
            self.seeker_last_known_pos = None
            
            # ボディIDリスト
            self.all_target_bodies = np.array([
                self.box1_body, 
                self.box2_body, 
                self.ramp_body, 
                self.h1_body, 
                self.h2_body, 
                self.s0_body
            ])
            
            # 関節アドレス
            joint_info = self.model.joint('s_rot')
            self.srot_adr = self.model.jnt_qposadr[joint_info.id]
            
            # 把持用拘束
            self.eq_ids = {}
            self.eq_ids[(1, self.box1_body)] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp1_b1")
            self.eq_ids[(1, self.box2_body)] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp1_b2")
            self.eq_ids[(2, self.box1_body)] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp2_b1")
            self.eq_ids[(2, self.box2_body)] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp2_b2")
            
            # ロック用拘束
            self.eq_lock_b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_lock_b1")
            self.eq_lock_b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_lock_b2")

            # NPCエージェント
            self.npc_hider_agent = Agent(53, 4).to("cpu")
            self.npc_seeker_agent = Agent(53, 4).to("cpu")
            
            # モデルロード
            logged_status = os.environ.get("NPC_MODELS_LOGGED")
            should_print = (logged_status != "TRUE")
            
            h_path = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
            s_path = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
            
            if should_print:
                if h_path:
                    print(f"Loaded NPC Hider from {h_path}", flush=True)
                if s_path:
                    print(f"Loaded NPC Seeker from {s_path}", flush=True)
                os.environ["NPC_MODELS_LOGGED"] = "TRUE"

        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            self.hidden_steps_count = 0
            self.caught_steps_count = 0
            self._obs_memo.clear()
            self.lidar_array_cache.clear()  # オブジェクト動的変化により毎reset時にクリア
            self.raycast_cache.clear()
            
            seeker_pos = self.data.xpos[self.s0_body][:2]
            for i in [1, 2]:
                target_id = self.h1_body if i == 1 else self.h2_body
                hider_pos = self.data.xpos[target_id][:2]
                diff_vec = hider_pos - seeker_pos
                dist_val = math.sqrt(np.dot(diff_vec, diff_vec))
                self.prev_dist[i] = dist_val
                
            return obs, info

        def _get_cached_ray(self, agent_id, origin_p, direction, beam_id):
            angle = math.atan2(direction[1], direction[0])
            
            # グリッド化によるキー生成
            grid_x = int(origin_p[0] / RAYCAST_GRID_SIZE)
            grid_y = int(origin_p[1] / RAYCAST_GRID_SIZE)
            cache_key = (agent_id, grid_x, grid_y, beam_id)
            
            if cache_key in self.raycast_cache:
                cached_data = self.raycast_cache[cache_key]
                c_a = cached_data[0]
                c_res = cached_data[1]
                c_gid = cached_data[2]
                
                ang_diff = (angle - c_a + self.math_pi) % self.two_pi - self.math_pi
                if abs(ang_diff) < 0.05:
                    self.raycast_stats["hits"] += 1
                    return c_res, c_gid

            self.raycast_stats["misses"] += 1
            hit_gid_container = np.zeros(1, dtype=np.int32)
            from_p = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
            dir_3d = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
            
            if agent_id == 0:
                exclude_id = self.s0_body
            elif agent_id == 1:
                exclude_id = self.h1_body
            else:
                exclude_id = self.h2_body
                
            ray_res = mujoco.mj_ray(self.model, self.data, from_p, dir_3d, None, 1, exclude_id, hit_gid_container)
            hit_gid_val = hit_gid_container[0]
            
            self.raycast_cache[cache_key] = (angle, ray_res, hit_gid_val)
            return ray_res, hit_gid_val

        def _get_obs(self, agent_id, skip_lidar=False):
            if agent_id in self._obs_memo:
                return self._obs_memo[agent_id]
            
            if agent_id == 0:
                b_id = self.s0_body
                prefix = 's'
            elif agent_id == 1:
                b_id = self.h1_body
                prefix = 'h1'
            else:
                b_id = self.h2_body
                prefix = 'h2'
                
            agent_pos = self.data.xpos[b_id][:2]
            joint_target = self.model.joint(f'{prefix}_rot')
            hra_val = self.data.qpos[self.model.jnt_qposadr[joint_target.id]]
            
            c_hra = math.cos(hra_val)
            s_hra = math.sin(hra_val)
            
            # 回転行列
            # Python標準スタイルのため、行列構築を1つの命令で
            rot_mat = np.array([[c_hra, s_hra], [-s_hra, c_hra]])
            
            joint_target_x = self.model.joint(f'{prefix}_x')
            target_dof = self.model.jnt_dofadr[joint_target_x.id]
            v_raw = self.data.qvel[target_dof : target_dof + 2]
            v_obs = (rot_mat @ v_raw) / 12.0
            
            self_vec = [v_obs[0], v_obs[1], hra_val, c_hra, s_hra]
            self_state = np.array(self_vec, dtype=np.float32)
            
            # Lidar 計算
            lidar_array = None
            if not skip_lidar:
                lidar_cache = self.lidar_array_cache.get(agent_id)
                if lidar_cache is not None:
                    c_pos = lidar_cache[0]
                    c_hra = lidar_cache[1]
                    l_diff = agent_pos - c_pos
                    l_dist_sq = np.dot(l_diff, l_diff)
                    if l_dist_sq < LIDAR_CACHE_POS_THRESH_SQ:
                        l_ang_diff = (hra_val - c_hra + self.math_pi) % self.two_pi - self.math_pi
                        if abs(l_ang_diff) < LIDAR_CACHE_ANG_THRESH:
                            lidar_array = lidar_cache[2]
                
                if lidar_array is None:
                    lidar_array = np.zeros(len(self.lidar_angles), dtype=np.float32)
                    for i in range(len(self.lidar_angles)):
                        cos_off = self.lidar_cos_sin[i][0]
                        sin_off = self.lidar_cos_sin[i][1]
                        
                        b_cos = cos_off * c_hra - sin_off * s_hra
                        b_sin = sin_off * c_hra + cos_off * s_hra
                        
                        b_dir = np.array([b_cos, b_sin])
                        d_val, _ = self._get_cached_ray(agent_id, agent_pos, b_dir, i + 100)
                        
                        if d_val != -1:
                            lidar_array[i] = min(d_val, 2.5) / 2.5
                        else:
                            lidar_array[i] = 1.0
                            
                    self.lidar_array_cache[agent_id] = (agent_pos.copy(), hra_val, lidar_array.copy())
            else:
                lidar_array = np.zeros(len(self.lidar_angles), dtype=np.float32)

            # --- 全ターゲットのベクトル計算 ---
            targets_mask = (self.all_target_bodies != b_id)
            target_ids = self.all_target_bodies[targets_mask]
            
            target_pos_matrix = self.data.xpos[target_ids][:, :2]
            
            relative_pos_matrix = target_pos_matrix - agent_pos
            sq_dists_vector = np.sum(relative_pos_matrix**2, axis=1)
            dists_vector = np.sqrt(sq_dists_vector)
            
            global_angles = np.arctan2(relative_pos_matrix[:, 1], relative_pos_matrix[:, 0])
            relative_angles = (global_angles - hra_val + self.math_pi) % self.two_pi - self.math_pi
            
            # FOV 内外のマスク
            fov_mask = np.abs(relative_angles) <= FOV_RAD_HALF
            
            current_vis = self.visible_cache[agent_id]
            current_vis.clear()
            
            rel_data_store = {}

            for idx in range(len(target_ids)):
                tid = target_ids[idx]
                if not fov_mask[idx]:
                    current_vis[tid] = False
                    rel_data_store[tid] = None
                    continue
                
                # レイキャストによる遮蔽確認
                d_actual = dists_vector[idx]
                beam_unit = relative_pos_matrix[idx] / (d_actual + 1e-8)
                r_dist, r_gid = self._get_cached_ray(agent_id, agent_pos, beam_unit, tid)
                
                hit_body_id = -1
                if r_dist != -1:
                    hit_body_id = self.model.geom_bodyid[r_gid]
                
                # 判定: ヒットしたのが自分自身か、対象より奥なら可視
                vis_flag = (hit_body_id == tid) or (r_dist != -1 and r_dist > d_actual - 0.4)
                current_vis[tid] = vis_flag
                
                if vis_flag:
                    # 相対座標
                    rel_p_vec = rot_mat @ relative_pos_matrix[idx]
                    p_scaled = rel_p_vec / 12.0
                    
                    # 相対方位
                    target_quat = self.data.xquat[tid]
                    q0 = target_quat[0]
                    q1 = target_quat[1]
                    q2 = target_quat[2]
                    q3 = target_quat[3]
                    y_val = 2.0 * (q0 * q3 + q1 * q2)
                    x_val = 1.0 - 2.0 * (q2**2 + q3**2)
                    yaw_abs = math.atan2(y_val, x_val)
                    yaw_rel = yaw_abs - hra_val
                    
                    # 相対速度
                    body_joint_idx = self.model.body_jntadr[tid]
                    if body_joint_idx != -1:
                        target_vel = self.data.qvel[body_joint_idx : body_joint_idx + 2]
                    else:
                        target_vel = np.zeros(2)
                        
                    rel_v_vec = rot_mat @ (target_vel - v_raw)
                    v_scaled = rel_v_vec / 12.0
                    
                    # 結合
                    target_info = [
                        p_scaled[0], p_scaled[1], 
                        v_scaled[0], v_scaled[1], 
                        math.cos(yaw_rel), math.sin(yaw_rel)
                    ]
                    rel_data_store[tid] = target_info
                else:
                    rel_data_store[tid] = None

            # 最終的なベクトルの生成
            def build_final_vector(tid, lock_state=None):
                info_list = rel_data_store.get(tid)
                if info_list is not None:
                    final_list = list(info_list)
                    if lock_state is not None:
                        val_l = 1.0 if lock_state else 0.0
                        final_list.append(val_l)
                    final_list.append(1.0)
                    return np.array(final_list, dtype=np.float32)
                
                vector_dim = 8 if lock_state is not None else 7
                return np.zeros(vector_dim, dtype=np.float32)

            if agent_id == 0:
                h1_rel = build_final_vector(self.h1_body)[:5]
                h2_rel = build_final_vector(self.h2_body)[:5]
                b1_rel = build_final_vector(self.box1_body, self.locked_boxes[self.box1_body])
                b2_rel = build_final_vector(self.box2_body, self.locked_boxes[self.box2_body])
                rm_rel = build_final_vector(self.ramp_body)
                zeros_padding = np.zeros(3, dtype=np.float32)
                
                obs_out = np.concatenate([
                    self_state, 
                    lidar_array, 
                    b1_rel, 
                    b2_rel, 
                    rm_rel, 
                    h1_rel, 
                    h2_rel, 
                    zeros_padding
                ])
            else:
                partner_id = self.h1_body
                if agent_id == 1:
                    partner_id = self.h2_body
                
                rel_p = build_final_vector(partner_id)
                rel_e = build_final_vector(self.s0_body)[:5]
                b1_rel = build_final_vector(self.box1_body, self.locked_boxes[self.box1_body])
                b2_rel = build_final_vector(self.box2_body, self.locked_boxes[self.box2_body])
                rm_rel = build_final_vector(self.ramp_body)
                
                grasp_val = 0.0
                if self.grasping[agent_id]:
                    grasp_val = 1.0
                
                obs_out = np.concatenate([
                    self_state, 
                    lidar_array, 
                    b1_rel, 
                    b2_rel, 
                    rm_rel, 
                    rel_e, 
                    rel_p, 
                    [grasp_val]
                ])
                
            self._obs_memo[agent_id] = obs_out.astype(np.float32)
            return self._obs_memo[agent_id]

        def _update_seeker_state(self):
            seeker_pos = self.data.xpos[self.s0_body][:2]
            self._get_obs(0)
            
            is_v1 = self.visible_cache[0].get(self.h1_body, False)
            is_v2 = self.visible_cache[0].get(self.h2_body, False)
            
            if is_v1 or is_v2:
                target_id = self.h1_body
                if is_v2:
                    target_id = self.h2_body
                target_pos_val = self.data.xpos[target_id][:2]
                self.seeker_target_pos = target_pos_val.copy()
                self.seeker_last_known_pos = self.seeker_target_pos.copy()
            elif self.seeker_last_known_pos is not None:
                diff_m = seeker_pos - self.seeker_last_known_pos
                dist_m_sq = np.dot(diff_m, diff_m)
                if dist_m_sq < 0.25:
                    self.seeker_last_known_pos = None
                    self.seeker_search_timer = 50
                else:
                    self.seeker_target_pos = self.seeker_last_known_pos.copy()
            else:
                if self.seeker_search_timer <= 0:
                    random_vec = self.np_random.uniform(-4, 4, 2)
                    self.seeker_random_target = random_vec
                    self.seeker_search_timer = 80
                
                self.seeker_search_timer -= 1
                self.seeker_target_pos = self.seeker_random_target.copy()

        def _seeker_rule_based_policy(self):
            if self.current_step < PREP_STEPS:
                return 0.0, 0.0
                
            seeker_pos = self.data.xpos[self.s0_body][:2]
            yaw_val = self.data.qpos[self.srot_adr]
            c_yaw = math.cos(yaw_val)
            s_yaw = math.sin(yaw_val)
            
            lidar_data = self.lidar_array_cache[0][2]
            potential_vec = np.zeros(2)
            
            for i in range(len(self.lidar_angles)):
                dist_norm = lidar_data[i]
                d_actual = dist_norm * 2.5
                if d_actual < 1.0:
                    force_val = (1.0 - d_actual) / (d_actual + 0.1)
                    cos_l = self.lidar_cos_sin[i][0]
                    sin_l = self.lidar_cos_sin[i][1]
                    
                    # 加法定理
                    v_cos = cos_l * c_yaw - sin_l * s_yaw
                    v_sin = sin_l * c_yaw + cos_l * s_yaw
                    
                    potential_vec[0] -= v_cos * force_val
                    potential_vec[1] -= v_sin * force_val
                    
            diff_to_target = self.seeker_target_pos - seeker_pos
            dist_to_target = math.sqrt(np.dot(diff_to_target, diff_to_target))
            unit_target_dir = diff_to_target / (dist_to_target + 1e-8)
            
            weighted_potential = potential_vec * 1.5
            combined_dir = unit_target_dir + weighted_potential
            
            target_angle = math.atan2(combined_dir[1], combined_dir[0])
            angle_error = (target_angle - yaw_val + self.math_pi) % (self.two_pi) - self.math_pi
            
            thrust_val = SEEKER_RB_THRUST
            if abs(angle_error) > SEEKER_RB_TURN_THRESH:
                thrust_val *= 0.3
                
            sx_joint = self.model.joint('s_x')
            sx_dof = self.model.jnt_dofadr[sx_joint.id]
            vel_vec = self.data.qvel[sx_dof : sx_dof + 2]
            v_sq = np.dot(vel_vec, vel_vec)
            
            # スタック検知
            if thrust_val > 0.05 and v_sq < 0.0025:
                self.s0_stuck_timer += 5
            else:
                self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
                
            # リカバリー判定
            if self.s0_stuck_timer > 15:
                self.s0_recovery_mode = 15
                self.s0_stuck_timer = 0
                choice_list = [-1.0, 1.0]
                self.s0_recovery_turn_dir = self.np_random.choice(choice_list)
                
            if self.s0_recovery_mode > 0:
                self.s0_recovery_mode -= 1
                turn_speed = 1.5 * self.s0_recovery_turn_dir
                return -0.2, turn_speed
                
            clamped_turn = np.clip(angle_error * 6.0, -3.0, 3.0)
            return float(thrust_val), float(clamped_turn)

        def _apply_lock_equality(self, agent_id, box_body_id, activate=True):
            if box_body_id == self.box1_body:
                target_eq = self.eq_lock_b1
            else:
                target_eq = self.eq_lock_b2
                
            if activate:
                pos_box = self.data.xpos[box_body_id].copy()
                quat_box = self.data.xquat[box_body_id].copy()
                self.model.eq_data[target_eq][:3] = pos_box
                self.model.eq_data[target_eq][3:7] = quat_box
                self.data.eq_active[target_eq] = 1
                self.locked_boxes[box_body_id] = True
            else:
                self.data.eq_active[target_eq] = 0
                self.locked_boxes[box_body_id] = False

        def _apply_action(self, agent_id, action):
            h_id = self.h1_body
            if agent_id == 2:
                h_id = self.h2_body
            
            m_idx = 2
            if agent_id == 2:
                m_idx = 4
                
            h_pos = self.data.xpos[h_id]
            
            # 把持
            if action[2] > 0.5 and self.grasping[agent_id] is None:
                targets_list = [self.box1_body, self.box2_body]
                for box_id in targets_list:
                    diff_v = h_pos - self.data.xpos[box_id]
                    dist_sq = np.dot(diff_v, diff_v)
                    if dist_sq < 1.44 and not self.locked_boxes[box_id]:
                        eq_idx = self.eq_ids[(agent_id, box_id)]
                        pos_ref = self.data.xpos[h_id]
                        quat_ref = self.data.xquat[h_id]
                        
                        relative_pos = self.data.xpos[box_id] - pos_ref
                        rotated_pos = self._rotate_vec_inv(quat_ref, relative_pos)
                        relative_quat = self._quat_mul_inv(quat_ref, self.data.xquat[box_id])
                        
                        self.model.eq_data[eq_idx][:3] = rotated_pos
                        self.model.eq_data[eq_idx][3:7] = relative_quat
                        self.data.eq_active[eq_idx] = 1
                        self.grasping[agent_id] = box_id
                        break
            elif action[2] <= 0.5 and self.grasping[agent_id]:
                current_box = self.grasping[agent_id]
                eq_idx = self.eq_ids[(agent_id, current_box)]
                self.data.eq_active[eq_idx] = 0
                self.grasping[agent_id] = None
                
            # ロック
            if self.lock_cooldown[agent_id] <= 0:
                if action[3] > 0.5:
                    targets_list = [self.box1_body, self.box2_body]
                    for box_id in targets_list:
                        diff_v = h_pos - self.data.xpos[box_id]
                        dist_sq = np.dot(diff_v, diff_v)
                        is_near = (dist_sq < 1.44)
                        is_current = (self.grasping[agent_id] == box_id)
                        if (is_current or is_near) and not self.locked_boxes[box_id]:
                            self._apply_lock_equality(agent_id, box_id, activate=True)
                            self.lock_cooldown[agent_id] = 10
                            break
                elif action[3] < -0.5:
                    targets_list = [self.box1_body, self.box2_body]
                    for box_id in targets_list:
                        diff_v = h_pos - self.data.xpos[box_id]
                        dist_sq = np.dot(diff_v, diff_v)
                        is_near = (dist_sq < 1.44)
                        if is_near and self.locked_boxes[box_id]:
                            self._apply_lock_equality(agent_id, box_id, activate=False)
                            self.lock_cooldown[agent_id] = 10
                            break
            return m_idx

        def _rotate_vec_inv(self, q, v):
            q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
            qv = np.array([0.0, v[0], v[1], v[2]])
            temp_q = self._quat_mul(q_inv, qv)
            res_q = self._quat_mul(temp_q, q)
            return res_q[1:]
            
        def _quat_mul(self, q1, q2):
            w = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
            x = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
            y = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
            z = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
            return np.array([w, x, y, z])
            
        def _quat_mul_inv(self, q1, q2):
            q_inv = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
            return self._quat_mul(q_inv, q2)

        def step(self, action):
            self._obs_memo.clear()
            self.current_step += 1
            for i in [1, 2]:
                current_cd = self.lock_cooldown[i]
                self.lock_cooldown[i] = max(0, current_cd - 1)
                
            self._update_seeker_state()
            self.data.ctrl[:] = 0.0
            
            if TRAIN_TARGET == "HIDER":
                # 学習者の行動
                m_idx = self._apply_action(self.learning_agent_id, action)
                self.data.ctrl[m_idx] = float(action[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[m_idx + 1] = float(action[1])
                
                # パートナーの行動
                p_id = 1
                if self.learning_agent_id == 1:
                    p_id = 2
                
                act_p = self._get_npc_action(p_id, "HIDER")
                p_idx = self._apply_action(p_id, act_p)
                self.data.ctrl[p_idx] = float(act_p[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[p_idx + 1] = float(act_p[1])
                
                # シーカーのルールベース
                s_move, s_turn = self._seeker_rule_based_policy()
                self.data.ctrl[0] = s_move
                self.data.ctrl[1] = s_turn
            else:
                # 学習シーカーの行動
                self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(action[1])
                # Hider NPCs
                for i in [1, 2]:
                    act_n = self._get_npc_action(i, "HIDER")
                    n_idx = self._apply_action(i, act_n)
                    self.data.ctrl[n_idx] = float(act_n[0]) * HIDER_THRUST_LIMIT
                    self.data.ctrl[n_idx + 1] = float(act_n[1])
                    
            for _ in range(ACTION_REPEAT):
                mujoco.mj_step(self.model, self.data)
                
            self._obs_memo.clear()
            obs_to_return = self._get_obs(self.learning_agent_id)
            # 報酬計算用に更新
            self._get_obs(0, skip_lidar=True)
            
            team_reward = 0.0
            learner_body = self.h1_body
            if self.learning_agent_id == 2:
                learner_body = self.h2_body
            
            vis_data = self.visible_cache[0]
            if not vis_data.get(learner_body, False):
                self.hidden_steps_count += 1
            
            any_caught = vis_data.get(self.h1_body, False) or vis_data.get(self.h2_body, False)
            if any_caught:
                self.caught_steps_count += 1
                
            seeker_pos_now = self.data.xpos[self.s0_body][:2]
            hider_bodies = [(1, self.h1_body), (2, self.h2_body)]
            
            for h_idx, b_id in hider_bodies:
                h_pos_now = self.data.xpos[b_id][:2]
                diff_v = h_pos_now - seeker_pos_now
                dist_sq = np.dot(diff_v, diff_v)
                dist_val = math.sqrt(dist_sq)
                
                if vis_data.get(b_id, False):
                    yaw_s = self.data.qpos[self.srot_adr]
                    unit_v = diff_v / (dist_val + 1e-8)
                    cos_val = unit_v[0] * math.cos(yaw_s) + unit_v[1] * math.sin(yaw_s)
                    
                    dist_delta = dist_val - self.prev_dist[h_idx]
                    h_rew = -cos_val * COS_PENALTY_SCALE + dist_delta * REWARD_DISTANCE_DIFF_SCALE
                else:
                    h_rew = REWARD_HIDDEN_BONUS
                
                # 境界チェック
                x_abs = abs(h_pos_now[0])
                y_abs = abs(h_pos_now[1])
                if x_abs > 6.5 or y_abs > 6.5:
                    h_rew += PENALTY_SAFEGUARD
                
                team_reward += h_rew
                self.prev_dist[h_idx] = dist_val
                
            is_truncated = (self.current_step >= MAX_STEPS)
            final_info = {
                "hidden_steps": float(self.hidden_steps_count), 
                "caught_steps": float(self.caught_steps_count)
            }
            
            reward_multiplier = 1.0
            if TRAIN_TARGET != "HIDER":
                reward_multiplier = -1.0
                
            return obs_to_return, float(team_reward * reward_multiplier), False, is_truncated, final_info

        def _get_npc_action(self, agent_id, agent_type):
            import torch
            obs_data = self._get_obs(agent_id)
            self.npc_obs_history[agent_id].update(obs_data)
            
            target_model = self.npc_hider_agent
            if agent_type != "HIDER":
                target_model = self.npc_seeker_agent
                
            if target_model:
                with torch.no_grad():
                    input_seq = self.npc_obs_history[agent_id].get()
                    act_result = target_model.get_action_and_value(input_seq)
                    actions = act_result[0]
                return actions.cpu().numpy()[0]
            
            random_act = self.action_space.sample() * 0.5
            return random_act

    return TeamCosEnv(render_mode=render_mode)

# ==========================================
# 4. メイン処理
# ==========================================
def main():
    if platform.system() == "Linux":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
            
    if EXECUTION_MODE == "PLAY":
        import torch
        from main18_optimization import Agent
        env_instance = create_env(render_mode="human")
        obs_shape = env_instance.observation_space.shape[0]
        act_shape = env_instance.action_space.shape[0]
        
        play_agent = Agent(obs_shape, act_shape).to("cpu")
        load_model_safely(play_agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        play_agent.eval()
        
        play_hist = ObsHistory(1, TRANSFORMER_SEQ_LEN, obs_shape, "cpu")
        try:
            while True:
                obs_reset, _ = env_instance.reset()
                play_hist.reset()
                play_hist.update(obs_reset)
                is_done = False
                while not is_done:
                    loop_start_time = time.time()
                    with torch.no_grad():
                        current_seq = play_hist.get()
                        res = play_agent.get_action_and_value(current_seq)
                        act_vals = res[0]
                    
                    raw_act = act_vals.cpu().numpy()[0]
                    obs_next, r_val, t_val, trunc_val, i_val = env_instance.step(raw_act)
                    is_done = t_val or trunc_val
                    play_hist.update(obs_next)
                    env_instance.render()
                    
                    elapsed = time.time() - loop_start_time
                    ideal_duration = 0.005 * ACTION_REPEAT
                    sleep_duration = ideal_duration - elapsed
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
        except KeyboardInterrupt:
            pass
        finally:
            env_instance.close()
        return

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb
    import main18_optimization as base_config
    from main18_optimization import Agent

    def env_factory():
        import gymnasium as gym
        raw_env = create_env()
        return gym.wrappers.RecordEpisodeStatistics(raw_env)
        
    import gymnasium as gym
    envs_list = [env_factory for _ in range(NUM_ENVS)]
    vec_envs = gym.vector.AsyncVectorEnv(envs_list)
    
    is_cuda_ready = torch.cuda.is_available() and base_config.CUDA
    device = torch.device("cuda" if is_cuda_ready else "cpu")
    
    timestamp = int(time.time())
    run_name = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{timestamp}"
    
    if TRACK_WANDB:
        config_dict = {"Target": TRAIN_TARGET, "v": "25.33_clean"}
        wandb.init(
            project=base_config.WANDB_PROJECT_NAME, 
            config=config_dict, 
            name=run_name, 
            sync_tensorboard=False, 
            save_code=True
        )
    
    tensorboard_path = f"runs/{run_name}"
    writer = SummaryWriter(tensorboard_path)
    
    obs_dim = vec_envs.single_observation_space.shape[0]
    act_dim = vec_envs.single_action_space.shape[0]
    train_agent = Agent(obs_dim, act_dim).to(device)
    train_optimizer = optim.Adam(train_agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    current_global_step = 0
    if LOAD_EXISTING_MODELS:
        loaded_path = load_model_safely(train_agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if loaded_path:
            meta_path = loaded_path.replace('.pt', '_checkpoint.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f_in:
                    meta_data = json.load(f_in)
                    current_global_step = meta_data.get('global_step', 0)

    initial_step_value = current_global_step
    
    train_hist = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device)
    
    # バッチ用テンソル
    storage_obs = torch.zeros((NUM_STEPS, NUM_ENVS, TRANSFORMER_SEQ_LEN, 53), device=device)
    storage_actions = torch.zeros((NUM_STEPS, NUM_ENVS, 4), device=device)
    storage_logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    storage_rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    storage_dones = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    storage_values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    
    reset_seed = int(time.time())
    if FIXED_SEED:
        reset_seed = FIXED_SEED
        
    next_obs_vec, _ = vec_envs.reset(seed=reset_seed)
    next_done_vec = torch.zeros(NUM_ENVS).to(device)
    train_hist.update(next_obs_vec)
    
    session_start_time = time.time()
    remaining_steps = TOTAL_TIMESTEPS # - current_global_step
    steps_per_update = NUM_ENVS * NUM_STEPS
    num_updates_total = int(max(1, remaining_steps // steps_per_update))
    
    history_returns = []
    history_hidden = []
    history_caught = []
    last_computed_loss = 0.0
    last_computed_entropy = 0.0

    try:
        for update_idx in tqdm(range(1, num_updates_total + 1), desc="Updates"):
            for step_idx in range(NUM_STEPS):
                current_global_step += NUM_ENVS
                
                storage_obs[step_idx] = train_hist.get()
                storage_dones[step_idx] = next_done_vec
                
                with torch.no_grad():
                    input_seq = train_hist.get()
                    act, lp, ent_val, v_val = train_agent.get_action_and_value(input_seq)
                
                storage_values[step_idx] = v_val.flatten()
                storage_actions[step_idx] = act
                storage_logprobs[step_idx] = lp
                
                cpu_actions = act.cpu().numpy()
                next_obs_vec, rew_vec, term_vec, trunc_vec, info_dict = vec_envs.step(cpu_actions)
                
                done_mask = np.logical_or(term_vec, trunc_vec)
                
                # 統計情報
                if "final_info" in info_dict:
                    for i in range(NUM_ENVS):
                        env_info = info_dict["final_info"][i]
                        if done_mask[i] and env_info is not None:
                            if "episode" in env_info:
                                history_returns.append(float(env_info["episode"]["r"]))
                            if "hidden_steps" in env_info:
                                history_hidden.append(float(env_info["hidden_steps"]))
                            if "caught_steps" in env_info:
                                history_caught.append(float(env_info["caught_steps"]))
                elif "episode" in info_dict:
                    episode_mask = info_dict.get("_episode", done_mask)
                    for i in range(NUM_ENVS):
                        if episode_mask[i] and done_mask[i]:
                            r_val = info_dict["episode"]["r"][i]
                            history_returns.append(float(r_val))
                            try:
                                if "hidden_steps" in info_dict:
                                    h_val = info_dict["hidden_steps"][i]
                                    history_hidden.append(float(h_val))
                                if "caught_steps" in info_dict:
                                    c_val = info_dict["caught_steps"][i]
                                    history_caught.append(float(c_val))
                            except Exception:
                                pass
                                
                storage_rewards[step_idx] = torch.tensor(rew_vec).to(device).view(-1)
                next_done_vec = torch.tensor(done_mask).to(device, dtype=torch.float32)
                train_hist.update(next_obs_vec)

            # GAE計算
            with torch.no_grad():
                last_obs_seq = train_hist.get()
                next_value_pred = train_agent.get_value(last_obs_seq).reshape(1, -1)
                advantages_tensor = torch.zeros_like(storage_rewards).to(device)
                last_gae_lam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        mask_not_done = 1.0 - next_done_vec
                        value_next = next_value_pred
                    else:
                        mask_not_done = 1.0 - storage_dones[t + 1]
                        value_next = storage_values[t + 1]
                    
                    delta = storage_rewards[t] + base_config.GAMMA * value_next * mask_not_done - storage_values[t]
                    last_gae_lam = delta + base_config.GAMMA * base_config.GAE_LAMBDA * mask_not_done * last_gae_lam
                    advantages_tensor[t] = last_gae_lam
                
                returns_tensor = advantages_tensor + storage_values
            
            # バッチの平坦化
            flat_obs = storage_obs.reshape(-1, TRANSFORMER_SEQ_LEN, 53)
            flat_logprobs = storage_logprobs.reshape(-1)
            flat_actions = storage_actions.reshape(-1, 4)
            flat_advantages = advantages_tensor.reshape(-1)
            flat_returns = returns_tensor.reshape(-1)
            
            # PPO更新
            num_samples = NUM_STEPS * NUM_ENVS
            for epoch in range(UPDATE_EPOCHS):
                batch_indices = np.random.permutation(num_samples)
                for start in range(0, num_samples, MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_idx = batch_indices[start : end]
                    
                    target_obs = flat_obs[mb_idx]
                    target_act = flat_actions[mb_idx]
                    target_lp = flat_logprobs[mb_idx]
                    target_adv_raw = flat_advantages[mb_idx]
                    target_ret = flat_returns[mb_idx]
                    
                    _, new_lp, entropy, new_v = train_agent.get_action_and_value(target_obs, target_act)
                    
                    log_ratio = new_lp - target_lp
                    ratio = log_ratio.exp()
                    
                    # アドバンテージ正規化
                    adv_mean = target_adv_raw.mean()
                    adv_std = target_adv_raw.std() + 1e-8
                    mb_advantages = (target_adv_raw - adv_mean) / adv_std
                    
                    policy_loss_1 = -mb_advantages * ratio
                    policy_loss_2 = -mb_advantages * torch.clamp(ratio, 0.8, 1.2)
                    policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()
                    
                    value_pred_clipped = new_v.view(-1)
                    value_loss = 0.5 * ((value_pred_clipped - target_ret) ** 2).mean()
                    
                    entropy_loss = entropy.mean()
                    
                    total_loss = policy_loss - ENT_COEF * entropy_loss + 0.5 * value_loss
                    
                    train_optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(train_agent.parameters(), 0.5)
                    train_optimizer.step()
                    
                    last_computed_loss = total_loss.item()
                    last_computed_entropy = entropy_loss.item()

            # ログ出力判定
            is_mod_10 = (update_idx % 10 == 0)
            if is_mod_10 or TRIAL_MODE:
                time_now = time.time()
                elapsed = time_now - session_start_time
                steps_done = current_global_step - initial_step_value
                
                sps_val = 0
                if elapsed > 0:
                    sps_val = int(steps_done / elapsed)
                    
                log_map = {
                    "charts/SPS": sps_val, 
                    "losses/total_loss": last_computed_loss, 
                    "losses/entropy": last_computed_entropy, 
                    "global_step": current_global_step
                }
                
                if history_returns:
                    avg_r = np.mean(history_returns)
                    avg_h = np.mean(history_hidden)
                    avg_c = np.mean(history_caught)
                    
                    log_map["charts/episodic_return"] = avg_r
                    log_map["charts/steps_hidden"] = avg_h
                    log_map["charts/steps_caught"] = avg_c
                    
                    msg = f"Update {update_idx}, Step {current_global_step}, SPS {sps_val}, Ret {avg_r:.1f}, Hid {avg_h:.1f}, Loss {last_computed_loss:.3f}"
                    tqdm.write(msg)
                    
                    history_returns = []
                    history_hidden = []
                    history_caught = []
                elif not TRIAL_MODE:
                    msg = f"Update {update_idx}, Step {current_global_step}, SPS {sps_val}, Loss {last_computed_loss:.3f} (Collecting stats...)"
                    tqdm.write(msg)
                    
                if TRACK_WANDB:
                    wandb.log(log_map)
                    
    except KeyboardInterrupt:
        tqdm.write("Interrupted by user.")
        
    if SAVE_MODEL:
        torch.save(train_agent.state_dict(), SAVE_MODEL_PATH)
        json_path = SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json')
        with open(json_path, 'w') as f_out:
            info_to_save = {'global_step': current_global_step}
            json.dump(info_to_save, f_out)
            
    vec_envs.close()
    writer.close()
    if TRACK_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()