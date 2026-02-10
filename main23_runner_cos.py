# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【修正内容 (v25.14)】
# 1. 行圧縮（セミコロン ;）の完全排除:
#    - 全てのロジックを1行1ステートメントで記述し、文法的な不安定さを根絶。
#    - if文、for文、代入、関数呼び出しを全て独立した行に展開。
# 2. AttributeError (_check_collision_all) の完全修正:
#    - クラス構造を整理し、全てのヘルパーメソッドがインスタンスから確実に参照可能に。
# 3. Ubuntu Linux + CUDA 起動スタックの解消:
#    - AsyncVectorEnv の生成が完了するまで親プロセスが torch/CUDA に触れない構造を徹底。
#    - multiprocessing.set_start_method("spawn") を最優先で実行。
# 4. Seeker 知能と高速化の維持:
#    - 障害物回避AI、Lidarキャッシュ、RayCastキャッシュ、ゼロコピーバッファを完備。

import os
import sys
import platform
import json
import time
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- 実行環境の最適化 (数値計算ライブラリのスレッド競合抑制) ---
if platform.processor() != 'arm':
    for k in ["OMP", "MKL", "OPENBLAS", "VECLIB", "NUMEXPR"]:
        os.environ[f"{k}_NUM_THREADS"] = "1"

# --- プロジェクトルートの特定とパス追加 ---
current_file_path = os.path.abspath(__file__)
search_path = os.path.dirname(current_file_path)

# プロジェクトルートを探して sys.path に追加
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
MODE = "refinement" # "initial" ０からの学習，"refinement"　継続学習
EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"
TRAIN_TARGET = "HIDER" 

EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"
LOAD_EXISTING_MODELS = True if MODE == "refinement" else False
EXECUTION_MODE = "TRAIN" #　"PLAY" 

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

# 環境定数
ACTION_REPEAT = 16          
PREP_STEPS = 80             
MAX_STEPS = 300             
FOV_DEG = 135               
TRANSFORMER_SEQ_LEN = 8

# キャッシュ定数
LIDAR_CACHE_POS_THRESH = 0.05
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)
RAYCAST_CACHE_POS_THRESH = 0.05

# 報酬定数
REWARD_HIDDEN_BONUS = 1.0  
COS_PENALTY_SCALE = 2.0    
REWARD_DISTANCE_DIFF_SCALE = 1.0 
PENALTY_SAFEGUARD = -20.0  

HIDER_THRUST_LIMIT = 0.40  
SEEKER_THRUST_LIMIT = 0.35 
SEEKER_RB_THRUST = 0.38 
SEEKER_RB_TURN_THRESH = np.pi/6 

SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# ==========================================
# 2. 高速化データ構造 (ObsHistory)
# ==========================================
class ObsHistory:
    def __init__(self, num_envs, seq_len, obs_dim, device):
        """ミラーリングバッファを用いたゼロコピー履歴管理"""
        import torch
        self.buffer = torch.zeros((num_envs, seq_len * 2, obs_dim), device=device)
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
        
        # 2箇所に書き込み
        self.buffer[:, self.ptr] = obs_tensor
        self.buffer[:, self.ptr + self.seq_len] = obs_tensor
        
        # ポインタを回す
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        # Viewを返す (コピーなし)
        return self.buffer[:, self.ptr : self.ptr + self.seq_len]

# ==========================================
# 3. ヘルパー関数
# ==========================================
def load_model_safely(model_obj, base_name, target_type):
    """複数のロード候補からモデルを安全に読み込む"""
    import torch
    candidates = [
        f"{base_name}_refinement_{target_type}.pt",
        f"{base_name}_initial_{target_type}.pt",
        f"{base_name}_{target_type}.pt"
    ]
    
    for path in candidates:
        if os.path.exists(path):
            try:
                model_obj.load_state_dict(torch.load(path, map_location="cpu"))
                model_obj.eval()
                return path
            except Exception:
                continue
    return None

# ==========================================
# 4. 環境作成用ファクトリ (定義を完全に展開)
# ==========================================
def create_env(render_mode=None):
    """Linux GPUハング対策のため、主要ライブラリのロードをワーカー起動まで遅延させる"""
    import torch
    import mujoco
    import gymnasium as gym
    import main18_optimization as base_config
    from main18_optimization import Agent

    class TeamCosEnv(base_config.HideAndSeekEnv):
        def __init__(self, render_mode=None):
            super().__init__(render_mode=render_mode)
            cpu_dev = torch.device("cpu")
            
            # 各エージェントの観測バッファ
            self.npc_obs_history = {
                0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev), 
                1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev), 
                2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
            }
            
            # キャッシュ類
            self.visible_cache = {0: {}, 1: {}, 2: {}}
            self.lidar_array_cache = {} 
            self.raycast_cache = {} 
            self.raycast_stats = {"hits": 0, "misses": 0}
            
            # 統計
            self.hidden_steps_count = 0
            self.caught_steps_count = 0 
            self.prev_dist = {1: 0.0, 2: 0.0}
            self._obs_memo = {}
            self.s0_recovery_turn_dir = 1.0

            # NPCモデルのロード
            self.npc_hider_agent = Agent(53, 4).to("cpu")
            self.npc_seeker_agent = Agent(53, 4).to("cpu")
            
            should_print = (os.environ.get("NPC_MODELS_LOGGED") != "TRUE")

            h_path = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
            if h_path:
                if should_print:
                    print(f"Loaded NPC Hider from {h_path}", flush=True)
            else:
                self.npc_hider_agent = None

            s_path = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
            if s_path:
                if should_print:
                    print(f"Loaded NPC Seeker from {s_path}", flush=True)
            else:
                self.npc_seeker_agent = None

            if should_print:
                os.environ["NPC_MODELS_LOGGED"] = "TRUE"

        def _check_collision_all(self, pos, threshold):
            """指定座標が障害物と衝突しているか判定"""
            # 壁との衝突
            for w_pos, w_size in self.wall_data:
                dx = abs(pos[0] - w_pos[0]) - w_size[0]
                dy = abs(pos[1] - w_pos[1]) - w_size[1]
                dx_clamped = max(dx, 0.0)
                dy_clamped = max(dy, 0.0)
                if (dx_clamped**2 + dy_clamped**2) < threshold**2:
                    return True
            
            # 箱とスロープとの衝突
            for wid in self.box_geoms + self.ramp_all_geoms:
                center = self.data.geom_xpos[wid][:2]
                size = self.model.geom(wid).size[:2]
                dx_obj = abs(pos[0] - center[0]) - size[0]
                dy_obj = abs(pos[1] - center[1]) - size[1]
                dx_obj_clamped = max(dx_obj, 0.0)
                dy_obj_clamped = max(dy_obj, 0.0)
                if (dx_obj_clamped**2 + dy_obj_clamped**2) < threshold**2:
                    return True
            
            return False

        def _get_cached_ray(self, agent_id, origin_p, direction, beam_id):
            """RayCast結果の座標ベースキャッシュ"""
            angle = np.arctan2(direction[1], direction[0])
            cache_key = (agent_id, beam_id)
            
            if cache_key in self.raycast_cache:
                c_hp, c_a, c_res, c_gid = self.raycast_cache[cache_key]
                pos_diff = np.linalg.norm(origin_p - c_hp)
                if pos_diff < RAYCAST_CACHE_POS_THRESH:
                    angle_diff = (angle - c_a + np.pi) % (2 * np.pi) - np.pi
                    if abs(angle_diff) < 0.05:
                        self.raycast_stats["hits"] += 1
                        return c_res, c_gid
            
            # キャッシュミス
            self.raycast_stats["misses"] += 1
            gid = np.zeros(1, dtype=np.int32)
            from_p = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
            dir_3d = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
            
            if agent_id == 0:
                exclude = self.s0_body
            elif agent_id == 1:
                exclude = self.h1_body
            else:
                exclude = self.h2_body
                
            res = mujoco.mj_ray(self.model, self.data, from_p, dir_3d, None, 1, exclude, gid)
            self.raycast_cache[cache_key] = (origin_p.copy(), angle, res, gid[0])
            return res, gid[0]

        def _get_obs(self, agent_id):
            """53次元観測の生成 (完全展開版)"""
            if agent_id in self._obs_memo:
                return self._obs_memo[agent_id]
                
            if agent_id == 0:
                b_id = self.s0_body
                p_pref = 's'
            elif agent_id == 1:
                b_id = self.h1_body
                p_pref = 'h1'
            else:
                b_id = self.h2_body
                p_pref = 'h2'
            
            hp = self.data.xpos[b_id][:2]
            j_id = self.model.joint(f'{p_pref}_rot').id
            hra = self.data.qpos[self.model.jnt_qposadr[j_id]]
            
            c_val = np.cos(-hra)
            s_val = np.sin(-hra)
            rot_mat = np.array([[c_val, -s_val], [s_val, c_val]])
            
            jx_id = self.model.joint(f'{p_pref}_x').id
            dof_adr = self.model.jnt_dofadr[jx_id]
            h_raw_vel = self.data.qvel[dof_adr : dof_adr + 2]
            h_local_vel = rot_mat @ h_raw_vel
            h_obs_vel = h_local_vel / 12.0
            
            self_state = np.concatenate([h_obs_vel, [hra, np.cos(hra), np.sin(hra)]])
            
            # Lidar一括キャッシュ
            lidar_data = None
            l_cache = self.lidar_array_cache.get(agent_id)
            if l_cache is not None:
                c_hp, c_hra, c_lidar = l_cache
                if np.linalg.norm(hp - c_hp) < LIDAR_CACHE_POS_THRESH:
                    angle_diff = (hra - c_hra + np.pi) % (2 * np.pi) - np.pi
                    if abs(angle_diff) < LIDAR_CACHE_ANG_THRESH:
                        lidar_data = c_lidar
            
            if lidar_data is None:
                lidar_data = np.zeros(len(self.lidar_angles), dtype=np.float32)
                for i, angle_offset in enumerate(self.lidar_angles):
                    beam_dir = angle_offset + hra
                    direction = np.array([np.cos(beam_dir), np.sin(beam_dir)])
                    dist, _ = self._get_cached_ray(agent_id, hp, direction, i + 100)
                    if dist != -1:
                        lidar_data[i] = min(dist, 2.5) / 2.5
                    else:
                        lidar_data[i] = 1.0
                self.lidar_array_cache[agent_id] = (hp.copy(), hra, lidar_data.copy())

            # 視界判定
            my_vis = self.visible_cache[agent_id]
            my_vis.clear()
            targets = [self.box1_body, self.box2_body, self.ramp_body, self.h1_body, self.h2_body, self.s0_body]
            targets = [t for t in targets if t != b_id]
            
            for tid in targets:
                tp_pos = self.data.xpos[tid]
                diff = tp_pos[:2] - hp
                dist_obj = np.linalg.norm(diff)
                angle_obj = (np.arctan2(diff[1], diff[0]) - hra + np.pi) % (2 * np.pi) - np.pi
                
                if abs(angle_obj) > np.deg2rad(FOV_DEG / 2.0):
                    my_vis[tid] = False
                    continue
                
                ray_dir = diff / (dist_obj + 1e-8)
                res, hit_gid = self._get_cached_ray(agent_id, hp, ray_dir, tid)
                hit_body = self.model.geom_bodyid[hit_gid] if res != -1 else -1
                
                is_vis = False
                if hit_body == tid:
                    is_vis = True
                elif res != -1 and res > dist_obj - 0.4:
                    is_vis = True
                my_vis[tid] = is_vis

            def get_rel_info(target_id, lock=None):
                if my_vis.get(target_id, False):
                    t_pos = self.data.xpos[target_id]
                    rel_p = rot_mat @ (t_pos[:2] - hp) / 12.0
                    q_rot = self.data.xquat[target_id]
                    yaw = np.arctan2(2 * (q_rot[0]*q_rot[3] + q_rot[1]*q_rot[2]), 1 - 2 * (q_rot[2]**2 + q_rot[3]**2))
                    
                    b_jnt = self.model.body_jntadr[target_id]
                    if b_jnt != -1:
                        t_vel = self.data.qvel[b_jnt : b_jnt + 2]
                    else:
                        t_vel = np.zeros(2)
                    
                    rel_v = rot_mat @ (t_vel - h_raw_vel) / 12.0
                    info_list = [rel_p, rel_v, [np.cos(yaw - hra), np.sin(yaw - hra)]]
                    if lock is not None:
                        val = 1.0 if lock else 0.0
                        info_list.append([val])
                    
                    info_list.append([1.0])
                    return np.concatenate(info_list)
                
                # 不可視時
                length = 8 if lock is not None else 7
                return np.zeros(length, dtype=np.float32)

            if agent_id == 0:
                h1_rel = get_rel_info(self.h1_body)[:5]
                h2_rel = get_rel_info(self.h2_body)[:5]
                obj_list = [
                    get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    get_rel_info(self.ramp_body)
                ]
                obs_out = np.concatenate([self_state, lidar_data, *obj_list, h1_rel, h2_rel, np.zeros(3, dtype=np.float32)])
            else:
                partner_id = self.h2_body if agent_id == 1 else self.h1_body
                enemy_rel = get_rel_info(self.s0_body)[:5]
                friend_rel = get_rel_info(partner_id)
                grasp_val = 1.0 if self.grasping[agent_id] else 0.0
                obj_list = [
                    get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    get_rel_info(self.ramp_body)
                ]
                obs_out = np.concatenate([self_state, lidar_data, *obj_list, enemy_rel, friend_rel, [grasp_val]])

            obs_final = obs_out.astype(np.float32)
            self._obs_memo[agent_id] = obs_final
            return obs_final

        def _update_seeker_state(self):
            """NPCシーカーの目標決定ロジック"""
            sp_pos = self.data.xpos[self.s0_body][:2]
            
            # 視界判定の更新
            self._get_obs(0)
            v1 = self.visible_cache[0].get(self.h1_body, False)
            v2 = self.visible_cache[0].get(self.h2_body, False)
            
            if v1 or v2:
                # 目撃時
                target_bid = self.h1_body if v1 else self.h2_body
                self.seeker_target_pos = self.data.xpos[target_bid][:2].copy()
                self.seeker_last_known_pos = self.seeker_target_pos.copy()
                self.seeker_mode = "CHASING"
            elif self.seeker_last_known_pos is not None:
                # 見失った後の記憶追跡
                dist_to_mem = np.linalg.norm(sp_pos - self.seeker_last_known_pos)
                if dist_to_mem > 0.5:
                    self.seeker_target_pos = self.seeker_last_known_pos.copy()
                    self.seeker_mode = "SEARCHING"
                else:
                    # 到着したが居なかった
                    self.seeker_last_known_pos = None
                    self.seeker_search_timer = 50
            else:
                # 巡回
                if self.seeker_search_timer <= 0:
                    self.seeker_random_target = self.np_random.uniform(-4, 4, 2)
                    self.seeker_search_timer = 80
                
                self.seeker_search_timer -= 1
                self.seeker_target_pos = self.seeker_random_target.copy()
                self.seeker_mode = "PATROLLING"

        def _seeker_rule_based_policy(self):
            """NPCシーカーの移動制御 (障害物回避AI)"""
            if self.current_step < PREP_STEPS:
                return 0.0, 0.0
            
            sp_pos = self.data.xpos[self.s0_body][:2]
            sr_val = self.data.qpos[self.srot_adr]
            target_pos = self.seeker_target_pos
            
            # --- 障害物回避ポテンシャル法 (ランダムゆらぎ付き) ---
            diff_vec = target_pos - sp_pos
            target_ang = np.arctan2(diff_vec[1], diff_vec[0])
            best_angle = target_ang
            max_score = -1e9
            
            # 32方向スキャン
            for scan_ang in np.linspace(0, 2 * np.pi, 32):
                # 目的地への向きスコア
                score = np.cos(scan_ang - target_ang) * 10.0
                # ジッター追加
                score += self.np_random.uniform(-0.8, 0.8)
                
                # 障害物チェック
                dist_clear = 1.5
                cos_s = np.cos(scan_ang)
                sin_s = np.sin(scan_ang)
                
                for ds in np.arange(0.2, 1.6, 0.2):
                    check_p = sp_pos + np.array([cos_s, sin_s]) * ds
                    # 衝突判定関数の呼び出し
                    if self._check_collision_all(check_p, 0.45):
                        dist_clear = ds
                        score -= 200.0 
                        break
                
                # 開放スペース評価
                score += (dist_clear * 4.0)
                
                if score > max_score:
                    max_score = score
                    best_angle = scan_ang
            
            # 制御出力
            angle_diff = (best_angle - sr_val + np.pi) % (2 * np.pi) - np.pi
            thrust = SEEKER_RB_THRUST
            turn = np.clip(angle_diff * 6.0, -3.0, 3.0)
            
            if abs(angle_diff) > SEEKER_RB_TURN_THRESH:
                thrust *= 0.3
            
            # スタック判定
            sx_id = self.model.joint('s_x').id
            sx_dof = self.model.jnt_dofadr[sx_id]
            s_vel = np.linalg.norm(self.data.qvel[sx_dof : sx_dof + 2])
            
            if thrust > 0.05 and s_vel < 0.05:
                self.s0_stuck_timer += 5
            else:
                self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            
            if self.s0_stuck_timer > 15:
                self.s0_recovery_mode = 15
                self.s0_stuck_timer = 0
                self.s0_recovery_turn_dir = self.np_random.choice([-1.0, 1.0])
            
            if self.s0_recovery_mode > 0:
                thrust = -0.2
                turn = 1.5 * self.s0_recovery_turn_dir
                self.s0_recovery_mode -= 1
                
            return float(thrust), float(turn)

        def reset(self, seed=None, options=None):
            obs, info = super().reset(seed=seed, options=options)
            self.hidden_steps_count = 0
            self.caught_steps_count = 0
            self._obs_memo.clear()
            self.lidar_array_cache.clear()
            self.s0_recovery_turn_dir = 1.0
            
            sp_pos = self.data.xpos[self.s0_body][:2]
            for agent_idx in [1, 2]:
                bid = self.h1_body if agent_idx == 1 else self.h2_body
                self.prev_dist[agent_idx] = np.linalg.norm(self.data.xpos[bid][:2] - sp_pos)
            
            return obs, info

        def step(self, action):
            """1ステップの進行"""
            self._obs_memo.clear()
            self.current_step += 1
            
            for agent_idx in [1, 2]:
                val = self.lock_cooldown[agent_idx] - 1
                self.lock_cooldown[agent_idx] = max(0, val)
            
            self._update_seeker_state()
            self.data.ctrl[:] = 0.0 
            
            if TRAIN_TARGET == "HIDER":
                # 学習エージェント
                m_idx = self._apply_action(self.learning_agent_id, action)
                self.data.ctrl[m_idx] = float(action[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[m_idx + 1] = float(action[1])
                
                # 相棒NPC
                partner_id = 2 if self.learning_agent_id == 1 else 1
                act_p = self._get_npc_action(partner_id, "HIDER")
                p_idx = self._apply_action(partner_id, act_p)
                self.data.ctrl[p_idx] = float(act_p[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[p_idx + 1] = float(act_p[1])
                
                # 敵シーカーNPC
                sf_s, sr_s = self._seeker_rule_based_policy()
                self.data.ctrl[0] = sf_s
                self.data.ctrl[1] = sr_s
            else:
                # 学習シーカー
                self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(action[1])
                
                # 敵Hider NPCs
                for agent_idx in [1, 2]:
                    act_n = self._get_npc_action(agent_idx, "HIDER")
                    n_idx = self._apply_action(agent_idx, act_n)
                    self.data.ctrl[n_idx] = float(act_n[0]) * HIDER_THRUST_LIMIT
                    self.data.ctrl[n_idx + 1] = float(act_n[1])
            
            # 物理ループ (Action Repeat)
            for _ in range(ACTION_REPEAT):
                for box_bid, box_pose in self.locked_pose.items():
                    if self.locked_boxes[box_bid]:
                        if box_bid == self.box1_body:
                            jid = self.box1_joint_id
                        else:
                            jid = self.box2_joint_id
                        
                        q_a = self.model.jnt_qposadr[jid]
                        d_a = self.model.jnt_dofadr[jid]
                        self.data.qpos[q_a : q_a + 7] = box_pose
                        self.data.qvel[d_a : d_a + 6] = 0
                
                mujoco.mj_step(self.model, self.data)
            
            # 更新後の観測
            self._obs_memo.clear()
            obs_learner = self._get_obs(self.learning_agent_id)
            self._get_obs(0) 
            
            team_reward = 0.0
            if self.learning_agent_id == 1:
                l_body_id = self.h1_body
            else:
                l_body_id = self.h2_body
            
            iv = self.visible_cache[0].get(l_body_id, False)
            iav = any(self.visible_cache[0].get(bid, False) for bid in [self.h1_body, self.h2_body])
            
            if not iv:
                self.hidden_steps_count += 1
            if iav:
                self.caught_steps_count += 1
                
            for hid_idx, body_id in [(1, self.h1_body), (2, self.h2_body)]:
                is_vis = self.visible_cache[0].get(body_id, False)
                sp_p = self.data.xpos[self.s0_body][:2]
                cur_d = np.linalg.norm(self.data.xpos[body_id][:2] - sp_p)
                
                if is_vis:
                    sr_pos = self.data.qpos[self.srot_adr]
                    dv = self.data.xpos[body_id][:2] - sp_p
                    dv_norm = dv / (np.linalg.norm(dv) + 1e-8)
                    cos_v = np.dot(dv_norm, np.array([np.cos(sr_pos), np.sin(sr_pos)]))
                    h_rew = -cos_v * COS_PENALTY_SCALE + (cur_d - self.prev_dist[hid_idx]) * REWARD_DISTANCE_DIFF_SCALE
                else:
                    h_rew = REWARD_HIDDEN_BONUS
                
                if max(abs(self.data.xpos[body_id][:2])) > 6.5:
                    h_rew += PENALTY_SAFEGUARD
                
                team_reward += h_rew
                self.prev_dist[hid_idx] = cur_d
                
            final_reward = team_reward if TRAIN_TARGET == "HIDER" else -team_reward
            truncated = (self.current_step >= MAX_STEPS)
            
            s_info = {
                "hidden_steps": float(self.hidden_steps_count), 
                "caught_steps": float(self.caught_steps_count)
            }
            return obs_learner, float(final_reward), False, truncated, s_info

        def _get_npc_action(self, agent_id, agent_type):
            """NPC自律行動の生成"""
            import torch
            obs = self._get_obs(agent_id)
            self.npc_obs_history[agent_id].update(obs)
            
            if agent_type == "HIDER":
                model = self.npc_hider_agent
            else:
                model = self.npc_seeker_agent
                
            if model is not None:
                with torch.no_grad():
                    obs_seq = self.npc_obs_history[agent_id].get()
                    act, _, _, _ = model.get_action_and_value(obs_seq)
                return act.cpu().numpy()[0]
            
            # フォールバック
            return self.action_space.sample() * 0.5

    return TeamCosEnv(render_mode=render_mode)

# ==========================================
# 5. メイン処理 (学習ループ)
# ==========================================
def main():
    # Linux環境のデッドロック防止
    if platform.system() == "Linux":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # 並列ワーカーの起動
    def env_factory():
        env_instance = create_env()
        import gymnasium as gym
        return gym.wrappers.RecordEpisodeStatistics(env_instance)
    
    import gymnasium as gym
    print(f"--- [Parent] 1. Initializing {NUM_ENVS} parallel workers ---", flush=True)
    
    try:
        vec_envs = gym.vector.AsyncVectorEnv([env_factory for _ in range(NUM_ENVS)])
        print("--- [Parent] 2. Workers initialized successfully ---", flush=True)
    except Exception as e:
        print(f"--- [Parent] [CRITICAL] Parallel startup failed: {e} ---", flush=True)
        sys.exit(1)

    # ワーカー起動後に親プロセスでのライブラリ初期化
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb
    import main18_optimization as base_config
    from main18_optimization import Agent

    device = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")
    run_ts = int(time.time())
    run_name = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{run_ts}"
    
    # PLAYモード処理
    if EXECUTION_MODE == "PLAY":
        print(f"--- Inference Mode (PLAY) ---")
        env = create_env(render_mode="human")
        o_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]
        agent = Agent(o_dim, a_dim).to(device)
        lp_path = load_model_safely(agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if lp_path:
            print(f"Loaded: {lp_path}")
        
        agent.eval()
        o_hist = ObsHistory(1, TRANSFORMER_SEQ_LEN, o_dim, device)
        s_dur = 0.005 * ACTION_REPEAT
        
        try:
            while True:
                next_obs, _ = env.reset()
                o_hist.reset()
                o_hist.update(next_obs)
                done = False
                total_rew = 0.0
                while not done:
                    loop_start = time.time()
                    with torch.no_grad():
                        action, _, _, _ = agent.get_action_and_value(o_hist.get())
                    
                    obs_arr, reward, term, trunc, info = env.step(action.cpu().numpy()[0])
                    done = term or trunc
                    total_rew += reward
                    o_hist.update(obs_arr)
                    env.render()
                    
                    process_time = time.time() - loop_start
                    wait = s_dur - process_time
                    if wait > 0:
                        time.sleep(wait)
                    
                    if env.viewer is not None and not env.viewer.is_running():
                        return
                
                print(f"EpRet: {total_rew:.1f}, Hidden: {info['hidden_steps']:.0f}")
        except KeyboardInterrupt:
            pass
        finally:
            env.close()
        return

    # TRAINモード
    if TRACK_WANDB:
        run = wandb.init(
            project=base_config.WANDB_PROJECT_NAME, 
            config={"Target": TRAIN_TARGET, "MODE": MODE, "v": "25.14_expanded"}, 
            name=run_name, 
            sync_tensorboard=False,
            save_code=True
        )
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")

    writer = SummaryWriter(f"runs/{run_name}")
    train_agent = Agent(vec_envs.single_observation_space.shape[0], vec_envs.single_action_space.shape[0]).to(device)
    optimizer = optim.Adam(train_agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    global_step = 0
    start_global_step = 0
    if LOAD_EXISTING_MODELS:
        lp = load_model_safely(train_agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if lp:
            print(f"★ Resumed from: {lp}")
            cp_path = lp.replace('.pt', '_checkpoint.json')
            if os.path.exists(cp_path):
                try:
                    with open(cp_path, 'r') as f_cp:
                        cp_data = json.load(f_cp)
                        global_step = cp_data.get('global_step', 0)
                        start_global_step = global_step
                except Exception:
                    pass

    # バッファ確保
    train_obs_hist = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device)
    S, E, O, A = NUM_STEPS, NUM_ENVS, 53, 4
    obs_b = torch.zeros((S, E, TRANSFORMER_SEQ_LEN, O), device=device)
    actions_b = torch.zeros((S, E, A), device=device)
    logprobs_b = torch.zeros((S, E), device=device)
    rewards_b = torch.zeros((S, E), device=device)
    dones_b = torch.zeros((S, E), device=device)
    values_b = torch.zeros((S, E), device=device)
    
    c_seed = FIXED_SEED if FIXED_SEED else int(time.time())
    next_obs, _ = vec_envs.reset(seed=c_seed)
    next_done = torch.zeros(E).to(device)
    train_obs_hist.reset()
    train_obs_hist.update(next_obs)
    
    num_updates = int(max(1, (TOTAL_TIMESTEPS - global_step) // (E * S)))
    history_returns, history_hidden, history_caught = [], [], []
    start_time = time.time()
    last_loss, last_entropy = 0.0, 0.0

    print(f"--- Training started ---")
    try:
        for update_idx in tqdm(range(1, num_updates + 1), desc="Updates"):
            # データ収集 (Rollout)
            for step_idx in range(S):
                global_step += E
                obs_b[step_idx] = train_obs_hist.get()
                dones_b[step_idx] = next_done
                
                with torch.no_grad():
                    act_batch, lp_batch, _, val_batch = train_agent.get_action_and_value(train_obs_hist.get())
                    values_b[step_idx] = val_batch.flatten()
                
                actions_b[step_idx] = act_batch
                logprobs_b[step_idx] = lp_batch
                
                next_obs, reward_vec, term_vec, trunc_vec, info_dict = vec_envs.step(act_batch.cpu().numpy())
                done_vec = np.logical_or(term_vec, trunc_vec)
                
                if "final_info" in info_dict:
                    for env_idx in range(E):
                        info_item = info_dict["final_info"][env_idx]
                        if done_vec[env_idx] and info_item is not None:
                            if "episode" in info_item:
                                history_returns.append(float(info_item["episode"]["r"]))
                            if "hidden_steps" in info_item:
                                history_hidden.append(float(info_item["hidden_steps"]))
                            if "caught_steps" in info_item:
                                history_caught.append(float(info_item["caught_steps"]))
                elif "episode" in info_dict:
                    mask = info_dict.get("_episode", [True] * E)
                    for env_idx in range(E):
                        if mask[env_idx] and done_vec[env_idx]:
                            history_returns.append(float(info_dict["episode"]["r"][env_idx]))
                            try:
                                if "hidden_steps" in info_dict:
                                    val_h = info_dict["hidden_steps"][env_idx]
                                    history_hidden.append(float(val_h))
                                if "caught_steps" in info_dict:
                                    val_c = info_dict["caught_steps"][env_idx]
                                    history_caught.append(float(val_c))
                            except Exception:
                                pass
                
                rewards_b[step_idx] = torch.tensor(reward_vec).to(device).view(-1)
                next_done = torch.tensor(done_vec).to(device, dtype=torch.float32)
                train_obs_hist.update(next_obs)

            # アドバンテージ計算
            with torch.no_grad():
                next_value = train_agent.get_value(train_obs_hist.get()).reshape(1, -1)
                advantages = torch.zeros_like(rewards_b).to(device)
                lastgaelam = 0
                for t in reversed(range(S)):
                    if t == S - 1:
                        nt = 1.0 - next_done
                        v_post = next_value
                    else:
                        nt = 1.0 - dones_b[t+1]
                        v_post = values_b[t+1]
                    
                    delta = rewards_b[t] + base_config.GAMMA * v_post * nt - values_b[t]
                    advantages[t] = lastgaelam = delta + base_config.GAMMA * base_config.GAE_LAMBDA * nt * lastgaelam
                
                returns = advantages + values_b

            # 最適化
            f_obs = obs_b.reshape((-1, TRANSFORMER_SEQ_LEN, O))
            f_lp = logprobs_b.reshape(-1)
            f_act = actions_b.reshape((-1, A))
            f_adv = advantages.reshape(-1)
            f_ret = returns.reshape(-1)
            
            for epoch_idx in range(UPDATE_EPOCHS):
                inds = np.arange(S * E)
                np.random.shuffle(inds)
                for s_ptr in range(0, S * E, MINIBATCH_SIZE):
                    mb_idx = inds[s_ptr : s_ptr + MINIBATCH_SIZE]
                    _, n_lp, ent, n_v = train_agent.get_action_and_value(f_obs[mb_idx], f_act[mb_idx])
                    
                    ratio = (n_lp - f_lp[mb_idx]).exp()
                    mb_adv_vals = f_adv[mb_idx]
                    mb_adv_norm = (mb_adv_vals - mb_adv_vals.mean()) / (mb_adv_vals.std() + 1e-8)
                    
                    pg_l1 = -mb_adv_norm * ratio
                    pg_l2 = -mb_adv_norm * torch.clamp(ratio, 0.8, 1.2)
                    pg_l = torch.max(pg_l1, pg_l2).mean()
                    
                    v_l = 0.5 * ((n_v.view(-1) - f_ret[mb_idx]) ** 2).mean()
                    loss = pg_l - ENT_COEF * ent.mean() + 0.5 * v_l
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(train_agent.parameters(), 0.5)
                    optimizer.step()
                    
                    last_loss = loss.item()
                    last_entropy = ent.mean().item()

            if (TRIAL_MODE) or (update_idx % 10 == 0):
                dt = time.time() - start_time
                sps = int((global_step - start_global_step) / dt) if dt > 0 else 0
                log_map = {"charts/SPS": sps, "losses/total_loss": last_loss, "losses/entropy": last_entropy, "global_step": global_step}
                if history_returns:
                    avg_h = np.mean(history_hidden)
                    avg_c = np.mean(history_caught)
                    avg_r = np.mean(history_returns)
                    log_map.update({"charts/episodic_return": avg_r, "charts/steps_hidden": avg_h, "charts/steps_caught": avg_c})
                    print(f"Update {update_idx}, Step {global_step}, SPS: {sps}, EpRet: {avg_r:.1f}, Hidden: {avg_h:.1f}, Caught: {avg_c:.1f}", flush=True)
                    history_returns, history_hidden, history_caught = [], [], []
                elif not TRIAL_MODE:
                    print(f"Update {update_idx}, Step {global_step}, SPS: {sps} (Collecting stats...)", flush=True)
                
                if TRACK_WANDB:
                    wandb.log(log_map)
                writer.add_scalar("charts/SPS", sps, global_step)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    if SAVE_MODEL:
        torch.save(train_agent.state_dict(), SAVE_MODEL_PATH)
        with open(SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json'), 'w') as f_s:
            json.dump({'global_step': global_step}, f_s)
        print(f"Model saved to {SAVE_MODEL_PATH}")
    
    vec_envs.close()
    writer.close()
    if TRACK_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()