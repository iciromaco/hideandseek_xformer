# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【概要】
# 2体の Hider と 1体の Seeker による、マルチエージェントかくれんぼ環境の強化学習スクリプト。
# Transformer 搭載 Actor-Critic (PPO) アルゴリズムを用い、共進化を目指します。
# 
# 【修正内容 (v25.47 - 構文エラー根絶・SPS回復・統計完全同期版)】
# 1. 構文エラーの物理的排除:
#    - 全てのセミコロン (;) を完全に削除。
#    - ご指摘の終了処理（envs_vec.close()等）を含め、全てのステートメントを独立行に展開。
# 2. 統計収集ロジックの堅牢化:
#    - ユーザー様のご指摘に基づき、episode 辞書と hidden_steps 配列から同期してデータを抽出。
#    - これにより Hidden が nan になる問題を解決し、Optuna への正確なフィードバックを保証。
# 3. 冗長処理の排除による SPS 回復:
#    - キャッシュ判定や停滞ペナルティ等の頻回計算において、平方根を避けた二乗距離比較を採用。
#    - _get_obs 内部でのローカル関数定義を廃止し、インライン処理に統合。
# 4. 詳細な日本語コメント:
#    - 各工程の実装意図と物理的意味を詳しく解説。

import os
import sys
import platform
import json
import time
import math
import numpy as np
import multiprocessing
import mujoco
import gymnasium as gym
from tqdm import tqdm

# --- 実行環境の最適化 ---
# 数値計算ライブラリのスレッド競合を抑制し、マルチプロセス時の実行効率を高めます。
if platform.processor() != 'arm':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# インポートパスの解決
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
# 1. 実験設定 (Configuration)
# ==========================================

# "initial": 新規学習 / "refinement": 既存モデルをロードして微調整
MODE = "initial" # "refinement" 

# "TRAIN": 並列環境で学習を実行 / "PLAY": 1環境での描画確認
EXECUTION_MODE = "TRAIN" 

# "HIDER": Hiderチームを学習 / "SEEKER": Seekerを学習
TRAIN_TARGET = "HIDER" 

EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"
EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"

# モデルのロード判定
LOAD_EXISTING_MODELS = False
if MODE == "refinement":
    LOAD_EXISTING_MODELS = True

# 実験フラグ
SAVE_MODEL = True
TRACK_WANDB = True
FIXED_SEED = None
TRIAL_MODE = False

# 強化学習 (PPO) パラメータ
TOTAL_TIMESTEPS = 5000000 
NUM_ENVS = 8
NUM_STEPS = 128
LEARNING_RATE = 2e-4
ENT_COEF = 0.001
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4

# 環境・シミュレーション設定
ACTION_REPEAT = 16
PREP_STEPS = 80
MAX_STEPS = 300
FOV_DEG = 135 
FOV_RAD_HALF = math.radians(FOV_DEG / 2.0)
TRANSFORMER_SEQ_LEN = 8

# 高速化キャッシュ用定数 (計算負荷の低い二乗値を保持)
RAYCAST_GRID_SIZE = 0.1
LIDAR_CACHE_POS_THRESH_SQ = 0.05**2 
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)

# 報酬設計
REWARD_HIDDEN_BONUS = 1.0
COS_PENALTY_SCALE = 2.0
REWARD_DISTANCE_DIFF_SCALE = 1.0
PENALTY_SAFEGUARD = -20.0

# 移動出力制限
HIDER_THRUST_LIMIT = 0.40  
SEEKER_THRUST_LIMIT = 0.35 
SEEKER_RB_THRUST = 0.38 
SEEKER_RB_TURN_THRESH = math.pi / 6 

SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# ==========================================
# 2. 高速化データ構造 (ObsHistory)
# ==========================================
class ObsHistory:
    """
    Transformer への入力を管理するゼロコピー履歴バッファ。
    ミラーリングにより、読み出し時の並べ替え（Cat/Roll）を排除します。
    """
    def __init__(self, num_envs, seq_len, obs_dim, device):
        import torch
        self.buffer = torch.zeros((num_envs, seq_len * 2, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.ptr = 0

    def reset(self):
        """バッファを初期化します。"""
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        """最新の観測値を 2 箇所に書き込みます。"""
        import torch
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # 現在地点とミラー地点に書き込み
        self.buffer[:, self.ptr] = obs_tensor
        mirrored_idx = self.ptr + self.seq_len
        self.buffer[:, mirrored_idx] = obs_tensor
        
        # ポインタを進める
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        """連続したメモリ領域として直近履歴を高速にスライス取得します。"""
        end_idx = self.ptr + self.seq_len
        return self.buffer[:, self.ptr : end_idx]

def load_model_safely(model_obj, base_name, target_type):
    """
    指定されたターゲットに合致するモデルファイルを検索してロードします。
    """
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
# 3. 環境クラス (TeamCosEnv)
# ==========================================
import main18_optimization as base_config

class TeamCosEnv(base_config.HideAndSeekEnv):
    """
    視界勾配報酬とポテンシャル法 AI を搭載した強化版環境。
    1行1ステートメントを徹底し、 redundant な処理を排除しています。
    """
    def __init__(self, render_mode=None):
        # 親クラス（物理環境）の初期化
        super().__init__(render_mode=render_mode)
        
        import torch
        from main18_optimization import Agent
        cpu_dev = torch.device("cpu")
        
        # NPCエージェント用の履歴管理
        self.npc_obs_history = {}
        self.npc_obs_history[0] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
        self.npc_obs_history[1] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
        self.npc_obs_history[2] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
        
        # 各種計算キャッシュ
        self.visible_cache = {0: {}, 1: {}, 2: {}}
        self.lidar_array_cache = {} 
        self.raycast_cache = {} 
        self._obs_memo = {}
        
        # 物理統計情報
        self.hidden_steps_count = 0
        self.caught_steps_count = 0
        self.prev_dist = {1: 0.0, 2: 0.0}
        self.s0_recovery_turn_dir = 1.0
        self.two_pi = 2.0 * math.pi
        
        # Lidar用三角関数テーブル (事前に計算して実行時の負荷を軽減)
        angles_vec = self.lidar_angles
        self.lidar_cos_sin = np.column_stack([
            np.cos(angles_vec), 
            np.sin(angles_vec)
        ])

        # 学習ターゲットエージェントの決定
        if TRAIN_TARGET == "HIDER":
            self.learning_agent_id = self.np_random.choice([1, 2])
        else:
            self.learning_agent_id = 0
            
        self.lock_cooldown = {1: 0, 2: 0}
        self.raycast_stats = {"hits": 0, "misses": 0}
        
        # Seeker の状態変数
        self.s0_stuck_timer = 0
        self.s0_recovery_mode = 0
        self.seeker_search_timer = 0
        self.seeker_random_target = np.zeros(2)
        self.seeker_target_pos = np.zeros(2)
        self.seeker_last_known_pos = None
        
        # ベクトル演算用の ID リスト
        bodies_list = [
            self.box1_body, 
            self.box2_body, 
            self.ramp_body, 
            self.h1_body, 
            self.h2_body, 
            self.s0_body
        ]
        self.all_target_bodies = np.array(bodies_list)
        
        # ジョイントのアドレス取得
        rot_jnt = self.model.joint('s_rot')
        self.srot_adr = self.model.jnt_qposadr[rot_jnt.id]
        
        # NPC用モデルのロード
        self.npc_hider_agent = Agent(53, 4).to("cpu")
        self.npc_seeker_agent = Agent(53, 4).to("cpu")
        
        log_flag_key = os.environ.get("NPC_MODELS_LOGGED")
        should_print_log = (log_flag_key != "TRUE")
        
        h_file = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
        s_file = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
        
        if should_print_log:
            if h_file: 
                print(f"Loaded NPC Hider from {h_file}", flush=True)
            if s_file: 
                print(f"Loaded NPC Seeker from {s_file}", flush=True)
            os.environ["NPC_MODELS_LOGGED"] = "TRUE"

    def reset(self, seed=None, options=None):
        """エピソード開始時のリセット処理。"""
        obs, info = super().reset(seed=seed, options=options)
        
        self.hidden_steps_count = 0
        self.caught_steps_count = 0
        self._obs_memo.clear()
        self.lidar_array_cache.clear()
        self.raycast_cache.clear()
        
        # 報酬基準となる初期距離を保存
        sp_pos = self.data.xpos[self.s0_body][:2]
        for i in [1, 2]:
            tid = self.h1_body
            if i == 2:
                tid = self.h2_body
            hp_pos = self.data.xpos[tid][:2]
            # 距離計算
            dist_val = np.linalg.norm(hp_pos - sp_pos)
            self.prev_dist[i] = dist_val
            
        return obs, info

    def _get_cached_ray(self, agent_id, origin_p, direction, beam_id):
        """レイキャスト結果を空間グリッド単位でキャッシュします。"""
        angle_rad = math.atan2(direction[1], direction[0])
        # 位置を 10cm グリッドに丸めてキャッシュキーを生成
        gx = int(origin_p[0] / RAYCAST_GRID_SIZE)
        gy = int(origin_p[1] / RAYCAST_GRID_SIZE)
        cache_key = (agent_id, gx, gy, beam_id)
        
        if cache_key in self.raycast_cache:
            c_ang, c_res, c_gid = self.raycast_cache[cache_key]
            # 向きの変化がわずかならキャッシュ採用
            a_diff = (angle_rad - c_ang + math.pi) % self.two_pi - math.pi
            if abs(a_diff) < 0.05:
                self.raycast_stats["hits"] += 1
                return c_res, c_gid

        # キャッシュミス
        self.raycast_stats["misses"] += 1
        gid_out = np.zeros(1, dtype=np.int32)
        f_xyz = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
        d_xyz = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        
        # 自己遮蔽の回避
        ex_id = self.s0_body
        if agent_id == 1:
            ex_id = self.h1_body
        elif agent_id == 2:
            ex_id = self.h2_body
        
        r_val = mujoco.mj_ray(self.model, self.data, f_xyz, d_xyz, None, 1, ex_id, gid_out)
        hit_id = gid_out[0]
        
        # キャッシュ登録
        self.raycast_cache[cache_key] = (angle_rad, r_val, hit_id)
        return r_val, hit_id

    def _get_obs(self, agent_id, skip_lidar=False):
        """観測ベクトル生成。行列演算により計算ステップを最小化しています。"""
        if agent_id in self._obs_memo: 
            return self._obs_memo[agent_id]
        
        # 対象特定
        if agent_id == 0:
            b_id = self.s0_body
            pref = 's'
        elif agent_id == 1:
            b_id = self.h1_body
            pref = 'h1'
        else:
            b_id = self.h2_body
            pref = 'h2'
            
        agent_xy = self.data.xpos[b_id][:2]
        rot_jnt = self.model.joint(f'{pref}_rot')
        hra = self.data.qpos[self.model.jnt_qposadr[rot_jnt.id]]
        
        cos_h = math.cos(hra)
        sin_h = math.sin(hra)
        # 自身の座標系への変換行列
        rot_mat = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
        
        xjnt = self.model.joint(f'{pref}_x')
        d_idx = self.model.jnt_dofadr[xjnt.id]
        v_raw = self.data.qvel[d_idx : d_idx + 2]
        v_obs = (rot_mat @ v_raw) / 12.0
        
        # 自己状態 (5次元)
        self_state = np.array([v_obs[0], v_obs[1], hra, cos_h, sin_h], dtype=np.float32)
        
        # Lidar (二乗距離比較により norm 計算の負荷を削減)
        lidar_data = np.zeros(len(self.lidar_angles), dtype=np.float32)
        if not skip_lidar:
            l_cache = self.lidar_array_cache.get(agent_id)
            if l_cache:
                c_xy, c_hra, c_vals = l_cache
                dx = agent_xy[0] - c_xy[0]
                dy = agent_xy[1] - c_xy[1]
                # 平方根をとらずに二乗和でしきい値を比較
                if (dx*dx + dy*dy) < LIDAR_CACHE_POS_THRESH_SQ:
                    da = (hra - c_hra + math.pi) % self.two_pi - math.pi
                    if abs(da) < LIDAR_CACHE_ANG_THRESH:
                        lidar_data = c_vals
            
            if l_cache is None or np.sum(lidar_data) == 0:
                for idx in range(len(self.lidar_angles)):
                    co = self.lidar_cos_sin[idx][0]
                    si = self.lidar_cos_sin[idx][1]
                    # 現在の向きを反映
                    bx = co * cos_h - si * sin_h
                    by = si * cos_h + co * sin_h
                    b_vec = np.array([bx, by])
                    dr, _ = self._get_cached_ray(agent_id, agent_xy, b_vec, idx + 100)
                    lidar_data[idx] = min(dr, 2.5) / 2.5 if dr != -1 else 1.0
                self.lidar_array_cache[agent_id] = (agent_xy.copy(), hra, lidar_data.copy())

        # オブジェクトベクトルの一括演算
        mask = (self.all_target_bodies != b_id)
        t_ids = self.all_target_bodies[mask]
        
        rel_pos_mat = self.data.xpos[t_ids][:, :2] - agent_xy
        dists = np.sqrt(np.sum(rel_pos_mat**2, axis=1))
        
        # FOV判定
        g_angs = np.arctan2(rel_pos_mat[:, 1], rel_pos_mat[:, 0])
        rel_angs = (g_angs - hra + math.pi) % self.two_pi - math.pi
        fov_mask = np.abs(rel_angs) <= FOV_RAD_HALF
        
        vis_dict = self.visible_cache[agent_id]
        vis_dict.clear()
        rel_info_store = {}
        
        for idx in range(len(t_ids)):
            tid = t_ids[idx]
            if not fov_mask[idx]:
                vis_dict[tid] = False
                continue
                
            u_dir = rel_pos_raw = rel_pos_mat[idx] / (dists[idx] + 1e-8)
            d_ray, h_gid = self._get_cached_ray(agent_id, agent_xy, u_dir, tid)
            
            hit_body = self.model.geom_bodyid[h_gid]
            is_vis = (hit_body == tid) or (d_ray != -1 and d_ray > dists[idx] - 0.4)
            vis_dict[tid] = is_vis
            
            if is_vis:
                p_loc = (rot_mat @ rel_pos_mat[idx]) / 12.0
                q = self.data.xquat[tid]
                y_val = 2.0 * (q[0] * q[3] + q[1] * q[2])
                x_val = 1.0 - 2.0 * (q[2]**2 + q[3]**2)
                yaw_rel = math.atan2(y_val, x_val) - hra
                
                j_adr = self.model.body_jntadr[tid]
                v_target = np.zeros(2)
                if j_adr != -1:
                    v_target = self.data.qvel[j_adr : j_adr + 2]
                
                rv_loc = (rot_mat @ (v_target - v_raw)) / 12.0
                rel_info_store[tid] = [
                    p_loc[0], p_loc[1], 
                    rv_loc[0], rv_loc[1], 
                    math.cos(yaw_rel), math.sin(yaw_rel)
                ]

        # 出力の結合 (関数オーバーヘッド削減)
        p_list = [self_state, lidar_data]
        
        if agent_id == 0:
            # シーカー視点
            order = [
                (self.box1_body, self.locked_boxes[self.box1_body]), 
                (self.box2_body, self.locked_boxes[self.box2_body]), 
                (self.ramp_body, None)
            ]
            for tid, lock in order:
                inf = rel_info_store.get(tid)
                if inf:
                    v = list(inf)
                    if lock is not None:
                        v.append(1.0 if lock else 0.0)
                    v.append(1.0)
                    p_list.append(np.array(v, dtype=np.float32))
                else:
                    p_list.append(np.zeros(8 if lock is not None else 7, dtype=np.float32))
            
            for tid in [self.h1_body, self.h2_body]:
                inf = rel_info_store.get(tid)
                if inf:
                    v = list(inf[:4])
                    v.append(1.0)
                    p_list.append(np.array(v, dtype=np.float32))
                else:
                    p_list.append(np.zeros(5, dtype=np.float32))
            p_list.append(np.zeros(3, dtype=np.float32))
        else:
            # ハイダー視点
            partner_id = self.h2_body if agent_id == 1 else self.h1_body
            for tid, lock in [(self.box1_body, self.locked_boxes[self.box1_body]), (self.box2_body, self.locked_boxes[self.box2_body]), (self.ramp_body, None)]:
                inf = rel_info_store.get(tid)
                if inf:
                    v = list(inf)
                    if lock is not None:
                        v.append(1.0 if lock else 0.0)
                    v.append(1.0)
                    p_list.append(np.array(v, dtype=np.float32))
                else:
                    p_list.append(np.zeros(8 if lock is not None else 7, dtype=np.float32))
            inf_en = rel_info_store.get(self.s0_body)
            p_list.append(np.array(list(inf_en[:4]) + [1.0], dtype=np.float32) if inf_en else np.zeros(5, dtype=np.float32))
            inf_fr = rel_info_store.get(partner_id)
            p_list.append(np.array(list(inf_fr) + [1.0], dtype=np.float32) if inf_fr else np.zeros(7, dtype=np.float32))
            p_list.append(np.array([1.0 if self.grasping[agent_id] else 0.0], dtype=np.float32))
            
        final_obs = np.concatenate(p_list).astype(np.float32)
        self._obs_memo[agent_id] = final_obs
        return final_obs

    def _update_seeker_state(self):
        """シーカーの AI 目標更新。視覚・記憶・巡回を管理します。"""
        sp_xy = self.data.xpos[self.s0_body][:2]
        self._get_obs(0)
        v_dict = self.visible_cache[0]
        v1 = v_dict.get(self.h1_body, False)
        v2 = v_dict.get(self.h2_body, False)
        
        if v1 or v2:
            target_bid = self.h1_body if v1 else self.h2_body
            t_pos = self.data.xpos[target_bid][:2]
            self.seeker_target_pos = t_pos.copy()
            self.seeker_last_known_pos = self.seeker_target_pos.copy()
            self.seeker_mode = "CHASING"
        elif self.seeker_last_known_pos is not None:
            dx = sp_xy[0] - self.seeker_last_known_pos[0]
            dy = sp_xy[1] - self.seeker_last_known_pos[1]
            if (dx*dx + dy*dy) < 0.25: 
                self.seeker_last_known_pos = None
                self.seeker_search_timer = 50
            else: 
                self.seeker_target_pos = self.seeker_last_known_pos.copy()
                self.seeker_mode = "SEARCHING"
        else:
            if self.seeker_search_timer <= 0:
                self.seeker_random_target = self.np_random.uniform(-4, 4, 2)
                self.seeker_search_timer = 80
            self.seeker_search_timer -= 1
            self.seeker_target_pos = self.seeker_random_target.copy()
            self.seeker_mode = "PATROLLING"

    def _seeker_rule_based_policy(self):
        """ポテンシャル法に基づいた物理制御。"""
        if self.current_step < PREP_STEPS:
            return 0.0, 0.0
            
        sp_xy = self.data.xpos[self.s0_body][:2]
        yaw = self.data.qpos[self.srot_adr]
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        
        l_vals = self.lidar_array_cache[0][2]
        repulsion = np.zeros(2)
        for i in range(len(self.lidar_angles)):
            d = l_vals[i] * 2.5
            if d < 1.0:
                f = (1.0 - d) / (d + 0.1)
                co = self.lidar_cos_sin[i][0]
                si = self.lidar_cos_sin[i][1]
                repulsion[0] -= (co * cy - si * sy) * f
                repulsion[1] -= (si * cy + co * sy) * f
                
        diff_v = self.seeker_target_pos - sp_xy
        dist_t = np.linalg.norm(diff_v)
        unit_t = diff_v / (dist_t + 1e-8)
        
        combined = unit_t + repulsion * 1.5
        target_a = math.atan2(combined[1], combined[0])
        err = (target_a - yaw + math.pi) % self.two_pi - math.pi
        
        thrust = SEEKER_RB_THRUST
        if abs(err) > SEEKER_RB_TURN_THRESH:
            thrust = thrust * 0.3
            
        sx_jnt = self.model.joint('s_x')
        d_adr = self.model.jnt_dofadr[sx_jnt.id]
        v_sq = self.data.qvel[d_adr]**2 + self.data.qvel[d_adr + 1]**2
        
        if thrust > 0.05 and v_sq < 0.0025:
            self.s0_stuck_timer += 5
        else:
            self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            
        if self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 15
            self.s0_stuck_timer = 0
            self.s0_recovery_turn_dir = self.np_random.choice([-1.0, 1.0])
            
        if self.s0_recovery_mode > 0:
            self.s0_recovery_mode -= 1
            return -0.2, 1.5 * self.s0_recovery_turn_dir
            
        return float(thrust), float(np.clip(err * 6.0, -3.0, 3.0))

    def step(self, action):
        """シミュレーション更新と報酬計算。"""
        self._obs_memo.clear()
        self.current_step += 1
        
        for idx in [1, 2]:
            self.lock_cooldown[idx] = max(0, self.lock_cooldown[idx] - 1)
            
        self._update_seeker_state()
        self.data.ctrl[:] = 0.0
        
        if TRAIN_TARGET == "HIDER":
            m_idx = self._apply_action(self.learning_agent_id, action)
            self.data.ctrl[m_idx] = float(action[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[m_idx + 1] = float(action[1])
            
            p_id = 2 if self.learning_agent_id == 1 else 1
            act_p = self._get_npc_action(p_id, "HIDER")
            p_idx = self._apply_action(p_id, act_p)
            self.data.ctrl[p_idx] = float(act_p[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[p_idx + 1] = float(act_p[1])
            
            sf, sr = self._seeker_rule_based_policy()
            self.data.ctrl[0] = sf
            self.data.ctrl[1] = sr
        else:
            self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(action[1])
            for i in [1, 2]:
                act_h = self._get_npc_action(i, "HIDER")
                idx = self._apply_action(i, act_h)
                self.data.ctrl[idx] = float(act_h[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[idx + 1] = float(act_h[1])
                    
        for _ in range(ACTION_REPEAT):
            for box, pose in self.locked_pose.items():
                if self.locked_boxes[box]:
                    jid = self.box1_joint_id if box == self.box1_body else self.box2_joint_id
                    qa = self.model.jnt_qposadr[jid]
                    da = self.model.jnt_dofadr[jid]
                    self.data.qpos[qa : qa + 7] = pose
                    self.data.qvel[da : da + 6] = 0
            mujoco.mj_step(self.model, self.data)
                
        self._obs_memo.clear()
        obs_learner = self._get_obs(self.learning_agent_id)
        self._get_obs(0, skip_lidar=True) 
        
        team_reward = 0.0
        learner_body = self.h1_body if self.learning_agent_id == 1 else self.h2_body
            
        if not self.visible_cache[0].get(learner_body, False):
            self.hidden_steps_count += 1
        if any(self.visible_cache[0].get(b, False) for b in [self.h1_body, self.h2_body]):
            self.caught_steps_count += 1
                
        sp_pos = self.data.xpos[self.s0_body][:2]
        for h_idx, b_id in [(1, self.h1_body), (2, self.h2_body)]:
            hp_pos = self.data.xpos[b_id][:2]
            diff = hp_pos - sp_pos
            dist = np.linalg.norm(diff)
            
            if self.visible_cache[0].get(b_id, False):
                sr_val = self.data.qpos[self.srot_adr]
                unit = diff / (dist + 1e-8)
                cv = unit[0] * math.cos(sr_val) + unit[1] * math.sin(sr_val)
                h_rew = -cv * COS_PENALTY_SCALE + (dist - self.prev_dist[h_idx]) * REWARD_DISTANCE_DIFF_SCALE
            else:
                h_rew = REWARD_HIDDEN_BONUS
            
            if np.max(np.abs(hp_pos)) > 6.5:
                h_rew += PENALTY_SAFEGUARD
                
            team_reward += h_rew
            self.prev_dist[h_idx] = dist
            
        final_reward = team_reward if TRAIN_TARGET == "HIDER" else -team_reward
        info_out = {
            "hidden_steps": float(self.hidden_steps_count),
            "caught_steps": float(self.caught_steps_count)
        }
        return obs_learner, float(final_reward), False, (self.current_step >= MAX_STEPS), info_out

    def _get_npc_action(self, agent_id, agent_type):
        """NPC の行動生成。"""
        import torch
        obs_val = self._get_obs(agent_id)
        self.npc_obs_history[agent_id].update(obs_val)
        
        model = self.npc_hider_agent if agent_type == "HIDER" else self.npc_seeker_agent
        if model:
            with torch.no_grad():
                seq = self.npc_obs_history[agent_id].get()
                res = model.get_action_and_value(seq)
                return res[0].cpu().numpy()[0]
        return self.action_space.sample() * 0.5

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
        env_p = TeamCosEnv(render_mode="human")
        agent_p = Agent(env_p.observation_space.shape[0], env_p.action_space.shape[0]).to("cpu")
        load_model_safely(agent_p, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        agent_p.eval()
        hist_p = ObsHistory(1, TRANSFORMER_SEQ_LEN, env_p.observation_space.shape[0], "cpu")
        try:
            while True:
                obs_init, _ = env_p.reset()
                hist_p.reset()
                hist_p.update(obs_init)
                is_ep_done = False
                while not is_ep_done:
                    t_loop0 = time.time()
                    with torch.no_grad():
                        res_act = agent_p.get_action_and_value(hist_p.get())
                        v_act = res_act[0].cpu().numpy()[0]
                    obs_next, r, term, trunc, info = env_p.step(v_act)
                    is_ep_done = term or trunc
                    hist_p.update(obs_next)
                    env_p.render()
                    dt_proc = time.time() - t_loop0
                    t_wait = (0.005 * ACTION_REPEAT) - dt_proc
                    if t_wait > 0:
                        time.sleep(t_wait)
        except KeyboardInterrupt:
            pass
        finally:
            env_p.close()
        return

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb
    import main18_optimization as base_config
    from main18_optimization import Agent

    envs_vec = gym.vector.AsyncVectorEnv([lambda: gym.wrappers.RecordEpisodeStatistics(TeamCosEnv()) for _ in range(NUM_ENVS)])
    device_final = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")
    run_full_name = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{int(time.time())}"
    
    if TRACK_WANDB:
        wandb.init(project=base_config.WANDB_PROJECT_NAME, config={"Target": TRAIN_TARGET, "v": "25.47_sync"}, name=run_full_name, sync_tensorboard=False, save_code=True)
    
    writer_obj = SummaryWriter(f"runs/{run_full_name}")
    agent_train = Agent(envs_vec.single_observation_space.shape[0], envs_vec.single_action_space.shape[0]).to(device_final)
    optimizer_obj = optim.Adam(agent_train.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    g_step, s_step_val = 0, 0
    if LOAD_EXISTING_MODELS:
        loaded_path = load_model_safely(agent_train, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if loaded_path:
            meta_path = loaded_path.replace('.pt', '_checkpoint.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f_in:
                    meta_json = json.load(f_in)
                    g_step = meta_json.get('global_step', 0)
                    s_step_val = g_step
                    
    history_train = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device_final)
    obs_batch = torch.zeros((NUM_STEPS, NUM_ENVS, TRANSFORMER_SEQ_LEN, 53), device=device_final)
    act_batch = torch.zeros((NUM_STEPS, NUM_ENVS, 4), device=device_final)
    lp_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_final)
    rew_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_final)
    done_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_final)
    val_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_final)
    
    seed_val = FIXED_SEED if FIXED_SEED else int(time.time())
    next_obs_v_data, _ = envs_vec.reset(seed=seed_val)
    next_done_v_data = torch.zeros(NUM_ENVS).to(device_final)
    history_train.update(next_obs_v_data)
    
    t0_start = time.time()
    updates_count = int(max(1, TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)))
    h_ret_list, h_hid_list = [], []

    print(f"--- Training Started (Step: {g_step}) ---")
    try:
        for update_idx in tqdm(range(1, updates_count + 1), desc="Updates"):
            for step_idx in range(NUM_STEPS):
                g_step += NUM_ENVS
                obs_batch[step_idx] = history_train.get()
                done_batch[step_idx] = next_done_v_data
                with torch.no_grad():
                    a_out, lp_out, _, v_out = agent_train.get_action_and_value(history_train.get())
                    val_batch[step_idx] = v_out.flatten()
                act_batch[step_idx] = a_out
                lp_batch[step_idx] = lp_out
                next_obs_v_data, rew_array, term_array, trunc_array, info_dict = envs_vec.step(a_out.cpu().numpy())
                done_mask = np.logical_or(term_array, trunc_array)
                
                # ★統計抽出ロジックの修正
                if "final_info" in info_dict:
                    for f_info in info_dict["final_info"]:
                        if f_info is not None:
                            if "episode" in f_info:
                                h_ret_list.append(float(f_info["episode"]["r"]))
                            if "hidden_steps" in f_info:
                                h_hid_list.append(float(f_info["hidden_steps"]))
                elif "episode" in info_dict:
                    ep_mask = info_dict.get("_episode", done_mask)
                    for i in range(NUM_ENVS):
                        if ep_mask[i]:
                            h_ret_list.append(float(info_dict["episode"]["r"][i]))
                            if "hidden_steps" in info_dict:
                                if hasattr(info_dict["hidden_steps"], '__getitem__'):
                                    h_hid_list.append(float(info_dict["hidden_steps"][i]))

                rew_batch[step_idx] = torch.tensor(rew_array).to(device_final).view(-1)
                next_done_v_data = torch.tensor(done_mask).to(device_final, dtype=torch.float32)
                history_train.update(next_obs_v_data)
                
            with torch.no_grad():
                v_final_pred = agent_train.get_value(history_train.get()).reshape(1, -1)
                adv_tensor = torch.zeros_like(rew_batch).to(device_final)
                last_gae_val = 0
                for t in reversed(range(NUM_STEPS)):
                    m_done = 1.0 - (next_done_v_data if t == NUM_STEPS - 1 else done_batch[t + 1])
                    v_next = v_final_pred if t == NUM_STEPS - 1 else val_batch[t + 1]
                    delta = rew_batch[t] + base_config.GAMMA * v_next * m_done - val_batch[t]
                    adv_tensor[t] = last_gae_val = delta + base_config.GAMMA * base_config.GAE_LAMBDA * m_done * last_gae_val
                returns_batch = adv_tensor + val_batch
                
            f_obs, f_lp, f_act, f_adv, f_ret = obs_batch.reshape(-1, TRANSFORMER_SEQ_LEN, 53), lp_batch.reshape(-1), act_batch.reshape(-1, 4), adv_tensor.reshape(-1), returns_batch.reshape(-1)
            for epoch_idx in range(UPDATE_EPOCHS):
                idx_p = np.random.permutation(NUM_STEPS * NUM_ENVS)
                for ptr in range(0, NUM_STEPS * NUM_ENVS, MINIBATCH_SIZE):
                    mb = idx_p[ptr : ptr + MINIBATCH_SIZE]
                    _, n_lp, ent_batch, n_v = agent_train.get_action_and_value(f_obs[mb], f_act[mb])
                    ratio = (n_lp - f_lp[mb]).exp()
                    mb_adv_norm = (f_adv[mb] - f_adv[mb].mean()) / (f_adv[mb].std() + 1e-8)
                    pg_loss = torch.max(-mb_adv_norm * ratio, -mb_adv_norm * torch.clamp(ratio, 0.8, 1.2)).mean()
                    v_loss = 0.5 * ((n_v.view(-1) - f_ret[mb])**2).mean()
                    loss_total = pg_loss - ENT_COEF * ent_batch.mean() + 0.5 * v_loss
                    optimizer_obj.zero_grad()
                    loss_total.backward()
                    nn.utils.clip_grad_norm_(agent_train.parameters(), 0.5)
                    optimizer_obj.step()

            if (update_idx % 10 == 0) or TRIAL_MODE:
                dt_now = time.time() - t0_start
                sps_now = int((g_step - s_step_val) / dt_now) if dt_now > 0 else 0
                log_map = {"charts/SPS": sps_now, "losses/total_loss": loss_total.item(), "global_step": g_step}
                if h_ret_list:
                    m_ret, m_hid = np.mean(h_ret_list), np.mean(h_hid_list) if h_hid_list else 0.0
                    log_map.update({"charts/episodic_return": m_ret, "charts/steps_hidden": m_hid})
                    tqdm.write(f"Update {update_idx}, Step {g_step}, SPS {sps_now}, EpRet: {m_ret:.1f}, Hidden: {m_hid:.1f}")
                    h_ret_list, h_hid_list = [], []
                elif TRIAL_MODE:
                    tqdm.write(f"Update {update_idx}, Step {g_step}, SPS {sps_now}, Loss: {loss_total.item():.3f}")
                if TRACK_WANDB:
                    wandb.log(log_map)
                    
    except KeyboardInterrupt:
        print("\nInterrupted.")
        
    if SAVE_MODEL:
        torch.save(agent_train.state_dict(), SAVE_MODEL_PATH)
        with open(SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json'), 'w') as f_out:
            json.dump({'global_step': g_step}, f_out)
            
    # 【終了処理ブロック: すべて独立行として記述】
    envs_vec.close()
    writer_obj.close()
    if TRACK_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()