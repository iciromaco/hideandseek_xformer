# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【概要】
# 2体の Hider と 1体の Seeker による、マルチエージェントかくれんぼ環境の強化学習スクリプト。
# 物理エンジン MuJoCo と Transformer 搭載 Actor-Critic (PPO) アルゴリズムを使用します。
# 
# 【修正内容 (v25.45 - 統計収集の堅牢化・完全展開版)】
# 1. 統計情報の完全取得: 
#    - AsyncVectorEnv から final_info を抽出し、"Hidden:" ステップ数が nan になる問題を解消。
#    - エピソード完了時に報酬(EpRet)と生存時間(Hidden)を確実に記録します。
# 2. 構文エラーの物理的排除: 
#    - 全てのセミコロン (;) を削除。終了処理や条件分岐も全て独立行へ展開。
# 3. 詳解日本語コメント:
#    - プログラムの各工程（観測、AI、報酬、学習）に動作原理の解説を付与。
# 4. ポテンシャル法 AI (Seeker):
#    - 高速かつ公平な NPC 制御を継続実装。

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
# マルチプロセス実行時の CPU スレッド競合を抑制し、並列環境の処理効率を最大化します。
if platform.processor() != 'arm':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# インポートパスの解決
# 設定ファイル main18_optimization.py が見つかるまで親ディレクトリを探索します。
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

# "initial":    新規モデルとして学習を開始
# "refinement": 既存モデル (.pt) をロードして追加学習
MODE = "initial" # "refinement" 

# "TRAIN": 並列環境で学習を実行
# "PLAY":  単一環境で学習済みモデルの挙動を人間が目視で確認
EXECUTION_MODE = "TRAIN" 

# "HIDER":  Hider チームを学習対象に指定
# "SEEKER": Seeker を学習対象に指定
TRAIN_TARGET = "HIDER" 

EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"
EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"

# モデルのロード判定
LOAD_EXISTING_MODELS = False
if MODE == "refinement":
    LOAD_EXISTING_MODELS = True

# 実験の基本フラグ
SAVE_MODEL = True           # 学習終了時にモデルを保存
TRACK_WANDB = True          # Weights & Biases にログを送信
FIXED_SEED = None           # None: ランダム / 整数: 結果を固定
TRIAL_MODE = False          # True: Optuna 等の外部スクリプトからの呼び出し用

# 強化学習 (PPO) パラメータ
TOTAL_TIMESTEPS = 5000000 
NUM_ENVS = 8                # 並列プロセス数 (CPUコア数に合わせて調整)
NUM_STEPS = 128             # 1更新あたりのデータ収集長 (各環境ごと)
LEARNING_RATE = 2e-4
ENT_COEF = 0.001            # 探索の強さ
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4           # 同一データに対する反復学習回数

# 環境・シミュレーション設定
ACTION_REPEAT = 16          # 1ステップあたりの MuJoCo 進展回数
PREP_STEPS = 80             # エピソード序盤の Seeker 停止期間
MAX_STEPS = 300             # 1エピソードの最大ステップ数（固定長）
FOV_DEG = 135               # エージェントの視野角（度）
FOV_RAD_HALF = math.radians(FOV_DEG / 2.0)
TRANSFORMER_SEQ_LEN = 8     # Transformer が参照する過去の履歴長

# 高速化キャッシュ設定
RAYCAST_GRID_SIZE = 0.1     # 空間解像度 (10cm 単位で結果をキャッシュ)
LIDAR_CACHE_POS_THRESH_SQ = 0.05**2 # 移動距離（二乗）のキャッシュ破棄しきい値
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0) # 回転のキャッシュ破棄しきい値

# 報酬設計の重み
REWARD_HIDDEN_BONUS = 1.0   # 隠れている時のステップ報酬
COS_PENALTY_SCALE = 2.0     # 視界内ペナルティ倍率 (真正面ほどマイナス)
REWARD_DISTANCE_DIFF_SCALE = 1.0 # 敵から遠ざかる行動への報酬倍率
PENALTY_SAFEGUARD = -20.0   # 場外脱走などの異常行動に対する罰則

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
    Transformer の入力となる時系列履歴をゼロコピーで管理するクラス。
    ダブルバッファ（ミラーリング）により、データの読み出し効率を最大化します。
    """
    def __init__(self, num_envs, seq_len, obs_dim, device):
        import torch
        # バッファを seq_len の 2 倍確保
        self.buffer = torch.zeros((num_envs, seq_len * 2, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.ptr = 0

    def reset(self):
        """バッファ全体をリセットします。"""
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        """最新の観測値を 2 箇所に同時に書き込みます。"""
        import torch
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # 循環バッファの現在位置に書き込み
        self.buffer[:, self.ptr] = obs_tensor
        # ミラー領域にも同時に書き込み
        mirrored_idx = self.ptr + self.seq_len
        self.buffer[:, mirrored_idx] = obs_tensor
        
        # ポインタを 1 つ進める
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        """最新の履歴をスライス（View）として高速に取得します（コピーなし）。"""
        end_idx = self.ptr + self.seq_len
        return self.buffer[:, self.ptr : end_idx]

def load_model_safely(model_obj, base_name, target_type):
    """
    複数のファイル名の候補からロード可能なモデルを検索します。
    MODE (initial/refinement) の切り替えに柔軟に対応します。
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
    視界勾配（Cos）報酬とチーム共有報酬を採用したマルチエージェント環境。
    1行1ステートメントを遵守し、計算負荷を最適化しています。
    """
    def __init__(self, render_mode=None):
        # 親クラスの初期化
        super().__init__(render_mode=render_mode)
        
        import torch
        from main18_optimization import Agent
        cpu_dev = torch.device("cpu")
        
        # NPCエージェント（学習対象以外）のための時系列バッファ
        self.npc_obs_history = {}
        self.npc_obs_history[0] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
        self.npc_obs_history[1] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
        self.npc_obs_history[2] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
        
        # 高速化キャッシュとメモ化バッファ
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
        
        # Lidar計算用三角関数テーブル
        angles_array = self.lidar_angles
        self.lidar_cos_sin = np.column_stack([
            np.cos(angles_array), 
            np.sin(angles_array)
        ])

        # 学習ターゲットIDの設定
        if TRAIN_TARGET == "HIDER":
            self.learning_agent_id = self.np_random.choice([1, 2])
        else:
            self.learning_agent_id = 0
            
        self.lock_cooldown = {1: 0, 2: 0}
        self.raycast_stats = {"hits": 0, "misses": 0}
        
        # NPC Seeker 用の状態変数
        self.s0_stuck_timer = 0
        self.s0_recovery_mode = 0
        self.seeker_search_timer = 0
        self.seeker_random_target = np.zeros(2)
        self.seeker_target_pos = np.zeros(2)
        self.seeker_last_known_pos = None
        
        # ボディIDの一括管理リスト
        target_bodies = [
            self.box1_body, 
            self.box2_body, 
            self.ramp_body, 
            self.h1_body, 
            self.h2_body, 
            self.s0_body
        ]
        self.all_target_bodies = np.array(target_bodies)
        
        # シーカー回転関節のアドレス
        joint_obj = self.model.joint('s_rot')
        self.srot_adr = self.model.jnt_qposadr[joint_obj.id]
        
        # NPC用学習済みモデルのロード
        self.npc_hider_agent = Agent(53, 4).to("cpu")
        self.npc_seeker_agent = Agent(53, 4).to("cpu")
        
        # ロード時のログ制御
        log_status = os.environ.get("NPC_MODELS_LOGGED")
        should_print = (log_status != "TRUE")
        
        h_model = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
        s_model = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
        
        if should_print:
            if h_model: 
                print(f"Loaded NPC Hider from {h_model}", flush=True)
            if s_model: 
                print(f"Loaded NPC Seeker from {s_model}", flush=True)
            os.environ["NPC_MODELS_LOGGED"] = "TRUE"

    def reset(self, seed=None, options=None):
        """環境を初期化し、キャッシュをリセットします。"""
        obs, info = super().reset(seed=seed, options=options)
        
        self.hidden_steps_count = 0
        self.caught_steps_count = 0
        self._obs_memo.clear()
        self.lidar_array_cache.clear()
        self.raycast_cache.clear()
        
        # 初期距離の記録（報酬計算の基準）
        sp_xy = self.data.xpos[self.s0_body][:2]
        for i in [1, 2]:
            target_bid = self.h1_body
            if i == 2:
                target_bid = self.h2_body
            hp_xy = self.data.xpos[target_bid][:2]
            dist_val = np.linalg.norm(hp_xy - sp_xy)
            self.prev_dist[i] = dist_val
            
        return obs, info

    def _get_cached_ray(self, agent_id, origin_p, direction, beam_id):
        """RayCast結果を空間グリッド単位で保持。物理演算の負荷を劇的に下げます。"""
        angle_rad = math.atan2(direction[1], direction[0])
        # 位置を 10cm グリッドに丸めてキャッシュキーを生成
        grid_x = int(origin_p[0] / RAYCAST_GRID_SIZE)
        grid_y = int(origin_p[1] / RAYCAST_GRID_SIZE)
        cache_key = (agent_id, grid_x, grid_y, beam_id)
        
        if cache_key in self.raycast_cache:
            c_angle, c_res, c_hit_gid = self.raycast_cache[cache_key]
            # 回転角の変化がわずかであればキャッシュを返す
            angle_diff = (angle_rad - c_angle + math.pi) % self.two_pi - math.pi
            if abs(angle_diff) < 0.05:
                self.raycast_stats["hits"] += 1
                return c_res, c_hit_gid

        # キャッシュミス時のみ物理エンジン MuJoCo を呼び出す
        self.raycast_stats["misses"] += 1
        gid_output = np.zeros(1, dtype=np.int32)
        from_xyz = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
        dir_xyz = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
        
        # 自己ヒットの除外
        exclude_id = self.s0_body
        if agent_id == 1:
            exclude_id = self.h1_body
        elif agent_id == 2:
            exclude_id = self.h2_body
        
        res_val = mujoco.mj_ray(self.model, self.data, from_xyz, dir_xyz, None, 1, exclude_id, gid_output)
        hit_geom_id = gid_output[0]
        
        # 新しい判定結果をキャッシュに登録
        self.raycast_cache[cache_key] = (angle_rad, res_val, hit_geom_id)
        return res_val, hit_geom_id

    def _get_obs(self, agent_id, skip_lidar=False):
        """エージェントへの入力ベクトルを生成。行列演算により高い SPS を維持します。"""
        if agent_id in self._obs_memo: 
            return self._obs_memo[agent_id]
        
        # 対象エージェントの特定
        if agent_id == 0:
            b_id = self.s0_body
            p_pref = 's'
        elif agent_id == 1:
            b_id = self.h1_body
            p_pref = 'h1'
        else:
            b_id = self.h2_body
            p_pref = 'h2'
            
        agent_xy = self.data.xpos[b_id][:2]
        joint_rot = self.model.joint(f'{p_pref}_rot')
        hra = self.data.qpos[self.model.jnt_qposadr[joint_rot.id]]
        
        c_val = math.cos(hra)
        s_val = math.sin(hra)
        # ローカル座標系への変換行列 (2D回転)
        rot_mat = np.array([[c_val, s_val], [-s_val, c_val]])
        
        joint_x = self.model.joint(f'{p_pref}_x')
        dof_start = self.model.jnt_dofadr[joint_x.id]
        v_raw = self.data.qvel[dof_start : dof_start + 2]
        # ローカル速度への変換とスケーリング
        v_obs_val = (rot_mat @ v_raw) / 12.0
        
        # [1] 自己状態
        self_state_vec = np.array([v_obs_val[0], v_obs_val[1], hra, c_val, s_val], dtype=np.float32)
        
        # [2] Lidar センサー
        lidar_vec = np.zeros(len(self.lidar_angles), dtype=np.float32)
        if not skip_lidar:
            l_cache = self.lidar_array_cache.get(agent_id)
            if l_cache:
                pos_diff = agent_xy - l_cache[0]
                # 平方根計算を避けるため二乗距離で判定
                if (pos_diff[0]**2 + pos_diff[1]**2) < LIDAR_CACHE_POS_THRESH_SQ:
                    ang_diff = (hra - l_cache[1] + math.pi) % self.two_pi - math.pi
                    if abs(ang_diff) < LIDAR_CACHE_ANG_THRESH:
                        lidar_vec = l_cache[2]
            
            # 再計算が必要な場合のみビームを飛ばす
            if l_cache is None or np.sum(lidar_vec) == 0:
                for idx in range(len(self.lidar_angles)):
                    co = self.lidar_cos_sin[idx][0]
                    si = self.lidar_cos_sin[idx][1]
                    # 加法定理を用いてエージェントの回転を反映
                    b_dir_x = co * c_val - si * s_val
                    b_dir_y = si * c_val + co * s_val
                    b_dir_vec = np.array([b_dir_x, b_dir_y])
                    
                    dist_r, _ = self._get_cached_ray(agent_id, agent_xy, b_dir_vec, idx + 100)
                    if dist_r != -1:
                        lidar_vec[idx] = min(dist_r, 2.5) / 2.5
                    else:
                        lidar_vec[idx] = 1.0
                # キャッシュの更新
                self.lidar_array_cache[agent_id] = (agent_xy.copy(), hra, lidar_vec.copy())

        # [3] オブジェクトベクトル演算
        mask_others = (self.all_target_bodies != b_id)
        other_ids = self.all_target_bodies[mask_others]
        
        # 相対位置ベクトルの一括計算
        rel_pos_mat = self.data.xpos[other_ids][:, :2] - agent_xy
        # 各物体との距離を一括算出
        dists_array = np.sqrt(np.sum(rel_pos_mat**2, axis=1))
        
        # 視野角(FOV)判定
        g_angles = np.arctan2(rel_pos_mat[:, 1], rel_pos_mat[:, 0])
        rel_angs_array = (g_angles - hra + math.pi) % self.two_pi - math.pi
        fov_mask_bool = np.abs(rel_angs_array) <= FOV_RAD_HALF
        
        v_cache = self.visible_cache[agent_id]
        v_cache.clear()
        rel_info_store = {}
        
        for idx in range(len(other_ids)):
            tid = other_ids[idx]
            if not fov_mask_bool[idx]:
                v_cache[tid] = False
                continue
                
            unit_vector = rel_pos_mat[idx] / (dists_array[idx] + 1e-8)
            dist_ray_val, hit_gid_val = self._get_cached_ray(agent_id, agent_xy, unit_vector, tid)
            
            # 遮蔽判定ロジック
            hit_body_id = self.model.geom_bodyid[hit_gid_val]
            is_vis_bool = (hit_body_id == tid) or (dist_ray_val != -1 and dist_ray_val > dists_array[idx] - 0.4)
            v_cache[tid] = is_vis_bool
            
            if is_vis_bool:
                # ローカル座標系での詳細情報を計算
                pos_local = (rot_mat @ rel_pos_mat[idx]) / 12.0
                q_data = self.data.xquat[tid]
                yaw_rel = math.atan2(2.0*(q_data[0]*q_data[3] + q_data[1]*q_data[2]), 1-2*(q_data[2]**2+q_data[3]**2)) - hra
                
                j_adr = self.model.body_jntadr[tid]
                v_target_vec = np.zeros(2)
                if j_adr != -1:
                    v_target_vec = self.data.qvel[j_adr : j_adr + 2]
                
                rel_v_local = (rot_mat @ (v_target_vec - v_raw)) / 12.0
                rel_info_store[tid] = [
                    pos_local[0], pos_local[1], 
                    rel_v_local[0], rel_v_local[1], 
                    math.cos(yaw_rel), math.sin(yaw_rel)
                ]

        # 観測配列の構築 (関数定義オーバーヘッドを排除するためリスト集計を採用)
        pack_list = [self_state_vec, lidar_vec]
        
        if agent_id == 0:
            # Seeker 視点の情報構築
            order = [
                (self.box1_body, self.locked_boxes[self.box1_body]), 
                (self.box2_body, self.locked_boxes[self.box2_body]), 
                (self.ramp_body, None)
            ]
            for tid, lock in order:
                found = rel_info_store.get(tid)
                if found:
                    vec = list(found)
                    if lock is not None:
                        vec.append(1.0 if lock else 0.0)
                    vec.append(1.0)
                    pack_list.append(np.array(vec, dtype=np.float32))
                else:
                    pack_list.append(np.zeros(8 if lock is not None else 7, dtype=np.float32))
            
            # 敵(Hider)の情報 (位置のみに短縮して入力)
            for tid in [self.h1_body, self.h2_body]:
                found = rel_info_store.get(tid)
                if found:
                    vec = list(found[:4])
                    vec.append(1.0)
                    pack_list.append(np.array(vec, dtype=np.float32))
                else:
                    pack_list.append(np.zeros(5, dtype=np.float32))
            pack_list.append(np.zeros(3, dtype=np.float32))
        else:
            # Hider 視点の情報構築
            p_id = self.h2_body if agent_id == 1 else self.h1_body
            for tid, lock in [(self.box1_body, self.locked_boxes[self.box1_body]), (self.box2_body, self.locked_boxes[self.box2_body]), (self.ramp_body, None)]:
                found = rel_info_store.get(tid)
                if found:
                    vec = list(found)
                    if lock is not None:
                        vec.append(1.0 if lock else 0.0)
                    vec.append(1.0)
                    pack_list.append(np.array(vec, dtype=np.float32))
                else:
                    pack_list.append(np.zeros(8 if lock is not None else 7, dtype=np.float32))
            # Seeker の相対情報
            found = rel_info_store.get(self.s0_body)
            if found:
                vec = list(found[:4])
                vec.append(1.0)
                pack_list.append(np.array(vec, dtype=np.float32))
            else:
                pack_list.append(np.zeros(5, dtype=np.float32))
            # 味方(Partner)の情報
            found = rel_info_store.get(p_id)
            if found:
                vec = list(found)
                vec.append(1.0)
                pack_list.append(np.array(vec, dtype=np.float32))
            else:
                pack_list.append(np.zeros(7, dtype=np.float32))
            # 掴み(Grasp)状態フラグ
            pack_list.append(np.array([1.0 if self.grasping[agent_id] else 0.0], dtype=np.float32))
            
        final_obs_array = np.concatenate(pack_list).astype(np.float32)
        self._obs_memo[agent_id] = final_obs_array
        return final_obs_array

    def _update_seeker_state(self):
        """NPCシーカーの目標地点を決定する知能ロジック。視覚・記憶・巡回を管理します。"""
        seeker_xy_pos = self.data.xpos[self.s0_body][:2]
        # 視界情報の更新
        self._get_obs(0)
        v_dict = self.visible_cache[0]
        is_h1_vis = v_dict.get(self.h1_body, False)
        is_h2_vis = v_dict.get(self.h2_body, False)
        
        if is_h1_vis or is_h2_vis:
            # 視界内の Hider をターゲットにする
            target_bid = self.h1_body if is_h1_vis else self.h2_body
            self.seeker_target_pos = self.data.xpos[target_bid][:2].copy()
            self.seeker_last_known_pos = self.seeker_target_pos.copy()
            self.seeker_mode = "CHASING"
        elif self.seeker_last_known_pos is not None:
            # 見失った場合、最後に目撃した場所へ急行する
            diff_to_mem = seeker_xy_pos - self.seeker_last_known_pos
            if (diff_to_mem[0]**2 + diff_to_mem[1]**2) < 0.25: 
                # 到着しても見当たらなければ記憶をリセット
                self.seeker_last_known_pos = None
                self.seeker_search_timer = 50
            else: 
                self.seeker_target_pos = self.seeker_last_known_pos.copy()
                self.seeker_mode = "SEARCHING"
        else:
            # 手がかりがない場合はランダムに巡回
            if self.seeker_search_timer <= 0:
                self.seeker_random_target = self.np_random.uniform(-4, 4, 2)
                self.seeker_search_timer = 80
            self.seeker_search_timer -= 1
            self.seeker_target_pos = self.seeker_random_target.copy()
            self.seeker_mode = "PATROLLING"

    def _seeker_rule_based_policy(self):
        """ポテンシャル法に基づいた Seeker NPC の移動制御。"""
        if self.current_step < PREP_STEPS:
            return 0.0, 0.0
            
        seeker_xy_pos = self.data.xpos[self.s0_body][:2]
        yaw_val = self.data.qpos[self.srot_adr]
        c_y = math.cos(yaw_val)
        s_y = math.sin(yaw_val)
        
        # [斥力] Lidar で見える障害物から離れる反発力
        lidar_values = self.lidar_array_cache[0][2]
        repulsion_force = np.zeros(2)
        for i in range(len(self.lidar_angles)):
            dist_actual = lidar_values[i] * 2.5
            if dist_actual < 1.0:
                # 距離が近いほど強い力を発生させる
                f_val = (1.0 - dist_actual) / (dist_actual + 0.1)
                co_val = self.lidar_cos_sin[i][0]
                si_val = self.lidar_cos_sin[i][1]
                # 反発方向をワールド座標系に合成
                repulsion_force[0] -= (co_val * c_y - si_val * s_y) * f_val
                repulsion_force[1] -= (si_val * c_y + co_val * s_y) * f_val
                
        # [引力] ターゲットへ向かう引き寄せの力
        diff_vec = self.seeker_target_pos - seeker_xy_pos
        dist_to_t = np.linalg.norm(diff_vec)
        unit_target = diff_vec / (dist_to_t + 1e-8)
        
        # 力の合成と目標角度の決定
        combined_vec = unit_target + repulsion_force * 1.5
        target_ang = math.atan2(combined_vec[1], combined_vec[0])
        error_rad = (target_ang - yaw_val + math.pi) % self.two_pi - math.pi
        
        thrust_output = SEEKER_RB_THRUST
        if abs(error_rad) > SEEKER_RB_TURN_THRESH:
            # 回転を優先するため減速
            thrust_output = thrust_output * 0.3
            
        # スタック検知
        sx_joint = self.model.joint('s_x')
        dof_adr_idx = self.model.jnt_dofadr[sx_joint.id]
        v_sq_total = self.data.qvel[dof_adr_idx]**2 + self.data.qvel[dof_adr_idx + 1]**2
        
        if thrust_output > 0.05 and v_sq_total < 0.0025:
            self.s0_stuck_timer += 5
        else:
            self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            
        if self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 15
            self.s0_stuck_timer = 0
            self.s0_recovery_turn_dir = self.np_random.choice([-1.0, 1.0])
            
        if self.s0_recovery_mode > 0:
            self.s0_recovery_mode -= 1
            # 後退しながら旋回してスタックを回避
            return -0.2, 1.5 * self.s0_recovery_turn_dir
            
        return float(thrust_output), float(np.clip(error_rad * 6.0, -3.0, 3.0))

    def step(self, action):
        """シミュレーションを 1 ステップ進めます。報酬計算と統計収集を含みます。"""
        self._obs_memo.clear()
        self.current_step += 1
        
        for idx in [1, 2]:
            self.lock_cooldown[idx] = max(0, self.lock_cooldown[idx] - 1)
            
        # 思考の更新
        self._update_seeker_state()
        self.data.ctrl[:] = 0.0
        
        # 制御入力の適用
        if TRAIN_TARGET == "HIDER":
            m_idx = self._apply_action(self.learning_agent_id, action)
            self.data.ctrl[m_idx] = float(action[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[m_idx + 1] = float(action[1])
            
            p_id = 2 if self.learning_agent_id == 1 else 1
            act_p = self._get_npc_action(p_id, "HIDER")
            p_idx = self._apply_action(p_id, act_p)
            self.data.ctrl[p_idx] = float(act_p[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[p_idx + 1] = float(act_p[1])
            
            sf_npc, sr_npc = self._seeker_rule_based_policy()
            self.data.ctrl[0] = sf_npc
            self.data.ctrl[1] = sr_npc
        else:
            # Seeker学習モード
            self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(action[1])
            for i in [1, 2]:
                act_h = self._get_npc_action(i, "HIDER")
                h_idx = self._apply_action(i, act_h)
                self.data.ctrl[h_idx] = float(act_h[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[h_idx + 1] = float(act_h[1])
                    
        # 物理サブステップの実行
        for _ in range(ACTION_REPEAT):
            # ロックされた箱の位置を固定（Velocity Freeze）
            for box_id_key, pose_val in self.locked_pose.items():
                if self.locked_boxes[box_id_key]:
                    if box_id_key == self.box1_body:
                        jid = self.box1_joint_id
                    else:
                        jid = self.box2_joint_id
                    qa_adr = self.model.jnt_qposadr[jid]
                    da_adr = self.model.jnt_dofadr[jid]
                    self.data.qpos[qa_adr : qa_adr + 7] = pose_val
                    self.data.qvel[da_adr : da_adr + 6] = 0
            mujoco.mj_step(self.model, self.data)
                
        self._obs_memo.clear()
        obs_array = self._get_obs(self.learning_agent_id)
        # 報酬判定用にシーカー視界を更新
        self._get_obs(0, skip_lidar=True) 
        
        # チーム報酬の集計
        team_reward_val = 0.0
        learner_body = self.h1_body if self.learning_agent_id == 1 else self.h2_body
            
        if not self.visible_cache[0].get(learner_body, False):
            self.hidden_steps_count += 1
        if any(self.visible_cache[0].get(b, False) for b in [self.h1_body, self.h2_body]):
            self.caught_steps_count += 1
                
        sp_pos_xyz = self.data.xpos[self.s0_body][:2]
        for h_idx_tmp, b_id_tmp in [(1, self.h1_body), (2, self.h2_body)]:
            hp_pos_xyz = self.data.xpos[b_id_tmp][:2]
            diff_vec_tmp = hp_pos_xyz - sp_pos_xyz
            dist_val_tmp = np.linalg.norm(diff_vec_tmp)
            
            if self.visible_cache[0].get(b_id_tmp, False):
                sr_val_tmp = self.data.qpos[self.srot_adr]
                unit_vec_tmp = diff_vec_tmp / (dist_val_tmp + 1e-8)
                # 視界内の角度勾配ペナルティ
                cos_val_tmp = unit_vec_tmp[0] * math.cos(sr_val_tmp) + unit_vec_tmp[1] * math.sin(sr_val_tmp)
                h_rew_tmp = -cos_val_tmp * COS_PENALTY_SCALE + (dist_val_tmp - self.prev_dist[h_idx_tmp]) * REWARD_DISTANCE_DIFF_SCALE
            else:
                h_rew_tmp = REWARD_HIDDEN_BONUS
            
            # 場外ペナルティ
            if np.max(np.abs(hp_pos_xyz)) > 6.5:
                h_rew_tmp += PENALTY_SAFEGUARD
                
            team_reward_val += h_rew_tmp
            self.prev_dist[h_idx_tmp] = dist_val_tmp
            
        # 学習対象による報酬の確定
        reward_final = team_reward_val if TRAIN_TARGET == "HIDER" else -team_reward_val
        
        # 客観的評価指標（Hiddenステップ数）のパッケージ化
        info_final_dict = {
            "hidden_steps": float(self.hidden_steps_count),
            "caught_steps": float(self.caught_steps_count)
        }
        
        # 固定長エピソードのため terminated は False
        return obs_array, float(reward_final), False, (self.current_step >= MAX_STEPS), info_final_dict

    def _get_npc_action(self, agent_id, agent_type):
        """NPCの行動（推論またはランダム）を実行。"""
        import torch
        obs_val = self._get_obs(agent_id)
        self.npc_obs_history[agent_id].update(obs_val)
        
        model_ptr = self.npc_hider_agent if agent_type == "HIDER" else self.npc_seeker_agent
        if model_ptr:
            with torch.no_grad():
                seq_data = self.npc_obs_history[agent_id].get()
                res_tuple = model_ptr.get_action_and_value(seq_data)
                return res_tuple[0].cpu().numpy()[0]
        return self.action_space.sample() * 0.5

# ==========================================
# 4. メイン処理
# ==========================================
def main():
    # デッドロック防止設定
    if platform.system() == "Linux":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
            
    # 【推論モード】
    if EXECUTION_MODE == "PLAY":
        import torch
        from main18_optimization import Agent
        env_play = TeamCosEnv(render_mode="human")
        agent_play = Agent(env_play.observation_space.shape[0], env_play.action_space.shape[0]).to("cpu")
        load_model_safely(agent_play, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        agent_play.eval()
        hist_play = ObsHistory(1, TRANSFORMER_SEQ_LEN, env_play.observation_space.shape[0], "cpu")
        try:
            while True:
                obs_init, _ = env_play.reset()
                hist_play.reset()
                hist_play.update(obs_init)
                is_done = False
                while not is_done:
                    t0_loop = time.time()
                    with torch.no_grad():
                        res_act = agent_play.get_action_and_value(hist_play.get())
                        val_act = res_act[0].cpu().numpy()[0]
                    obs_next, r_step, term_step, trunc_step, info_step = env_play.step(val_act)
                    is_done = term_step or trunc_step
                    hist_play.update(obs_next)
                    env_play.render()
                    # 描画速度調整
                    dt_elapsed = time.time() - t0_loop
                    t_wait_val = (0.005 * ACTION_REPEAT) - dt_elapsed
                    if t_wait_val > 0:
                        time.sleep(t_wait_val)
        except KeyboardInterrupt:
            pass
        finally:
            env_play.close()
        return

    # 【学習モード】
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb
    import main18_optimization as base_config
    from main18_optimization import Agent

    def env_thunk_factory():
        """環境作成のための無名関数。並列ワーカーによって呼び出されます。"""
        new_env_obj = TeamCosEnv()
        return gym.wrappers.RecordEpisodeStatistics(new_env_obj)

    envs_vec = gym.vector.AsyncVectorEnv([env_thunk_factory for _ in range(NUM_ENVS)])
    device_final = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")
    full_experiment_name = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{int(time.time())}"
    
    if TRACK_WANDB:
        wandb.init(project=base_config.WANDB_PROJECT_NAME, config={"Target": TRAIN_TARGET, "v": "25.45_fix_hid"}, name=full_experiment_name, sync_tensorboard=False, save_code=True)
    
    writer_obj = SummaryWriter(f"runs/{full_experiment_name}")
    agent_train = Agent(envs_vec.single_observation_space.shape[0], envs_vec.single_action_space.shape[0]).to(device_final)
    optimizer_obj = optim.Adam(agent_train.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    g_step_counter, start_step_val = 0, 0
    if LOAD_EXISTING_MODELS:
        loaded_path = load_model_safely(agent_train, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if loaded_path:
            json_meta_path = loaded_path.replace('.pt', '_checkpoint.json')
            if os.path.exists(json_meta_path):
                with open(json_meta_path, 'r') as f_meta:
                    meta_data_dict = json.load(f_meta)
                    g_step_counter = meta_data_dict.get('global_step', 0)
                    start_step_val = g_step_counter
                    
    history_train = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device_final)
    
    # ロールアウト・バッファ
    obs_batch = torch.zeros((NUM_STEPS, NUM_ENVS, TRANSFORMER_SEQ_LEN, 53), device=device_final)
    act_batch = torch.zeros((NUM_STEPS, NUM_ENVS, 4), device=device_final)
    lp_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_final)
    rew_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_final)
    done_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_final)
    val_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_final)
    
    seed_final = FIXED_SEED if FIXED_SEED else int(time.time())
    next_obs_v_data, _ = envs_vec.reset(seed=seed_final)
    next_done_v_data = torch.zeros(NUM_ENVS).to(device_final)
    history_train.update(next_obs_v_data)
    
    t0_start = time.time()
    num_total_updates = int(max(1, TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)))
    
    # 統計用リスト
    h_ret_list, h_hid_list = [], []

    print(f"--- Training Started (Step: {g_step_counter}) ---")
    try:
        for update_idx in tqdm(range(1, num_total_updates + 1), desc="Updates"):
            # データ収集フェーズ (Rollout)
            for step_idx in range(NUM_STEPS):
                g_step_counter += NUM_ENVS
                obs_batch[step_idx] = history_train.get()
                done_batch[step_idx] = next_done_v_data
                
                with torch.no_grad():
                    a_out, lp_out, _, v_out = agent_train.get_action_and_value(history_train.get())
                    val_batch[step_idx] = v_out.flatten()
                    
                act_batch[step_idx] = a_out
                lp_batch[step_idx] = lp_out
                
                next_obs_v_data, rew_array, term_array, trunc_array, info_dict = envs_vec.step(a_out.cpu().numpy())
                done_mask_array = np.logical_or(term_array, trunc_array)
                
                # ★修正: 統計情報の堅牢な抽出 (final_info を優先的に走査)
                if "final_info" in info_dict:
                    for f_info in info_dict["final_info"]:
                        if f_info is not None:
                            # 1. 報酬合計の抽出
                            if "episode" in f_info:
                                h_ret_list.append(float(f_info["episode"]["r"]))
                            # 2. カスタム生存時間の抽出
                            if "hidden_steps" in f_info:
                                h_hid_list.append(float(f_info["hidden_steps"]))
                elif "episode" in info_dict:
                    # final_info が無い場合、トップレベルの episode 辞書から取得を試みる
                    m = info_dict.get("_episode", [False] * NUM_ENVS)
                    for i in range(NUM_ENVS):
                        if m[i]:
                            h_ret_list.append(float(info_dict["episode"]["r"][i]))
                
                rew_batch[step_idx] = torch.tensor(rew_array).to(device_final).view(-1)
                next_done_v_data = torch.tensor(done_mask_array).to(device_final, dtype=torch.float32)
                history_train.update(next_obs_v_data)
                
            # 利得計算 (GAE)
            with torch.no_grad():
                v_final_pred = agent_train.get_value(history_train.get()).reshape(1, -1)
                adv_tensor = torch.zeros_like(rew_batch).to(device_final)
                last_gae_val = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        m_done_val = 1.0 - next_done_v_data
                        v_next_val = v_final_pred
                    else:
                        m_done_val = 1.0 - done_batch[t + 1]
                        v_next_val = val_batch[t + 1]
                    delta_val = rew_batch[t] + base_config.GAMMA * v_next_val * m_done_val - val_batch[t]
                    adv_tensor[t] = last_gae_val = delta_val + base_config.GAMMA * base_config.GAE_LAMBDA * m_done_val * last_gae_val
                returns_batch = adv_tensor + val_batch
                
            # 最適化
            f_obs = obs_batch.reshape(-1, TRANSFORMER_SEQ_LEN, 53)
            f_lp = lp_batch.reshape(-1)
            f_act = act_batch.reshape(-1, 4)
            f_adv = adv_tensor.reshape(-1)
            f_ret = returns_batch.reshape(-1)
            
            for epoch_idx in range(UPDATE_EPOCHS):
                idx_perm = np.random.permutation(NUM_STEPS * NUM_ENVS)
                for ptr in range(0, NUM_STEPS * NUM_ENVS, MINIBATCH_SIZE):
                    mb_idx = idx_perm[ptr : ptr + MINIBATCH_SIZE]
                    _, n_lp, ent_batch, n_v = agent_train.get_action_and_value(f_obs[mb_idx], f_act[mb_idx])
                    ratio_val = (n_lp - f_lp[mb_idx]).exp()
                    mb_adv_norm = (f_adv[mb_idx] - f_adv[mb_idx].mean()) / (f_adv[mb_idx].std() + 1e-8)
                    pg_loss = torch.max(-mb_adv_norm * ratio_val, -mb_adv_norm * torch.clamp(ratio_val, 0.8, 1.2)).mean()
                    v_loss = 0.5 * ((n_v.view(-1) - f_ret[mb_idx])**2).mean()
                    loss_total = pg_loss - ENT_COEF * ent_batch.mean() + 0.5 * v_loss
                    optimizer_obj.zero_grad()
                    loss_total.backward()
                    nn.utils.clip_grad_norm_(agent_train.parameters(), 0.5)
                    optimizer_obj.step()

            # ログ表示 (Optuna 用キーワード EpRet: と Hidden: を厳守)
            if (update_idx % 10 == 0) or TRIAL_MODE:
                dt_now = time.time() - t0_start
                sps_now = int((g_step_counter - start_step_val) / dt_now) if dt_now > 0 else 0
                log_map = {"charts/SPS": sps_now, "losses/total_loss": loss_total.item(), "global_step": g_step_counter}
                if h_ret_list:
                    m_ret, m_hid = np.mean(h_ret_list), np.mean(h_hid_list) if h_hid_list else 0.0
                    log_map.update({"charts/episodic_return": m_ret, "charts/steps_hidden": m_hid})
                    tqdm.write(f"Update {update_idx}, Step {g_step_counter}, SPS {sps_now}, EpRet: {m_ret:.1f}, Hidden: {m_hid:.1f}")
                    h_ret_list, h_hid_list = [], []
                elif TRIAL_MODE:
                    tqdm.write(f"Update {update_idx}, Step {g_step_counter}, SPS {sps_now}, Loss: {loss_total.item():.3f}")
                if TRACK_WANDB:
                    wandb.log(log_map)
                    
    except KeyboardInterrupt:
        print("\nInterrupted.")
        
    # モデルの保存
    if SAVE_MODEL:
        torch.save(agent_train.state_dict(), SAVE_MODEL_PATH)
        with open(SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json'), 'w') as f_out:
            json.dump({'global_step': g_step_counter}, f_out)
            
    # 【終了処理ブロック: すべて独立行として記述】
    envs_vec.close()
    writer_obj.close()
    if TRACK_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()