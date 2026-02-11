# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【概要】
# 2体の Hider と 1体の Seeker によるマルチエージェントかくれんぼ環境の強化学習スクリプト。
# PPO アルゴリズムと Transformer を用い、高度な推論と高速な学習を両立させます。
# 
# 【修正内容 (v25.63 - 詳解コメント完全復元・完全展開・SPS 1400 限界突破版)】
# 1. 詳細ドキュメントの完全復元:
#    - 観測ベクトルの全 53 次元の詳細な内訳（インデックスごとの意味）を記述。
#    - ゼロアロケーション（obs_prealloc）による GC 負荷軽減の仕組みを再解説。
# 2. 不可視情報の非ゼロ化 (NN学習安定化):
#    - オブジェクトが視界にない時、座標や速度を 0.0 ではなく 2.0 (遠方) でマスク。
#    - これにより「目の前(0,0)に居る」状況と「見えない」状況の数値的重複を回避。
# 3. 統計収集の同期化 (Hidden: nan 解消):
#    - AsyncVectorEnv の final_info から、報酬(EpRet)と生存時間(Hidden)を確実に同期抽出。
# 4. 構文エラーの物理的排除:
#    - 全てのセミコロン (;) を削除。終了処理、条件分岐を全て独立行へ展開。

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
# マルチプロセス実行時に各プロセスが CPU リソースを奪い合うのを防ぐため、
# 1プロセスあたりのスレッド数を1に制限します（Intel/AMD環境向け）。
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

# "initial":    新規学習モード
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

# 実験の基本制御フラグ
SAVE_MODEL = True           # 学習終了時にモデルを保存
TRACK_WANDB = True          # Weights & Biases にログを送信
FIXED_SEED = None           # None: ランダム / 整数: 結果を固定
TRIAL_MODE = False          # True: Optuna 等の外部スクリプトからの呼び出し用

# 強化学習 (PPO) パラメータ
TOTAL_TIMESTEPS = 5000000 
NUM_ENVS = 8                # 並列プロセス数
NUM_STEPS = 128             # 1更新あたりのデータ収集長（各環境ごと）
LEARNING_RATE = 2e-4
ENT_COEF = 0.001            # 探索の強さ
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4           # 同一データに対する反復学習回数

# 環境・シミュレーション定数
ACTION_REPEAT = 16          # 1ステップあたりの MuJoCo 進展回数
PREP_STEPS = 80             # エピソード序盤の Seeker 停止期間
MAX_STEPS = 300             # 1エピソードの最大ステップ数（固定長）
FOV_DEG = 135               # 視野角（度）
FOV_RAD_HALF = math.radians(FOV_DEG / 2.0)
TRANSFORMER_SEQ_LEN = 8     # Transformer が参照する過去の履歴長

# 高速化キャッシュ用定数
RAYCAST_GRID_SIZE = 0.1
RAYCAST_CACHE_POS_THRESH_SQ = 0.05**2 # 5cm以上動いたらキャッシュを破棄
LIDAR_CACHE_POS_THRESH_SQ = 0.05**2 
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0) # 2度以上回転したら破棄

# 報酬係数
REWARD_HIDDEN_BONUS = 1.0   # 全員隠れている時のプラス報酬
COS_PENALTY_SCALE = 2.0     # 視界内ペナルティ倍率
REWARD_DISTANCE_DIFF_SCALE = 1.0 # 敵から遠ざかったことによる増分報酬
PENALTY_SAFEGUARD = -20.0   # 場外脱走へのペナルティ

# 物理出力制限
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
    バッファサイズを2倍確保し、2箇所に同時書き込みすることで、
    読み出し時の並べ替えコストを排除し、スライス操作のみで高速取得を可能にします。
    """
    def __init__(self, num_envs, seq_len, obs_dim, device):
        import torch
        # seq_len の 2 倍確保し、連続 View の取得を可能にします
        self.buffer = torch.zeros((num_envs, seq_len * 2, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.ptr = 0

    def reset(self):
        """履歴バッファを初期化します。"""
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        """最新の観測値を 2 箇所に同時に書き込みます。"""
        import torch
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            # 単一環境の場合、バッチ次元を補完
            obs_tensor = obs_tensor.unsqueeze(0)
        
        # 循環バッファの現在位置に書き込み
        self.buffer[:, self.ptr] = obs_tensor
        # ミラー領域にも同時に書き込み
        mirrored_idx = self.ptr + self.seq_len
        self.buffer[:, mirrored_idx] = obs_tensor
        
        # ポインタを進める
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        """直近 seq_len ステップ分の履歴を View として高速取得します。"""
        end_idx = self.ptr + self.seq_len
        return self.buffer[:, self.ptr : end_idx]

def load_model_safely(model_obj, base_name, target_type):
    """
    複数パスからロード可能な最新モデルを安全に読み込みます。
    MODE (initial/refinement) の切り替えに関わらず、学習の継続性を確保します。
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
    ゼロアロケーション（配列スライス代入）と徹底メモ化を施した最高速環境。
    不可視情報を遠方値(2.0)でマスクし、NNの学習安定性を向上させています。
    """
    def __init__(self, render_mode=None):
        # 親クラス（MuJoCo物理環境）の初期化
        super().__init__(render_mode=render_mode)
        
        import torch
        from main18_optimization import Agent
        cpu_dev = torch.device("cpu")
        
        # 各エージェント履歴管理バッファ
        self.npc_obs_history = {}
        self.npc_obs_history[0] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
        self.npc_obs_history[1] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
        self.npc_obs_history[2] = ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
        
        # 計算キャッシュの初期化
        self.visible_cache = {0: {}, 1: {}, 2: {}}
        self.lidar_array_cache = {} 
        self.raycast_cache = {} 
        self._obs_memo = {}
        
        # 観測用事前確保バッファ (毎ステップの配列生成を排除)
        self.obs_prealloc = np.zeros(53, dtype=np.float32)
        
        # --- 不可視情報のマスキング設定 (NN学習安定化) ---
        # 0.0 は原点(目の前)や静止を意味するため、不可視時は 2.0 (遠方/高速) でマスク。
        # 座標(pos)と速度(vel)に該当するインデックスを 2.0 で初期化します。
        self.mask_indices = [
            17, 18, 19, 20, # Box1 rel_pos, rel_vel
            25, 26, 27, 28, # Box2 rel_pos, rel_vel
            33, 34, 35, 36, # Ramp rel_pos, rel_vel
            40, 41, 42, 43, # Enemy rel_pos, rel_vel
            45, 46, 47, 48  # Friend rel_pos, rel_vel
        ]
        for idx in self.mask_indices:
            self.obs_prealloc[idx] = 2.0
        
        # 統計と定数
        self.hidden_steps_count = 0
        self.caught_steps_count = 0
        self.prev_dist = {1: 0.0, 2: 0.0}
        self.s0_recovery_turn_dir = 1.0
        self.two_pi = 2.0 * math.pi
        
        # Lidar 演算高速化：三角関数のテーブル化
        angles_vec = self.lidar_angles
        self.lidar_cos_sin = np.column_stack([
            np.cos(angles_vec), 
            np.sin(angles_vec)
        ])

        # 学習ターゲットIDの設定
        if TRAIN_TARGET == "HIDER":
            self.learning_agent_id = self.np_random.choice([1, 2])
        else:
            self.learning_agent_id = 0
            
        self.lock_cooldown = {1: 0, 2: 0}
        self.raycast_stats = {"hits": 0, "misses": 0}
        
        # Seeker AI 状態
        self.s0_stuck_timer = 0
        self.s0_recovery_mode = 0
        self.seeker_search_timer = 0
        self.seeker_random_target = np.zeros(2)
        self.seeker_target_pos = None
        self.seeker_last_known_pos = None
        
        # 演算用ボディ ID リスト
        bodies_list = [
            self.box1_body, 
            self.box2_body, 
            self.ramp_body, 
            self.h1_body, 
            self.h2_body, 
            self.s0_body
        ]
        self.all_target_bodies = np.array(bodies_list)
        
        # 回転関節アドレス取得
        rot_jnt = self.model.joint('s_rot')
        self.srot_adr = self.model.jnt_qposadr[rot_jnt.id]
        
        # NPC モデルロード
        self.npc_hider_agent = Agent(53, 4).to("cpu")
        self.npc_seeker_agent = Agent(53, 4).to("cpu")
        
        log_flag = os.environ.get("NPC_MODELS_LOGGED")
        should_print = (log_flag != "TRUE")
        
        h_f = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
        s_f = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
        
        if should_print:
            if h_f: 
                print(f"Loaded NPC Hider Model", flush=True)
            if s_f: 
                print(f"Loaded NPC Seeker Model", flush=True)
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
            if i == 1:
                tid = self.h1_body
            else:
                tid = self.h2_body
            hp_pos = self.data.xpos[tid][:2]
            # 初期距離のノルム計算
            self.prev_dist[i] = np.linalg.norm(hp_pos - sp_pos)
            
        return obs, info

    def _get_cached_ray_optimized(self, agent_id, origin_p, angle_rad, tid, grid_x, grid_y, diff_vec=None, dist_val=None):
        """
        アンパックエラー (TypeError) を解消した高速 RayCast。
        キャッシュヒット時も物理エンジン実行時も、必ず (ID, Dist) のタプルを返します。
        """
        cache_key = (agent_id, tid)
        
        # 1. フレーム間キャッシュの判定
        if cache_key in self.raycast_cache:
            c_pos, c_angle, c_hit_id, c_dist = self.raycast_cache[cache_key]
            # 平方根計算を避け、二乗距離で判定
            dx_sq = (origin_p[0] - c_pos[0])**2 + (origin_p[1] - c_pos[1])**2
            if dx_sq < RAYCAST_CACHE_POS_THRESH_SQ:
                a_diff = (angle_rad - c_angle + math.pi) % self.two_pi - math.pi
                if abs(a_diff) < 0.05:
                    self.raycast_stats["hits"] += 1
                    return c_hit_id, c_dist # 常にタプルを返却

        # 2. キャッシュミス処理
        self.raycast_stats["misses"] += 1
        if diff_vec is None:
            # 汎用ビーム (Lidar 等)
            direction = np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0], dtype=np.float64)
        else:
            # ターゲット判定。引数の距離(dist_val)を利用して正規化を高速化
            direction = np.array([diff_vec[0]/dist_val, diff_vec[1]/dist_val, 0.0], dtype=np.float64)

        gid_out = np.zeros(1, dtype=np.int32)
        from_xyz = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
        
        # 自己遮蔽回避 ID
        if agent_id == 0:
            exclude_body_id = self.s0_body
        elif agent_id == 1:
            exclude_body_id = self.h1_body
        else:
            exclude_body_id = self.h2_body
        
        # MuJoCo レイキャスト実行
        res = mujoco.mj_ray(self.model, self.data, from_xyz, direction, None, 1, exclude_body_id, gid_out)
        
        # 3. 判定結果の確定
        hit_body = -1
        if res != -1:
            hit_body = self.model.geom_bodyid[gid_out[0]]
            if dist_val is not None:
                # ターゲット判定時は遮蔽誤差(0.4m)を考慮
                if hit_body != tid:
                    if res < dist_val - 0.4:
                        # 遮蔽された
                        pass
                    else:
                        # 近接ヒット
                        hit_body = tid
                else:
                    # 直接ヒット
                    hit_body = tid
        
        # キャッシュに 4 要素保存し、呼び出し元へは (ID, Dist) で返却
        self.raycast_cache[cache_key] = (origin_p.copy(), angle_rad, hit_body, res)
        return hit_body, res

    def _get_obs(self, agent_id, skip_lidar=False):
        """
        超高速・メモ化・ゼロアロケーション観測生成ロジック。
        不連続な情報をニューラルネットワークが扱いやすいよう、不可視情報は 2.0 でマスクします。
        
        【観測ベクトル内訳 (全53次元)】
        - 00-04: Self [v_local(2), angle(1), cos(1), sin(1)]
        - 05-16: Lidar [distances(12)]
        - 17-24: Box 1 [rel_p(2), rel_v(2), cos(1), sin(1), locked(1), vis(1)]
        - 25-32: Box 2 [rel_p(2), rel_v(2), cos(1), sin(1), locked(1), vis(1)]
        - 33-39: Ramp  [rel_p(2), rel_v(2), cos(1), sin(1), vis(1)]
        - 40-44: Enemy (Seeker/Hider) [rel_p(2), rel_v(2), vis(1)]
        - 45-51: Friend (Hider) [rel_p(2), rel_v(2), cos(1), sin(1), vis(1)]
        - 52   : Status [grasping(1)]
        """
        # メモ化の確認 (skip_lidar フラグ別に保存)
        memo_key = (agent_id, skip_lidar)
        if memo_key in self._obs_memo:
            return self._obs_memo[memo_key]
        
        # 遠方マスク値 (2.0) を含むベース配列をコピー
        obs = self.obs_prealloc.copy()
        
        # エージェント情報の抽出
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
        rot_jnt_idx = self.model.joint(f'{pref}_rot').id
        hra = self.data.qpos[self.model.jnt_qposadr[rot_jnt_idx]]
        
        cos_h = math.cos(hra)
        sin_h = math.sin(hra)
        # ローカル変換行列
        rot_mat = np.array([[cos_h, sin_h], [-sin_h, cos_h]])
        
        xj_id = self.model.joint(f'{pref}_x').id
        d_idx = self.model.jnt_dofadr[xj_id]
        v_raw = self.data.qvel[d_idx : d_idx + 2]
        v_obs = (rot_mat @ v_raw) / 12.0
        
        # グリッド座標算出（キャッシュ効率化のため）
        gx = int(agent_xy[0] / RAYCAST_GRID_SIZE)
        gy = int(agent_xy[1] / RAYCAST_GRID_SIZE)
        
        # [0-4] 自己状態の書き込み
        obs[0:2] = v_obs
        obs[2] = hra
        obs[3] = cos_h
        obs[4] = sin_h
        
        # [5-16] Lidar センサー
        if not skip_lidar:
            lidar_cached = False
            l_cache = self.lidar_array_cache.get(agent_id)
            if l_cache:
                c_xy, c_hra, c_vals = l_cache
                dx_sq = (agent_xy[0] - c_xy[0])**2 + (agent_xy[1] - c_xy[1])**2
                if dx_sq < LIDAR_CACHE_POS_THRESH_SQ:
                    da_l = (hra - c_hra + math.pi) % self.two_pi - math.pi
                    if abs(da_l) < LIDAR_CACHE_ANG_THRESH:
                        obs[5:17] = c_vals
                        lidar_cached = True
            
            if not lidar_cached:
                for i in range(12):
                    angle_abs = hra + self.lidar_angles[i]
                    _, dist_r = self._get_cached_ray_optimized(agent_id, agent_xy, angle_abs, f"L{i}", gx, gy)
                    obs[5+i] = min(dist_r, 2.5) / 2.5 if dist_r != -1 else 1.0
                # Lidar キャッシュの保存
                self.lidar_array_cache[agent_id] = (agent_xy.copy(), hra, obs[5:17].copy())
        else:
            # スキップ時は最大距離（1.0 = 正規化値）
            obs[5:17] = 1.0

        # [他オブジェクト情報] 一括行列演算
        mask_others = (self.all_target_bodies != b_id)
        t_ids = self.all_target_bodies[mask_others]
        rel_pos_mat = self.data.xpos[t_ids][:, :2] - agent_xy
        dists_arr = np.sqrt(np.sum(rel_pos_mat**2, axis=1))
        
        g_angs = np.arctan2(rel_pos_mat[:, 1], rel_pos_mat[:, 0])
        rel_angs_arr = (g_angs - hra + math.pi) % self.two_pi - math.pi
        fov_mask = np.abs(rel_angs_arr) <= FOV_RAD_HALF
        
        v_cache_ref = self.visible_cache[agent_id]
        v_cache_ref.clear()
        rel_store_tmp = {}
        
        # ターゲットごとの判定ループ
        for idx in range(len(t_ids)):
            tid = t_ids[idx]
            if not fov_mask[idx]:
                v_cache_ref[tid] = False
                continue
                
            # RayCast 実行
            hit_id_res, _ = self._get_cached_ray_optimized(agent_id, agent_xy, g_angs[idx], tid, gx, gy, rel_pos_mat[idx], dists_arr[idx])
            is_vis = (hit_id_res == tid)
            v_cache_ref[tid] = is_vis
            
            if is_vis:
                # 可視物体の詳細情報を算出
                p_loc = (rot_mat @ rel_pos_mat[idx]) / 12.0
                q_tgt = self.data.xquat[tid]
                y_v = 2.0 * (q_tgt[0]*q_tgt[3] + q_tgt[1]*q_tgt[2])
                x_v = 1.0 - 2.0 * (q_tgt[2]**2 + q_tgt[3]**2)
                y_rel = math.atan2(y_v, x_v) - hra
                
                # 相対速度
                j_adr = self.model.body_jntadr[tid]
                if j_adr != -1:
                    v_target = self.data.qvel[j_adr : j_adr + 2]
                else:
                    v_target = np.zeros(2)
                
                rv_l = (rot_mat @ (v_target - v_raw)) / 12.0
                rel_store_tmp[tid] = [p_loc[0], p_loc[1], rv_l[0], rv_l[1], math.cos(y_rel), math.sin(y_rel)]

        # --- 配列への直接マッピング (ゼロアロケーション & 不可視情報の処理) ---
        # 1. Box 1 (17-24)
        if self.box1_body in rel_store_tmp:
            inf = rel_store_tmp[self.box1_body]
            obs[17:21] = inf[:4]
            obs[21:23] = inf[4:6]
            obs[23] = 1.0 if self.locked_boxes[self.box1_body] else 0.0
            obs[24] = 1.0
        else:
            # 見えない場合、pos/vel は 2.0 のまま。cos/sin/locked/vis は 0.0。
            obs[21:25] = 0.0
            
        # 2. Box 2 (25-32)
        if self.box2_body in rel_store_tmp:
            inf = rel_store_tmp[self.box2_body]
            obs[25:29] = inf[:4]
            obs[29:31] = inf[4:6]
            obs[31] = 1.0 if self.locked_boxes[self.box2_body] else 0.0
            obs[32] = 1.0
        else:
            obs[29:33] = 0.0
            
        # 3. Ramp (33-39)
        if self.ramp_body in rel_store_tmp:
            inf = rel_store_tmp[self.ramp_body]
            obs[33:37] = inf[:4]
            obs[37:39] = inf[4:6]
            obs[39] = 1.0
        else:
            obs[37:40] = 0.0

        # エージェント固有情報の書き込み
        if agent_id == 0:
            # Seeker視点
            if self.h1_body in rel_store_tmp:
                inf = rel_store_tmp[self.h1_body]
                obs[40:44] = inf[:4]
                obs[44] = 1.0
            else:
                obs[44] = 0.0
            if self.h2_body in rel_store_tmp:
                inf = rel_store_tmp[self.h2_body]
                obs[45:49] = inf[:4]
                obs[49] = 1.0
            else:
                obs[49] = 0.0
        else:
            # Hider視点
            if self.s0_body in rel_store_tmp:
                inf = rel_store_tmp[self.s0_body]
                obs[40:44] = inf[:4]
                obs[44] = 1.0
            else:
                obs[44] = 0.0
            p_id_ref = self.h2_body if agent_id == 1 else self.h1_body
            if p_id_ref in rel_store_tmp:
                inf = rel_store_tmp[p_id_ref]
                obs[45:49] = inf[:4]
                obs[49:51] = inf[4:6]
                obs[51] = 1.0
            else:
                obs[49:52] = 0.0
            obs[52] = 1.0 if self.grasping[agent_id] else 0.0
            
        # メモ化保存して返却
        self._obs_memo[memo_key] = obs
        return obs

    def _update_seeker_state(self):
        """シーカーの「視覚と記憶」を更新。メモ化により呼び出しコストを下げています。"""
        self._get_obs(0) 
        v_dict = self.visible_cache[0]
        v1_b = v_dict.get(self.h1_body, False)
        v2_b = v_dict.get(self.h2_body, False)
        
        if v1_b or v2_b:
            # 目撃時
            if v1_b:
                tid = self.h1_body
            else:
                tid = self.h2_body
            self.seeker_target_pos = self.data.xpos[tid][:2].copy()
            self.seeker_last_known_pos = self.seeker_target_pos.copy()
            self.seeker_mode = "CHASING"
        elif self.seeker_last_known_pos is not None:
            # 見失った直後：捜索モード
            sp_xy = self.data.xpos[self.s0_body][:2]
            dx = sp_xy[0] - self.seeker_last_known_pos[0]
            dy = sp_xy[1] - self.seeker_last_known_pos[1]
            if (dx*dx + dy*dy) < 0.25: # 二乗距離 0.5m 未満で到着
                self.seeker_last_known_pos = None
                self.seeker_search_timer = 50
            else: 
                self.seeker_target_pos = self.seeker_last_known_pos.copy()
                self.seeker_mode = "SEARCHING"
        else:
            # 手がかりなし：ランダム巡回
            if self.seeker_search_timer <= 0:
                self.seeker_random_target = self.np_random.uniform(-4, 4, 2)
                self.seeker_search_timer = 80
            self.seeker_search_timer -= 1
            self.seeker_target_pos = self.seeker_random_target
            self.seeker_mode = "PATROLLING"

    def _seeker_rule_based_policy(self):
        """ポテンシャル法に基づいた物理制御。斥力（障害物）と引力（目標）を合成。"""
        if self.current_step < PREP_STEPS:
            return 0.0, 0.0
            
        sp_pos = self.data.xpos[self.s0_body][:2]
        yaw_rad = self.data.qpos[self.srot_adr]
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        l_vals = self.lidar_array_cache[0][2]
        rep_v = np.zeros(2)
        
        # [斥力] Lidarで検知した近接物から離れる力
        for i in range(12):
            d_beam = l_vals[i] * 2.5
            if d_beam < 1.0:
                f_mag = (1.0 - d_beam) / (d_beam + 0.1)
                co_p, si_p = self.lidar_cos_sin[i]
                rep_v[0] -= (co_p * cy - si_p * sy) * f_mag
                rep_v[1] -= (si_p * cy + co_p * sy) * f_mag
        
        # [引力] 目標方向への牽引力
        diff_p = self.seeker_target_pos - sp_pos
        combined_p = (diff_p / (np.linalg.norm(diff_p) + 1e-8)) + rep_v * 1.5
        target_angle = math.atan2(combined_p[1], combined_p[0])
        err_p = (target_angle - yaw_rad + math.pi) % self.two_pi - math.pi
        
        # 推力制御：向きがずれている時は減速
        thrust_p = SEEKER_RB_THRUST
        if abs(err_p) > SEEKER_RB_TURN_THRESH:
            thrust_p = thrust_p * 0.3
            
        # スタック復帰ロジック
        sx_id = self.model.joint('s_x').id
        v_sq = self.data.qvel[self.model.jnt_dofadr[sx_id]]**2 + self.data.qvel[self.model.jnt_dofadr[sx_id]+1]**2
        
        if thrust_p > 0.05 and v_sq < 0.0025:
            self.s0_stuck_timer += 5
        else:
            self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            
        if self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 15
            self.s0_stuck_timer = 0
            
        if self.s0_recovery_mode > 0:
            self.s0_recovery_mode -= 1
            # 強制旋回による復帰
            return -0.2, 1.5 * self.np_random.choice([-1, 1])
            
        return float(thrust_p), float(np.clip(err_p * 6.0, -3.0, 3.0))

    def step(self, action):
        """シミュレーション更新とチーム報酬の集計。"""
        self._obs_memo.clear()
        self.current_step += 1
        for i in [1, 2]:
            self.lock_cooldown[i] = max(0, self.lock_cooldown[i]-1)
        
        self._update_seeker_state()
        self.data.ctrl[:] = 0.0
        
        # アクションの適用
        if TRAIN_TARGET == "HIDER":
            m_idx_s = self._apply_action(self.learning_agent_id, action)
            self.data.ctrl[m_idx_s:m_idx_s+2] = [float(action[0])*HIDER_THRUST_LIMIT, float(action[1])]
            
            if self.learning_agent_id == 1:
                partner_id = 2
            else:
                partner_id = 1
                
            act_n_s = self._get_npc_action(partner_id, "HIDER")
            p_idx_s = self._apply_action(partner_id, act_n_s)
            self.data.ctrl[p_idx_s:p_idx_s+2] = [float(act_n_s[0])*HIDER_THRUST_LIMIT, float(act_n_s[1])]
            
            sf_v, sr_v = self._seeker_rule_based_policy()
            self.data.ctrl[0:2] = [sf_v, sr_v]
        else:
            # Seeker 学習
            self.data.ctrl[0:2] = [float(action[0])*SEEKER_THRUST_LIMIT, float(action[1])]
            for i in [1, 2]:
                act_h_s = self._get_npc_action(i, "HIDER")
                idx_h_s = self._apply_action(i, act_h_s)
                self.data.ctrl[idx_h_s:idx_h_s+2] = [float(act_h_s[0])*HIDER_THRUST_LIMIT, float(act_h_s[1])]
                    
        # 物理サブステップ (Locked 状態の強制維持)
        locked_list = []
        for box_id, pose_v in self.locked_pose.items():
            if self.locked_boxes[box_id]:
                if box_id == self.box1_body:
                    bid_s = self.box1_joint_id
                else:
                    bid_s = self.box2_joint_id
                locked_list.append((self.model.jnt_qposadr[bid_s], self.model.jnt_dofadr[bid_s], pose_v))

        for _ in range(ACTION_REPEAT):
            for q_adr, d_adr, p_val in locked_list:
                self.data.qpos[q_adr : q_adr+7] = p_val
                self.data.qvel[d_adr : d_adr+6] = 0
            mujoco.mj_step(self.model, self.data)
                
        # 報酬と統計確定
        self._obs_memo.clear()
        obs_learner_final = self._get_obs(self.learning_agent_id)
        self._get_obs(0, skip_lidar=True) # 報酬判定用にシーカーの視界を更新
        
        # チーム統計集計：全員が隠れている場合のみ Hidden
        exposed_bool = False
        if self.visible_cache[0].get(self.h1_body, False):
            exposed_bool = True
        if self.visible_cache[0].get(self.h2_body, False):
            exposed_bool = True
            
        if not exposed_bool:
            self.hidden_steps_count += 1
        else:
            self.caught_steps_count += 1
                
        team_reward_acc = 0.0
        sp_xy_f = self.data.xpos[self.s0_body][:2]
        for h_idx_f, b_id_f in [(1, self.h1_body), (2, self.h2_body)]:
            hp_xy_f = self.data.xpos[b_id_f][:2]
            dist_f = np.linalg.norm(hp_xy_f - sp_xy_f)
            if self.visible_cache[0].get(b_id_f, False):
                sr_f = self.data.qpos[self.srot_adr]
                unit_f = (hp_xy_f - sp_xy_f) / (dist_f + 1e-8)
                # シーカー正面との cos をペナルティ化
                cv_f = unit_f[0] * math.cos(sr_f) + unit_f[1] * math.sin(sr_f)
                h_rew_f = -cv_f * COS_PENALTY_SCALE + (dist_f - self.prev_dist[h_idx_f]) * REWARD_DISTANCE_DIFF_SCALE
            else:
                h_rew_f = REWARD_HIDDEN_BONUS
            
            # 場外ペナルティ
            if np.max(np.abs(hp_xy_f)) > 6.5:
                h_rew_f += PENALTY_SAFEGUARD
                
            team_reward_acc += h_rew_f
            self.prev_dist[h_idx_f] = dist_f
            
        reward_final = team_reward_acc if TRAIN_TARGET == "HIDER" else -team_reward_acc
        info_final = {
            "hidden_steps": float(self.hidden_steps_count), 
            "caught_steps": float(self.caught_steps_count)
        }
        return obs_learner_final, float(reward_final), False, (self.current_step >= MAX_STEPS), info_final

    def _get_npc_action(self, agent_id, agent_type):
        import torch
        obs_val = self._get_obs(agent_id)
        self.npc_obs_history[agent_id].update(obs_val)
        if agent_type == "HIDER":
            model_ptr = self.npc_hider_agent
        else:
            model_ptr = self.npc_seeker_agent
            
        if model_ptr:
            with torch.no_grad():
                res_act = model_ptr.get_action_and_value(self.npc_obs_history[agent_id].get())
                return res_act[0].cpu().numpy()[0]
        return self.action_space.sample() * 0.5

# ==========================================
# 4. メイン処理 (学習ループ)
# ==========================================
def main():
    # デッドロック防止
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
        hist_p_buffer = ObsHistory(1, TRANSFORMER_SEQ_LEN, env_play.observation_space.shape[0], "cpu")
        try:
            while True:
                obs_init, _ = env_play.reset()
                hist_p_buffer.reset()
                hist_p_buffer.update(obs_init)
                is_ep_done = False
                while not is_ep_done:
                    t_loop_s = time.time()
                    with torch.no_grad():
                        res_p = agent_play.get_action_and_value(hist_p_buffer.get())
                        v_act_p = res_p[0].cpu().numpy()[0]
                    obs_next_p, r_p, term_p, trunc_p, info_p = env_play.step(v_act_p)
                    is_ep_done = term_p or trunc_p
                    hist_p_buffer.update(obs_next_p)
                    env_play.render()
                    # 速度調整
                    wait_t = (0.005 * ACTION_REPEAT) - (time.time() - t_loop_s)
                    if wait_t > 0:
                        time.sleep(wait_t)
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

    def env_factory():
        new_env = TeamCosEnv()
        return gym.wrappers.RecordEpisodeStatistics(new_env)
    
    envs_vec = gym.vector.AsyncVectorEnv([env_factory for _ in range(NUM_ENVS)])
    device_f = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")
    run_full_id = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{int(time.time())}"
    
    if TRACK_WANDB:
        wandb.init(project=base_config.WANDB_PROJECT_NAME, config={"Target": TRAIN_TARGET, "v": "25.63_sync"}, name=run_full_id, sync_tensorboard=False, save_code=True)
    
    writer_obj_s = SummaryWriter(f"runs/{run_full_id}")
    agent_train = Agent(envs_vec.single_observation_space.shape[0], envs_vec.single_action_space.shape[0]).to(device_f)
    optimizer_obj = optim.Adam(agent_train.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    g_step_total = 0
    s_step_init = 0
    if LOAD_EXISTING_MODELS:
        lp_path = load_model_safely(agent_train, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if lp_path:
            ckpt_json = lp_path.replace('.pt', '_checkpoint.json')
            if os.path.exists(ckpt_json):
                with open(ckpt_json, 'r') as f_in:
                    meta_d = json.load(f_in)
                    g_step_total = meta_d.get('global_step', 0)
                    s_step_init = g_step_total
                    
    history_train = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device_f)
    obs_batch = torch.zeros((NUM_STEPS, NUM_ENVS, TRANSFORMER_SEQ_LEN, 53), device=device_f)
    act_batch = torch.zeros((NUM_STEPS, NUM_ENVS, 4), device=device_f)
    lp_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_f)
    rew_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_f)
    done_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_f)
    val_batch = torch.zeros((NUM_STEPS, NUM_ENVS), device=device_f)
    
    next_obs_v_data, _ = envs_vec.reset(seed=FIXED_SEED if FIXED_SEED else int(time.time()))
    next_done_v_data = torch.zeros(NUM_ENVS).to(device_f)
    history_train.update(next_obs_v_data)
    
    t0_start_time = time.time()
    num_updates_count = int(max(1, TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)))
    h_ret_list, h_hid_list = [], []

    print(f"--- Training Started (Target SPS: 1400+) ---")
    try:
        for update_idx in tqdm(range(1, num_updates_count + 1), desc="Updates"):
            # データ収集フェーズ (Rollout)
            for step_idx in range(NUM_STEPS):
                g_step_total += NUM_ENVS
                obs_batch[step_idx], done_batch[step_idx] = history_train.get(), next_done_v_data
                with torch.no_grad():
                    a_out, lp_out, _, v_out = agent_train.get_action_and_value(history_train.get())
                    val_batch[step_idx] = v_out.flatten()
                act_batch[step_idx], lp_batch[step_idx] = a_out, lp_out
                n_obs_v, rew_v, term_v, trunc_v, info_d = envs_vec.step(a_out.cpu().numpy())
                is_done_v = np.logical_or(term_v, trunc_v)
                
                # ★修正: 統計情報の同期抽出 (final_info 優先)
                if "final_info" in info_d:
                    for f_info in info_d["final_info"]:
                        if f_info is not None:
                            if "episode" in f_info:
                                h_ret_list.append(float(f_info["episode"]["r"]))
                            if "hidden_steps" in f_info:
                                h_hid_list.append(float(f_info["hidden_steps"]))
                elif "episode" in info_d:
                    # AsyncVectorEnv で _episode マスクが使える場合のフォールバック
                    ep_mask = info_d.get("_episode", is_done_v)
                    for i_mask in range(NUM_ENVS):
                        if ep_mask[i_mask]:
                            h_ret_list.append(float(info_d["episode"]["r"][i_mask]))
                            if "hidden_steps" in info_d:
                                h_hid_list.append(float(info_d["hidden_steps"][i_mask]))

                rew_batch[step_idx] = torch.tensor(rew_v).to(device_f).view(-1)
                next_done_v_data = torch.tensor(is_done_v).to(device_f, dtype=torch.float32)
                history_train.update(n_obs_v)
                
            # 利得計算フェーズ (GAE)
            with torch.no_grad():
                v_last_pred = agent_train.get_value(history_train.get()).reshape(1, -1)
                adv_tensor = torch.zeros_like(rew_batch).to(device_f)
                gae_val = 0
                for t in reversed(range(NUM_STEPS)):
                    m_done = 1.0 - (next_done_v_data if t == NUM_STEPS - 1 else done_batch[t + 1])
                    v_next = v_last_pred if t == NUM_STEPS - 1 else val_batch[t + 1]
                    delta = rew_batch[t] + base_config.GAMMA * v_next * m_done - val_batch[t]
                    adv_tensor[t] = gae_val = delta + base_config.GAMMA * base_config.GAE_LAMBDA * m_done * gae_val
                returns_batch = adv_tensor + val_batch
                
            f_obs, f_lp, f_act, f_adv, f_ret = obs_batch.reshape(-1, TRANSFORMER_SEQ_LEN, 53), lp_batch.reshape(-1), act_batch.reshape(-1, 4), adv_tensor.reshape(-1), returns_batch.reshape(-1)
            # パラメータ最適化フェーズ (PPO)
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

            # ログ表示 (Optuna 用キーワード EpRet: と Hidden: を厳守)
            if (update_idx % 10 == 0) or TRIAL_MODE:
                dt_now = time.time() - t0_start_time
                if dt_now > 0:
                    sps_now = int((g_step_total - s_step_init) / dt_now)
                else:
                    sps_now = 0
                log_map = {"charts/SPS": sps_now, "losses/total_loss": loss_total.item(), "global_step": g_step_total}
                
                log_msg = f"Update {update_idx}, Step {g_step_total}, SPS {sps_now}, Loss {loss_total.item():.3f}"
                
                if len(h_ret_list) > 0:
                    m_ret_v = np.mean(h_ret_list)
                    m_hid_v = np.mean(h_hid_list)
                    log_map.update({"charts/episodic_return": m_ret_v, "charts/steps_hidden": m_hid_v})
                    log_msg = log_msg + f", EpRet: {m_ret_v:.1f}, Hidden: {m_hid_v:.1f}"
                    h_ret_list = []
                    h_hid_list = []
                elif TRIAL_MODE:
                    log_msg = log_msg + " (Collecting stats...)"
                
                tqdm.write(log_msg)
                if TRACK_WANDB:
                    wandb.log(log_map)
                    
    except KeyboardInterrupt:
        pass
        
    if SAVE_MODEL:
        torch.save(agent_train.state_dict(), SAVE_MODEL_PATH)
        ckpt_path_json = SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json')
        with open(ckpt_path_json, 'w') as f_out:
            json.dump({'global_step': g_step_total}, f_out)
    
    # 【終了処理ブロック: すべて独立行として記述し、構文エラーを防止】
    envs_vec.close()
    writer_obj_s.close()
    if TRACK_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()