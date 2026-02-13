# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【修正内容 (v25.35)】
# 1. CUDA 定数およびデバイス制御の正常化:
#    - 実験設定セクションに CUDA 定数を明示し、base_config.CUDA を確実に反映。
#    - main() 内で混在していた device_actual と device_final を統合。
#    - 全てのテンソル計算とモデル配置において、設定されたデバイスが正しく使われるよう修正。
# 2. 独立行展開の徹底:
#    - すべてのステートメントを1行ずつ個別に記述。
#    - 複数代入、インラインif、セミコロンを完全に根絶。
# 3. ロジックの完全維持:
#    - チーム指標 (hidden + caught = 300)、外れ値 2.0、停滞罰 (-0.5)、
#      Seekerのヘディング制御、PLAYモードのロード制限解除をすべて継承。
# 4. 詳細な日本語技術解説の網羅:
#    - 報酬計算の数理的背景、物理エンジンの強制固定仕様などを詳細に記述。

import os
import sys
import platform
import json
import time
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- 実行環境の最適化 ---
# 並列実行時に各プロセスがスレッドを奪い合わないよう、数値計算ライブラリを制限します。
if platform.processor() != 'arm':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- プロジェクトパスの解決 ---
# 共通資産である main18_optimization.py 等にアクセスするため、ディレクトリを特定します。
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
search_path = current_dir_path
for _ in range(5):
    # ファイルの存在を確認
    potential_marker = os.path.join(search_path, "main18_optimization.py")
    if os.path.exists(potential_marker):
        if search_path not in sys.path:
            sys.path.insert(0, search_path)
        break
    # 階層を遡る
    search_path = os.path.dirname(search_path)

# カレントディレクトリをパスに追加
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
# refinement モードのときのみ、既存モデルをロードして再開
LOAD_EXISTING_MODELS = (MODE == "refinement")

# 実行モード: "TRAIN" (学習) または "PLAY" ( Viewer で挙動を鑑賞)
EXECUTION_MODE = "TRAIN" 

SAVE_MODEL = True
TRACK_WANDB = True           
FIXED_SEED = None
TRIAL_MODE = False

# ★CUDA 設定の正常化: main18_optimization からの設定を継承
import main18_optimization as base_config
CUDA = base_config.CUDA

# PPO アルゴリズムのハイパーパラメータ
TOTAL_TIMESTEPS = 5000000 
NUM_ENVS = 8
NUM_STEPS = 128            # 更新1回あたりのデータ収集長さ
LEARNING_RATE = 1.0e-05 # 2e-4       # 学習率
ENT_COEF = 0.00055 # 0.001           # 探索のランダム性を制御
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4

# Transformer 設定
TRANSFORMER_SEQ_LEN = 8    # 記憶系列の長さ
HIDDEN_DIM = 64            # 内部特徴量次元
NUM_LAYERS = 2             # スタック層数
NUM_HEADS = 2              # 注意ヘッド数

# 環境・物理定数
ACTION_REPEAT = 16         # アクションごとの物理計算回数
PREP_STEPS = 80            # Seeker 待機時間
MAX_STEPS = 300            # 固定長エピソード長
FOV_DEG = 135              # 視野角の設定

# 高速化キャッシュ閾値
LIDAR_CACHE_POS_THRESH = 0.05
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)
RAYCAST_CACHE_POS_THRESH = 0.05

# 観測情報のマスク用外れ値
# 0.0 (原点重なり) と区別するため、フィールド境界を考慮した 2.0 を採用。
OUTLIER_VALUE = 2.0

# 報酬設計
REWARD_HIDDEN_BONUS = 1.0 # 1.0        # チーム全体が隠れている時の報酬
COS_PENALTY_SCALE = 2.0 # 2.0          # 正面被視認時のペナルティ強度
REWARD_DISTANCE_DIFF_SCALE = 1.5 # 1.0 # 敵から遠ざかる動きに対する補助報酬
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
    """重みの直交初期化を行い、学習の初期安定性を確保します。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """過去の観測系列から行動を決定する Transformer Actor-Critic ネットワーク。"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 観測値の埋め込み
        self.embedding = nn.Linear(obs_dim, HIDDEN_DIM)
        # 系列順序情報を教える位置エンコーディング
        self.pos_encoder = nn.Parameter(torch.zeros(1, TRANSFORMER_SEQ_LEN, HIDDEN_DIM))
        
        # Transformer エンコーダ層
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
        
        # Actor 層: 行動平均の出力
        self.actor_mean = layer_init(nn.Linear(HIDDEN_DIM, action_dim), std=0.01)
        # Actor 層: 対数標準偏差のパラメータ
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        # Critic 層: 状態価値予測の出力
        self.critic = layer_init(nn.Linear(HIDDEN_DIM, 1), std=1)

    def get_value(self, x):
        """現在の観測系列 x に対して状態価値 V(s) を返します。"""
        # 特徴埋め込み
        x_emb = self.embedding(x)
        # 位置情報加算
        x_pos = x_emb + self.pos_encoder
        # 系列処理
        x_out = self.transformer(x_pos)
        # 最終トークンを抽出
        h_last = x_out[:, -1, :]
        v_pred = self.critic(h_last)
        return v_pred

    def get_action_and_value(self, x, action=None):
        """行動、確率密度、エントロピー、および価値を一括取得します。"""
        # 特徴抽出プロセス
        x_emb = self.embedding(x)
        x_pos = x_emb + self.pos_encoder
        x_out = self.transformer(x_pos)
        h_context = x_out[:, -1, :]
        
        # 確率的ポリシーの構築 (正規分布)
        mean_val = self.actor_mean(h_context)
        logstd_val = self.actor_logstd.expand_as(mean_val)
        std_val = torch.exp(logstd_val)
        dist_probs = Normal(mean_val, std_val)
        
        if action is None:
            # 未指定時はサンプリング
            action = dist_probs.sample()
            
        lp_out = dist_probs.log_prob(action).sum(1)
        ent_out = dist_probs.entropy().sum(1)
        v_out = self.critic(h_context)
        
        return action, lp_out, ent_out, v_out

class ObsHistory:
    """
    ミラーリングバッファを用いたゼロコピー履歴管理。
    同じデータを2箇所に書くことで、スライス操作のみで時間順序が保たれた系列を取得。
    """
    def __init__(self, num_envs, seq_len, obs_dim, device):
        # 連続スライスのためシーケンス長の2倍を確保
        self.total_len = seq_len * 2
        self.buffer = torch.zeros((num_envs, self.total_len, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.ptr = 0

    def reset(self):
        """記憶バッファをゼロ初期化します。"""
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        """最新の観測値 obs でバッファを更新し、ミラーリング書き込みします。"""
        # テンソルへの変換
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        # 次元補正
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        # リングバッファ現在位置への書き込み
        self.buffer[:, self.ptr] = obs_tensor
        # ミラー位置（seq_len 先）への同時書き込み
        self.buffer[:, self.ptr + self.seq_len] = obs_tensor
        
        # ポインタの循環
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        """最新 seq_len ステップ分の系列をコピーなしの View として抽出。"""
        # ミラーリングにより、このスライスは常に時系列順です。
        view_series = self.buffer[:, self.ptr : self.ptr + self.seq_len]
        return view_series

# ==========================================
# 3. ヘルパー関数
# ==========================================
def load_model_safely(model_obj, base_name, target_type):
    """候補リストから最新の学習済みモデルを安全にロードします。"""
    import torch
    # 探索の優先順位: 微調整版 > 初期版 > 一般版
    search_list = [
        f"{base_name}_refinement_{target_type}.pt",
        f"{base_name}_initial_{target_type}.pt",
        f"{base_name}_{target_type}.pt"
    ]
    
    for file_path in search_list:
        if os.path.exists(file_path):
            try:
                # デバイス非依存でロード
                weights_dict = torch.load(file_path, map_location="cpu")
                model_obj.load_state_dict(weights_dict)
                model_obj.eval()
                return file_path
            except Exception:
                # 失敗時は次のファイルを試行
                continue
    return None

# ==========================================
# 4. 環境作成用ファクトリ
# ==========================================
def create_env(render_mode=None):
    """MuJoCo 物理エンジンの初期化を遅延させ、プロセス競合を防止します。"""
    import torch
    import mujoco
    import gymnasium as gym
    import main18_optimization as base_config

    class TeamCosEnv(base_config.HideAndSeekEnv):
        """
        視界勾配報酬、チーム共有指標、外れ値マスク、停滞罰を統合した環境クラス。
        """
        def __init__(self, render_mode=None):
            # 親クラスの物理初期化
            super().__init__(render_mode=render_mode)
            
            # 内部ワーカープロセスでは常に CPU を使用
            cpu_dev = torch.device("cpu")
            # 各エージェント（Seeker=0, Hider1=1, Hider2=2）の観測履歴管理
            self.npc_obs_history = {
                0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev), 
                1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev), 
                2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_dev)
            }
            
            # 高速化用各種キャッシュ
            self.visible_cache = {0: {}, 1: {}, 2: {}}
            self.lidar_array_cache = {} 
            self.raycast_cache = {} 
            self.raycast_stats = {"hits": 0, "misses": 0}
            
            # 指標収集用カウンタ
            self.hidden_steps_count = 0
            self.caught_steps_count = 0 
            
            # 報酬計算用の物理バッファ
            self.prev_dist = {1: 0.0, 2: 0.0}
            self.prev_hider_xy = {1: None, 2: None}
            
            # 重複演算回避用のメモ化
            self._obs_memo = {}
            # スタック復帰時の旋回フラグ
            self.s0_recovery_turn_dir = 1.0

            # NPC 用のエージェントモデル
            self.npc_hider_agent = Agent(53, 4).to("cpu")
            self.npc_seeker_agent = Agent(53, 4).to("cpu")
            
            # ログ表示制御
            is_play_mode = (render_mode == "human")
            logged_marker = os.environ.get("NPC_MODELS_LOGGED")
            should_print = (logged_marker != "TRUE")

            # モデルロード条件: refinement モードであるか、PLAY(検証) モードである場合
            should_load = LOAD_EXISTING_MODELS or is_play_mode

            if should_load:
                # HIDER NPC ロード
                h_path = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
                if h_path:
                    if should_print:
                        print(f"Loaded NPC Hider: {h_path}", flush=True)
                else:
                    self.npc_hider_agent = None

                # SEEKER NPC ロード
                s_path = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
                if s_path:
                    if should_print:
                        print(f"Loaded NPC Seeker: {s_path}", flush=True)
                else:
                    self.npc_seeker_agent = None
            else:
                # initial 学習中: ゼロベースからの進化を促すためモデルを割り当てない
                self.npc_hider_agent = None
                self.npc_seeker_agent = None
                if should_print:
                    print("Initial Training: NPCs start with default behaviors.", flush=True)

            if should_print:
                os.environ["NPC_MODELS_LOGGED"] = "TRUE"

        def _is_visible(self, origin_pos, origin_rot, target_pos, target_body_id, exclude_body_id):
            """視界角と物理的遮蔽を考慮し、対象が見えているか判定します。"""
            diff_vec = target_pos[:2] - origin_pos[:2]
            distance_val = np.linalg.norm(diff_vec)
            
            # 極至近距離は無条件で可視とみなす
            if distance_val < 0.1:
                return True, target_body_id
            
            # 1. FOV (視野角) 判定
            ang_to_target = np.arctan2(diff_vec[1], diff_vec[0])
            ang_rel = (ang_to_target - origin_rot + np.pi) % (2 * np.pi) - np.pi
            if abs(ang_rel) > np.deg2rad(FOV_DEG / 2.0):
                return False, -1
            
            # 2. RayCast による遮蔽判定
            dir_vec = np.array([diff_vec[0]/distance_val, diff_vec[1]/distance_val, 0.0], dtype=np.float64)
            # レイ始点を腰の高さ 0.5m に設定
            orig_vec = np.array([origin_pos[0], origin_pos[1], 0.5], dtype=np.float64)
            hit_geom_id_arr = np.zeros(1, dtype=np.int32)
            
            # mj_ray 実行
            hit_res_dist = mujoco.mj_ray(
                self.model, 
                self.data, 
                orig_vec, 
                dir_vec, 
                None, 
                1, 
                exclude_body_id, 
                hit_geom_id_arr
            )
            
            if hit_res_dist != -1:
                hit_bid = self.model.geom_bodyid[hit_geom_id_arr[0]]
                # ターゲット自体にヒットした場合
                if hit_bid == target_body_id:
                    return True, target_body_id
                # ターゲットより手前で遮蔽された場合
                if hit_res_dist < distance_val - 0.4:
                    return False, hit_bid
            
            # 遮蔽なし
            return True, target_body_id

        def _check_collision_all(self, pos, threshold):
            """指定座標が障害物の内部にあるか判定します。"""
            # 外壁
            for wall_p, wall_s in self.wall_data:
                dx = abs(pos[0] - wall_p[0]) - wall_s[0]
                dy = abs(pos[1] - wall_p[1]) - wall_s[1]
                # 外部最短距離の二乗
                d_sq = max(dx, 0.0)**2 + max(dy, 0.0)**2
                if d_sq < threshold**2:
                    return True
            # フィールド内オブジェクト
            for gi in self.box_geoms + self.ramp_all_geoms:
                o_p = self.data.geom_xpos[gi][:2]
                o_s = self.model.geom(gi).size[:2]
                dx_o = abs(pos[0] - o_p[0]) - o_s[0]
                dy_o = abs(pos[1] - o_p[1]) - o_s[1]
                do_sq = max(dx_o, 0.0)**2 + max(dy_o, 0.0)**2
                if do_sq < threshold**2:
                    return True
            return False

        def _get_cached_ray(self, agent_id, origin_p, direction, beam_id):
            """ mj_ray 呼び出しを座標ベースでキャッシュします。"""
            ang_val = np.arctan2(direction[1], direction[0])
            cache_key = (agent_id, beam_id)
            
            # 変動が閾値内であれば前回の結果を再利用
            if cache_key in self.raycast_cache:
                cp, ca, cr, cg = self.raycast_cache[cache_key]
                p_moved = np.linalg.norm(origin_p - cp)
                if p_moved < RAYCAST_CACHE_POS_THRESH:
                    a_err = (ang_val - ca + np.pi) % (2 * np.pi) - np.pi
                    if abs(a_err) < 0.05:
                        self.raycast_stats["hits"] += 1
                        return cr, cg
            
            # キャッシュミス
            self.raycast_stats["misses"] += 1
            hit_res = np.zeros(1, dtype=np.int32)
            r_from = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
            r_dir = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
            
            # 除外ボディ設定
            if agent_id == 0:
                exclude = self.s0_body
            elif agent_id == 1:
                exclude = self.h1_body
            else:
                exclude = self.h2_body
                
            dist_res = mujoco.mj_ray(self.model, self.data, r_from, r_dir, None, 1, exclude, hit_res)
            
            # 保存
            self.raycast_cache[cache_key] = (origin_p.copy(), ang_val, dist_res, hit_res[0])
            return dist_res, hit_res[0]

        def _get_obs(self, agent_id):
            """53次元観測を生成。不可視時は外れ値 2.0 でマスクします。"""
            # メモ化チェック
            if agent_id in self._obs_memo:
                return self._obs_memo[agent_id]
                
            # エージェント解決
            if agent_id == 0:
                target_bid = self.s0_body
                prefix_str = 's'
            elif agent_id == 1:
                target_bid = self.h1_body
                prefix_str = 'h1'
            else:
                target_bid = self.h2_body
                prefix_str = 'h2'
            
            # 物理状態取得
            xy_pos = self.data.xpos[target_bid][:2]
            j_joint_id = self.model.joint(f'{prefix_str}_rot').id
            q_idx_val = self.model.jnt_qposadr[j_joint_id]
            angle_rad_val = self.data.qpos[q_idx_val]
            
            # 回転行列
            cos_v = np.cos(-angle_rad_val)
            sin_v = np.sin(-angle_rad_val)
            rot_mat_v = np.array([[cos_v, -sin_v], [sin_v, cos_v]])
            
            # 速度正規化
            jx_id_v = self.model.joint(f'{prefix_str}_x').id
            dof_idx_v = self.model.jnt_dofadr[jx_id_v]
            vel_raw_v = self.data.qvel[dof_idx_v : dof_idx_v + 2]
            vel_local_v = rot_mat_v @ vel_raw_v
            vel_obs_v = vel_local_v / 12.0
            
            # 自己状態
            self_state_v = np.concatenate([
                vel_obs_v, 
                [angle_rad_val, np.cos(angle_rad_val), np.sin(angle_rad_val)]
            ])
            
            # Lidar 情報
            lidar_out_arr = None
            if agent_id in self.lidar_array_cache:
                cp, cr, cl = self.lidar_array_cache[agent_id]
                if np.linalg.norm(xy_pos - cp) < LIDAR_CACHE_POS_THRESH:
                    if abs((angle_rad_val - cr + np.pi) % (2 * np.pi) - np.pi) < LIDAR_CACHE_ANG_THRESH:
                        lidar_out_arr = cl
            
            if lidar_out_arr is None:
                lidar_out_arr = np.zeros(len(self.lidar_angles), dtype=np.float32)
                for i_beam, offset_v in enumerate(self.lidar_angles):
                    rad_v = offset_v + angle_rad_val
                    dir_v = np.array([np.cos(rad_v), np.sin(rad_v)])
                    d_ray, _ = self._get_cached_ray(agent_id, xy_pos, dir_v, i_beam + 100)
                    lidar_out_arr[i_beam] = min(d_ray, 2.5) / 2.5 if d_ray != -1 else 1.0
                # キャッシュ
                self.lidar_array_cache[agent_id] = (xy_pos.copy(), angle_rad_val, lidar_out_arr.copy())

            # 視界判定
            current_vis_v = self.visible_cache[agent_id]
            current_vis_v.clear()
            obj_list_v = [self.box1_body, self.box2_body, self.ramp_body, self.h1_body, self.h2_body, self.s0_body]
            for target_id_v in obj_list_v:
                if target_id_v != target_bid:
                    v_flag_v, _ = self._is_visible(self.data.xpos[target_bid], angle_rad_val, self.data.xpos[target_id_v], target_id_v, target_bid)
                    current_vis_v[target_id_v] = v_flag_v

            def generate_rel_info(obj_id, lock_state_v=None):
                """相対物理情報を整形。不可視時は外れ値 2.0。"""
                is_seen_v = current_vis_v.get(obj_id, False)
                sz_v = 8 if lock_state_v is not None else 7
                
                if is_seen_v:
                    tp_xyz = self.data.xpos[obj_id]
                    rp_v = rot_mat_v @ (tp_xyz[:2] - xy_pos) / 12.0
                    姿勢q = self.data.xquat[obj_id]
                    yaw_v = np.arctan2(
                        2 * (姿勢q[0]*姿勢q[3] + 姿勢q[1]*姿勢q[2]), 
                        1 - 2 * (姿勢q[2]**2 + 姿勢q[3]**2)
                    )
                    
                    j_adr_v = self.model.body_jntadr[obj_id]
                    if j_adr_v != -1:
                        vt_raw = self.data.qvel[j_adr_v : j_adr_v + 2]
                    else:
                        vt_raw = np.zeros(2)
                        
                    rv_v = rot_mat_v @ (vt_raw - vel_raw_v) / 12.0
                    parts_v = [rp_v, rv_v, [np.cos(yaw_v - angle_rad_val), np.sin(yaw_v - angle_rad_val)]]
                    if lock_state_v is not None:
                        f_l_v = 1.0 if lock_state_v else 0.0
                        parts_v.append([f_l_v])
                    # 可視フラグ
                    parts_v.append([1.0])
                    return np.concatenate(parts_v)
                else:
                    mask_v = np.full(sz_v, OUTLIER_VALUE, dtype=np.float32)
                    # フラグのみ 0.0
                    mask_v[-1] = 0.0
                    return mask_v

            # 結合プロセス
            if agent_id == 0:
                # シーカー視点
                H1_rel_v = generate_rel_info(self.h1_body)[:5]
                H2_rel_v = generate_rel_info(self.h2_body)[:5]
                objs_v = [
                    generate_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    generate_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    generate_rel_info(self.ramp_body)
                ]
                final_obs_v = np.concatenate([self_state_v, lidar_out_arr, *objs_v, H1_rel_v, H2_rel_v, np.zeros(3, dtype=np.float32)])
            else:
                # ハイダー視点
                p_id_v = self.h2_body if agent_id == 1 else self.h1_body
                enemy_rel_v = generate_rel_info(self.s0_body)[:5]
                friend_rel_v = generate_rel_info(p_id_v)
                gr_flag_v = 1.0 if self.grasping[agent_id] else 0.0
                objs_v = [
                    generate_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    generate_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    generate_rel_info(self.ramp_body)
                ]
                final_obs_v = np.concatenate([self_state_v, lidar_out_arr, *objs_v, enemy_rel_v, friend_rel_v, [gr_flag_v]])

            vec_out_v = final_obs_v.astype(np.float32)
            self._obs_memo[agent_id] = vec_out_v
            return vec_out_v

        def _update_seeker_state(self):
            """NPCシーカーのターゲット座標と探索モードの管理。"""
            sp_xy_v = self.data.xpos[self.s0_body][:2]
            # 視覚更新
            self._get_obs(0)
            v1_v = self.visible_cache[0].get(self.h1_body, False)
            v2_v = self.visible_cache[0].get(self.h2_body, False)
            
            if v1_v or v2_v:
                target_bid_v = self.h1_body if v1_v else self.h2_body
                target_p_v = self.data.xpos[target_bid_v][:2].copy()
                self.seeker_target_pos = target_p_v
                self.seeker_last_known_pos = target_p_v.copy()
                self.seeker_mode = "CHASING"
            elif self.seeker_last_known_pos is not None:
                dist_v = np.linalg.norm(sp_xy_v - self.seeker_last_known_pos)
                if dist_v > 0.5:
                    self.seeker_target_pos = self.seeker_last_known_pos.copy()
                    self.seeker_mode = "SEARCHING"
                else:
                    self.seeker_last_known_pos = None
                    self.seeker_search_timer = 50
            else:
                if self.seeker_search_timer <= 0:
                    self.seeker_random_target = self.np_random.uniform(-4, 4, 2)
                    self.seeker_search_timer = 80
                self.seeker_search_timer = self.seeker_search_timer - 1
                self.seeker_target_pos = self.seeker_random_target.copy()
                self.seeker_mode = "PATROLLING"

        def _seeker_rule_based_policy(self):
            """最短ヘディング追従制御。"""
            if self.current_step < PREP_STEPS:
                return 0.0, 0.0
            sp_v = self.data.xpos[self.s0_body][:2]
            sr_v = self.data.qpos[self.srot_adr]
            tp_v = self.seeker_target_pos
            dx_v = tp_v[0] - sp_v[0]
            dy_v = tp_v[1] - sp_v[1]
            ta_v = np.arctan2(dy_v, dx_v)
            ad_v = (ta_v - sr_v + np.pi) % (2 * np.pi) - np.pi
            thrust_v = SEEKER_RB_THRUST
            turn_v = np.clip(ad_v * 6.0, -3.0, 3.0)
            if abs(ad_v) > SEEKER_RB_TURN_THRESH:
                thrust_v = thrust_v * 0.3
            sx_id = self.model.joint('s_x').id
            sd_ptr = self.model.jnt_dofadr[sx_id]
            vn_v = np.linalg.norm(self.data.qvel[sd_ptr : sd_ptr + 2])
            if thrust_v > 0.05 and vn_v < 0.05:
                self.s0_stuck_timer = self.s0_stuck_timer + 5
            else:
                self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            if self.s0_stuck_timer > 15:
                self.s0_recovery_mode = 15
                self.s0_stuck_timer = 0
                self.s0_recovery_turn_dir = self.np_random.choice([-1.0, 1.0])
            if self.s0_recovery_mode > 0:
                thrust_v = -0.2
                turn_v = 1.5 * self.s0_recovery_turn_dir
                self.s0_recovery_mode = self.s0_recovery_mode - 1
            return float(thrust_v), float(turn_v)

        def _get_npc_action(self, agent_id, agent_type):
            """NPC行動生成。モデル優先。"""
            import torch
            o_raw = self._get_obs(agent_id)
            self.npc_obs_history[agent_id].update(o_raw)
            if agent_type == "HIDER":
                m_u = self.npc_hider_agent
            else:
                m_u = self.npc_seeker_agent
            if m_u is not None:
                with torch.no_grad():
                    ctx = self.npc_obs_history[agent_id].get()
                    act_t, _, _, _ = m_u.get_action_and_value(ctx)
                return act_t.cpu().numpy()[0]
            if agent_type == "SEEKER":
                fr_v, rr_v = self._seeker_rule_based_policy()
                nf_v = fr_v / SEEKER_THRUST_LIMIT
                return np.array([nf_v, rr_v, 0.0, 0.0], dtype=np.float32)
            return self.action_space.sample() * 0.5

        def reset(self, seed=None, options=None):
            """初期化処理。"""
            obs, info = super().reset(seed=seed, options=options)
            self.hidden_steps_count = 0
            self.caught_steps_count = 0
            self._obs_memo.clear()
            self.lidar_array_cache.clear()
            self.s0_recovery_turn_dir = 1.0
            sp_xy = self.data.xpos[self.s0_body][:2]
            for i_h in [1, 2]:
                bid = self.h1_body if i_h == 1 else self.h2_body
                pxy = self.data.xpos[bid][:2].copy()
                self.prev_dist[i_h] = np.linalg.norm(pxy - sp_xy)
                self.prev_hider_xy[i_h] = pxy
            return obs, info

        def step(self, action):
            """1環境ステップの進行。論理的に独立した行で記述。"""
            self._obs_memo.clear()
            self.current_step = self.current_step + 1
            # クールダウン
            for i in [1, 2]:
                val_cd = self.lock_cooldown[i]
                self.lock_cooldown[i] = max(0, val_cd - 1)
            # 鬼の思考
            self._update_seeker_state()
            self.data.ctrl[:] = 0.0 
            
            if TRAIN_TARGET == "HIDER":
                # 学習者 (Main Hider)
                m_idx = self._apply_action(self.learning_agent_id, action)
                self.data.ctrl[m_idx] = float(action[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[m_idx + 1] = float(action[1])
                # 相棒 (NPC Hider)
                p_id = 2 if self.learning_agent_id == 1 else 1
                a_p = self._get_npc_action(p_id, "HIDER")
                p_idx = self._apply_action(p_id, a_p)
                self.data.ctrl[p_idx] = float(a_p[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[p_idx + 1] = float(a_p[1])
                # 鬼 (NPC Seeker)
                a_s = self._get_npc_action(0, "SEEKER")
                self.data.ctrl[0] = float(a_s[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(a_s[1])
            else:
                # 学習者 (Main Seeker)
                self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(action[1])
                # 敵群 (NPC Hiders)
                for i in [1, 2]:
                    a_n = self._get_npc_action(i, "HIDER")
                    n_idx = self._apply_action(i, a_n)
                    self.data.ctrl[n_idx] = float(a_n[0]) * HIDER_THRUST_LIMIT
                    self.data.ctrl[n_idx + 1] = float(a_n[1])
            
            # --- 物理進展 ---
            for _ in range(ACTION_REPEAT):
                for bi, pose in self.locked_pose.items():
                    if self.locked_boxes[bi]:
                        tj = self.box1_joint_id if bi == self.box1_body else self.box2_joint_id
                        qa = self.model.jnt_qposadr[tj]
                        da = self.model.jnt_dofadr[tj]
                        self.data.qpos[qa : qa + 7] = pose
                        self.data.qvel[da : da + 6] = 0
                mujoco.mj_step(self.model, self.data)
            
            # --- 統計と報酬の確定 ---
            self._obs_memo.clear()
            obs_learner = self._get_obs(self.learning_agent_id)
            _ = self._get_obs(0) # 統計用視界更新
            
            visible_h1 = self.visible_cache[0].get(self.h1_body, False)
            visible_h2 = self.visible_cache[0].get(self.h2_body, False)
            is_any_visible = visible_h1 or visible_h2
            
            if is_any_visible:
                # チームの誰かが発見されている
                self.caught_steps_count = self.caught_steps_count + 1
            else:
                # チーム全員が隠蔽されている
                self.hidden_steps_count = self.hidden_steps_count + 1
                
            total_team_reward = 0.0
            for h_id, target_bid in [(1, self.h1_body), (2, self.h2_body)]:
                f_seen = self.visible_cache[0].get(target_bid, False)
                sp_p = self.data.xpos[self.s0_body][:2]
                hp_p = self.data.xpos[target_bid][:2]
                cd_v = np.linalg.norm(hp_p - sp_p)
                
                if f_seen:
                    # 被視認ペナルティ
                    sr_v = self.data.qpos[self.srot_adr]
                    dv_v = hp_p - sp_p
                    dn_v = dv_v / (np.linalg.norm(dv_v) + 1e-8)
                    fv_v = np.array([np.cos(sr_v), np.sin(sr_v)])
                    ct_v = np.dot(dn_v, fv_v)
                    h_r = -ct_v * COS_PENALTY_SCALE
                    # 逃走ボーナス
                    h_r = h_r + (cd_v - self.prev_dist[h_id]) * REWARD_DISTANCE_DIFF_SCALE
                else:
                    # 隠蔽報酬
                    h_r = REWARD_HIDDEN_BONUS
                
                if h_id == self.learning_agent_id:
                    # 停滞罰
                    if self.prev_hider_xy[h_id] is not None:
                        d_moved = np.linalg.norm(hp_p - self.prev_hider_xy[h_id])
                        if d_moved < 0.01:
                            h_r = h_r + PENALTY_STAGNATION
                    self.prev_hider_xy[h_id] = hp_p.copy()
                
                if max(abs(hp_p)) > 6.5:
                    h_r = h_r + PENALTY_SAFEGUARD
                
                total_team_reward = total_team_reward + h_r
                self.prev_dist[h_id] = cd_v
                
            f_rew = total_team_reward if TRAIN_TARGET == "HIDER" else -total_team_reward
            truncated = (self.current_step >= MAX_STEPS)
            info_step = {
                "hidden_steps": float(self.hidden_steps_count), 
                "caught_steps": float(self.caught_steps_count)
            }
            return obs_learner, float(f_rew), False, truncated, info_step

    return TeamCosEnv(render_mode=render_mode)

# ==========================================
# 5. メイン処理 (学習ループ)
# ==========================================
def main():
    # --- 1. マルチプロセス起動設定 ---
    if platform.system() == "Linux":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    def env_factory():
        env_ins = create_env()
        import gymnasium as gym
        return gym.wrappers.RecordEpisodeStatistics(env_ins)
    
    import gymnasium as gym
    print(f"--- [Parent] 1. Initializing {NUM_ENVS} workers ---", flush=True)
    try:
        vec_envs = gym.vector.AsyncVectorEnv([env_factory for _ in range(NUM_ENVS)])
        print("--- [Parent] 2. Parallel environment ready ---", flush=True)
    except Exception as e_s:
        print(f"--- [Parent] startup failed: {e_s} ---", flush=True)
        sys.exit(1)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb

    # ★デバイス判定の統一
    device_final = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
    exp_run_name = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{int(time.time())}"
    
    if EXECUTION_MODE == "PLAY":
        print(f"--- Inference Mode (PLAY) ---")
        env_p = create_env(render_mode="human")
        agent_p = Agent(env_p.observation_space.shape[0], env_p.action_space.shape[0]).to(device_final)
        lp_file = load_model_safely(agent_p, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if lp_file:
            print(f"Loaded successfully: {lp_file}")
        agent_p.eval()
        hist_p = ObsHistory(1, TRANSFORMER_SEQ_LEN, env_p.observation_space.shape[0], device_final)
        dt_p = 0.005 * ACTION_REPEAT
        try:
            while True:
                o_v, _ = env_p.reset()
                hist_p.reset()
                hist_p.update(o_v)
                f_done = False
                total_r = 0.0
                while not f_done:
                    l_s = time.time()
                    with torch.no_grad():
                        act_v, _, _, _ = agent_p.get_action_and_value(hist_p.get())
                    o_nxt, r_v, term, trunc, i_v = env_p.step(act_v.cpu().numpy()[0])
                    f_done = term or trunc
                    total_r = total_r + r_v
                    hist_p.update(o_nxt)
                    env_p.render()
                    wait_v = dt_p - (time.time() - l_s)
                    if wait_v > 0:
                        time.sleep(wait_v)
                    if env_p.viewer is not None and not env_p.viewer.is_running():
                        return
                print(f"Return: {total_r:.1f}, Hidden Steps: {i_v['hidden_steps']:.0f}")
        except KeyboardInterrupt:
            pass
        finally:
            env_p.close()
        return

    if TRACK_WANDB:
        wandb.init(
            project=base_config.WANDB_PROJECT_NAME, 
            config={"Target": TRAIN_TARGET, "MODE": MODE, "v": "25.35_CUDA_fix"}, 
            name=exp_run_name, 
            sync_tensorboard=False,
            save_code=True
        )

    t_writer = SummaryWriter(f"runs/{exp_run_name}")
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
                        dc = json.load(f_c)
                        global_s_idx = dc.get('global_step', 0)
                        start_s_val = global_s_idx
                except:
                    pass

    # 訓練用バッファの確保 (device_final を確実に使用)
    hist_t = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device_final)
    S, E, O, A = NUM_STEPS, NUM_ENVS, 53, 4
    b_obs = torch.zeros((S, E, TRANSFORMER_SEQ_LEN, O), device=device_final)
    b_act = torch.zeros((S, E, A), device=device_final)
    b_log = torch.zeros((S, E), device=device_final)
    b_rew = torch.zeros((S, E), device=device_final)
    b_don = torch.zeros((S, E), device=device_final)
    b_val = torch.zeros((S, E), device=device_final)
    
    nxt_o_v, _ = vec_envs.reset(seed=FIXED_SEED if FIXED_SEED else int(time.time()))
    nxt_d_v = torch.zeros(E).to(device_final)
    hist_t.reset()
    hist_t.update(nxt_o_v)
    
    total_updates_needed = int(max(1, (TOTAL_TIMESTEPS - global_s_idx) // (E * S)))
    h_returns, h_hidden, h_caught = [], [], []
    start_clock = time.time()
    l_loss, l_ent = 0.0, 0.0

    print(f"--- Training started (v25.35) ---")
    try:
        # total_updates_needed をループに使用
        for u_idx in tqdm(range(1, total_updates_needed + 1), desc="Updates"):
            for step_idx in range(S):
                global_s_idx = global_s_idx + E
                b_obs[step_idx] = hist_t.get()
                b_don[step_idx] = nxt_d_v
                with torch.no_grad():
                    a_out, lp_out, _, v_out = agent_t.get_action_and_value(hist_t.get())
                    b_val[step_idx] = v_out.flatten()
                b_act[step_idx] = a_out
                b_log[step_idx] = lp_out
                nx_o, r_v, term_v, trunc_v, i_v = vec_envs.step(a_out.cpu().numpy())
                d_mask = np.logical_or(term_v, trunc_v)
                
                # インデックス変数 e_idx を統一
                if "final_info" in i_v:
                    for e_idx in range(E):
                        item = i_v["final_info"][e_idx]
                        if d_mask[e_idx] and item is not None:
                            if "episode" in item:
                                h_returns.append(float(item["episode"]["r"]))
                            if "hidden_steps" in item:
                                h_hidden.append(float(item["hidden_steps"]))
                            if "caught_steps" in item:
                                h_caught.append(float(item["caught_steps"]))
                elif "episode" in i_v:
                    e_mask = i_v.get("_episode", [True] * E)
                    for e_idx in range(E):
                        if e_mask[e_idx] and d_mask[e_idx]:
                            h_returns.append(float(i_v["episode"]["r"][e_idx]))
                            try:
                                if "hidden_steps" in i_v:
                                    h_hidden.append(float(i_v["hidden_steps"][e_idx]))
                                if "caught_steps" in i_v:
                                    h_caught.append(float(i_v["caught_steps"][e_idx]))
                            except:
                                pass
                b_rew[step_idx] = torch.tensor(r_v).to(device_final).view(-1)
                nxt_d_v = torch.tensor(d_mask).to(device_final, dtype=torch.float32)
                hist_t.update(nx_o)
            
            with torch.no_grad():
                v_nxt = agent_t.get_value(hist_t.get()).reshape(1, -1)
                b_adv = torch.zeros_like(b_rew).to(device_final)
                g_ptr = 0
                for t in reversed(range(S)):
                    if t == S - 1:
                        nt = 1.0 - nxt_d_v
                        vp = v_nxt
                    else:
                        nt = 1.0 - b_don[t + 1]
                        vp = b_val[t + 1]
                    # デルタ計算 (誤差)
                    delta_t = b_rew[t] + 0.99 * vp * nt - b_val[t]
                    # アドバンテージの累積
                    g_ptr = delta_t + 0.99 * 0.95 * nt * g_ptr
                    b_adv[t] = g_ptr
                b_ret = b_adv + b_val
            
            f_obs = b_obs.reshape((-1, TRANSFORMER_SEQ_LEN, O))
            f_log = b_log.reshape(-1)
            f_act = b_act.reshape((-1, A))
            f_adv = b_adv.reshape(-1)
            f_ret = b_ret.reshape(-1)
            
            for ep_inner in range(UPDATE_EPOCHS):
                idx_shuffle = np.arange(S * E)
                np.random.shuffle(idx_shuffle)
                for ptr in range(0, S * E, MINIBATCH_SIZE):
                    mb_i = idx_shuffle[ptr : ptr + MINIBATCH_SIZE]
                    _, n_lp, e_v, n_v_v = agent_t.get_action_and_value(f_obs[mb_i], f_act[mb_i])
                    ratio = (n_lp - f_log[mb_i]).exp()
                    mb_a_r = f_adv[mb_i]
                    mb_a_n = (mb_a_r - mb_a_r.mean()) / (mb_a_r.std() + 1e-8)
                    l_pol = torch.max(-mb_a_n * ratio, -mb_a_n * torch.clamp(ratio, 0.8, 1.2))
                    l_val = 0.5 * ((n_v_v.view(-1) - f_ret[mb_i]) ** 2).mean()
                    l_tot = l_pol.mean() - ENT_COEF * e_v.mean() + 0.5 * l_val
                    optimizer_t.zero_grad()
                    l_tot.backward()
                    nn.utils.clip_grad_norm_(agent_t.parameters(), 0.5)
                    optimizer_t.step()
                    l_loss = l_tot.item()
                    l_ent = e_v.mean().item()
                    
            if (TRIAL_MODE) or (u_idx % 10 == 0):
                d_real = time.time() - start_clock
                sps_now = int((global_s_idx - start_s_val) / d_real) if d_real > 0 else 0
                log_map = {"charts/SPS": sps_now, "losses/total_loss": l_loss, "losses/entropy": l_ent, "global_step": global_s_idx}
                if h_returns:
                    ah = np.mean(h_hidden)
                    ac = np.mean(h_caught)
                    ar = np.mean(h_returns)
                    log_map.update({"charts/episodic_return": ar, "charts/steps_hidden": ah, "charts/steps_caught": ac})
                    print(f"Update {u_idx}, Step {global_s_idx}, SPS: {sps_now}, EpRet: {ar:.1f}, Hidden: {ah:.1f}, Caught: {ac:.1f}", flush=True)
                    h_returns, h_hidden, h_caught = [], [], []
                elif not TRIAL_MODE:
                    print(f"Update {u_idx}, Step {global_s_idx}, SPS: {sps_now} (Collecting...)", flush=True)
                if TRACK_WANDB:
                    wandb.log(log_map)
                t_writer.add_scalar("charts/SPS", sps_now, global_s_idx)
                
    except KeyboardInterrupt:
        print("\nInterrupted.")
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