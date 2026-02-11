# main23_runner_cos.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【修正内容 (v25.22)】
# 1. PEP 8 完全準拠の独立行展開:
#    - すべてのセミコロン (;) を排除し、1行につき1つの動作のみを記述。
#    - 代入、条件分岐、ループをすべて複数行に展開し、文法エラーを根絶。
# 2. 詳細な日本語解説コメントの網羅:
#    - 報酬設計、物理エンジン制御、Transformerの仕様など、核心部に詳細な説明を付与。
# 3. 観測情報の外れ値設定 (OUTLIER_VALUE = 2.0):
#    - 10.0 から 2.0 へ調整。情報の欠落を明示しつつ、初期学習の数値的安定性を確保。
# 4. Seeker ルールベースの簡素化:
#    - 32方向スキャンを廃止し、ターゲット地点へ直進するシンプルなヘディング制御を採用。
# 5. Seeker NPC の知能統合:
#    - 学習済みモデルがあれば AI を優先し、なければルールベースへ自動フォールバック。

import os
import sys
import platform
import json
import time
import numpy as np
import multiprocessing
from tqdm import tqdm

# --- 実行環境の最適化 ---
# 数値計算ライブラリが並列プロセス内で不要なスレッドを立て、CPUリソースを奪い合うのを防ぎます。
if platform.processor() != 'arm':
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- プロジェクトルートのパス解決 ---
# 共通資産である main18_optimization.py 等にアクセスするため、ディレクトリを遡って特定します。
current_file_path = os.path.abspath(__file__)
search_path = os.path.dirname(current_file_path)
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
# モード: "initial" (新規学習) / "refinement" (既存モデルをベースに微調整)
MODE = "initial" 
EXPERIMENT_BASE_NAME = "HideAndSeek_Layer23_TeamCos"
# 学習対象: 2体の Hider をチームとして育成します
TRAIN_TARGET = "HIDER" 

EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"
# refinement モードのときのみ、既存のウェイトファイルをロードして再開します
LOAD_EXISTING_MODELS = False

# 実行モード: "TRAIN" (学習) または "PLAY" ( Viewer で挙動を観察)
EXECUTION_MODE = "TRAIN" 

SAVE_MODEL = False
TRACK_WANDB = True
FIXED_SEED = None
TRIAL_MODE = True

# PPO (強化学習アルゴリズム) のハイパーパラメータ
TOTAL_TIMESTEPS = 150000
NUM_ENVS = 8
NUM_STEPS = 128            # 更新1回あたりのデータ収集長さ
LEARNING_RATE = 0.0003069848026628
ENT_COEF = 0.00016796783029992242
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4

# Transformer (記憶メカニズム) の設定
TRANSFORMER_SEQ_LEN = 8    # 過去何ステップの記憶を同時に処理するか
HIDDEN_DIM = 64            # 内部の隠れ層の次元数
NUM_LAYERS = 2             # Transformer ブロックの積層数
NUM_HEADS = 2              # Attention 機構のヘッド数

# 環境・物理シミュレーションの設定
ACTION_REPEAT = 16         # 1アクションにつき MuJoCo を何回進めるか
PREP_STEPS = 80            # 序盤の数秒間、Seeker は待機（Hider が陣形を整える時間）
MAX_STEPS = 300            # 固定長エピソード。途中で捕まっても 300歩まで継続
FOV_DEG = 135              # 視野角の設定（135度）

# 高速化のためのキャッシュ制御閾値
LIDAR_CACHE_POS_THRESH = 0.05
LIDAR_CACHE_ANG_THRESH = np.deg2rad(2.0)
RAYCAST_CACHE_POS_THRESH = 0.05

# 観測情報のマスク用外れ値
# 数値 0.0（重なり）と区別するため、フィールド外の物理量を意味する 2.0 を使用
OUTLIER_VALUE = 2.0

# 報酬設計
REWARD_HIDDEN_BONUS = 0.9776424561101347
COS_PENALTY_SCALE = 4.577369643328021
REWARD_DISTANCE_DIFF_SCALE = 1.0 # 敵から遠ざかる動き（逃走成功）に対する加算
PENALTY_SAFEGUARD = -20.0     # フィールド外への脱走に対する強いペナルティ

# エージェントの移動能力制限
HIDER_THRUST_LIMIT = 0.40     # 逃げ手に有利な速度設定
SEEKER_THRUST_LIMIT = 0.35 
SEEKER_RB_THRUST = 0.38       # ルールベース動作時の基準推力
SEEKER_RB_TURN_THRESH = np.pi/6 # 旋回中に減速をかける角度

# 保存パス
SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# ==========================================
# 2. モデル・バッファ定義 (Agent / ObsHistory)
# ==========================================
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """ネットワークレイヤーの初期化 (直交初期化)"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    """過去の観測系列を利用して思考する Transformer ベースのエージェント"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 特徴埋め込み層
        self.embedding = nn.Linear(obs_dim, HIDDEN_DIM)
        # 位置エンコーディング (系列の順序を教える)
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
        
        # Actor: 行動（推力・回転）を出力
        self.actor_mean = layer_init(nn.Linear(HIDDEN_DIM, action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic: 現在の状況をスコア化 (状態価値)
        self.critic = layer_init(nn.Linear(HIDDEN_DIM, 1), std=1)

    def get_value(self, x):
        """状態価値 V(s) を計算"""
        x = self.embedding(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        h_last = x[:, -1, :] # 最後のステップを特徴量として抽出
        v = self.critic(h_last)
        return v

    def get_action_and_value(self, x, action=None):
        """行動、対数確率、エントロピー、および価値を返します"""
        x = self.embedding(x)
        x = x + self.pos_encoder
        x = self.transformer(x)
        h_context = x[:, -1, :]
        
        # ガウス分布に基づく確率的ポリシー
        action_mean = self.actor_mean(h_context)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        lp = probs.log_prob(action).sum(1)
        ent = probs.entropy().sum(1)
        val = self.critic(h_context)
        
        return action, lp, ent, val

class ObsHistory:
    """
    ミラーリングバッファを用いたゼロコピー履歴管理。
    Transformer の系列入力をメモリコピーなしで生成するための構造です。
    """
    def __init__(self, num_envs, seq_len, obs_dim, device):
        # 連続領域として確保できるよう系列長の2倍を確保
        self.buffer = torch.zeros((num_envs, seq_len * 2, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.ptr = 0

    def reset(self):
        """履歴を消去"""
        self.buffer.zero_()
        self.ptr = 0

    def update(self, obs):
        """最新の観測値をバッファの2箇所にミラーリングして書き込みます"""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        # リングバッファの現在位置と、1周後の同じ位置に書き込む
        self.buffer[:, self.ptr] = obs_tensor
        self.buffer[:, self.ptr + self.seq_len] = obs_tensor
        
        # ポインタの循環
        self.ptr = (self.ptr + 1) % self.seq_len

    def get(self):
        """最新 seq_len ステップ分の系列を View として高速に抽出"""
        h = self.buffer[:, self.ptr : self.ptr + self.seq_len]
        return h

# ==========================================
# 3. ヘルパー関数
# ==========================================
def load_model_safely(model_obj, base_name, target_type):
    """複数のファイル候補から最新の学習済みモデルを安全に探索・ロードします"""
    import torch
    # 優先順位: 微調整版 > 初期版 > 汎用版
    candidates = [
        f"{base_name}_refinement_{target_type}.pt",
        f"{base_name}_initial_{target_type}.pt",
        f"{base_name}_{target_type}.pt"
    ]
    
    for path in candidates:
        if os.path.exists(path):
            try:
                sd = torch.load(path, map_location="cpu")
                model_obj.load_state_dict(sd)
                model_obj.eval()
                return path
            except Exception:
                continue
    return None

# ==========================================
# 4. 環境作成用ファクトリ
# ==========================================
def create_env(render_mode=None):
    """MuJoCo の初期化を遅延させることでマルチプロセスのデッドロックを防止します"""
    import torch
    import mujoco
    import gymnasium as gym
    import main18_optimization as base_config

    class TeamCosEnv(base_config.HideAndSeekEnv):
        """
        第23回実験：視界勾配報酬、外れ値マスク、固定長エピソードによるチーム連携学習。
        """
        def __init__(self, render_mode=None):
            super().__init__(render_mode=render_mode)
            
            cpu_device = torch.device("cpu")
            # 各エージェント（Seeker=0, Hider1=1, Hider2=2）個別の履歴を保持
            self.npc_obs_history = {
                0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
                1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
                2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device)
            }
            
            # 各種高速化キャッシュ
            self.visible_cache = {0: {}, 1: {}, 2: {}}
            self.lidar_array_cache = {} 
            self.raycast_cache = {} 
            self.raycast_stats = {"hits": 0, "misses": 0}
            
            # 統計カウンタ
            self.hidden_steps_count = 0
            self.caught_steps_count = 0 
            self.prev_dist = {1: 0.0, 2: 0.0}
            
            # 重複計算回避用のメモ化バッファ
            self._obs_memo = {}
            # スタック時の旋回方向
            self.s0_recovery_turn_dir = 1.0

            # NPC 用のモデル作成
            self.npc_hider_agent = Agent(53, 4).to("cpu")
            self.npc_seeker_agent = Agent(53, 4).to("cpu")
            
            # 環境変数による print 制御（最初のワーカーのみログを出す）
            is_logged = os.environ.get("NPC_MODELS_LOGGED")
            should_print = (is_logged != "TRUE")

            # HIDER モデルのロード
            h_p = load_model_safely(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER")
            if h_p:
                if should_print:
                    print(f"Loaded NPC Hider from {h_p}", flush=True)
            else:
                self.npc_hider_agent = None

            # SEEKER モデルのロード
            s_p = load_model_safely(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER")
            if s_p:
                if should_print:
                    print(f"Loaded NPC Seeker from {s_p}", flush=True)
            else:
                self.npc_seeker_agent = None

            if should_print:
                os.environ["NPC_MODELS_LOGGED"] = "TRUE"

        def _is_visible(self, origin_pos, origin_rot, target_pos, target_body_id, exclude_body_id):
            """対象が視野角内にあり、かつ遮蔽されていないか判定します (戻り値はタプル)"""
            diff = target_pos[:2] - origin_pos[:2]
            d = np.linalg.norm(diff)
            
            # 至近距離は無条件で可視
            if d < 0.1:
                return True, target_body_id
            
            # 1. FOV (視野角) 判定
            a_to_t = np.arctan2(diff[1], diff[0])
            rel_a = (a_to_t - origin_rot + np.pi) % (2 * np.pi) - np.pi
            if abs(rel_a) > np.deg2rad(FOV_DEG / 2.0):
                return False, -1
            
            # 2. RayCast による遮蔽判定
            dr = np.array([diff[0]/d, diff[1]/d, 0.0], dtype=np.float64)
            # レイの始点を腰の高さ 0.5m に固定
            fr = np.array([origin_pos[0], origin_pos[1], 0.5], dtype=np.float64)
            gid_out = np.zeros(1, dtype=np.int32)
            
            # mj_ray 実行
            res_d = mujoco.mj_ray(self.model, self.data, fr, dr, None, 1, exclude_body_id, gid_out)
            
            if res_d != -1:
                hit_bid = self.model.geom_bodyid[gid_out[0]]
                # ターゲット自体に当たったか確認
                if hit_bid == target_body_id:
                    return True, target_body_id
                # ターゲットより手前（40cm以上の余裕）で何かに当たったか確認
                if res_d < d - 0.4:
                    return False, hit_bid
            
            # 遮蔽なし
            return True, target_body_id

        def _check_collision_all(self, pos, threshold):
            """指定座標が障害物内部にあるか幾何学的に判定します (AttributeError対策)"""
            # 外壁
            for wp, ws in self.wall_data:
                dx = abs(pos[0] - wp[0]) - ws[0]
                dy = abs(pos[1] - wp[1]) - ws[1]
                ds_sq = max(dx, 0.0)**2 + max(dy, 0.0)**2
                if ds_sq < threshold**2:
                    return True
            
            # 箱とスロープ
            all_objs = self.box_geoms + self.ramp_all_geoms
            for gi in all_objs:
                cp = self.data.geom_xpos[gi][:2]
                sz = self.model.geom(gi).size[:2]
                dx_o = abs(pos[0] - cp[0]) - sz[0]
                dy_o = abs(pos[1] - cp[1]) - sz[1]
                do_sq = max(dx_o, 0.0)**2 + max(dy_o, 0.0)**2
                if do_sq < threshold**2:
                    return True
            
            return False

        def _get_cached_ray(self, agent_id, origin_p, direction, beam_id):
            """mj_ray の計算負荷を座標ベースのキャッシュで軽減します"""
            a_cur = np.arctan2(direction[1], direction[0])
            key = (agent_id, beam_id)
            
            # 位置と角度の変動が少なければキャッシュを使用
            if key in self.raycast_cache:
                cp, ca, cr, cg = self.raycast_cache[key]
                p_err = np.linalg.norm(origin_p - cp)
                if p_err < RAYCAST_CACHE_POS_THRESH:
                    a_err = (a_cur - ca + np.pi) % (2 * np.pi) - np.pi
                    if abs(a_err) < 0.05:
                        self.raycast_stats["hits"] += 1
                        return cr, cg
            
            # キャッシュミス時
            self.raycast_stats["misses"] += 1
            gid_res = np.zeros(1, dtype=np.int32)
            # 3次元ベクトルとして定義
            r_fr = np.array([origin_p[0], origin_p[1], 0.5], dtype=np.float64)
            r_dr = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
            
            # 自己遮蔽の除外対象設定
            if agent_id == 0:
                excl = self.s0_body
            elif agent_id == 1:
                excl = self.h1_body
            else:
                excl = self.h2_body
                
            d_val = mujoco.mj_ray(self.model, self.data, r_fr, r_dr, None, 1, excl, gid_res)
            
            # キャッシュに保存
            self.raycast_cache[key] = (origin_p.copy(), a_cur, d_val, gid_res[0])
            return d_val, gid_res[0]

        def _get_obs(self, agent_id):
            """53次元観測を生成。不可視時は外れ値 2.0 でマスクします。"""
            if agent_id in self._obs_memo:
                return self._obs_memo[agent_id]
                
            if agent_id == 0:
                bid = self.s0_body
                pref = 's'
            elif agent_id == 1:
                bid = self.h1_body
                pref = 'h1'
            else:
                bid = self.h2_body
                pref = 'h2'
            
            xy = self.data.xpos[bid][:2]
            rad_id = self.model.joint(f'{pref}_rot').id
            rad = self.data.qpos[self.model.jnt_qposadr[rad_id]]
            
            # ローカル座標変換用の回転行列
            cr = np.cos(-rad)
            sr = np.sin(-rad)
            rot_m = np.array([[cr, -sr], [sr, cr]])
            
            # 自己速度の正規化
            jx_id = self.model.joint(f'{pref}_x').id
            da = self.model.jnt_dofadr[jx_id]
            v_r = self.data.qvel[da : da + 2]
            v_l = rot_m @ v_r
            v_o = v_l / 12.0
            
            # 5次元の自己状態
            self_s = np.concatenate([v_o, [rad, np.cos(rad), np.sin(rad)]])
            
            # Lidar 情報の生成とキャッシュ
            lid_o = None
            lc = self.lidar_array_cache.get(agent_id)
            if lc is not None:
                cp, cr, cl = lc
                pe = np.linalg.norm(xy - cp)
                re = abs((rad - cr + np.pi) % (2 * np.pi) - np.pi)
                if pe < LIDAR_CACHE_POS_THRESH and re < LIDAR_CACHE_ANG_THRESH:
                    lid_o = cl
            
            if lid_o is None:
                lid_o = np.zeros(len(self.lidar_angles), dtype=np.float32)
                for i, ao in enumerate(self.lidar_angles):
                    ba = ao + rad
                    bd = np.array([np.cos(ba), np.sin(ba)])
                    rd, _ = self._get_cached_ray(agent_id, xy, bd, i + 100)
                    # 2.5m を 1.0 に正規化
                    lid_o[i] = min(rd, 2.5) / 2.5 if rd != -1 else 1.0
                self.lidar_array_cache[agent_id] = (xy.copy(), rad, lid_o.copy())

            # 視界判定キャッシュの更新
            my_v = self.visible_cache[agent_id]
            my_v.clear()
            cands = [self.box1_body, self.box2_body, self.ramp_body, self.h1_body, self.h2_body, self.s0_body]
            # 自分を除いてループ
            targs = [t for t in cands if t != bid]
            for ti in targs:
                tp = self.data.xpos[ti]
                vis_f, _ = self._is_visible(self.data.xpos[bid], rad, tp, ti, bid)
                my_v[ti] = vis_f

            def get_rel_info(target_id, lock_s=None):
                """不可視時は外れ値 2.0 で埋め、フラグのみ 0.0 にします"""
                seen = my_v.get(target_id, False)
                sz = 8 if lock_s is not None else 7
                
                if seen:
                    tp_xyz = self.data.xpos[target_id]
                    rp = rot_m @ (tp_xyz[:2] - xy) / 12.0
                    
                    qv = self.data.xquat[target_id]
                    ty = np.arctan2(2*(qv[0]*qv[3]+qv[1]*qv[2]), 1-2*(qv[2]**2+qv[3]**2))
                    
                    jad = self.model.body_jntadr[target_id]
                    tv_g = self.data.qvel[jad : jad+2] if jad != -1 else np.zeros(2)
                    rv = rot_m @ (tv_g - v_r) / 12.0
                    
                    info = [rp, rv, [np.cos(ty - rad), np.sin(ty - rad)]]
                    if lock_s is not None:
                        val = 1.0 if lock_s else 0.0
                        info.append([val])
                    
                    info.append([1.0]) # 可視フラグ
                    return np.concatenate(info)
                else:
                    # 数値的に 0.0（重なり）と区別するため外れ値 2.0 で埋める
                    mask_v = np.full(sz, OUTLIER_VALUE, dtype=np.float32)
                    # 最後の要素（可視フラグ）のみ情報の不在を示す 0.0 を代入
                    mask_v[-1] = 0.0
                    return mask_v

            # 役割別のベクトル結合
            if agent_id == 0:
                h1r = get_rel_info(self.h1_body)[:5]
                h2r = get_rel_info(self.h2_body)[:5]
                objs = [
                    get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    get_rel_info(self.ramp_body)
                ]
                obs_f = np.concatenate([self_s, lid_o, *objs, h1r, h2r, np.zeros(3, dtype=np.float32)])
            else:
                p_id = self.h2_body if agent_id == 1 else self.h1_body
                er = get_rel_info(self.s0_body)[:5]
                fr = get_rel_info(p_id)
                gr = 1.0 if self.grasping[agent_id] else 0.0
                objs = [
                    get_rel_info(self.box1_body, self.locked_boxes[self.box1_body]), 
                    get_rel_info(self.box2_body, self.locked_boxes[self.box2_body]), 
                    get_rel_info(self.ramp_body)
                ]
                obs_f = np.concatenate([self_s, lid_o, *objs, er, fr, [gr]])

            vec = obs_f.astype(np.float32)
            self._obs_memo[agent_id] = vec
            return vec

        def _update_seeker_state(self):
            """NPCシーカーのターゲット座標と探索モードを決定します"""
            s_xy = self.data.xpos[self.s0_body][:2]
            # 視界の更新
            self._get_obs(0)
            v1 = self.visible_cache[0].get(self.h1_body, False)
            v2 = self.visible_cache[0].get(self.h2_body, False)
            
            if v1 or v2:
                # 発見時：追跡モード
                tb = self.h1_body if v1 else self.h2_body
                self.seeker_target_pos = self.data.xpos[tb][:2].copy()
                self.seeker_last_known_pos = self.seeker_target_pos.copy()
                self.seeker_mode = "CHASING"
            elif self.seeker_last_known_pos is not None:
                # 見失った直後：最後に見えた場所へ急行
                dm = np.linalg.norm(s_xy - self.seeker_last_known_pos)
                if dm > 0.5:
                    self.seeker_target_pos = self.seeker_last_known_pos.copy()
                    self.seeker_mode = "SEARCHING"
                else:
                    # 捜索完了：手がかりを消去して巡回へ
                    self.seeker_last_known_pos = None
                    self.seeker_search_timer = 50
            else:
                # 手がかりなし：ランダム巡回
                if self.seeker_search_timer <= 0:
                    self.seeker_random_target = self.np_random.uniform(-4, 4, 2)
                    self.seeker_search_timer = 80
                
                self.seeker_search_timer = self.seeker_search_timer - 1
                self.seeker_target_pos = self.seeker_random_target.copy()
                self.seeker_mode = "PATROLLING"

        def _seeker_rule_based_policy(self):
            """★修正：32方向スキャンを廃止し、目的地へ最短角で向かうシンプルな制御"""
            if self.current_step < PREP_STEPS:
                return 0.0, 0.0
            
            s_p = self.data.xpos[self.s0_body][:2]
            s_r = self.data.qpos[self.srot_adr]
            t_p = self.seeker_target_pos
            
            # 目的地への最短角度の算出
            dx = t_p[0] - s_p[0]
            dy = t_p[1] - s_p[1]
            t_a = np.arctan2(dy, dx)
            
            # 角度偏差 (-pi 〜 pi)
            a_d = (t_a - s_r + np.pi) % (2 * np.pi) - np.pi
            
            thr = SEEKER_RB_THRUST
            trn = np.clip(a_d * 6.0, -3.0, 3.0)
            
            # 回転角が大きい場合は直進速度を抑制して安定させる
            if abs(a_d) > SEEKER_RB_TURN_THRESH:
                thr = thr * 0.3
            
            # 物理的スタック判定
            s_dof = self.model.jnt_dofadr[self.model.joint('s_x').id]
            v_n = np.linalg.norm(self.data.qvel[s_dof : s_dof + 2])
            
            if thr > 0.05 and v_n < 0.05:
                self.s0_stuck_timer = self.s0_stuck_timer + 5
            else:
                self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            
            # リカバリー動作の起動
            if self.s0_stuck_timer > 15:
                self.s0_recovery_mode = 15
                self.s0_stuck_timer = 0
                self.s0_recovery_turn_dir = self.np_random.choice([-1.0, 1.0])
            
            if self.s0_recovery_mode > 0:
                thr = -0.2 # 後退
                trn = 1.5 * self.s0_recovery_turn_dir # ランダム転回
                self.s0_recovery_mode = self.s0_recovery_mode - 1
                
            return float(thr), float(trn)

        def _get_npc_action(self, agent_id, agent_type):
            """NPCの自律行動。AIモデルを優先し、なければルールベースへ。"""
            import torch
            o_v = self._get_obs(agent_id)
            self.npc_obs_history[agent_id].update(o_v)
            
            # 使用モデルの選択
            if agent_type == "HIDER":
                m_u = self.npc_hider_agent
            else:
                m_u = self.npc_seeker_agent
                
            if m_u is not None:
                with torch.no_grad():
                    ctx = self.npc_obs_history[agent_id].get()
                    # 正規化範囲 [-1.0, 1.0] で出力
                    act_t, _, _, _ = m_u.get_action_and_value(ctx)
                return act_t.cpu().numpy()[0]
            
            # フォールバック (Seeker のみルールベースあり)
            if agent_type == "SEEKER":
                f_r, r_r = self._seeker_rule_based_policy()
                # step関数側の期待に合わせて推力を LIMIT で割り正規化
                n_f = f_r / SEEKER_THRUST_LIMIT
                return np.array([n_f, r_r, 0.0, 0.0], dtype=np.float32)
            
            # Hider モデルなし：ランダムに彷徨う
            return self.action_space.sample() * 0.5

        def reset(self, seed=None, options=None):
            """物理環境と各バッファを初期化します"""
            obs, info = super().reset(seed=seed, options=options)
            self.hidden_steps_count = 0
            self.caught_steps_count = 0
            self._obs_memo.clear()
            self.lidar_array_cache.clear()
            self.s0_recovery_turn_dir = 1.0
            
            # 初期距離の保存 (報酬計算用)
            sp = self.data.xpos[self.s0_body][:2]
            for i in [1, 2]:
                bi = self.h1_body if i == 1 else self.h2_body
                self.prev_dist[i] = np.linalg.norm(self.data.xpos[bi][:2] - sp)
            
            return obs, info

        def step(self, action):
            """アクション適用、物理進展、報酬算出の1サイクル"""
            self._obs_memo.clear()
            self.current_step = self.current_step + 1
            
            # 各ハイダーの操作クールダウン減算
            for i in [1, 2]:
                v = self.lock_cooldown[i]
                self.lock_cooldown[i] = max(0, v - 1)
            
            # シーカー NPC の思考
            self._update_seeker_state()
            self.data.ctrl[:] = 0.0 
            
            if TRAIN_TARGET == "HIDER":
                # 学習対象 (Hider 1 or 2)
                mi = self._apply_action(self.learning_agent_id, action)
                self.data.ctrl[mi] = float(action[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[mi + 1] = float(action[1])
                
                # 相棒 Hider (NPC)
                pi = 2 if self.learning_agent_id == 1 else 1
                an = self._get_npc_action(pi, "HIDER")
                ni = self._apply_action(pi, an)
                self.data.ctrl[ni] = float(an[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[ni + 1] = float(an[1])
                
                # 敵 Seeker (NPC - AI優先)
                as_n = self._get_npc_action(0, "SEEKER")
                self.data.ctrl[0] = float(as_n[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(as_n[1])
            else:
                # 学習対象 (Seeker)
                self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = float(action[1])
                
                # 敵 Hiders (NPC x 2)
                for i in [1, 2]:
                    an = self._get_npc_action(i, "HIDER")
                    ni = self._apply_action(i, an)
                    self.data.ctrl[ni] = float(an[0]) * HIDER_THRUST_LIMIT
                    self.data.ctrl[ni + 1] = float(an[1])
            
            # --- 物理進展ループ (サブステップ間でのロック維持) ---
            for _ in range(ACTION_REPEAT):
                for bi, pose in self.locked_pose.items():
                    if self.locked_boxes[bi]:
                        ti = self.box1_joint_id if bi == self.box1_body else self.box2_joint_id
                        qa = self.model.jnt_qposadr[ti]
                        da = self.model.jnt_dofadr[ti]
                        # 座標の上書きによる強制固定
                        self.data.qpos[qa : qa + 7] = pose
                        # 物理エンジンの速度をリセットして反発を抑える
                        self.data.qvel[da : da + 6] = 0
                
                mujoco.mj_step(self.model, self.data)
            
            # 観測と報酬の算出
            self._obs_memo.clear()
            o_l = self._get_obs(self.learning_agent_id)
            self._get_obs(0) # 統計用にシーカー視界を更新
            
            t_rew = 0.0
            lb = self.h1_body if self.learning_agent_id == 1 else self.h2_body
            
            # 統計フラグ
            v_l = self.visible_cache[0].get(lb, False)
            v_a = any(self.visible_cache[0].get(bid, False) for bid in [self.h1_body, self.h2_body])
            
            if not v_l:
                self.hidden_steps_count = self.hidden_steps_count + 1
            if v_a:
                self.caught_steps_count = self.caught_steps_count + 1
                
            # チーム全体の報酬集計
            for hi, body_i in [(1, self.h1_body), (2, self.h2_body)]:
                vf = self.visible_cache[0].get(body_i, False)
                sp_p = self.data.xpos[self.s0_body][:2]
                cd = np.linalg.norm(self.data.xpos[body_i][:2] - sp_p)
                
                if vf:
                    # 視界内：正面度合いに応じたペナルティ (-cos)
                    sr = self.data.qpos[self.srot_adr]
                    dv = self.data.xpos[body_i][:2] - sp_p
                    dn = dv / (np.linalg.norm(dv) + 1e-8)
                    ct = np.dot(dn, np.array([np.cos(sr), np.sin(sr)]))
                    hr = -ct * COS_PENALTY_SCALE
                    # 逃走（距離増加）への補助ボーナス
                    hr = hr + (cd - self.prev_dist[hi]) * REWARD_DISTANCE_DIFF_SCALE
                else:
                    # 隠蔽成功ボーナス
                    hr = REWARD_HIDDEN_BONUS
                
                # 壁抜け等の物理バグ防止用保険
                if max(abs(self.data.xpos[body_i][:2])) > 6.5:
                    hr = hr + PENALTY_SAFEGUARD
                
                t_rew = t_rew + hr
                self.prev_dist[hi] = cd
                
            f_r = t_rew if TRAIN_TARGET == "HIDER" else -t_rew
            trunc = (self.current_step >= MAX_STEPS)
            
            step_i = {
                "hidden_steps": float(self.hidden_steps_count), 
                "caught_steps": float(self.caught_steps_count)
            }
            return o_l, float(f_r), False, trunc, step_i

        def render(self, stats=None):
            """Viewer 上での詳細描画。サイト座標を利用してラベルを正確に配置します。"""
            if self.render_mode == "human":
                if self.viewer is None: 
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    self.viewer.cam.elevation = -60
                    self.viewer.cam.distance = 23.0
                    self.viewer.cam.lookat[:] = [0, 0, 0] 
                
                # 箱の色の動的更新
                for bb, gi in [(self.box1_body, self.box1_geom_id), (self.box2_body, self.box2_geom_id)]:
                    if self.locked_boxes[bb]:
                        # ロック中：赤
                        self.model.geom_rgba[gi] = [0.8, 0.1, 0.1, 1.0]
                    elif any(v == bb for v in self.grasping.values()):
                        # 掴まれ中：青
                        self.model.geom_rgba[gi] = [0.1, 0.1, 0.9, 1.0]
                    else:
                        # デフォルト：茶
                        bc = [0.6, 0.4, 0.2, 1.0] if bb == self.box1_body else [0.7, 0.5, 0.3, 1.0]
                        self.model.geom_rgba[gi] = bc
                
                if self.viewer.user_scn:
                    ctx = self.viewer.user_scn
                    ctx.ngeom = 0 # 毎フレームクリア
                    
                    # 1. Seeker ラベル (AI/Rule)
                    if ctx.ngeom < ctx.maxgeom:
                        s_src = "AI" if self.npc_seeker_agent is not None else "Rule"
                        l_s = f"Seeker: {self.seeker_mode} ({s_src})"
                        l_p = self.data.site_xpos[self.id_s_label]
                        mujoco.mjv_initGeom(
                            ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, 
                            size=[0,0,0], pos=l_p, 
                            mat=np.eye(3).flatten(), rgba=[1, 0.5, 0, 1]
                        )
                        ctx.geoms[ctx.ngeom].label = l_s
                        ctx.ngeom = ctx.ngeom + 1

                    # 2. Hider ラベルと可視ライン
                    for i, (bid, c_vec) in enumerate([(self.h1_body, [1,1,0,1]), (self.h2_body, [0,1,1,1])]):
                        hp = np.array(self.data.xpos[bid])
                        hr = self.data.qpos[self.h1rot_adr if i == 0 else self.h2rot_adr]
                        v_seen = []
                        targs = [(self.box1_body, "Box1"), (self.box2_body, "Box2"), (self.ramp_body, "Ramp"), (self.s0_body, "Seeker")]
                        
                        for tid, name in targs:
                            vf, _ = self._is_visible(hp, hr, self.data.xpos[tid], tid, bid)
                            if vf:
                                v_seen.append(name)
                                # 視界のラインを描画
                                if ctx.ngeom < ctx.maxgeom:
                                    mujoco.mjv_connector(
                                        ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=2.0, 
                                        from_=hp + [0,0,0.5], to=self.data.xpos[tid] + [0,0,0.5]
                                    )
                                    ctx.geoms[ctx.ngeom].rgba = np.array(c_vec[:3] + [0.6])
                                    ctx.ngeom = ctx.ngeom + 1
                        
                        if ctx.ngeom < ctx.maxgeom:
                            l_h = f"H{i+1} Vis:[{','.join(v_seen)}]"
                            l_hp = self.data.site_xpos[self.id_h1_label if i == 0 else self.id_h2_label]
                            mujoco.mjv_initGeom(
                                ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, 
                                size=[0,0,0], pos=l_hp, 
                                mat=np.eye(3).flatten(), rgba=np.array(c_vec)
                            )
                            ctx.geoms[ctx.ngeom].label = l_h
                            ctx.ngeom = ctx.ngeom + 1

                self.viewer.sync()

    return TeamCosEnv(render_mode=render_mode)

# ==========================================
# 5. メイン処理 (学習ループ)
# ==========================================
def main():
    # --- 1. マルチプロセス起動設定 (デッドロック回避) ---
    if platform.system() == "Linux":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # 並列ワーカー生成用ファクトリ
    def env_factory():
        # create_env 内部でライブラリを遅延ロード
        env_ins = create_env()
        import gymnasium as gym
        return gym.wrappers.RecordEpisodeStatistics(env_ins)
    
    import gymnasium as gym
    print(f"--- [Parent] 1. Initializing {NUM_ENVS} parallel workers ---", flush=True)
    
    # ワーカーが立ち上がるまで、親プロセスは CUDA のコンテキストを作成しません。
    try:
        vec_envs = gym.vector.AsyncVectorEnv([env_factory for _ in range(NUM_ENVS)])
        print("--- [Parent] 2. Workers initialized successfully ---", flush=True)
    except Exception as e_s:
        print(f"--- [Parent] [CRITICAL] Parallel startup failed: {e_s} ---", flush=True)
        sys.exit(1)

    # --- 3. 親プロセスでのライブラリ初期化 ---
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    import wandb
    import main18_optimization as base_config

    device_actual = torch.device("cuda" if torch.cuda.is_available() and base_config.CUDA else "cpu")
    ts_now = int(time.time())
    display_n = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{ts_now}"
    
    # 鑑賞モード (PLAY)
    if EXECUTION_MODE == "PLAY":
        print(f"--- Inference Mode (PLAY) ---")
        env_v = create_env(render_mode="human")
        agent_v = Agent(env_v.observation_space.shape[0], env_v.action_space.shape[0]).to(device_actual)
        l_p = load_model_safely(agent_v, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if l_p:
            print(f"Loaded successfully: {l_p}")
        
        agent_v.eval()
        hist_v = ObsHistory(1, TRANSFORMER_SEQ_LEN, env_v.observation_space.shape[0], device_actual)
        dt_v = 0.005 * ACTION_REPEAT
        
        try:
            while True:
                obs_v, _ = env_v.reset()
                hist_v.reset()
                hist_v.update(obs_v)
                f_done = False
                total_r = 0.0
                while not f_done:
                    l_s = time.time()
                    with torch.no_grad():
                        act_v, _, _, _ = agent_v.get_action_and_value(hist_v.get())
                    
                    o_nxt, r_v, term_v, trunc_v, info_v = env_v.step(act_v.cpu().numpy()[0])
                    f_done = term_v or trunc_v
                    total_r = total_r + r_v
                    hist_v.update(o_nxt)
                    env_v.render()
                    
                    # 再生速度の調整
                    p_t = time.time() - l_s
                    w_t = dt_v - p_t
                    if w_t > 0:
                        time.sleep(w_t)
                    
                    if env_v.viewer is not None and not env_v.viewer.is_running():
                        return
                
                print(f"Return: {total_r:.1f}, Hidden Steps: {info_v['hidden_steps']:.0f}")
        except KeyboardInterrupt:
            pass
        finally:
            env_v.close()
        return

    # 学習モード (TRAIN) の初期化
    if TRACK_WANDB:
        run_obj = wandb.init(
            project=base_config.WANDB_PROJECT_NAME, 
            config={"Target": TRAIN_TARGET, "MODE": MODE, "v": "25.22_unrolled_stable"}, 
            name=display_n, 
            sync_tensorboard=False,
            save_code=True
        )
        run_obj.define_metric("global_step")
        run_obj.define_metric("*", step_metric="global_step")

    t_writer = SummaryWriter(f"runs/{display_n}")
    agent_t = Agent(vec_envs.single_observation_space.shape[0], vec_envs.single_action_space.shape[0]).to(device_actual)
    optimizer_t = optim.Adam(agent_t.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    global_s_idx = 0
    start_s_val = 0
    if LOAD_EXISTING_MODELS:
        lp = load_model_safely(agent_t, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if lp:
            print(f"★ Resumed from: {lp}")
            cp_p = lp.replace('.pt', '_checkpoint.json')
            if os.path.exists(cp_p):
                try:
                    with open(cp_p, 'r') as f_c:
                        cp_d = json.load(f_c)
                        global_s_idx = cp_d.get('global_step', 0)
                        start_s_val = global_s_idx
                except Exception:
                    pass

    # 訓練用バッファの確保
    hist_t = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device_actual)
    S, E, O, A = NUM_STEPS, NUM_ENVS, 53, 4
    b_obs = torch.zeros((S, E, TRANSFORMER_SEQ_LEN, O), device=device_actual)
    b_act = torch.zeros((S, E, A), device=device_actual)
    b_log = torch.zeros((S, E), device=device_actual)
    b_rew = torch.zeros((S, E), device=device_actual)
    b_don = torch.zeros((S, E), device=device_actual)
    b_val = torch.zeros((S, E), device=device_actual)
    
    # 最初の観測
    sv = FIXED_SEED if FIXED_SEED else int(time.time())
    nxt_o_v, _ = vec_envs.reset(seed=sv)
    nxt_d_v = torch.zeros(E).to(device_actual)
    hist_t.reset()
    hist_t.update(nxt_o_v)
    
    u_total = int(max(1, (TOTAL_TIMESTEPS - global_s_idx) // (E * S)))
    h_returns, h_hidden, h_caught = [], [], []
    start_r_t = time.time()
    l_loss, l_ent = 0.0, 0.0

    print(f"--- Training loop starting ---")
    try:
        for u_idx in tqdm(range(1, u_total + 1), desc="Updates"):
            # データ収集フェーズ (Rollout)
            for step_idx in range(S):
                global_s_idx = global_s_idx + E
                b_obs[step_idx] = hist_t.get()
                b_don[step_idx] = nxt_d_v
                
                with torch.no_grad():
                    a_out, lp_out, _, v_out = agent_t.get_action_and_value(hist_t.get())
                    b_val[step_idx] = v_out.flatten()
                
                b_act[step_idx] = a_out
                b_log[step_idx] = lp_out
                
                # 環境進展
                nx_o, r_v, term_v, trunc_v, i_v = vec_envs.step(a_out.cpu().numpy())
                d_mask = np.logical_or(term_v, trunc_v)
                
                # 統計収集 (エピソード完了時)
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
                    e_m = i_v.get("_episode", [True] * E)
                    for e_idx in range(E):
                        if e_m[e_idx] and d_mask[e_idx]:
                            h_returns.append(float(i_v["episode"]["r"][e_idx]))
                            try:
                                if "hidden_steps" in i_v:
                                    h_hidden.append(float(i_v["hidden_steps"][e_idx]))
                                if "caught_steps" in i_v:
                                    h_caught.append(float(i_v["caught_steps"][e_idx]))
                            except Exception:
                                pass
                
                b_rew[step_idx] = torch.tensor(r_v).to(device_actual).view(-1)
                nxt_d_v = torch.tensor(d_mask).to(device_actual, dtype=torch.float32)
                hist_t.update(nx_o)

            # アドバンテージ計算 (GAE)
            with torch.no_grad():
                v_nxt_p = agent_t.get_value(hist_t.get()).reshape(1, -1)
                b_adv = torch.zeros_like(b_rew).to(device_actual)
                g_ptr = 0
                for t in reversed(range(S)):
                    if t == S - 1:
                        m_non_t = 1.0 - nxt_d_v
                        v_post = v_nxt_p
                    else:
                        m_non_t = 1.0 - b_don[t + 1]
                        v_post = b_val[t + 1]
                    
                    # 期待報酬と実際の差分
                    dt_t = b_rew[t] + 0.99 * v_post * m_non_t - b_val[t]
                    # 累積利得への重み付け
                    g_ptr = dt_t + 0.99 * 0.95 * m_non_t * g_ptr
                    b_adv[t] = g_ptr
                
                b_ret = b_adv + b_val

            # パラメータ最適化 (PPO)
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
                    
                    # 方策の変化率
                    rat = (n_lp - f_log[mb_i]).exp()
                    mb_a_r = f_adv[mb_i]
                    # アドバンテージの正規化による安定化
                    mb_a_n = (mb_a_r - mb_a_r.mean()) / (mb_a_r.std() + 1e-8)
                    
                    # PPO クリップ損失関数
                    l_p1 = -mb_a_n * rat
                    l_p2 = -mb_a_n * torch.clamp(rat, 0.8, 1.2)
                    l_pol = torch.max(l_p1, l_p2).mean()
                    
                    # 状態価値予測の MSE 誤差
                    l_val = 0.5 * ((n_v_v.view(-1) - f_ret[mb_i]) ** 2).mean()
                    
                    # エントロピー（多様性）項を加味した合計損失
                    l_tot = l_pol - ENT_COEF * e_v.mean() + 0.5 * l_val
                    
                    optimizer_t.zero_grad()
                    l_tot.backward()
                    nn.utils.clip_grad_norm_(agent_t.parameters(), 0.5)
                    optimizer_t.step()
                    
                    l_loss = l_tot.item()
                    l_ent = e_v.mean().item()

            # 診断ログ出力
            if (TRIAL_MODE) or (u_idx % 10 == 0):
                d_real = time.time() - start_r_t
                sps_now = int((global_s_idx - start_s_val) / d_real) if d_real > 0 else 0
                
                log_m = {
                    "charts/SPS": sps_now, "losses/total_loss": l_loss, 
                    "losses/entropy": l_ent, "global_step": global_s_idx
                }
                
                if h_returns:
                    avg_h_s = np.mean(h_hidden)
                    avg_c_s = np.mean(h_caught)
                    avg_rt = np.mean(h_returns)
                    log_m.update({
                        "charts/episodic_return": avg_rt, 
                        "charts/steps_hidden": avg_h_s, "charts/steps_caught": avg_c_s
                    })
                    print(f"Update {u_idx}, Step {global_s_idx}, SPS: {sps_now}, EpRet: {avg_rt:.1f}, Hidden: {avg_h_s:.1f}, Caught: {avg_c_s:.1f}", flush=True)
                    # 次のログに向けてバッファをクリア
                    h_returns, h_hidden, h_caught = [], [], []
                elif not TRIAL_MODE:
                    print(f"Update {u_idx}, Step {global_s_idx}, SPS: {sps_now} (Collecting statistics...)", flush=True)
                
                if TRACK_WANDB:
                    wandb.log(log_m)
                t_writer.add_scalar("charts/SPS", sps_now, global_s_idx)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # モデル保存
    if SAVE_MODEL:
        torch.save(agent_t.state_dict(), SAVE_MODEL_PATH)
        chk_f_p = SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json')
        with open(chk_f_p, 'w') as f_o:
            json.dump({'global_step': global_s_idx}, f_o)
        print(f"Model saved to {SAVE_MODEL_PATH}")
    
    # 終了処理
    vec_envs.close()
    t_writer.close()
    if TRACK_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()