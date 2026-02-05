# main18_optimization.py
# 演習第17回：並列処理のボトルネック分析と高速化
# 
# 【概要】
# main17_transformer.py (CleanRL) をベースに、Windows/Linux環境等での
# 「並列学習効率」を最大化するようチューニングしたバージョンです。
#
# 【主な変更点 (v18.9)】
# 1. 視界判定(_is_visible)の高速化 (ユーザー提案):
#    - 戻り値を (is_visible, hit_body_id) に拡張。
#    - ターゲットが見えない場合、遮っている物体(hit_body_id)を返すように変更。
# 2. 観測生成(_get_obs)のキャッシング:
#    - _is_visibleの結果を利用し、遮蔽物として検出されたオブジェクトは「見えている」として
#      キャッシュ(visible_cache)に登録。これにより重複するRayCastを削減。
# 3. その他v18.8の最適化を維持:
#    - AsyncVectorEnv(マルチプロセス) + 環境変数スレッド制限。
#    - 観測配列の事前確保。
#    - コードクリーンアップ。
#
# 【実行準備】
# uv add "gymnasium>=1.2.3" "mujoco>=3.4.0" "mujoco-python-viewer>=0.1.4" "numpy>=2.4.1" "tensorboard>=2.20.0" "tensorboardx>=2.6.4" "torch>=2.10.0" "wandb>=0.24.0" matplotlib tqdm

import os

# --- 並列計算の競合を防ぐ環境変数設定 ---
# AsyncVectorEnv(マルチプロセス)を使用するため、各プロセス内の数値計算ライブラリを
# シングルスレッドに制限し、プロセス並列の効果を最大化します。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

# ==========================================
# 1. ハイパーパラメータ & 設定
# ==========================================
EXPERIMENT_NAME = "HideAndSeek_Transformer"
SEED = 1
TORCH_DETERMINISTIC = True
CUDA = True

# ★ モード設定
TRAIN_MODE = True         # True: 学習 / False: 推論
USE_VIEWER = False        # 学習中はFalse推奨 (Trueにすると並列化が無効になります)
TRACK_WANDB = True        # WandBログ

# WandB
WANDB_PROJECT_NAME = "HideAndSeek_Transformer_Project"
WANDB_ENTITY = None 

# PPO / Transformer
TOTAL_TIMESTEPS = 3000000
LEARNING_RATE = 3e-4
NUM_ENVS = 4              # 並列環境数 (CPUコア数に合わせて調整推奨: 4~16)
NUM_STEPS = 256           # 1回の更新で収集するステップ数
MINIBATCH_SIZE = 256
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TRANSFORMER_SEQ_LEN = 16
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4

# 視界判定の再計算間隔（ステップ）
VIS_UPDATE_INTERVAL = 5

# 環境設定
TRAIN_TARGET = "HIDER" 
SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"
MODEL_PATH_HIDER = f"{EXPERIMENT_NAME}_HIDER.pt"
MODEL_PATH_SEEKER = f"{EXPERIMENT_NAME}_SEEKER.pt"

# --- 環境定数 ---
ENT_COEF_ENV = 0.02             
LSTM_HIDDEN_SIZE = 256      
ACTION_REPEAT = 16          
PREP_STEPS = 80             
MAX_STEPS = 300             
FOV_DEG = 135               
REWARD_SURVIVAL = 0.2       
REWARD_DISTANCE_COEFF = 0.08 
PENALTY_CAPTURE = -30.0     
PENALTY_STAGNATION = -0.5   
REWARD_CAPTURE_BONUS = 30.0 
HIDER_THRUST_LIMIT = 0.35   
SEEKER_THRUST_LIMIT = 0.40  
BOOST_MULTIPLIER = 5.0      
SEEKER_RB_THRUST = 0.38 
SEEKER_RB_TURN_THRESH = np.pi/6 

# ==========================================
# 2. 数学ヘルパー関数
# ==========================================
def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def rotate_vec(q, v):
    q_vec = np.array([0, v[0], v[1], v[2]])
    q_inv = quat_inv(q)
    return quat_mul(quat_mul(q, q_vec), q_inv)[1:]

def get_body_xy(data, body_id):
    return data.xpos[body_id][:2]

def get_body_rot(data, body_id):
    q = data.xquat[body_id]
    return np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))

# ==========================================
# 3. モデル・バッファ定義
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding = nn.Linear(self.obs_dim, HIDDEN_DIM)
        self.pos_encoder = nn.Parameter(torch.zeros(1, TRANSFORMER_SEQ_LEN, HIDDEN_DIM))
        encoder_layer = nn.TransformerEncoderLayer(d_model=HIDDEN_DIM, nhead=NUM_HEADS, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.actor_mean = layer_init(nn.Linear(HIDDEN_DIM, self.action_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
        self.critic = layer_init(nn.Linear(HIDDEN_DIM, 1), std=1)

    def get_value(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        return self.critic(x[:, -1, :])

    def get_action_and_value(self, x, action=None):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer(x)
        h = x[:, -1, :] 
        action_mean = self.actor_mean(h)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(h)

class ObsHistory:
    def __init__(self, num_envs, seq_len, obs_dim, device):
        self.buffer = torch.zeros((num_envs, seq_len, obs_dim)).to(device)
        self.device = device
    def reset(self):
        self.buffer.zero_()
    def update(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        self.buffer = torch.roll(self.buffer, -1, dims=1)
        self.buffer[:, -1, :] = obs
    def get(self):
        return self.buffer

# ==========================================
# 4. MJCF (XML) 物理環境定義
# ==========================================
XML_CONTENT = """
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.005"/>
    <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/></visual>
    <asset>
        <texture name="grid_tex" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid_mat" texture="grid_tex" texrepeat="1 1" reflectance="0.2"/>
        <mesh name="ramp_mesh" vertex="-0.6666 -0.5 0.0 0.6666 -0.5 0.0 0.6666 -0.5 1.0 -0.6666 0.5 0.0 0.6666 0.5 0.0 0.6666 0.5 1.0" face="0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0"/>
    </asset>
    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="6 6 0.1" material="grid_mat" friction="1.0 0.05 0.0001" solref="0.04 1"/>
        <geom name="wall_n" type="box" size="6.2 0.1 4.0" pos="0 6.1 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="wall_s" type="box" size="6.2 0.1 4.0" pos="0 -6.1 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="wall_e" type="box" size="0.1 6 4.0" pos="6.1 0 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="wall_w" type="box" size="0.1 6 4.0" pos="-6.1 0 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="maze_w0" type="box" size="1.5 0.2 0.5" pos="3.0 1.5 0.5" rgba="0.0 0.7 0.7 1"/>
        <geom name="maze_w1" type="box" size="1.5 0.2 0.5" pos="-3.0 -1.5 0.5" rgba="0.0 0.7 0.7 1"/>
        <geom name="maze_w2" type="box" size="0.2 1.5 0.5" pos="0 -3.0 0.5" rgba="0.0 0.7 0.7 1"/>
        <geom name="maze_w3" type="box" size="0.2 1.5 0.5" pos="0 3.0 0.5" rgba="0.0 0.7 0.7 1"/>
        <body name="ramp_body" pos="0 0 0">
            <inertial pos="0.3 0 0.25" mass="50" diaginertia="10 10 20"/>
            <joint type="free" name="ramp_joint" damping="500.0"/>
            <geom type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0" rgba="0 1 0 1"/>
            <geom name="ramp_slope_surface" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.3" friction="1.2 0.01 0"/>
            <geom name="ramp_back_panel" type="box" size="0.02 0.5 0.5" pos="0.6666 0 0.5" rgba="0 1 0 0.3"/>
            <geom name="ramp_inner_weight" type="box" size="0.3333 0.5 0.25" pos="0.3333 0 0.25" rgba="0 1 0 0.3" mass="30" solimp="0.95 0.99 0.001"/> 
        </body>
        <body name="box1_body" pos="2 -2 0.5"><joint name="box1_joint" type="free" damping="100.0"/><geom name="box1_geom" type="box" size="0.6 0.6 0.5" rgba="0.6 0.4 0.2 1" mass="100" solref="0.02 1" condim="3" friction="1.0 0.005 0.0001"/></body>
        <body name="box2_body" pos="-2 2 0.5"><joint name="box2_joint" type="free" damping="100.0"/><geom name="box2_geom" type="box" size="0.6 0.6 0.5" rgba="0.7 0.5 0.3 1" mass="100" solref="0.02 1" condim="3" friction="1.0 0.005 0.0001"/></body>
        <body name="seeker_anchor" pos="0 0 0.5">
            <joint name="s_x" type="slide" axis="1 0 0" damping="40"/><joint name="s_y" type="slide" axis="0 1 0" damping="40"/><joint name="s_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2" damping="20"/><joint name="s_rot" type="hinge" axis="0 0 1" damping="50" armature="3.0"/>
            <body name="seeker_body">
                <site name="seeker_thrust_site" pos="0 0 0"/>
                <site name="site_s_label" pos="0 0 1.8" type="sphere" size="0.01" rgba="0 0 0 0"/>
                <geom name="seeker_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="1.2 0.01 0"/><geom name="seeker_capsule" type="capsule" size="0.3 0.2" rgba="0.9 0.1 0.1 1" mass="5"/><geom name="seeker_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/><geom name="seeker_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.9 0.1 0.1 1" contype="0" conaffinity="0"/>
            </body>
        </body>
        <body name="hider1_anchor" pos="0 0 0.5">
            <joint name="h1_x" type="slide" axis="1 0 0" damping="40"/><joint name="h1_y" type="slide" axis="0 1 0" damping="40"/><joint name="h1_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2" damping="20"/><joint name="h1_rot" type="hinge" axis="0 0 1" damping="50" armature="3.0"/>
            <body name="hider1_body">
                <site name="hider1_thrust_site" pos="0 0 0"/>
                <site name="site_h1_label" pos="0 0 1.2" type="sphere" size="0.01" rgba="0 0 0 0"/>
                <geom name="hider1_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="1.2 0.01 0"/><geom name="hider1_capsule" type="capsule" size="0.3 0.2" rgba="0.1 0.1 0.9 1" mass="5"/><geom name="hider1_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/><geom name="hider1_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.1 0.1 0.9 1" contype="0" conaffinity="0"/>
            </body>
        </body>
        <body name="hider2_anchor" pos="0 0 0.5">
            <joint name="h2_x" type="slide" axis="1 0 0" damping="40"/><joint name="h2_y" type="slide" axis="0 1 0" damping="40"/><joint name="h2_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2" damping="20"/><joint name="h2_rot" type="hinge" axis="0 0 1" damping="50" armature="3.0"/>
            <body name="hider2_body">
                <site name="hider2_thrust_site" pos="0 0 0"/>
                <site name="site_h2_label" pos="0 0 1.5" type="sphere" size="0.01" rgba="0 0 0 0"/>
                <geom name="hider2_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="1.2 0.01 0"/><geom name="hider2_capsule" type="capsule" size="0.3 0.2" rgba="0.1 0.6 0.9 1" mass="5"/><geom name="hider2_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/><geom name="hider2_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.1 0.6 0.9 1" contype="0" conaffinity="0"/>
            </body>
        </body>
    </worldbody>
    <equality>
        <weld name="eq_grasp1_b1" body1="hider1_body" body2="box1_body" active="false" relpose="0 0 0 1 0 0 0" solref="0.08 1" solimp="0.9 0.95 0.001"/>
        <weld name="eq_grasp1_b2" body1="hider1_body" body2="box2_body" active="false" relpose="0 0 0 1 0 0 0" solref="0.08 1" solimp="0.9 0.95 0.001"/>
        <weld name="eq_grasp2_b1" body1="hider2_body" body2="box1_body" active="false" relpose="0 0 0 1 0 0 0" solref="0.08 1" solimp="0.9 0.95 0.001"/>
        <weld name="eq_grasp2_b2" body1="hider2_body" body2="box2_body" active="false" relpose="0 0 0 1 0 0 0" solref="0.08 1" solimp="0.9 0.95 0.001"/>
        <weld name="eq_lock_b1" body1="world" body2="box1_body" active="false" solref="0.02 1" solimp="0.95 0.99 0.001"/>
        <weld name="eq_lock_b2" body1="world" body2="box2_body" active="false" solref="0.02 1" solimp="0.95 0.99 0.001"/>
    </equality>
    <actuator>
        <general name="s_fwd" site="seeker_thrust_site" gear="1 0 0 0 0 0" gainprm="9000" ctrlrange="-1 1"/>
        <general name="s_turn" joint="s_rot" gear="0.6" gainprm="500" ctrlrange="-1 1"/>
        <general name="h1_fwd" site="hider1_thrust_site" gear="1 0 0 0 0 0" gainprm="9000" ctrlrange="-1 1"/>
        <general name="h1_turn" joint="h1_rot" gear="0.6" gainprm="500" ctrlrange="-1 1"/>
        <general name="h2_fwd" site="hider2_thrust_site" gear="1 0 0 0 0 0" gainprm="9000" ctrlrange="-1 1"/>
        <general name="h2_turn" joint="h2_rot" gear="0.6" gainprm="500" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""

# ==========================================
# 4. 環境クラス実装 (HideAndSeekEnv)
# ==========================================
class HideAndSeekEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(XML_CONTENT)
        self.data = mujoco.MjData(self.model)
        
        # 観測空間 (53次元)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        # アクション空間 (4次元): [Move, Turn, Grasp, Lock]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self._setup_ids()
        self.viewer = None
        self.render_mode = render_mode
        self.current_step = 0
        self.episode_count = 0 
        self.s0_stuck_timer = 0
        self.s0_recovery_mode = 0
        
        self.grasping = {1: None, 2: None}
        self.locked_boxes = {self.box1_body: False, self.box2_body: False}
        self.locked_pose = {}
        self.lock_cooldown = {1: 0, 2: 0}
        
        self.seeker_last_known_pos = None
        self.seeker_search_timer = 0
        self.seeker_random_target = np.zeros(2)
        self.seeker_target_pos = None 
        
        # 学習ターゲットID
        self.learning_agent_id = self.np_random.choice([1, 2]) if TRAIN_TARGET == "HIDER" else 0
        
        surround = np.linspace(0, 2*np.pi, 8, endpoint=False)
        front = np.linspace(-np.pi/6, np.pi/6, 5)
        self.lidar_angles = np.unique(np.concatenate([surround, front]))

        # --- NPCモデルのロード & 管理 ---
        self.npc_hider_agent = None
        self.npc_seeker_agent = None
        
        if os.path.exists(MODEL_PATH_HIDER):
            try:
                self.npc_hider_agent = Agent(self.observation_space.shape[0], self.action_space.shape[0])
                self.npc_hider_agent.load_state_dict(torch.load(MODEL_PATH_HIDER, map_location="cpu"))
                self.npc_hider_agent.eval() 
                print(f"Loaded NPC Hider Model from {MODEL_PATH_HIDER}")
            except Exception as e:
                print(f"Failed to load Hider NPC model: {e}")

        if os.path.exists(MODEL_PATH_SEEKER):
            try:
                self.npc_seeker_agent = Agent(self.observation_space.shape[0], self.action_space.shape[0])
                self.npc_seeker_agent.load_state_dict(torch.load(MODEL_PATH_SEEKER, map_location="cpu"))
                self.npc_seeker_agent.eval()
                print(f"Loaded NPC Seeker Model from {MODEL_PATH_SEEKER}")
            except Exception as e:
                print(f"Failed to load Seeker NPC model: {e}")

        # NPC用観測バッファ (CPU)
        self.npc_obs_history = {
            0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, torch.device("cpu")), # Seeker
            1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, torch.device("cpu")), # Hider1
            2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, torch.device("cpu")), # Hider2
        }
        
        # ★計測用：レイキャストスキップ率
        self.raycast_stats = {'total_attempts': 0, 'total_skips': 0}

        # ★視界キャッシュ（エージェントごと）と更新ステップ
        self.visible_cache = {0: {}, 1: {}, 2: {}}  # agent_id -> {body_id: bool}
        self.visible_cache_step = {0: -1, 1: -1, 2: -1}

    def __del__(self):
        self.close()

    def _setup_ids(self):
        self.s0_body = self.model.body('seeker_body').id
        self.h1_body = self.model.body('hider1_body').id
        self.h2_body = self.model.body('hider2_body').id
        self.box1_body = self.model.body('box1_body').id
        self.box2_body = self.model.body('box2_body').id
        self.ramp_body = self.model.body('ramp_body').id
        
        self.srot_adr = self.model.jnt_qposadr[self.model.joint('s_rot').id]
        self.h1rot_adr = self.model.jnt_qposadr[self.model.joint('h1_rot').id]
        self.h2rot_adr = self.model.jnt_qposadr[self.model.joint('h2_rot').id]
        
        self.s0_geoms = [i for i in range(self.model.ngeom) if "seeker" in self.model.geom(i).name]
        self.h0_geoms = [i for i in range(self.model.ngeom) if "hider" in self.model.geom(i).name]
        
        self.wall_ids = [i for i in range(self.model.ngeom) if "wall" in self.model.geom(i).name or "maze" in self.model.geom(i).name]
        self.wall_data = [(self.model.geom(wi).pos[:2], self.model.geom(wi).size[:2]) for wi in self.wall_ids]
        
        self.box_geoms = [i for i in range(self.model.ngeom) if "box" in self.model.geom(i).name]
        self.box1_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box1_geom")
        self.box2_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box2_geom")
        
        self.box1_joint_id = self.model.joint('box1_joint').id
        self.box2_joint_id = self.model.joint('box2_joint').id
        self.ramp_slope_geoms = [i for i in range(self.model.ngeom) if "ramp_slope" in self.model.geom(i).name]
        self.ramp_all_geoms = [i for i in range(self.model.ngeom) if "ramp" in self.model.geom(i).name and "mesh" not in self.model.geom(i).name]
        
        self.eq_grasp_b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp_box1")
        self.eq_grasp_b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp_box2")
        self.eq_lock_b1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_lock_box1")
        self.eq_lock_b2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_lock_box2")
        
        self.eq_ids = {}
        self.eq_ids[(1, self.box1_body)] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp1_b1")
        self.eq_ids[(1, self.box2_body)] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp1_b2")
        self.eq_ids[(2, self.box1_body)] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp2_b1")
        self.eq_ids[(2, self.box2_body)] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "eq_grasp2_b2")
        
        # ラベル用サイトID
        self.id_s_label = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_s_label")
        self.id_h1_label = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_h1_label")
        self.id_h2_label = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "site_h2_label")

    def _is_safe_pos(self, pos, check_radius, existing=[]):
        if np.linalg.norm(pos) < 1.0: return False
        for wp, ws in self.wall_data:
            dx, dy = max(abs(pos[0]-wp[0])-ws[0], 0), max(abs(pos[1]-wp[1])-ws[1], 0)
            if np.sqrt(dx**2 + dy**2) < check_radius + 0.1: return False
        for o_p, o_r in existing:
            if np.linalg.norm(pos - o_p) < (check_radius + o_r + 0.2): return False
        return True

    # ★修正(v18.9): 視界判定ロジックの修正
    # 戻り値を (is_visible, hit_body_id) に変更し、詳細な情報を返す
    def _is_visible(self, origin_pos, origin_rot, target_pos, target_body_id, exclude_body_id):
        diff = target_pos[:2] - origin_pos[:2]
        dist = np.linalg.norm(diff)
        
        if dist < 0.1: return True, target_body_id # 距離が極端に近い場合は見えるとする
        
        angle = (np.arctan2(diff[1], diff[0]) - origin_rot + np.pi) % (2*np.pi) - np.pi
        if abs(angle) > np.deg2rad(FOV_DEG / 2.0): return False, None # 視野角外
        
        direction = np.array([diff[0]/dist, diff[1]/dist, 0.0], dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        
        # 位置引数でレイキャスト
        res = mujoco.mj_ray(self.model, self.data, np.array([origin_pos[0], origin_pos[1], 0.5], dtype=np.float64), direction, None, 1, exclude_body_id, geomid)
        
        if res != -1:
            hit_body = self.model.geom_bodyid[geomid[0]]
            if hit_body == target_body_id:
                return True, hit_body # ターゲットにヒットした
            if res < dist - 0.4:
                return False, hit_body # 手前の何かにヒットした（そのhit_bodyを返す）
        
        return True, None # 遮蔽物なしで見える

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.episode_count += 1
        self.current_step = 0
        self.s0_stuck_timer = 0
        self.s0_recovery_mode = 0
        
        self.data.eq_active[:] = 0
        self.grasping = {1: None, 2: None}
        self.locked_boxes = {self.box1_body: False, self.box2_body: False}
        self.locked_pose = {}
        self.lock_cooldown = {1: 0, 2: 0}
        self.seeker_last_known_pos = None
        self.seeker_mode = "PATROLLING"
        self.seeker_search_timer = 0
        self.seeker_random_target = np.zeros(2)
        self.seeker_target_pos = None 
        
        # 学習ターゲットがHIDERならどちらかをランダム選択
        self.learning_agent_id = self.np_random.choice([1, 2]) if TRAIN_TARGET == "HIDER" else 0
        
        ex = []
        ramp_p = np.zeros(2)
        for _ in range(100):
            p = self.np_random.uniform(-4.5, 4.5, size=2)
            if self._is_safe_pos(p, 1.5, ex): ramp_p = p; break
        self.data.qpos[self.model.jnt_qposadr[self.model.joint('ramp_joint').id]:self.model.jnt_qposadr[self.model.joint('ramp_joint').id]+3] = [ramp_p[0], ramp_p[1], 0]; ex.append((ramp_p, 1.5))
        
        for jid in [self.box1_joint_id, self.box2_joint_id]:
            bp = np.zeros(2)
            for _ in range(100):
                p = self.np_random.uniform(-4.5, 4.5, 2)
                if self._is_safe_pos(p, 1.0, ex): bp = p; break
            self.data.qpos[self.model.jnt_qposadr[jid]:self.model.jnt_qposadr[jid]+3] = [bp[0], bp[1], 0.5]; ex.append((bp, 1.0))
            
        for bpref in ['s', 'h1', 'h2']:
            ap = np.zeros(2)
            for _ in range(100):
                p = self.np_random.uniform(-5, 5, 2)
                if self._is_safe_pos(p, 0.6, ex): ap = p; break
            jx = self.model.jnt_qposadr[self.model.joint(f'{bpref}_x').id]
            self.data.qpos[jx:jx+2] = ap; ex.append((ap, 0.6))
        
        mujoco.mj_forward(self.model, self.data)
        
        # NPCバッファのリセット
        for i in [0, 1, 2]:
            self.npc_obs_history[i].reset()
            obs = self._get_obs(i)
            self.npc_obs_history[i].update(obs)

        return self._get_obs(self.learning_agent_id), {}

    # ★修正(v18.9): _get_obs での視界判定キャッシュ導入
    def _get_obs(self, agent_id):
        # 共通情報取得
        if agent_id == 0: b_id, p_pref = self.s0_body, 's'
        elif agent_id == 1: b_id, p_pref = self.h1_body, 'h1'
        else: b_id, p_pref = self.h2_body, 'h2'

        # ★視界キャッシュ（サブサンプリング）
        refresh_vis = (self.current_step % VIS_UPDATE_INTERVAL == 0)
        cache = self.visible_cache[agent_id]
        if refresh_vis:
            cache.clear()
            self.visible_cache_step[agent_id] = self.current_step

        hp, hra = self.data.xpos[b_id][:2], self.data.qpos[self.model.jnt_qposadr[self.model.joint(f'{p_pref}_rot').id]]
        c, s = np.cos(-hra), np.sin(-hra); rot_mat = np.array([[c, -s], [s, c]])
        dof = self.model.jnt_dofadr[self.model.joint(f'{p_pref}_x').id]
        h_raw_vel = self.data.qvel[dof : dof+2]
        h_local_vel = rot_mat @ h_raw_vel
        h_obs_vel = h_local_vel / 12.0
        self_s = np.concatenate([h_obs_vel, [hra, np.cos(hra), np.sin(hra)]])
        
        # [最適化] NumPy配列の事前確保 (リストappend廃止)
        lidar = np.zeros(len(self.lidar_angles), dtype=np.float32)
        geomid = np.zeros(1, dtype=np.int32)
        
        for i, angle_offset in enumerate(self.lidar_angles):
            beam_dir = angle_offset + hra
            direction = np.array([np.cos(beam_dir), np.sin(beam_dir), 0.0], dtype=np.float64)
            # 位置引数
            dist = mujoco.mj_ray(self.model, self.data, np.array([hp[0], hp[1], 0.5], dtype=np.float64), direction, None, 1, b_id, geomid)
            lidar[i] = min(dist, 2.5)/2.5 if dist != -1 else 1.0

        # 3. Objects (Masked)
        def get_rel_info(target_id, lock=None):
            # サブサンプリング：refresh_vis=Trueのときだけ_is_visible()を呼ぶ
            self.raycast_stats['total_attempts'] += 1
            if not refresh_vis:
                # refresh_vis=False のときはキャッシュから取得（計算スキップ）
                if target_id in cache:
                    is_vis = cache[target_id]
                    self.raycast_stats['total_skips'] += 1
                else:
                    is_vis = False  # 初回は False
            else:
                # refresh_vis=True のときだけ計算
                tp = self.data.xpos[target_id]
                is_vis, hit_id = self._is_visible(hp, hra, tp, target_id, exclude_body_id=b_id)
                cache[target_id] = is_vis # ターゲットの結果を保存
                # 遮蔽物が見えていた場合、その遮蔽物も「見える」としてキャッシュ
                if not is_vis and hit_id is not None:
                     cache[hit_id] = True

            if is_vis:
                tp = self.data.xpos[target_id] # 位置は再取得（コスト小）
                rel_pos = rot_mat @ (tp[:2] - hp) / 12.0
                q = self.data.xquat[target_id]; yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
                tv = self.data.qvel[self.model.body_jntadr[target_id]:self.model.body_jntadr[target_id]+2] if self.model.body_jntadr[target_id] != -1 else np.zeros(2)
                rel_v = rot_mat @ (tv - self.data.qvel[dof:dof+2]) / 12.0
                info = [rel_pos, rel_v, [np.cos(yaw-hra), np.sin(yaw-hra)]]
                if lock is not None: info.append([1.0 if lock else 0.0])
                info.append([1.0]); return np.concatenate(info)
            return np.zeros(8 if lock is not None else 7, dtype=np.float32)

        # 遠い→近い順に判定し、近距離の遮蔽で後の対象をスキップできるようにする
        def collect_rel_infos(target_specs):
            # target_specs: List[Tuple[target_id, lock_or_None]]
            ordered = sorted(
                target_specs,
                key=lambda spec: np.linalg.norm(self.data.xpos[spec[0]][:2] - hp),
                reverse=True,  # 遠いものから
            )
            out = {}
            for tid, lock in ordered:
                out[tid] = get_rel_info(tid, lock)
            return out

        if agent_id == 0:
            specs = [
                (self.box1_body, self.locked_boxes[self.box1_body]),
                (self.box2_body, self.locked_boxes[self.box2_body]),
                (self.ramp_body, None),
                (self.h1_body, None),
                (self.h2_body, None),
            ]
            infos = collect_rel_infos(specs)
            h1 = infos[self.h1_body][:5]; h2 = infos[self.h2_body][:5]
            objs = [infos[self.box1_body], infos[self.box2_body], infos[self.ramp_body]]
            return np.concatenate([self_s, lidar, *objs, h1, h2, np.zeros(3, dtype=np.float32)]).astype(np.float32)
        
        partner = self.h2_body if agent_id == 1 else self.h1_body
        specs = [
            (self.box1_body, self.locked_boxes[self.box1_body]),
            (self.box2_body, self.locked_boxes[self.box2_body]),
            (self.ramp_body, None),
            (self.s0_body, None),
            (partner, None),
        ]
        infos = collect_rel_infos(specs)
        enemy = infos[self.s0_body][:5]; friend = infos[partner]
        st = np.array([1.0 if self.grasping[agent_id] else 0.0], dtype=np.float32)
        objs = [infos[self.box1_body], infos[self.box2_body], infos[self.ramp_body]]
        return np.concatenate([self_s, lidar, *objs, enemy, friend, st]).astype(np.float32)

    def _activate_lock(self, eq_id, box_body_id):
        p_box = self.data.xpos[box_body_id].copy()
        q_box = self.data.xquat[box_body_id].copy()
        self.model.eq_data[eq_id][:3] = p_box
        self.model.eq_data[eq_id][3:7] = q_box
        self.data.eq_active[eq_id] = 1
        
    def _activate_weld(self, eq_id, body1_id, body2_id):
        p1 = self.data.xpos[body1_id]
        q1 = self.data.xquat[body1_id]
        current_p2 = self.data.xpos[body2_id]
        current_q2 = self.data.xquat[body2_id]
        p_diff = current_p2 - p1
        p_rel = rotate_vec(quat_inv(q1), p_diff)
        q_rel = quat_mul(quat_inv(q1), current_q2)
        self.model.eq_data[eq_id][:3] = p_rel
        self.model.eq_data[eq_id][3:7] = q_rel
        self.data.eq_active[eq_id] = 1

    def _apply_action(self, agent_id, action):
        if agent_id == 0: return 0
        b_id, c_idx = (self.h1_body, 2) if agent_id == 1 else (self.h2_body, 4)
        hp = self.data.xpos[b_id]
        
        if action[2] > 0.5 and self.grasping[agent_id] is None:
            for box in [self.box1_body, self.box2_body]:
                if np.linalg.norm(hp - self.data.xpos[box]) < 1.2 and not self.locked_boxes[box]:
                    eq = self.eq_ids[(agent_id, box)]
                    p1, q1 = self.data.xpos[b_id], self.data.xquat[b_id]
                    self.model.eq_data[eq][:3] = rotate_vec(quat_inv(q1), self.data.xpos[box] - p1)
                    self.model.eq_data[eq][3:7] = quat_mul(quat_inv(q1), self.data.xquat[box])
                    self.data.eq_active[eq] = 1; self.grasping[agent_id] = box; break
        elif action[2] <= 0.5 and self.grasping[agent_id]:
            self.data.eq_active[self.eq_ids[(agent_id, self.grasping[agent_id])]] = 0; self.grasping[agent_id] = None
        if self.lock_cooldown[agent_id] <= 0:
            if action[3] > 0.5:
                for box in [self.box1_body, self.box2_body]:
                    if (self.grasping[agent_id] == box or np.linalg.norm(hp - self.data.xpos[box]) < 1.2) and not self.locked_boxes[box]:
                        self.locked_boxes[box] = True; jid = self.box1_joint_id if box == self.box1_body else self.box2_joint_id
                        self.locked_pose[box] = self.data.qpos[self.model.jnt_qposadr[jid]:self.model.jnt_qposadr[jid]+7].copy()
                        self.lock_cooldown[agent_id] = 10; break
            elif action[3] < -0.5:
                for box in [self.box1_body, self.box2_body]:
                    if np.linalg.norm(hp - self.data.xpos[box]) < 1.2 and self.locked_boxes[box]:
                        self.locked_boxes[box] = False; self.locked_pose.pop(box, None); self.lock_cooldown[agent_id] = 10; break
        return c_idx

    def _update_seeker_state(self):
        sp, sr = self.data.xpos[self.s0_body][:2], self.data.qpos[self.srot_adr]
        # _is_visibleの戻り値が変わったので修正
        v1, _ = self._is_visible(sp, sr, self.data.xpos[self.h1_body], self.h1_body, exclude_body_id=self.s0_body)
        v2, _ = self._is_visible(sp, sr, self.data.xpos[self.h2_body], self.h2_body, exclude_body_id=self.s0_body)
        if v1 or v2:
            self.seeker_target_pos = self.data.xpos[self.h1_body if v1 else self.h2_body][:2].copy()
            self.seeker_last_known_pos, self.seeker_mode = self.seeker_target_pos.copy(), "CHASING"
        elif self.seeker_last_known_pos is not None:
            if np.linalg.norm(sp - self.seeker_last_known_pos) > 0.5:
                target_pos = self.seeker_last_known_pos
                self.seeker_mode = "SEARCHING"
                self.seeker_target_pos = target_pos
                return target_pos
            else:
                self.seeker_last_known_pos = None
                self.seeker_search_timer = 50
        else:
            if self.seeker_search_timer <= 0: self.seeker_random_target, self.seeker_search_timer = self.np_random.uniform(-4, 4, 2), 80
            self.seeker_search_timer -= 1; self.seeker_target_pos, self.seeker_mode = self.seeker_random_target, "PATROLLING"

    def _seeker_rule_based_policy(self):
        if self.current_step < PREP_STEPS: return 0.0, 0.0
        sp, sr = self.data.xpos[self.s0_body][:2], self.data.qpos[self.srot_adr]
        ad = (np.arctan2(self.seeker_target_pos[1]-sp[1], self.seeker_target_pos[0]-sp[0]) - sr + np.pi)%(2*np.pi)-np.pi
        
        # スタック回避ロジック
        dof_s = self.model.jnt_dofadr[self.model.joint('s_x').id]
        s_vel = np.linalg.norm(self.data.qvel[dof_s:dof_s+2])
        sf = SEEKER_RB_THRUST
        sr_val = np.clip(ad * 6.0, -3.0, 3.0)
        
        if abs(ad) > SEEKER_RB_TURN_THRESH: sf *= 0.3
        
        if sf > 0.05 and s_vel < 0.05: self.s0_stuck_timer += 5
        else: self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
        
        if self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 15; self.s0_stuck_timer = 0
            
        if self.s0_recovery_mode > 0:
            sf = -0.2; sr_val = 1.5; self.s0_recovery_mode -= 1
            
        return sf, sr_val

    # NPC行動決定ロジック (CleanRL版)
    def _get_npc_action(self, agent_id, agent_type):
        obs = self._get_obs(agent_id)
        # 1. バッファ更新
        self.npc_obs_history[agent_id].update(obs)
        
        # 2. モデルがある場合は推論
        model = self.npc_seeker_agent if agent_type == "SEEKER" else self.npc_hider_agent
        if model is not None:
            obs_seq = self.npc_obs_history[agent_id].get()
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs_seq)
            return action.cpu().numpy()[0]
            
        # 3. モデルがない場合のフォールバック
        if agent_type == "SEEKER":
            sf, sr = self._seeker_rule_based_policy() # ルールベース
            return np.array([sf/SEEKER_THRUST_LIMIT, sr, 0.0, 0.0], dtype=np.float32)
        else:
            return self.action_space.sample() * 0.5 # ランダム

    def step(self, action):
        self.current_step += 1
        for i in [1, 2]: self.lock_cooldown[i] = max(0, self.lock_cooldown[i]-1)
        self._update_seeker_state()
        self.data.ctrl[:] = 0.0
        
        # 1. Main Agent (Learner)
        idx_main = 2 if self.learning_agent_id == 1 else (4 if self.learning_agent_id == 2 else 0)
        
        if TRAIN_TARGET == "HIDER":
            # Main Hider
            self.data.ctrl[idx_main] = action[0]*HIDER_THRUST_LIMIT
            self.data.ctrl[idx_main+1] = action[1]
            self._apply_action(self.learning_agent_id, action)

            # Partner Hider (NPC)
            partner_id = 2 if self.learning_agent_id == 1 else 1
            act_npc = self._get_npc_action(partner_id, "HIDER")
            idx_npc = 4 if partner_id == 2 else 2
            self.data.ctrl[idx_npc] = act_npc[0]*HIDER_THRUST_LIMIT
            self.data.ctrl[idx_npc+1] = act_npc[1]
            self._apply_action(partner_id, act_npc)

            # Enemy Seeker (NPC)
            act_seeker = self._get_npc_action(0, "SEEKER")
            if self.npc_seeker_agent: # Model
                self.data.ctrl[0] = act_seeker[0]*SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = act_seeker[1]
            else: # Rule (already scaled inside _get_npc_action if rule, but need consistent handling)
                # _get_npc_action returns normalized or raw depending on source.
                # If rule: normalized by division. So multiply back.
                self.data.ctrl[0] = act_seeker[0]*SEEKER_THRUST_LIMIT
                self.data.ctrl[1] = act_seeker[1]
            
        else: # SEEKER Training
            # Main Seeker
            self.data.ctrl[0] = action[0]*SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = action[1]
            
            # Hider NPCs
            for hid in [1, 2]:
                act_h = self._get_npc_action(hid, "HIDER")
                idx = 2 if hid == 1 else 4
                self.data.ctrl[idx] = act_h[0]*HIDER_THRUST_LIMIT
                self.data.ctrl[idx+1] = act_h[1]
                self._apply_action(hid, act_h)

        if self.current_step < PREP_STEPS:
            self.data.ctrl[0:2] = [0.0, 0.0]
            self.seeker_mode = "WAITING"

        for _ in range(ACTION_REPEAT):
            for box, pose in self.locked_pose.items():
                if self.locked_boxes[box]:
                    bid = self.box1_joint_id if box==self.box1_body else self.box2_joint_id
                    q = self.model.jnt_qposadr[bid]
                    d = self.model.jnt_dofadr[bid]
                    self.data.qpos[q:q+7] = pose
                    self.data.qvel[d:d+6] = 0
            mujoco.mj_step(self.model, self.data)

        h1p, h2p, sp = self.data.xpos[self.h1_body][:2], self.data.xpos[self.h2_body][:2], self.data.xpos[self.s0_body][:2]
        captured = any(np.linalg.norm(p - sp) < 0.85 for p in [h1p, h2p])
        min_dist = min(np.linalg.norm(h1p-sp), np.linalg.norm(h2p-sp))
        if TRAIN_TARGET == "HIDER":
            reward = REWARD_SURVIVAL + min(min_dist, 13.0)*REWARD_DISTANCE_COEFF
            if captured: reward += PENALTY_CAPTURE
        else:
            reward = -REWARD_DISTANCE_COEFF*min_dist + (REWARD_CAPTURE_BONUS if captured else 0)
        
        # ★計測：100ステップごとにスキップ率をログ出力
        if self.current_step % 100 == 0 and self.raycast_stats['total_attempts'] > 0:
            skip_rate = 100.0 * self.raycast_stats['total_skips'] / self.raycast_stats['total_attempts']
            print(f"[Step {self.current_step}] Raycast Skip Rate: {skip_rate:.1f}% ({self.raycast_stats['total_skips']}/{self.raycast_stats['total_attempts']})")
        
        terminated = captured or (self.current_step >= MAX_STEPS)
        return self._get_obs(self.learning_agent_id), reward, terminated, False, {}

    def render(self, stats=None):
        if self.render_mode == "human":
            if self.viewer is None: 
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.elevation, self.viewer.cam.distance = -60, 23.0
                self.viewer.cam.lookat[:] = [0, 0, 0] 
            
            for b, g in [(self.box1_body, self.box1_geom_id), (self.box2_body, self.box2_geom_id)]:
                if self.locked_boxes[b]: self.model.geom_rgba[g] = [0.8, 0.1, 0.1, 1.0]
                elif any(v == b for v in self.grasping.values()): self.model.geom_rgba[g] = [0.1, 0.1, 0.9, 1.0]
                else: self.model.geom_rgba[g] = [0.6, 0.4, 0.2, 1.0] if b==self.box1_body else [0.7, 0.5, 0.3, 1.0]
            
            if self.viewer.user_scn:
                ctx = self.viewer.user_scn
                ctx.ngeom = 0 # Reset
                
                # --- Unrolled Rendering ---
                pos_h1 = np.array(self.data.xpos[self.h1_body])
                rot_h1 = self.data.qpos[self.h1rot_adr]
                vis_h1 = []
                targets_h1 = [(self.box1_body, "Box1"), (self.box2_body, "Box2"), (self.ramp_body, "Ramp"), (self.s0_body, "Seeker"), (self.h2_body, "Friend")]
                for tid, name in targets_h1:
                    if self._is_visible(pos_h1, rot_h1, self.data.xpos[tid], tid, exclude_body_id=self.h1_body)[0]:
                        vis_h1.append(name)
                        if ctx.ngeom < ctx.maxgeom:
                            ctx.geoms[ctx.ngeom].label = "" 
                            mujoco.mjv_connector(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=2.0, from_=pos_h1+np.array([0,0,0.5]), to=self.data.xpos[tid]+np.array([0,0,0.5]))
                            ctx.geoms[ctx.ngeom].rgba = np.array([1, 1, 0, 0.6])
                            ctx.ngeom += 1
                if ctx.ngeom < ctx.maxgeom:
                    label_pos = self.data.site_xpos[self.id_h1_label]
                    ctx.geoms[ctx.ngeom].label = ""
                    mujoco.mjv_initGeom(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.array([0,0,0]), pos=label_pos, mat=np.eye(3).flatten(), rgba=np.array([1, 1, 0, 1]))
                    ctx.geoms[ctx.ngeom].label = f"H1 Vis:[{','.join(vis_h1)}]"
                    ctx.ngeom += 1

                pos_h2 = np.array(self.data.xpos[self.h2_body])
                rot_h2 = self.data.qpos[self.h2rot_adr]
                vis_h2 = []
                targets_h2 = [(self.box1_body, "Box1"), (self.box2_body, "Box2"), (self.ramp_body, "Ramp"), (self.s0_body, "Seeker"), (self.h1_body, "Friend")]
                for tid, name in targets_h2:
                    if self._is_visible(pos_h2, rot_h2, self.data.xpos[tid], tid, exclude_body_id=self.h2_body)[0]:
                        vis_h2.append(name)
                        if ctx.ngeom < ctx.maxgeom:
                            ctx.geoms[ctx.ngeom].label = ""
                            mujoco.mjv_connector(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=2.0, from_=pos_h2+np.array([0,0,0.5]), to=self.data.xpos[tid]+np.array([0,0,0.5]))
                            ctx.geoms[ctx.ngeom].rgba = np.array([0, 1, 1, 0.6])
                            ctx.ngeom += 1
                if ctx.ngeom < ctx.maxgeom:
                    label_pos = self.data.site_xpos[self.id_h2_label]
                    ctx.geoms[ctx.ngeom].label = ""
                    mujoco.mjv_initGeom(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.array([0,0,0]), pos=label_pos, mat=np.eye(3).flatten(), rgba=np.array([0, 1, 1, 1]))
                    ctx.geoms[ctx.ngeom].label = f"H2 Vis:[{','.join(vis_h2)}]"
                    ctx.ngeom += 1

                pos_s = np.array(self.data.xpos[self.s0_body])
                rot_s = self.data.qpos[self.srot_adr]
                for tid in [self.h1_body, self.h2_body]:
                    if self._is_visible(pos_s, rot_s, self.data.xpos[tid], tid, exclude_body_id=self.s0_body)[0]:
                        if ctx.ngeom < ctx.maxgeom:
                            ctx.geoms[ctx.ngeom].label = ""
                            mujoco.mjv_connector(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=3.0, from_=pos_s+np.array([0,0,0.5]), to=self.data.xpos[tid]+np.array([0,0,0.5]))
                            ctx.geoms[ctx.ngeom].rgba = np.array([1, 0, 0, 0.6])
                            ctx.ngeom += 1
                if ctx.ngeom < ctx.maxgeom:
                    c_dict = {"CHASING": [1,0,0,1], "SEARCHING": [1,0.5,0,1], "SCANNING": [0,1,1,1], "PATROLLING": [1,1,1,1], "WAITING": [0.5,0.5,0.5,1]}
                    rgba = c_dict.get(self.seeker_mode, [1,1,1,1])
                    label_pos = self.data.site_xpos[self.id_s_label]
                    ctx.geoms[ctx.ngeom].label = ""
                    mujoco.mjv_initGeom(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.array([0,0,0]), pos=label_pos, mat=np.eye(3).flatten(), rgba=np.array(rgba))
                    ctx.geoms[ctx.ngeom].label = f"S:{self.seeker_mode}"
                    ctx.ngeom += 1

            self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close(); self.viewer = None

# ==========================================
# 6. 学習ループ
# ==========================================
def main():
    run_name = f"{EXPERIMENT_NAME}_{int(time.time())}"
    
    if TRACK_WANDB:
        wandb.init(
            project=WANDB_PROJECT_NAME,
            entity=WANDB_ENTITY,
            sync_tensorboard=True,
            config=vars(),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
    
    # 観測履歴バッファ
    class ObsHistory:
        def __init__(self, num_envs, seq_len, obs_dim):
            self.buffer = torch.zeros((num_envs, seq_len, obs_dim)).to(device)
        def reset(self):
            self.buffer.zero_()
        def update(self, obs):
            self.buffer = torch.roll(self.buffer, -1, dims=1)
            self.buffer[:, -1, :] = torch.Tensor(obs).to(device)
        def get(self):
            return self.buffer

    # 環境構築: gym.vector.SyncVectorEnvを使用 (SubprocVecEnvの代替)
    def make_env(render_mode=None):
        def thunk():
            env = HideAndSeekEnv(render_mode=render_mode)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk

    if TRAIN_MODE:
        # ★AsyncVectorEnv (マルチプロセス) + 環境変数有効化 (Windows/Linux高速化)
        # 学習モードかつViewerなしの場合のみ
        if not USE_VIEWER:
             envs = gym.vector.AsyncVectorEnv(
                 [make_env(render_mode=None) for i in range(NUM_ENVS)]
             )
        else:
             # Viewerありの場合はSync (デバッグ用)
             envs = gym.vector.SyncVectorEnv([make_env(render_mode="human")])
    else:
        # 推論モード
        envs = gym.vector.SyncVectorEnv([make_env(render_mode="human")])

    agent = Agent(envs.single_observation_space.shape[0], envs.single_action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, TRANSFORMER_SEQ_LEN, envs.single_observation_space.shape[0])).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    
    global_step = 0
    obs_history = ObsHistory(NUM_ENVS if TRAIN_MODE else 1, TRANSFORMER_SEQ_LEN, envs.single_observation_space.shape[0])
    
    next_obs, _ = envs.reset(seed=SEED)
    next_done = torch.zeros(NUM_ENVS if TRAIN_MODE else 1).to(device)
    obs_history.reset(); obs_history.update(next_obs)
    writer.add_scalar("charts/step", 0, 0); writer.flush()

    # 推論モード (学習済みモデルがある場合、学習して更新する)
    if TRAIN_MODE and os.path.exists(SAVE_MODEL_PATH):
        print(f"Resuming training from: {SAVE_MODEL_PATH}")
        try: agent.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
        except Exception as e: print(f"Failed to resume model: {e}")

    # 推論モードのみ
    if not TRAIN_MODE:
        if os.path.exists(SAVE_MODEL_PATH):
            agent.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device)); print(f"Loaded model: {SAVE_MODEL_PATH}")
        else: print("No saved model found.")
        agent.eval()
        try:
            for ep in range(5):
                next_obs, _ = envs.reset(seed=SEED+ep); obs_history.reset(); obs_history.update(next_obs); done = False
                while not done:
                    ls = time.time()
                    with torch.no_grad(): action, _, _, _ = agent.get_action_and_value(obs_history.get())
                    next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                    done = np.logical_or(terminations, truncations)[0]
                    obs_history.update(next_obs)
                    envs.envs[0].unwrapped.render(stats={"Ep": ep})
                    time.sleep(max(0, (0.005*ACTION_REPEAT) - (time.time()-ls)))
                print(f"Episode {ep} finished")
        except KeyboardInterrupt: print("Inference interrupted")
        finally: envs.close()
        return

    print(f"Start Training: {TOTAL_TIMESTEPS} steps")
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

    try:
        for update in tqdm(range(1, num_updates + 1), desc="Updates"):
            episodic_returns = []; episodic_lengths = []
            for step in range(NUM_STEPS):
                global_step += NUM_ENVS
                obs[step] = obs_history.get(); dones[step] = next_done
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs_history.get())
                    values[step] = value.flatten()
                actions[step] = action; logprobs[step] = logprob
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.tensor(next_done).to(device, dtype=torch.float32)
                
                # Episodic Return Logging (v130: dict対応)
                if "episode" in infos:
                    mask = infos.get("_episode", [True] * len(infos["episode"]))
                    ep_data = infos["episode"]
                    if isinstance(ep_data, dict) and "r" in ep_data:
                        for i, is_done in enumerate(mask):
                            if is_done:
                                r = ep_data["r"][i]; l = ep_data["l"][i]
                                writer.add_scalar("charts/episodic_return", r, global_step)
                                writer.add_scalar("charts/episodic_length", l, global_step)
                                episodic_returns.append(r); episodic_lengths.append(l)

                if torch.any(next_done):
                    for i, d in enumerate(next_done):
                        if d: obs_history.buffer[i].zero_()
                obs_history.update(next_obs)
                if USE_VIEWER: envs.envs[0].unwrapped.render()

            with torch.no_grad():
                next_value = agent.get_value(obs_history.get()).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done; nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]; nextvalues = values[t + 1]
                    delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + values

            b_obs = obs.reshape((-1, TRANSFORMER_SEQ_LEN, agent.obs_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, agent.action_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            b_inds = np.arange(NUM_STEPS * NUM_ENVS)
            clipfracs = []
            
            for epoch in range(UPDATE_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, NUM_STEPS * NUM_ENVS, MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]
                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss
                    optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM); optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, global_step)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/reward", rewards.mean().item(), global_step)
            
            # 平均エピソード情報の記録
            if len(episodic_returns) > 0:
                writer.add_scalar("charts/mean_episodic_return", np.mean(episodic_returns), global_step)
                writer.add_scalar("charts/mean_episodic_length", np.mean(episodic_lengths), global_step)

            writer.flush()

            if update % 10 == 0:
                print(f"Update {update}, Step {global_step}, Loss: {loss.item():.3f}, Reward: {rewards.mean().item():.3f}")
                
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    torch.save(agent.state_dict(), SAVE_MODEL_PATH)
    print(f"Model saved to {SAVE_MODEL_PATH}")
    envs.close()
    writer.close()
    if TRACK_WANDB: wandb.finish()

if __name__ == "__main__":
    start_time = time.time()
    main()