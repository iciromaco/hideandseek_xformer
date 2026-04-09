# main17_transformer.py
# 演習第16回：Transformerによる高次推論 (CleanRL Style)
# 
# 【概要】
# PyTorchネイティブな実装で Transformer Actor-Critic (PPO) を構築したスクリプトです。
#
# 【修正点 (v110)】
# - renderメソッドの描画ロジックを「ヘルパー関数なしの完全ベタ書き」に変更。
# - 変数名を h1_*, h2_*, s_* と厳密に分離し、情報の混線・重複表示バグを根絶。
# - HideAndSeekEnvクラスの構造を main15.py の安定版と完全に一致。
#
# 【実行準備】
# uv add torch numpy gymnasium mujoco tensorboard matplotlib wandb

import os
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

# ==========================================
# 1. ハイパーパラメータ & 設定
# ==========================================
EXPERIMENT_NAME = "HideAndSeek_Transformer"
SEED = 1
TORCH_DETERMINISTIC = True
CUDA = True

# ★ モード設定
TRAIN_MODE = False         # True: 学習を実行 / False: 推論(鑑賞)モード
USE_VIEWER = True        # 学習中はFalse推奨。推論時は自動でTrueになります。
TRACK_WANDB = True        # WandBでログを取る場合はTrue

# WandB プロジェクト設定
WANDB_PROJECT_NAME = "HideAndSeek_Transformer_Project"
WANDB_ENTITY = None 

# PPO / Transformer設定
TOTAL_TIMESTEPS = 3000000
LEARNING_RATE = 3e-4
NUM_ENVS = 1              # 学習時の並列環境数
NUM_STEPS = 256           # 1回の更新で収集するステップ数 (per env)
MINIBATCH_SIZE = 256      # PPO更新時のミニバッチサイズ
UPDATE_EPOCHS = 4         # PPO更新回数
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Transformer構造
TRANSFORMER_SEQ_LEN = 16  # 過去何ステップを見るか (Context Length)
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_HEADS = 4

# 環境設定
TRAIN_TARGET = "HIDER" 
SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# --- 環境定数 ---
SAVE_PATH_HIDER = "ppo_lstm_hider_model_v15"
SAVE_PATH_SEEKER = "ppo_lstm_seeker_model_v15"
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

# SB3 (NPCモデルロード用: なければスキップ)
try:
    from sb3_contrib import RecurrentPPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

# ==========================================
# 2. 数学ヘルパー関数
# ==========================================
def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

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
# 3. MJCF (XML) 物理環境定義
# ==========================================
XML_CONTENT = """
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.005"/>
    <visual>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    </visual>
    <asset>
        <texture name="grid_tex" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid_mat" texture="grid_tex" texrepeat="1 1" reflectance="0.2"/>
        <mesh name="ramp_mesh" 
              vertex="-0.6666 -0.5 0.0   0.6666 -0.5 0.0   0.6666 -0.5 1.0   -0.6666 0.5 0.0   0.6666 0.5 0.0   0.6666 0.5 1.0" 
              face="0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0"/>
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
        
        <body name="box1_body" pos="2 -2 0.5">
            <joint name="box1_joint" type="free" damping="100.0"/>
            <geom name="box1_geom" type="box" size="0.6 0.6 0.5" rgba="0.6 0.4 0.2 1" mass="100" solref="0.02 1" condim="3" friction="1.0 0.005 0.0001"/>
        </body>

        <body name="box2_body" pos="-2 2 0.5">
            <joint name="box2_joint" type="free" damping="100.0"/>
            <geom name="box2_geom" type="box" size="0.6 0.6 0.5" rgba="0.7 0.5 0.3 1" mass="100" solref="0.02 1" condim="3" friction="1.0 0.005 0.0001"/>
        </body>
        
        <body name="seeker_anchor" pos="0 0 0.5">
            <joint name="s_x" type="slide" axis="1 0 0" damping="40"/>
            <joint name="s_y" type="slide" axis="0 1 0" damping="40"/>
            <joint name="s_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2" damping="20"/>
            <joint name="s_rot" type="hinge" axis="0 0 1" damping="50" armature="3.0"/>
            <body name="seeker_body">
                <site name="seeker_thrust_site" pos="0 0 0"/>
                <geom name="seeker_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="1.2 0.01 0"/>
                <geom name="seeker_capsule" type="capsule" size="0.3 0.2" rgba="0.9 0.1 0.1 1" mass="5"/>
                <geom name="seeker_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <geom name="seeker_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.9 0.1 0.1 1" contype="0" conaffinity="0"/>
            </body>
        </body>
        
        <!-- Hider 1 (青) -->
        <body name="hider1_anchor" pos="0 0 0.5">
            <joint name="h1_x" type="slide" axis="1 0 0" damping="40"/>
            <joint name="h1_y" type="slide" axis="0 1 0" damping="40"/>
            <joint name="h1_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2" damping="20"/>
            <joint name="h1_rot" type="hinge" axis="0 0 1" damping="50" armature="3.0"/>
            <body name="hider1_body">
                <site name="hider1_thrust_site" pos="0 0 0"/>
                <geom name="hider1_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="1.2 0.01 0"/>
                <geom name="hider1_capsule" type="capsule" size="0.3 0.2" rgba="0.1 0.1 0.9 1" mass="5"/>
                <geom name="hider1_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <geom name="hider1_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.1 0.1 0.9 1" contype="0" conaffinity="0"/>
            </body>
        </body>

        <!-- Hider 2 (水色) -->
        <body name="hider2_anchor" pos="0 0 0.5">
            <joint name="h2_x" type="slide" axis="1 0 0" damping="40"/>
            <joint name="h2_y" type="slide" axis="0 1 0" damping="40"/>
            <joint name="h2_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2" damping="20"/>
            <joint name="h2_rot" type="hinge" axis="0 0 1" damping="50" armature="3.0"/>
            <body name="hider2_body">
                <site name="hider2_thrust_site" pos="0 0 0"/>
                <geom name="hider2_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="1.2 0.01 0"/>
                <geom name="hider2_capsule" type="capsule" size="0.3 0.2" rgba="0.1 0.6 0.9 1" mass="5"/>
                <geom name="hider2_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <geom name="hider2_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.1 0.6 0.9 1" contype="0" conaffinity="0"/>
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
        self.seeker_mode = "PATROLLING"
        self.seeker_target_pos = None 
        
        self.learning_agent_id = 1 
        
        surround = np.linspace(0, 2*np.pi, 8, endpoint=False)
        front = np.linspace(-np.pi/6, np.pi/6, 5)
        self.lidar_angles = np.unique(np.concatenate([surround, front]))

        # --- NPCモデルロード ---
        self.hider_npc_model = None
        self.seeker_npc_model = None
        
        if SB3_AVAILABLE:
            if os.path.exists(SAVE_PATH_HIDER + ".zip"):
                try:
                    self.hider_npc_model = RecurrentPPO.load(SAVE_PATH_HIDER)
                    print("Loaded Hider NPC Model.")
                except Exception as e:
                    print(f"Warning: Failed to load Hider model: {e}")

            if TRAIN_TARGET == "HIDER" and os.path.exists(SAVE_PATH_SEEKER + ".zip"):
                try:
                    self.seeker_npc_model = RecurrentPPO.load(SAVE_PATH_SEEKER)
                    print("Loaded Seeker NPC Model.")
                except Exception as e:
                    print(f"Warning: Failed to load Seeker model: {e}")

        self.hider_npc_states = {1: None, 2: None}
        self.hider_npc_dones = {1: np.ones((1,), dtype=bool), 2: np.ones((1,), dtype=bool)}
        self.seeker_npc_states, self.seeker_npc_dones = None, np.ones((1,), dtype=bool)

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

    def _is_safe_pos(self, pos, check_radius, existing=[]):
        if np.linalg.norm(pos) < 1.0: return False
        for wp, ws in self.wall_data:
            dx, dy = max(abs(pos[0]-wp[0])-ws[0], 0), max(abs(pos[1]-wp[1])-ws[1], 0)
            if np.sqrt(dx**2 + dy**2) < check_radius + 0.1: return False
        for o_p, o_r in existing:
            if np.linalg.norm(pos - o_p) < (check_radius + o_r + 0.2): return False
        return True

    def _is_visible(self, origin_pos, origin_rot, target_pos, target_body_id, exclude_body_id):
        diff = target_pos[:2] - origin_pos[:2]
        dist = np.linalg.norm(diff)
        if dist < 0.1: return True 
        angle = (np.arctan2(diff[1], diff[0]) - origin_rot + np.pi) % (2*np.pi) - np.pi
        if abs(angle) > np.deg2rad(FOV_DEG / 2.0): return False 
        direction = np.array([diff[0]/dist, diff[1]/dist, 0.0], dtype=np.float64)
        geomid = np.zeros(1, dtype=np.int32)
        res = mujoco.mj_ray(self.model, self.data, np.array([origin_pos[0], origin_pos[1], 0.5], dtype=np.float64), direction, None, 1, exclude_body_id, geomid)
        if res != -1:
            hit_body = self.model.geom_bodyid[geomid[0]]
            if hit_body == target_body_id: return True
            if res < dist - 0.4: return False
        return True

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
        
        # 学習ターゲットがHIDERならどちらかをランダム選択
        self.learning_agent_id = self.np_random.choice([1, 2]) if TRAIN_TARGET == "HIDER" else 0
        
        # オブジェクト配置
        ex = []
        ramp_p = np.zeros(2)
        for _ in range(100):
            p = self.np_random.uniform(-4.5, 4.5, 2)
            if self._is_safe_pos(p, 1.5, ex): ramp_p = p; break
        self.data.qpos[self.model.jnt_qposadr[self.model.joint('ramp_joint').id]:self.model.jnt_qposadr[self.model.joint('ramp_joint').id]+3] = [ramp_p[0], ramp_p[1], 0]; ex.append((ramp_p, 1.5))
        
        for bid, jid in [(self.box1_body, self.box1_joint_id), (self.box2_body, self.box2_joint_id)]:
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
        
        # NPC初期化
        if self.hider_npc_model:
            self.hider_npc_states = {1: None, 2: None}
            self.hider_npc_dones = {1: np.ones((1,), dtype=bool), 2: np.ones((1,), dtype=bool)}
        if self.seeker_npc_model:
            self.seeker_npc_states, self.seeker_npc_dones = None, np.ones((1,), dtype=bool)

        return self._get_obs(self.learning_agent_id), {}

    def _get_obs(self, agent_id):
        # 共通情報取得
        if agent_id == 0: b_id, p_pref = self.s0_body, 's'
        elif agent_id == 1: b_id, p_pref = self.h1_body, 'h1'
        else: b_id, p_pref = self.h2_body, 'h2'
        
        hp = self.data.xpos[b_id][:2]
        hra = self.data.qpos[self.model.jnt_qposadr[self.model.joint(f'{p_pref}_rot').id]]
        
        # 2D回転行列 (Global -> Local)
        c, s = np.cos(-hra), np.sin(-hra)
        rot_mat = np.array([[c, -s], [s, c]])

        # 1. Self State (5)
        dof = self.model.jnt_dofadr[self.model.joint(f'{p_pref}_x').id]
        h_raw_vel = self.data.qvel[dof : dof+2]
        h_local_vel = rot_mat @ h_raw_vel
        h_obs_vel = h_local_vel / 12.0
        self_s = np.concatenate([h_obs_vel, [hra, np.cos(hra), np.sin(hra)]])

        # 2. Lidar (12)
        lidar = []
        for angle_offset in self.lidar_angles:
            beam_dir = angle_offset + hra
            direction = np.array([np.cos(beam_dir), np.sin(beam_dir), 0.0], dtype=np.float64)
            from_p = np.array([hp[0], hp[1], 0.5], dtype=np.float64)
            geomid = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(self.model, self.data, from_p, direction, None, 1, b_id, geomid)
            if dist != -1: lidar.append(min(dist, 2.5) / 2.5)
            else: lidar.append(1.0)
        lidar = np.array(lidar, dtype=np.float32)

        # 3. Objects (Masked)
        def get_rel_info(target_id, name, lock=None): 
            tp = self.data.xpos[target_id]
            is_vis = self._is_visible(hp, hra, tp, target_id, exclude_body_id=b_id)
            if is_vis:
                rel_pos_global = tp[:2] - hp
                rel_pos_local = rot_mat @ rel_pos_global
                rel_pos = rel_pos_local / 12.0
                
                q = self.data.xquat[target_id]; yaw = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
                tv = self.data.qvel[self.model.body_jntadr[target_id]:self.model.body_jntadr[target_id]+2] if self.model.body_jntadr[target_id] != -1 else np.zeros(2)
                rel_v = rot_mat @ (tv - self.data.qvel[dof:dof+2]) / 12.0
                info = [rel_pos, rel_v, [np.cos(yaw-hra), np.sin(yaw-hra)]]
                if lock is not None: info.append([1.0 if lock else 0.0])
                info.append([1.0]); return np.concatenate(info)
            return np.zeros(8 if lock is not None else 7, dtype=np.float32)

        if agent_id == 0:
            h1 = get_rel_info(self.h1_body, "H1")[:5]; h2 = get_rel_info(self.h2_body, "H2")[:5]
            objs = [get_rel_info(self.box1_body, "B1", self.locked_boxes[self.box1_body]), get_rel_info(self.box2_body, "B2", self.locked_boxes[self.box2_body]), get_rel_info(self.ramp_body, "R")]
            return np.concatenate([self_s, lidar, *objs, h1, h2, np.zeros(3, dtype=np.float32)]).astype(np.float32)
        
        partner = self.h2_body if agent_id == 1 else self.h1_body
        enemy = get_rel_info(self.s0_body, "S")[:5]; friend = get_rel_info(partner, "F")
        st = np.array([1.0 if self.grasping[agent_id] else 0.0], dtype=np.float32)
        objs = [get_rel_info(self.box1_body, "B1", self.locked_boxes[self.box1_body]), get_rel_info(self.box2_body, "B2", self.locked_boxes[self.box2_body]), get_rel_info(self.ramp_body, "R")]
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
        v1 = self._is_visible(sp, sr, self.data.xpos[self.h1_body], self.h1_body, self.s0_body)
        v2 = self._is_visible(sp, sr, self.data.xpos[self.h2_body], self.h2_body, self.s0_body)
        if v1 or v2:
            self.seeker_target_pos = self.data.xpos[self.h1_body if v1 else self.h2_body][:2].copy()
            self.seeker_last_known_pos, self.seeker_mode = self.seeker_target_pos.copy(), "CHASING"
        elif self.seeker_last_known_pos is not None:
            if np.linalg.norm(sp - self.seeker_last_known_pos) > 0.5: self.seeker_target_pos, self.seeker_mode = self.seeker_last_known_pos, "SEARCHING"
            else: self.seeker_last_known_pos, self.seeker_search_timer = None, 50
        else:
            if self.seeker_search_timer <= 0: self.seeker_random_target, self.seeker_search_timer = self.np_random.uniform(-4, 4, 2), 80
            self.seeker_search_timer -= 1; self.seeker_target_pos, self.seeker_mode = self.seeker_random_target, "PATROLLING"

    def _seeker_rule_based_policy(self):
        if self.current_step < PREP_STEPS: return 0.0, 0.0
        sp, sr = self.data.xpos[self.s0_body][:2], self.data.qpos[self.srot_adr]
        ad = (np.arctan2(self.seeker_target_pos[1]-sp[1], self.seeker_target_pos[0]-sp[0]) - sr + np.pi)%(2*np.pi)-np.pi
        return SEEKER_RB_THRUST * (0.3 if abs(ad) > SEEKER_RB_TURN_THRESH else 1.0), np.clip(ad*6.0, -3, 3)

    def step(self, action):
        self.current_step += 1
        for i in [1, 2]: self.lock_cooldown[i] = max(0, self.lock_cooldown[i]-1)
        self._update_seeker_state()
        self.data.ctrl[:] = 0.0
        
        if TRAIN_TARGET == "HIDER":
            idx_m = self._apply_action(self.learning_agent_id, action)
            self.data.ctrl[idx_m:idx_m+2] = [action[0]*HIDER_THRUST_LIMIT, action[1]]
            
            partner = 2 if self.learning_agent_id == 1 else 1
            if self.hider_npc_model:
                act_n, self.hider_npc_states[partner] = self.hider_npc_model.predict(self._get_obs(partner), state=self.hider_npc_states[partner], episode_start=self.hider_npc_dones[partner], deterministic=False)
                self.hider_npc_dones[partner][0] = False
                n_idx = self._apply_action(partner, act_n); self.data.ctrl[n_idx:n_idx+2] = [act_n[0]*HIDER_THRUST_LIMIT, act_n[1]]
            else:
                act_rand = self.action_space.sample() * 0.5
                n_idx = self._apply_action(partner, act_rand); self.data.ctrl[n_idx:n_idx+2] = [act_rand[0]*HIDER_THRUST_LIMIT, act_rand[1]]

            if self.seeker_npc_model:
                act_s, self.seeker_npc_states = self.seeker_npc_model.predict(self._get_obs(0), state=self.seeker_npc_states, episode_start=self.seeker_npc_dones, deterministic=False)
                self.seeker_npc_dones[0] = False
                sf, sr = act_s[0]*SEEKER_THRUST_LIMIT, act_s[1]
            else:
                sf, sr = self._seeker_rule_based_policy()
            self.data.ctrl[0:2] = [sf, sr]
            
        else: # SEEKER
            self.data.ctrl[0:2] = [action[0]*SEEKER_THRUST_LIMIT, action[1]]
            for i in [1, 2]:
                if self.hider_npc_model:
                    act_n, self.hider_npc_states[i] = self.hider_npc_model.predict(self._get_obs(i), state=self.hider_npc_states[i], episode_start=self.hider_npc_dones[i], deterministic=False)
                    self.hider_npc_dones[i][0] = False
                    n_idx = self._apply_action(i, act_n); self.data.ctrl[n_idx:n_idx+2] = [act_n[0]*HIDER_THRUST_LIMIT, act_n[1]]
                else:
                    act_n = self.action_space.sample() * 0.5
                    n_idx = self._apply_action(i, act_n); self.data.ctrl[n_idx:n_idx+2] = [act_n[0]*HIDER_THRUST_LIMIT, act_n[1]]

        if self.current_step < PREP_STEPS:
            self.data.ctrl[0:2] = [0.0, 0.0]
            self.seeker_mode = "WAITING"

        for _ in range(ACTION_REPEAT):
            for box, pose in self.locked_pose.items():
                if self.locked_boxes[box]:
                    bid = self.box1_joint_id if box==self.box1_body else self.box2_joint_id
                    self.data.qpos[self.model.jnt_qposadr[bid]:self.model.jnt_qposadr[bid]+7] = pose
                    self.data.qvel[self.model.jnt_dofadr[bid]:self.model.jnt_dofadr[bid]+6] = 0
            mujoco.mj_step(self.model, self.data)

        h1p, h2p, sp = self.data.xpos[self.h1_body][:2], self.data.xpos[self.h2_body][:2], self.data.xpos[self.s0_body][:2]
        captured = any(np.linalg.norm(p - sp) < 0.85 for p in [h1p, h2p])
        min_dist = min(np.linalg.norm(h1p-sp), np.linalg.norm(h2p-sp))
        if TRAIN_TARGET == "HIDER":
            reward = REWARD_SURVIVAL + min(min_dist, 13.0)*REWARD_DISTANCE_COEFF
            if captured: reward += PENALTY_CAPTURE
        else:
            reward = -REWARD_DISTANCE_COEFF*min_dist + (REWARD_CAPTURE_BONUS if captured else 0)
        
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
                ctx = self.viewer.user_scn; ctx.ngeom = 0
                
                # --- Unrolled Rendering Logic to prevent variable shadowing ---
                # Hider 1 (Yellow)
                pos_h1 = np.array(self.data.xpos[self.h1_body])
                rot_h1 = self.data.qpos[self.h1rot_adr]
                vis_h1 = []
                targets_h1 = [(self.box1_body, "Box1"), (self.box2_body, "Box2"), (self.ramp_body, "Ramp"), (self.s0_body, "Seeker"), (self.h2_body, "Friend")]
                for tid, name in targets_h1:
                    if self._is_visible(pos_h1, rot_h1, self.data.xpos[tid], tid, exclude_body_id=self.h1_body):
                        vis_h1.append(name)
                        if ctx.ngeom < ctx.maxgeom:
                            mujoco.mjv_connector(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=2.0, from_=pos_h1+np.array([0,0,0.5]), to=self.data.xpos[tid]+np.array([0,0,0.5]))
                            ctx.geoms[ctx.ngeom].rgba = np.array([1, 1, 0, 0.6])
                            ctx.ngeom += 1
                if ctx.ngeom < ctx.maxgeom:
                    mujoco.mjv_initGeom(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.array([0,0,0]), pos=pos_h1+np.array([0,0,1.2]), mat=np.eye(3).flatten(), rgba=np.array([1, 1, 0, 1]))
                    ctx.geoms[ctx.ngeom].label = f"H1 Vis:[{','.join(vis_h1)}]"
                    ctx.ngeom += 1

                # Hider 2 (Cyan)
                pos_h2 = np.array(self.data.xpos[self.h2_body])
                rot_h2 = self.data.qpos[self.h2rot_adr]
                vis_h2 = []
                targets_h2 = [(self.box1_body, "Box1"), (self.box2_body, "Box2"), (self.ramp_body, "Ramp"), (self.s0_body, "Seeker"), (self.h1_body, "Friend")]
                for tid, name in targets_h2:
                    if self._is_visible(pos_h2, rot_h2, self.data.xpos[tid], tid, exclude_body_id=self.h2_body):
                        vis_h2.append(name)
                        if ctx.ngeom < ctx.maxgeom:
                            mujoco.mjv_connector(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=2.0, from_=pos_h2+np.array([0,0,0.5]), to=self.data.xpos[tid]+np.array([0,0,0.5]))
                            ctx.geoms[ctx.ngeom].rgba = np.array([0, 1, 1, 0.6])
                            ctx.ngeom += 1
                if ctx.ngeom < ctx.maxgeom:
                    mujoco.mjv_initGeom(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.array([0,0,0]), pos=pos_h2+np.array([0,0,1.2]), mat=np.eye(3).flatten(), rgba=np.array([0, 1, 1, 1]))
                    ctx.geoms[ctx.ngeom].label = f"H2 Vis:[{','.join(vis_h2)}]"
                    ctx.ngeom += 1

                # Seeker (Red)
                pos_s = np.array(self.data.xpos[self.s0_body])
                rot_s = self.data.qpos[self.srot_adr]
                for tid in [self.h1_body, self.h2_body]:
                    if self._is_visible(pos_s, rot_s, self.data.xpos[tid], tid, exclude_body_id=self.s0_body):
                        if ctx.ngeom < ctx.maxgeom:
                            mujoco.mjv_connector(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=3.0, from_=pos_s+np.array([0,0,0.5]), to=self.data.xpos[tid]+np.array([0,0,0.5]))
                            ctx.geoms[ctx.ngeom].rgba = np.array([1, 0, 0, 0.6])
                            ctx.ngeom += 1
                if ctx.ngeom < ctx.maxgeom:
                    c_dict = {"CHASING": [1,0,0,1], "SEARCHING": [1,0.5,0,1], "SCANNING": [0,1,1,1], "PATROLLING": [1,1,1,1], "WAITING": [0.5,0.5,0.5,1]}
                    rgba = c_dict.get(self.seeker_mode, [1,1,1,1])
                    mujoco.mjv_initGeom(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.array([0,0,0]), pos=pos_s+np.array([0,0,1.2]), mat=np.eye(3).flatten(), rgba=np.array(rgba))
                    ctx.geoms[ctx.ngeom].label = f"S:{self.seeker_mode}"
                    ctx.ngeom += 1

            self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close(); self.viewer = None

# ==========================================
# 5. モデル定義 (Transformer Actor-Critic)
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.obs_dim = np.array(envs.single_observation_space.shape).prod()
        self.action_dim = np.array(envs.single_action_space.shape).prod()
        
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
        if action is None: action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(h)

# ==========================================
# 6. 学習ループ (CleanRL Style)
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

    # 環境構築
    if TRAIN_MODE:
        envs = gym.vector.SyncVectorEnv(
            [lambda: HideAndSeekEnv(render_mode="human" if i==0 and USE_VIEWER else None) for i in range(NUM_ENVS)]
        )
    else:
        envs = gym.vector.SyncVectorEnv([lambda: HideAndSeekEnv(render_mode="human")])

    agent = Agent(envs).to(device)
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
    
    # ★修正: ログ初期化
    writer.add_scalar("charts/step", 0, 0); writer.flush()

    # 推論モード
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
                    # ★修正: unwrapped経由でrenderを呼ぶ
                    envs.envs[0].unwrapped.render()
                    time.sleep(max(0, (0.005*ACTION_REPEAT) - (time.time()-ls)))
                print(f"Episode {ep} finished")
        except KeyboardInterrupt: print("Inference interrupted")
        finally: envs.close()
        return

    print(f"Start Training: {TOTAL_TIMESTEPS} steps")
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)

    try:
        for update in range(1, num_updates + 1):
            for step in range(NUM_STEPS):
                global_step += NUM_ENVS
                obs[step] = obs_history.get()
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(obs_history.get())
                    values[step] = value.flatten()
                
                actions[step] = action
                logprobs[step] = logprob
                
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.tensor(next_done).to(device, dtype=torch.float32)
                
                if torch.any(next_done):
                    for i, d in enumerate(next_done):
                        if d: obs_history.buffer[i].zero_()
                
                obs_history.update(next_obs)
                # ★修正: unwrapped経由でrenderを呼ぶ
                if USE_VIEWER: envs.envs[0].unwrapped.render()

            with torch.no_grad():
                next_value = agent.get_value(obs_history.get()).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
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

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/reward", rewards.mean().item(), global_step)
            
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
    main()