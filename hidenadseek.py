# main15.py
# 演習第14回：共進化への道 〜マルチエージェントと敵対的学習〜
# 
# 【Main14(第13回)からの変更点】
# 1. マルチエージェント環境 (2 vs 1):
#    - Hiderを2体 (hider1, hider2) に増員。Seekerは1体。
# 2. 交互学習 (Co-evolution) 基盤:
#    - TRAIN_TARGET = "HIDER" または "SEEKER" で学習対象を切り替え可能。
#    - 学習対象以外のエージェントは「NPC」として自律行動する。
# 3. 観測空間の拡張と統一:
#    - 学習対象の切り替えに対応するため、観測次元を53次元に統一。
#    - Seeker/Hider(Enemy)の相対速度情報を追加し、公平性を確保。
#    - Hider視点: Friend情報などを含む。
#    - Seeker視点: Hider1/2の情報を含む（Paddingあり）。
#    - ★修正(v52-95): 各種バグ修正適用済み。
#    - ★修正(v96-97): 消失していた _update_seeker_state メソッドを復元し、AttributeErrorを解消。                                                                                                                                                                                                                                                                       
import os

# --- 並列計算の競合を防ぐ環境変数設定 ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
import logging
import copy

# --- 警告抑制設定 ---
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import traceback
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from sb3_contrib import RecurrentPPO

# GUI競合対策
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# 1. グローバル設定・ハイパーパラメータ
# ==========================================
# ★学習モード設定: "HIDER" または "SEEKER"
TRAIN_TARGET = "HIDER" 
# TRAIN_TARGET = "SEEKER" 

TRAIN_MODE = True         
N_ENVS = 4 # ★修正: Viewerありの場合は1推奨
USE_VIEWER = True # ★修正: 動作確認のためTrue
ENABLE_RICH_PROGRESS = True 
RENDER_EVERY = 10           

# モデル保存名
SAVE_PATH_HIDER = "xtransformer_hider_model"
SAVE_PATH_SEEKER = "xtransformer_seeker_model"

# 今回学習・保存するパス
SAVE_PATH = SAVE_PATH_HIDER if TRAIN_TARGET == "HIDER" else SAVE_PATH_SEEKER
PLOT_PATH = f"learning_curve_xtransformer_{TRAIN_TARGET.lower()}.png"
TOTAL_TIMESTEPS = 5000000 

# 実験用パラメータ
ENT_COEF = 0.02             
LSTM_HIDDEN_SIZE = 256      

# 環境定数
ACTION_REPEAT = 16          
PREP_STEPS = 80             
MAX_STEPS = 300             
FOV_DEG = 135               

# 報酬設計
REWARD_SURVIVAL = 0.2       
REWARD_DISTANCE_COEFF = 0.08 
PENALTY_CAPTURE = -30.0     
PENALTY_STAGNATION = -0.5   
REWARD_CAPTURE_BONUS = 30.0 

# エージェント特性
HIDER_THRUST_LIMIT = 0.35   
SEEKER_THRUST_LIMIT = 0.40  
BOOST_MULTIPLIER = 5.0      

# Seeker Rule-Based AI Params (NPC用)
SEEKER_RB_THRUST = 0.38 
SEEKER_RB_TURN_THRESH = np.pi/6 

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
    # quaternion to yaw angle 
    q = data.xquat[body_id]
    return np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))

# ==========================================
# 3. MJCF (XML) 物理環境定義 (Hider x2)
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
# 学習対象設定 (HIDER or SEEKER)
DEFAULT_TRAIN_TARGET = "HIDER"

class HideAndSeekEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, train_target=DEFAULT_TRAIN_TARGET, render_mode=None,
                 save_path_hider=SAVE_PATH_HIDER, save_path_seeker=SAVE_PATH_SEEKER):
        super(HideAndSeekEnv, self).__init__()
        self.train_target = train_target
        self.render_mode = render_mode
        self.save_path_hider = save_path_hider
        self.save_path_seeker = save_path_seeker
        self.model = mujoco.MjModel.from_xml_string(XML_CONTENT)
        self.data = mujoco.MjData(self.model)
        
        # --- 観測空間 (53次元) ---
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        self._setup_ids()
        self.viewer, self.render_mode = None, render_mode
        self.current_step, self.episode_count = 0, 0
        self.s0_stuck_timer, self.s0_recovery_mode = 0, 0
        self.s0_prev_pos, self.h0_prev_pos = np.zeros(2), np.zeros(2)
        self.grasping_box_id = None
        self.locked_boxes = {self.box1_body: False, self.box2_body: False}
        self.locked_pose = {}
        self.lock_cooldown = 0
        self.seeker_last_known_pos = None 
        self.seeker_search_timer = 0
        self.seeker_random_target = np.zeros(2)
        self.seeker_mode = "PATROLLING" 
        self.seeker_target_pos = None
        self.episode_return = 0.0
        
        # ★修正(v82): visible_obj_names を完全に削除
        
        self.learning_agent_id = 1 
        
        surround = np.linspace(0, 2*np.pi, 8, endpoint=False)
        front = np.linspace(-np.pi/6, np.pi/6, 5)
        self.lidar_angles = np.unique(np.concatenate([surround, front]))

        # Hider NPCモデルロード
        self.hider_npc_model = None
        print(os.path.exists(SAVE_PATH_HIDER+ ".zip"))
        if TRAIN_TARGET == "SEEKER" and os.path.exists(self.save_path_hider + ".zip"):
             self.hider_npc_model = RecurrentPPO.load(self.save_path_hider)
             print("Loaded Hider NPC Model.")
        elif TRAIN_TARGET == "HIDER" and os.path.exists(self.save_path_hider + ".zip"):
             self.hider_npc_model = RecurrentPPO.load(self.save_path_hider)
             print("Loaded Hider Partner Model.")
        else:
             print("No Hider model found. Hider NPC will act randomly.")
        
        # 2. Seeker NPC (Hider学習時のみロード試行)
        self.seeker_npc_model = None
        if TRAIN_TARGET == "HIDER" and os.path.exists(self.save_path_seeker + ".zip"):
             self.seeker_npc_model = RecurrentPPO.load(self.save_path_seeker)
             print("Loaded Seeker NPC Model.")

        # NPC State Buffers
        self.hider_npc_states = {1: None, 2: None}
        self.hider_npc_dones = {1: np.ones((1,), dtype=bool), 2: np.ones((1,), dtype=bool)}
        self.seeker_npc_states = None
        self.seeker_npc_dones = np.ones((1,), dtype=bool)

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

    def _is_safe_pos(self, pos, check_radius, existing_objects=[]):
        if np.linalg.norm(pos) < 1.0: return False
        for w_pos, w_size in self.wall_data:
            dx = max(abs(pos[0]-w_pos[0]) - w_size[0], 0)
            dy = max(abs(pos[1]-w_pos[1]) - w_size[1], 0)
            if np.sqrt(dx**2 + dy**2) < check_radius + 0.1: return False
        for obj_pos, obj_radius in existing_objects:
            if np.linalg.norm(pos - obj_pos) < (check_radius + obj_radius + 0.2): return False
        return True

    def _check_collision_all(self, pos, threshold):
        for w_pos, w_size in self.wall_data:
            dx, dy = max(abs(pos[0]-w_pos[0])-w_size[0], 0), max(abs(pos[1]-w_pos[1])-w_size[1], 0)
            if dx**2 + dy**2 < threshold**2: return True
        for wid in self.box_geoms + self.ramp_all_geoms:
            center, size = self.data.geom_xpos[wid][:2], self.model.geom(wid).size[:2]
            dx, dy = max(abs(pos[0]-center[0])-size[0], 0), max(abs(pos[1]-center[1])-size[1], 0)
            if dx**2 + dy**2 < threshold**2: return True
        return False

    def _is_visible(self, origin_pos, origin_rot, target_pos, target_body_id, fov_deg=FOV_DEG, exclude_body_id=None):
        diff = target_pos[:2] - origin_pos[:2]
        dist = np.linalg.norm(diff)
        if dist < 0.1: return True 
        
        angle_to_target = np.arctan2(diff[1], diff[0])
        rel_angle = (angle_to_target - origin_rot + np.pi) % (2*np.pi) - np.pi
        if abs(rel_angle) > np.deg2rad(fov_deg / 2.0): return False 
        
        direction_2d = diff / dist
        direction = np.array([direction_2d[0], direction_2d[1], 0.0], dtype=np.float64)
        from_p = np.array([origin_pos[0], origin_pos[1], 0.5], dtype=np.float64)
        
        geomid = np.zeros(1, dtype=np.int32)
        if exclude_body_id is None: exclude_body_id = self.h1_body 
        
        dist_ray = mujoco.mj_ray(self.model, self.data, from_p, direction, None, 1, exclude_body_id, geomid)
        
        if dist_ray != -1:
             hit_geom_id = geomid[0]
             hit_body_id = self.model.geom_bodyid[hit_geom_id]
             if hit_body_id == target_body_id:
                 return True 
             if dist_ray < dist - 0.5:
                 return False

        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.episode_count += 1
        self.current_step = 0
        self.s0_stuck_timer = 0
        self.s0_recovery_mode = 0
        self.episode_return = 0.0
        
        self.data.eq_active[:] = 0
        self.grasping = {1: None, 2: None}
        self.locked_boxes = {self.box1_body: False, self.box2_body: False}
        self.locked_pose = {}
        self.lock_cooldown = {1: 0, 2: 0}
        self.seeker_last_known_pos = None
        self.seeker_mode = "PATROLLING"
        self.seeker_target_pos = None 
        
        if TRAIN_TARGET == "HIDER":
            self.learning_agent_id = self.np_random.choice([1, 2])
        else: 
            self.learning_agent_id = 0

        dof_b1 = self.model.jnt_dofadr[self.box1_joint_id]
        dof_b2 = self.model.jnt_dofadr[self.box2_joint_id]
        self.model.dof_damping[dof_b1 : dof_b1+6] = 100.0
        self.model.dof_damping[dof_b2 : dof_b2+6] = 100.0
        
        ramp_pos = np.zeros(2)
        for _ in range(100):
            p = self.np_random.uniform(-4.5, 4.5, size=2)
            if self._is_safe_pos(p, 1.5, []): ramp_pos = p; break
        ramp_q = self.model.jnt_qposadr[self.model.joint('ramp_joint').id]
        self.data.qpos[ramp_q:ramp_q+3] = [ramp_pos[0], ramp_pos[1], 0]
        
        box1_pos = np.zeros(2)
        for _ in range(100):
            p = self.np_random.uniform(-4.5, 4.5, size=2)
            if self._is_safe_pos(p, 1.0, [(ramp_pos, 1.5)]): box1_pos = p; break
        b1_q = self.model.jnt_qposadr[self.box1_joint_id]
        self.data.qpos[b1_q:b1_q+3] = [box1_pos[0], box1_pos[1], 0.5]
        
        box2_pos = np.zeros(2)
        for _ in range(100):
            p = self.np_random.uniform(-4.5, 4.5, size=2)
            if self._is_safe_pos(p, 1.0, [(ramp_pos, 1.5), (box1_pos, 1.0)]): box2_pos = p; break
        b2_q = self.model.jnt_qposadr[self.box2_joint_id]
        self.data.qpos[b2_q:b2_q+3] = [box2_pos[0], box2_pos[1], 0.5]
        
        existing = [(ramp_pos, 1.5), (box1_pos, 0.8), (box2_pos, 0.8)]
        sx, sy = -2.5, -2.5
        for _ in range(100):
            p = self.np_random.uniform(-5.0, 5.0, size=2)
            if self._is_safe_pos(p, 0.6, existing): sx, sy = p[0], p[1]; break
        
        hx, hy = 2.5, 2.5
        for _ in range(100):
            p = self.np_random.uniform(-5.0, 5.0, size=2)
            if self._is_safe_pos(p, 0.6, existing) and np.linalg.norm(p - np.array([sx, sy])) > 3.5:
                hx, hy = p[0], p[1]; break
        
        h2x, h2y = 2.5, -2.5 # Hider2
        for _ in range(100):
            p = self.np_random.uniform(-5.0, 5.0, size=2)
            if self._is_safe_pos(p, 0.6, existing) and np.linalg.norm(p - np.array([sx, sy])) > 3.5 and np.linalg.norm(p - np.array([hx, hy])) > 1.0:
                h2x, h2y = p[0], p[1]; break
        
        sx_adr = self.model.jnt_qposadr[self.model.joint('s_x').id]
        sy_adr = self.model.jnt_qposadr[self.model.joint('s_y').id]
        hx_adr = self.model.jnt_qposadr[self.model.joint('h1_x').id]
        hy_adr = self.model.jnt_qposadr[self.model.joint('h1_y').id]
        h2x_adr = self.model.jnt_qposadr[self.model.joint('h2_x').id]
        h2y_adr = self.model.jnt_qposadr[self.model.joint('h2_y').id]
        
        self.data.qpos[sx_adr] = sx
        self.data.qpos[sy_adr] = sy
        self.data.qpos[hx_adr] = hx
        self.data.qpos[hy_adr] = hy
        self.data.qpos[h2x_adr] = h2x
        self.data.qpos[h2y_adr] = h2y
        
        s_rot_adr = self.model.jnt_qposadr[self.model.joint('s_rot').id]
        h1_rot_adr = self.model.jnt_qposadr[self.model.joint('h1_rot').id]
        h2_rot_adr = self.model.jnt_qposadr[self.model.joint('h2_rot').id]
        self.data.qpos[s_rot_adr] = self.np_random.uniform(0, 2*np.pi)
        self.data.qpos[h1_rot_adr] = self.np_random.uniform(0, 2*np.pi)
        self.data.qpos[h2_rot_adr] = self.np_random.uniform(0, 2*np.pi)
        
        self.s0_prev_pos, self.h0_prev_pos = np.array([sx, sy]), np.array([hx, hy])

        # ★修正(v62): 初期ブースト
        if self.learning_agent_id == 1:
             target_h = np.array([hx, hy])
        elif self.learning_agent_id == 2:
             target_h = np.array([h2x, h2y])
        else: # Seeker学習時はHider1
             target_h = np.array([hx, hy])
        self.seeker_last_known_pos = target_h
        self.seeker_search_timer = 0
        self.seeker_mode = "SEARCHING"

        mujoco.mj_forward(self.model, self.data)
        
        # NPCの初期化 (状態リセット)
        if self.hider_npc_model:
            self.hider_npc_states = {1: None, 2: None}
            self.hider_npc_dones = {1: np.ones((1,), dtype=bool), 2: np.ones((1,), dtype=bool)}
        if self.seeker_npc_model:
            self.seeker_npc_states = None
            self.seeker_npc_dones = np.ones((1,), dtype=bool)

        return self._get_obs(self.learning_agent_id), {}

    def _get_obs(self, agent_id):
        # 共通情報取得
        if agent_id == 0: # Seeker
            my_body, my_joint = self.s0_body, 's'
        elif agent_id == 1: # Hider1
            my_body, my_joint = self.h1_body, 'h1'
        else: # Hider2
            my_body, my_joint = self.h2_body, 'h2'
            
        hp = self.data.xpos[my_body][:2]
        hra = self.data.qpos[self.model.jnt_qposadr[self.model.joint(f'{my_joint}_rot').id]]
        
        # 2D回転行列 (Global -> Local)
        c, s = np.cos(-hra), np.sin(-hra)
        rot_mat = np.array([[c, -s], [s, c]])

        # 1. Self State (5)
        dof = self.model.jnt_dofadr[self.model.joint(f'{my_joint}_x').id]
        h_raw_vel = self.data.qvel[dof : dof+2]
        h_local_vel = rot_mat @ h_raw_vel
        h_obs_vel = h_local_vel / 12.0
        self_state = np.concatenate([h_obs_vel, [hra, np.cos(hra), np.sin(hra)]])

        # 2. Lidar (12)
        lidar = []
        for angle_offset in self.lidar_angles:
            beam_dir = angle_offset + hra
            direction = np.array([np.cos(beam_dir), np.sin(beam_dir), 0.0], dtype=np.float64)
            from_p = np.array([hp[0], hp[1], 0.5], dtype=np.float64)
            geomid = np.zeros(1, dtype=np.int32)
            dist = mujoco.mj_ray(self.model, self.data, from_p, direction, None, 1, my_body, geomid)
            if dist != -1: lidar.append(min(dist, 2.5) / 2.5)
            else: lidar.append(1.0)
        lidar = np.array(lidar, dtype=np.float32)

        # 3. Objects (Masked)
        # ★修正: 副作用なしで観測のみを生成
        def get_rel_info(target_id, name, is_locked=None): 
            tp = self.data.xpos[target_id]
            is_vis = self._is_visible(hp, hra, tp, target_id, exclude_body_id=my_body)
            if is_vis:
                rel_pos_global = tp[:2] - hp
                rel_pos_local = rot_mat @ rel_pos_global
                rel_pos = rel_pos_local / 12.0
                
                q = self.data.xquat[target_id]
                yaw = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
                rel_yaw = yaw - hra
                
                j_id = self.model.body_jntadr[target_id]
                if j_id != -1:
                    dof = self.model.jnt_dofadr[j_id]
                    obj_raw_vel = self.data.qvel[dof : dof+2]
                    # 相対速度 = (相手 - 自分) のローカル成分
                    rel_vel_global = obj_raw_vel - h_raw_vel
                    rel_vel_local = rot_mat @ rel_vel_global
                    vel = rel_vel_local / 12.0
                else: 
                    # 固定物
                    rel_vel_global = -h_raw_vel
                    rel_vel_local = rot_mat @ rel_vel_global
                    vel = rel_vel_local / 12.0

                info = [rel_pos, vel, [np.cos(rel_yaw), np.sin(rel_yaw)]]
                if is_locked is not None:
                    info.append([1.0 if is_locked else 0.0])
                info.append([1.0]) # Visible Flag
                return np.concatenate(info)
            else:
                # 見えない場合はゼロ埋め (可視フラグも0)
                dims = 8 if is_locked is not None else 7
                return np.zeros(dims, dtype=np.float32)

        # 構築
        b1_info = get_rel_info(self.box1_body, "Box1", self.locked_boxes[self.box1_body])
        b2_info = get_rel_info(self.box2_body, "Box2", self.locked_boxes[self.box2_body])
        r_info = get_rel_info(self.ramp_body, "Ramp", None)

        # Target/Partner Info
        if agent_id == 0: # Seeker
            def get_agent_info(target_id, name):
                # get_rel_info は [pos, vel, rot, vis] (7dim) を返す
                # SeekerはHiderの回転を知る必要性は薄いので [pos, vel, vis] (5dim) に削る
                info = get_rel_info(target_id, name, None)
                # info = [px, py, vx, vy, cos, sin, vis]
                return np.concatenate([info[:4], info[6:]]) # rot(4,5) skip

            h1_info = get_agent_info(self.h1_body, "Hider1")
            h2_info = get_agent_info(self.h2_body, "Hider2")
            pad = np.zeros(3, dtype=np.float32)
            
            return np.concatenate([self_state, lidar, b1_info, b2_info, r_info, h1_info, h2_info, pad]).astype(np.float32)

        else: # Hider
            partner_body = self.h2_body if agent_id == 1 else self.h1_body
            enemy_body = self.s0_body
            
            # Enemy(5) - [pos(2), vel(2), vis(1)]
            def get_enemy_info(target_id, name):
                info = get_rel_info(target_id, name, None)
                return np.concatenate([info[:4], info[6:]]) # rot skip

            e_info = get_enemy_info(enemy_body, "Seeker")
            
            # Friend(7) - フル情報 (Lockなし, Visあり)
            f_info = get_rel_info(partner_body, "Friend", None)
            
            # Status(1)
            st = np.array([1.0 if self.grasping[agent_id] is not None else 0.0], dtype=np.float32)
            
            # Self(5)+Lidar(12)+B1(8)+B2(8)+R(7)+E(5)+F(7)+S(1) = 53
            return np.concatenate([self_state, lidar, b1_info, b2_info, r_info, e_info, f_info, st]).astype(np.float32)

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
        if agent_id == 0: # Seeker
            # SeekerはGrasp/Lock不可
            ctrl_idx = 0
            return ctrl_idx

        # Hider Logic
        if agent_id == 1:
            body, joint_prefix = self.h1_body, 'h1'
            ctrl_idx = 2 
        else:
            body, joint_prefix = self.h2_body, 'h2'
            ctrl_idx = 4 

        hp = self.data.xpos[body]
        # Grasp
        if action[2] > 0.5:
            if self.grasping[agent_id] is None:
                d1 = np.linalg.norm(hp - self.data.xpos[self.box1_body])
                d2 = np.linalg.norm(hp - self.data.xpos[self.box2_body])
                target = None
                if d1 < 1.2: target = (self.eq_ids[(agent_id, self.box1_body)], self.box1_body)
                elif d2 < 1.2: target = (self.eq_ids[(agent_id, self.box2_body)], self.box2_body)
                
                if target:
                    eq, box = target
                    if not self.locked_boxes[box]:
                        self._activate_weld(eq, body, box)
                        self.grasping[agent_id] = box
        else:
            if self.grasping[agent_id] is not None:
                box = self.grasping[agent_id]
                self.data.eq_active[self.eq_ids[(agent_id, box)]] = 0
                self.grasping[agent_id] = None

        # Lock
        if self.lock_cooldown[agent_id] <= 0:
            lock_cmd = action[3]
            target_box = None
            d1 = np.linalg.norm(hp - self.data.xpos[self.box1_body])
            d2 = np.linalg.norm(hp - self.data.xpos[self.box2_body])
            
            if self.grasping[agent_id] == self.box1_body or (self.grasping[agent_id] is None and d1 < 1.2):
                target_box = (self.eq_lock_b1, self.box1_body, self.box1_geom_id, self.box1_joint_id)
            elif self.grasping[agent_id] == self.box2_body or (self.grasping[agent_id] is None and d2 < 1.2):
                target_box = (self.eq_lock_b2, self.box2_body, self.box2_geom_id, self.box2_joint_id)
                
            if target_box:
                eq, box, geom, joint = target_box
                # Lock
                if lock_cmd > 0.5 and not self.locked_boxes[box]:
                    dof = self.model.jnt_dofadr[joint]
                    self.data.qvel[dof:dof+6] = 0
                    self.model.dof_damping[dof:dof+6] = 100000.0
                    
                    self.data.eq_active[eq] = 0 # Weldなし

                    if self.grasping[agent_id] == box:
                        self.data.eq_active[self.eq_ids[(agent_id, box)]] = 0
                        self.grasping[agent_id] = None
                    
                    self.locked_boxes[box] = True
                    q_adr = self.model.jnt_qposadr[joint]
                    self.locked_pose[box] = self.data.qpos[q_adr:q_adr+7].copy()
                    
                    self.model.geom_rgba[geom][:] = [0.8, 0.1, 0.1, 1.0]
                    self.lock_cooldown[agent_id] = 10
                    
                # Unlock
                elif lock_cmd < -0.5 and self.locked_boxes[box]:
                    dof = self.model.jnt_dofadr[joint]
                    self.model.dof_damping[dof:dof+6] = 100.0
                    self.data.eq_active[eq] = 0 
                    
                    self.locked_boxes[box] = False
                    if box in self.locked_pose: del self.locked_pose[box]
                    
                    color = [0.6, 0.4, 0.2, 1.0] if box == self.box1_body else [0.7, 0.5, 0.3, 1.0]
                    self.model.geom_rgba[geom][:] = color
                    self.lock_cooldown[agent_id] = 10

        return ctrl_idx 

    def step(self, action):
        
        self.current_step += 1
        for i in [1, 2]:
            if self.lock_cooldown[i] > 0: self.lock_cooldown[i] -= 1
            
        # ★修正(v96): メソッド復元
        self._update_seeker_state()
        
        # --- Actions ---
        if TRAIN_TARGET == "HIDER":
            # 1. Main Hider
            idx_main = self._apply_action(self.learning_agent_id, action)
            
            # 2. NPC Hider (Model)
            partner_id = 2 if self.learning_agent_id == 1 else 1
            if self.hider_npc_model:
                obs_npc = self._get_obs(partner_id)
                act_npc, self.hider_npc_states[partner_id] = self.hider_npc_model.predict(
                    obs_npc, state=self.hider_npc_states[partner_id], episode_start=self.hider_npc_dones[partner_id], deterministic=False
                )
                self.hider_npc_dones[partner_id][0] = False
            else:
                act_npc = self.action_space.sample() * 0.5
            idx_npc = self._apply_action(partner_id, act_npc)
            
            # 3. Seeker
            if self.seeker_npc_model:
                obs_seeker = self._get_obs(0) # Seeker(0)視点
                act_seeker, self.seeker_npc_states = self.seeker_npc_model.predict(
                    obs_seeker, state=self.seeker_npc_states, episode_start=self.seeker_npc_dones, deterministic=False
                )
                self.seeker_npc_dones[0] = False
                sf0 = act_seeker[0] * SEEKER_THRUST_LIMIT
                sr0 = act_seeker[1]
            else:
                sf0, sr0 = self._seeker_rule_based_policy()
            
            idx_seeker = 0

        else: # SEEKER Training
            # 1. Main Seeker
            sf0, sr0 = action[0]*SEEKER_THRUST_LIMIT, action[1]
            idx_seeker = 0
            
            # 2. NPC Hiders (Model x2)
            acts_hider = {}
            for hid in [1, 2]:
                if self.hider_npc_model:
                    obs_npc = self._get_obs(hid)
                    act_h, self.hider_npc_states[hid] = self.hider_npc_model.predict(
                        obs_npc, state=self.hider_npc_states[hid], episode_start=self.hider_npc_dones[hid], deterministic=False
                    )
                    self.hider_npc_dones[hid][0] = False
                else:
                    act_h = self.action_space.sample() * 0.5
                acts_hider[hid] = act_h
            
            idx_h1 = self._apply_action(1, acts_hider[1])
            idx_h2 = self._apply_action(2, acts_hider[2])

        # ★修正(v95): 準備期間中のSeeker強制停止
        if self.current_step < PREP_STEPS:
            sf0, sr0 = 0.0, 0.0
            self.seeker_mode = "WAITING"

        # --- Physics Loop ---
        for _ in range(ACTION_REPEAT):
            if TRAIN_TARGET == "HIDER":
                self.data.ctrl[idx_main] = action[0]*HIDER_THRUST_LIMIT
                self.data.ctrl[idx_main+1] = action[1]
                self.data.ctrl[idx_npc] = act_npc[0]*HIDER_THRUST_LIMIT
                self.data.ctrl[idx_npc+1] = act_npc[1]
                self.data.ctrl[0] = sf0
                self.data.ctrl[1] = sr0
            else:
                self.data.ctrl[0] = sf0
                self.data.ctrl[1] = sr0
                self.data.ctrl[2] = acts_hider[1][0]*HIDER_THRUST_LIMIT
                self.data.ctrl[3] = acts_hider[1][1]
                self.data.ctrl[4] = acts_hider[2][0]*HIDER_THRUST_LIMIT
                self.data.ctrl[5] = acts_hider[2][1]
            
            # Velocity Freeze
            for box, pose in self.locked_pose.items():
                if self.locked_boxes[box]:
                    bid = self.box1_joint_id if box==self.box1_body else self.box2_joint_id
                    q = self.model.jnt_qposadr[bid]
                    d = self.model.jnt_dofadr[bid]
                    self.data.qpos[q:q+7] = pose
                    self.data.qvel[d:d+6] = 0
                    
            mujoco.mj_step(self.model, self.data)

        # --- Reward & Done ---
        captured = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            n1, n2 = self.model.geom(c.geom1).name, self.model.geom(c.geom2).name
            if ("seeker" in n1 and "hider" in n2) or ("seeker" in n2 and "hider" in n1):
                captured = True; break

        h1p = self.data.xpos[self.h1_body][:2]
        h2p = self.data.xpos[self.h2_body][:2]
        if max(abs(h1p)) > 6.5 or max(abs(h2p)) > 6.5: captured = True

        sp = self.data.xpos[self.s0_body][:2]
        d1 = np.linalg.norm(h1p - sp)
        d2 = np.linalg.norm(h2p - sp)
        min_dist = min(d1, d2)
        
        if TRAIN_TARGET == "HIDER":
            reward = REWARD_SURVIVAL + min(min_dist, 13.0) * REWARD_DISTANCE_COEFF
            if captured: reward += PENALTY_CAPTURE
        else: # SEEKER
            reward = -REWARD_DISTANCE_COEFF * min_dist
            if captured: reward += REWARD_CAPTURE_BONUS

        truncated = self.current_step >= MAX_STEPS  
        terminated = captured
        self.episode_return += reward
        return self._get_obs(self.learning_agent_id), reward, terminated, truncated, {}
    
    # ★修正(v96-97): メソッド復元
    def _update_seeker_state(self):
        sp = self.data.xpos[self.s0_body][:2]
        s_rot = self.data.qpos[self.srot_adr]
        
        vis1 = self._is_visible(sp, s_rot, self.data.xpos[self.h1_body], self.h1_body, exclude_body_id=self.s0_body)
        vis2 = self._is_visible(sp, s_rot, self.data.xpos[self.h2_body], self.h2_body, exclude_body_id=self.s0_body)
        
        target_pos = None
        # CHASING
        if vis1: target_pos = self.data.xpos[self.h1_body][:2]
        elif vis2: target_pos = self.data.xpos[self.h2_body][:2]
        
        if target_pos is not None:
            self.seeker_last_known_pos = target_pos.copy()
            self.seeker_search_timer = 0
            self.seeker_mode = "CHASING"
            self.seeker_target_pos = target_pos 
            return target_pos

        # SEARCHING
        if self.seeker_last_known_pos is not None:
            if np.linalg.norm(sp - self.seeker_last_known_pos) > 0.5:
                target_pos = self.seeker_last_known_pos
                self.seeker_mode = "SEARCHING"
                self.seeker_target_pos = target_pos
                return target_pos
            else:
                self.seeker_last_known_pos = None
                self.seeker_search_timer = 50
        
        # PATROLLING
        if self.seeker_search_timer <= 0:
             self.seeker_random_target = self.np_random.uniform(-4.0, 4.0, size=2)
             self.seeker_search_timer = 80
        self.seeker_search_timer -= 1
        
        # SCANNING Check
        if self.seeker_search_timer % 40 < 10:
            self.seeker_mode = "SCANNING"
            self.seeker_target_pos = None 
        else:
            self.seeker_mode = "PATROLLING"
            self.seeker_target_pos = self.seeker_random_target
            
        return self.seeker_target_pos

    def _seeker_rule_based_policy(self):
        target_pos = self.seeker_target_pos
        
        sx_dof = self.model.jnt_dofadr[self.model.joint('s_x').id]
        s0_vel = np.linalg.norm(self.data.qvel[sx_dof : sx_dof+2])
        
        if self.current_step < PREP_STEPS:
            self.seeker_mode = "WAITING"
            return 0.0, 0.0

        if self.seeker_mode == "SCANNING":
            return 0.0, 1.0 

        if target_pos is None: 
            return 0.0, 0.0

        sp = self.data.xpos[self.s0_body][:2]
        s_rot = self.data.qpos[self.srot_adr]
        ta = np.arctan2(target_pos[1]-sp[1], target_pos[0]-sp[0])
        ad = (ta - s_rot + np.pi) % (2*np.pi) - np.pi
        
        sf0 = SEEKER_RB_THRUST
        sr0 = np.clip(ad * 6.0, -3.0, 3.0)
        if abs(ad) > SEEKER_RB_TURN_THRESH: sf0 *= 0.3
        
        if sf0 > 0.05 and s0_vel < 0.05: self.s0_stuck_timer += 5
        else: self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
        
        if self.s0_recovery_mode > 0:
            sf0, sr0 = -0.2, 1.5
            self.s0_recovery_mode -= 1
        elif self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 10
            self.s0_stuck_timer = 0
            
        return sf0, sr0

    def render(self, stats=None):
        if self.render_mode == "human":
            loop_start = time.time()
            if self.viewer is None: 
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.elevation, self.viewer.cam.distance = -60, 23.0
                self.viewer.cam.lookat[:] = [0, 0, 0] 
            
            for box, geom in [(self.box1_body, self.box1_geom_id), (self.box2_body, self.box2_geom_id)]:
                if self.locked_boxes[box]:
                    self.model.geom_rgba[geom][:] = [0.8, 0.1, 0.1, 1.0]
                elif (self.grasping[1] == box) or (self.grasping[2] == box):
                    self.model.geom_rgba[geom][:] = [0.1, 0.1, 0.9, 1.0]
                else:
                    self.model.geom_rgba[geom][:] = [0.6, 0.4, 0.2, 1.0]
            
            # ★修正(v94): 描画ロジックを完全Unroll化
            if stats and self.viewer.user_scn:
                ctx = self.viewer.user_scn
                ctx.ngeom = 0 # Reset

                def add_line(pos1, pos2, color):
                    if ctx.ngeom < ctx.maxgeom:
                        mujoco.mjv_connector(
                            ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=2.0, 
                            from_=pos1, to=pos2
                        )
                        ctx.geoms[ctx.ngeom].rgba = color
                        ctx.ngeom += 1

                def add_label(pos, text, color):
                    if ctx.ngeom < ctx.maxgeom:
                        mujoco.mjv_initGeom(
                            ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.array([0,0,0]), 
                            pos=pos, mat=np.eye(3).flatten(), rgba=color
                        )
                        ctx.geoms[ctx.ngeom].label = text
                        ctx.ngeom += 1

                targets_common = [
                    (self.box1_body, "Box1"), (self.box2_body, "Box2"), 
                    (self.ramp_body, "Ramp"), (self.s0_body, "Seeker"), 
                    (self.h1_body, "H1"), (self.h2_body, "H2")
                ]

                # --- 1. Hider 1 (Yellow) ---
                h1_pos = np.array(self.data.xpos[self.h1_body])
                h1_rot = self.data.qpos[self.h1rot_adr]
                vis_h1 = []
                for tid, name in targets_common:
                    if tid == self.h1_body: continue
                    if self._is_visible(h1_pos, h1_rot, self.data.xpos[tid], tid, exclude_body_id=self.h1_body):
                        disp_name = "Friend" if name == "H2" else name
                        vis_h1.append(disp_name)
                        add_line(h1_pos+np.array([0,0,0.5]), self.data.xpos[tid]+np.array([0,0,0.5]), np.array([1, 1, 0, 0.6]))
                
                txt_h1 = f"H1:[{','.join(vis_h1)}]" if vis_h1 else "H1:[]"
                add_label(h1_pos+np.array([0,0,1.2]), txt_h1, np.array([1, 1, 0, 1]))

                # --- 2. Hider 2 (Cyan) ---
                h2_pos = np.array(self.data.xpos[self.h2_body])
                h2_rot = self.data.qpos[self.h2rot_adr]
                vis_h2 = []
                for tid, name in targets_common:
                    if tid == self.h2_body: continue
                    if self._is_visible(h2_pos, h2_rot, self.data.xpos[tid], tid, exclude_body_id=self.h2_body):
                        disp_name = "Friend" if name == "H1" else name
                        vis_h2.append(disp_name)
                        add_line(h2_pos+np.array([0,0,0.5]), self.data.xpos[tid]+np.array([0,0,0.5]), np.array([0, 1, 1, 0.6]))

                txt_h2 = f"H2:[{','.join(vis_h2)}]" if vis_h2 else "H2:[]"
                add_label(h2_pos+np.array([0,0,1.2]), txt_h2, np.array([0, 1, 1, 1]))

                # --- 3. Seeker (Red) ---
                s_pos = np.array(self.data.xpos[self.s0_body])
                s_rot = self.data.qpos[self.srot_adr]
                for tid in [self.h1_body, self.h2_body]:
                    if self._is_visible(s_pos, s_rot, self.data.xpos[tid], tid, exclude_body_id=self.s0_body):
                        add_line(s_pos+np.array([0,0,0.5]), self.data.xpos[tid]+np.array([0,0,0.5]), np.array([1, 0, 0, 0.6]))

                c_dict = {"CHASING": [1,0,0,1], "SEARCHING": [1,0.5,0,1], "SCANNING": [0,1,1,1], "PATROLLING": [1,1,1,1], "WAITING": [0.5,0.5,0.5,1]}
                s_color = np.array(c_dict.get(self.seeker_mode, [1,1,1,1]))
                add_label(s_pos+np.array([0,0,1.2]), f"S:{self.seeker_mode}", s_color)

                ACTION_REPEAT = 0.08
                step_duration = 0.005 * ACTION_REPEAT  # 例: 0.08秒くらい

                process_time = time.time() - loop_start
                wait_time = step_duration - process_time
                if wait_time > 0:
                    time.sleep(wait_time)

            self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close()

def make_env(rank, seed=0, render_mode=None):
    def _init():
        env = HideAndSeekEnv(render_mode=render_mode)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class PlotUpdateCallback(BaseCallback):
    def __init__(self, plot_every=10): 
        super().__init__()
        self.plot_every = plot_every
        self.history = []
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.history.append(info["episode"]["r"])
                if len(self.history) % self.plot_every == 0: self._save()
        return True
    def _save(self):
        self.ax.clear(); self.ax.set_title(f"PPO (LSTM) Progress v15 - Ep: {len(self.history)}")
        y = np.array(self.history); self.ax.plot(y, alpha=0.3, color="blue", label="Raw Reward")
        if len(y) >= 10:
            ma = np.convolve(y, np.ones(10)/10, mode='valid')
            self.ax.plot(np.arange(9, len(y)), ma, color="red", linewidth=2, label="MA10")
        self.ax.legend(loc='upper left'); self.ax.set_xlabel("Episode"); self.ax.set_ylabel("Reward")
        self.fig.savefig(PLOT_PATH)

# ★修正(v79): RenderCallbackのエラー回避
class RenderCallback(BaseCallback):
    def __init__(self, render_every: int):
        super().__init__()
        self.render_every = render_every
    def _on_step(self) -> bool:
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0].unwrapped
            if env.episode_count % self.render_every == 0:
                env.render(stats={"Ep": env.episode_count})
                time.sleep(0.002)
        return True

'''
def main():
    print(f"--- Multi-Agent Training Stage: Target = {TRAIN_TARGET} ---")
    
    if TRAIN_MODE:
        # ★修正(v79): USE_VIEWER=True時はDummyVecEnv(直列・描画可)を使用するよう分岐
        if USE_VIEWER:
            env = DummyVecEnv([make_env(i, render_mode="human") for i in range(N_ENVS)])
        else:
            env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
        
        if os.path.exists(SAVE_PATH + ".zip"):
            print(f"Loading existing model: {SAVE_PATH}")
            model = RecurrentPPO.load(SAVE_PATH, env=env, device="cpu")
        else:
            print("Creating new model...")
            model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, 
                                 n_steps=2048, batch_size=128, ent_coef=ENT_COEF, 
                                 policy_kwargs=dict(lstm_hidden_size=LSTM_HIDDEN_SIZE))
        
        try:
            # ★修正(v72): RenderCallbackの追加と条件分岐
            callbacks = [PlotUpdateCallback()]
            if USE_VIEWER:
                callbacks.append(RenderCallback(RENDER_EVERY))
                
            model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=ENABLE_RICH_PROGRESS, callback=callbacks)
            model.save(SAVE_PATH)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            model.save(SAVE_PATH)
            print("Model saved.")
        finally:
            try:
                env.close()
                time.sleep(0.5) # ★修正(v79): 終了待機時間を復元
            except Exception:
                pass
        
        # ★修正(v61): 学習モードの場合はここで終了し、推論モードへ進まないようにする
        print("Training finished.")
        return

    print("--- Inference Stage ---")
    env = DummyVecEnv([lambda: Monitor(HideAndSeekEnv(render_mode="human"))])
    model = RecurrentPPO.load(SAVE_PATH)
    
    # 1ステップのシミュレーション時間 (秒)
    step_duration = 0.005 * ACTION_REPEAT

    try:  
        for ep in range(5):
            obs = env.reset()
            done = [False]
            states = None
            start = np.ones((1,), dtype=bool)
            total_r = 0
            
            real_env = env.envs[0].unwrapped

            while not done[0]:
                loop_start = time.time() # ループ開始時刻
                
                action, states = model.predict(obs, state=states, episode_start=start, deterministic=False)
                start[0] = False
                obs, rew, done, _ = env.step(action)
                total_r += rew[0]
                
                # 描画更新: statsを渡すことでデバッグ情報を表示
                is_locked = any(real_env.locked_boxes.values())
                
                real_env.render(stats={
                    "Ep": ep, 
                    "Step": real_env.current_step, 
                    # ★修正(v65): total_rew -> total_r
                    "Rew": f"{total_r:.1f}", 
                    "Grasp": "ON" if (real_env.grasping[1] or real_env.grasping[2]) else "OFF",
                    "Lock": "ON" if is_locked else "OFF"
                })
                
                # 実時間調整
                process_time = time.time() - loop_start
                wait_time = step_duration - process_time
                if wait_time > 0:
                    time.sleep(wait_time)
                
            print(f"Episode {ep}: Reward {total_r:.1f}")
    except KeyboardInterrupt:
        print("\nInference interrupted.")
    finally:
        env.close()
        print("Environment closed.")
'''
if __name__ == "__main__":
    main()