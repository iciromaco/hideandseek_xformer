# hns_environment.py
# 演習第25回：高速ハイブリッド判定エンジン統合版
# 物理干渉を排除し、命名規則を検証スクリプトと完全に一致させた修正版。

import os
import tempfile
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from visibility_engine import VisibilityEngine

# 命名規則を検証スクリプト (hider1_body等) と完全に一致させたXML
HNS_XML = """
<mujoco model="hideandseek">
    <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
    <option integrator="RK4" timestep="0.01"/>
    <worldbody>
        <light pos="0 0 10" dir="0 0 -1"/>
        <geom name="floor" pos="0 0 0" size="6 6 0.1" type="plane" rgba="0.8 0.8 0.8 1"/>
        
        <!-- 外壁 -->
        <geom name="wall_n" pos="0 6.1 0.5" size="6.1 0.1 0.5" type="box" rgba="0.5 0.5 0.5 1"/>
        <geom name="wall_s" pos="0 -6.1 0.5" size="6.1 0.1 0.5" type="box" rgba="0.5 0.5 0.5 1"/>
        <geom name="wall_e" pos="6.1 0 0.5" size="0.1 6.1 0.5" type="box" rgba="0.5 0.5 0.5 1"/>
        <geom name="wall_w" pos="-6.1 0 0.5" size="0.1 6.1 0.5" type="box" rgba="0.5 0.5 0.5 1"/>

        <!-- 内壁 (Maze) -->
        <geom name="maze_w1" pos="0 -3.0 0.5" size="3.0 0.1 0.5" type="box" rgba="0.4 0.4 0.4 1"/>
        <geom name="maze_w2" pos="0 1.5 0.5" size="0.1 1.5 0.5" type="box" rgba="0.4 0.4 0.4 1"/>
        <geom name="maze_w3" pos="0 4.5 0.5" size="3.0 0.1 0.5" type="box" rgba="0.4 0.4 0.4 1"/>

        <!-- 動的オブジェクト: 箱 -->
        <body name="box1_body" pos="2 2 0.5">
            <joint name="box1_joint" type="free"/>
            <geom name="box1_geom" size="0.6 0.6 0.5" type="box" rgba="0.8 0.5 0.2 1" mass="2.0"/>
        </body>
        <body name="box2_body" pos="-2 2 0.5">
            <joint name="box2_joint" type="free"/>
            <geom name="box2_geom" size="0.6 0.6 0.5" type="box" rgba="0.8 0.5 0.2 1" mass="2.0"/>
        </body>
        <!-- 動的オブジェクト: ランプ（厚みを考慮） -->
        <body name="ramp_body" pos="2 -2 0.5">
            <joint name="ramp_joint" type="free"/>
            <geom name="ramp_base" size="0.66 0.5 0.5" type="box" rgba="0.2 0.8 0.2 1" mass="5.0"/>
            <geom name="ramp_slope" pos="0 0 0.5" size="0.66 0.5 0.1" type="box" euler="30 0 0" rgba="0.2 0.7 0.2 1"/>
        </body>

        <!-- エージェント (検証用命名規則: hider1, hider2, seeker) -->
        <body name="hider1_body" pos="-2 -2 0.5">
            <joint name="hider1_x" type="slide" axis="1 0 0" damping="0.5"/>
            <joint name="hider1_y" type="slide" axis="0 1 0" damping="0.5"/>
            <joint name="hider1_z" type="slide" axis="0 0 1" damping="0.5"/>
            <joint name="hider1_rot" type="hinge" axis="0 0 1" damping="0.5"/>
            <geom name="hider1_geom" size="0.4 0.5" type="cylinder" rgba="0 1 0 1"/>
        </body>
        <body name="hider2_body" pos="-1 -2 0.5">
            <joint name="hider2_x" type="slide" axis="1 0 0" damping="0.5"/>
            <joint name="hider2_y" type="slide" axis="0 1 0" damping="0.5"/>
            <joint name="hider2_z" type="slide" axis="0 0 1" damping="0.5"/>
            <joint name="hider2_rot" type="hinge" axis="0 0 1" damping="0.5"/>
            <geom name="hider2_geom" size="0.4 0.5" type="cylinder" rgba="0 0.8 0.8 1"/>
        </body>
        <body name="seeker_body" pos="0 0 0.5">
            <joint name="seeker_x" type="slide" axis="1 0 0" damping="0.5"/>
            <joint name="seeker_y" type="slide" axis="0 1 0" damping="0.5"/>
            <joint name="seeker_z" type="slide" axis="0 0 1" damping="0.5"/>
            <joint name="seeker_rot" type="hinge" axis="0 0 1" damping="0.5"/>
            <geom name="seeker_geom" size="0.4 0.5" type="cylinder" rgba="1 0 0 1"/>
        </body>
    </worldbody>
    <actuator>
        <motor joint="hider1_x" gear="150"/> <motor joint="hider1_y" gear="150"/> <motor joint="hider1_rot" gear="100"/>
        <motor joint="hider2_x" gear="150"/> <motor joint="hider2_y" gear="150"/> <motor joint="hider2_rot" gear="100"/>
        <motor joint="seeker_x" gear="150"/> <motor joint="seeker_y" gear="150"/> <motor joint="seeker_rot" gear="100"/>
    </actuator>
</mujoco>
"""

class TeamCosEnv(gym.Env):
    def __init__(self, layout_name="Maze"):
        super().__init__()
        xml_path = os.path.join(os.path.dirname(__file__), "hideandseek.xml")
        if not os.path.exists(xml_path):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
                f.write(HNS_XML)
                xml_path = f.name
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data, layout_name=layout_name)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.agent_prefixes = ["hider1", "hider2", "seeker"]
        self.agent_body_ids = [self.model.body(f"{p}_body").id for p in self.agent_prefixes]
        self.box_body_ids = [self.model.body(f"box{i}_body").id for i in [1, 2]]
        self.ramp_body_id = self.model.body("ramp_body").id

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # 初期配置
        for prefix in self.agent_prefixes:
            pos = np.random.uniform(-4.0, 4.0, 2)
            rot = np.random.uniform(-np.pi, np.pi)
            try:
                self.data.qpos[self.model.jnt_qposadr[self.model.joint(f"{prefix}_x").id]] = pos[0]
                self.data.qpos[self.model.jnt_qposadr[self.model.joint(f"{prefix}_y").id]] = pos[1]
                self.data.qpos[self.model.jnt_qposadr[self.model.joint(f"{prefix}_z").id]] = 0.5
                self.data.qpos[self.model.jnt_qposadr[self.model.joint(f"{prefix}_rot").id]] = rot
            except: pass
            
        for name in ["box1", "box2", "ramp"]:
            try:
                q_adr = self.model.jnt_qposadr[self.model.joint(f"{name}_joint").id]
                self.data.qpos[q_adr:q_adr+2] = np.random.uniform(-4.0, 4.0, 2)
                self.data.qpos[q_adr+2] = 0.5
            except: pass
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(0), {}

    def _get_obs(self, agent_idx):
        obs = np.zeros(53, dtype=np.float32)
        d = self.data; m = self.model
        prefix = self.agent_prefixes[agent_idx]
        a_id = self.agent_body_ids[agent_idx]
        
        a_pos = d.xpos[a_id][:2]
        a_rot = d.qpos[m.jnt_qposadr[m.joint(f"{prefix}_rot").id]]
        
        obs[0:2] = d.cvel[a_id][:2]
        obs[2] = a_rot
        obs[3:5] = [np.cos(a_rot), np.sin(a_rot)]
        
        # Lidar計測 (Headingを考慮)
        obs[5:17] = self.vis_engine.cast_lidar(a_pos, heading=a_rot, body_exclude=a_id)
        
        # オブジェクト相対情報
        for i, b_id in enumerate(self.box_body_ids):
            off = 17 + i*8
            b_pos = d.xpos[b_id][:2]
            if self.vis_engine.is_visible(a_pos, b_pos, body_exclude=a_id):
                obs[off:off+2] = b_pos - a_pos
                obs[off+7] = 1.0
        
        r_pos = d.xpos[self.ramp_body_id][:2]
        if self.vis_engine.is_visible(a_pos, r_pos, body_exclude=a_id):
            obs[33:35] = r_pos - a_pos
            obs[39] = 1.0
            
        s_pos = d.xpos[self.agent_body_ids[2]][:2]
        if self.vis_engine.is_visible(a_pos, s_pos, body_exclude=a_id):
            obs[40:42] = s_pos - a_pos
            obs[44] = 1.0
            
        return obs

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, 5)
        return self._get_obs(0), 0.0, False, False, {}