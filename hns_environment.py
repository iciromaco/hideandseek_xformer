# hns_environment.py v1.37
# 演習第25回：壁との衝突回避・完全配置（めり込み防止）版
# 
# 修正履歴:
# v1.36: 壁衝突判定。
# v1.37: ユーザー報告の「内壁との重なり」を完全に解消。
#        1. 各オブジェクト（エージェント, Box, Ramp）のサイズに応じたマージンを適用。
#        2. 配置候補地の AABB 判定を厳格化し、壁の角へのめり込みも防止。

import os, tempfile, numpy as np, mujoco, gymnasium as gym
from gymnasium import spaces
from visibility_engine import VisibilityEngine
import math

XML_CONTENT = """
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.005"/>
    <asset>
        <texture name="grid_tex" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid_mat" texture="grid_tex" texrepeat="1 1" reflectance="0.2"/>
    </asset>
    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="6 6 0.1" material="grid_mat" friction="1.0 0.05 0.0001"/>
        
        <!-- 外壁 -->
        <geom name="wall_n" type="box" size="6.2 0.1 4.0" pos="0 6.1 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="wall_s" type="box" size="6.2 0.1 4.0" pos="0 -6.1 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="wall_e" type="box" size="0.1 6 4.0" pos="6.1 0 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="wall_w" type="box" size="0.1 6 4.0" pos="-6.1 0 4.0" rgba="0.7 0.7 0.7 0.3"/>
        
        <!-- 内壁 (Group 0: 計測対象) -->
        <geom name="maze_w0" type="box" size="1.5 0.2 0.5" pos="3.0 1.5 0.5" rgba="0.0 0.7 0.7 1"/>
        <geom name="maze_w1" type="box" size="1.5 0.2 0.5" pos="-3.0 -1.5 0.5" rgba="0.0 0.7 0.7 1"/>
        <geom name="maze_w2" type="box" size="0.2 1.5 0.5" pos="0 -3.0 0.5" rgba="0.0 0.7 0.7 1"/>
        <geom name="maze_w3" type="box" size="0.2 1.5 0.5" pos="0 3.0 0.5" rgba="0.0 0.7 0.7 1"/>
        
        <body name="ramp_body" pos="0 0 0">
            <joint type="free" name="ramp_joint"/>
            <geom name="ramp_slope_surface" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.5"/>
            <geom name="ramp_base" type="box" size="0.6 0.5 0.5" pos="0.6 0 0.25" rgba="0 1 0 0.2" group="1"/>
        </body>
        
        <body name="box1_body" pos="2 -2 0.5"><joint name="box1_joint" type="free"/><geom name="box1_geom" type="box" size="0.6 0.6 0.5" rgba="0.6 0.4 0.2 1"/></body>
        <body name="box2_body" pos="-2 2 0.5"><joint name="box2_joint" type="free"/><geom name="box2_geom" type="box" size="0.6 0.6 0.5" rgba="0.7 0.5 0.3 1"/></body>
        
        <body name="seeker_anchor" pos="0 0 0.5">
            <joint name="s_x" type="slide" axis="1 0 0"/><joint name="s_y" type="slide" axis="0 1 0"/><joint name="s_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2"/><joint name="s_rot" type="hinge" axis="0 0 1" damping="50"/>
            <body name="seeker_body">
                <geom name="seeker_btm" type="sphere" size="0.4" pos="0 0 -0.1" rgba="0.9 0.1 0.1 1"/>
                <geom name="seeker_capsule" type="capsule" size="0.3 0.2" rgba="0.9 0.1 0.1 1" group="1"/>
            </body>
        </body>
        
        <body name="hider1_anchor" pos="0 0 0.5">
            <joint name="h1_x" type="slide" axis="1 0 0"/><joint name="h1_y" type="slide" axis="0 1 0"/><joint name="h1_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2"/><joint name="h1_rot" type="hinge" axis="0 0 1" damping="50"/>
            <body name="hider1_body">
                <geom name="hider1_btm" type="sphere" size="0.4" pos="0 0 -0.1" rgba="0.1 0.1 0.9 1"/>
                <geom name="hider1_capsule" type="capsule" size="0.3 0.2" rgba="0.1 0.1 0.9 1" group="1"/>
            </body>
        </body>
        
        <body name="hider2_anchor" pos="0 0 0.5">
            <joint name="h2_x" type="slide" axis="1 0 0"/><joint name="h2_y" type="slide" axis="0 1 0"/><joint name="h2_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2"/><joint name="h2_rot" type="hinge" axis="0 0 1" damping="50"/>
            <body name="hider2_body">
                <geom name="hider2_btm" type="sphere" size="0.4" pos="0 0 -0.1" rgba="0.1 0.6 0.9 1"/>
                <geom name="hider2_capsule" type="capsule" size="0.3 0.2" rgba="0.1 0.6 0.9 1" group="1"/>
            </body>
        </body>
    </worldbody>
</mujoco>
"""

class TeamCosEnv(gym.Env):
    def __init__(self, lidar_mode=1):
        super().__init__()
        self.lidar_mode = lidar_mode
        fd, path = tempfile.mkstemp(suffix='.xml', text=True)
        with os.fdopen(fd, 'w') as f: f.write(XML_CONTENT)
        self.xml_path = path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data)
        
        self.agent_prefixes = ["s", "h1", "h2"] 
        self.body_ids = {}; self.qpos_indices = {}
        self._setup_indices()
        
        # 内壁データ (cx, cy, sx, sy)
        self.maze_walls = [
            (3.0, 1.5, 1.5, 0.2), (-3.0, -1.5, 1.5, 0.2),
            (0.0, -3.0, 0.2, 1.5), (0.0, 3.0, 0.2, 1.5)
        ]
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def _setup_indices(self):
        m = self.model
        for p in self.agent_prefixes:
            try:
                b_name = "seeker_body" if p == "s" else f"hider{p[1]}_body"
                self.body_ids[p] = m.body(b_name).id
                self.qpos_indices[p] = {'x': m.joint(f"{p}_x").id, 'y': m.joint(f"{p}_y").id, 'z': m.joint(f"{p}_z").id, 'rot': m.joint(f"{p}_rot").id}
            except: pass

    def _get_obs(self, agent_idx):
        obs = np.zeros(53, dtype=np.float32); d, m = self.data, self.model; p = self.agent_prefixes[agent_idx]
        if p not in self.body_ids: return obs
        b_id = self.body_ids[p]; a_pos = d.xpos[b_id][:2]
        q_idx_vx = m.jnt_dofadr[self.qpos_indices[p]['x']]
        obs[0:2] = d.qvel[q_idx_vx : q_idx_vx+2]
        rot_rad = float(d.qpos[m.jnt_qposadr[self.qpos_indices[p]['rot']]])
        obs[2], obs[3], obs[4] = rot_rad, np.cos(rot_rad), np.sin(rot_rad)
        raw_lidar = self.vis_engine.cast_lidar(a_pos, heading=rot_rad, mode=self.lidar_mode, body_exclude=b_id)
        obs[5:17] = np.maximum(0.0, raw_lidar - 0.45)
        return obs

    def step(self, action):
        self.data.ctrl[:6] = action
        for _ in range(5): mujoco.mj_step(self.model, self.data)
        return self._get_obs(0), 0.0, False, False, {}

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self._randomize_all_objects()
        return self._get_obs(0), {}

    def _randomize_all_objects(self):
        m, d = self.model, self.data; mujoco.mj_resetData(m, d); placed = []
        # オブジェクトごとの占有半径
        radii = {"s": 0.45, "h1": 0.45, "h2": 0.45, "box1": 0.7, "box2": 0.7, "ramp": 0.9}
        targets = self.agent_prefixes + ["box1", "box2", "ramp"]
        np.random.shuffle(targets)
        
        for t in targets:
            r_obj = radii.get(t, 0.5)
            for _ in range(250):
                # フィールド端を考慮した範囲
                new_pos = np.random.uniform(-5.3, 5.3, size=2)
                
                # 1. 内壁との衝突チェック (AABB + オブジェクト半径)
                in_wall = False
                for (cx, cy, sx, sy) in self.maze_walls:
                    if abs(new_pos[0] - cx) < (sx + r_obj) and abs(new_pos[1] - cy) < (sy + r_obj):
                        in_wall = True; break
                if in_wall: continue
                
                # 2. 他の配置済みオブジェクトとの衝突
                if all(np.linalg.norm(new_pos - p) > (r_obj + 0.6) for p in placed):
                    if t in self.agent_prefixes:
                        q = self.qpos_indices[t]
                        d.qpos[m.jnt_qposadr[q['x']]] = new_pos[0]
                        d.qpos[m.jnt_qposadr[q['y']]] = new_pos[1]
                        d.qpos[m.jnt_qposadr[q['z']]] = 0.0
                        d.qpos[m.jnt_qposadr[q['rot']]] = np.random.uniform(-np.pi, np.pi)
                    else:
                        j_id = m.joint(f"{t}_joint").id; q_adr = m.jnt_qposadr[j_id]
                        d.qpos[q_adr : q_adr+2] = new_pos
                        d.qpos[q_adr+2] = 0.5
                        d.qpos[q_adr+3:q_adr+7] = [1.0, 0.0, 0.0, 0.0]
                    placed.append(new_pos); break
        mujoco.mj_forward(m, d)

    def __del__(self):
        if hasattr(self, 'xml_path') and os.path.exists(self.xml_path):
            try: os.remove(self.xml_path)
            except: pass