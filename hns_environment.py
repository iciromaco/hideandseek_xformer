# hns_environment.py v1.22
# 演習第25回：XML記述順（Seeker先頭）完全同期版
# 
# 修正履歴:
# v1.21: 壁吸着解消。
# v1.22: ユーザーの指摘に基づき、エージェントのインデックス順をXMLの記述順
#        [Seeker, Hider1, Hider2] に完全に同期。
#        これにより制御入力の「ねじれ」を解消し、意図通りの駆動を実現。

import os, tempfile, numpy as np, mujoco, gymnasium as gym
from gymnasium import spaces
from visibility_engine import VisibilityEngine
import math

try:
    from main18_optimization import XML_CONTENT
except ImportError:
    XML_CONTENT = ""

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
        
        # 【重要】XMLの <actuator> 記述順に合わせる
        self.agent_prefixes = ["s", "h1", "h2"] 
        
        self.obj_prefixes = ["box1", "box2", "ramp"]
        self.qpos_indices = {}
        self.body_ids = {} 
        self.stuck_counts = {p: 0 for p in self.agent_prefixes}
        self.nudge_cooldown = {p: 0 for p in self.agent_prefixes}
        self._setup_indices()
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

    def _setup_indices(self):
        for p in self.agent_prefixes:
            try:
                jx = self.model.joint(f"{p}_x").id
                self.body_ids[p] = self.model.jnt_bodyid[jx]
                self.qpos_indices[p] = {
                    'x': jx, 'y': self.model.joint(f"{p}_y").id, 
                    'z': self.model.joint(f"{p}_z").id, 'rot': self.model.joint(f"{p}_rot").id
                }
            except: pass
        for o in self.obj_prefixes:
            try:
                j_id = self.model.joint(f"{o}_joint").id
                self.qpos_indices[o] = self.model.jnt_qposadr[j_id]
            except: pass

    def _is_in_wall(self, pos):
        walls = [(3.0, 1.5, 0.2, 1.6), (-3.0, -1.5, 0.2, 1.6), (1.5, -3.0, 1.6, 0.2), (-1.5, 3.0, 1.6, 0.2)]
        margin = 0.55
        for (cx, cy, sx, sy) in walls:
            if abs(pos[0] - cx) < (sx + margin) and abs(pos[1] - cy) < (sy + margin): return True
        return False

    def _randomize_all_objects(self):
        mujoco.mj_resetData(self.model, self.data)
        placed_positions = []
        targets = self.agent_prefixes + self.obj_prefixes
        np.random.shuffle(targets)
        for t in targets:
            success = False
            for _ in range(100):
                new_pos = np.random.uniform(-5.0, 5.0, size=2)
                if all(np.linalg.norm(new_pos - p) > 1.4 for p in placed_positions):
                    if not self._is_in_wall(new_pos):
                        if t in self.agent_prefixes:
                            q = self.qpos_indices[t]
                            self.data.qpos[self.model.jnt_qposadr[q['x']]] = new_pos[0]
                            self.data.qpos[self.model.jnt_qposadr[q['y']]] = new_pos[1]
                            self.data.qpos[self.model.jnt_qposadr[q['z']]] = 0.0
                            self.data.qpos[self.model.jnt_qposadr[q['rot']]] = np.random.uniform(-np.pi, np.pi)
                        else:
                            q_adr = self.qpos_indices[t]
                            self.data.qpos[q_adr : q_adr+2] = new_pos
                            self.data.qpos[q_adr+2] = 0.5
                            yaw = np.random.uniform(-np.pi, np.pi)
                            self.data.qpos[q_adr+3:q_adr+7] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                        placed_positions.append(new_pos); success = True; break
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self, agent_idx):
        obs = np.zeros(53, dtype=np.float32); d, m = self.data, self.model; p = self.agent_prefixes[agent_idx]
        if p not in self.body_ids: return obs
        b_id = self.body_ids[p]; a_pos = d.xpos[b_id][:2]
        obs[0:2] = d.cvel[b_id][:2]
        rot_rad = float(d.qpos[m.jnt_qposadr[self.qpos_indices[p]['rot']]])
        obs[2], obs[3], obs[4] = rot_rad, np.cos(rot_rad), np.sin(rot_rad)
        raw_lidar = self.vis_engine.cast_lidar(a_pos, heading=rot_rad, mode=self.lidar_mode, body_exclude=b_id)
        obs[5:17] = np.maximum(0.0, raw_lidar - 0.45)
        return obs

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self._randomize_all_objects()
        for p in self.agent_prefixes: self.stuck_counts[p] = 0; self.nudge_cooldown[p] = 0
        # XML順に合わせ、Seeker(index 0)の観測を返す
        return self._get_obs(0), {}

    def step(self, action):
        # アクション [S_T, S_R, H1_T, H1_R, H2_T, H2_R] をそのまま ctrl に適用
        self.data.ctrl[:6] = action
        for _ in range(5): mujoco.mj_step(self.model, self.data)
        
        # --- Nudge ロジック ---
        for p_idx, p in enumerate(self.agent_prefixes):
            if self.nudge_cooldown[p] > 0: self.nudge_cooldown[p] -= 1; continue
            b_id = self.body_ids[p]; vel = np.linalg.norm(self.data.cvel[b_id][:2])
            p_ctrl = self.data.ctrl[p_idx*2 : p_idx*2+2]
            if vel < 0.015 and np.linalg.norm(p_ctrl) > 0.1: self.stuck_counts[p] += 1
            else: self.stuck_counts[p] = 0
            
            if self.stuck_counts[p] >= 15: 
                obs = self._get_obs(p_idx); lidar = obs[5:17]
                angles = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
                repel_vec = np.zeros(2); found_near = False
                for i in range(12):
                    if lidar[i] < 0.15:
                        found_near = True; wall_ang = obs[2] + np.deg2rad(angles[i])
                        repel_vec[0] -= np.cos(wall_ang) * (1.0/(lidar[i]+0.02))
                        repel_vec[1] -= np.sin(wall_ang) * (1.0/(lidar[i]+0.02))
                if found_near:
                    norm = np.linalg.norm(repel_vec)
                    if norm > 1e-5:
                        push_dir = repel_vec / norm; qx, qy, qz, qr = self.qpos_indices[p]['x'], self.qpos_indices[p]['y'], self.qpos_indices[p]['z'], self.qpos_indices[p]['rot']
                        self.data.qpos[self.model.jnt_qposadr[qx]] += push_dir[0] * 0.10
                        self.data.qpos[self.model.jnt_qposadr[qy]] += push_dir[1] * 0.10
                        self.data.qpos[self.model.jnt_qposadr[qz]] = 0.0
                        self.data.ctrl[p_idx*2 : p_idx*2+2] = 0.0
                        dof_x, dof_y, dof_z, dof_r = self.model.jnt_dofadr[qx], self.model.jnt_dofadr[qy], self.model.jnt_dofadr[qz], self.model.jnt_dofadr[qr]
                        self.data.qvel[dof_x] = self.data.qvel[dof_y] = self.data.qvel[dof_z] = self.data.qvel[dof_r] = 0.0
                        self.data.qacc[dof_x] = self.data.qacc[dof_y] = self.data.qacc[dof_z] = self.data.qacc[dof_r] = 0.0
                        self.stuck_counts[p] = 0; self.nudge_cooldown[p] = 30
                        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(0), 0.0, False, False, {}

    def __del__(self):
        if hasattr(self, 'xml_path') and os.path.exists(self.xml_path):
            try: os.remove(self.xml_path)
            except: pass