# hns_environment.py
# 演習第25回：本番環境完全同期・自動キャッシュ生成・マルチエージェント対応

import os
import tempfile
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from visibility_engine import VisibilityEngine

# 本番環境 main18_optimization.py から XML をそのまま引き継ぐ
try:
    from main18_optimization import XML_CONTENT
except ImportError:
    print("⚠️ Error: main18_optimization.py が見つかりません。")
    XML_CONTENT = ""

class TeamCosEnv(gym.Env):
    def __init__(self, layout_name="Maze"):
        super().__init__()
        fd, path = tempfile.mkstemp(suffix='.xml', text=True)
        with os.fdopen(fd, 'w') as f: f.write(XML_CONTENT)
        self.xml_path = path
        
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 幾何エンジンの初期化 (内部でキャッシュ不在を検知し自動生成する)
        self.vis_engine = VisibilityEngine(self.model, self.data, layout_name=layout_name)
        
        self.agent_prefixes = ["h1", "h2", "s"]
        self.qpos_indices = {}
        self.body_ids = {}
        self._setup_indices()
        self._randomize_all_objects()
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

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

    def _randomize_all_objects(self):
        """SDF を利用して非衝突な初期配置を行う"""
        placed = []
        c = self.vis_engine.cache
        
        def safe(x, y, r):
            if c:
                ix, iy = int((x + c["bounds"])/c["cell_size"]), int((c["bounds"] - y)/c["cell_size"])
                if not (0 <= ix < c["n_side"] and 0 <= iy < c["n_side"]) or c["sdf_map"][iy, ix] < r + 0.15: return False
            for px, py, pr in placed:
                if np.sqrt((x-px)**2 + (y-py)**2) < r + pr + 0.4: return False
            return True

        # 障害物配置
        for obj in ["box1_joint", "box2_joint", "ramp_joint"]:
            try:
                q_adr = self.model.jnt_qposadr[self.model.joint(obj).id]
                rad = 1.1 if "ramp" in obj else 0.8
                for _ in range(200):
                    pos = np.random.uniform(-4.5, 4.5, 2)
                    if safe(pos[0], pos[1], rad):
                        self.data.qpos[q_adr:q_adr+2], self.data.qpos[q_adr+2] = pos, 0.5
                        self.data.qpos[q_adr+3:q_adr+7] = [1,0,0,0]
                        placed.append((pos[0], pos[1], rad)); break
            except: pass

        # エージェント配置
        for p in self.agent_prefixes:
            if p in self.qpos_indices:
                for _ in range(200):
                    pos = np.random.uniform(-5.0, 5.0, 2)
                    if safe(pos[0], pos[1], 0.5):
                        q = self.qpos_indices[p]
                        self.data.qpos[self.model.jnt_qposadr[q['x']]] = pos[0]
                        self.data.qpos[self.model.jnt_qposadr[q['y']]] = pos[1]
                        self.data.qpos[self.model.jnt_qposadr[q['z']]] = 0.4
                        self.data.qpos[self.model.jnt_qposadr[q['rot']]] = np.random.uniform(-180, 180)
                        placed.append((pos[0], pos[1], 0.5)); break
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self, agent_idx):
        obs = np.zeros(53, dtype=np.float32)
        d, m = self.data, self.model
        p = self.agent_prefixes[agent_idx]
        if p not in self.body_ids: return obs
        b_id = self.body_ids[p]
        obs[0:2] = d.cvel[b_id][:2]
        rot_deg = d.qpos[m.jnt_qposadr[self.qpos_indices[p]['rot']]]
        rot_rad = np.deg2rad(rot_deg)
        obs[2], obs[3:5] = rot_rad, [np.cos(rot_rad), np.sin(rot_rad)]
        obs[5:17] = self.vis_engine.cast_lidar(d.xpos[b_id][:2], heading=rot_rad, body_exclude=b_id)
        print(b_id, d.xpos[b_id][:2], rot_deg, obs[5:17])
        # ターゲット可視性などは以前のロジックと同様
        return obs

    def step(self, action):
        act = np.asarray(action).flatten()
        if act.size == 6: self.data.ctrl[:6] = act
        elif act.size == 2: self.data.ctrl[0:2] = act # Seeker
        mujoco.mj_step(self.model, self.data, 5)
        return self._get_obs(2), 0.0, False, False, {}

    def __del__(self):
        if hasattr(self, 'xml_path') and os.path.exists(self.xml_path):
            try: os.remove(self.xml_path)
            except: pass