# hns_environment.py v1.4
# 演習第25回：実体ボディID特定（_btm優先）およびキャッシュアクセス・バグ修正版
# 
# 修正履歴:
# v1.2: 主要Geom (_btm) を保持するボディを優先的に探索。
# v1.3: AttributeError (vis_engine.cache) および NameError (inf) を修正。mj_step をサブステップ形式に変更。
# v1.4: _get_obsにて、Lidarの生データからエージェント自身のサイズ＋マージン(0.45)を引き、移動可能な自由空間として観測ベクトルに格納するよう修正。

import os, tempfile, numpy as np, mujoco, gymnasium as gym
from gymnasium import spaces
from visibility_engine import VisibilityEngine

try: from main18_optimization import XML_CONTENT
except: XML_CONTENT = ""

class TeamCosEnv(gym.Env):
    def __init__(self, layout_name="Maze"):
        super().__init__()
        fd, path = tempfile.mkstemp(suffix='.xml', text=True)
        with os.fdopen(fd, 'w') as f: f.write(XML_CONTENT)
        self.xml_path = path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data, layout_name=layout_name)
        self.agent_prefixes = ["h1", "h2", "s"]
        self.qpos_indices = {}
        self.body_ids = {} 
        self._setup_indices()
        self._randomize_all_objects()
        
        # 修正: inf -> np.inf
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _find_actual_body_with_btm(self, start_body_id):
        """指定されたボディ配下で '_btm' geomを保持しているボディIDを最優先で探索"""
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == start_body_id:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                if "_btm" in name: return start_body_id
        for b in range(self.model.nbody):
            if self.model.body_parentid[b] == start_body_id:
                res = self._find_actual_body_with_btm(b)
                if res != -1: return res
        return -1

    def _setup_indices(self):
        for p in self.agent_prefixes:
            try:
                jx = self.model.joint(f"{p}_x").id
                anchor_id = self.model.jnt_bodyid[jx]
                actual_id = self._find_actual_body_with_btm(anchor_id)
                self.body_ids[p] = actual_id if actual_id != -1 else anchor_id
                
                self.qpos_indices[p] = {
                    'x': jx, 
                    'y': self.model.joint(f"{p}_y").id, 
                    'z': self.model.joint(f"{p}_z").id, 
                    'rot': self.model.joint(f"{p}_rot").id
                }
            except: pass

    def _is_collision_free(self, x, y, r, placed_objects):
        margin = 0.2
        # 修正: 属性の存在をチェック
        c = getattr(self.vis_engine, 'cache', None)
        if c is not None:
            ix, iy = int((x + c["bounds"]) / c["cell_size"]), int((c["bounds"] - y) / c["cell_size"])
            if 0 <= ix < c["n_side"] and 0 <= iy < c["n_side"]:
                if c["sdf_map"][iy, ix] < (r + margin): return False
            else: return False
        for px, py, pr in placed_objects:
            if np.sqrt((x - px)**2 + (y - py)**2) < (r + pr + 0.4): return False
        return True

    def _randomize_all_objects(self):
        placed = []
        for obj in ["box1_joint", "box2_joint", "ramp_joint"]:
            try:
                q_adr = self.model.jnt_qposadr[self.model.joint(obj).id]
                radius = 1.2 if "ramp" in obj else 0.85
                for _ in range(300):
                    pos = np.random.uniform(-4.5, 4.5, 2)
                    if self._is_collision_free(pos[0], pos[1], radius, placed):
                        self.data.qpos[q_adr:q_adr+2], self.data.qpos[q_adr+2], self.data.qpos[q_adr+3:q_adr+7] = pos, 0.55, [1,0,0,0]
                        placed.append((pos[0], pos[1], radius)); break
            except: pass
        for p in self.agent_prefixes:
            if p in self.qpos_indices:
                for _ in range(300):
                    pos = np.random.uniform(-5.0, 5.0, 2)
                    if self._is_collision_free(pos[0], pos[1], 0.5, placed):
                        q = self.qpos_indices[p]
                        # 修正: Zジョイントに 0.0 を設定。
                        # XMLのオフセット(0.5 - 0.1)により、これで球体中心が正確に 0.4m となります。
                        self.data.qpos[self.model.jnt_qposadr[q['x']]] = pos[0]
                        self.data.qpos[self.model.jnt_qposadr[q['y']]] = pos[1]
                        self.data.qpos[self.model.jnt_qposadr[q['z']]] = 0.0
                        self.data.qpos[self.model.jnt_qposadr[q['rot']]] = np.random.uniform(-180, 180)
                        placed.append((pos[0], pos[1], 0.5)); break
        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self, agent_idx):
        obs = np.zeros(53, dtype=np.float32); d, m = self.data, self.model; p = self.agent_prefixes[agent_idx]
        if p not in self.body_ids: return obs
        b_id = self.body_ids[p]
        a_pos = d.xpos[b_id][:2]
        obs[0:2] = d.cvel[b_id][:2]
        rot_rad = np.deg2rad(d.qpos[m.jnt_qposadr[self.qpos_indices[p]['rot']]])
        obs[2], obs[3:5] = rot_rad, [np.cos(rot_rad), np.sin(rot_rad)]
        
        # Lidar計測（生データ）
        raw_lidar = self.vis_engine.cast_lidar(a_pos, heading=rot_rad, body_exclude=b_id)
        # 自身の半径(0.4) + 余裕マージン(0.05) を引き、0未満にならないよう制限
        safe_margin = 0.45
        free_space = np.maximum(0.0, raw_lidar - safe_margin)
        obs[5:17] = free_space
        
        targets = [0, 1] if agent_idx == 2 else [2, 1 if agent_idx == 0 else 0]
        for i, t_idx in enumerate(targets):
            t_p = self.agent_prefixes[t_idx]
            if t_p in self.body_ids:
                t_id = self.body_ids[t_p]
                rel = d.xpos[t_id][:2] - a_pos
                off = 40 if i == 0 else 45
                if self.vis_engine.is_visible(a_pos, d.xpos[t_id][:2], body_exclude=b_id):
                    obs[off:off+2], obs[off+4 if i==0 else off+6] = rel, 1.0
        return obs

    def step(self, action):
        self.data.ctrl[:6] = np.asarray(action).flatten()
        # 修正: サブステップをループで実行
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        return self._get_obs(2), 0.0, False, False, {}

    def __del__(self):
        if hasattr(self, 'xml_path') and os.path.exists(self.xml_path):
            try: os.remove(self.xml_path)
            except: pass