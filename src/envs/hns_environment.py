# hns_environment.py v1.48
# 演習第25回：XML構造完全自動解析 ＆ インポート・パス解決強化版
# 
# 修正履歴:
# v1.47: 物理構造自動解析。
# v1.48: KeyError: 's' を根絶。
#        1. main18_optimization.py のインポートをルートディレクトリから確実に実行。
#        2. XML内のジョイント名から接頭語 (s, h1, h2...) を自動検出し正規化。
#        3. 内壁 (maze_walls) を geom 定義から全自動で抽出 (pos, size を取得)。
#        4. src. を付けないインポートルールを遵守。

import os
import sys
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import math
from pathlib import Path

# 環境設定済みのパッケージから直接インポート (src.なし)
from core.visibility_engine import VisibilityEngine

# --- XML_CONTENT のインポート解決 (最優先) ---
def _load_xml_content():
    """プロジェクトルートから XML_CONTENT を安全にロードする"""
    try:
        import main18_optimization
        return getattr(main18_optimization, "XML_CONTENT", "")
    except ImportError:
        # このファイル (src/envs/hns_environment.py) から3つ上がルート
        _root = str(Path(__file__).resolve().parent.parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        try:
            import main18_optimization
            return getattr(main18_optimization, "XML_CONTENT", "")
        except ImportError:
            return ""

XML_CONTENT = _load_xml_content()

class TeamCosEnv(gym.Env):
    def __init__(self, lidar_mode=1):
        super().__init__()
        self.lidar_mode = lidar_mode
        
        # 1. モデルの構築
        if not XML_CONTENT or len(XML_CONTENT) < 100:
            print("⚠️  Warning: XML_CONTENT could not be loaded from main18_optimization.")
            # 最小限の動作保証用
            self._xml = "<mujoco><worldbody><light pos='0 0 2'/><geom type='plane' size='5 5 .1' name='floor'/></worldbody></mujoco>"
        else:
            self._xml = XML_CONTENT

        try:
            self.model = mujoco.MjModel.from_xml_string(self._xml)
        except Exception as e:
            print(f"❌ Critical XML Error: {e}")
            raise
            
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data)
        
        # 2. 自動解析用メンバ
        self.body_ids = {}
        self.qpos_indices = {}
        self.maze_walls = []
        self.seekers = []
        self.hiders = []
        
        self._analyze_xml_structure()
        self.agents = self.seekers + self.hiders
        self.num_agents = len(self.agents)
        
        # 3. 空間の定義
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(max(1, self.num_agents) * 2,), dtype=np.float32)

    def _analyze_xml_structure(self):
        """XMLからエージェントと内壁の情報を動的に抽出する"""
        m = self.model
        
        # --- A. エージェント (Joint名から Prefix を特定) ---
        found_prefixes = set()
        for i in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and '_' in name:
                prefix = name.split('_')[0].lower()
                if prefix in ['s', 'h1', 'h2', 'h3', 'agent']:
                    found_prefixes.add(prefix)
        
        for p in sorted(list(found_prefixes)):
            # チーム分け
            if p.startswith('s'):
                self.seekers.append(p)
            else:
                self.hiders.append(p)
            
            # Body ID 紐付け (main18命名規則優先)
            b_name = "seeker_body" if p == 's' else f"hider{p[1:] or '1'}_body"
            try:
                self.body_ids[p] = m.body(b_name).id
                self.qpos_indices[p] = {
                    'x': m.joint(f"{p}_x").id, 
                    'y': m.joint(f"{p}_y").id, 
                    'z': m.joint(f"{p}_z").id, 
                    'rot': m.joint(f"{p}_rot").id
                }
            except:
                pass

        # --- B. 内壁 (Geomから自動抽出) ---
        for i in range(m.ngeom):
            name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
            bid = m.geom_bodyid[i]
            
            # 条件: 名前が床ではなく、かつジョイントを持たない静的ボディに属している
            is_floor = any(k in name for k in ["floor", "ground", "plane"])
            if not is_floor and m.body_jntnum[bid] == 0:
                pos = m.geom_pos[i]
                size = m.geom_size[i]
                # (cx, cy, sx, sy)
                self.maze_walls.append((pos[0], pos[1], size[0], size[1]))

        print(f"✅ XML Analysis Complete:")
        print(f"   - Seekers: {self.seekers} | Hiders: {self.hiders}")
        print(f"   - Detected Walls: {len(self.maze_walls)} objects")

    def _get_obs(self, agent_idx):
        obs = np.zeros(53, dtype=np.float32)
        d, m = self.data, self.model
        if not self.agents or agent_idx >= len(self.agents): return obs
        
        p = self.agents[agent_idx]
        if p not in self.body_ids or p not in self.qpos_indices: return obs
        
        b_id = self.body_ids[p]
        a_pos = d.xpos[b_id][:2]
        
        # 速度
        q_idx_vx = m.jnt_dofadr[self.qpos_indices[p]['x']]
        obs[0:2] = d.qvel[q_idx_vx : q_idx_vx+2]
        
        # 回転
        rot_rad = float(d.qpos[m.jnt_qposadr[self.qpos_indices[p]['rot']]])
        obs[2], obs[3], obs[4] = rot_rad, np.cos(rot_rad), np.sin(rot_rad)
        
        # Lidar
        raw_lidar = self.vis_engine.cast_lidar(a_pos, heading=rot_rad, mode=self.lidar_mode, body_exclude=b_id)
        obs[5:17] = np.maximum(0.0, raw_lidar - 0.45)
        return obs

    def step(self, action):
        if self.num_agents > 0:
            self.data.ctrl[:len(action)] = action
        for _ in range(5): mujoco.mj_step(self.model, self.data)
        return self._get_obs(0), 0.0, False, False, {}

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self._randomize_all_objects()
        return self._get_obs(0), {}

    def _randomize_all_objects(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.num_agents == 0: return
        
        placed = []
        radii = {"s": 0.45, "h": 0.45, "box": 0.7, "ramp": 0.9}
        targets = [('agent', p) for p in self.agents]
        for i in range(m.nbody):
            name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
            if any(k in name for k in ["box", "ramp"]):
                targets.append(('object', name))
        
        np.random.shuffle(targets)
        for t_type, t_name in targets:
            r_obj = radii["s"] if t_type == 'agent' else (radii["ramp"] if "ramp" in t_name else radii["box"])
            for _ in range(250):
                new_pos = np.random.uniform(-5.3, 5.3, size=2)
                in_wall = False
                for (cx, cy, sx, sy) in self.maze_walls:
                    if abs(new_pos[0] - cx) < (sx + r_obj) and abs(new_pos[1] - cy) < (sy + r_obj):
                        in_wall = True; break
                if in_wall: continue
                if all(np.linalg.norm(new_pos - p) > (r_obj + 0.6) for p in placed):
                    if t_type == 'agent':
                        q = self.qpos_indices[t_name]
                        d.qpos[m.jnt_qposadr[q['x']]] = new_pos[0]
                        d.qpos[m.jnt_qposadr[q['y']]] = new_pos[1]
                        d.qpos[m.jnt_qposadr[q['rot']]] = np.random.uniform(-np.pi, np.pi)
                    else:
                        try:
                            j_id = m.joint(f"{t_name.split('_')[0]}_joint").id
                            q_adr = m.jnt_qposadr[j_id]
                            d.qpos[q_adr : q_adr+2] = new_pos
                            d.qpos[q_adr+2] = 0.5
                        except: pass
                    placed.append(new_pos); break
        mujoco.mj_forward(m, d)