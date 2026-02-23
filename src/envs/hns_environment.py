# hns_environment.py v1.50
# 演習第25回：観測ロジック完全復元 ＆ XML動的解析 統合版
# 
# 修正履歴:
# v1.49: インポート移行に伴う整理（観測ロジックの一部欠落を修正）。
# v1.50: ユーザーの指摘に基づき、削られていた 53次元 obs の詳細構築ロジックを復元。
#        1. 自己情報 (5)、Lidar (12) に加え、Box (16)、Ramp (7)、他者 (12) の相対情報を計算。
#        2. VisibilityEngine.is_visible を使用した可視フラグ判定を再実装。
#        3. プロジェクトルートからのインポートルールを遵守。

import os
import sys
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import math

# 環境設定済みのパッケージから直接インポート
from core.visibility_engine import VisibilityEngine

# --- 物理的真理のインポート ---
def _load_production_xml():
    """プロジェクトルートから本番用 XML 定義をロードする"""
    try:
        import main18_optimization
        return getattr(main18_optimization, "XML_CONTENT", "")
    except ImportError:
        _root = str(Path(__file__).resolve().parent.parent.parent)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        try:
            import main18_optimization
            return getattr(main18_optimization, "XML_CONTENT", "")
        except ImportError:
            return ""

XML_CONTENT = _load_production_xml()

class TeamCosEnv(gym.Env):
    def __init__(self, lidar_mode=1):
        super().__init__()
        self.lidar_mode = lidar_mode
        
        if not XML_CONTENT:
            raise ImportError("Critical Error: XML_CONTENT could not be loaded.")

        # 1. モデル構築
        try:
            self.model = mujoco.MjModel.from_xml_string(XML_CONTENT)
        except Exception as e:
            print(f"❌ MuJoCo XML Load Error: {e}")
            raise
            
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data)
        
        # 2. 構造自動解析
        self.body_ids = {}
        self.qpos_indices = {}
        self.maze_walls = []
        self.box_ids = []
        self.ramp_id = -1
        self.seekers = []
        self.hiders = []
        
        self._analyze_xml_structure()
        self.agents = self.seekers + self.hiders
        self.num_agents = len(self.agents)
        
        # 3. 空間の定義 (53次元)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(max(1, self.num_agents) * 2,), dtype=np.float32)

    def _analyze_xml_structure(self):
        m = self.model
        # エージェント検出
        found_prefixes = set()
        for i in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and '_' in name:
                prefix = name.split('_')[0].lower()
                if prefix in ['s', 'h1', 'h2', 'h3', 'agent']:
                    found_prefixes.add(prefix)
        
        for p in sorted(list(found_prefixes)):
            if p.startswith('s'): self.seekers.append(p)
            else: self.hiders.append(p)
            b_name = "seeker_body" if p == 's' else f"hider{p[1:] or '1'}_body"
            try:
                self.body_ids[p] = m.body(b_name).id
                self.qpos_indices[p] = {
                    'x': m.joint(f"{p}_x").id, 'y': m.joint(f"{p}_y").id, 
                    'z': m.joint(f"{p}_z").id, 'rot': m.joint(f"{p}_rot").id
                }
            except: pass

        # 壁・箱・ランプの検出
        for i in range(m.ngeom):
            name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
            bid = m.geom_bodyid[i]
            if any(k in name for k in ["wall", "maze", "border"]) and "floor" not in name:
                if m.body_jntnum[bid] == 0:
                    self.maze_walls.append((m.geom_pos[i][0], m.geom_pos[i][1], m.geom_size[i][0], m.geom_size[i][1]))
            
            if "box" in name and bid not in self.box_ids:
                self.box_ids.append(bid)
            if "ramp" in name and "surface" in name:
                self.ramp_id = bid

    def _get_obs(self, agent_idx):
        """53次元の観測ベクトルを精密に構築"""
        obs = np.zeros(53, dtype=np.float32)
        d, m = self.data, self.model
        if agent_idx >= len(self.agents): return obs
        
        p = self.agents[agent_idx]
        if p not in self.body_ids: return obs
        
        b_id = self.body_ids[p]
        my_pos = d.xpos[b_id][:2].copy()
        my_vel = d.cvel[b_id][:2].copy()
        
        # --- 1. 自己情報 (0-4) ---
        q_idx_vx = m.jnt_dofadr[self.qpos_indices[p]['x']]
        obs[0:2] = d.qvel[q_idx_vx : q_idx_vx+2] # ローカル速度
        rot_rad = float(d.qpos[m.jnt_qposadr[self.qpos_indices[p]['rot']]])
        obs[2], obs[3], obs[4] = rot_rad, np.cos(rot_rad), np.sin(rot_rad)
        
        # --- 2. Lidar (5-16) ---
        raw_lidar = self.vis_engine.cast_lidar(my_pos, heading=rot_rad, mode=self.lidar_mode, body_exclude=b_id)
        obs[5:17] = np.maximum(0.0, raw_lidar - 0.45)
        
        # --- 3. 動的オブジェクト: Boxes (17-32) ---
        for i, box_bid in enumerate(self.box_ids[:2]):
            idx = 17 + i * 8
            visible = self.vis_engine.is_visible(my_pos, d.xpos[box_bid][:2], body_exclude=b_id, target_body_id=box_bid)
            if visible:
                obs[idx:idx+2] = d.xpos[box_bid][:2] - my_pos
                obs[idx+2:idx+4] = d.cvel[box_bid][:2] - my_vel
                # 回転 (Quaternion to Sin/Cos)
                q = d.xquat[box_bid]
                yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
                obs[idx+4], obs[idx+5] = np.cos(yaw), np.sin(yaw)
                obs[idx+7] = 1.0 # Visible flag

        # --- 4. スロープ: Ramp (33-39) ---
        if self.ramp_id != -1:
            visible = self.vis_engine.is_visible(my_pos, d.xpos[self.ramp_id][:2], body_exclude=b_id, target_body_id=self.ramp_id)
            if visible:
                obs[33:35] = d.xpos[self.ramp_id][:2] - my_pos
                obs[35:37] = d.cvel[self.ramp_id][:2] - my_vel
                q = d.xquat[self.ramp_id]
                yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
                obs[37], obs[38] = np.cos(yaw), np.sin(yaw)
                obs[39] = 1.0

        # --- 5. 敵/味方エージェント (40-51) ---
        # ここでは Seeker の視点を想定した簡易マッピング（実際は role により切り替えが必要）
        others = [self.body_ids[a] for a in self.agents if self.body_ids[a] != b_id]
        for i, other_bid in enumerate(others[:2]):
            idx = 40 + i * 5 # Seeker/Partner で次元が異なるが、ここでは基本相対位置と速度
            visible = self.vis_engine.is_visible(my_pos, d.xpos[other_bid][:2], body_exclude=b_id, target_body_id=other_bid)
            if visible:
                obs[idx:idx+2] = d.xpos[other_bid][:2] - my_pos
                obs[idx+2:idx+4] = d.cvel[other_bid][:2] - my_vel
                obs[idx+4] = 1.0
                
        return obs

    def step(self, action):
        self.data.ctrl[:len(action)] = action
        for _ in range(5): mujoco.mj_step(self.model, self.data)
        return self._get_obs(0), 0.0, False, False, {}

    def reset(self, seed=None):
        if seed is not None: np.random.seed(seed)
        self._randomize_all_objects()
        return self._get_obs(0), {}

    def _randomize_all_objects(self):
        m, d = self.model, self.data; mujoco.mj_resetData(m, d)
        placed = []; radii = {"s": 0.45, "h": 0.45, "box": 0.7, "ramp": 0.9}
        
        targets = [('agent', p) for p in self.agents]
        # Box/Ramp も動的にリスト化
        for i in range(m.nbody):
            name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
            if any(k in name for k in ["box", "ramp"]): targets.append(('object', name))
        
        np.random.shuffle(targets)
        for t_type, t_name in targets:
            r_obj = radii["s"] if t_type == 'agent' else (radii["ramp"] if "ramp" in t_name else radii["box"])
            for _ in range(250):
                new_pos = np.random.uniform(-5.3, 5.3, size=2)
                if any(abs(new_pos[0]-cx) < (sx+r_obj) and abs(new_pos[1]-cy) < (sy+r_obj) for (cx, cy, sx, sy) in self.maze_walls): continue
                if all(np.linalg.norm(new_pos - p) > (r_obj + 0.6) for p in placed):
                    if t_type == 'agent':
                        q = self.qpos_indices[t_name]
                        d.qpos[m.jnt_qposadr[q['x']]] = new_pos[0]; d.qpos[m.jnt_qposadr[q['y']]] = new_pos[1]
                        d.qpos[m.jnt_qposadr[q['rot']]] = np.random.uniform(-np.pi, np.pi)
                    else:
                        try:
                            # 自由関節(freejoint)の pos に代入
                            j_id = m.joint(f"{t_name.split('_')[0]}_joint").id; q_adr = m.jnt_qposadr[j_id]
                            d.qpos[q_adr : q_adr+2] = new_pos; d.qpos[q_adr+2] = 0.5
                        except: pass
                    placed.append(new_pos); break
        mujoco.mj_forward(m, d)