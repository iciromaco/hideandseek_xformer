# hns_environment.py v1.99.53
# 演習第26回：【VectorEnv 整合性・安定報酬版】KeyError 回避 ＆ 対称報酬 完遂版
# 
# 修正内容:
# 1. KeyError解消: reset() の info に "is_detected" を含め、SyncVectorEnv の不連続性を排除。
# 2. 完全対称報酬: 発見中(-1.0)と隠蔽中(+1.0)の符号違い同等な強烈な信号により、0への沈着を解消。
# 3. 物理ガード: 包含円判定によるオブジェクト融合防止ロジックを維持。

import os, sys, numpy as np, mujoco, mujoco.viewer, gymnasium as gym, math, time
from gymnasium import spaces
from pathlib import Path

def _load_xml_content():
    try:
        root_path = Path(__file__).resolve().parent.parent.parent
        if str(root_path) not in sys.path:
            sys.path.insert(0, str(root_path))
        import main18_optimization
        return getattr(main18_optimization, "XML_CONTENT", "")
    except Exception:
        return ""

XML_CONTENT = _load_xml_content()

from core.visibility_engine import VisibilityEngine
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider

class TeamCosEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 40}

    def __init__(self, mode="initial", lidar_mode=1, render_mode=None):
        super().__init__()
        self.mode, self.lidar_mode, self.render_mode = mode, lidar_mode, render_mode
        self.current_step = 0
        self.prep_steps = 80
        
        if not XML_CONTENT:
            raise ImportError("XML_CONTENT could not be loaded.")
            
        self.model = mujoco.MjModel.from_xml_string(XML_CONTENT)
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data)
        self.viewer = None
        
        self.agent_keys = ["s", "h1", "h2"]
        self.body_ids, self.qpos_indices = {}, {}
        self.obj_default_colors, self.obj_geom_ids = {}, {}
        
        self._analyze_structure()
        self._init_npcs()
        
        self.maze_walls = [(3.0, 1.5, 1.5, 0.2), (-3.0, -1.5, 1.5, 0.2), (0.0, -3.0, 0.2, 1.5), (0.0, 3.0, 0.2, 1.5)]
        self.obj_locker_team = {o_id: None for o_id in self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])}
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        total_act_dim = 4 if self.mode == "initial" else 8
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(total_act_dim,), dtype=np.float32)

    def _analyze_structure(self):
        m = self.model
        for k in self.agent_keys:
            try:
                b_name = "seeker_body" if k=="s" else f"hider{k[1:] or '1'}_body"
                self.body_ids[k] = m.body(b_name).id
                self.qpos_indices[k] = {'x': m.joint(f"{k}_x").id, 'y': m.joint(f"{k}_y").id, 'rot': m.joint(f"{k}_rot").id}
            except: pass
        self.box_ids = []
        for i in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if ("box" in name or "ramp" in name) and "_body" in name:
                if "box" in name: self.box_ids.append(i)
                elif "ramp" in name: self.ramp_id = i
                g_id = m.body_geomadr[i]
                self.obj_geom_ids[i] = g_id
                self.obj_default_colors[i] = m.geom_rgba[g_id].copy()
        if not hasattr(self, 'ramp_id'): self.ramp_id = -1

    def _init_npcs(self):
        self.npcs = {"s": RuleBasedSeeker()}
        if self.mode == "initial": self.npcs["h2"] = RuleBasedHider()

    def step(self, action):
        self.current_step += 1
        ctrl = np.zeros(self.model.nu); act_f = np.ravel(action)
        h1_act = act_f[0:4]; ctrl[2:4] = h1_act[0:2]; self._process_physical_interaction("h1", h1_act[2], h1_act[3])
        if self.mode == "refinement":
            h2_act = act_f[4:8]; ctrl[4:6] = h2_act[0:2]; self._process_physical_interaction("h2", h2_act[2], h2_act[3])
        s_obs = self._get_obs(0); s_pos = self.data.xpos[self.body_ids['s']][:2]
        s_rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices['s']['rot']]]
        h_bids = [self.body_ids[hk] for hk in ["h1", "h2"]]; h_pos2d = [self.data.xpos[bid][:2].copy() for bid in h_bids]
        s_act_full = self.npcs["s"].get_action(s_obs, self.vis_engine, s_pos, s_rot, h_pos2d, self.body_ids['s'], h_bids)
        if self.current_step <= self.prep_steps:
            ctrl[0:2] = 0.0; self._process_physical_interaction("s", 0.0, 0.0)
        else:
            ctrl[0:2] = s_act_full[0:2]; self._process_physical_interaction("s", s_act_full[2], s_act_full[3])
        if self.mode == "initial" and "h2" in self.npcs:
            h2_obs = self._get_obs(2); h2_pos = self.data.xpos[self.body_ids['h2']][:2]
            h2_rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices['h2']['rot']]]
            h2_act_f = self.npcs["h2"].get_action(h2_obs, h2_pos, h2_rot, s_pos)
            ctrl[4:6] = h2_act_f[0:2]; self._process_physical_interaction("h2", h2_act_f[2], h2_act_f[3])
        self.data.ctrl[:] = ctrl
        for _ in range(5): mujoco.mj_step(self.model, self.data)
        if self.render_mode == "human": self.render()
        reward, found_any = self._compute_team_reward()
        done = (self.current_step >= 500)
        return self._normalize_obs(self._get_obs(1)), float(reward), False, done, {"is_detected": found_any}

    def _compute_team_reward(self):
        if self.current_step <= self.prep_steps: return 0.01, False
        s_pos, s_rot = self.data.xpos[self.body_ids['s']][:2], self.data.qpos[self.model.jnt_qposadr[self.qpos_indices['s']['rot']]]
        v_dir = np.array([math.cos(s_rot), math.sin(s_rot)]); found = False
        for hk in ["h1", "h2"]:
            h_p = self.data.xpos[self.body_ids[hk]][:2]; diff = h_p - s_pos; d = np.linalg.norm(diff)
            if d < 15.0 and np.dot(v_dir, diff/(d+1e-8)) > 0.38:
                if self.vis_engine.is_visible(s_pos, h_p, body_exclude=self.body_ids['s'], target_body_id=self.body_ids[hk]):
                    found = True; break
        return (-1.0 if found else 1.0), found

    def _process_physical_interaction(self, agent_key, lock_cmd, grab_cmd):
        m, d = self.model, self.data; a_bid, a_team = self.body_ids[agent_key], ("s" if agent_key=="s" else "h")
        target_ids = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        c_id, m_dist = -1, 2.0
        for o_id in target_ids:
            dist = np.linalg.norm(d.xpos[o_id][:2] - d.xpos[a_bid][:2])
            if dist <= m_dist: m_dist, c_id = dist, o_id
        if c_id != -1:
            o_n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, c_id) or ""
            eq_n = (f"eq_lock_b{'1' if 'box1' in o_n else '2'}" if "box" in o_n else "ramp_lock")
            eq_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eq_n)
            if eq_id != -1:
                cur_l = self.obj_locker_team.get(c_id)
                if lock_cmd > 0.0:
                    if cur_l is None or cur_l == a_team:
                        d.eq_active[eq_id] = 1; self.obj_locker_team[c_id] = a_team
                        m.geom_rgba[self.obj_geom_ids[c_id]] = [1, 0, 0, 1]
                else:
                    if cur_l == a_team:
                        d.eq_active[eq_id] = 0; self.obj_locker_team[c_id] = None
                        m.geom_rgba[self.obj_geom_ids[c_id]] = self.obj_default_colors[c_id]
            ag_n = agent_key[1:] if agent_key != "s" else "_s"
            eg_n = (f"eq_grasp{ag_n}_b{'1' if 'box1' in o_n else '2'}" if "box" in o_n else f"eq_grasp{ag_n}_ramp")
            eg_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, eg_n)
            if eg_id != -1:
                is_l = (eq_id != -1 and d.eq_active[eq_id])
                d.eq_active[eg_id] = 1 if (grab_cmd > 0.0 and not is_l) else 0

    def _get_obs(self, agent_idx):
        obs = np.zeros(53, dtype=np.float32); d, m = self.data, self.model
        k = self.agent_keys[agent_idx]; b_id = self.body_ids[k]
        pos3d = d.xpos[b_id].copy(); rot = float(d.qpos[m.jnt_qposadr[self.qpos_indices[k]['rot']]])
        v_gl = d.qvel[m.jnt_dofadr[self.qpos_indices[k]['x']] : m.jnt_dofadr[self.qpos_indices[k]['x']]+2]
        c, s = np.cos(-rot), np.sin(-rot); obs[0], obs[1] = v_gl[0]*c - v_gl[1]*s, v_gl[0]*s + v_gl[1]*c
        obs[2] = d.qvel[m.jnt_dofadr[self.qpos_indices[k]['rot']]]; obs[3], obs[4] = 1.0, 0.0
        obs[5:17] = np.maximum(0.0, self.vis_engine.cast_lidar(pos3d[:2], heading=rot, body_exclude=b_id) - 0.45)
        return obs

    def _normalize_obs(self, r):
        n = r.copy(); n[0:2] /= 10.0; n[2] /= 5.0; n[5:17] /= 15.0; n[17:52] /= 12.0
        return n

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None: np.random.seed(seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        self.obj_locker_team = {o_id: None for o_id in self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])}
        self._init_npcs()
        m, d, placed = self.model, self.data, []
        r_map = {"s": 0.6, "h1": 0.6, "h2": 0.6, "box1": 1.0, "box2": 1.0, "ramp": 1.5}
        targets = ["ramp", "box1", "box2"] + np.random.permutation(self.agent_keys).tolist()
        for key in targets:
            r_s = r_map.get(key, 0.7)
            for _ in range(500):
                pos = np.random.uniform(-5.2, 5.2, 2); overlap = False
                for (cx, cy, sx, sy) in self.maze_walls:
                    if abs(pos[0]-cx) < (sx+r_s) and abs(pos[1]-cy) < (sy+r_s): overlap = True; break
                if overlap: continue
                for (p_pos, p_rad) in placed:
                    if np.linalg.norm(pos-p_pos) < (r_s+p_rad+0.2): overlap = True; break
                if overlap: continue
                if key in self.agent_keys:
                    q = self.qpos_indices[key]; d.qpos[m.jnt_qposadr[q['x']]:m.jnt_qposadr[q['x']]+2] = pos; d.qpos[m.jnt_qposadr[q['rot']]] = np.random.uniform(-np.pi, np.pi)
                else:
                    target_body = f"{key}_body"; j_id = m.joint(m.body(target_body).jntadr[0]).id; d.qpos[m.jnt_qposadr[j_id]:m.jnt_qposadr[j_id]+2] = pos; d.qpos[m.jnt_qposadr[j_id]+2] = 0.5
                placed.append((pos, r_s)); break
        d.qvel[:] = 0.0
        for o_id in self.obj_geom_ids: m.geom_rgba[self.obj_geom_ids[o_id]] = self.obj_default_colors[o_id].copy()
        for i_eq in range(m.neq): d.eq_active[i_eq] = 0
        mujoco.mj_forward(m, d)
        if self.render_mode == "human": self.render()
        return self._normalize_obs(self._get_obs(1)), {"is_detected": False}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = [0, 0, 0.5]; self.viewer.cam.distance = 18.0; self.viewer.cam.elevation = -75.0
        self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close(); self.viewer = None