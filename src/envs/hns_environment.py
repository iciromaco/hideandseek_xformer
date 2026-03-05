# src/envs/hns_environment.py
# hns_environment.py v4.58 (学習対象を 1 エージェントに限定するロジックの修正)

import math
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from core.visibility_engine import VisibilityEngine
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider
from core.obs_indices import ObsIdx


def _euler_z_to_quat(yaw):
    """Z軸回転角からクォータニオン文字列を生成。"""
    w, z = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return f"{w:.6f} 0 0 {z:.6f}"


class TeamCosEnv(gym.Env):
    """
    Hide and Seek 高度物理環境 (単一エージェント学習最適化版)
    """
    ARENA_HALF = 6.0
    SAFE_HALF = 5.0
    R_AGENT = 0.55
    R_BOX = 0.95
    R_RAMP = 1.30

    AGENT_DAMPING_XY = 30.0
    AGENT_DAMPING_ROT = 25.0
    AGENT_ACTUATOR_FWD = 1000

    def __init__(self, mode="initial", target="hider", n_seekers=1,
                 n_hiders=2, n_boxes=2, n_ramps=1, render_mode=None):
        super().__init__()
        self.mode, self.target, self.render_mode = mode, target, render_mode
        self.current_step, self.prep_steps, self.max_episode_steps = 0, 80, 500
        self.agent_keys = ["s", "h1", "h2"]
        self.idx = ObsIdx(n_boxes, n_ramps, n_others=2)

        self.model = mujoco.MjModel.from_xml_string(self._build_dynamic_xml())
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data)
        self.viewer = None
        
        self.last_debug_ctrl = {k: (0.0, 0.0) for k in self.agent_keys}
        
        self.body_ids, self.qpos_indices, self.actuator_ids = {}, {}, {}
        self.maze_walls = [(3, 1.5, 1.5, 0.2), (-3, -1.5, 1.5, 0.2),
                          (0, -3, 0.2, 1.5), (0, 3, 0.2, 1.5)]
        self._analyze_structure()
        self._init_agent_intelligence()
        
        # 観測空間
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.idx.total_dim,), np.float32)
        
        # 【修正】アクションスペースを 4 次元に固定。
        # 外部（学習アルゴリズム）からは常に 1 体分の入力を受け取る。
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    def _build_dynamic_xml(self):
        arena = self._xml_static_scene()
        r = self._xml_ramp(0, [0, 0], 0)
        b1, b2 = self._xml_box(1, [0, 0], 0), self._xml_box(2, [0, 0], 0)
        s = self._xml_agent("seeker", 0, [0, 0], 0, (0.9, 0.1, 0.1))
        h1 = self._xml_agent("hider", 1, [0, 0], 0, (0.1, 0.1, 0.9))
        h2 = self._xml_agent("hider", 2, [0, 0], 0, (0.1, 0.6, 0.9))
        acts = (self._xml_actuators("seeker", 0) + 
                self._xml_actuators("hider", 1) + 
                self._xml_actuators("hider", 2))
        
        return f"""
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>
    <mesh name="ramp_mesh" 
          vertex="-0.6666 -0.5 0.0 0.6666 -0.5 0.0 0.6666 -0.5 1.0 -0.6666 0.5 0.0 0.6666 0.5 0.0 0.6666 0.5 1.0"
          face="0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0"/>
  </asset>
  <worldbody>
    <light pos="0 0 12" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="0 13 13" euler="2.35 0 -3.14" mode="fixed" />
    {arena} {r} {b1} {b2} {s} {h1} {h2}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>"""

    def _xml_static_scene(self):
        s = self.ARENA_HALF
        attr = 'rgba="0.65 0.65 0.65 0.35" friction="0.1 0.1 0.1" solref="0.01 1" solimp="0.95 0.99 0.001"'
        return f"""
    <geom name="floor" type="plane" size="{s} {s} 0.1" material="grid" friction="0.1 0.05 0.001"/>
    <geom name="wall_n" type="box" size="{s+0.15} 0.1 2.0" pos="0 6.1 2.0" {attr}/>
    <geom name="wall_s" type="box" size="{s+0.15} 0.1 2.0" pos="0 -6.1 2.0" {attr}/>
    <geom name="wall_e" type="box" size="0.1 {s} 2.0" pos="6.1 0 2.0" {attr}/>
    <geom name="wall_w" type="box" size="0.1 {s} 2.0" pos="-6.1 0 2.0" {attr}/>
    <geom name="maze_w0" type="box" size="1.5 0.2 0.5" pos="3.0 1.5 0.5" rgba="0 0.7 0.7 1"/>
    <geom name="maze_w1" type="box" size="1.5 0.2 0.5" pos="-3.0 -1.5 0.5" rgba="0 0.7 0.7 1"/>
    <geom name="maze_w2" type="box" size="0.2 1.5 0.5" pos="0.0 -3.0 0.5" rgba="0 0.7 0.7 1"/>
    <geom name="maze_w3" type="box" size="0.2 1.5 0.5" pos="0.0 3.0 0.5" rgba="0 0.7 0.7 1"/>
"""

    def _xml_ramp(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        return f"""
    <body name="ramp_body" pos="{xy[0]} {xy[1]} 0" quat="{q}">
      <joint type="free" name="ramp_joint" damping="60.0"/>
      <geom type="mesh" mesh="ramp_mesh" rgba="0 1 0 1" friction="0.5 0.1 0.1"/>
    </body>"""

    def _xml_box(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        return f"""
    <body name="box{i}_body" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
      <joint name="box{i}_joint" type="free" damping="20"/>
      <geom name="box{i}_geom" type="box" size="0.6 0.6 0.5" rgba="0.75 0.55 0.3 1" friction="0.5 0.005 0.001"/>
    </body>"""

    def _xml_agent(self, tag, i, xy, rot, color):
        q, pre = _euler_z_to_quat(rot), ("s" if tag == "seeker" else f"h{i}")
        r, g, b = color
        return f"""
    <body name="{tag}{i}_anchor" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
      <joint name="{pre}_x" type="slide" axis="1 0 0" damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{pre}_y" type="slide" axis="0 1 0" damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{pre}_rot" type="hinge" axis="0 0 1" damping="{self.AGENT_DAMPING_ROT}"/>
      <body name="{'seeker_body' if tag=='seeker' else f'hider{i}_body'}">
        <site name="{pre}_thrust" pos="0 0 0"/>
        <geom name="{pre}_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="12" friction="0.1 0.1 0.1"/>
        <geom name="{pre}_capsule" type="capsule" size="0.3 0.3" rgba="{r} {g} {b} 1" mass="4" contype="0" conaffinity="0"/>
        <geom name="{pre}_nose" type="capsule" fromto="0 0 0.3 0.3 0 0.3" size="0.09" rgba="1 1 1 1" contype="0" conaffinity="0"/>
      </body>
    </body>"""

    def _xml_actuators(self, tag, i):
        pre = "s" if tag == "seeker" else f"h{i}"
        return f"""
    <general name="{pre}_fwd" site="{pre}_thrust" gear="1 0 0 0 0 0" gainprm="{self.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{pre}_turn" joint="{pre}_rot" gear="0.5" gainprm="120" ctrlrange="-1 1"/>\n"""

    def _analyze_structure(self):
        m = self.model
        self.body_ids["s"] = m.body("seeker_body").id
        self.qpos_indices["s"] = {'x': m.joint("s_x").id, 'y': m.joint("s_y").id, 'rot': m.joint("s_rot").id}
        self.actuator_ids["s_fwd"] = m.actuator("s_fwd").id
        self.actuator_ids["s_turn"] = m.actuator("s_turn").id
        for i in [1, 2]:
            self.body_ids[f"h{i}"] = m.body(f"hider{i}_body").id
            self.qpos_indices[f"h{i}"] = {'x': m.joint(f"h{i}_x").id, 'y': m.joint(f"h{i}_y").id, 'rot': m.joint(f"h{i}_rot").id}
            self.actuator_ids[f"h{i}_fwd"] = m.actuator(f"h{i}_fwd").id
            self.actuator_ids[f"h{i}_turn"] = m.actuator(f"h{i}_turn").id
        self.ramp_id, self.box_ids = m.body("ramp_body").id, [m.body("box1_body").id, m.body("box2_body").id]

    def _init_agent_intelligence(self):
        self.npcs = {"s": RuleBasedSeeker(), "h1": RuleBasedHider(), "h2": RuleBasedHider()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        self._init_agent_intelligence()
        placed = []
        for bid, rad, z in [(self.ramp_id, self.R_RAMP, 0.1)] + [(b, self.R_BOX, 0.5) for b in self.box_ids]:
            for _ in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                if not any(np.linalg.norm(p-pp) < (rad+pr+0.2) for pp, pr in placed):
                    adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
                    self.data.qpos[adr:adr+7] = [p[0], p[1], z, 1, 0, 0, 0]
                    placed.append((p, rad)); break
        for ak in self.agent_keys:
            for _ in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2); rot = np.random.uniform(-np.pi, np.pi)
                if not any(np.linalg.norm(p-pp) < (self.R_AGENT+pr+0.3) for pp, pr in placed):
                    jx, jy, jr = self.qpos_indices[ak]['x'], self.qpos_indices[ak]['y'], self.qpos_indices[ak]['rot']
                    self.data.qpos[self.model.jnt_qposadr[jx]], self.data.qpos[self.model.jnt_qposadr[jy]], self.data.qpos[self.model.jnt_qposadr[jr]] = p[0], p[1], rot
                    placed.append((p, self.R_AGENT)); break
        mujoco.mj_forward(self.model, self.data)
        
        # 学習対象の観測を返す (seeker なら s, hider なら h1)
        idx_to_obs = 0 if self.target == "seeker" else 1
        return self._normalize_obs(self._get_obs(idx_to_obs)), {"is_detected": False}

    def step(self, action):
        self.current_step += 1
        af, cv = np.ravel(action), np.zeros(self.model.nu)
        
        for i, ak in enumerate(self.agent_keys):
            # 【核心】学習対象の判定: seeker なら s, hider なら h1 のみ。
            # h2 は常に NPC (RuleBasedHider) または 推論モデル (将来実装) で制御する。
            if (self.target == "seeker" and ak == "s") or (self.target == "hider" and ak == "h1"):
                if ak == "s" and self.current_step <= self.prep_steps:
                    f, t = 0.0, 0.0
                else:
                    # 外部アクション（常に4要素）を適用
                    f, t = af[0], af[1]
            else:
                # それ以外（非ターゲットの Seeker や、Hider2）は内部 NPC が制御
                if ak == "s" and self.current_step <= self.prep_steps:
                    f, t = 0.0, 0.0
                else:
                    norm_obs = self._normalize_obs(self._get_obs(i))
                    f, t, _, _ = self.npcs[ak].get_action(norm_obs, self.idx)
            
            cv[self.actuator_ids[f"{ak}_fwd"]], cv[self.actuator_ids[f"{ak}_turn"]] = f, t
            self.last_debug_ctrl[ak] = (f, t)
            
        self.data.ctrl[:] = cv
        for _ in range(5): mujoco.mj_step(self.model, self.data)
        
        rb, find = self._compute_team_reward()
        # 学習対象に合わせて観測を生成
        idx_to_obs = 0 if self.target == "seeker" else 1
        
        return (self._normalize_obs(self._get_obs(idx_to_obs)), 
                float(rb if self.target == "hider" else -rb), 
                False, 
                self.current_step >= self.max_episode_steps, 
                {"is_detected": find})

    def _compute_team_reward(self):
        if self.current_step <= self.prep_steps: return 0.0, False
        sid, h1id, h2id = self.body_ids['s'], self.body_ids['h1'], self.body_ids['h2']
        sp, sr = self.data.xpos[sid][:2], self.data.qpos[self.model.jnt_qposadr[self.qpos_indices['s']['rot']]]
        f1 = self._is_vis(sp, sr, self.data.xpos[h1id][:2], sid, h1id)
        f2 = self._is_vis(sp, sr, self.data.xpos[h2id][:2], sid, h2id)
        return (-1.0 if (f1 or f2) else 1.0), bool(f1 or f2)

    def _is_vis(self, pos, rot, t_pos, my_id, t_id):
        rel = t_pos - pos; dist = math.sqrt(np.sum(rel**2)) + 1e-8
        if dist > 15.0 or (math.cos(rot)*(rel[0]/dist) + math.sin(rot)*(rel[1]/dist)) < 0.38: return False
        return self.vis_engine.is_visible(pos, t_pos, body_exclude=my_id, target_body_id=t_id)

    def _normalize_obs(self, o):
        v = o.copy(); idx = self.idx
        v[idx.SELF.VEL_X] /= 10.0; v[idx.SELF.VEL_Y] /= 10.0; v[idx.SELF.ROT] /= 5.0; v[idx.LIDAR] /= 15.0
        for b in idx.B: v[b.REL_X] /= 12.0; v[b.REL_Y] /= 12.0; v[b.VEL_X] /= 10.0; v[b.VEL_Y] /= 10.0
        for r in idx.RAMP: v[r.REL_X] /= 12.0; v[r.REL_Y] /= 12.0; v[r.VEL_X] /= 10.0; v[r.VEL_Y] /= 10.0
        for en in idx.OTHERS: v[en.REL_X] /= 12.0; v[en.REL_Y] /= 12.0; v[en.VEL_X] /= 10.0; v[en.VEL_Y] /= 10.0
        return v

    def _get_obs(self, idx):
        o = np.zeros(self.idx.total_dim, dtype=np.float32); ak, m, d = self.agent_keys[idx], self.model, self.data
        ps, rv = d.xpos[self.body_ids[ak]], float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        vax, vay = m.jnt_dofadr[self.qpos_indices[ak]['x']], m.jnt_dofadr[self.qpos_indices[ak]['y']]
        cos_r, sin_r = math.cos(-rv), math.sin(-rv)
        si = self.idx.SELF
        o[si.VEL_X] = d.qvel[vax] * cos_r - d.qvel[vay] * sin_r
        o[si.VEL_Y] = d.qvel[vax] * sin_r + d.qvel[vay] * cos_r
        o[si.ROT] = rv; o[si.COS_ROT], o[si.SIN_ROT] = math.cos(rv), math.sin(rv)
        o[self.idx.LIDAR] = self.vis_engine.cast_lidar(ps[:2], rv, 1, self.body_ids[ak])
        for i, tid in enumerate(self.box_ids):
            b_idx = self.idx.B[i]; d_w = d.xpos[tid][:2] - ps[:2]
            o[b_idx.REL_X] = d_w[0] * cos_r - d_w[1] * sin_r; o[b_idx.REL_Y] = d_w[0] * sin_r + d_w[1] * cos_r; o[b_idx.IS_MOVING] = 1.0
        r_idx = self.idx.RAMP[0]; d_w_r = d.xpos[self.ramp_id][:2] - ps[:2]
        o[r_idx.REL_X] = d_w_r[0] * cos_r - d_w_r[1] * sin_r; o[r_idx.REL_Y] = d_w_r[0] * sin_r + d_w_r[1] * cos_r; o[r_idx.IS_MOVING] = 1.0
        ens = ["h1", "h2"] if ak == "s" else ["s", "h2"] if ak == "h1" else ["s", "h1"]
        for i, enm in enumerate(ens):
            en_idx = self.idx.OTHERS[i]; eid = self.body_ids[enm]
            if self._is_vis(ps[:2], rv, d.xpos[eid][:2], self.body_ids[ak], eid):
                d_w = d.xpos[eid][:2] - ps[:2]; o[en_idx.REL_X] = d_w[0] * cos_r - d_w[1] * sin_r; o[en_idx.REL_Y] = d_w[0] * sin_r + d_w[1] * cos_r; o[en_idx.VISIBLE] = 1.0
            else: o[en_idx.REL_X], o[en_idx.REL_Y] = 15.0, 15.0; o[en_idx.VISIBLE] = 0.0
        return o

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0, 0, 0.8]; self.viewer.cam.distance, self.viewer.cam.elevation, self.viewer.cam.azimuth = 18.0, -35.0, 90.0
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = 0
            for ak in self.agent_keys:
                sid = self.body_ids[ak]; pos = self.data.xpos[sid]; rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]['rot']]]
                t_val = self.last_debug_ctrl[ak][1]
                if abs(t_val) > 0.005:
                    h = 1.1 if ak == "s" else 1.3
                    color = [0, 1, 1, 1.0] if ak == "s" else [0, 0.2, 1, 1.0]
                    p_start = pos.copy(); p_start[2] = h; p_end = p_start.copy()
                    p_end[0] += -math.sin(rot) * t_val * 2.0; p_end[1] += math.cos(rot) * t_val * 2.0
                    if self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                        g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 8.0 + abs(t_val)*25, p_start, p_end)
                        g.rgba[:] = color
                        self.viewer.user_scn.ngeom += 1
                targets = [(self.body_ids[k], [1,0,0,1] if ak=='s' else [0,0,1,1]) for k in self.agent_keys if k != ak]
                for tid, color in targets:
                    if self._is_vis(pos[:2], rot, self.data.xpos[tid][:2], sid, tid):
                        if self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                            g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]
                            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 2.0, pos, self.data.xpos[tid])
                            g.rgba[:] = color
                            self.viewer.user_scn.ngeom += 1
            obs_raw = self._get_obs(0); h1_idx = self.idx.OTHERS[0]
            if obs_raw[h1_idx.VISIBLE] > 0.5:
                lx, ly = obs_raw[h1_idx.REL_X], obs_raw[h1_idx.REL_Y]; sid = self.body_ids["s"]; spos = self.data.xpos[sid][:2]; srot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices["s"]["rot"]]]
                wx = spos[0] + lx * math.cos(srot) - ly * math.sin(srot); wy = spos[1] + lx * math.sin(srot) + ly * math.cos(srot)
                if self.viewer.user_scn.ngeom < self.viewer.user_scn.maxgeom:
                    g = self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom]; g.type, g.size[:], g.pos[:], g.rgba[:], g.matid = mujoco.mjtGeom.mjGEOM_SPHERE, [0.2, 0.2, 0.2], [wx, wy, 1.2], [1, 1, 0, 1], -1; self.viewer.user_scn.ngeom += 1
        self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close()