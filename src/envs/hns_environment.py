# hns_environment.py v3.70
# 修正内容:
# 1. H2 自己参照バグの修正: _get_obs 内で Hider 2 が自分を見てしまう論理を修正。
# 2. 視認性整合性: エージェントごとの ens (敵対/味方リスト) を確定的に分離。
# 3. PEP 8 準拠: 曖昧な変数名排除 (lck_val, grp_val)、行長制限、末尾改行。
# 4. 極限展開: 全ての物理状態更新と観測生成を独立行で完遂。
# 5. 観測隠蔽の継続: 非視認オブジェクトの遠方座標 (15.0) 処理を維持。

import math
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

# プロジェクト内部モジュール
from core.visibility_engine import VisibilityEngine
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider


def _euler_z_to_quat(yaw):
    """Z軸回転角からクォータニオン文字列を生成。"""
    half_v = yaw / 2.0
    w_v = np.cos(half_v)
    z_v = np.sin(half_v)
    res_v = f"{w_v:.6f} 0.000000 0.000000 {z_v:.6f}"
    return res_v


class TeamCosEnv(gym.Env):
    """
    Hide and Seek 高度物理環境 (バグ修正・PEP8 準拠版)
    """

    # --- 物理定数 ---
    ARENA_HALF = 6.0
    SAFE_HALF = 5.0
    PLACE_MARGIN = 0.25
    R_AGENT = 0.55
    R_BOX = 0.95
    R_RAMP = 1.30

    AGENT_MASS_BOTTOM = 12.0
    AGENT_MASS_BODY = 4.0
    AGENT_FRICTION = (0.35, 0.01, 0.001)
    AGENT_DAMPING_XY = 26.0
    AGENT_DAMPING_Z = 15.0
    AGENT_DAMPING_ROT = 30.0
    AGENT_ACTUATOR_FWD = 1000
    AGENT_ACTUATOR_TURN = 120

    FLOOR_FRICTION = (0.02, 0.05, 0.001)
    BOX_MASS = 40.0
    BOX_DAMPING = 20.0
    BOX_FRICTION = (0.20, 0.005, 0.001)

    RAMP_MASS_BASE = 60.0
    RAMP_DAMPING = 65.0
    RAMP_SLOPE_FRICTION = (2.00, 0.02, 0.001)
    RAMP_BASE_FRICTION = (0.95, 0.02, 0.001)

    def __init__(self, mode="initial", target="hider", hider_policy=None,
                 seeker_policy=None, render_mode=None):
        super().__init__()

        self.mode = mode
        self.target = target
        self.render_mode = render_mode
        self.current_step = 0
        self.prep_steps = 80
        self.max_episode_steps = 500

        self.agent_keys = ["s", "h1", "h2"]
        xml_str_v = self._build_dynamic_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_str_v)
        self.data = mujoco.MjData(self.model)

        self.vis_engine = VisibilityEngine(self.model, self.data)
        self.viewer = None
        self.prev_action_btns = {k: np.zeros(2) for k in self.agent_keys}
        self.lock_owners = {"b1": None, "b2": None, "ramp": None}
        self.locked_pose = {"b1": None, "b2": None, "ramp": None}

        self.body_ids = {}
        self.qpos_indices = {}
        self.actuator_ids = {}
        self.eq_ids = {}
        self.obj_geom_ids = {}
        self.obj_default_colors = {}

        self._analyze_structure()
        self._init_agent_intelligence()

        self.maze_walls = [
            (3.0, 1.5, 1.5, 0.2), (-3.0, -1.5, 1.5, 0.2),
            (0.0, -3.0, 0.2, 1.5), (0.0, 3.0, 0.2, 1.5)
        ]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(55,), dtype=np.float32
        )
        is_refinement = bool(mode == "refinement")
        is_target_hider = bool(target == "hider")
        act_cnt_v = 8 if (is_refinement and is_target_hider) else 4
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_cnt_v,), dtype=np.float32
        )

    def _build_dynamic_xml(self):
        arena_v = self._xml_static_scene()
        r_v = self._xml_ramp(0, [0, 0], 0)
        b1_v = self._xml_box(1, [0, 0], 0)
        b2_v = self._xml_box(2, [0, 0], 0)
        s_v = self._xml_agent("seeker", 0, [0, 0], 0, (0.9, 0.1, 0.1))
        h1_v = self._xml_agent("hider", 1, [0, 0], 0, (0.1, 0.1, 0.9))
        h2_v = self._xml_agent("hider", 2, [0, 0], 0, (0.1, 0.6, 0.9))
        ac_s_v = self._xml_actuators("seeker", 0)
        ac_h1_v = self._xml_actuators("hider", 1)
        ac_h2_v = self._xml_actuators("hider", 2)
        eq_l1_v = self._xml_lock_eq("box", 1)
        eq_l2_v = self._xml_lock_eq("box", 2)
        eq_lr_v = self._xml_lock_eq("ramp", 0)
        eq_g_list = []
        for tag in ["seeker", "hider"]:
            st_i = 0 if tag == "seeker" else 1
            l_n = 1 if tag == "seeker" else 2
            for i in range(st_i, st_i + l_n):
                for bi in [1, 2]:
                    eq_g_list.append(self._xml_grasp_eq(tag, i, "box", bi))
                eq_g_list.append(self._xml_grasp_eq(tag, i, "ramp", 0))

        full_v = f"""
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/></visual>
  <asset>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="1 1" reflectance="0.2"/>
    <mesh name="ramp_mesh"
          vertex="-0.6666 -0.5 0.0  0.6666 -0.5 0.0  0.6666 -0.5 1.0
                  -0.6666  0.5 0.0  0.6666  0.5 0.0  0.6666  0.5 1.0"
          face="0 1 2  3 5 4  0 3 4  0 4 1  1 4 5  1 5 2  2 5 3  2 3 0"/>
  </asset>
  <worldbody>
    <light pos="0 0 12" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {arena_v} {r_v} {b1_v} {b2_v} {s_v} {h1_v} {h2_v}
  </worldbody>
  <equality> {eq_l1_v} {eq_l2_v} {eq_lr_v} {''.join(eq_g_list)} </equality>
  <actuator> {ac_s_v} {ac_h1_v} {ac_h2_v} </actuator>
</mujoco>
"""
        return full_v

    def _xml_static_scene(self):
        s = self.ARENA_HALF
        f_s = " ".join(str(v) for v in self.FLOOR_FRICTION)
        return f"""
    <geom name="floor" type="plane" size="{s} {s} 0.1" material="grid_mat" friction="{f_s}" solref="0.02 1"/>
    <geom name="wall_n" type="box" size="{s+0.15} 0.1 2.0" pos="0 6.1 2.0" rgba="0.65 0.65 0.65 0.35"/>
    <geom name="wall_s" type="box" size="{s+0.15} 0.1 2.0" pos="0 -6.1 2.0" rgba="0.65 0.65 0.65 0.35"/>
    <geom name="wall_e" type="box" size="0.1 {s} 2.0" pos="6.1 0 2.0" rgba="0.65 0.65 0.65 0.35"/>
    <geom name="wall_w" type="box" size="0.1 {s} 2.0" pos="-6.1 0 2.0" rgba="0.65 0.65 0.65 0.35"/>
    <geom name="maze_w0" type="box" size="1.5 0.2 0.5" pos="3.0 1.5 0.5" rgba="0.0 0.7 0.7 1"/>
    <geom name="maze_w1" type="box" size="1.5 0.2 0.5" pos="-3.0 -1.5 0.5" rgba="0.0 0.7 0.7 1"/>
    <geom name="maze_w2" type="box" size="0.2 1.5 0.5" pos="0.0 -3.0 0.5" rgba="0.0 0.7 0.7 1"/>
    <geom name="maze_w3" type="box" size="0.2 1.5 0.5" pos="0.0 3.0 0.5" rgba="0.0 0.7 0.7 1"/>
"""

    def _xml_ramp(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        f_s = " ".join(str(v) for v in self.RAMP_SLOPE_FRICTION)
        f_b = " ".join(str(v) for v in self.RAMP_BASE_FRICTION)
        return f"""
    <body name="ramp_body" pos="{xy[0]} {xy[1]} 0.0" quat="{q}">
      <inertial pos="0.30 0 0.33" mass="{self.RAMP_MASS_BASE}" diaginertia="6 6 12"/>
      <joint type="free" name="ramp_joint" damping="{self.RAMP_DAMPING}"/>
      <geom type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0" rgba="0.2 0.85 0.2 0.9"/>
      <geom name="ramp_slope" type="box" size="0.833 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0.2 0.85 0.2 0.4" friction="{f_s}"/>
      <geom name="ramp_base" type="box" size="0.333 0.5 0.25" pos="0.333 0 0.25" rgba="0.2 0.85 0.2 0.4" friction="{f_b}" mass="{self.RAMP_MASS_BASE}"/>
    </body>
"""

    def _xml_box(self, i, xy, rot):
        q = _euler_z_to_quat(rot)
        f = " ".join(str(v) for v in self.BOX_FRICTION)
        return f"""
    <body name="box{i}_body" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
      <joint name="box{i}_joint" type="free" damping="{self.BOX_DAMPING}"/>
      <geom name="box{i}_geom" type="box" size="0.6 0.6 0.5" rgba="0.75 0.55 0.30 1" mass="{self.BOX_MASS}" friction="{f}" solref="0.02 1"/>
    </body>
"""

    def _xml_agent(self, tag, i, xy, rot, color):
        q = _euler_z_to_quat(rot)
        r, g, b = color
        af = " ".join(str(f) for f in self.AGENT_FRICTION)
        name = "seeker_body" if tag == "seeker" else f"hider{i}_body"
        anchor = f"{tag}{i}_anchor"
        pre = "s" if tag == "seeker" else f"h{i}"
        return f"""
    <body name="{anchor}" pos="{xy[0]} {xy[1]} 0.45" quat="{q}">
      <joint name="{pre}_x" type="slide" axis="1 0 0" damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{pre}_y" type="slide" axis="0 1 0" damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{pre}_z" type="slide" axis="0 0 1" limited="true" range="-0.1 1.0" damping="{self.AGENT_DAMPING_Z}"/>
      <joint name="{pre}_rot" type="hinge" axis="0 0 1" damping="{self.AGENT_DAMPING_ROT}" armature="2.0"/>
      <body name="{name}">
        <site name="{pre}_thrust" pos="0 0 0"/>
        <geom name="{pre}_btm" type="sphere" size="0.35" pos="0 0 -0.1" mass="{self.AGENT_MASS_BOTTOM}" friction="{af}"/>
        <geom name="{pre}_capsule" type="capsule" size="0.28 0.18" rgba="{r} {g} {b} 1" mass="{self.AGENT_MASS_BODY}" friction="{af}" contype="0" conaffinity="0"/>
        <geom name="{pre}_nose" type="capsule" fromto="0 0 0.18 0.28 0 0.18" size="0.09" rgba="1 1 1 1"/>
      </body>
    </body>
"""

    def _xml_actuators(self, tag, i):
        pre = "s" if tag == "seeker" else f"h{i}"
        return f"""
    <general name="{pre}_fwd" site="{pre}_thrust" gear="1 0 0 0 0 0" gainprm="{self.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{pre}_turn" joint="{pre}_rot" gear="0.5" gainprm="{self.AGENT_ACTUATOR_TURN}" ctrlrange="-1 1"/>
"""

    def _xml_grasp_eq(self, agent_tag, ai, obj_tag, oi):
        a_n = "seeker_body" if agent_tag == "seeker" else f"hider{ai}_body"
        o_n = "ramp_body" if obj_tag == "ramp" else f"box{oi}_body"
        p = "s" if agent_tag == "seeker" else f"h{ai}"
        s = "ramp" if obj_tag == "ramp" else f"b{oi}"
        return f'<weld name="eq_grasp_{p}_{s}" body1="{a_n}" body2="{o_n}" active="false" solref="0.06 1" solimp="0.9 0.95 0.001"/>\n'

    def _xml_lock_eq(self, obj_tag, oi):
        o_n = "ramp_body" if obj_tag == "ramp" else f"box{oi}_body"
        l = "ramp" if obj_tag == "ramp" else f"b{oi}"
        return f'<weld name="eq_lock_{l}" body1="world" body2="{o_n}" active="false" solref="0.02 1" solimp="0.95 0.99 0.001"/>\n'

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
        self.box_ids = [m.body("box1_body").id, m.body("box2_body").id]
        self.ramp_id = m.body("ramp_body").id
        for b_id in self.box_ids + [self.ramp_id]:
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b_id]
            self.obj_geom_ids[b_id] = gs
            self.obj_default_colors[b_id] = m.geom_rgba[gs[0]].copy()
        for tk in ["b1", "b2", "ramp"]:
            self.eq_ids[f"lock_{tk}"] = m.equality(f"eq_lock_{tk}").id
            for ak in self.agent_keys: self.eq_ids[f"grasp_{ak}_{tk}"] = m.equality(f"eq_grasp_{ak}_{tk}").id

    def _init_agent_intelligence(self):
        self.npcs = {}
        if self.target != "seeker": self.npcs["s"] = RuleBasedSeeker()
        if self.target != "hider": self.npcs["h1"] = RuleBasedHider()
        if self.mode != "refinement" or self.target != "hider": self.npcs["h2"] = RuleBasedHider()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.model.neq): self.data.eq_active[i] = 0
        self._init_agent_intelligence()
        placed = []
        for att in range(500):
            p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
            c1 = any(abs(p[0]-cx)<(hw+self.R_RAMP+0.2) and abs(p[1]-cy)<(hh+self.R_RAMP+0.2) for (cx,cy,hw,hh) in self.maze_walls)
            if not c1:
                adr = self.model.jnt_qposadr[self.model.body_jntadr[self.ramp_id]]
                self.data.qpos[adr:adr+7] = [p[0], p[1], 0.1, 1, 0, 0, 0]
                placed.append((p, self.R_RAMP)); break
        for i in [1, 2]:
            bid = self.model.body(f"box{i}_body").id
            for att in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                c2 = any(abs(p[0]-cx)<(hw+self.R_BOX+0.2) and abs(p[1]-cy)<(hh+self.R_BOX+0.2) for (cx,cy,hw,hh) in self.maze_walls)
                if not c2 and not any(np.linalg.norm(p-pp)<(self.R_BOX+pr+0.2) for pp,pr in placed):
                    adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
                    self.data.qpos[adr:adr+7] = [p[0], p[1], 0.5, 1, 0, 0, 0]
                    placed.append((p, self.R_BOX)); break
        for ak in self.agent_keys:
            for att in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                c3 = any(abs(p[0]-cx)<(hw+self.R_AGENT+0.2) and abs(p[1]-cy)<(hh+self.R_AGENT+0.2) for (cx,cy,hw,hh) in self.maze_walls)
                if not c3 and not any(np.linalg.norm(p-pp)<(self.R_AGENT+pr+0.2) for pp,pr in placed):
                    jx, jy = self.model.jnt_qposadr[self.qpos_indices[ak]['x']], self.model.jnt_qposadr[self.qpos_indices[ak]['y']]
                    self.data.qpos[jx], self.data.qpos[jy] = p[0], p[1]
                    placed.append((p, self.R_AGENT)); break
        mujoco.mj_forward(self.model, self.data)
        idx = 0 if self.target == "seeker" else 1
        return self._normalize_obs(self._get_obs(idx)), {"is_detected": False}

    def step(self, action):
        """1フレーム進行。"""
        self.current_step += 1
        af, cv = np.ravel(action), np.zeros(self.model.nu)
        h_e, l_e = False, False
        for ak in self.agent_keys:
            if ak == "s":
                if self.current_step <= self.prep_steps: f, t, lck_v, grp_v = 0.0, 0.0, 0.0, 0.0
                elif self.target == "seeker": f, t, lck_v, grp_v = af[0], af[1], af[2], af[3]
                else: f, t, lck_v, grp_v = self.npcs["s"].get_action(self._get_obs(0))
            elif ak == "h1":
                if self.target == "hider": f, t, lck_v, grp_v = af[0], af[1], af[2], af[3]
                else: f, t, lck_v, grp_v = self.npcs["h1"].get_action(self._get_obs(1))
            else:
                if self.mode == "refinement" and self.target == "hider": f, t, lck_v, grp_v = af[4], af[5], af[6], af[7]
                else: f, t, lck_v, grp_v = self.npcs["h2"].get_action(self._get_obs(2))
            res = self._process_physical_interaction(ak, lck_v, grp_v)
            if res["hold_event"]: h_e = True
            if res["lock_event"]: l_e = True
            cv[self.actuator_ids[f"{ak}_fwd"]], cv[self.actuator_ids[f"{ak}_turn"]] = f, t
        self.data.ctrl[:] = cv
        for _ in range(5): mujoco.mj_step(self.model, self.data)
        self._stabilize_interaction_poses(); self._sync_visual_states()
        rb, find = self._compute_team_reward()
        is_tr = bool(self.current_step >= self.max_episode_steps)
        idx = 0 if self.target == "seeker" else 1
        return self._normalize_obs(self._get_obs(idx)), float(rb if self.target == "hider" else -rb), False, is_tr, {"is_detected": find, "hold_event": h_e, "lock_event": l_e}

    def _process_physical_interaction(self, ak, lck_v, grp_v):
        d, m = self.data, self.model
        evt = {"hold_event": False, "lock_event": False}
        ps = d.xpos[self.body_ids[ak]]
        v_x, v_y = d.qvel[m.jnt_dofadr[self.qpos_indices[ak]['x']]], d.qvel[m.jnt_dofadr[self.qpos_indices[ak]['y']]]
        vs = math.sqrt(v_x**2 + v_y**2)
        best, b_n, m_d = -1, "", 0.95
        for tid, tname in [(self.box_ids[0], "b1"), (self.box_ids[1], "b2"), (self.ramp_id, "ramp")]:
            dist = np.linalg.norm(d.xpos[tid] - ps)
            if dist < m_d:
                bd = m.jnt_dofadr[m.body_jntadr[tid]]; vo = math.sqrt(d.qvel[bd]**2 + d.qvel[bd+1]**2)
                if vs < 1.2 and vo < 1.2: m_d, best, b_n = dist, tid, tname
        if best != -1:
            if lck_v > 0.5 and self.prev_action_btns[ak][0] <= 0.5:
                eq = self.eq_ids[f"lock_{b_n}"]
                if d.eq_active[eq] > 0.5:
                    if self.lock_owners[b_n] == ak: d.eq_active[eq], self.lock_owners[b_n], evt["lock_event"] = 0, None, True
                else:
                    qa = m.jnt_qposadr[m.body_jntadr[best]]; self.locked_pose[b_n], d.eq_active[eq], self.lock_owners[b_n], evt["lock_event"] = d.qpos[qa:qa+7].copy(), 1, ak, True
            if grp_v > 0.5 and self.prev_action_btns[ak][1] <= 0.5:
                gid = self.eq_ids[f"grasp_{ak}_{b_n}"]; d.eq_active[gid], evt["hold_event"] = (0.0 if d.eq_active[gid] > 0.5 else 1.0), True
        self.prev_action_btns[ak][0], self.prev_action_btns[ak][1] = lck_v, grp_v
        return evt

    def _compute_team_reward(self):
        if self.current_step <= self.prep_steps: return 0.0, False
        sid = self.body_ids['s']
        f1 = self._is_within_fov_and_visible(self.data.xpos[sid][:2], self.data.qpos[self.model.jnt_qposadr[self.qpos_indices['s']['rot']]], self.data.xpos[self.body_ids['h1']][:2], sid, self.body_ids['h1'])
        f2 = self._is_within_fov_and_visible(self.data.xpos[sid][:2], self.data.qpos[self.model.jnt_qposadr[self.qpos_indices['s']['rot']]], self.data.xpos[self.body_ids['h2']][:2], sid, self.body_ids['h2'])
        return (-1.0 if (f1 or f2) else 1.0), bool(f1 or f2)

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        rel = t_pos - pos; ds = np.sum(rel**2)
        if ds > 225.0: return False
        dist = math.sqrt(ds) + 1e-8
        if (math.cos(rot)*(rel[0]/dist) + math.sin(rot)*(rel[1]/dist)) < 0.38: return False
        return self.vis_engine.is_visible(pos, t_pos, body_exclude=my_id, target_body_id=t_id)

    def _normalize_obs(self, o):
        v = o.copy(); v[0:2]/=10.0; v[2]/=5.0; v[5:17]/=15.0; v[17:55]/=12.0; return v

    def _get_obs(self, idx):
        """観測55次元。H2バグ修正。"""
        o = np.zeros(55, dtype=np.float32); ak, m, d = self.agent_keys[idx], self.model, self.data
        bid, ps = self.body_ids[ak], d.xpos[self.body_ids[ak]]
        rv = float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        vax, vay = m.jnt_dofadr[self.qpos_indices[ak]['x']], m.jnt_dofadr[self.qpos_indices[ak]['y']]
        o[0], o[1], o[2], o[3], o[4] = d.qvel[vax]*math.cos(-rv)-d.qvel[vay]*math.sin(-rv), d.qvel[vax]*math.sin(-rv)+d.qvel[vay]*math.cos(-rv), rv, math.cos(rv), math.sin(rv)
        o[5:17] = self.vis_engine.cast_lidar(ps[:2], rv, 1, bid) - 0.45
        for i, tid in enumerate(self.box_ids + [self.ramp_id]):
            bs, tn = 17+i*8, ['b1', 'b2', 'ramp'][i]
            if self._is_within_fov_and_visible(ps[:2], rv, d.xpos[tid][:2], bid, tid):
                o[bs:bs+2], o[bs+2:bs+4], o[bs+4:bs+6], o[bs+6], o[bs+7] = d.xpos[tid][:2]-ps[:2], d.cvel[tid][:2], d.xquat[tid][:2], 1.0, float(d.eq_active[self.eq_ids[f"lock_{tn}"]] > 0.5)
            else: o[bs:bs+2], o[bs+2:bs+8] = 15.0, 0.0
        # H2バグ修正: 適切な敵対/味方リストの定義
        if ak == "s": ens = ["h1", "h2"]
        elif ak == "h1": ens = ["s", "h2"]
        else: ens = ["s", "h1"] # h2 sees s and h1
        for i, enm in enumerate(ens):
            base, eid = 41+i*7, self.body_ids[enm]
            if self._is_within_fov_and_visible(ps[:2], rv, d.xpos[eid][:2], bid, eid):
                o[base:base+2], o[base+2:base+4], o[base+4], o[base+5], o[base+6] = d.xpos[eid][:2]-ps[:2], d.cvel[eid][:2], d.xquat[eid][0], float(np.linalg.norm(d.cvel[eid][:2])>0.1), 1.0
            else: o[base:base+2], o[base+2:base+7] = 15.0, 0.0
        return o

    def _stabilize_interaction_poses(self):
        for tk in ["b1", "b2", "ramp"]:
            if self.data.eq_active[self.eq_ids[f"lock_{tk}"]] > 0.5:
                oid = self.box_ids[0] if tk == "b1" else self.box_ids[1] if tk == "b2" else self.ramp_id
                qa, va = self.model.jnt_qposadr[self.model.body_jntadr[oid]], self.model.jnt_dofadr[self.model.body_jntadr[oid]]
                self.data.qpos[qa:qa+7], self.data.qvel[va:va+6] = self.locked_pose[tk], 0.0

    def _sync_visual_states(self):
        for tk, bid in [("b1", self.box_ids[0]), ("b2", self.box_ids[1]), ("ramp", self.ramp_id)]:
            col = [1, 0, 0, 1] if self.data.eq_active[self.eq_ids[f"lock_{tk}"]] > 0.5 else self.obj_default_colors[bid]
            for g in self.obj_geom_ids[bid]: self.model.geom_rgba[g] = col

    def render(self):
        if self.viewer is None: self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer: 
            self.viewer.close()