# src/envs/hns_environment.py
# hns_environment.py v4.26 (ブレーキ強化とセンサー感度の最適化)

import math
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

# プロジェクト内部モジュール
from core.visibility_engine import VisibilityEngine
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider
from core.obs_indices import ObsIdx


def _euler_z_to_quat(yaw):
    """Z軸回転角からクォータニオン文字列を生成。"""
    half_v = yaw / 2.0
    w_v = np.cos(half_v)
    z_v = np.sin(half_v)
    res_v = f"{w_v:.6f} 0.000000 0.000000 {z_v:.6f}"
    return res_v


class TeamCosEnv(gym.Env):
    """
    Hide and Seek 高度物理環境 (能動的減速支援版)
    """

    # --- 物理定数 (v3.70 準拠 + ブレーキ性能強化) ---
    ARENA_HALF = 6.0
    SAFE_HALF = 5.0
    PLACE_MARGIN = 0.25
    # エージェントの当たり判定半径
    R_AGENT = 0.60
    R_BOX = 0.95
    R_RAMP = 1.30

    # 質量設定
    AGENT_MASS_BOTTOM = 12.0
    AGENT_MASS_BODY = 4.0
    AGENT_FRICTION = (0.35, 0.01, 0.001)

    # ブレーキ（減衰）設定を 26.0 から 45.0 に大幅強化。
    # これにより推力をオフにした瞬間に急停止しやすくなります。
    AGENT_DAMPING_XY = 45.0
    AGENT_DAMPING_Z = 15.0
    AGENT_DAMPING_ROT = 40.0

    # 推進力は維持しつつ、減衰とのバランスで制御しやすくします
    AGENT_ACTUATOR_FWD = 1000
    AGENT_ACTUATOR_TURN = 120

    # 環境オブジェクトの物理設定
    FLOOR_FRICTION = (0.02, 0.05, 0.001)
    BOX_MASS = 40.0
    BOX_DAMPING = 20.0
    BOX_FRICTION = (0.20, 0.005, 0.001)

    RAMP_MASS_BASE = 60.0
    RAMP_DAMPING = 65.0
    RAMP_SLOPE_FRICTION = (2.00, 0.02, 0.001)
    RAMP_BASE_FRICTION = (0.95, 0.02, 0.001)

    def __init__(self, mode="initial", target="hider", hider_policy=None,
                 seeker_policy=None, render_mode=None, **kwargs):
        super().__init__()

        self.mode = mode
        self.target = target
        self.render_mode = render_mode
        self.current_step = 0
        self.prep_steps = 80
        self.max_episode_steps = 500

        self.agent_keys = ["s", "h1", "h2"]
        # 観測インデックスの初期化
        self.idx = ObsIdx(n_boxes=2, n_ramps=1, n_others=2)

        # XML の動的ビルド
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
        is_refinement = bool(self.mode == "refinement")
        is_target_hider = bool(self.target == "hider")
        act_cnt_v = 8 if (is_refinement and is_target_hider) else 4
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_cnt_v,), dtype=np.float32
        )

    def _build_dynamic_xml(self):
        """v3.70 の構成と最適化された描画設定で XML を生成。"""
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
  <statistic center="0 0 0.8" extent="10"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
    <global azimuth="90" elevation="-35"/>
  </visual>
  <asset>
    <texture name="grid_tex" type="2d" builtin="checker" rgb1=".1 .2 .3" 
             rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="1 1" 
              reflectance="0.2"/>
    <mesh name="ramp_mesh"
          vertex="-0.6666 -0.5 0.0  0.6666 -0.5 0.0  0.6666 -0.5 1.0
                  -0.6666  0.5 0.0  0.6666  0.5 0.0  0.6666  0.5 1.0"
          face="0 1 2  3 5 4  0 3 4  0 4 1  1 4 5  1 5 2  2 5 3  2 3 0"/>
  </asset>
  <worldbody>
    <light pos="0 0 12" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="0 13 13" euler="2.35 0 -3.14" mode="fixed" />
    {arena_v} {r_v} {b1_v} {b2_v} {s_v} {h1_v} {h2_v}
  </worldbody>
  <equality> {eq_l1_v} {eq_l2_v} {eq_lr_v} {''.join(eq_g_list)} </equality>
  <actuator> {ac_s_v} {ac_h1_v} {ac_h2_v} </actuator>
</mujoco>
"""
        return full_v

    def _xml_static_scene(self):
        """アリーナの外壁と迷路内壁を生成します。"""
        s = self.ARENA_HALF
        f_s = " ".join(str(v) for v in self.FLOOR_FRICTION)
        return f"""
    <geom name="floor" type="plane" size="{s} {s} 0.1" material="grid_mat" 
          friction="{f_s}" solref="0.02 1"/>
    <geom name="wall_n" type="box" size="{s+0.15} 0.1 2.0" pos="0 6.1 2.0" 
          rgba="0.65 0.65 0.65 0.35"/>
    <geom name="wall_s" type="box" size="{s+0.15} 0.1 2.0" pos="0 -6.1 2.0" 
          rgba="0.65 0.65 0.65 0.35"/>
    <geom name="wall_e" type="box" size="0.1 {s} 2.0" pos="6.1 0 2.0" 
          rgba="0.65 0.65 0.65 0.35"/>
    <geom name="wall_w" type="box" size="0.1 {s} 2.0" pos="-6.1 0 2.0" 
          rgba="0.65 0.65 0.65 0.35"/>
    <geom name="maze_w0" type="box" size="1.5 0.2 0.5" pos="3.0 1.5 0.5" 
          rgba="0.0 0.7 0.7 1"/>
    <geom name="maze_w1" type="box" size="1.5 0.2 0.5" pos="-3.0 -1.5 0.5" 
          rgba="0.0 0.7 0.7 1"/>
    <geom name="maze_w2" type="box" size="0.2 1.5 0.5" pos="0.0 -3.0 0.5" 
          rgba="0.0 0.7 0.7 1"/>
    <geom name="maze_w3" type="box" size="0.2 1.5 0.5" pos="0.0 3.0 0.5" 
          rgba="0.0 0.7 0.7 1"/>
"""

    def _xml_ramp(self, i, xy, rot):
        """スロープの XML ボディを生成。"""
        q = _euler_z_to_quat(rot)
        f_s = " ".join(str(v) for v in self.RAMP_SLOPE_FRICTION)
        f_b = " ".join(str(v) for v in self.RAMP_BASE_FRICTION)
        return f"""
    <body name="ramp_body" pos="{xy[0]} {xy[1]} 0.0" quat="{q}">
      <inertial pos="0.30 0 0.33" mass="{self.RAMP_MASS_BASE}" diaginertia="6 6 12"/>
      <joint type="free" name="ramp_joint" damping="{self.RAMP_DAMPING}"/>
      <geom type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0" 
            rgba="0.2 0.85 0.2 0.9"/>
      <geom name="ramp_slope" type="box" size="0.833 0.5 0.02" pos="0 0 0.516" 
            euler="0 -36.87 0" rgba="0.2 0.85 0.2 0.4" friction="{f_s}"/>
      <geom name="ramp_base" type="box" size="0.333 0.5 0.25" pos="0.333 0 0.25" 
            rgba="0.2 0.85 0.2 0.4" friction="{f_b}" mass="{self.RAMP_MASS_BASE}"/>
    </body>
"""

    def _xml_box(self, i, xy, rot):
        """ボックスの XML ボディを生成。"""
        q = _euler_z_to_quat(rot)
        f = " ".join(str(v) for v in self.BOX_FRICTION)
        return f"""
    <body name="box{i}_body" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
      <joint name="box{i}_joint" type="free" damping="{self.BOX_DAMPING}"/>
      <geom name="box{i}_geom" type="box" size="0.6 0.6 0.5" 
            rgba="0.75 0.55 0.30 1" mass="{self.BOX_MASS}" friction="{f}" 
            solref="0.02 1"/>
    </body>
"""

    def _xml_agent(self, tag, i, xy, rot, color):
        """エージェントを生成。"""
        q = _euler_z_to_quat(rot)
        r, g, b = color
        af = " ".join(str(f) for f in self.AGENT_FRICTION)
        name = "seeker_body" if tag == "seeker" else f"hider{i}_body"
        anchor = f"{tag}{i}_anchor"
        pre = "s" if tag == "seeker" else f"h{i}"
        return f"""
    <body name="{anchor}" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
      <joint name="{pre}_x" type="slide" axis="1 0 0" damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{pre}_y" type="slide" axis="0 1 0" damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{pre}_z" type="slide" axis="0 0 1" limited="true" 
             range="-0.1 1.0" damping="{self.AGENT_DAMPING_Z}"/>
      <joint name="{pre}_rot" type="hinge" axis="0 0 1" 
             damping="{self.AGENT_DAMPING_ROT}" armature="2.0"/>
      <body name="{name}">
        <site name="{pre}_thrust" pos="0 0 0"/>
        <geom name="{pre}_btm" type="sphere" size="0.4" pos="0 0 -0.1" 
              mass="{self.AGENT_MASS_BOTTOM}" friction="{af}"/>
        <geom name="{pre}_capsule" type="capsule" size="0.3 0.3" 
              rgba="{r} {g} {b} 1" mass="{self.AGENT_MASS_BODY}" friction="{af}" 
              contype="0" conaffinity="0"/>
        <geom name="{pre}_nose" type="capsule" fromto="0 0 0.3 0.3 0 0.3" 
              size="0.09" rgba="1 1 1 1" contype="0" conaffinity="0"/>
        <geom name="{pre}_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" 
              size="0.05" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0" />
      </body>
    </body>
"""

    def _xml_actuators(self, tag, i):
        """推力用のアクチュエータを生成。"""
        pre = "s" if tag == "seeker" else f"h{i}"
        return f"""
    <general name="{pre}_fwd" site="{pre}_thrust" gear="1 0 0 0 0 0" 
             gainprm="{self.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{pre}_turn" joint="{pre}_rot" gear="0.5" 
             gainprm="{self.AGENT_ACTUATOR_TURN}" ctrlrange="-1 1"/>
"""

    def _xml_grasp_eq(self, agent_tag, ai, obj_tag, oi):
        """掴み判定用の等価制約を生成。"""
        a_n = "seeker_body" if agent_tag == "seeker" else f"hider{ai}_body"
        o_n = "ramp_body" if obj_tag == "ramp" else f"box{oi}_body"
        p = "s" if agent_tag == "seeker" else f"h{ai}"
        s = "ramp" if obj_tag == "ramp" else f"b{oi}"
        return (f'<weld name="eq_grasp_{p}_{s}" body1="{a_n}" body2="{o_n}" '
                f'active="false" solref="0.06 1" solimp="0.9 0.95 0.001"/>\n')

    def _xml_lock_eq(self, obj_tag, oi):
        """ロック判定用の等価制約を生成。"""
        o_n = "ramp_body" if obj_tag == "ramp" else f"box{oi}_body"
        l = "ramp" if obj_tag == "ramp" else f"b{oi}"
        return (f'<weld name="eq_lock_{l}" body1="world" body2="{o_n}" '
                f'active="false" solref="0.02 1" solimp="0.95 0.99 0.001"/>\n')

    def _analyze_structure(self):
        """モデル内の ID を取得します。"""
        m = self.model
        self.body_ids["s"] = m.body("seeker_body").id
        self.qpos_indices["s"] = {
            'x': m.joint("s_x").id, 'y': m.joint("s_y").id,
            'z': m.joint("s_z").id, 'rot': m.joint("s_rot").id
        }
        self.actuator_ids["s_fwd"] = m.actuator("s_fwd").id
        self.actuator_ids["s_turn"] = m.actuator("s_turn").id

        for i in [1, 2]:
            self.body_ids[f"h{i}"] = m.body(f"hider{i}_body").id
            self.qpos_indices[f"h{i}"] = {
                'x': m.joint(f"h{i}_x").id, 'y': m.joint(f"h{i}_y").id,
                'z': m.joint(f"h{i}_z").id, 'rot': m.joint(f"h{i}_rot").id
            }
            self.actuator_ids[f"h{i}_fwd"] = m.actuator(f"h{i}_fwd").id
            self.actuator_ids[f"h{i}_turn"] = m.actuator(f"h{i}_turn").id

        self.box_ids = [m.body("box1_body").id, m.body("box2_body").id]
        self.ramp_id = m.body("ramp_body").id

        for b_id in self.box_ids + [self.ramp_id]:
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b_id]
            self.obj_geom_ids[b_id] = gs
            self.obj_default_colors[b_id] = m.geom_rgba[gs[0]].copy()

        # 等価制約 ID
        for tk in ["b1", "b2", "ramp"]:
            self.eq_ids[f"lock_{tk}"] = m.equality(f"eq_lock_{tk}").id
            for ak in self.agent_keys:
                self.eq_ids[f"grasp_{ak}_{tk}"] = m.equality(
                    f"eq_grasp_{ak}_{tk}").id

    def _init_agent_intelligence(self):
        """NPC の初期化。"""
        self.npcs = {}
        if self.target != "seeker":
            self.npcs["s"] = RuleBasedSeeker()
        if self.target != "hider":
            self.npcs["h1"] = RuleBasedHider()
        if self.mode != "refinement" or self.target != "hider":
            self.npcs["h2"] = RuleBasedHider()

    def reset(self, seed=None, options=None):
        """環境を初期化。"""
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.model.neq):
            self.data.eq_active[i] = 0
        self._init_agent_intelligence()

        placed = []
        # 配置ロジック
        for name in ["ramp_body", "box1_body", "box2_body"]:
            bid = self.model.body(name).id
            rad = self.R_RAMP if "ramp" in name else self.R_BOX
            for att in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                col = any(abs(p[0]-cx) < (hw+rad+0.2) and
                          abs(p[1]-cy) < (hh+rad+0.2)
                          for (cx, cy, hw, hh) in self.maze_walls)
                if not col:
                    adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
                    self.data.qpos[adr:adr+7] = [p[0], p[1], 0.1, 1, 0, 0, 0]
                    placed.append((p, rad))
                    break

        for ak in self.agent_keys:
            for att in range(500):
                p = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
                col = any(abs(p[0]-cx) < (hw+self.R_AGENT+0.2) and
                          abs(p[1]-cy) < (hh+self.R_AGENT+0.2)
                          for (cx, cy, hw, hh) in self.maze_walls)
                if not col:
                    jx = self.model.jnt_qposadr[self.qpos_indices[ak]['x']]
                    jy = self.model.jnt_qposadr[self.qpos_indices[ak]['y']]
                    jz = self.model.jnt_qposadr[self.qpos_indices[ak]['z']]
                    self.data.qpos[jx] = p[0]
                    self.data.qpos[jy] = p[1]
                    self.data.qpos[jz] = 0.5
                    break

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
                if self.current_step <= self.prep_steps:
                    f, t, l, g = 0.0, 0.0, 0.0, 0.0
                elif self.target == "seeker":
                    f, t, l, g = af[0], af[1], af[2], af[3]
                else:
                    f, t, l, g = self.npcs["s"].get_action(
                        self._get_obs(0), self.idx)
            elif ak == "h1":
                if self.target == "hider":
                    f, t, l, g = af[0], af[1], af[2], af[3]
                else:
                    f, t, l, g = self.npcs["h1"].get_action(
                        self._get_obs(1), self.idx)
            else:
                is_ref = bool(self.mode == "refinement" and
                              self.target == "hider")
                if is_ref:
                    f, t, l, g = af[4], af[5], af[6], af[7]
                else:
                    f, t, l, g = self.npcs["h2"].get_action(
                        self._get_obs(2), self.idx)

            res = self._process_physical_interaction(ak, l, g)
            if res["hold_event"]:
                h_e = True
            if res["lock_event"]:
                l_e = True
            cv[self.actuator_ids[f"{ak}_fwd"]] = f
            cv[self.actuator_ids[f"{ak}_turn"]] = t

        self.data.ctrl[:] = cv
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self._stabilize_interaction_poses()
        self._sync_visual_states()

        rb, find = self._compute_team_reward()
        idx = 0 if self.target == "seeker" else 1
        return (self._normalize_obs(self._get_obs(idx)),
                float(rb if self.target == "hider" else -rb),
                False, self.current_step >= self.max_episode_steps,
                {"is_detected": find, "hold_event": h_e, "lock_event": l_e})

    def _process_physical_interaction(self, ak, lck_v, grp_v):
        """物理的な干渉（ロック・掴み）を処理。"""
        d, m = self.data, self.model
        evt = {"hold_event": False, "lock_event": False}
        ps = d.xpos[self.body_ids[ak]]
        v_x = d.qvel[m.jnt_dofadr[self.qpos_indices[ak]['x']]]
        v_y = d.qvel[m.jnt_dofadr[self.qpos_indices[ak]['y']]]
        vs = math.sqrt(v_x**2 + v_y**2)
        best, b_n, m_d = -1, "", 0.95

        objs = [(self.box_ids[0], "b1"), (self.box_ids[1], "b2"),
                (self.ramp_id, "ramp")]
        for tid, tname in objs:
            dist = np.linalg.norm(d.xpos[tid] - ps)
            if dist < m_d:
                bd = m.jnt_dofadr[m.body_jntadr[tid]]
                vo = math.sqrt(d.qvel[bd]**2 + d.qvel[bd+1]**2)
                if vs < 1.2 and vo < 1.2:
                    m_d, best, b_n = dist, tid, tname

        if best != -1:
            if lck_v > 0.5 and self.prev_action_btns[ak][0] <= 0.5:
                eq = self.eq_ids[f"lock_{b_n}"]
                if d.eq_active[eq] > 0.5:
                    if self.lock_owners[b_n] == ak:
                        d.eq_active[eq], self.lock_owners[b_n] = 0, None
                        evt["lock_event"] = True
                else:
                    qa = m.jnt_qposadr[m.body_jntadr[best]]
                    self.locked_pose[b_n] = d.qpos[qa:qa+7].copy()
                    d.eq_active[eq], self.lock_owners[b_n] = 1, ak
                    evt["lock_event"] = True
            if grp_v > 0.5 and self.prev_action_btns[ak][1] <= 0.5:
                gid = self.eq_ids[f"grasp_{ak}_{b_n}"]
                d.eq_active[gid] = 0.0 if d.eq_active[gid] > 0.5 else 1.0
                evt["hold_event"] = True

        self.prev_action_btns[ak][0] = lck_v
        self.prev_action_btns[ak][1] = grp_v
        return evt

    def _compute_team_reward(self):
        """報酬の計算。"""
        if self.current_step <= self.prep_steps:
            return 0.0, False
        sid = self.body_ids['s']
        sp = self.data.xpos[sid][:2]
        sr = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices['s']['rot']]]
        f1 = self._is_within_fov_and_visible(
            sp, sr, self.data.xpos[self.body_ids['h1']][:2], sid,
            self.body_ids['h1'])
        f2 = self._is_within_fov_and_visible(
            sp, sr, self.data.xpos[self.body_ids['h2']][:2], sid,
            self.body_ids['h2'])
        return (-1.0 if (f1 or f2) else 1.0), bool(f1 or f2)

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        """視野・遮蔽判定。"""
        rel = t_pos - pos
        ds = np.sum(rel**2)
        if ds > 225.0:
            return False
        dist = math.sqrt(ds) + 1e-8
        dot = (math.cos(rot)*(rel[0]/dist) + math.sin(rot)*(rel[1]/dist))
        if dot < 0.38:
            return False
        return self.vis_engine.is_visible(
            pos, t_pos, body_exclude=my_id, target_body_id=t_id)

    def _normalize_obs(self, o):
        """観測の正規化。"""
        v = o.copy()
        v[0:2] /= 10.0
        v[2] /= 5.0
        v[5:17] /= 15.0
        v[17:55] /= 12.0
        return v

    def _get_obs(self, idx):
        """
        観測生成。
        LiDAR オフセットを 0.50 から 0.65 に引き上げ。
        これにより、NPC は物理的に接触するかなり手前から「壁がゼロ距離」と
        判断するようになり、早めの回避行動（減速や旋回）を促します。
        """
        o = np.zeros(55, dtype=np.float32)
        ak, m, d = self.agent_keys[idx], self.model, self.data
        bid, ps = self.body_ids[ak], d.xpos[self.body_ids[ak]]
        rv = float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        vax = m.jnt_dofadr[self.qpos_indices[ak]['x']]
        vay = m.jnt_dofadr[self.qpos_indices[ak]['y']]
        
        o[0] = d.qvel[vax]*math.cos(-rv) - d.qvel[vay]*math.sin(-rv)
        o[1] = d.qvel[vax]*math.sin(-rv) + d.qvel[vay]*math.cos(-rv)
        o[2], o[3], o[4] = rv, math.cos(rv), math.sin(rv)
        
        # オフセットを 0.65 にしてセンサーの感度を上げ、慎重な動きを実現
        o[5:17] = self.vis_engine.cast_lidar(ps[:2], rv, 1, bid) - 0.65
        
        for i, tid in enumerate(self.box_ids + [self.ramp_id]):
            bs, tn = 17+i*8, ['b1', 'b2', 'ramp'][i]
            if self._is_within_fov_and_visible(ps[:2], rv, d.xpos[tid][:2], bid, tid):
                o[bs:bs+2] = d.xpos[tid][:2] - ps[:2]
                o[bs+2:bs+4] = d.cvel[tid][:2]
                o[bs+4:bs+6] = d.xquat[tid][:2]
                o[bs+6], o[bs+7] = 1.0, float(d.eq_active[self.eq_ids[f"lock_{tn}"]] > .5)
            else:
                o[bs:bs+2] = 15.0
                o[bs+2:bs+8] = 0.0

        ens = ["h1", "h2"] if ak == "s" else ["s", "h2"] if ak == "h1" else ["s", "h1"]
        for i, enm in enumerate(ens):
            base, eid = 41+i*7, self.body_ids[enm]
            if self._is_within_fov_and_visible(ps[:2], rv, d.xpos[eid][:2], bid, eid):
                o[base:base+2] = d.xpos[eid][:2] - ps[:2]
                o[base+2:base+4] = d.cvel[eid][:2]
                o[base+4] = d.xquat[eid][0]
                o[base+5] = float(np.linalg.norm(d.cvel[eid][:2]) > 0.1)
                o[base+6] = 1.0
            else:
                o[base:base+2] = 15.0
                o[base+2:base+7] = 0.0
        return o

    def _stabilize_interaction_poses(self):
        """ロック済みポーズの維持。"""
        for tk in ["b1", "b2", "ramp"]:
            if self.data.eq_active[self.eq_ids[f"lock_{tk}"]] > 0.5:
                oid = self.box_ids[0] if tk == "b1" else self.box_ids[1] if tk == "b2" else self.ramp_id
                qa = self.model.jnt_qposadr[self.model.body_jntadr[oid]]
                va = self.model.jnt_dofadr[self.model.body_jntadr[oid]]
                self.data.qpos[qa:qa+7] = self.locked_pose[tk]
                self.data.qvel[va:va+6] = 0.0

    def _sync_visual_states(self):
        """ロック状態の色変更。"""
        objs = [("b1", self.box_ids[0]), ("b2", self.box_ids[1]), ("ramp", self.ramp_id)]
        for tk, bid in objs:
            is_l = bool(self.data.eq_active[self.eq_ids[f"lock_{tk}"]] > 0.5)
            col = [1, 0, 0, 1] if is_l else self.obj_default_colors[bid]
            for g in self.obj_geom_ids[bid]:
                self.model.geom_rgba[g] = col

    def render(self):
        """Viewer 起動。"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
                self.viewer.cam.distance = 18.0
                self.viewer.cam.elevation = -35.0
                self.viewer.cam.azimuth = 90.0
        self.viewer.sync()

    def close(self):
        """終了。"""
        if self.viewer:
            self.viewer.close()