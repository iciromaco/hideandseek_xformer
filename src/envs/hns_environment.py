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
    INTERACT_RANGE = 1.95
    BTN_ON = 0.1
    BTN_COOLDOWN = 8
    GRAB_OFFSET = 1.45
    GRAB_FOLLOW_GAIN = 8.0
    GRAB_MAX_SPEED = 2.6

    def __init__(self, mode="initial", target="hider", n_seekers=1,
                 n_hiders=2, n_boxes=2, n_ramps=1, render_mode=None,
                 inference_policies=None):
        super().__init__()
        self.mode, self.target, self.render_mode = mode, target, render_mode
        self.current_step, self.prep_steps, self.max_episode_steps = 0, 80, 500
        self.agent_keys = ["s", "h1", "h2"]
        self.idx = ObsIdx(n_boxes, n_ramps, n_others=2)

        self.model = mujoco.MjModel.from_xml_string(self._build_dynamic_xml())
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data)
        self.viewer = None
        self.inference_policies = inference_policies or {}
        
        self.last_debug_ctrl = {k: (0.0, 0.0) for k in self.agent_keys}
        
        self.body_ids, self.qpos_indices, self.actuator_ids = {}, {}, {}
        self.obj_body_map = {}
        self.obj_geom_ids = {}
        self.obj_default_rgba = {}
        self.maze_walls = [(3, 1.5, 1.5, 0.2), (-3, -1.5, 1.5, 0.2),
                          (0, -3, 0.2, 1.5), (0, 3, 0.2, 1.5)]
        self._analyze_structure()
        self._init_agent_intelligence()
        self._init_interaction_state()
        
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
            <geom name="ramp_geom" type="mesh" mesh="ramp_mesh" rgba="0 1 0 1" friction="0.5 0.1 0.1"/>
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
        self.obj_body_map = {"b1": self.box_ids[0], "b2": self.box_ids[1], "ramp": self.ramp_id}
        self.obj_geom_ids = {
            "b1": [m.geom("box1_geom").id],
            "b2": [m.geom("box2_geom").id],
            "ramp": [m.geom("ramp_geom").id],
        }
        self.obj_default_rgba = {
            k: m.geom_rgba[v[0]].copy() for k, v in self.obj_geom_ids.items()
        }

    def _init_agent_intelligence(self):
        self.npcs = {"s": RuleBasedSeeker(), "h1": RuleBasedHider(), "h2": RuleBasedHider()}

    def _init_interaction_state(self):
        self.object_state = {
            "b1": {"mode": "free", "owner": None, "locked_pose": None},
            "b2": {"mode": "free", "owner": None, "locked_pose": None},
            "ramp": {"mode": "free", "owner": None, "locked_pose": None},
        }
        self.prev_action_btns = {ak: np.zeros(2, dtype=np.float32) for ak in self.agent_keys}
        self.btn_cooldown = {ak: 0 for ak in self.agent_keys}

    def _obj_addr(self, obj_key):
        bid = self.obj_body_map[obj_key]
        jadr = self.model.body_jntadr[bid]
        return self.model.jnt_qposadr[jadr], self.model.jnt_dofadr[jadr]

    def _select_target(self, ak, for_grab=False):
        apos = self.data.xpos[self.body_ids[ak]][:2]
        rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]['rot']]]
        fwd = np.array([math.cos(rot), math.sin(rot)], dtype=np.float32)
        aid = self.body_ids[ak]
        best_key, best_dist = None, 1e9
        for tk, bid in self.obj_body_map.items():
            st = self.object_state[tk]
            if for_grab and st["mode"] == "locked" and st["owner"] != ak:
                continue
            opos = self.data.xpos[bid][:2]
            rel = opos - apos
            dist = float(np.linalg.norm(rel))
            if dist > self.INTERACT_RANGE:
                continue
            if for_grab and float(np.dot(fwd, rel)) <= -0.2:
                continue
            if not self.vis_engine.is_visible(
                apos, opos, body_exclude=aid, target_body_id=bid
            ):
                continue
            if dist < best_dist:
                best_key, best_dist = tk, dist
        return best_key

    def _current_grabbed_by(self, ak):
        for tk, st in self.object_state.items():
            if st["mode"] == "grabbed" and st["owner"] == ak:
                return tk
        return None

    def _toggle_lock(self, ak):
        tk = self._select_target(ak, for_grab=False)
        if tk is None:
            return False
        st = self.object_state[tk]
        qadr, _ = self._obj_addr(tk)
        if st["mode"] == "locked" and st["owner"] == ak:
            st["mode"], st["owner"], st["locked_pose"] = "free", None, None
            return True
        if st["mode"] in ("free", "grabbed") and (st["owner"] is None or st["owner"] == ak):
            st["mode"] = "locked"
            st["owner"] = ak
            st["locked_pose"] = self.data.qpos[qadr:qadr+7].copy()
            return True
        return False

    def _toggle_grab(self, ak):
        cur = self._current_grabbed_by(ak)
        tk = self._select_target(ak, for_grab=True)

        if cur is not None:
            aid = self.body_ids[ak]
            cid = self.obj_body_map[cur]
            apos = self.data.xpos[aid][:2]
            cpos = self.data.xpos[cid][:2]
            if not self.vis_engine.is_visible(
                apos, cpos, body_exclude=aid, target_body_id=cid
            ):
                return False

        if cur is not None and (tk is None or tk == cur):
            self.object_state[cur]["mode"] = "free"
            self.object_state[cur]["owner"] = None
            return True
        if tk is None:
            return False
        st = self.object_state[tk]
        if st["mode"] == "free":
            if cur is not None and cur != tk:
                self.object_state[cur]["mode"] = "free"
                self.object_state[cur]["owner"] = None
            st["mode"] = "grabbed"
            st["owner"] = ak
            return True
        return False

    def _handle_buttons(self, ak, lock_btn, grab_btn):
        prev = self.prev_action_btns[ak]
        lock_edge = lock_btn > self.BTN_ON and prev[0] <= self.BTN_ON
        grab_edge = grab_btn > self.BTN_ON and prev[1] <= self.BTN_ON
        self.prev_action_btns[ak][0] = lock_btn
        self.prev_action_btns[ak][1] = grab_btn
        if self.btn_cooldown[ak] > 0:
            return False, False
        if lock_edge:
            lock_evt = self._toggle_lock(ak)
            self.btn_cooldown[ak] = self.BTN_COOLDOWN
            return lock_evt, False
        if grab_edge:
            grab_evt = self._toggle_grab(ak)
            self.btn_cooldown[ak] = self.BTN_COOLDOWN
            return False, grab_evt
        return False, False

    def _apply_object_constraints(self):
        for ak in self.agent_keys:
            if self.btn_cooldown[ak] > 0:
                self.btn_cooldown[ak] -= 1
        for tk, st in self.object_state.items():
            qadr, vadr = self._obj_addr(tk)
            if st["mode"] == "locked" and st["locked_pose"] is not None:
                self.data.qpos[qadr:qadr+7] = st["locked_pose"]
                self.data.qvel[vadr:vadr+6] = 0.0
            elif st["mode"] == "grabbed" and st["owner"] is not None:
                owner = st["owner"]
                opos = self.data.xpos[self.body_ids[owner]][:2]
                rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[owner]['rot']]]
                target_xy = opos + np.array([math.cos(rot), math.sin(rot)]) * self.GRAB_OFFSET
                cur_xy = self.data.qpos[qadr:qadr+2].copy()
                err_xy = target_xy - cur_xy
                desired_xy = np.clip(
                    err_xy * self.GRAB_FOLLOW_GAIN,
                    -self.GRAB_MAX_SPEED,
                    self.GRAB_MAX_SPEED,
                )
                owner_xy = self.data.qvel[
                    self.model.jnt_dofadr[self.qpos_indices[owner]['x']]:
                    self.model.jnt_dofadr[self.qpos_indices[owner]['x']] + 2
                ]
                self.data.qvel[vadr:vadr+2] = 0.75 * desired_xy + 0.25 * owner_xy
                self.data.qvel[vadr+2] = 0.0
                self.data.qvel[vadr+3:vadr+6] *= 0.6
        for tk, geom_ids in self.obj_geom_ids.items():
            mode = self.object_state[tk]["mode"]
            if mode == "locked":
                rgba = np.array([0.2, 0.2, 0.2, 1.0])
            elif mode == "grabbed":
                rgba = np.array([1.0, 0.85, 0.1, 1.0])
            else:
                rgba = self.obj_default_rgba[tk]
            for gid in geom_ids:
                self.model.geom_rgba[gid] = rgba

    def _policy_action(self, agent_key, norm_obs):
        """推論モデルがあれば優先。失敗時は RuleBased にフォールバック。"""
        policy = self.inference_policies.get(agent_key)
        if policy is not None:
            try:
                pred = policy(norm_obs)
                arr = np.asarray(pred).reshape(-1)
                if arr.size >= 4:
                    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
                if arr.size >= 2:
                    return float(arr[0]), float(arr[1]), 0.0, 0.0
            except Exception:
                pass
        arr = np.asarray(self.npcs[agent_key].get_action(norm_obs, self.idx)).reshape(-1)
        if arr.size >= 4:
            return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        return float(arr[0]), float(arr[1]), 0.0, 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        self._init_agent_intelligence()
        self._init_interaction_state()
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
        any_lock_event = False
        any_grab_event = False
        any_lock_pressed = False
        any_grab_pressed = False
        any_lock_target = False
        any_grab_target = False
        max_lock_btn = 0.0
        max_grab_btn = 0.0
        
        for i, ak in enumerate(self.agent_keys):
            # 【核心】学習対象の判定: seeker なら s, hider なら h1 のみ。
            # h2 は常に NPC (RuleBasedHider) または 推論モデル (将来実装) で制御する。
            if (self.target == "seeker" and ak == "s") or (self.target == "hider" and ak == "h1"):
                if ak == "s" and self.current_step <= self.prep_steps:
                    f, t, lck, grb = 0.0, 0.0, 0.0, 0.0
                else:
                    # 外部アクション（常に4要素）を適用
                    f = af[0] if len(af) > 0 else 0.0
                    t = af[1] if len(af) > 1 else 0.0
                    lck = af[2] if len(af) > 2 else 0.0
                    grb = af[3] if len(af) > 3 else 0.0
            else:
                # それ以外（非ターゲットの Seeker や、Hider2）は内部 NPC が制御
                if ak == "s" and self.current_step <= self.prep_steps:
                    f, t, lck, grb = 0.0, 0.0, 0.0, 0.0
                else:
                    norm_obs = self._normalize_obs(self._get_obs(i))
                    f, t, lck, grb = self._policy_action(ak, norm_obs)

            max_lock_btn = max(max_lock_btn, float(lck))
            max_grab_btn = max(max_grab_btn, float(grb))
            lock_pressed = bool(lck > self.BTN_ON)
            grab_pressed = bool(grb > self.BTN_ON)
            any_lock_pressed = any_lock_pressed or lock_pressed
            any_grab_pressed = any_grab_pressed or grab_pressed
            if lock_pressed and self._select_target(ak, for_grab=False) is not None:
                any_lock_target = True
            if grab_pressed and self._select_target(ak, for_grab=True) is not None:
                any_grab_target = True

            lock_evt, grab_evt = self._handle_buttons(ak, lck, grb)
            any_lock_event = any_lock_event or lock_evt
            any_grab_event = any_grab_event or grab_evt
            
            cv[self.actuator_ids[f"{ak}_fwd"]], cv[self.actuator_ids[f"{ak}_turn"]] = f, t
            self.last_debug_ctrl[ak] = (f, t)
            
        self.data.ctrl[:] = cv
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            self._apply_object_constraints()
        mujoco.mj_forward(self.model, self.data)
        
        rb, find = self._compute_team_reward()
        # 学習対象に合わせて観測を生成
        idx_to_obs = 0 if self.target == "seeker" else 1
        
        return (self._normalize_obs(self._get_obs(idx_to_obs)), 
                float(rb if self.target == "hider" else -rb), 
                False, 
                self.current_step >= self.max_episode_steps, 
                {
                    "is_detected": find,
                    "lock_event": any_lock_event,
                    "grab_event": any_grab_event,
                    "dbg_lock_pressed": any_lock_pressed,
                    "dbg_grab_pressed": any_grab_pressed,
                    "dbg_lock_target": any_lock_target,
                    "dbg_grab_target": any_grab_target,
                    "dbg_lock_btn_max": max_lock_btn,
                    "dbg_grab_btn_max": max_grab_btn,
                })

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
        ignore_body_id = -1
        grabbed_key = self._current_grabbed_by(ak)
        if grabbed_key is not None:
            ignore_body_id = self.obj_body_map[grabbed_key]
        lidar_raw = self.vis_engine.cast_lidar(
            ps[:2], rv, 1, self.body_ids[ak], ignore_body_id
        )
        if grabbed_key is not None:
            carry_rel = self.data.xpos[ignore_body_id][:2] - ps[:2]
            h_cos = math.cos(rv)
            h_sin = math.sin(rv)
            for i in range(lidar_raw.shape[0]):
                vx = self.vis_engine.base_cos[i] * h_cos - self.vis_engine.base_sin[i] * h_sin
                vy = self.vis_engine.base_sin[i] * h_cos + self.vis_engine.base_cos[i] * h_sin
                proj = carry_rel[0] * vx + carry_rel[1] * vy
                if proj > 0.0:
                    lidar_raw[i] = max(0.02, float(lidar_raw[i] - proj))
        o[self.idx.LIDAR] = lidar_raw
        for i, tid in enumerate(self.box_ids):
            b_idx = self.idx.B[i]; d_w = d.xpos[tid][:2] - ps[:2]
            o[b_idx.REL_X] = d_w[0] * cos_r - d_w[1] * sin_r
            o[b_idx.REL_Y] = d_w[0] * sin_r + d_w[1] * cos_r
            b_vadr = m.jnt_dofadr[m.body_jntadr[tid]]
            b_speed = math.sqrt(d.qvel[b_vadr] ** 2 + d.qvel[b_vadr + 1] ** 2)
            o[b_idx.IS_MOVING] = 1.0 if b_speed > 0.05 else 0.0
            o[b_idx.IS_LOCKED] = 1.0 if self.object_state[f"b{i+1}"]["mode"] == "locked" else 0.0
        r_idx = self.idx.RAMP[0]; d_w_r = d.xpos[self.ramp_id][:2] - ps[:2]
        o[r_idx.REL_X] = d_w_r[0] * cos_r - d_w_r[1] * sin_r
        o[r_idx.REL_Y] = d_w_r[0] * sin_r + d_w_r[1] * cos_r
        r_vadr = m.jnt_dofadr[m.body_jntadr[self.ramp_id]]
        r_speed = math.sqrt(d.qvel[r_vadr] ** 2 + d.qvel[r_vadr + 1] ** 2)
        o[r_idx.IS_MOVING] = 1.0 if r_speed > 0.05 else 0.0
        o[r_idx.IS_LOCKED] = 1.0 if self.object_state["ramp"]["mode"] == "locked" else 0.0
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