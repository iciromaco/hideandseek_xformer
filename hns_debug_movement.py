# src/envs/hns_environment.py
# hns_environment.py v4.1 (GitHub オリジナル XML 復元版)
# 修正内容:
# 1. オリジナルの物理定義 (Texture, Material, Friction, Damping, Gear) を復元。
# 2. 構成数に応じてこれらを動的に生成する _build_xml を実装。
# 3. 立ち上がりエッジ検出、垂直展開、安定化処理を動的リストに適用。

import math
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from core.visibility_engine import VisibilityEngine
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider
from core.obs_indices import ObsIdx


def _build_xml(n_seekers, n_hiders, n_boxes, n_ramps):
    """
    GitHub オリジナルの物理設定をベースに XML を動的生成する。
    """
    # オリジナルのアセットと基本設定を維持
    xml = f"""
    <mujoco model="hide_and_seek_classic_restored">
        <compiler angle="radian" coordinate="local" meshdir="assets" />
        <option timestep="0.02" gravity="0 0 -9.81" />
        <statistic center="0 0 0.8" extent="10" />
        <visual>
            <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7" />
            <global azimuth="90" elevation="-35" />
        </visual>
        
        <asset>
            <texture builtin="flat" height="32" name="texplane" rgb1=".2 .3 .4" rgb2=".1 .15 .2" type="2d" width="32"/>
            <material name="matplane" reflectance="0.5" texture="texplane" texrepeat="1 1"/>
        </asset>

        <worldbody>
            <light directional="true" pos="0 0 10" dir="0 0 -1" />
            <camera name="overview" pos="0 13 13" euler="2.35 0 -3.14" mode="fixed" />
            <geom name="floor" size="6 6 0.1" type="plane" material="matplane" friction="1 0.005 0.0001" />
            
            <!-- 外壁 -->
            <geom name="wall_n" type="box" size="6.2 0.2 1.0" pos="0 6.2 1.0" rgba="0.8 0.8 0.8 1" />
            <geom name="wall_s" type="box" size="6.2 0.2 1.0" pos="0 -6.2 1.0" rgba="0.8 0.8 0.8 1" />
            <geom name="wall_e" type="box" size="0.2 6.2 1.0" pos="6.2 0 1.0" rgba="0.8 0.8 0.8 1" />
            <geom name="wall_w" type="box" size="0.2 6.2 1.0" pos="-6.2 0 1.0" rgba="0.8 0.8 0.8 1" />
            
            <!-- オリジナルの迷路壁 -->
            <geom name="maze_1" type="box" size="1.5 0.2 0.8" pos="3 1.5 0.8" rgba="0.7 0.7 0.7 1" />
            <geom name="maze_2" type="box" size="1.5 0.2 0.8" pos="-3 -1.5 0.8" rgba="0.7 0.7 0.7 1" />
            <geom name="maze_3" type="box" size="0.2 1.5 0.8" pos="0 -3 0.8" rgba="0.7 0.7 0.7 1" />
            <geom name="maze_4" type="box" size="0.2 1.5 0.8" pos="0 3 0.8" rgba="0.7 0.7 0.7 1" />
    """

    # Seeker 生成 (オリジナルの damping と sphere 定義)
    for i in range(n_seekers):
        xml += f"""
            <body name="seeker{i}_body" pos="0 0 0.4">
                <joint name="s{i}_x" type="slide" axis="1 0 0" damping="10" frictionloss="0.1" />
                <joint name="s{i}_y" type="slide" axis="0 1 0" damping="10" frictionloss="0.1" />
                <joint name="s{i}_rot" type="hinge" axis="0 0 1" damping="5" frictionloss="0.1" />
                <geom name="seeker{i}_geom" type="sphere" size="0.4" rgba="0 0.2 0.8 1" mass="1" />
            </body>"""

    # Hider 生成
    for i in range(n_hiders):
        xml += f"""
            <body name="hider{i}_body" pos="0 0 0.4">
                <joint name="h{i}_x" type="slide" axis="1 0 0" damping="10" frictionloss="0.1" />
                <joint name="h{i}_y" type="slide" axis="0 1 0" damping="10" frictionloss="0.1" />
                <joint name="h{i}_rot" type="hinge" axis="0 0 1" damping="5" frictionloss="0.1" />
                <geom name="hider{i}_geom" type="sphere" size="0.4" rgba="0.2 0.8 0.2 1" mass="1" />
            </body>"""

    # Box 生成 (オリジナルの摩擦と質量)
    for i in range(n_boxes):
        xml += f"""
            <body name="box{i}_body" pos="0 0 0.5">
                <freejoint name="box{i}_joint" />
                <geom name="box{i}_geom" type="box" size="0.5 0.5 0.5" rgba="0.8 0.8 0.2 1" mass="5" friction="1 0.005 0.0001" />
            </body>"""

    # Ramp 生成
    for i in range(n_ramps):
        xml += f"""
            <body name="ramp{i}_body" pos="0 0 0.5">
                <freejoint name="ramp{i}_joint" />
                <geom name="ramp{i}_geom" type="box" size="0.8 0.5 0.5" rgba="0.8 0.4 0.1 1" mass="8" friction="1 0.005 0.0001" />
            </body>"""

    xml += "\n</worldbody>\n<equality>\n"
    
    # Weld Constraints (Lock & Grasp)
    for i in range(n_boxes): xml += f'<weld name="eq_lock_b{i}" body1="box{i}_body" active="false" />\n'
    for i in range(n_ramps): xml += f'<weld name="eq_lock_r{i}" body1="ramp{i}_body" active="false" />\n'
    
    agents = [f"s{i}" for i in range(n_seekers)] + [f"h{i}" for i in range(n_hiders)]
    for ak in agents:
        b_pfx = "seeker" if ak.startswith("s") else "hider"
        b_num = ak[1:]
        for b in range(n_boxes):
            xml += f'<weld name="eq_grasp_{ak}_b{b}" body1="{b_pfx}{b_num}_body" body2="box{b}_body" active="false" />\n'
        for r in range(n_ramps):
            xml += f'<weld name="eq_grasp_{ak}_r{r}" body1="{b_pfx}{b_num}_body" body2="ramp{r}_body" active="false" />\n'

    xml += "</equality>\n<actuator>\n"
    # オリジナルの Gear 比
    for i in range(n_seekers):
        xml += f'<motor name="s{i}_fwd" joint="s{i}_x" gear="100" />\n<motor name="s{i}_turn" joint="s{i}_rot" gear="50" />\n'
    for i in range(n_hiders):
        xml += f'<motor name="h{i}_fwd" joint="h{i}_x" gear="100" />\n<motor name="h{i}_turn" joint="h{i}_rot" gear="50" />\n'

    xml += "</actuator>\n</mujoco>"
    return xml


class TeamCosEnv(gym.Env):
    def __init__(self, mode="initial", target="hider", 
                 n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1,
                 render_mode=None, action_repeat=16):
        super().__init__()
        self.mode, self.target, self.render_mode = mode, target, render_mode
        self.counts = {"s": n_seekers, "h": n_hiders, "box": n_boxes, "ramp": n_ramps}
        self.current_step, self.prep_steps, self.max_episode_steps = 0, 80, 500
        self.action_repeat = action_repeat

        self.model = mujoco.MjModel.from_xml_string(_build_xml(n_seekers, n_hiders, n_boxes, n_ramps))
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data)
        self.viewer = None

        self.idx = ObsIdx(n_boxes, n_ramps, n_seekers + n_hiders - 1)
        self.agent_keys = [f"s{i}" for i in range(n_seekers)] + [f"h{i}" for i in range(n_hiders)]
        self.prev_action_btns = {ak: np.zeros(2) for ak in self.agent_keys}
        self.lock_owners, self.locked_pose = {}, {}

        self.body_ids, self.qpos_indices, self.actuator_ids = {}, {}, {}
        self.eq_ids, self.obj_geom_ids, self.obj_default_colors = {}, {}, {}

        self._analyze_structure()
        self._init_agent_intelligence()

        self.maze_walls = [(3.0, 1.5, 1.5, 0.2), (-3.0, -1.5, 1.5, 0.2), (0.0, -3.0, 0.2, 1.5), (0.0, 3.0, 0.2, 1.5)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.idx.total_dim,), dtype=np.float32)
        
        act_cnt = 8 if (self.mode == "refinement" and self.target == "hider") else 4
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_cnt,), dtype=np.float32)

    def _analyze_structure(self):
        m = self.model
        for ak in self.agent_keys:
            pfx = "seeker" if ak.startswith("s") else "hider"
            num = ak[1:]
            self.body_ids[ak] = m.body(f"{pfx}{num}_body").id
            self.qpos_indices[ak] = {
                'x': m.joint(f"{ak[0]}{num}_x").id, 
                'y': m.joint(f"{ak[0]}{num}_y").id, 
                'rot': m.joint(f"{ak[0]}{num}_rot").id
            }
            self.actuator_ids[f"{ak}_fwd"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{ak}_fwd")
            self.actuator_ids[f"{ak}_turn"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{ak}_turn")

        self.box_body_ids = [m.body(f"box{i}_body").id for i in range(self.counts["box"])]
        self.ramp_body_ids = [m.body(f"ramp{i}_body").id for i in range(self.counts["ramp"])]
        for bid in self.box_body_ids + self.ramp_body_ids:
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
            self.obj_geom_ids[bid], self.obj_default_colors[bid] = gs, m.geom_rgba[gs[0]].copy()

        for i in range(self.counts["box"]): self.eq_ids[f"lock_b{i}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_lock_b{i}")
        for i in range(self.counts["ramp"]): self.eq_ids[f"lock_r{i}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_lock_r{i}")
        for ak in self.agent_keys:
            for i in range(self.counts["box"]): self.eq_ids[f"grasp_{ak}_b{i}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_grasp_{ak}_b{i}")
            for i in range(self.counts["ramp"]): self.eq_ids[f"grasp_{ak}_r{i}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_grasp_{ak}_r{i}")

    def _init_agent_intelligence(self):
        self.npcs = {}
        for ak in self.agent_keys:
            if ak.startswith("s") and self.target != "seeker": self.npcs[ak] = RuleBasedSeeker()
            if ak.startswith("h") and self.target != "hider": self.npcs[ak] = RuleBasedHider()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        for i in range(self.model.neq): self.data.eq_active[i] = 0
        self._init_agent_intelligence()

        # 配置リトライ (オリジナルの 500 回方式)
        for name in [f"box{i}_body" for i in range(self.counts["box"])] + [f"ramp{i}_body" for i in range(self.counts["ramp"])]:
            for _ in range(500):
                p = np.random.uniform(-5.2, 5.2, 2)
                if not any(abs(p[0]-cx) < sw+1.2 and abs(p[1]-cy) < sh+1.2 for cx, cy, sw, sh in self.maze_walls):
                    bid = self.model.body(name).id
                    j_adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
                    self.data.qpos[j_adr:j_adr+7] = [p[0], p[1], 0.5, 1, 0, 0, 0]
                    break
        for ak in self.agent_keys:
            for _ in range(500):
                p = np.random.uniform(-5.2, 5.2, 2)
                if not any(abs(p[0]-cx) < sw+0.6 and abs(p[1]-cy) < sh+0.6 for cx, cy, sw, sh in self.maze_walls):
                    jx, jy = self.model.jnt_qposadr[self.qpos_indices[ak]['x']], self.model.jnt_qposadr[self.qpos_indices[ak]['y']]
                    self.data.qpos[jx], self.data.qpos[jy] = p[0], p[1]
                    break

        mujoco.mj_forward(self.model, self.data)
        return self._normalize_obs(self._get_obs(0)), {"is_detected": False}

    def _get_obs(self, idx):
        o = np.zeros(self.idx.total_dim, dtype=np.float32)
        ak, m, d = self.agent_keys[idx], self.model, self.data
        bid, ps = self.body_ids[ak], d.xpos[self.body_ids[ak]]
        rv = float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        vx, vy = d.qvel[m.jnt_dofadr[self.qpos_indices[ak]['x']]], d.qvel[m.jnt_dofadr[self.qpos_indices[ak]['y']]]

        # SELF
        o[self.idx.SELF.VEL_X], o[self.idx.SELF.VEL_Y] = vx * math.cos(-rv) - vy * math.sin(-rv), vx * math.sin(-rv) + vy * math.cos(-rv)
        o[self.idx.SELF.ROT], o[self.idx.SELF.COS_ROT], o[self.idx.SELF.SIN_ROT] = rv, math.cos(rv), math.sin(rv)
        o[self.idx.LIDAR] = self.vis_engine.cast_lidar(ps[:2], rv, 1, bid) - 0.45

        # OBJECTS
        for i, sc in enumerate(self.idx.B):
            bid_o = self.box_body_ids[i]
            o[sc.REL_X:sc.REL_X+2] = d.xpos[bid_o][:2] - ps[:2]
            o[sc.IS_LOCKED] = float(d.eq_active[self.eq_ids[f"lock_b{i}"]] > 0.5)
        for i, sc in enumerate(self.idx.RAMP):
            bid_o = self.ramp_body_ids[i]
            o[sc.REL_X:sc.REL_X+2] = d.xpos[bid_o][:2] - ps[:2]
            o[sc.IS_LOCKED] = float(d.eq_active[self.eq_ids[f"lock_r{i}"]] > 0.5)

        # OTHERS
        others = [k for k in self.agent_keys if k != ak]
        for i, sc in enumerate(self.idx.OTHERS):
            enm, eid = others[i], self.body_ids[others[i]]
            o[sc.REL_X:sc.REL_X+2] = d.xpos[eid][:2] - ps[:2]
            o[sc.VISIBLE] = float(self._is_within_fov_and_visible(ps[:2], rv, d.xpos[eid][:2], bid, eid))
        return o

    def _normalize_obs(self, o):
        v = o.copy()
        v[self.idx.SELF.SLICE] /= 5.0
        v[self.idx.LIDAR] /= 15.0
        for sc in self.idx.B + self.idx.RAMP + self.idx.OTHERS: v[sc.SLICE] /= 12.0
        return v

    def step(self, action):
        self.current_step += 1
        af = np.ravel(action)
        cv = np.zeros(self.model.nu)

        for i, ak in enumerate(self.agent_keys):
            if ak in self.npcs: f, t, l, g = self.npcs[ak].get_action(self._get_obs(i), self.idx)
            else: f, t, l, g = af[0], af[1], af[2], af[3]
            res = self._process_physical_interaction(ak, l, g)
            cv[self.actuator_ids[f"{ak}_fwd"]], cv[self.actuator_ids[f"{ak}_turn"]] = f, t

        self.data.ctrl[:] = cv
        for _ in range(self.action_repeat):
            mujoco.mj_step(self.model, self.data)
        self._stabilize_interaction_poses()
        self._sync_visual_states()

        rb, find = self._compute_team_reward()
        final_r = float(rb if self.target == "hider" else -rb)
        return self._normalize_obs(self._get_obs(0)), final_r, False, self.current_step >= self.max_episode_steps, \
               {"is_detected": find, "hold_event": res["hold_event"], "lock_event": res["lock_event"]}

    def _process_physical_interaction(self, ak, lck, grb):
        d, m = self.data, self.model
        event = {"hold_event": False, "lock_event": False}
        if lck <= 0.5 and grb <= 0.5:
            self.prev_action_btns[ak] = np.array([lck, grb])
            return event

        p_s = d.xpos[self.body_ids[ak]]
        v_s = math.sqrt(d.qvel[m.jnt_dofadr[self.qpos_indices[ak]['x']]]**2 + d.qvel[m.jnt_dofadr[self.qpos_indices[ak]['y']]]**2)
        best_id, b_name, m_dist = -1, "", 0.85

        all_objs = [(bid, f"b{i}") for i, bid in enumerate(self.box_body_ids)] + [(bid, f"r{i}") for i, bid in enumerate(self.ramp_body_ids)]
        for bid_o, name in all_objs:
            dist = np.linalg.norm(d.xpos[bid_o] - p_s)
            if dist < m_dist:
                bd = m.jnt_dofadr[m.body_jntadr[bid_o]]
                if v_s < 1.2 and math.sqrt(d.qvel[bd]**2 + d.qvel[bd+1]**2) < 1.2:
                    m_dist, best_id, b_name = dist, bid_o, name

        if best_id != -1:
            if lck > 0.5 and self.prev_action_btns[ak][0] <= 0.5:
                eq_id = self.eq_ids[f"lock_{b_name}"]
                if d.eq_active[eq_id] > 0.5:
                    if self.lock_owners.get(b_name) == ak: d.eq_active[eq_id], self.lock_owners[b_name] = 0, None; event["lock_event"] = True
                else:
                    qa = m.jnt_qposadr[m.body_jntadr[best_id]]
                    self.locked_pose[b_name] = d.qpos[qa:qa+7].copy()
                    d.eq_active[eq_id], self.lock_owners[b_name] = 1, ak; event["lock_event"] = True
            if grb > 0.5 and self.prev_action_btns[ak][1] <= 0.5:
                gid = self.eq_ids[f"grasp_{ak}_{b_name}"]
                d.eq_active[gid] = 0.0 if d.eq_active[gid] > 0.5 else 1.0; event["hold_event"] = True

        self.prev_action_btns[ak] = np.array([lck, grb])
        return event

    def _compute_team_reward(self):
        if self.current_step <= self.prep_steps: return 0.0, False
        any_found = False
        for i in range(self.counts["s"]):
            sid = self.body_ids[f"s{i}"]
            sp, sr = self.data.xpos[sid][:2], self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[f"s{i}"]['rot']]]
            for j in range(self.counts["h"]):
                hid = self.body_ids[f"h{j}"]
                if self._is_within_fov_and_visible(sp, sr, self.data.xpos[hid][:2], sid, hid): any_found = True; break
            if any_found: break
        return (-1.0 if any_found else 1.0), any_found

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        rel = t_pos - pos
        ds = np.sum(rel**2)
        if ds > 225.0: return False
        dist = math.sqrt(ds) + 1e-8
        if (math.cos(rot)*(rel[0]/dist) + math.sin(rot)*(rel[1]/dist)) < 0.38: return False
        return self.vis_engine.is_visible(pos, t_pos, body_exclude=my_id, target_body_id=t_id)

    def _stabilize_interaction_poses(self):
        all_objs = [(bid, f"b{i}") for i, bid in enumerate(self.box_body_ids)] + [(bid, f"r{i}") for i, bid in enumerate(self.ramp_body_ids)]
        for bid, name in all_objs:
            if self.data.eq_active[self.eq_ids[f"lock_{name}"]] > 0.5:
                qa, va = self.model.jnt_qposadr[self.model.body_jntadr[bid]], self.model.jnt_dofadr[self.model.body_jntadr[bid]]
                self.data.qpos[qa:qa+7], self.data.qvel[va:va+6] = self.locked_pose[name], 0.0

    def _sync_visual_states(self):
        all_objs = [(bid, f"b{i}") for i, bid in enumerate(self.box_body_ids)] + [(bid, f"r{i}") for i, bid in enumerate(self.ramp_body_ids)]
        for bid, name in all_objs:
            is_l = self.data.eq_active[self.eq_ids[f"lock_{name}"]] > 0.5
            col = [1, 0, 0, 1] if is_l else self.obj_default_colors[bid]
            for g in self.obj_geom_ids[bid]: self.model.geom_rgba[g] = col

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            cam_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "overview"
            )
            if cam_id >= 0:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self.viewer.cam.fixedcamid = cam_id
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
                self.viewer.cam.distance = 18.0
                self.viewer.cam.elevation = -35.0
                self.viewer.cam.azimuth = 90.0
        self.viewer.sync()

    def close(self):
        if self.viewer: self.viewer.close()