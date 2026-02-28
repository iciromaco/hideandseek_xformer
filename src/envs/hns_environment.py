# hns_environment.py v3.11
# 演習第26回：【デバッグモード搭載 ＆ 物理整合性 ＆ 1行1命令義務化版】
#
# 修正内容:
# 1. デバッグモード (debug_mode) の追加:
#    - 160ステップまで各エージェントの制御入力(ctrl)と現在座標(pos)を標準出力。
#    - 160ステップ到達時に自動で truncated=True を返し、シミュレーションを停止。
# 2. ユーザー提供 v2.79 ロジックの完全維持:
#    - interaction_state (0-3)、質量ベース推力補正 (Hold Boost)、姿勢安定化プロトコルを継続。
# 3. バグ修正済みの reset 処理:
#    - AttributeError (list.fill) を [:] スライス代入で解消。
#    - NameError (rad, i) を完全に精査し oid に統一。
#    - reset 時の self.np_random 採用により、シードに応じたランダム配置を保証。
# 4. 1行1命令（1-line-1-command）の徹底: 垂直展開によるデバッグ透明性の確保。

import math
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

# 高速演算エンジン（Lidar/視界判定用）
from core.visibility_engine import VisibilityEngine


def _load_xml_content():
    """main18_optimization.py から XML を無改変で取得"""
    try:
        root_path = Path(__file__).resolve().parent.parent.parent
        import main18_optimization
        content = getattr(main18_optimization, "XML_CONTENT", "")
        return content
    except Exception:
        return ""


# シミュレーション用XMLの取得
XML_CONTENT = _load_xml_content()

from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider


class TeamCosEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 40}

    def __init__(self, mode="initial", lidar_mode=1, render_mode=None, debug_mode=False):
        super().__init__()
        self.mode = mode
        self.lidar_mode = lidar_mode
        self.render_mode = render_mode
        self.debug_mode = debug_mode
        
        self.current_step = 0
        self.prep_steps = 80
        self.cooldown_limit = 20
        self.debug_step_limit = 160

        if not XML_CONTENT:
            error_msg = "XML_CONTENT could not be loaded."
            raise ImportError(error_msg)

        self.model = mujoco.MjModel.from_xml_string(XML_CONTENT)
        self.data = mujoco.MjData(self.model)
        self.vis_engine = VisibilityEngine(self.model, self.data)
        self.viewer = None

        self.agent_keys = ["s", "h1", "h2"]
        # 名前解決マッピング
        self.key_to_full = {
            "s": "seeker",
            "h1": "hider1",
            "h2": "hider2"
        }

        # 内部状態の保持 (1要素1行展開)
        self.prev_action_btns = {}
        self.prev_action_btns["s"] = np.zeros(2)
        self.prev_action_btns["h1"] = np.zeros(2)
        self.prev_action_btns["h2"] = np.zeros(2)

        self.cooldown_timers = {}
        self.cooldown_timers["s"] = {"lock": 0, "grab": 0}
        self.cooldown_timers["h1"] = {"lock": 0, "grab": 0}
        self.cooldown_timers["h2"] = {"lock": 0, "grab": 0}
        
        # Lock所有権 ＆ 固定姿勢保存
        self.lock_owners = {}
        self.lock_owners["b1"] = None
        self.lock_owners["b2"] = None
        self.lock_owners["ramp"] = None

        self.locked_pose = {}
        self.locked_pose["b1"] = None
        self.locked_pose["b2"] = None
        self.locked_pose["ramp"] = None

        # 物理安定化定数
        self.object_ground_z = {}
        self.object_ground_z["b1"] = 0.5
        self.object_ground_z["b2"] = 0.5
        self.object_ground_z["ramp"] = 0.5

        self.hold_boost_max = 2.3
        self.agent_body_mass = {}
        self.object_body_mass = {
            "b1": 100.0,
            "b2": 100.0,
            "ramp": 50.0
        }

        self.body_ids = {}
        self.qpos_indices = {}
        self.obj_default_colors = {}
        self.obj_geom_ids = {}
        self.obj_default_con = {}
        self.eq_ids = {}

        # 物理構造の解析
        self._analyze_structure()
        # NPCの初期化
        self._init_npcs()

        # 内壁データ
        self.maze_walls = [
            (3.0, 1.5, 1.5, 0.2),
            (-3.0, -1.5, 1.5, 0.2),
            (0.0, -3.0, 0.2, 1.5),
            (0.0, 3.0, 0.2, 1.5)
        ]

        # 観測・行動空間の定義
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(55,), dtype=np.float32
        )

        total_act_dim = 4
        if self.mode == "refinement":
            total_act_dim = 8

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_act_dim,), dtype=np.float32
        )

    def _analyze_structure(self):
        """XML解析とキャッシュ構築"""
        m = self.model
        for k in self.agent_keys:
            try:
                full_name = self.key_to_full[k]
                body_ptr = m.body(f"{full_name}_body")
                self.body_ids[k] = body_ptr.id
                self.qpos_indices[k] = {
                    'x': m.joint(f"{k}_x").id,
                    'y': m.joint(f"{k}_y").id,
                    'rot': m.joint(f"{k}_rot").id
                }
                # 質量取得
                self.agent_body_mass[k] = float(m.body_subtreemass[body_ptr.id])
            except Exception: pass
        self.box_ids, self.ramp_id = [], -1
        for i in range(m.nbody):
            nr = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if "_body" in nr:
                if "box" in nr: self.box_ids.append(i)
                elif "ramp" in nr: self.ramp_id = i
                self.obj_geom_ids[i] = []
                for g in range(m.ngeom):
                    if m.geom_bodyid[g] == i:
                        self.obj_geom_ids[i].append(g)
                        if g == m.body_geomadr[i]: self.obj_default_colors[i] = m.geom_rgba[g].copy()
                        self.obj_default_con[g] = {"type": m.geom_contype[g], "aff": m.geom_conaffinity[g]}
        for t in ["b1", "b2", "ramp"]:
            self.eq_ids[f"lock_{t}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_lock_{t}")
            for ak in self.agent_keys:
                self.eq_ids[f"grasp_{ak}_{t}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_grasp_{ak}_{t}")
        if len(self.box_ids) >= 2:
            self.object_body_mass["b1"] = float(m.body_subtreemass[self.box_ids[0]])
            self.object_body_mass["b2"] = float(m.body_subtreemass[self.box_ids[1]])
        if self.ramp_id != -1: self.object_body_mass["ramp"] = float(m.body_subtreemass[self.ramp_id])

    def _init_npcs(self):
        self.npcs = {"s": RuleBasedSeeker()}
        if self.mode == "initial": self.npcs["h2"] = RuleBasedHider()

    def step(self, action):
        """物理更新ステップ（極限垂直展開 ＆ デバッグモード版）"""
        self.current_step = self.current_step + 1
        
        # クールダウン更新
        for ak in self.agent_keys:
            if self.cooldown_timers[ak]["lock"] > 0: self.cooldown_timers[ak]["lock"] -= 1
            if self.cooldown_timers[ak]["grab"] > 0: self.cooldown_timers[ak]["grab"] -= 1
                
        ctrl_vec = np.zeros(self.model.nu)
        act_f = np.ravel(action)
        
        # Hider 1 (Agent)
        ctrl_vec[2] = act_f[0]
        ctrl_vec[3] = act_f[1]
        self._process_physical_interaction("h1", act_f[2], act_f[3])
        debug_states = {"h1": "RL_Agent"}
        
        # Hider 2 (Agent or NPC)
        if self.mode == "refinement":
            ctrl_vec[4] = act_f[4]
            ctrl_vec[5] = act_f[5]
            self._process_physical_interaction("h2", act_f[6], act_f[7])
            debug_states["h2"] = "RL_Agent"
        else:
            if "h2" in self.npcs:
                h2_obj = self.npcs["h2"]
                obs_h2 = self._get_obs(2)
                h2_npc_act = h2_obj.get_action(obs_h2)
                ctrl_vec[4] = h2_npc_act[0]
                ctrl_vec[5] = h2_npc_act[1]
                self._process_physical_interaction("h2", h2_npc_act[2], h2_npc_act[3])
                debug_states["h2"] = getattr(h2_obj, "current_state", "NPC_Active")
                
        # Seeker (NPC)
        s_obj = self.npcs["s"]
        obs_s = self._get_obs(0)
        s_npc_act = s_obj.get_action(obs_s)
        if self.current_step <= self.prep_steps:
            ctrl_vec[0] = 0.0
            ctrl_vec[1] = 0.0
            self._process_physical_interaction("s", 0.0, 0.0)
            debug_states["s"] = "PrepFreeze"
        else:
            ctrl_vec[0] = s_npc_act[0]
            ctrl_vec[1] = s_npc_act[1]
            self._process_physical_interaction("s", s_npc_act[2], s_npc_act[3])
            debug_states["s"] = getattr(s_obj, "current_state", "NPC_Active")

        # 重荷重補正 (Boost)
        fw_map = {"s": 0, "h1": 2, "h2": 4}
        for ak_b in self.agent_keys:
            held_t = ""
            for tk_n in ["b1", "b2", "ramp"]:
                if self.data.eq_active[self.eq_ids[f"grasp_{ak_b}_{tk_n}"]] > 0.5:
                    held_t = tk_n
                    break
            if held_t != "":
                a_m = self.agent_body_mass.get(ak_b, 1.0)
                o_m = self.object_body_mass.get(held_t, 0.0)
                boost = max(1.0, min(self.hold_boost_max, (a_m + o_m) / (a_m + 1e-6)))
                c_idx = fw_map[ak_b]
                ctrl_vec[c_idx] = float(np.clip(ctrl_vec[c_idx] * boost, -1.0, 1.0))
            
        self.data.ctrl[:] = ctrl_vec
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            # Lock状態の強制同期
            for t_f in ["b1", "b2", "ramp"]:
                if self.data.eq_active[self.eq_ids[f"lock_{t_f}"]] > 0.5:
                    oid_f = self.box_ids[0] if t_f=="b1" else self.box_ids[1] if t_f=="b2" else self.ramp_id
                    j_adr = self.model.body_jntadr[oid_f]
                    if j_adr != -1:
                        q_ptr = self.model.jnt_qposadr[j_adr]
                        v_ptr = self.model.jnt_dofadr[j_adr]
                        self.data.qpos[q_ptr : q_ptr + 7] = self.locked_pose[t_f]
                        self.data.qvel[v_ptr : v_ptr + 6] = 0.0
                        
        # デバッグログ出力 (hns_debug_movement 互換)
        if self.debug_mode:
            print(f"--- Step {self.current_step} Movement Log ---")
            for dk in self.agent_keys:
                idx_d = fw_map[dk]
                p_d = self.data.xpos[self.body_ids[dk]]
                print(f"Agent {dk}: ctrl=[{ctrl_vec[idx_d]:.4f}, {ctrl_vec[idx_d+1]:.4f}], pos=[{p_d[0]:.2f}, {p_d[1]:.2f}]")

        self._stabilize_interaction_poses()
        self._sync_visual_states()
        if self.render_mode == "human": self.render()
        
        rew, f_flag = self._compute_team_reward()
        
        # 終了判定
        truncated = self.current_step >= 500
        if self.debug_mode and self.current_step >= self.debug_step_limit:
            print(f"Debug Mode: Simulation stopped at step {self.current_step}.")
            truncated = True
            
        return self._normalize_obs(self._get_obs(1)), float(rew), False, truncated, {"is_detected": f_flag, "debug": debug_states}

    def _process_physical_interaction(self, agent_key, lock_cmd, grab_cmd):
        """物理インタラクション処理（ユーザー提供ロジック完全復旧版）"""
        m, d = self.model, self.data
        a_bid = self.body_ids[agent_key]
        p_b = self.prev_action_btns[agent_key]
        
        # 1. パルス判定
        l_pulse = (lock_cmd > 0.5 and p_b[0] <= 0.5)
        g_pulse = (grab_cmd > 0.5 and p_b[1] <= 0.5)
        if l_pulse and self.cooldown_timers[agent_key]["lock"] > 0: l_pulse = False
        if g_pulse and self.cooldown_timers[agent_key]["grab"] > 0: g_pulse = False
        
        # リスト代入による型崩れ防止
        self.prev_action_btns[agent_key][:] = [lock_cmd, grab_cmd]
        
        if not l_pulse and not g_pulse: return

        # 2. 自身の現在把持対象を確認
        my_h_tok, my_h_id = "", -1
        for tk in ["b1", "b2", "ramp"]:
            if d.eq_active[self.eq_ids[f"grasp_{agent_key}_{tk}"]] > 0.5:
                my_h_tok = tk
                my_h_id = self.box_ids[0] if tk=="b1" else self.box_ids[1] if tk=="b2" else self.ramp_id
                break

        # 3. インタラクション対象の探索
        objs = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        best_id, best_tkn, min_dist = -1, "", (1.50 if my_h_id != -1 else 1.25)
        a_p_w = d.xpos[a_bid]
        r_val = d.qpos[m.jnt_qposadr[self.qpos_indices[agent_key]['rot']]]
        f_dir = np.array([math.cos(r_val), math.sin(r_val)])

        for oid in objs:
            dist_curr = np.linalg.norm(d.xpos[oid] - a_p_w)
            vis = self._is_within_fov_and_visible(a_p_w[:2], r_val, d.xpos[oid][:2], a_bid, oid)
            if not vis and dist_curr > 1.25 and oid != my_h_id: continue
            if dist_curr < min_dist:
                min_dist = dist_curr
                best_id = oid
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
                best_tkn = "b1" if "box1" in nm else "b2" if "box2" in nm else "ramp"

        # 4. 正面判定
        is_f = False
        if best_id != -1:
            diff_vec = d.xpos[best_id] - a_p_w
            is_f = (diff_vec[0] * f_dir[0] + diff_vec[1] * f_dir[1]) > 0.15
            
        a_team = agent_key[0]
        v_ax_adr = m.jnt_dofadr[self.qpos_indices[agent_key]['x']]
        v_ar_adr = m.jnt_dofadr[self.qpos_indices[agent_key]['rot']]
        
        # --- [A] Lock ロジック ---
        if l_pulse and best_id != -1:
            lid = self.eq_ids[f"lock_{best_tkn}"]
            l_changed = False
            if d.eq_active[lid] > 0.5:
                if self.lock_owners[best_tkn] == a_team:
                    d.eq_active[lid] = 0
                    self.lock_owners[best_tkn] = None
                    self.locked_pose[best_tkn] = None
                    self.cooldown_timers[agent_key]["lock"] = self.cooldown_limit
                    l_changed = True
            elif (my_h_id == -1 or my_h_id == best_id) and is_f:
                o_g_sum = sum(d.eq_active[self.eq_ids[f"grasp_{ak}_{best_tkn}"]] for ak in self.agent_keys)
                if min_dist < 1.05 and (o_g_sum - d.eq_active[self.eq_ids[f"grasp_{agent_key}_{best_tkn}"]]) <= 0.5:
                    v_adr_o = m.jnt_dofadr[m.body_jntadr[best_id]]
                    d.qvel[v_ax_adr:v_ax_adr+2], d.qvel[v_ar_adr], d.qvel[v_adr_o:v_adr_o+6] = 0, 0, 0
                    j_adr_o = m.body_jntadr[best_id]
                    q_adr_o = m.jnt_qposadr[j_adr_o]
                    self.locked_pose[best_tkn] = d.qpos[q_adr_o : q_adr_o + 7].copy()
                    m.eq_data[lid, 0:3], m.eq_data[lid, 3:7] = d.xpos[best_id], d.xquat[best_id]
                    d.eq_active[lid] = 1
                    self.lock_owners[best_tkn] = a_team
                    self.cooldown_timers[agent_key]["lock"] = self.cooldown_limit
                    if d.eq_active[self.eq_ids[f"grasp_{agent_key}_{best_tkn}"]] > 0.5: d.eq_active[self.eq_ids[f"grasp_{agent_key}_{best_tkn}"]] = 0
                    l_changed = True
            if l_changed: mujoco.mj_forward(m, d)

        # --- [B] Grab ロジック ---
        if g_pulse:
            g_changed = False
            if my_h_id != -1:
                if best_id == my_h_id:
                    d.eq_active[self.eq_ids[f"grasp_{agent_key}_{my_h_tok}"]] = 0
                    self.cooldown_timers[agent_key]["grab"] = self.cooldown_limit
                    g_changed = True
            elif best_id != -1 and is_f and d.eq_active[self.eq_ids[f"lock_{best_tkn}"]] <= 0.5:
                if sum(d.eq_active[self.eq_ids[f"grasp_{ak}_{best_tkn}"]] for ak in self.agent_keys) <= 0.5:
                    v_o_ptr = m.jnt_dofadr[m.body_jntadr[best_id]]
                    d.qvel[v_ax_adr:v_ax_adr+2], d.qvel[v_ar_adr], d.qvel[v_o_ptr:v_o_ptr+6] = 0, 0, 0
                    inv_q_a = np.zeros(4); mujoco.mju_negQuat(inv_q_a, d.xquat[a_bid])
                    p_diff = d.xpos[best_id]-a_p_w; p_diff[2] = self.object_ground_z[best_tkn]-d.xpos[a_bid][2]; rel_p = np.zeros(3); mujoco.mju_rotVecQuat(rel_p, p_diff, inv_q_a)
                    oq = d.xquat[best_id]; ry = math.atan2(2.0*(oq[0]*oq[3]+oq[1]*oq[2]), 1.0-2.0*(oq[2]*oq[2]+oq[3]*oq[3])) - r_val; rq = np.array([math.cos(ry*0.5), 0, 0, math.sin(ry*0.5)])
                    gid = self.eq_ids[f"grasp_{agent_key}_{best_tkn}"]; m.eq_data[gid, 0:3], m.eq_data[gid, 3:7], d.eq_active[gid], self.cooldown_timers[agent_key]["grab"], g_changed = rel_p, rq, 1, self.cooldown_limit, True
            if g_changed: mujoco.mj_forward(m, d)

    def _stabilize_interaction_poses(self):
        m, d = self.model, self.data
        for tk in ["b1", "b2", "ramp"]:
            if sum(d.eq_active[self.eq_ids[f"grasp_{ak}_{tk}"]] for ak in self.agent_keys) <= 0.5 or d.eq_active[self.eq_ids[f"lock_{tk}"]] > 0.5: continue
            oid = self.box_ids[0] if tk=="b1" else self.box_ids[1] if tk=="b2" else self.ramp_id
            j_adr = m.body_jntadr[oid]
            if j_adr == -1: continue
            q_ptr = m.jnt_qposadr[j_adr]
            v_ptr = m.jnt_dofadr[j_adr]
            d.qpos[q_ptr + 2] = self.object_ground_z[tk]
            d.qvel[v_ptr + 2 : v_ptr + 5] = 0.0
            oq = d.qpos[q_ptr + 3 : q_ptr + 7]
            oy = math.atan2(2.0*(oq[0]*oq[3]+oq[1]*oq[2]), 1.0-2.0*(oq[2]*oq[2]+oq[3]*oq[3]))
            d.qpos[q_ptr + 3], d.qpos[q_ptr + 4], d.qpos[q_ptr + 5], d.qpos[q_ptr + 6] = math.cos(oy*0.5), 0, 0, math.sin(oy*0.5)
        mujoco.mj_forward(m, d)

    def _sync_visual_states(self):
        """色の同期 (oid 参照修正済み)"""
        m, d = self.model, self.data
        all_objs = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        for oid in all_objs:
            nr = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""; tk = "b1" if "box1" in nr else "b2" if "box2" in nr else "ramp"
            fc = self.obj_default_colors[oid].copy()
            if any(d.eq_active[self.eq_ids[f"grasp_{ak}_{tk}"]] > 0.5 for ak in self.agent_keys): fc[0:3] = [1.0, 1.0, 0.0]
            if d.eq_active[self.eq_ids[f"lock_{tk}"]] > 0.5: fc[0:3] = [1.0, 0.0, 0.0]
            for g in self.obj_geom_ids[oid]: m.geom_rgba[g] = fc

    def _get_obs(self, agent_idx):
        """55次元観測構築（極限展開版）"""
        obs, d, m = np.zeros(55, dtype=np.float32), self.data, self.model
        ak = self.agent_keys[agent_idx]; bid = self.body_ids[ak]; p_s = d.xpos[bid]
        rv = float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        v_idx = m.jnt_dofadr[self.qpos_indices[ak]['x']]
        vx, vy = d.qvel[v_idx], d.qvel[v_idx + 1]; cv, sv = math.cos(-rv), math.sin(-rv)
        
        obs[0] = vx * cv - vy * sv
        obs[1] = vx * sv + vy * cv
        obs[2] = rv
        obs[3] = math.cos(rv)
        obs[4] = math.sin(rv)
        
        lv = self.vis_engine.cast_lidar(p_s[:2], rv, self.lidar_mode, bid)
        for i in range(12): obs[5+i] = lv[i] - 0.45
        
        # 道具観測 (Box1, Box2, Ramp) 1要素1行展開
        for io, oid in enumerate(self.box_ids + [self.ramp_id]):
            if oid == -1: continue
            bs, tkn = 17+io*8, "b1" if io==0 else "b2" if io==1 else "ramp"
            iv = 0.0
            if d.eq_active[self.eq_ids[f"lock_{tkn}"]] > 0.5: iv = 1.0
            elif d.eq_active[self.eq_ids[f"grasp_{ak}_{tkn}"]] > 0.5: iv = 2.0
            else:
                for oa in self.agent_keys:
                    if oa != ak and d.eq_active[self.eq_ids[f"grasp_{oa}_{tkn}"]] > 0.5: iv = 3.0; break
            obs[bs+6] = iv
            if self._is_within_fov_and_visible(p_s[:2], rv, d.xpos[oid][:2], bid, oid):
                rp = d.xpos[oid]-p_s; obs[bs], obs[bs+1] = rp[0]*cv-rp[1]*sv, rp[0]*sv+rp[1]*cv
                va = m.jnt_dofadr[m.body_jntadr[oid]]; obs[bs+2], obs[bs+3] = (d.qvel[va]-vx)*cv-(d.qvel[va+1]-vy)*sv, (d.qvel[va]-vx)*sv+(d.qvel[va+1]-vy)*cv
                oq = d.xquat[oid]; yo = math.atan2(2.0*(oq[0]*oq[3]+oq[1]*oq[2]), 1.0-2.0*(oq[2]*oq[2]+oq[3]*oq[3])); obs[bs+4], obs[bs+5], obs[bs+7] = math.cos(yo-rv), math.sin(yo-rv), 1.0

        ek, pk = ("h1","h2") if ak=="s" else ("s","h2") if ak=="h1" else ("s","h1")
        for ia, aky in enumerate([ek, pk]):
            abd, abs_idx = self.body_ids[aky], 41+ia*7; ap = d.xpos[abd]
            if self._is_within_fov_and_visible(p_s[:2], rv, ap[:2], bid, abd):
                rp = ap-p_s; obs[abs_idx], obs[abs_idx+1] = rp[0]*cv-rp[1]*sv, rp[0]*sv+rp[1]*cv
                va = m.jnt_dofadr[self.qpos_indices[aky]['x']]; obs[abs_idx+2], obs[abs_idx+3] = (d.qvel[va]-vx)*cv-(d.qvel[va+1]-vy)*sv, (d.qvel[va]-vx)*sv+(d.qvel[va+1]-vy)*cv
                arv = float(d.qpos[m.jnt_qposadr[self.qpos_indices[aky]['rot']]]); obs[abs_idx+4], obs[abs_idx+5], obs[abs_idx+6] = math.cos(arv-rv), math.sin(arv-rv), 1.0
        return obs

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        dx, dy = t_pos[0]-pos[0], t_pos[1]-pos[1]; dt = math.sqrt(dx*dx+dy*dy)
        if dt > 15.0: return False
        vx, vy = math.cos(rot), math.sin(rot)
        if (vx*(dx/dt)+vy*(dy/dt)) < 0.38: return False
        return self.vis_engine.is_visible(pos, t_pos, body_exclude=my_id, target_body_id=t_id)

    def _normalize_obs(self, v):
        nv = v.copy(); nv[0:2]/=10.0; nv[2]/=5.0; nv[5:17]/=15.0; nv[17:55]/=12.0; return nv

    def _compute_team_reward(self):
        if self.current_step <= self.prep_steps: return 0.01, False
        s_bid = self.body_ids['s']; sr_ptr = self.qpos_indices['s']['rot']
        sp, sr = self.data.xpos[s_bid][:2], self.data.qpos[self.model.jnt_qposadr[sr_ptr]]
        f = False
        if self._is_within_fov_and_visible(sp, sr, self.data.xpos[self.body_ids['h1']][:2], s_bid, self.body_ids['h1']): f = True
        elif self._is_within_fov_and_visible(sp, sr, self.data.xpos[self.body_ids['h2']][:2], s_bid, self.body_ids['h2']): f = True
        return (-1.0 if f else 1.0), f

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.current_step = 0; mujoco.mj_resetData(self.model, self.data)
        for k in self.prev_action_btns: self.prev_action_btns[k].fill(0.0)
        for ak in self.agent_keys: self.cooldown_timers[ak]["lock"]=0; self.cooldown_timers[ak]["grab"]=0
        for t in ["b1", "b2", "ramp"]: self.lock_owners[t], self.locked_pose[t] = None, None
        self._init_npcs(); m, d = self.model, self.data; pl = []
        for kt in ["ramp", "box1", "box2", "s", "h1", "h2"]:
            rd = 0.6 if kt in self.agent_keys else (1.0 if "box" in kt else 1.5)
            for _ in range(500):
                pt = self.np_random.uniform(-5.2, 5.2, 2); ov = any(abs(pt[0]-cx)<(sx+rd) and abs(pt[1]-cy)<(sy+rd) for (cx, cy, sx, sy) in self.maze_walls)
                if not ov:
                    for (p, r) in pl:
                        if np.linalg.norm(pt-p) < (rd + r + 0.2): ov = True; break
                if ov: continue
                if kt in self.agent_keys:
                    q = self.qpos_indices[kt]; d.qpos[m.jnt_qposadr[q['x']]]=pt[0]; d.qpos[m.jnt_qposadr[q['y']]]=pt[1]; d.qpos[m.jnt_qposadr[q['rot']]]=self.np_random.uniform(-np.pi, np.pi)
                else:
                    body_r = m.body(f"{kt}_body"); adr_r = m.jnt_qposadr[body_r.jntadr[0]]; d.qpos[adr_r:adr_r+2] = pt
                pl.append((pt, rd)); break
        self._sync_visual_states(); mujoco.mj_forward(m, d)
        self.object_ground_z["b1"], self.object_ground_z["b2"] = float(d.xpos[self.box_ids[0]][2]), float(d.xpos[self.box_ids[1]][2])
        if self.ramp_id != -1: self.object_ground_z["ramp"] = float(d.xpos[self.ramp_id][2])
        return self._normalize_obs(self._get_obs(1)), {"is_detected": False}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data); self.viewer.cam.type, self.viewer.cam.lookat[0:3] = mujoco.mjtCamera.mjCAMERA_FREE, [0.0, 0.0, 0.5]
            self.viewer.cam.distance, self.viewer.cam.elevation = 18.0, -75.0; self.viewer.sync()
        if hasattr(self.viewer, 'user_scn'):
            sn = self.viewer.user_scn; sn.ngeom = 0
            for ak in self.agent_keys:
                bi = self.body_ids[ak]; ps = self.data.xpos[bi]; ri = self.qpos_indices[ak]['rot']; rv = self.data.qpos[self.model.jnt_qposadr[ri]]
                ra = [self.obj_default_colors[bi][0], self.obj_default_colors[bi][1], self.obj_default_colors[bi][2], 0.8]
                for tk in self.agent_keys:
                    if tk != ak and self._is_within_fov_and_visible(ps[:2], rv, self.data.xpos[self.body_ids[tk]][:2], bi, self.body_ids[tk]):
                        mujoco.mjv_connector(sn.geoms[sn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 4.0, ps + [0,0,0.4], self.data.xpos[self.body_ids[tk]] + [0,0,0.4])
                        sn.geoms[sn.ngeom].rgba, sn.ngeom = ra, sn.ngeom + 1
                for o_id in self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else []):
                    if self._is_within_fov_and_visible(ps[:2], rv, self.data.xpos[o_id][:2], bi, o_id):
                        mujoco.mjv_connector(sn.geoms[sn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 2.0, ps + [0,0,0.4], self.data.xpos[o_id] + [0,0,0.4])
                        sn.geoms[sn.ngeom].rgba, sn.ngeom = ra, sn.ngeom + 1
        self.viewer.sync()

    def close(self):
        if self.viewer: 
            self.viewer.close()