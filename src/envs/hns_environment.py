# hns_environment.py v2.56
# 演習第26回：【Lockプロトコル厳格化・安全マージンスナップ ＆ 把持弾き解消版】
#
# 修正内容:
# 1. Lock動作の物理的安定化:
#    - Lock実行時、対象物だけでなくエージェント自身の速度(qvel)・加速度(qacc)も零化。
#    - これにより、押し込み中の慣性による異常な反発力（弾き飛ばし）を抑制。
# 2. Lock時の重なり回避スナップ:
#    - オブジェクトとの距離が 1.02m 未満で Lock する場合、Grab 同様に 1.02m 地点へ強制配置。
#    - エージェントとの「初期重なり」がないクリーンな状態で世界固定拘束を開始。
# 3. 拘束整合性の維持: 
#    - weld 拘束のインデックス (0:3, 3:7) および 衝突属性の維持設定を継承。
# 4. 1行1命令・極限展開の厳守: 速度リセット、スナップ、mj_forward の全工程を独立行で記述。

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
        root_str = str(root_path)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        import main18_optimization
        content = getattr(main18_optimization, "XML_CONTENT", "")
        return content
    except Exception as e:
        print(f"Error loading XML: {e}")
        return ""


# シミュレーション用XMLの取得
XML_CONTENT = _load_xml_content()

from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider


class TeamCosEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 40}

    def __init__(self, mode="initial", lidar_mode=1, render_mode=None):
        super().__init__()
        self.mode = mode
        self.lidar_mode = lidar_mode
        self.render_mode = render_mode
        self.current_step = 0
        self.prep_steps = 80

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

        # トグル制御用の前回ボタン入力保持
        self.prev_action_btns = {
            "s": np.zeros(2),
            "h1": np.zeros(2),
            "h2": np.zeros(2)
        }

        self.body_ids = {}
        self.qpos_indices = {}
        self.obj_default_colors = {}
        self.obj_geom_ids = {}
        self.obj_default_con = {} # 衝突設定のキャッシュ
        
        # 拘束(Equality) ID のキャッシュ
        self.eq_ids = {}

        # 物理構造の解析（キャッシュ化）
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
                body_name = f"{full_name}_body"
                body_obj = m.body(body_name)
                self.body_ids[k] = body_obj.id

                self.qpos_indices[k] = {
                    'x': m.joint(f"{k}_x").id,
                    'y': m.joint(f"{k}_y").id,
                    'rot': m.joint(f"{k}_rot").id
                }
            except Exception:
                pass

        self.box_ids = []
        self.ramp_id = -1
        num_bodies = m.nbody
        
        for i in range(num_bodies):
            name_raw = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            name_str = name_raw or ""
            
            if "_body" in name_str:
                if "box" in name_str:
                    self.box_ids.append(i)
                elif "ramp" in name_str:
                    self.ramp_id = i

                self.obj_geom_ids[i] = []
                for g_idx in range(m.ngeom):
                    if m.geom_bodyid[g_idx] == i:
                        self.obj_geom_ids[i].append(g_idx)
                        # 色のキャッシュ
                        if g_idx == m.body_geomadr[i]:
                            self.obj_default_colors[i] = m.geom_rgba[g_idx].copy()
                        # 衝突設定のキャッシュ
                        self.obj_default_con[g_idx] = {
                            "type": m.geom_contype[g_idx],
                            "aff": m.geom_conaffinity[g_idx]
                        }

        # 拘束IDのキャッシュ
        tokens = ["b1", "b2", "ramp"]
        for t in tokens:
            self.eq_ids[f"lock_{t}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_lock_{t}")
            for ak in self.agent_keys:
                self.eq_ids[f"grasp_{ak}_{t}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_grasp_{ak}_{t}")

    def _init_npcs(self):
        """NPC の生成"""
        self.npcs = {}
        self.npcs["s"] = RuleBasedSeeker()
        if self.mode == "initial":
            self.npcs["h2"] = RuleBasedHider()

    def step(self, action):
        """物理更新ステップ"""
        self.current_step = self.current_step + 1
        ctrl_vec = np.zeros(self.model.nu)
        act_flat = np.ravel(action)

        # 1. Hider 1
        h1_fwd = act_flat[0]
        h1_turn = act_flat[1]
        h1_lock = act_flat[2]
        h1_grab = act_flat[3]
        ctrl_vec[2] = h1_fwd
        ctrl_vec[3] = h1_turn
        self._process_physical_interaction("h1", h1_lock, h1_grab)

        # 2. Hider 2
        if self.mode == "refinement":
            h2_fwd = act_flat[4]
            h2_turn = act_flat[5]
            h2_lock = act_flat[6]
            h2_grab = act_flat[7]
            ctrl_vec[4] = h2_fwd
            ctrl_vec[5] = h2_turn
            self._process_physical_interaction("h2", h2_lock, h2_grab)
        else:
            if "h2" in self.npcs:
                obs_h2 = self._get_obs(2)
                h2_act = self.npcs["h2"].get_action(obs_h2)
                ctrl_vec[4] = h2_act[0]
                ctrl_vec[5] = h2_act[1]
                self._process_physical_interaction("h2", h2_act[2], h2_act[3])

        # 3. Seeker
        obs_s = self._get_obs(0)
        s_act = self.npcs["s"].get_action(obs_s)
        if self.current_step <= self.prep_steps:
            ctrl_vec[0] = 0.0
            ctrl_vec[1] = 0.0
            self._process_physical_interaction("s", 0.0, 0.0)
        else:
            ctrl_vec[0] = s_act[0]
            ctrl_vec[1] = s_act[1]
            self._process_physical_interaction("s", s_act[2], s_act[3])

        self.data.ctrl[:] = ctrl_vec
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._sync_visual_states()
        if self.render_mode == "human":
            self.render()

        rew, found = self._compute_team_reward()
        truncated = self.current_step >= 500
        norm_obs = self._normalize_obs(self._get_obs(1))

        return norm_obs, float(rew), False, truncated, {"is_detected": found}

    def _process_physical_interaction(self, agent_key, lock_cmd, grab_cmd):
        """物理インタラクション処理（拘束整合性 ＆ 安全マージン確保版）"""
        m, d = self.model, self.data
        agent_bid = self.body_ids[agent_key]
        prev = self.prev_action_btns[agent_key]
        lock_toggled = (lock_cmd > 0.5) and (prev[0] <= 0.5)
        grab_toggled = (grab_cmd > 0.5) and (prev[1] <= 0.5)
        self.prev_action_btns[agent_key][0] = lock_cmd
        self.prev_action_btns[agent_key][1] = grab_cmd

        if not lock_toggled and not grab_toggled:
            return

        # 把持中の対象特定
        my_held_tok = ""
        my_held_id = -1
        for t in ["b1", "b2", "ramp"]:
            gid = self.eq_ids[f"grasp_{agent_key}_{t}"]
            if gid != -1 and d.eq_active[gid] > 0.5:
                my_held_tok = t
                my_held_id = self.box_ids[0] if t == "b1" else self.box_ids[1] if t == "b2" else self.ramp_id
                break

        objs = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        best_id, best_tok = -1, ""
        interact_th = 1.50 if my_held_id != -1 else 1.15
        min_d = interact_th
        
        a_pos = d.xpos[agent_bid]
        r_abs = d.qpos[m.jnt_qposadr[self.qpos_indices[agent_key]['rot']]]
        dir_v = np.array([math.cos(r_abs), math.sin(r_abs)])

        for oid in objs:
            diff = d.xpos[oid] - a_pos
            dist = np.linalg.norm(diff)
            if dist > 1.8: continue
            norm_v = diff[:2] / (dist + 1e-8)
            if np.dot(dir_v, norm_v) < 0.4 and oid != my_held_id: continue
            if dist < min_d:
                min_d = dist
                best_id = oid
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
                best_tok = "b1" if "box1" in nm else "b2" if "box2" in nm else "ramp"

        # --- [A] Lock処理 ---
        if lock_toggled and best_id != -1:
            lid = self.eq_ids[f"lock_{best_tok}"]
            if d.eq_active[lid] > 0.5:
                d.eq_active[lid] = 0
            elif my_held_id == -1:
                gs = self.eq_ids[f"grasp_s_{best_tok}"]
                gh1 = self.eq_ids[f"grasp_h1_{best_tok}"]
                gh2 = self.eq_ids[f"grasp_h2_{best_tok}"]
                if (d.eq_active[gs] + d.eq_active[gh1] + d.eq_active[gh2]) <= 0.5:
                    
                    # 1. 徹底停止プロトコル (エージェント ＆ オブジェクト)
                    v_ax = m.jnt_dofadr[self.qpos_indices[agent_key]['x']]
                    v_ay = m.jnt_dofadr[self.qpos_indices[agent_key]['y']]
                    v_ar = m.jnt_dofadr[self.qpos_indices[agent_key]['rot']]
                    d.qvel[v_ax], d.qvel[v_ay], d.qvel[v_ar] = 0.0, 0.0, 0.0
                    d.qacc[v_ax], d.qacc[v_ay], d.qacc[v_ar] = 0.0, 0.0, 0.0
                    
                    o_jnt = m.body_jntadr[best_id]
                    v_ob = m.jnt_dofadr[o_jnt]
                    d.qvel[v_ob:v_ob+6], d.qacc[v_ob:v_ob+6] = 0.0, 0.0
                    
                    # 2. 安全マージン確保スナップ (接触状態でのLock弾き防止)
                    if min_d < 1.02:
                        q_o_base_l = m.jnt_qposadr[o_jnt]
                        d.qpos[q_o_base_l + 0] = d.xpos[agent_bid][0] + dir_v[0] * 1.02
                        d.qpos[q_o_base_l + 1] = d.xpos[agent_bid][1] + dir_v[1] * 1.02
                    
                    # 3. 定着 ＆ 再停止
                    mujoco.mj_forward(m, d)
                    d.qvel[v_ax:v_ax+2], d.qvel[v_ar] = 0.0, 0.0
                    d.qvel[v_ob:v_ob+6] = 0.0
                    
                    # 4. 拘束設定 (World local)
                    m.eq_data[lid, 0:3] = d.xpos[best_id]
                    m.eq_data[lid, 3:7] = d.xquat[best_id]
                    d.eq_active[lid] = 1
            mujoco.mj_forward(m, d)

        # --- [B] Grab処理 ---
        if grab_toggled:
            if my_held_id != -1:
                # 解放：Weld拘束をオフにする
                d.eq_active[self.eq_ids[f"grasp_{agent_key}_{my_held_tok}"]] = 0
                mujoco.mj_forward(m, d)
            elif best_id != -1:
                l_chk_id = self.eq_ids[f"lock_{best_tok}"]
                if d.eq_active[l_chk_id] <= 0.5:
                    on_gid = self.eq_ids[f"grasp_{agent_key}_{best_tok}"]
                    
                    # 1. 徹底停止プロトコル (衝撃排除)
                    v_ax = m.jnt_dofadr[self.qpos_indices[agent_key]['x']]
                    v_ay = m.jnt_dofadr[self.qpos_indices[agent_key]['y']]
                    v_ar = m.jnt_dofadr[self.qpos_indices[agent_key]['rot']]
                    d.qvel[v_ax], d.qvel[v_ay], d.qvel[v_ar] = 0.0, 0.0, 0.0
                    d.qacc[v_ax], d.qacc[v_ay], d.qacc[v_ar] = 0.0, 0.0, 0.0
                    
                    o_jnt = m.body_jntadr[best_id]
                    v_ob = m.jnt_dofadr[o_jnt]
                    d.qvel[v_ob:v_ob+6], d.qacc[v_ob:v_ob+6] = 0.0, 0.0
                    
                    # 2. 正面物理配置 (安全マージン 1.02m)
                    q_o_base = m.jnt_qposadr[o_jnt]
                    d.qpos[q_o_base + 0] = d.xpos[agent_bid][0] + dir_v[0] * 1.02
                    d.qpos[q_o_base + 1] = d.xpos[agent_bid][1] + dir_v[1] * 1.02
                    
                    # 3. 定着 ＆ 再停止
                    mujoco.mj_forward(m, d)
                    d.qvel[v_ax:v_ax+2], d.qvel[v_ar] = 0.0, 0.0
                    d.qvel[v_ob:v_ob+6] = 0.0
                    
                    # 4. 拘束パラメータの設定 (Body1 local)
                    inv_q_agent = np.zeros(4)
                    mujoco.mju_negQuat(inv_q_agent, d.xquat[agent_bid])
                    
                    # 相対位置（スナップ後の 1.02m に基づく算出）
                    p_diff_actual = d.xpos[best_id] - d.xpos[agent_bid]
                    rel_p_theory = np.zeros(3)
                    mujoco.mju_rotVecQuat(rel_p_theory, p_diff_actual, inv_q_agent)
                    
                    # 相対回転 (Body1 local)
                    rel_q_theory = np.zeros(4)
                    mujoco.mju_mulQuat(rel_q_theory, inv_q_agent, d.xquat[best_id])
                    
                    # 5. 拘束有効化 (修正インデックス 0:3, 3:7)
                    m.eq_data[on_gid, 0:3] = rel_p_theory 
                    m.eq_data[on_gid, 3:7] = rel_q_theory 
                    d.eq_active[on_gid] = 1
                    
                    # 6. 最終同期
                    mujoco.mj_forward(m, d)
                    d.qvel[v_ax:v_ax+2], d.qvel[v_ar] = 0.0, 0.0
                    d.qvel[v_ob:v_ob+6] = 0.0

    def _sync_visual_states(self):
        """色の同期"""
        m, d = self.model, self.data
        all_tools = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        for oid in all_tools:
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
            tok = "b1" if "box1" in nm else "b2" if "box2" in nm else "ramp"
            f_col = self.obj_default_colors[oid].copy()
            is_held = any(d.eq_active[self.eq_ids[f"grasp_{ak}_{tok}"]] > 0.5 for ak in self.agent_keys)
            if is_held: f_col[0:3] = [1.0, 1.0, 0.0]
            lid = self.eq_ids[f"lock_{tok}"]
            if lid != -1 and d.eq_active[lid] > 0.5: f_col[0:3] = [1.0, 0.0, 0.0]
            for g_id in self.obj_geom_ids[oid]:
                m.geom_rgba[g_id] = f_col

    def _get_obs(self, agent_idx):
        """55次元観測の構築"""
        obs = np.zeros(55, dtype=np.float32)
        d, m = self.data, self.model
        ak = self.agent_keys[agent_idx]
        self_bid = self.body_ids[ak]
        p_self = d.xpos[self_bid]
        rot = float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        v_adr = m.jnt_dofadr[self.qpos_indices[ak]['x']]
        vx_w, vy_w = d.qvel[v_adr], d.qvel[v_adr + 1]
        c, s = math.cos(-rot), math.sin(-rot)

        obs[0] = vx_w * c - vy_w * s
        obs[1] = vx_w * s + vy_w * c
        obs[2] = rot
        obs[3] = math.cos(rot)
        obs[4] = math.sin(rot)
        obs[5:17] = self.vis_engine.cast_lidar(p_self[:2], rot, self.lidar_mode, self_bid) - 0.45

        for i, oid in enumerate(self.box_ids + [self.ramp_id]):
            if oid == -1: continue
            base = 17 + i * 8
            if self._is_within_fov_and_visible(p_self[:2], rot, d.xpos[oid][:2], self_bid, oid):
                rel = d.xpos[oid] - p_self
                obs[base] = rel[0] * c - rel[1] * s
                obs[base+1] = rel[0] * s + rel[1] * c
                v_o_a = m.jnt_dofadr[m.body_jntadr[oid]]
                obs[base+2] = (d.qvel[v_o_a] - vx_w) * c - (d.qvel[v_o_a+1] - vy_w) * s
                obs[base+3] = (d.qvel[v_o_a] - vx_w) * s + (d.qvel[v_o_a+1] - vy_w) * c
                oq = d.xquat[oid]
                oy = math.atan2(2.0*(oq[0]*oq[3]+oq[1]*oq[2]), 1.0-2.0*(oq[2]*oq[2]+oq[3]*oq[3]))
                obs[base+4] = math.cos(oy - rot)
                obs[base+5] = math.sin(oy - rot)
                tok = "b1" if i==0 else "b2" if i==1 else "ramp"
                obs[base+6] = float(d.eq_active[self.eq_ids[f"lock_{tok}"]])
                obs[base+7] = 1.0

        if ak == "s": ek, pk = "h1", "h2"
        elif ak == "h1": ek, pk = "s", "h2"
        else: ek, pk = "s", "h1"

        # Enemy
        ebid = self.body_ids[ek]
        if self._is_within_fov_and_visible(p_self[:2], rot, d.xpos[ebid][:2], self_bid, ebid):
            rel_e = d.xpos[ebid] - p_self
            obs[41] = rel_e[0] * c - rel_e[1] * s
            obs[42] = rel_e[0] * s + rel_e[1] * c
            ev_a = m.jnt_dofadr[self.qpos_indices[ek]['x']]
            obs[43] = (d.qvel[ev_a] - vx_w) * c - (d.qvel[ev_a+1] - vy_w) * s
            obs[44] = (d.qvel[ev_a] - vx_w) * s + (d.qvel[ev_a+1] - vy_w) * c
            er = float(d.qpos[m.jnt_qposadr[self.qpos_indices[ek]['rot']]])
            obs[45] = math.cos(er - rot)
            obs[46] = math.sin(er - rot)
            obs[47] = 1.0

        # Partner
        pbid = self.body_ids[pk]
        if self._is_within_fov_and_visible(p_self[:2], rot, d.xpos[pbid][:2], self_bid, pbid):
            rel_p = d.xpos[pbid] - p_self
            obs[48] = rel_p[0] * c - rel_p[1] * s
            obs[49] = rel_p[0] * s + rel_p[1] * c
            pv_a = m.jnt_dofadr[self.qpos_indices[pk]['x']]
            obs[50] = (d.qvel[pv_a] - vx_w) * c - (d.qvel[pv_a+1] - vy_w) * s
            obs[51] = (d.qvel[pv_a] - vx_w) * s + (d.qvel[pv_a+1] - vy_w) * c
            pr = float(d.qpos[m.jnt_qposadr[self.qpos_indices[pk]['rot']]])
            obs[52] = math.cos(pr - rot)
            obs[53] = math.sin(pr - rot)
            obs[54] = 1.0
            
        return obs

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        """2Dレイキャスト判定"""
        dx, dy = t_pos[0]-pos[0], t_pos[1]-pos[1]
        dist = math.sqrt(dx*dx+dy*dy)
        if dist > 15.0: return False
        vx, vy = math.cos(rot), math.sin(rot)
        if (vx*(dx/dist) + vy*(dy/dist)) < 0.38: return False
        return self.vis_engine.is_visible(pos, t_pos, body_exclude=my_id, target_body_id=t_id)

    def _normalize_obs(self, v):
        nv = v.copy()
        nv[0:2] /= 10.0
        nv[2] /= 5.0
        nv[5:17] /= 15.0
        nv[17:55] /= 12.0
        return nv

    def _compute_team_reward(self):
        if self.current_step <= self.prep_steps: return 0.01, False
        s_bid = self.body_ids['s']
        s_pos = self.data.xpos[s_bid][:2]
        s_rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices['s']['rot']]]
        found = any(self._is_within_fov_and_visible(s_pos, s_rot, self.data.xpos[self.body_ids[h]][:2], s_bid, self.body_ids[h]) for h in ["h1","h2"])
        return (-1.0 if found else 1.0), found

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        for k in self.prev_action_btns: self.prev_action_btns[k].fill(0.0)
        self._init_npcs()
        m, d = self.model, self.data
        placed = []
        targets = ["ramp", "box1", "box2", "s", "h1", "h2"]
        for kt in targets:
            rad = 0.6 if kt in self.agent_keys else (1.0 if "box" in kt else 1.5)
            for _ in range(500):
                pt = np.random.uniform(-5.2, 5.2, 2)
                ov = False
                for (cx, cy, sx, sy) in self.maze_walls:
                    if abs(pt[0]-cx)<(sx+rad) and abs(pt[1]-cy)<(sy+rad): ov=True; break
                if ov or any(np.linalg.norm(pt-p)<(rad+r+0.2) for (p,r) in placed): continue
                if kt in self.agent_keys:
                    q = self.qpos_indices[kt]
                    d.qpos[m.jnt_qposadr[q['x']]] = pt[0]
                    d.qpos[m.jnt_qposadr[q['y']]] = pt[1]
                    d.qpos[m.jnt_qposadr[q['rot']]] = np.random.uniform(-np.pi, np.pi)
                else:
                    body = m.body(f"{kt}_body")
                    adr = m.jnt_qposadr[body.jntadr[0]]
                    d.qpos[adr:adr+2] = pt
                placed.append((pt, rad))
                break
        self._sync_visual_states()
        mujoco.mj_forward(m, d)
        return self._normalize_obs(self._get_obs(1)), {"is_detected": False}

    def render(self):
        """ビューア描画"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.lookat[0:3] = [0.0, 0.0, 0.5]
            self.viewer.cam.distance = 18.0
            self.viewer.cam.elevation = -75.0
            self.viewer.sync()
            
        if hasattr(self.viewer, 'user_scn'):
            scn = self.viewer.user_scn
            scn.ngeom = 0
            for ak in self.agent_keys:
                bid = self.body_ids[ak]
                p_s = self.data.xpos[bid]
                rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]['rot']]]
                rgba_orig = self.obj_default_colors[bid]
                rgba = [rgba_orig[0], rgba_orig[1], rgba_orig[2], 0.8]
                
                for target_k in self.agent_keys:
                    if target_k == ak: continue
                    t_bid = self.body_ids[target_k]
                    if self._is_within_fov_and_visible(p_s[:2], rot, self.data.xpos[t_bid][:2], bid, t_bid):
                        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 4.0, p_s + [0,0,0.4], self.data.xpos[t_bid] + [0,0,0.4])
                        scn.geoms[scn.ngeom].rgba = rgba
                        scn.ngeom += 1
                        
                tools = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
                for o_id in tools:
                    if self._is_within_fov_and_visible(p_s[:2], rot, self.data.xpos[o_id][:2], bid, o_id):
                        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 2.0, p_s + [0,0,0.4], self.data.xpos[o_id] + [0,0,0.4])
                        scn.geoms[scn.ngeom].rgba = rgba
                        scn.ngeom += 1
                        
        self.viewer.sync()

    def close(self):
        if self.viewer: 
            self.viewer.close()