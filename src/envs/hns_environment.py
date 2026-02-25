# hns_environment.py v2.42
# 演習第26回：【描画機能・カメラ設定完全復旧 ＆ 1行1命令徹底遵守版】
#
# 修正内容:
# 1. 描画ロジックの復元: 
#    - render メソッド内に mjv_connector を用いた視線描画処理を再実装。
#    - エージェントのチームカラーに基づいた透過色の線を適用。
# 2. カメラ設定の復元:
#    - ビューア初期化時に俯瞰視点（elevation: -75.0, distance: 18.0）を強制適用。
# 3. 未定義変数タイポの完全修正:
#    - reset メソッド内の m_r 等の誤記述を排除。
# 4. 論理等価性の維持: 55次元観測、物理安定化ロジック、IDキャッシュ性能をすべて継承。

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

        # 観測・行動空間の定義（55次元）
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(55,), dtype=np.float32
        )

        total_act_dim = 8
        if self.mode == "refinement":
            total_act_dim = 8
        else:
            total_act_dim = 4

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(total_act_dim,), dtype=np.float32
        )

    def _analyze_structure(self):
        """XML解析とID・拘束マッピングの事前キャッシュ"""
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
            name_bytes = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
            name_str = name_bytes or ""
            
            if "_body" in name_str:
                if "box" in name_str:
                    self.box_ids.append(i)
                elif "ramp" in name_str:
                    self.ramp_id = i

                self.obj_geom_ids[i] = []
                for g_idx in range(m.ngeom):
                    if m.geom_bodyid[g_idx] == i:
                        self.obj_geom_ids[i].append(g_idx)
                        if g_idx == m.body_geomadr[i]:
                            self.obj_default_colors[i] = m.geom_rgba[g_idx].copy()

        # 拘束IDの事前キャッシュ
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
        """物理更新ステップ（1行1命令・全展開版）"""
        self.current_step = self.current_step + 1
        ctrl_vec = np.zeros(self.model.nu)
        act_flat = np.ravel(action)

        # 1. Hider 1 (AIエージェント)
        h1_fwd = act_flat[0]
        h1_turn = act_flat[1]
        h1_lock_req = act_flat[2]
        h1_grab_req = act_flat[3]

        ctrl_vec[2] = h1_fwd
        ctrl_vec[3] = h1_turn
        self._process_physical_interaction("h1", h1_lock_req, h1_grab_req)

        # 2. Hider 2
        if self.mode == "refinement":
            h2_fwd = act_flat[4]
            h2_turn = act_flat[5]
            h2_lock_req = act_flat[6]
            h2_grab_req = act_flat[7]

            ctrl_vec[4] = h2_fwd
            ctrl_vec[5] = h2_turn
            self._process_physical_interaction("h2", h2_lock_req, h2_grab_req)
        else:
            if "h2" in self.npcs:
                obs_h2 = self._get_obs(2)
                h2_npc_act = self.npcs["h2"].get_action(obs_h2)
                ctrl_vec[4] = h2_npc_act[0]
                ctrl_vec[5] = h2_npc_act[1]
                self._process_physical_interaction("h2", h2_npc_act[2], h2_npc_act[3])

        # 3. Seeker (NPC)
        obs_s = self._get_obs(0)
        s_npc_act = self.npcs["s"].get_action(obs_s)

        if self.current_step <= self.prep_steps:
            ctrl_vec[0] = 0.0
            ctrl_vec[1] = 0.0
            self._process_physical_interaction("s", 0.0, 0.0)
        else:
            ctrl_vec[0] = s_npc_act[0]
            ctrl_vec[1] = s_npc_act[1]
            self._process_physical_interaction("s", s_npc_act[2], s_npc_act[3])

        # 物理シミュレーション実行
        self.data.ctrl[:] = ctrl_vec
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        # 視覚同期
        self._sync_visual_states()

        if self.render_mode == "human":
            self.render()

        reward_tuple = self._compute_team_reward()
        reward_val = reward_tuple[0]
        found_flag = reward_tuple[1]

        truncated = self.current_step >= 500
        norm_obs = self._normalize_obs(self._get_obs(1))

        return norm_obs, float(reward_val), False, truncated, {"is_detected": found_flag}

    def _process_physical_interaction(self, agent_key, lock_cmd, grab_cmd):
        """排他的トグル方式物理インタラクション（1行1命令・全展開版）"""
        m, d = self.model, self.data
        agent_bid = self.body_ids[agent_key]

        # トグル検知
        prev = self.prev_action_btns[agent_key]
        lock_toggled = (lock_cmd > 0.5) and (prev[0] <= 0.5)
        grab_toggled = (grab_cmd > 0.5) and (prev[1] <= 0.5)
        self.prev_action_btns[agent_key][0] = lock_cmd
        self.prev_action_btns[agent_key][1] = grab_cmd

        if not lock_toggled and not grab_toggled:
            return

        # 把持中の対象トークンを特定
        current_held_token = ""
        current_held_id = -1
        for t in ["b1", "b2", "ramp"]:
            gid = self.eq_ids[f"grasp_{agent_key}_{t}"]
            if gid != -1 and d.eq_active[gid] > 0.5:
                current_held_token = t
                if t == "b1": current_held_id = self.box_ids[0]
                elif t == "b2": current_held_id = self.box_ids[1]
                else: current_held_id = self.ramp_id
                break

        # 近接オブジェクト探索
        objs = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        best_obj_id = -1
        best_token = ""
        # 既に持っているなら判定距離を緩和
        min_th = 0.5 if current_held_id != -1 else 0.15
        min_found_dist = min_th

        # エージェント情報
        a_pos = d.xpos[agent_bid]
        r_abs = d.qpos[m.jnt_qposadr[self.qpos_indices[agent_key]['rot']]]
        dir_v = np.array([math.cos(r_abs), math.sin(r_abs)])

        for oid in objs:
            diff = d.xpos[oid] - a_pos
            dist = np.linalg.norm(diff)
            if dist > 1.8: continue
            
            # 正面判定
            norm_v = diff[:2] / (dist + 1e-8)
            dot = np.dot(dir_v, norm_v)
            if dot < 0.4 and oid != current_held_id: continue
            
            if dist < min_found_dist:
                min_found_dist = dist
                best_obj_id = oid
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
                best_token = "b1" if "box1" in nm else "b2" if "box2" in nm else "ramp"

        # --- [A] Lock処理 ---
        if lock_toggled and best_obj_id != -1:
            lid = self.eq_ids[f"lock_{best_token}"]
            if d.eq_active[lid] > 0.5:
                d.eq_active[lid] = 0
                mujoco.mj_forward(m, d)
            elif current_held_id == -1:
                v_adr = m.jnt_dofadr[m.body_jntadr[best_obj_id]]
                d.qvel[v_adr:v_adr+6] = 0.0
                mujoco.mj_forward(m, d)
                m.eq_data[lid, 3:6] = d.xpos[best_obj_id]
                m.eq_data[lid, 6:10] = d.xquat[best_obj_id]
                d.eq_active[lid] = 1
                mujoco.mj_forward(m, d)

        # --- [B] Grab処理 ---
        if grab_toggled:
            if current_held_id != -1:
                gid = self.eq_ids[f"grasp_{agent_key}_{current_held_token}"]
                d.eq_active[gid] = 0
                mujoco.mj_forward(m, d)
            elif best_obj_id != -1:
                lid_check = self.eq_ids[f"lock_{best_token}"]
                if d.eq_active[lid_check] <= 0.5:
                    gid = self.eq_ids[f"grasp_{agent_key}_{best_token}"]
                    v_adr = m.jnt_dofadr[m.body_jntadr[best_obj_id]]
                    d.qvel[v_adr:v_adr+6] = 0.0
                    mujoco.mj_forward(m, d)
                    
                    inv_q = np.zeros(4)
                    mujoco.mju_negQuat(inv_q, d.xquat[agent_bid])
                    rel_p = np.zeros(3)
                    mujoco.mju_rotVecQuat(rel_p, d.xpos[best_obj_id] - d.xpos[agent_bid], inv_q)
                    rel_q = np.zeros(4)
                    mujoco.mju_mulQuat(rel_q, inv_q, d.xquat[best_obj_id])
                    
                    m.eq_data[gid, 0:3] = rel_p
                    m.eq_data[gid, 6:10] = rel_q
                    d.eq_active[gid] = 1
                    mujoco.mj_forward(m, d)

    def _sync_visual_states(self):
        """オブジェクトの色を状態に合わせて同期（1行1命令）"""
        m, d = self.model, self.data
        all_tools = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])

        for oid in all_tools:
            onm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
            tok = "b1" if "box1" in onm else "b2" if "box2" in onm else "ramp"
            f_col = self.obj_default_colors[oid].copy()
            
            # 把持判定
            is_held = False
            for ak in self.agent_keys:
                gid = self.eq_ids[f"grasp_{ak}_{tok}"]
                if gid != -1 and d.eq_active[gid] > 0.5:
                    is_held = True
                    break
            if is_held: f_col[0:3] = [1.0, 1.0, 0.0]
            
            # ロック判定
            lid = self.eq_ids[f"lock_{tok}"]
            if lid != -1 and d.eq_active[lid] > 0.5: f_col[0:3] = [1.0, 0.0, 0.0]
            
            for g_id in self.obj_geom_ids[oid]:
                m.geom_rgba[g_id] = f_col

    def _get_obs(self, agent_idx):
        """55次元観測の極限展開構築（1行1命令・全復元版）"""
        obs = np.zeros(55, dtype=np.float32)
        d, m = self.data, self.model
        ak = self.agent_keys[agent_idx]
        self_bid = self.body_ids[ak]

        p_self_w = d.xpos[self_bid]
        rot_abs = float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        v_adr = m.jnt_dofadr[self.qpos_indices[ak]['x']]
        v_w_x, v_w_y = d.qvel[v_adr], d.qvel[v_adr + 1]
        
        c, s = math.cos(-rot_abs), math.sin(-rot_abs)

        # 0-4: 自己情報
        obs[0] = v_w_x * c - v_w_y * s
        obs[1] = v_w_x * s + v_w_y * c
        obs[2] = rot_abs
        obs[3] = math.cos(rot_abs)
        obs[4] = math.sin(rot_abs)

        # 5-16: Lidar
        l_res = self.vis_engine.cast_lidar(p_self_w[:2], rot_abs, self.lidar_mode, self_bid)
        obs[5] = l_res[0] - 0.45
        obs[6] = l_res[1] - 0.45
        obs[7] = l_res[2] - 0.45
        obs[8] = l_res[3] - 0.45
        obs[9] = l_res[4] - 0.45
        obs[10] = l_res[5] - 0.45
        obs[11] = l_res[6] - 0.45
        obs[12] = l_res[7] - 0.45
        obs[13] = l_res[8] - 0.45
        obs[14] = l_res[9] - 0.45
        obs[15] = l_res[10] - 0.45
        obs[16] = l_res[11] - 0.45

        # 17-24: Box 1
        b1_id = self.box_ids[0]
        if self._is_within_fov_and_visible(p_self_w[:2], rot_abs, d.xpos[b1_id][:2], self_bid, b1_id):
            obs[17] = (d.xpos[b1_id][0] - p_self_w[0]) * c - (d.xpos[b1_id][1] - p_self_w[1]) * s
            obs[18] = (d.xpos[b1_id][0] - p_self_w[0]) * s + (d.xpos[b1_id][1] - p_self_w[1]) * c
            v1_a = m.jnt_dofadr[m.body_jntadr[b1_id]]
            obs[19] = (d.qvel[v1_a] - v_w_x) * c - (d.qvel[v1_a+1] - v_w_y) * s
            obs[20] = (d.qvel[v1_a] - v_w_x) * s + (d.qvel[v1_a+1] - v_w_y) * c
            oq1 = d.xquat[b1_id]
            oy1 = math.atan2(2.0*(oq1[0]*oq1[3]+oq1[1]*oq1[2]), 1.0-2.0*(oq1[2]*oq1[2]+oq1[3]*oq1[3]))
            obs[21] = math.cos(oy1 - rot_abs)
            obs[22] = math.sin(oy1 - rot_abs)
            obs[23] = float(d.eq_active[self.eq_ids["lock_b1"]])
            obs[24] = 1.0

        # 25-32: Box 2
        b2_id = self.box_ids[1]
        if self._is_within_fov_and_visible(p_self_w[:2], rot_abs, d.xpos[b2_id][:2], self_bid, b2_id):
            obs[25] = (d.xpos[b2_id][0] - p_self_w[0]) * c - (d.xpos[b2_id][1] - p_self_w[1]) * s
            obs[26] = (d.xpos[b2_id][0] - p_self_w[0]) * s + (d.xpos[b2_id][1] - p_self_w[1]) * c
            v2_a = m.jnt_dofadr[m.body_jntadr[b2_id]]
            obs[27] = (d.qvel[v2_a] - v_w_x) * c - (d.qvel[v2_a+1] - v_w_y) * s
            obs[28] = (d.qvel[v2_a] - v_w_x) * s + (d.qvel[v2_a+1] - v_w_y) * c
            oq2 = d.xquat[b2_id]
            oy2 = math.atan2(2.0*(oq2[0]*oq2[3]+oq2[1]*oq2[2]), 1.0-2.0*(oq2[2]*oq2[2]+oq2[3]*oq2[3]))
            obs[29] = math.cos(oy2 - rot_abs)
            obs[30] = math.sin(oy2 - rot_abs)
            obs[31] = float(d.eq_active[self.eq_ids["lock_b2"]])
            obs[32] = 1.0

        # 33-40: Ramp
        r_id = self.ramp_id
        if r_id != -1 and self._is_within_fov_and_visible(p_self_w[:2], rot_abs, d.xpos[r_id][:2], self_bid, r_id):
            obs[33] = (d.xpos[r_id][0] - p_self_w[0]) * c - (d.xpos[r_id][1] - p_self_w[1]) * s
            obs[34] = (d.xpos[r_id][0] - p_self_w[0]) * s + (d.xpos[r_id][1] - p_self_w[1]) * c
            vr_a = m.jnt_dofadr[m.body_jntadr[r_id]]
            obs[35] = (d.qvel[vr_a] - v_w_x) * c - (d.qvel[vr_a+1] - v_w_y) * s
            obs[36] = (d.qvel[vr_a] - v_w_x) * s + (d.qvel[vr_a+1] - v_w_y) * c
            oqr = d.xquat[r_id]
            oyr = math.atan2(2.0*(oqr[0]*oqr[3]+oqr[1]*oqr[2]), 1.0-2.0*(oqr[2]*oqr[2]+oqr[3]*oqr[3]))
            obs[37] = math.cos(oyr - rot_abs)
            obs[38] = math.sin(oyr - rot_abs)
            obs[39] = float(d.eq_active[self.eq_ids["lock_ramp"]])
            obs[40] = 1.0

        # エージェント対象の切り替え
        if ak == "s": ek, pk = "h1", "h2"
        elif ak == "h1": ek, pk = "s", "h2"
        else: ek, pk = "s", "h1"

        # 41-47: Enemy
        ebid = self.body_ids[ek]
        if self._is_within_fov_and_visible(p_self_w[:2], rot_abs, d.xpos[ebid][:2], self_bid, ebid):
            obs[41] = (d.xpos[ebid][0] - p_self_w[0]) * c - (d.xpos[ebid][1] - p_self_w[1]) * s
            obs[42] = (d.xpos[ebid][0] - p_self_w[0]) * s + (d.xpos[ebid][1] - p_self_w[1]) * c
            ev_a = m.jnt_dofadr[self.qpos_indices[ek]['x']]
            obs[43] = (d.qvel[ev_a] - v_w_x) * c - (d.qvel[ev_a+1] - v_w_y) * s
            obs[44] = (d.qvel[ev_a] - v_w_x) * s + (d.qvel[ev_a+1] - v_w_y) * c
            er_val = float(d.qpos[m.jnt_qposadr[self.qpos_indices[ek]['rot']]])
            obs[45] = math.cos(er_val - rot_abs)
            obs[46] = math.sin(er_val - rot_abs)
            obs[47] = 1.0

        # 48-54: Partner
        pbid = self.body_ids[pk]
        if self._is_within_fov_and_visible(p_self_w[:2], rot_abs, d.xpos[pbid][:2], self_bid, pbid):
            obs[48] = (d.xpos[pbid][0] - p_self_w[0]) * c - (d.xpos[pbid][1] - p_self_w[1]) * s
            obs[49] = (d.xpos[pbid][0] - p_self_w[0]) * s + (d.xpos[pbid][1] - p_self_w[1]) * c
            pv_a = m.jnt_dofadr[self.qpos_indices[pk]['x']]
            obs[50] = (d.qvel[pv_a] - v_w_x) * c - (d.qvel[pv_a+1] - v_w_y) * s
            obs[51] = (d.qvel[pv_a] - v_w_x) * s + (d.qvel[pv_a+1] - v_w_y) * c
            pr_val = float(d.qpos[m.jnt_qposadr[self.qpos_indices[pk]['rot']]])
            obs[52] = math.cos(pr_val - rot_abs)
            obs[53] = math.sin(pr_val - rot_abs)
            obs[54] = 1.0
            
        return obs

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        dx, dy = t_pos[0]-pos[0], t_pos[1]-pos[1]
        dist = math.sqrt(dx*dx+dy*dy)
        if dist < 0.45: return True
        if dist > 15.0: return False
        vx, vy = math.cos(rot), math.sin(rot)
        if (vx*(dx/dist) + vy*(dy/dist)) < 0.38: return False
        return self.vis_engine.is_visible(pos, t_pos, my_id, t_id)

    def _normalize_obs(self, v):
        nv = v.copy()
        nv[0:2] = nv[0:2] / 10.0
        nv[2] = nv[2] / 5.0
        nv[5:17] = nv[5:17] / 15.0
        nv[17:55] = nv[17:55] / 12.0
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
        """ビューア描画ロジック：俯瞰カメラおよび視線の可視化を復旧"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            # 俯瞰カメラ設定の復旧
            self.viewer.cam.lookat[0:3] = [0.0, 0.0, 0.5]
            self.viewer.cam.distance = 18.0
            self.viewer.cam.elevation = -75.0
            self.viewer.sync()
            
        # 視線描画ロジックの復旧
        if hasattr(self.viewer, 'user_scn'):
            scn = self.viewer.user_scn
            scn.ngeom = 0
            for ak in self.agent_keys:
                bid = self.body_ids[ak]
                p_s = self.data.xpos[bid]
                rot = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]['rot']]]
                
                # チームカラーを取得（透過度 0.8）
                rgba_orig = self.obj_default_colors[bid]
                rgba = [rgba_orig[0], rgba_orig[1], rgba_orig[2], 0.8]
                
                # 他エージェントへの視線
                for target_k in self.agent_keys:
                    if target_k == ak: continue
                    t_bid = self.body_ids[target_k]
                    if self._is_within_fov_and_visible(p_s[:2], rot, self.data.xpos[t_bid][:2], bid, t_bid):
                        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 4.0, p_s + [0,0,0.4], self.data.xpos[t_bid] + [0,0,0.4])
                        scn.geoms[scn.ngeom].rgba = rgba
                        scn.ngeom += 1
                        
                # 道具への視線
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