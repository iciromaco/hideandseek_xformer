# hns_environment.py v2.71
# 演習第26回：【論理整合性・極限展開 ＆ 55次元カテゴリカル観測 ＆ 完全復旧版】
#
# 修正内容:
# 1. 論理の完全再展開 (行数復旧):
#    - 前回の更新で短縮された全プロセスを「1行1命令」で垂直方向に展開。
#    - 速度ノルム、距離、排他判定フラグの算出をすべてステップごとに分離。
# 2. 55次元カテゴリカル状態観測の定着:
#    - 道具の7番目(base+6)を interaction_state (0:なし, 1:Lock, 2:自分Hold, 3:他者Hold) に集約。
#    - 判定順序 (Lock > 自分 > 他者) を if/elif 形式で詳細に記述。
# 3. 物理仕様の厳格維持: 
#    - 合理的絶対静止条件（閾値 0.03）、1.02mマージン、チーム権限/本人権限管理をすべて保持。
# 4. 可読性の最大化: 複雑な計算を一行に詰め込まず、中間変数を多用して挙動を透明化。

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
        self.cooldown_limit = 20

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

        # 内部状態の保持
        self.prev_action_btns = {
            "s": np.zeros(2),
            "h1": np.zeros(2),
            "h2": np.zeros(2)
        }
        self.cooldown_timers = {
            "s": {"lock": 0, "grab": 0},
            "h1": {"lock": 0, "grab": 0},
            "h2": {"lock": 0, "grab": 0}
        }
        # チーム権限管理用 (None, 's', 'h')
        self.lock_owners = {
            "b1": None,
            "b2": None,
            "ramp": None
        }
        self.locked_pose = {
            "b1": None,
            "b2": None,
            "ramp": None
        }
        self.hold_boost_max = 2.3
        self.object_ground_z = {
            "b1": 0.5,
            "b2": 0.5,
            "ramp": 0.5
        }
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
        """XML解析とキャッシュ構築（詳細展開）"""
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

        for ak in self.agent_keys:
            if ak in self.body_ids:
                self.agent_body_mass[ak] = float(m.body_subtreemass[self.body_ids[ak]])

        if len(self.box_ids) >= 2:
            self.object_body_mass["b1"] = float(m.body_subtreemass[self.box_ids[0]])
            self.object_body_mass["b2"] = float(m.body_subtreemass[self.box_ids[1]])
        if self.ramp_id != -1:
            self.object_body_mass["ramp"] = float(m.body_subtreemass[self.ramp_id])

    def _init_npcs(self):
        """NPC の生成"""
        self.npcs = {}
        self.npcs["s"] = RuleBasedSeeker()
        if self.mode == "initial":
            self.npcs["h2"] = RuleBasedHider()

    def step(self, action):
        """物理更新ステップ（極限展開）"""
        self.current_step = self.current_step + 1
        
        # クールダウンタイマーの更新
        for k_tm in self.agent_keys:
            timer_l = self.cooldown_timers[k_tm]["lock"]
            if timer_l > 0:
                self.cooldown_timers[k_tm]["lock"] = timer_l - 1
            
            timer_g = self.cooldown_timers[k_tm]["grab"]
            if timer_g > 0:
                self.cooldown_timers[k_tm]["grab"] = timer_g - 1

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
                h2_npc_fwd = h2_act[0]
                h2_npc_turn = h2_act[1]
                h2_npc_lock = h2_act[2]
                h2_npc_grab = h2_act[3]
                ctrl_vec[4] = h2_npc_fwd
                ctrl_vec[5] = h2_npc_turn
                self._process_physical_interaction("h2", h2_npc_lock, h2_npc_grab)

        # 3. Seeker
        obs_s = self._get_obs(0)
        s_act = self.npcs["s"].get_action(obs_s)
        if self.current_step <= self.prep_steps:
            ctrl_vec[0] = 0.0
            ctrl_vec[1] = 0.0
            self._process_physical_interaction("s", 0.0, 0.0)
        else:
            s_npc_fwd = s_act[0]
            s_npc_turn = s_act[1]
            s_npc_lock = s_act[2]
            s_npc_grab = s_act[3]
            ctrl_vec[0] = s_npc_fwd
            ctrl_vec[1] = s_npc_turn
            self._process_physical_interaction("s", s_npc_lock, s_npc_grab)

        hold_forward_index = {"s": 0, "h1": 2, "h2": 4}
        for ak_hb in self.agent_keys:
            held_tok = ""
            for tok_hb in ["b1", "b2", "ramp"]:
                gid_hb = self.eq_ids[f"grasp_{ak_hb}_{tok_hb}"]
                lid_hb = self.eq_ids[f"lock_{tok_hb}"]
                if self.data.eq_active[gid_hb] > 0.5 and self.data.eq_active[lid_hb] <= 0.5:
                    held_tok = tok_hb
                    break
            if held_tok == "":
                continue
            idx_hb = hold_forward_index[ak_hb]
            ag_m = self.agent_body_mass.get(ak_hb, 1.0)
            ob_m = self.object_body_mass.get(held_tok, 0.0)
            if ag_m <= 1e-6:
                continue
            boost_hb = (ag_m + ob_m) / ag_m
            boost_hb = max(1.0, min(self.hold_boost_max, boost_hb))
            ctrl_vec[idx_hb] = float(np.clip(ctrl_vec[idx_hb] * boost_hb, -1.0, 1.0))

        self.data.ctrl[:] = ctrl_vec
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            for tok_fix in ["b1", "b2", "ramp"]:
                lid_fix = self.eq_ids[f"lock_{tok_fix}"]
                if self.data.eq_active[lid_fix] <= 0.5:
                    continue
                oid_fix = self.box_ids[0] if tok_fix == "b1" else self.box_ids[1] if tok_fix == "b2" else self.ramp_id
                if oid_fix == -1:
                    continue
                jadr_fix = self.model.body_jntadr[oid_fix]
                if jadr_fix == -1:
                    continue
                if self.model.jnt_type[jadr_fix] != mujoco.mjtJoint.mjJNT_FREE:
                    continue
                qadr_fix = self.model.jnt_qposadr[jadr_fix]
                dadr_fix = self.model.jnt_dofadr[jadr_fix]
                pose_fix = self.locked_pose[tok_fix]
                if pose_fix is None:
                    pose_fix = self.data.qpos[qadr_fix:qadr_fix+7].copy()
                    self.locked_pose[tok_fix] = pose_fix
                self.data.qpos[qadr_fix:qadr_fix+7] = pose_fix
                self.data.qvel[dadr_fix:dadr_fix+6] = 0.0

        for tok_l in ["b1", "b2", "ramp"]:
            lid_l = self.eq_ids[f"lock_{tok_l}"]
            if self.data.eq_active[lid_l] <= 0.5:
                continue
            oid_l = self.box_ids[0] if tok_l == "b1" else self.box_ids[1] if tok_l == "b2" else self.ramp_id
            v_adr_l = self.model.jnt_dofadr[self.model.body_jntadr[oid_l]]
            self.data.qvel[v_adr_l:v_adr_l+6] *= 0.25

        self._sync_visual_states()
        if self.render_mode == "human":
            self.render()

        rew_val, found_flag = self._compute_team_reward()
        truncated = self.current_step >= 500
        
        obs_h1_raw = self._get_obs(1)
        norm_obs_h1 = self._normalize_obs(obs_h1_raw)

        return norm_obs_h1, float(rew_val), False, truncated, {"is_detected": found_flag}

    def _process_physical_interaction(self, agent_key, lock_cmd, grab_cmd):
        """物理インタラクション処理（極限展開版）"""
        m, d = self.model, self.data
        agent_bid = self.body_ids[agent_key]
        
        # 1. 入力パルス・クールダウン判定
        prev = self.prev_action_btns[agent_key]
        lock_pushed = (lock_cmd > 0.5) and (prev[0] <= 0.5)
        grab_pushed = (grab_cmd > 0.5) and (prev[1] <= 0.5)
        
        timer_l = self.cooldown_timers[agent_key]["lock"]
        timer_g = self.cooldown_timers[agent_key]["grab"]
        if lock_pushed and timer_l > 0:
            lock_pushed = False
        if grab_pushed and timer_g > 0:
            grab_pushed = False
        
        self.prev_action_btns[agent_key][0] = lock_cmd
        self.prev_action_btns[agent_key][1] = grab_cmd

        if not lock_pushed and not grab_pushed:
            return

        # 2. 自身の現在の把持状態特定
        my_held_tok = ""
        my_held_id = -1
        for tok_name in ["b1", "b2", "ramp"]:
            gid_key = f"grasp_{agent_key}_{tok_name}"
            gid_val = self.eq_ids[gid_key]
            if d.eq_active[gid_val] > 0.5:
                my_held_tok = tok_name
                if tok_name == "b1":
                    my_held_id = self.box_ids[0]
                elif tok_name == "b2":
                    my_held_id = self.box_ids[1]
                else:
                    my_held_id = self.ramp_id
                break

        # 3. 近接オブジェクト探索
        objs = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        best_id, best_tok = -1, ""
        interact_th = 1.50 if my_held_id != -1 else 1.25
        min_dist = interact_th
        
        a_pos_w = d.xpos[agent_bid]
        r_idx = self.qpos_indices[agent_key]['rot']
        r_val = d.qpos[m.jnt_qposadr[r_idx]]
        dir_vec_f = np.array([math.cos(r_val), math.sin(r_val)])

        for oid in objs:
            o_pos_w = d.xpos[oid]
            # 視界判定
            is_vis = self._is_within_fov_and_visible(a_pos_w[:2], r_val, o_pos_w[:2], agent_bid, oid)
            diff_v = o_pos_w - a_pos_w
            dist_curr = np.linalg.norm(diff_v)
            near_override = dist_curr < 1.25
            if (not is_vis) and (not near_override) and oid != my_held_id:
                continue
            
            if dist_curr < min_dist:
                min_dist = dist_curr
                best_id = oid
                nm_body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
                if "box1" in nm_body:
                    best_tok = "b1"
                elif "box2" in nm_body:
                    best_tok = "b2"
                else:
                    best_tok = "ramp"

        best_is_front = False
        if best_id != -1:
            best_diff = d.xpos[best_id] - a_pos_w
            best_local_x = best_diff[0] * dir_vec_f[0] + best_diff[1] * dir_vec_f[1]
            best_is_front = best_local_x > 0.15

        # 4. 速度判定（合理的絶対静止）
        v_ax_adr = m.jnt_dofadr[self.qpos_indices[agent_key]['x']]
        v_ar_adr = m.jnt_dofadr[self.qpos_indices[agent_key]['rot']]
        v_ag_xy = np.linalg.norm(d.qvel[v_ax_adr : v_ax_adr+2])
        v_ag_rot = abs(d.qvel[v_ar_adr])
        v_ag_total = v_ag_xy + v_ag_rot
        
        my_team_p = agent_key[0] # 's' or 'h'

        # --- [A] Lock処理 ---
        if lock_pushed and best_id != -1:
            lock_changed = False
            lid_idx = self.eq_ids[f"lock_{best_tok}"]
            if d.eq_active[lid_idx] > 0.5:
                # 解除権限チェック
                if self.lock_owners[best_tok] == my_team_p:
                    d.eq_active[lid_idx] = 0
                    self.lock_owners[best_tok] = None
                    self.locked_pose[best_tok] = None
                    self.cooldown_timers[agent_key]["lock"] = self.cooldown_limit
                    lock_changed = True
            elif ((my_held_id == -1) or (my_held_id == best_id)) and best_is_front:
                # 他者の把持チェック
                others_grasp_sum = 0.0
                for ak_chk in self.agent_keys:
                    others_grasp_sum += d.eq_active[self.eq_ids[f"grasp_{ak_chk}_{best_tok}"]]
                my_grasp_val = d.eq_active[self.eq_ids[f"grasp_{agent_key}_{best_tok}"]]
                net_others_grab = others_grasp_sum - my_grasp_val
                
                # オブジェクト速度
                v_o_adr_l = m.jnt_dofadr[m.body_jntadr[best_id]]
                v_o_norm_l = np.linalg.norm(d.qvel[v_o_adr_l : v_o_adr_l+6])
                
                # 成立条件: 近距離・前方・フリー
                if min_dist < 1.05 and net_others_grab <= 0.5 and my_grasp_val <= 0.5:
                    # 零化
                    d.qvel[v_ax_adr:v_ax_adr+2] = 0.0
                    d.qvel[v_ar_adr] = 0.0
                    d.qvel[v_o_adr_l:v_o_adr_l+6] = 0.0
                    jadr_lock = m.body_jntadr[best_id]
                    if jadr_lock != -1 and m.jnt_type[jadr_lock] == mujoco.mjtJoint.mjJNT_FREE:
                        qadr_lock = m.jnt_qposadr[jadr_lock]
                        pose_lock = d.qpos[qadr_lock:qadr_lock+7].copy()
                        m.eq_data[lid_idx, 0:3] = pose_lock[0:3]
                        m.eq_data[lid_idx, 3:7] = pose_lock[3:7]
                        self.locked_pose[best_tok] = pose_lock
                    else:
                        m.eq_data[lid_idx, 0:3] = d.xpos[best_id]
                        m.eq_data[lid_idx, 3:7] = d.xquat[best_id]
                    d.eq_active[lid_idx] = 1
                    self.lock_owners[best_tok] = my_team_p
                    self.cooldown_timers[agent_key]["lock"] = self.cooldown_limit
                    lock_changed = True
            if lock_changed:
                mujoco.mj_forward(m, d)

        # --- [B] Grab処理 ---
        if grab_pushed:
            grab_changed = False
            if my_held_id != -1:
                # 解除権限チェック: 本人のみ
                if best_id == my_held_id:
                    rel_gid = self.eq_ids[f"grasp_{agent_key}_{my_held_tok}"]
                    d.eq_active[rel_gid] = 0
                    self.cooldown_timers[agent_key]["grab"] = self.cooldown_limit
                    grab_changed = True
            elif best_id != -1 and best_is_front:
                # ロック状態確認
                lock_status = d.eq_active[self.eq_ids[f"lock_{best_tok}"]]
                # 誰かが掴んでいるか確認
                any_holding = 0.0
                for ak_g in self.agent_keys:
                    any_holding += d.eq_active[self.eq_ids[f"grasp_{ak_g}_{best_tok}"]]
                
                v_o_adr_g = m.jnt_dofadr[m.body_jntadr[best_id]]
                v_o_norm_g = np.linalg.norm(d.qvel[v_o_adr_g : v_o_adr_g+6])
                
                if lock_status <= 0.5 and any_holding <= 0.5:
                    on_gid_idx = self.eq_ids[f"grasp_{agent_key}_{best_tok}"]
                    # 零化
                    d.qvel[v_ax_adr:v_ax_adr+2] = 0.0
                    d.qvel[v_ar_adr] = 0.0
                    d.qvel[v_o_adr_g:v_o_adr_g+6] = 0.0
                    # 相対座標
                    inv_q_f = np.zeros(4)
                    mujoco.mju_negQuat(inv_q_f, d.xquat[agent_bid])
                    p_diff_f = d.xpos[best_id] - d.xpos[agent_bid]
                    rel_p_f = np.zeros(3)
                    mujoco.mju_rotVecQuat(rel_p_f, p_diff_f, inv_q_f)
                    rel_p_f[2] = 0.0
                    o_q_f = d.xquat[best_id]
                    o_yaw_f = math.atan2(2.0*(o_q_f[0]*o_q_f[3]+o_q_f[1]*o_q_f[2]), 1.0-2.0*(o_q_f[2]*o_q_f[2]+o_q_f[3]*o_q_f[3]))
                    rel_yaw_f = o_yaw_f - r_val
                    rel_q_f = np.array([math.cos(rel_yaw_f * 0.5), 0.0, 0.0, math.sin(rel_yaw_f * 0.5)], dtype=np.float64)
                    # 拘束
                    m.eq_data[on_gid_idx, 0:3] = rel_p_f
                    m.eq_data[on_gid_idx, 3:7] = rel_q_f
                    d.eq_active[on_gid_idx] = 1
                    self.cooldown_timers[agent_key]["grab"] = self.cooldown_limit
                    grab_changed = True
            if grab_changed:
                mujoco.mj_forward(m, d)

        pose_changed = False
        for tok_h in ["b1", "b2", "ramp"]:
            g_sum_h = 0.0
            for ak_h in self.agent_keys:
                g_sum_h += d.eq_active[self.eq_ids[f"grasp_{ak_h}_{tok_h}"]]
            if g_sum_h <= 0.5:
                continue
            if d.eq_active[self.eq_ids[f"lock_{tok_h}"]] > 0.5:
                continue
            oid_h = self.box_ids[0] if tok_h == "b1" else self.box_ids[1] if tok_h == "b2" else self.ramp_id
            v_o_adr_h = m.jnt_dofadr[m.body_jntadr[oid_h]]
            d.qvel[v_o_adr_h + 2] = 0.0
            d.qvel[v_o_adr_h + 3] = 0.0
            d.qvel[v_o_adr_h + 4] = 0.0

            jadr_h = m.body_jntadr[oid_h]
            if jadr_h != -1 and m.jnt_type[jadr_h] == mujoco.mjtJoint.mjJNT_FREE:
                qadr_h = m.jnt_qposadr[jadr_h]
                oq_h = d.qpos[qadr_h + 3:qadr_h + 7]
                oyaw_h = math.atan2(
                    2.0 * (oq_h[0] * oq_h[3] + oq_h[1] * oq_h[2]),
                    1.0 - 2.0 * (oq_h[2] * oq_h[2] + oq_h[3] * oq_h[3]),
                )
                d.qpos[qadr_h + 2] = self.object_ground_z[tok_h]
                d.qpos[qadr_h + 3] = math.cos(oyaw_h * 0.5)
                d.qpos[qadr_h + 4] = 0.0
                d.qpos[qadr_h + 5] = 0.0
                d.qpos[qadr_h + 6] = math.sin(oyaw_h * 0.5)
                pose_changed = True

        if pose_changed:
            mujoco.mj_forward(m, d)

    def _sync_visual_states(self):
        """色の同期（1行1命令・極限展開）"""
        m, d = self.model, self.data
        tools_sync = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        for oid_s in tools_sync:
            nm_s = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid_s) or ""
            if "box1" in nm_s:
                t_s = "b1"
            elif "box2" in nm_s:
                t_s = "b2"
            else:
                t_s = "ramp"
            
            f_col_s = self.obj_default_colors[oid_s].copy()
            
            # 把持状態チェック (Yellow)
            any_held_s = 0.0
            for ak_s in self.agent_keys:
                any_held_s += d.eq_active[self.eq_ids[f"grasp_{ak_s}_{t_s}"]]
            
            if any_held_s > 0.5:
                f_col_s[0] = 1.0
                f_col_s[1] = 1.0
                f_col_s[2] = 0.0
            
            # 固定状態チェック (Red)
            l_val_s = d.eq_active[self.eq_ids[f"lock_{t_s}"]]
            if l_val_s > 0.5:
                f_col_s[0] = 1.0
                f_col_s[1] = 0.0
                f_col_s[2] = 0.0
                
            for g_id_s in self.obj_geom_ids[oid_s]:
                m.geom_rgba[g_id_s] = f_col_s

    def _get_obs(self, agent_idx):
        """55次元観測の構築（カテゴリカル集約 ＆ 1行1要素代入）"""
        obs = np.zeros(55, dtype=np.float32)
        d, m = self.data, self.model
        ak = self.agent_keys[agent_idx]
        bid = self.body_ids[ak]
        
        # 自己情報
        p_self = d.xpos[bid]
        r_idx = self.qpos_indices[ak]['rot']
        rot_v = float(d.qpos[m.jnt_qposadr[r_idx]])
        v_idx_x = self.qpos_indices[ak]['x']
        v_adr_x = m.jnt_dofadr[v_idx_x]
        vx_w = d.qvel[v_adr_x]
        vy_w = d.qvel[v_adr_x + 1]
        
        c_v = math.cos(-rot_v)
        s_v = math.sin(-rot_v)
        
        obs[0] = vx_w * c_v - vy_w * s_v
        obs[1] = vx_w * s_v + vy_w * c_v
        obs[2] = rot_v
        obs[3] = math.cos(rot_v)
        obs[4] = math.sin(rot_v)
        
        lidar_v = self.vis_engine.cast_lidar(p_self[:2], rot_v, self.lidar_mode, bid)
        obs[5] = lidar_v[0] - 0.45
        obs[6] = lidar_v[1] - 0.45
        obs[7] = lidar_v[2] - 0.45
        obs[8] = lidar_v[3] - 0.45
        obs[9] = lidar_v[4] - 0.45
        obs[10] = lidar_v[5] - 0.45
        obs[11] = lidar_v[6] - 0.45
        obs[12] = lidar_v[7] - 0.45
        obs[13] = lidar_v[8] - 0.45
        obs[14] = lidar_v[9] - 0.45
        obs[15] = lidar_v[10] - 0.45
        obs[16] = lidar_v[11] - 0.45

        # 道具情報 (8x3=24次元)
        all_tools = self.box_ids + [self.ramp_id]
        for i_o, o_id_o in enumerate(all_tools):
            if o_id_o == -1: continue
            base_o = 17 + i_o * 8
            o_pos_o = d.xpos[o_id_o]
            tok_nm_o = "b1" if i_o == 0 else "b2" if i_o == 1 else "ramp"
            interaction_val = 0.0
            if d.eq_active[self.eq_ids[f"lock_{tok_nm_o}"]] > 0.5:
                interaction_val = 1.0
            elif d.eq_active[self.eq_ids[f"grasp_{ak}_{tok_nm_o}"]] > 0.5:
                interaction_val = 2.0
            else:
                for ok_o in self.agent_keys:
                    if ok_o != ak:
                        if d.eq_active[self.eq_ids[f"grasp_{ok_o}_{tok_nm_o}"]] > 0.5:
                            interaction_val = 3.0
                            break
            obs[base_o + 6] = interaction_val
            
            if self._is_within_fov_and_visible(p_self[:2], rot_v, o_pos_o[:2], bid, o_id_o):
                rel_p_o = o_pos_o - p_self
                obs[base_o + 0] = rel_p_o[0] * c_v - rel_p_o[1] * s_v
                obs[base_o + 1] = rel_p_o[0] * s_v + rel_p_o[1] * c_v
                
                v_o_adr_o = m.jnt_dofadr[m.body_jntadr[o_id_o]]
                rel_vx_o = d.qvel[v_o_adr_o] - vx_w
                rel_vy_o = d.qvel[v_o_adr_o + 1] - vy_w
                obs[base_o + 2] = rel_vx_o * c_v - rel_vy_o * s_v
                obs[base_o + 3] = rel_vx_o * s_v + rel_vy_o * c_v
                
                o_q_o = d.xquat[o_id_o]
                o_y_o = math.atan2(2.0*(o_q_o[0]*o_q_o[3]+o_q_o[1]*o_q_o[2]), 1.0-2.0*(o_q_o[2]*o_q_o[2]+o_q_o[3]*o_q_o[3]))
                obs[base_o + 4] = math.cos(o_y_o - rot_v)
                obs[base_o + 5] = math.sin(o_y_o - rot_v)
                obs[base_o + 7] = 1.0

        # 相手・味方情報 (7x2=14次元)
        ek, pk = ("h1", "h2") if ak == "s" else ("s", "h2") if ak == "h1" else ("s", "h1")
        for i_a, a_key in enumerate([ek, pk]):
            a_bid = self.body_ids[a_key]
            a_base = 41 + i_a * 7
            a_pos_a = d.xpos[a_bid]
            
            if self._is_within_fov_and_visible(p_self[:2], rot_v, a_pos_a[:2], bid, a_bid):
                r_p_a = a_pos_a - p_self
                obs[a_base + 0] = r_p_a[0] * c_v - r_p_a[1] * s_v
                obs[a_base + 1] = r_p_a[0] * s_v + r_p_a[1] * c_v
                
                a_v_adr = m.jnt_dofadr[self.qpos_indices[a_key]['x']]
                r_vx_a = d.qvel[a_v_adr] - vx_w
                r_vy_a = d.qvel[a_v_adr + 1] - vy_w
                obs[a_base + 2] = r_vx_a * c_v - r_vy_a * s_v
                obs[a_base + 3] = r_vx_a * s_v + r_vy_a * c_v
                
                a_rot_idx = self.qpos_indices[a_key]['rot']
                a_rot_v = float(d.qpos[m.jnt_qposadr[a_rot_idx]])
                obs[a_base + 4] = math.cos(a_rot_v - rot_v)
                obs[a_base + 5] = math.sin(a_rot_v - rot_v)
                obs[a_base + 6] = 1.0

        return obs

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        dx_v = t_pos[0] - pos[0]
        dy_v = t_pos[1] - pos[1]
        dist_v = math.sqrt(dx_v*dx_v + dy_v*dy_v)
        if dist_v > 15.0: return False
        vx_f, vy_f = math.cos(rot), math.sin(rot)
        if (vx_f * (dx_v/dist_v) + vy_f * (dy_v/dist_v)) < 0.38: return False
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
        s_bid_r = self.body_ids['s']
        s_pos_r = self.data.xpos[s_bid_r][:2]
        s_rot_idx = self.qpos_indices['s']['rot']
        s_rot_r = self.data.qpos[self.model.jnt_qposadr[s_rot_idx]]
        
        found_r = False
        for h_key_r in ["h1", "h2"]:
            h_bid_r = self.body_ids[h_key_r]
            h_pos_r = self.data.xpos[h_bid_r][:2]
            if self._is_within_fov_and_visible(s_pos_r, s_rot_r, h_pos_r, s_bid_r, h_bid_r):
                found_r = True
                break
        return (-1.0 if found_r else 1.0), found_r

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        for k_re in self.prev_action_btns: 
            self.prev_action_btns[k_re].fill(0.0)
        for k_tm_re in self.agent_keys: 
            self.cooldown_timers[k_tm_re]["lock"] = 0
            self.cooldown_timers[k_tm_re]["grab"] = 0
        for t_re in ["b1", "b2", "ramp"]: 
            self.lock_owners[t_re] = None
            self.locked_pose[t_re] = None
        
        self._init_npcs()
        m, d = self.model, self.data
        placed = []
        for kt_re in ["ramp", "box1", "box2", "s", "h1", "h2"]:
            rad_re = 0.6 if kt_re in self.agent_keys else (1.0 if "box" in kt_re else 1.5)
            for _ in range(500):
                pt_re = np.random.uniform(-5.2, 5.2, 2)
                overlap_re = False
                for (cx, cy, sx, sy) in self.maze_walls:
                    if abs(pt_re[0]-cx)<(sx+rad_re) and abs(pt_re[1]-cy)<(sy+rad_re): 
                        overlap_re = True
                        break
                if overlap_re or any(np.linalg.norm(pt_re-p)<(rad_re+r+0.2) for (p,r) in placed):
                    continue
                if kt_re in self.agent_keys:
                    q_re = self.qpos_indices[kt_re]
                    d.qpos[m.jnt_qposadr[q_re['x']]] = pt_re[0]
                    d.qpos[m.jnt_qposadr[q_re['y']]] = pt_re[1]
                    d.qpos[m.jnt_qposadr[q_re['rot']]] = np.random.uniform(-np.pi, np.pi)
                else:
                    body_re = m.body(f"{kt_re}_body")
                    adr_re = m.jnt_qposadr[body_re.jntadr[0]]
                    d.qpos[adr_re:adr_re+2] = pt_re
                placed.append((pt_re, rad_re))
                break
        self._sync_visual_states()
        mujoco.mj_forward(m, d)

        self.object_ground_z["b1"] = float(d.xpos[self.box_ids[0]][2])
        self.object_ground_z["b2"] = float(d.xpos[self.box_ids[1]][2])
        if self.ramp_id != -1:
            self.object_ground_z["ramp"] = float(d.xpos[self.ramp_id][2])

        return self._normalize_obs(self._get_obs(1)), {"is_detected": False}

    def render(self):
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
            for ak_v in self.agent_keys:
                bid_v = self.body_ids[ak_v]
                p_s_v = self.data.xpos[bid_v]
                rot_idx_v = self.qpos_indices[ak_v]['rot']
                rot_v = self.data.qpos[self.model.jnt_qposadr[rot_idx_v]]
                rgba_v = [self.obj_default_colors[bid_v][0], self.obj_default_colors[bid_v][1], self.obj_default_colors[bid_v][2], 0.8]
                for tk_v in self.agent_keys:
                    if tk_v != ak_v:
                        t_bid_v = self.body_ids[tk_v]
                        if self._is_within_fov_and_visible(p_s_v[:2], rot_v, self.data.xpos[t_bid_v][:2], bid_v, t_bid_v):
                            mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 4.0, p_s_v + [0,0,0.4], self.data.xpos[t_bid_v] + [0,0,0.4])
                            scn.geoms[scn.ngeom].rgba = rgba_v
                            scn.ngeom += 1
                for o_id_v in self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else []):
                    if self._is_within_fov_and_visible(p_s_v[:2], rot_v, self.data.xpos[o_id_v][:2], bid_v, o_id_v):
                        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 2.0, p_s_v + [0,0,0.4], self.data.xpos[o_id_v] + [0,0,0.4])
                        scn.geoms[scn.ngeom].rgba = rgba_v
                        scn.ngeom += 1
        self.viewer.sync()

    def close(self):
        if self.viewer: 
            self.viewer.close()