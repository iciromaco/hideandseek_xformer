# hns_environment.py v2.69
# 演習第26回：【全ロジック極限展開 ＆ 合理的絶対静止条件 ＆ 権限管理・完全復元版】
#
# 修正内容:
# 1. 論理の再展開（行数復元）:
#    - 前回の更新で凝縮された全メソッドを「1行1命令」の原則に基づき、垂直方向に再展開。
#    - 速度ノルム、距離、判定フラグの算出プロセスをすべて中間変数に分離。
# 2. 合理的絶対静止条件の厳格適用:
#    - Lock/Hold の開始条件として「エージェントとオブジェクト双方が絶対的に静止（閾値 0.03）」であることを唯一の条件に設定。
#    - 相対静止（移動しながらの操作）を排除し、物理的な安定性を最大化。
# 3. インタラクション権限の厳密化:
#    - Hold OFF: そのオブジェクトを現在掴んでいる「本人」のみが解除パルスを送信可能。
#    - Lock OFF: そのオブジェクトをロックした者と同じ「チーム（Seeker/Hider）」のみが解除可能。
# 4. 物理仕様の完全維持: 1.02m マージン、3段階速度・加速度リセット、55次元観測、トグル制御をすべて維持。

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
        self.intent_ttl_limit = 35
        self.intent_stable_needed_lock = 2
        self.intent_stable_needed_grab = 1
        self.stationary_thresh_agent_lock = 0.03
        self.stationary_thresh_obj_lock = 0.04
        self.stationary_thresh_agent_grab = 0.10
        self.stationary_thresh_obj_grab = 0.12
        self.interaction_intents = {
            "s": {
                "lock": {"active": False, "tok": "", "id": -1, "ttl": 0, "stable": 0},
                "grab": {"active": False, "tok": "", "id": -1, "ttl": 0, "stable": 0}
            },
            "h1": {
                "lock": {"active": False, "tok": "", "id": -1, "ttl": 0, "stable": 0},
                "grab": {"active": False, "tok": "", "id": -1, "ttl": 0, "stable": 0}
            },
            "h2": {
                "lock": {"active": False, "tok": "", "id": -1, "ttl": 0, "stable": 0},
                "grab": {"active": False, "tok": "", "id": -1, "ttl": 0, "stable": 0}
            }
        }
        # チーム権限管理用 (None, 's', 'h')
        self.lock_owners = {
            "b1": None,
            "b2": None,
            "ramp": None
        }
        self.collision_guard_steps = {
            "b1": 0,
            "b2": 0,
            "ramp": 0,
        }
        self.velocity_guard_steps = {
            "b1": 0,
            "b2": 0,
            "ramp": 0,
        }
        self.pending_grab_release = {
            "s": {"active": False, "tok": "", "steps": 0},
            "h1": {"active": False, "tok": "", "steps": 0},
            "h2": {"active": False, "tok": "", "steps": 0},
        }
        self.velocity_guard_agent_xy_max = 0.15
        self.velocity_guard_agent_rot_max = 0.15
        self.velocity_guard_obj_max = 0.12
        self.body_ids = {}
        self.qpos_indices = {}
        self.obj_default_colors = {}
        self.obj_geom_ids = {}
        self.obj_default_con = {}
        self.eq_ids = {}

        # 物理構造の解析（詳細展開）
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
        """XML解析とキャッシュ構築（1行1命令）"""
        m = self.model
        for k in self.agent_keys:
            full_name = self.key_to_full[k]
            body_name = f"{full_name}_body"
            body_obj = m.body(body_name)
            self.body_ids[k] = body_obj.id

            joint_prefix_candidates = [k, full_name]

            def _resolve_joint_id(suffix):
                last_error = None
                for prefix in joint_prefix_candidates:
                    joint_name = f"{prefix}_{suffix}"
                    try:
                        return m.joint(joint_name).id
                    except Exception as exc:
                        last_error = exc
                raise KeyError(
                    f"Joint not found for agent '{k}' and suffix '{suffix}'. "
                    f"Tried: {joint_prefix_candidates}"
                ) from last_error

            self.qpos_indices[k] = {
                'x': _resolve_joint_id("x"),
                'y': _resolve_joint_id("y"),
                'rot': _resolve_joint_id("rot")
            }

        missing_agents = [k for k in self.agent_keys if k not in self.qpos_indices]
        if missing_agents:
            raise RuntimeError(f"Missing qpos indices for agents: {missing_agents}")

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

    def _init_npcs(self):
        """NPC の生成"""
        self.npcs = {}
        self.npcs["s"] = RuleBasedSeeker()
        if self.mode == "initial":
            self.npcs["h2"] = RuleBasedHider()

    def step(self, action):
        """物理更新ステップ（極限展開）"""
        self.current_step = self.current_step + 1
        self._update_collision_guards()
        
        # クールダウン更新
        for k_tm in self.agent_keys:
            if self.cooldown_timers[k_tm]["lock"] > 0:
                self.cooldown_timers[k_tm]["lock"] -= 1
            if self.cooldown_timers[k_tm]["grab"] > 0:
                self.cooldown_timers[k_tm]["grab"] -= 1

        ctrl_vec = np.zeros(self.model.nu)
        act_flat = np.ravel(action)

        # 1. Hider 1 行動展開
        h1_fwd = act_flat[0]
        h1_turn = act_flat[1]
        h1_lock = act_flat[2]
        h1_grab = act_flat[3]
        ctrl_vec[2] = h1_fwd
        ctrl_vec[3] = h1_turn
        self._process_physical_interaction("h1", h1_lock, h1_grab)

        # 2. Hider 2 行動展開
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
                h2_npc_trn = h2_act[1]
                h2_npc_lck = h2_act[2]
                h2_npc_grb = h2_act[3]
                ctrl_vec[4] = h2_npc_fwd
                ctrl_vec[5] = h2_npc_trn
                self._process_physical_interaction("h2", h2_npc_lck, h2_npc_grb)

        # 3. Seeker 行動展開
        obs_s = self._get_obs(0)
        s_act = self.npcs["s"].get_action(obs_s)
        if self.current_step <= self.prep_steps:
            ctrl_vec[0] = 0.0
            ctrl_vec[1] = 0.0
            self._process_physical_interaction("s", 0.0, 0.0)
        else:
            s_npc_fwd = s_act[0]
            s_npc_trn = s_act[1]
            s_npc_lck = s_act[2]
            s_npc_grb = s_act[3]
            ctrl_vec[0] = s_npc_fwd
            ctrl_vec[1] = s_npc_trn
            self._process_physical_interaction("s", s_npc_lck, s_npc_grb)

        # Hold成立前（grab intent中）のみ回転入力を抑止
        if self.interaction_intents["s"]["grab"]["active"] and (not self._is_agent_holding("s")):
            ctrl_vec[1] = 0.0
        if self.interaction_intents["h1"]["grab"]["active"] and (not self._is_agent_holding("h1")):
            ctrl_vec[3] = 0.0
        if self.interaction_intents["h2"]["grab"]["active"] and (not self._is_agent_holding("h2")):
            ctrl_vec[5] = 0.0

        self.data.ctrl[:] = ctrl_vec
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        self._stabilize_active_grasp_rotation()
        mujoco.mj_forward(self.model, self.data)
        self._apply_velocity_guard()

        self._sync_visual_states()
        if self.render_mode == "human":
            self.render()

        rew_val, found_flag = self._compute_team_reward()
        truncated = self.current_step >= 500
        
        obs_h1_raw = self._get_obs(1)
        norm_obs_h1 = self._normalize_obs(obs_h1_raw)

        return norm_obs_h1, float(rew_val), False, truncated, {"is_detected": found_flag}

    def _process_physical_interaction(self, agent_key, lock_cmd, grab_cmd):
        """物理インタラクション処理（Intent→減速→静止Commit）"""
        m, d = self.model, self.data
        agent_bid = self.body_ids[agent_key]
        
        # 1. 入力パルス判定
        prev = self.prev_action_btns[agent_key]
        lock_pushed = (lock_cmd > 0.5) and (prev[0] <= 0.5)
        grab_pushed = (grab_cmd > 0.5) and (prev[1] <= 0.5)
        
        # クールダウンチェック
        l_tmr = self.cooldown_timers[agent_key]["lock"]
        g_tmr = self.cooldown_timers[agent_key]["grab"]
        if lock_pushed and l_tmr > 0: lock_pushed = False
        if grab_pushed and g_tmr > 0: grab_pushed = False
        
        self.prev_action_btns[agent_key][0] = lock_cmd
        self.prev_action_btns[agent_key][1] = grab_cmd

        # 2. 自身の把持状態の特定
        my_held_tok = ""
        my_held_id = -1
        for tok_chk in ["b1", "b2", "ramp"]:
            g_id_chk = self.eq_ids[f"grasp_{agent_key}_{tok_chk}"]
            if d.eq_active[g_id_chk] > 0.5:
                my_held_tok = tok_chk
                if tok_chk == "b1": my_held_id = self.box_ids[0]
                elif tok_chk == "b2": my_held_id = self.box_ids[1]
                else: my_held_id = self.ramp_id
                break

        # 3. 近接オブジェクト探索（視界判定込み）
        objs = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        best_id, best_tok = -1, ""
        interact_th = 1.50 if my_held_id != -1 else 1.15
        min_dist = interact_th
        
        a_pos_w = d.xpos[agent_bid]
        r_idx = self.qpos_indices[agent_key]['rot']
        r_val = d.qpos[m.jnt_qposadr[r_idx]]
        dir_vec = np.array([math.cos(r_val), math.sin(r_val)])

        for oid in objs:
            o_pos_w = d.xpos[oid]
            # 視界または把持中の特権
            is_vis = self._is_within_fov_and_visible(a_pos_w[:2], r_val, o_pos_w[:2], agent_bid, oid)
            if not is_vis and oid != my_held_id:
                continue
            
            diff_w = o_pos_w - a_pos_w
            dist_curr = np.linalg.norm(diff_w)
            
            if dist_curr < min_dist:
                min_dist = dist_curr
                best_id = oid
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
                if "box1" in nm: best_tok = "b1"
                elif "box2" in nm: best_tok = "b2"
                else: best_tok = "ramp"

        # 4. 速度判定（絶対静止）
        v_ax_adr = m.jnt_dofadr[self.qpos_indices[agent_key]['x']]
        v_ar_adr = m.jnt_dofadr[self.qpos_indices[agent_key]['rot']]
        
        my_team = agent_key[0] # 's' or 'h'
        lock_int = self.interaction_intents[agent_key]["lock"]
        grab_int = self.interaction_intents[agent_key]["grab"]

        def _clear_intent(intent_obj):
            intent_obj["active"] = False
            intent_obj["tok"] = ""
            intent_obj["id"] = -1
            intent_obj["ttl"] = 0
            intent_obj["stable"] = 0

        def _arm_intent(intent_obj, tok, oid):
            intent_obj["active"] = True
            intent_obj["tok"] = tok
            intent_obj["id"] = oid
            intent_obj["ttl"] = self.intent_ttl_limit
            intent_obj["stable"] = 0

        def _refresh_candidate(intent_obj):
            if not intent_obj["active"]:
                return -1, "", 999.0, False
            if intent_obj["id"] == -1:
                _clear_intent(intent_obj)
                return -1, "", 999.0, False
            tok_now = intent_obj["tok"]
            oid_now = intent_obj["id"]
            o_pos_now = d.xpos[oid_now]
            vis_now = self._is_within_fov_and_visible(a_pos_w[:2], r_val, o_pos_now[:2], agent_bid, oid_now)
            if (not vis_now) and (oid_now != my_held_id):
                intent_obj["ttl"] -= 1
                if intent_obj["ttl"] <= 0:
                    _clear_intent(intent_obj)
                return -1, "", 999.0, False
            dist_now = np.linalg.norm(o_pos_now - a_pos_w)
            if dist_now > 1.50:
                intent_obj["ttl"] -= 1
                if intent_obj["ttl"] <= 0:
                    _clear_intent(intent_obj)
                return -1, "", 999.0, False
            return oid_now, tok_now, dist_now, True

        def _apply_brake_for_object(oid_brake):
            v_obj_adr_b = m.jnt_dofadr[m.body_jntadr[oid_brake]]
            d.qvel[v_ax_adr:v_ax_adr+2] *= 0.35
            d.qvel[v_ar_adr] *= 0.35
            d.qvel[v_obj_adr_b:v_obj_adr_b+6] *= 0.45
            return v_obj_adr_b

        def _is_stationary(v_obj_adr_s, agent_th, obj_th):
            v_ag_xy_s = np.linalg.norm(d.qvel[v_ax_adr : v_ax_adr+2])
            v_ag_rot_s = abs(d.qvel[v_ar_adr])
            v_ag_total_s = v_ag_xy_s + v_ag_rot_s
            v_obj_total_s = np.linalg.norm(d.qvel[v_obj_adr_s : v_obj_adr_s+6])
            stat_s = (v_ag_total_s < agent_th) and (v_obj_total_s < obj_th)
            return stat_s

        # --- [A] Lock処理（チーム権限管理） ---
        if lock_pushed and best_id != -1:
            lid_push = self.eq_ids[f"lock_{best_tok}"]
            if d.eq_active[lid_push] > 0.5:
                if self.lock_owners[best_tok] == my_team:
                    d.eq_active[lid_push] = 0
                    self.lock_owners[best_tok] = None
                    self.cooldown_timers[agent_key]["lock"] = self.cooldown_limit
                    _clear_intent(lock_int)
            else:
                _arm_intent(lock_int, best_tok, best_id)

        lock_oid, lock_tok, lock_dist, lock_ok = _refresh_candidate(lock_int)
        if lock_ok:
            lid = self.eq_ids[f"lock_{lock_tok}"]
            if d.eq_active[lid] > 0.5:
                _clear_intent(lock_int)
            else:
                gs = self.eq_ids[f"grasp_s_{lock_tok}"]
                gh1 = self.eq_ids[f"grasp_h1_{lock_tok}"]
                gh2 = self.eq_ids[f"grasp_h2_{lock_tok}"]
                others_grab = (d.eq_active[gs] + d.eq_active[gh1] + d.eq_active[gh2])
                my_grab_val = d.eq_active[self.eq_ids[f"grasp_{agent_key}_{lock_tok}"]]
                net_others = others_grab - my_grab_val
                if (my_held_id == -1) or (my_held_id == lock_oid):
                    v_obj_adr_lock = _apply_brake_for_object(lock_oid)
                    if _is_stationary(v_obj_adr_lock, self.stationary_thresh_agent_lock, self.stationary_thresh_obj_lock):
                        lock_int["stable"] += 1
                    else:
                        lock_int["stable"] = 0
                    if lock_int["stable"] >= self.intent_stable_needed_lock and net_others <= 0.5 and lock_dist <= 1.12:
                        if my_grab_val > 0.5:
                            d.eq_active[self.eq_ids[f"grasp_{agent_key}_{lock_tok}"]] = 0
                            mujoco.mj_forward(m, d)
                        d.qvel[v_ax_adr:v_ax_adr+2] = 0.0
                        d.qvel[v_ar_adr] = 0.0
                        d.qvel[v_obj_adr_lock:v_obj_adr_lock+6] = 0.0
                        mujoco.mj_forward(m, d)
                        m.eq_data[lid, 0:3] = d.xpos[lock_oid]
                        m.eq_data[lid, 3:7] = d.xquat[lock_oid]
                        d.eq_active[lid] = 1
                        self.lock_owners[lock_tok] = my_team
                        self._activate_collision_guard(lock_tok, 6)
                        self._activate_velocity_guard(lock_tok, 12)
                        self.cooldown_timers[agent_key]["lock"] = self.cooldown_limit
                        _clear_intent(lock_int)
                else:
                    _clear_intent(lock_int)
            if lock_int["active"]:
                lock_int["ttl"] -= 1
                if lock_int["ttl"] <= 0:
                    _clear_intent(lock_int)
            mujoco.mj_forward(m, d)

        # --- [B] Grab処理（本人解除制限） ---
        if grab_pushed:
            if my_held_id != -1:
                # 解除権限: 本人が今掴んでいるものなら許可
                lock_block_release = (
                    lock_int["active"] and
                    lock_int["tok"] == my_held_tok and
                    lock_int["id"] == my_held_id
                )
                if (best_id == my_held_id) and (not lock_block_release):
                    d.eq_active[self.eq_ids[f"grasp_{agent_key}_{my_held_tok}"]] = 0
                    self.cooldown_timers[agent_key]["grab"] = self.cooldown_limit
            elif best_id != -1:
                _arm_intent(grab_int, best_tok, best_id)
            mujoco.mj_forward(m, d)

        grab_oid, grab_tok, grab_dist, grab_ok = _refresh_candidate(grab_int)
        if grab_ok:
            l_chk_id = self.eq_ids[f"lock_{grab_tok}"]
            gs_g = self.eq_ids[f"grasp_s_{grab_tok}"]
            gh1_g = self.eq_ids[f"grasp_h1_{grab_tok}"]
            gh2_g = self.eq_ids[f"grasp_h2_{grab_tok}"]
            any_grab = (d.eq_active[gs_g] + d.eq_active[gh1_g] + d.eq_active[gh2_g]) > 0.5
            if d.eq_active[l_chk_id] <= 0.5 and (not any_grab):
                v_obj_adr_g = _apply_brake_for_object(grab_oid)
                if _is_stationary(v_obj_adr_g, self.stationary_thresh_agent_grab, self.stationary_thresh_obj_grab):
                    grab_int["stable"] += 1
                else:
                    grab_int["stable"] = 0
                if grab_int["stable"] >= self.intent_stable_needed_grab and grab_dist <= 1.25:
                    on_gid = self.eq_ids[f"grasp_{agent_key}_{grab_tok}"]
                    d.qvel[v_ax_adr:v_ax_adr+2] = 0.0
                    d.qvel[v_ar_adr] = 0.0
                    d.qvel[v_obj_adr_g:v_obj_adr_g+6] = 0.0
                    mujoco.mj_forward(m, d)
                    inv_q_a = np.zeros(4)
                    mujoco.mju_negQuat(inv_q_a, d.xquat[agent_bid])
                    rel_p_l = np.zeros(3)
                    mujoco.mju_rotVecQuat(rel_p_l, d.xpos[grab_oid] - d.xpos[agent_bid], inv_q_a)
                    rel_q_l = np.zeros(4)
                    mujoco.mju_mulQuat(rel_q_l, inv_q_a, d.xquat[grab_oid])
                    m.eq_data[on_gid, 0:3] = rel_p_l
                    m.eq_data[on_gid, 3:7] = rel_q_l
                    d.eq_active[on_gid] = 1
                    self._activate_collision_guard(grab_tok, 2)
                    self._activate_velocity_guard(grab_tok, 5)
                    self.cooldown_timers[agent_key]["grab"] = self.cooldown_limit
                    _clear_intent(grab_int)
            else:
                _clear_intent(grab_int)
            if grab_int["active"]:
                grab_int["ttl"] -= 1
                if grab_int["ttl"] <= 0:
                    _clear_intent(grab_int)
            mujoco.mj_forward(m, d)

        if not lock_pushed and not grab_pushed and (not lock_int["active"]) and (not grab_int["active"]):
            return

    def _sync_visual_states(self):
        """色の同期（1行1命令展開）"""
        m, d = self.model, self.data
        tools = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        for oid_v in tools:
            nm_v = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid_v) or ""
            if "box1" in nm_v: t_v = "b1"
            elif "box2" in nm_v: t_v = "b2"
            else: t_v = "ramp"
            
            f_col = self.obj_default_colors[oid_v].copy()
            
            # 把持中 (Yellow)
            gs_v = d.eq_active[self.eq_ids[f"grasp_s_{t_v}"]]
            gh1_v = d.eq_active[self.eq_ids[f"grasp_h1_{t_v}"]]
            gh2_v = d.eq_active[self.eq_ids[f"grasp_h2_{t_v}"]]
            if (gs_v + gh1_v + gh2_v) > 0.5:
                f_col[0] = 1.0
                f_col[1] = 1.0
                f_col[2] = 0.0
            
            # 固定中 (Red)
            l_v = d.eq_active[self.eq_ids[f"lock_{t_v}"]]
            if l_v > 0.5:
                f_col[0] = 1.0
                f_col[1] = 0.0
                f_col[2] = 0.0
                
            for g_id_v in self.obj_geom_ids[oid_v]:
                m.geom_rgba[g_id_v] = f_col

    def _token_to_oid(self, tok):
        if tok == "b1":
            return self.box_ids[0] if len(self.box_ids) >= 1 else -1
        if tok == "b2":
            return self.box_ids[1] if len(self.box_ids) >= 2 else -1
        if tok == "ramp":
            return self.ramp_id
        return -1

    def _is_agent_holding(self, agent_key):
        d = self.data
        for tok in ["b1", "b2", "ramp"]:
            gid = self.eq_ids.get(f"grasp_{agent_key}_{tok}", -1)
            if gid != -1 and d.eq_active[gid] > 0.5:
                return True
        return False

    def _stabilize_active_grasp_rotation(self):
        m = self.model
        d = self.data
        for ak in self.agent_keys:
            for tok in ["b1", "b2", "ramp"]:
                gid = self.eq_ids.get(f"grasp_{ak}_{tok}", -1)
                if gid == -1 or d.eq_active[gid] <= 0.5:
                    continue
                oid = self._token_to_oid(tok)
                if oid == -1:
                    continue
                v_obj_adr = m.jnt_dofadr[m.body_jntadr[oid]]
                d.qvel[v_obj_adr+3:v_obj_adr+6] = 0.0

    def _set_token_collision_enabled(self, tok, enabled):
        oid = self._token_to_oid(tok)
        if oid == -1:
            return
        for g_idx in self.obj_geom_ids.get(oid, []):
            if enabled:
                self.model.geom_contype[g_idx] = self.obj_default_con[g_idx]["type"]
                self.model.geom_conaffinity[g_idx] = self.obj_default_con[g_idx]["aff"]
            else:
                self.model.geom_contype[g_idx] = 0
                self.model.geom_conaffinity[g_idx] = 0

    def _activate_collision_guard(self, tok, steps):
        if tok not in self.collision_guard_steps:
            return
        self.collision_guard_steps[tok] = max(self.collision_guard_steps[tok], steps)

    def _activate_velocity_guard(self, tok, steps):
        if tok not in self.velocity_guard_steps:
            return
        self.velocity_guard_steps[tok] = max(self.velocity_guard_steps[tok], steps)

    def _apply_velocity_guard(self):
        m = self.model
        d = self.data
        guard_active = any(self.velocity_guard_steps[t] > 0 for t in ["b1", "b2", "ramp"])
        if not guard_active:
            return

        v_max_agent_xy = self.velocity_guard_agent_xy_max
        v_max_agent_rot = self.velocity_guard_agent_rot_max
        v_max_obj = self.velocity_guard_obj_max

        for ak in self.agent_keys:
            v_ax_adr = m.jnt_dofadr[self.qpos_indices[ak]['x']]
            v_ar_adr = m.jnt_dofadr[self.qpos_indices[ak]['rot']]

            vxy = d.qvel[v_ax_adr:v_ax_adr+2]
            nxy = np.linalg.norm(vxy)
            if nxy > v_max_agent_xy and nxy > 1e-8:
                d.qvel[v_ax_adr:v_ax_adr+2] = vxy * (v_max_agent_xy / nxy)

            vro = abs(d.qvel[v_ar_adr])
            if vro > v_max_agent_rot:
                d.qvel[v_ar_adr] = np.sign(d.qvel[v_ar_adr]) * v_max_agent_rot

        obj_ids = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        for oid in obj_ids:
            v_obj_adr = m.jnt_dofadr[m.body_jntadr[oid]]
            vobj = d.qvel[v_obj_adr:v_obj_adr+6]
            nobj = np.linalg.norm(vobj)
            if nobj > v_max_obj and nobj > 1e-8:
                d.qvel[v_obj_adr:v_obj_adr+6] = vobj * (v_max_obj / nobj)

    def _update_collision_guards(self):
        for tok in ["b1", "b2", "ramp"]:
            if self.collision_guard_steps[tok] > 0:
                self.collision_guard_steps[tok] -= 1
                if self.collision_guard_steps[tok] == 0:
                    self._set_token_collision_enabled(tok, True)
        self._apply_velocity_guard()
        for tok in ["b1", "b2", "ramp"]:
            if self.velocity_guard_steps[tok] > 0:
                self.velocity_guard_steps[tok] -= 1
        for ak in self.agent_keys:
            rel = self.pending_grab_release[ak]
            if not rel["active"]:
                continue
            if rel["steps"] > 0:
                rel["steps"] -= 1
            if rel["steps"] <= 0:
                tok = rel["tok"]
                if tok:
                    gid = self.eq_ids.get(f"grasp_{ak}_{tok}", -1)
                    if gid != -1 and self.data.eq_active[gid] > 0.5:
                        self.data.eq_active[gid] = 0
                rel["active"] = False
                rel["tok"] = ""
                rel["steps"] = 0

    def _get_obs(self, agent_idx):
        """55次元観測の構築（極限展開版）"""
        obs = np.zeros(55, dtype=np.float32)
        d, m = self.data, self.model
        ak = self.agent_keys[agent_idx]
        bid = self.body_ids[ak]
        
        # 自己情報の取得
        p_self = d.xpos[bid]
        r_idx = self.qpos_indices[ak]['rot']
        rot_val = float(d.qpos[m.jnt_qposadr[r_idx]])
        v_adr = m.jnt_dofadr[self.qpos_indices[ak]['x']]
        vx_w = d.qvel[v_adr]
        vy_w = d.qvel[v_adr + 1]
        
        # 座標変換用
        cos_r = math.cos(-rot_val)
        sin_r = math.sin(-rot_val)
        
        # 自己速度 (Local)
        obs[0] = vx_w * cos_r - vy_w * sin_r
        obs[1] = vx_w * sin_r + vy_w * cos_r
        # 自己回転
        obs[2] = rot_val
        obs[3] = math.cos(rot_val)
        obs[4] = math.sin(rot_val)
        # Lidar (12本)
        lidar_data = self.vis_engine.cast_lidar(p_self[:2], rot_val, self.lidar_mode, bid)
        obs[5:17] = lidar_data - 0.45

        # 道具情報 (8x3=24次元)
        all_tools = self.box_ids + [self.ramp_id]
        for i_obj, o_id in enumerate(all_tools):
            if o_id == -1: continue
            base = 17 + i_obj * 8
            tok_nm = "b1" if i_obj == 0 else "b2" if i_obj == 1 else "ramp"
            o_pos = d.xpos[o_id]
            
            if self._is_within_fov_and_visible(p_self[:2], rot_val, o_pos[:2], bid, o_id):
                rel_p = o_pos - p_self
                obs[base + 0] = rel_p[0] * cos_r - rel_p[1] * sin_r
                obs[base + 1] = rel_p[0] * sin_r + rel_p[1] * cos_r
                
                v_o_adr = m.jnt_dofadr[m.body_jntadr[o_id]]
                v_ox, v_oy = d.qvel[v_o_adr], d.qvel[v_o_adr + 1]
                rel_vx, rel_vy = v_ox - vx_w, v_oy - vy_w
                obs[base + 2] = rel_vx * cos_r - rel_vy * sin_r
                obs[base + 3] = rel_vx * sin_r + rel_vy * cos_r
                
                o_quat = d.xquat[o_id]
                o_yaw = math.atan2(2.0*(o_quat[0]*o_quat[3]+o_quat[1]*o_quat[2]), 1.0-2.0*(o_quat[2]*o_quat[2]+o_quat[3]*o_quat[3]))
                obs[base + 4] = math.cos(o_yaw - rot_val)
                obs[base + 5] = math.sin(o_yaw - rot_val)
                
                gs = d.eq_active[self.eq_ids[f"grasp_s_{tok_nm}"]]
                gh1 = d.eq_active[self.eq_ids[f"grasp_h1_{tok_nm}"]]
                gh2 = d.eq_active[self.eq_ids[f"grasp_h2_{tok_nm}"]]
                has_grasp = (gs + gh1 + gh2) > 0.5
                has_lock = d.eq_active[self.eq_ids[f"lock_{tok_nm}"]] > 0.5
                tool_state = 2.0 if has_lock else (1.0 if has_grasp else 0.0)
                obs[base + 6] = tool_state
                obs[base + 7] = 1.0 # 存在フラグ

        # 相手・味方情報
        if ak == "s": ek, pk = "h1", "h2"
        elif ak == "h1": ek, pk = "s", "h2"
        else: ek, pk = "s", "h1"

        # Enemy (7次元) ＆ Partner (7次元) のループ
        for i_t, t_key in enumerate([ek, pk]):
            t_bid = self.body_ids[t_key]
            t_base = 41 + i_t * 7
            t_pos = d.xpos[t_bid]
            
            if self._is_within_fov_and_visible(p_self[:2], rot_val, t_pos[:2], bid, t_bid):
                r_p = t_pos - p_self
                obs[t_base + 0] = r_p[0] * cos_r - r_p[1] * sin_r
                obs[t_base + 1] = r_p[0] * sin_r + r_p[1] * cos_r
                
                t_v_adr = m.jnt_dofadr[self.qpos_indices[t_key]['x']]
                t_vx, t_vy = d.qvel[t_v_adr], d.qvel[t_v_adr + 1]
                r_vx, r_vy = t_vx - vx_w, t_vy - vy_w
                obs[t_base + 2] = r_vx * cos_r - r_vy * sin_r
                obs[t_base + 3] = r_vx * sin_r + r_vy * cos_r
                
                t_rot_idx = self.qpos_indices[t_key]['rot']
                t_rot = float(d.qpos[m.jnt_qposadr[t_rot_idx]])
                obs[t_base + 4] = math.cos(t_rot - rot_val)
                obs[t_base + 5] = math.sin(t_rot - rot_val)
                obs[t_base + 6] = 1.0

        return obs

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        """2Dレイキャスト判定（バイパスなし）"""
        dx = t_pos[0] - pos[0]
        dy = t_pos[1] - pos[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 15.0: return False
        vx, vy = math.cos(rot), math.sin(rot)
        # FOV判定
        dot_v = vx * (dx/dist) + vy * (dy/dist)
        if dot_v < 0.38: return False
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
        s_idx = self.qpos_indices['s']['rot']
        s_rot = self.data.qpos[self.model.jnt_qposadr[s_idx]]
        
        found = False
        for h_key in ["h1", "h2"]:
            h_bid = self.body_ids[h_key]
            h_pos = self.data.xpos[h_bid][:2]
            if self._is_within_fov_and_visible(s_pos, s_rot, h_pos, s_bid, h_bid):
                found = True
                break
        return (-1.0 if found else 1.0), found

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        for k in self.prev_action_btns: self.prev_action_btns[k].fill(0.0)
        for k in self.agent_keys: 
            self.cooldown_timers[k]["lock"] = 0
            self.cooldown_timers[k]["grab"] = 0
            self.interaction_intents[k]["lock"]["active"] = False
            self.interaction_intents[k]["lock"]["tok"] = ""
            self.interaction_intents[k]["lock"]["id"] = -1
            self.interaction_intents[k]["lock"]["ttl"] = 0
            self.interaction_intents[k]["lock"]["stable"] = 0
            self.interaction_intents[k]["grab"]["active"] = False
            self.interaction_intents[k]["grab"]["tok"] = ""
            self.interaction_intents[k]["grab"]["id"] = -1
            self.interaction_intents[k]["grab"]["ttl"] = 0
            self.interaction_intents[k]["grab"]["stable"] = 0
        for t in ["b1", "b2", "ramp"]: self.lock_owners[t] = None
        for t in ["b1", "b2", "ramp"]:
            self.collision_guard_steps[t] = 0
            self._set_token_collision_enabled(t, True)
            self.velocity_guard_steps[t] = 0
        for ak in self.agent_keys:
            self.pending_grab_release[ak]["active"] = False
            self.pending_grab_release[ak]["tok"] = ""
            self.pending_grab_release[ak]["steps"] = 0
        
        self._init_npcs()
        m, d = self.model, self.data
        placed = []
        for kt in ["ramp", "box1", "box2", "s", "h1", "h2"]:
            rad = 0.6 if kt in self.agent_keys else (1.0 if "box" in kt else 1.5)
            for _ in range(500):
                pt = np.random.uniform(-5.2, 5.2, 2)
                ov = any(abs(pt[0]-cx)<(sx+rad) and abs(pt[1]-cy)<(sy+rad) for (cx, cy, sx, sy) in self.maze_walls)
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
                r_idx = self.qpos_indices[ak]['rot']
                rot = self.data.qpos[self.model.jnt_qposadr[r_idx]]
                rgba_orig = self.obj_default_colors[bid]
                rgba = [rgba_orig[0], rgba_orig[1], rgba_orig[2], 0.8]
                
                for target_k in self.agent_keys:
                    if target_k != ak:
                        t_bid = self.body_ids[target_k]
                        if self._is_within_fov_and_visible(p_s[:2], rot, self.data.xpos[t_bid][:2], bid, t_bid):
                            mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 4.0, p_s + [0,0,0.4], self.data.xpos[t_bid] + [0,0,0.4])
                            scn.geoms[scn.ngeom].rgba = rgba
                            scn.ngeom += 1
                for o_id in self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else []):
                    if self._is_within_fov_and_visible(p_s[:2], rot, self.data.xpos[o_id][:2], bid, o_id):
                        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 2.0, p_s + [0,0,0.4], self.data.xpos[o_id] + [0,0,0.4])
                        scn.geoms[scn.ngeom].rgba = rgba
                        scn.ngeom += 1
        self.viewer.sync()

    def close(self):
        if self.viewer: 
            self.viewer.close()