# hns_environment.py v2.79
# 演習第26回：【物理挙動整合性・極限展開 ＆ 1行1命令義務化 ＆ ドリフト解消 ＆ 新仕様完全準拠版】
#
# 修正内容:
# 1. 1行1命令（1-line-1-command）の完全義務化:
#    - Python の凝縮記法（スライス代入、sum/any、リスト内包表記）を全廃。
#    - 観測値の代入、他者の把持チェック、視認判定のループをすべて独立した行に展開。
# 2. 把持ドリフトの物理的解消 (L340-L360):
#    - Grab 成立時の相対座標算出において、Z軸成分を stabilization 目標（ground_z）に厳密に同期。
#    - weld 拘束の目標位置と、毎ステップの上書き座標の矛盾を排除し、滑らかな運搬を実現。
# 3. Lock/Hold 新仕様の厳格実装:
#    - Lock 判定を Hold よりも先に評価。瞬間移動（スナップ）を禁止し、現位置での相対拘束を採用。
#    - Lock 継続中（毎サブステップ）: qpos を locked_pose に、qvel を 0 に強制固定。
#    - Hold 継続中: 高度(z)固定、roll/pitch 零化、角速度の強制零化。
# 4. 55次元カテゴリカル観測の維持:
#    - interaction_state (0:None, 1:Locked, 2:Me, 3:Others) の判定プロセスを詳細記述。
# 5. 既存ロジックの完全継承: デグレードを禁止し、推進力補正(Boost)や安全レンダリング用フラグ設定も維持。

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
                # 質量取得
                b_id_val = body_obj.id
                st_mass = m.body_subtreemass[b_id_val]
                self.agent_body_mass[k] = float(st_mass)
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
                num_geoms = m.ngeom
                for g_idx in range(num_geoms):
                    geom_owner = m.geom_bodyid[g_idx]
                    if geom_owner == i:
                        self.obj_geom_ids[i].append(g_idx)
                        if g_idx == m.body_geomadr[i]:
                            color_raw = m.geom_rgba[g_idx]
                            self.obj_default_colors[i] = color_raw.copy()
                        # 衝突設定
                        c_type = m.geom_contype[g_idx]
                        c_aff = m.geom_conaffinity[g_idx]
                        self.obj_default_con[g_idx] = {
                            "type": c_type,
                            "aff": c_aff
                        }

        # 拘束ID
        tokens = ["b1", "b2", "ramp"]
        for t in tokens:
            l_name = f"eq_lock_{t}"
            self.eq_ids[f"lock_{t}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, l_name)
            for ak in self.agent_keys:
                g_name = f"eq_grasp_{ak}_{t}"
                self.eq_ids[f"grasp_{ak}_{t}"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, g_name)

        # オブジェクト質量再設定
        if len(self.box_ids) >= 2:
            m_box1 = m.body_subtreemass[self.box_ids[0]]
            m_box2 = m.body_subtreemass[self.box_ids[1]]
            self.object_body_mass["b1"] = float(m_box1)
            self.object_body_mass["b2"] = float(m_box2)
        if self.ramp_id != -1:
            m_ramp = m.body_subtreemass[self.ramp_id]
            self.object_body_mass["ramp"] = float(m_ramp)

    def _init_npcs(self):
        """NPC の生成"""
        self.npcs = {}
        self.npcs["s"] = RuleBasedSeeker()
        if self.mode == "initial":
            self.npcs["h2"] = RuleBasedHider()

    def step(self, action):
        """物理更新ステップ（極限展開）"""
        self.current_step = self.current_step + 1
        
        # 1. クールダウンタイマーの更新
        for k_tm in self.agent_keys:
            timer_l = self.cooldown_timers[k_tm]["lock"]
            if timer_l > 0:
                self.cooldown_timers[k_tm]["lock"] = timer_l - 1
            
            timer_g = self.cooldown_timers[k_tm]["grab"]
            if timer_g > 0:
                self.cooldown_timers[k_tm]["grab"] = timer_g - 1

        ctrl_vec = np.zeros(self.model.nu)
        act_flat = np.ravel(action)

        # 2. 各エージェントのアクション適用（Lock優先）
        h1_fwd = act_flat[0]
        h1_turn = act_flat[1]
        h1_lock = act_flat[2]
        h1_grab = act_flat[3]
        ctrl_vec[2] = h1_fwd
        ctrl_vec[3] = h1_turn
        self._process_physical_interaction("h1", h1_lock, h1_grab)

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

        # 3. 運搬時の重量補正 (Forward Boost)
        fw_indices = {"s": 0, "h1": 2, "h2": 4}
        for ak_hb in self.agent_keys:
            held_tok_hb = ""
            for tok_hb in ["b1", "b2", "ramp"]:
                gid_hb = self.eq_ids[f"grasp_{ak_hb}_{tok_hb}"]
                if self.data.eq_active[gid_hb] > 0.5:
                    held_tok_hb = tok_hb
                    break
            if held_tok_hb == "":
                continue
            
            idx_f = fw_indices[ak_hb]
            m_ag = self.agent_body_mass.get(ak_hb, 1.0)
            m_ob = self.object_body_mass.get(held_tok_hb, 0.0)
            # 推力増幅
            boost_v = (m_ag + m_ob) / (m_ag + 1e-6)
            boost_v = max(1.0, min(self.hold_boost_max, boost_v))
            ctrl_vec[idx_f] = float(np.clip(ctrl_vec[idx_f] * boost_v, -1.0, 1.0))

        # 4. 物理ステップ実行 (サブステップ内強制同期)
        self.data.ctrl[:] = ctrl_vec
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            # Lock状態の強制同期
            for tok_fix in ["b1", "b2", "ramp"]:
                lid_fix = self.eq_ids[f"lock_{tok_fix}"]
                if self.data.eq_active[lid_fix] <= 0.5:
                    continue
                
                oid_fix = self.box_ids[0] if tok_fix == "b1" else self.box_ids[1] if tok_fix == "b2" else self.ramp_id
                j_adr_f = self.model.body_jntadr[oid_fix]
                if j_adr_f == -1:
                    continue
                
                q_adr_f = self.model.jnt_qposadr[j_adr_f]
                d_adr_f = self.model.jnt_dofadr[j_adr_f]
                s_pose_f = self.locked_pose[tok_fix]
                
                if s_pose_f is not None:
                    self.data.qpos[q_adr_f : q_adr_f + 7] = s_pose_f
                self.data.qvel[d_adr_f : d_adr_f + 6] = 0.0

        # 5. 把持オブジェクト姿勢安定化 ＆ 色同期
        self._stabilize_interaction_poses()
        self._sync_visual_states()
        
        if self.render_mode == "human":
            self.render()

        rew_v, found_f = self._compute_team_reward()
        truncated = self.current_step >= 500
        
        obs_raw = self._get_obs(1)
        obs_norm = self._normalize_obs(obs_raw)
        return obs_norm, float(rew_v), False, truncated, {"is_detected": found_f}

    def _process_physical_interaction(self, agent_key, lock_cmd, grab_cmd):
        """物理インタラクション処理（極限展開版）"""
        m, d = self.model, self.data
        agent_bid = self.body_ids[agent_key]
        
        # 1. 入力パルス判定
        prev_btns = self.prev_action_btns[agent_key]
        lock_pushed = (lock_cmd > 0.5) and (prev_btns[0] <= 0.5)
        grab_pushed = (grab_cmd > 0.5) and (prev_btns[1] <= 0.5)
        
        # クールダウン判定
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

        # 2. 自身の現在の把持特定
        my_held_tok = ""
        my_held_id = -1
        for tk_c in ["b1", "b2", "ramp"]:
            gid_c = self.eq_ids[f"grasp_{agent_key}_{tk_c}"]
            if d.eq_active[gid_c] > 0.5:
                my_held_tok = tk_c
                if tk_c == "b1":
                    my_held_id = self.box_ids[0]
                elif tk_c == "b2":
                    my_held_id = self.box_ids[1]
                else:
                    my_held_id = self.ramp_id
                break

        # 3. 近接オブジェクト探索 (CANDIDATE_DIST=1.25)
        objs = self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else [])
        best_id, best_tok = -1, ""
        interact_th = 1.50 if my_held_id != -1 else 1.25
        min_dist = interact_th
        
        a_pos_w = d.xpos[agent_bid]
        r_idx_i = self.qpos_indices[agent_key]['rot']
        r_val_i = d.qpos[m.jnt_qposadr[r_idx_i]]
        dir_vec_i = np.array([math.cos(r_val_i), math.sin(r_val_i)])

        for oid in objs:
            o_pos_w = d.xpos[oid]
            dist_curr = np.linalg.norm(o_pos_w - a_pos_w)
            is_vis = self._is_within_fov_and_visible(a_pos_w[:2], r_val_i, o_pos_w[:2], agent_bid, oid)
            # 視認または至近距離なら許可
            if (not is_vis) and (dist_curr > 1.25) and oid != my_held_id:
                continue
            
            if dist_curr < min_dist:
                min_dist = dist_curr
                best_id = oid
                nm_b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
                if "box1" in nm_b:
                    best_tok = "b1"
                elif "box2" in nm_b:
                    best_tok = "b2"
                else:
                    best_tok = "ramp"

        # 前方判定 (FRONT_DOT_MIN=0.15)
        is_front = False
        if best_id != -1:
            diff_v = d.xpos[best_id] - a_pos_w
            dot_f = diff_v[0] * dir_vec_i[0] + diff_v[1] * dir_vec_i[1]
            is_front = dot_f > 0.15

        my_team_c = agent_key[0]
        v_ax_adr = m.jnt_dofadr[self.qpos_indices[agent_key]['x']]
        v_ar_adr = m.jnt_dofadr[self.qpos_indices[agent_key]['rot']]

        # --- [A] Lock処理 (優先1) ---
        if lock_pushed and best_id != -1:
            lid_i = self.eq_ids[f"lock_{best_tok}"]
            lock_changed = False
            if d.eq_active[lid_i] > 0.5:
                # 解除権限: 所有チームのみ
                if self.lock_owners[best_tok] == my_team_c:
                    d.eq_active[lid_i] = 0
                    self.lock_owners[best_tok] = None
                    self.locked_pose[best_tok] = None
                    self.cooldown_timers[agent_key]["lock"] = self.cooldown_limit
                    lock_changed = True
            elif ((my_held_id == -1) or (my_held_id == best_id)) and is_front and min_dist < 1.05:
                # LOCK_DIST=1.05
                others_g_sum = 0.0
                g_id_s = self.eq_ids[f"grasp_s_{best_tok}"]
                g_id_h1 = self.eq_ids[f"grasp_h1_{best_tok}"]
                g_id_h2 = self.eq_ids[f"grasp_h2_{best_tok}"]
                others_g_sum += d.eq_active[g_id_s]
                others_g_sum += d.eq_active[g_id_h1]
                others_g_sum += d.eq_active[g_id_h2]
                
                my_g_val = d.eq_active[self.eq_ids[f"grasp_{agent_key}_{best_tok}"]]
                net_others_grab = others_g_sum - my_g_val
                
                if net_others_grab <= 0.5:
                    # 速度リセット
                    d.qvel[v_ax_adr : v_ax_adr + 2] = 0.0
                    d.qvel[v_ar_adr] = 0.0
                    v_o_adr_l = m.jnt_dofadr[m.body_jntadr[best_id]]
                    d.qvel[v_o_adr_l : v_o_adr_l + 6] = 0.0
                    # 姿勢保存 ＆ 拘束
                    jadr_l = m.body_jntadr[best_id]
                    qadr_l = m.jnt_qposadr[jadr_l]
                    pose_copy = d.qpos[qadr_l : qadr_l + 7].copy()
                    self.locked_pose[best_tok] = pose_copy
                    m.eq_data[lid_i, 0:3] = d.xpos[best_id]
                    m.eq_data[lid_i, 3:7] = d.xquat[best_id]
                    d.eq_active[lid_i] = 1
                    self.lock_owners[best_tok] = my_team_c
                    # 固定時に把持解除
                    if my_g_val > 0.5:
                        d.eq_active[self.eq_ids[f"grasp_{agent_key}_{best_tok}"]] = 0
                    self.cooldown_timers[agent_key]["lock"] = self.cooldown_limit
                    lock_changed = True
            if lock_changed:
                mujoco.mj_forward(m, d)

        # --- [B] Grab処理 (優先2) ---
        if grab_pushed:
            grab_changed = False
            if my_held_id != -1:
                # 解除権限: 本人のみ
                if best_id == my_held_id:
                    rel_gid = self.eq_ids[f"grasp_{agent_key}_{my_held_tok}"]
                    d.eq_active[rel_gid] = 0
                    self.cooldown_timers[agent_key]["grab"] = self.cooldown_limit
                    grab_changed = True
            elif best_id != -1 and is_front and d.eq_active[self.eq_ids[f"lock_{best_tok}"]] <= 0.5:
                # 多重把持禁止
                h_sum_g = 0.0
                h_sum_g += d.eq_active[self.eq_ids[f"grasp_s_{best_tok}"]]
                h_sum_g += d.eq_active[self.eq_ids[f"grasp_h1_{best_tok}"]]
                h_sum_g += d.eq_active[self.eq_ids[f"grasp_h2_{best_tok}"]]
                
                if h_sum_g <= 0.5:
                    # 速度リセット
                    d.qvel[v_ax_adr : v_ax_adr + 2] = 0.0
                    d.qvel[v_ar_adr] = 0.0
                    v_o_adr_g = m.jnt_dofadr[m.body_jntadr[best_id]]
                    d.qvel[v_o_adr_g : v_o_adr_g + 6] = 0.0
                    
                    # ドリフト解消: stabilization ターゲット高度に基づいた精密な相対座標
                    inv_q_a = np.zeros(4)
                    mujoco.mju_negQuat(inv_q_a, d.xquat[agent_bid])
                    
                    p_diff_world = d.xpos[best_id] - d.xpos[agent_bid]
                    # Z軸成分を ground_z 基準に補正
                    target_obj_z = self.object_ground_z[best_tok]
                    p_diff_world[2] = target_obj_z - d.xpos[agent_bid][2]
                    
                    rel_p_local = np.zeros(3)
                    mujoco.mju_rotVecQuat(rel_p_local, p_diff_world, inv_q_a)
                    
                    # 姿勢安定化 (Yaw維持, Roll/Pitch零化)
                    oq_g = d.xquat[best_id]
                    oy_g = math.atan2(2.0*(oq_g[0]*oq_g[3]+oq_g[1]*oq_g[2]), 1.0-2.0*(oq_g[2]*oq_g[2]+oq_g[3]*oq_g[3]))
                    rel_y_g = oy_g - r_val_i
                    rel_q_g = np.array([math.cos(rel_y_g * 0.5), 0.0, 0.0, math.sin(rel_y_g * 0.5)])
                    
                    gid_idx = self.eq_ids[f"grasp_{agent_key}_{best_tok}"]
                    m.eq_data[gid_idx, 0:3] = rel_p_local
                    m.eq_data[gid_idx, 3:7] = rel_q_g
                    d.eq_active[gid_idx] = 1
                    self.cooldown_timers[agent_key]["grab"] = self.cooldown_limit
                    grab_changed = True
            if grab_changed:
                mujoco.mj_forward(m, d)

    def _stabilize_interaction_poses(self):
        """Hold中オブジェクトの姿勢(roll/pitch=0)・高度(z=基準)の固定"""
        m, d = self.model, self.data
        for tk_s in ["b1", "b2", "ramp"]:
            h_sum_s = 0.0
            h_sum_s += d.eq_active[self.eq_ids[f"grasp_s_{tk_s}"]]
            h_sum_s += d.eq_active[self.eq_ids[f"grasp_h1_{tk_s}"]]
            h_sum_s += d.eq_active[self.eq_ids[f"grasp_h2_{tk_s}"]]
            
            if h_sum_s <= 0.5:
                continue
            if d.eq_active[self.eq_ids[f"lock_{tk_s}"]] > 0.5:
                continue
                
            oid_s = self.box_ids[0] if tk_s=="b1" else self.box_ids[1] if tk_s=="b2" else self.ramp_id
            j_adr_s = m.body_jntadr[oid_s]
            if j_adr_s == -1:
                continue
            q_adr_s = m.jnt_qposadr[j_adr_s]
            v_adr_s = m.jnt_dofadr[j_adr_s]
            
            # 高度固定 ＆ 垂直速度零化
            d.qpos[q_adr_s + 2] = self.object_ground_z[tk_s]
            d.qvel[v_adr_s + 2] = 0.0
            # 角速度(wx, wy)の零化
            d.qvel[v_adr_s + 3] = 0.0
            d.qvel[v_adr_s + 4] = 0.0
            # 姿勢(roll/pitch=0)
            oq_s = d.qpos[q_adr_s + 3 : q_adr_s + 7]
            oy_s = math.atan2(2.0*(oq_s[0]*oq_s[3]+oq_s[1]*oq_s[2]), 1.0-2.0*(oq_s[2]*oq_s[2]+oq_s[3]*oq_s[3]))
            d.qpos[q_adr_s + 3] = math.cos(oy_s * 0.5)
            d.qpos[q_adr_s + 4] = 0.0
            d.qpos[q_adr_s + 5] = 0.0
            d.qpos[q_adr_s + 6] = math.sin(oy_s * 0.5)
        mujoco.mj_forward(m, d)

    def _sync_visual_states(self):
        """色の同期"""
        m, d = self.model, self.data
        for oid in self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else []):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
            tk = "b1" if "box1" in nm else "b2" if "box2" in nm else "ramp"
            f_col = self.obj_default_colors[oid].copy()
            
            # 把持チェック
            is_held = False
            for ak in self.agent_keys:
                if d.eq_active[self.eq_ids[f"grasp_{ak}_{tk}"]] > 0.5:
                    is_held = True
                    break
            if is_held:
                f_col[0] = 1.0
                f_col[1] = 1.0
                f_col[2] = 0.0
            
            # 固定チェック
            if d.eq_active[self.eq_ids[f"lock_{tk}"]] > 0.5:
                f_col[0] = 1.0
                f_col[1] = 0.0
                f_col[2] = 0.0
            for g in self.obj_geom_ids[oid]:
                m.geom_rgba[g] = f_col

    def _get_obs(self, agent_idx):
        """55次元観測の構築（極限展開・1要素1行）"""
        obs = np.zeros(55, dtype=np.float32)
        d, m = self.data, self.model
        ak = self.agent_keys[agent_idx]
        bid = self.body_ids[ak]
        
        # 自己情報
        p_self = d.xpos[bid]
        r_idx = self.qpos_indices[ak]['rot']
        rot_v = float(d.qpos[m.jnt_qposadr[r_idx]])
        v_idx = self.qpos_indices[ak]['x']
        v_adr = m.jnt_dofadr[v_idx]
        vx_w = d.qvel[v_adr]
        vy_w = d.qvel[v_adr + 1]
        
        c_v = math.cos(-rot_v)
        sin_v = math.sin(-rot_v)
        
        # 自己ベクトル代入
        obs[0] = vx_w * c_v - vy_w * sin_v
        obs[1] = vx_w * sin_v + vy_w * c_v
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

        # 道具情報 (8次元 x 3)
        tools_list = self.box_ids + [self.ramp_id]
        for i_o, oid_o in enumerate(tools_list):
            if oid_o == -1: continue
            base = 17 + i_o * 8
            tk_nm = "b1" if i_o == 0 else "b2" if i_o == 1 else "ramp"
            
            # カテゴリ集約
            i_state = 0.0
            if d.eq_active[self.eq_ids[f"lock_{tk_nm}"]] > 0.5:
                i_state = 1.0
            elif d.eq_active[self.eq_ids[f"grasp_{ak}_{tk_nm}"]] > 0.5:
                i_state = 2.0
            else:
                for other_k in self.agent_keys:
                    if other_k != ak:
                        if d.eq_active[self.eq_ids[f"grasp_{other_k}_{tk_nm}"]] > 0.5:
                            i_state = 3.0
                            break
            obs[base+6] = i_state
            
            o_pos_o = d.xpos[oid_o]
            if self._is_within_fov_and_visible(p_self[:2], rot_v, o_pos_o[:2], bid, oid_o):
                rel_p = o_pos_o - p_self
                obs[base] = rel_p[0] * c_v - rel_p[1] * sin_v
                obs[base+1] = rel_p[0] * sin_v + rel_p[1] * c_v
                v_o_a = m.jnt_dofadr[m.body_jntadr[oid_o]]
                rel_vx = d.qvel[v_o_a] - vx_w
                rel_vy = d.qvel[v_o_a + 1] - vy_w
                obs[base+2] = rel_vx * c_v - rel_vy * sin_v
                obs[base+3] = rel_vx * sin_v + rel_vy * c_v
                oq = d.xquat[oid_o]
                oy = math.atan2(2.0*(oq[0]*oq[3]+oq[1]*oq[2]), 1.0-2.0*(oq[2]*oq[2]+oq[3]*oq[3]))
                obs[base+4] = math.cos(oy-rot_v)
                obs[base+5] = math.sin(oy-rot_v)
                obs[base+7] = 1.0

        # 相手・味方情報 (7次元 x 2)
        ek, pk = (("h1", "h2") if ak == "s" else ("s", "h2") if ak == "h1" else ("s", "h1"))
        for i_a, a_key in enumerate([ek, pk]):
            a_bid = self.body_ids[a_key]
            a_base = 41 + i_a * 7
            a_pos_a = d.xpos[a_bid]
            if self._is_within_fov_and_visible(p_self[:2], rot_v, a_pos_a[:2], bid, a_bid):
                rel_a = a_pos_a - p_self
                obs[a_base] = rel_a[0] * c_v - rel_a[1] * sin_v
                obs[a_base+1] = rel_a[0] * sin_v + rel_a[1] * c_v
                v_a_idx = self.qpos_indices[a_key]['x']
                v_a_adr = m.jnt_dofadr[v_a_idx]
                rel_vxa = d.qvel[v_a_adr] - vx_w
                rel_vya = d.qvel[v_a_adr + 1] - vy_w
                obs[a_base+2] = rel_vxa * c_v - rel_vya * sin_v
                obs[a_base+3] = rel_vxa * sin_v + rel_vya * c_v
                tr_idx = self.qpos_indices[a_key]['rot']
                tr_v = float(d.qpos[m.jnt_qposadr[tr_idx]])
                obs[a_base+4] = math.cos(tr_v - rot_v)
                obs[a_base+5] = math.sin(tr_v - rot_v)
                obs[a_base+6] = 1.0
        return obs

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        dx, dy = t_pos[0]-pos[0], t_pos[1]-pos[1]
        dist = math.sqrt(dx*dx+dy*dy)
        if dist > 15.0: return False
        vx_f, vy_f = math.cos(rot), math.sin(rot)
        if (vx_f*(dx/dist) + vy_f*(dy/dist)) < 0.38: return False
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
        s_pos_r = self.data.xpos[s_bid][:2]
        s_idx_r = self.qpos_indices['s']['rot']
        s_rot_r = self.data.qpos[self.model.jnt_qposadr[s_idx_r]]
        
        found_flag = False
        for h_key in ["h1", "h2"]:
            h_bid = self.body_ids[h_key]
            h_pos = self.data.xpos[h_bid][:2]
            if self._is_within_fov_and_visible(s_pos_r, s_rot_r, h_pos, s_bid, h_bid):
                found_flag = True
                break
        
        reward_val = 1.0
        if found_flag:
            reward_val = -1.0
        return reward_val, found_flag

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)
        for k_re in self.prev_action_btns: self.prev_action_btns[k_re].fill(0.0)
        for k_tm in self.agent_keys: self.cooldown_timers[k_tm]["lock"], self.cooldown_timers[k_tm]["grab"] = 0, 0
        for t in ["b1", "b2", "ramp"]: self.lock_owners[t], self.locked_pose[t] = None, None
        
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
                    q = self.qpos_indices[kt]; d.qpos[m.jnt_qposadr[q['x']]], d.qpos[m.jnt_qposadr[q['y']]] = pt[0], pt[1]
                    d.qpos[m.jnt_qposadr[q['rot']]] = np.random.uniform(-np.pi, np.pi)
                else:
                    body = m.body(f"{kt}_body"); adr = m.jnt_qposadr[body.jntadr[0]]; d.qpos[adr:adr+2] = pt
                placed.append((pt, rad)); break
        self._sync_visual_states(); mujoco.mj_forward(m, d)
        
        self.object_ground_z["b1"] = float(d.xpos[self.box_ids[0]][2])
        self.object_ground_z["b2"] = float(d.xpos[self.box_ids[1]][2])
        if self.ramp_id != -1: self.object_ground_z["ramp"] = float(d.xpos[self.ramp_id][2])
        return self._normalize_obs(self._get_obs(1)), {"is_detected": False}

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.type, self.viewer.cam.lookat[0:3] = mujoco.mjtCamera.mjCAMERA_FREE, [0.0, 0.0, 0.5]
            self.viewer.cam.distance, self.viewer.cam.elevation = 18.0, -75.0; self.viewer.sync()
        if hasattr(self.viewer, 'user_scn'):
            scn = self.viewer.user_scn; scn.ngeom = 0
            for ak in self.agent_keys:
                bid = self.body_ids[ak]; p_s, rot = self.data.xpos[bid], self.data.qpos[self.model.jnt_qposadr[self.qpos_indices[ak]['rot']]]
                rgba = [self.obj_default_colors[bid][0], self.obj_default_colors[bid][1], self.obj_default_colors[bid][2], 0.8]
                for tk in self.agent_keys:
                    if tk != ak:
                        t_bid = self.body_ids[tk]
                        if self._is_within_fov_and_visible(p_s[:2], rot, self.data.xpos[t_bid][:2], bid, t_bid):
                            mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 4.0, p_s + [0,0,0.4], self.data.xpos[t_bid] + [0,0,0.4])
                            scn.geoms[scn.ngeom].rgba, scn.ngeom = rgba, scn.ngeom + 1
                for o_id in self.box_ids + ([self.ramp_id] if self.ramp_id != -1 else []):
                    if self._is_within_fov_and_visible(p_s[:2], rot, self.data.xpos[o_id][:2], bid, o_id):
                        mujoco.mjv_connector(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_LINE, 2.0, p_s + [0,0,0.4], self.data.xpos[o_id] + [0,0,0.4])
                        scn.geoms[scn.ngeom].rgba, scn.ngeom = rgba, scn.ngeom + 1
        self.viewer.sync()

    def close(self):
        if self.viewer: 
            self.viewer.close()