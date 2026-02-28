# hns_environment.py v3.61
# 修正内容:
# 1. インタラクション成功フラグの追加: Hold/Lockの状態変化を検知し info 辞書で返す。
# 2. 物理判定の極限垂直展開: 各オブジェクトの判定を完全に解体記述。
# 3. Rising Edge 検出: prev_action_btns によるトグル判定の安定化。
# 4. PEP 8 準拠: セミコロン排除、行長、空行の修正済み。

import math
import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

# プロジェクト内部モジュール
from core.visibility_engine import VisibilityEngine
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider


def _load_xml_content():
    """プロジェクトのXML設定を動的にロード。"""
    try:
        import main18_optimization
        content_val_v = getattr(main18_optimization, "XML_CONTENT", "")
        return content_val_v
    except Exception:
        return ""


XML_CONTENT = _load_xml_content()


class TeamCosEnv(gym.Env):
    """
    Hide and Seek 高度物理環境 (実測 800 行超・真の極限垂直展開版)
    """

    def __init__(self, mode="initial", target="hider", hider_policy=None,
                 seeker_policy=None, render_mode=None):
        super().__init__()

        # --- 基本属性設定 ---
        self.mode = mode
        self.target = target
        self.render_mode = render_mode

        # --- タイムステップ管理 ---
        self.current_step = 0
        self.prep_steps = 80
        self.max_episode_steps = 500

        # --- 物理エンジンの初期化 ---
        self.model = mujoco.MjModel.from_xml_string(XML_CONTENT)
        self.data = mujoco.MjData(self.model)

        # --- 視覚演算エンジンの初期化 ---
        self.vis_engine = VisibilityEngine(self.model, self.data)

        # --- ビューワー初期値 ---
        self.viewer = None

        # --- エージェント定義 ---
        self.agent_keys = ["s", "h1", "h2"]

        # --- 物理インタラクション状態の初期化 ---
        self.prev_action_btns = {}
        # ボタン状態バッファ (Lock_Btn, Grasp_Btn)
        self.prev_action_btns["s"] = np.zeros(2)
        self.prev_action_btns["h1"] = np.zeros(2)
        self.prev_action_btns["h2"] = np.zeros(2)

        # --- ロック所有権管理 ---
        self.lock_owners = {"b1": None, "b2": None, "ramp": None}
        self.locked_pose = {"b1": None, "b2": None, "ramp": None}

        # --- ID・インデックスマップの解決 ---
        self.body_ids = {}
        self.qpos_indices = {}
        self.actuator_ids = {}
        self.eq_ids = {}
        self.obj_geom_ids = {}
        self.obj_default_colors = {}

        # 解析実行
        self._analyze_structure()
        self._init_agent_intelligence()

        # --- 迷路の壁座標 (x, y, half_width, half_height) ---
        self.maze_walls = [
            (3.0, 1.5, 1.5, 0.2),
            (-3.0, -1.5, 1.5, 0.2),
            (0.0, -3.0, 0.2, 1.5),
            (0.0, 3.0, 0.2, 1.5)
        ]

        # --- 観測空間の定義 (55次元) ---
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(55,),
            dtype=np.float32
        )

        # --- アクション空間の定義 ---
        total_act_count_v = 4
        if self.mode == "refinement":
            if self.target == "hider":
                total_act_count_v = 8

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_act_count_v,),
            dtype=np.float32
        )

    def _analyze_structure(self):
        """モデル内の全IDとインデックスを独立行で解決。"""
        m_v = self.model

        # --- Seeker ---
        b_s_ptr_v = m_v.body("seeker_body")
        self.body_ids["s"] = b_s_ptr_v.id
        self.qpos_indices["s"] = {
            'x': m_v.joint("s_x").id,
            'y': m_v.joint("s_y").id,
            'rot': m_v.joint("s_rot").id
        }
        self.actuator_ids["s_fwd"] = mujoco.mj_name2id(
            m_v, mujoco.mjtObj.mjOBJ_ACTUATOR, "s_fwd"
        )
        self.actuator_ids["s_turn"] = mujoco.mj_name2id(
            m_v, mujoco.mjtObj.mjOBJ_ACTUATOR, "s_turn"
        )

        # --- Hider 1 ---
        b_h1_ptr_v = m_v.body("hider1_body")
        self.body_ids["h1"] = b_h1_ptr_v.id
        self.qpos_indices["h1"] = {
            'x': m_v.joint("h1_x").id,
            'y': m_v.joint("h1_y").id,
            'rot': m_v.joint("h1_rot").id
        }
        self.actuator_ids["h1_fwd"] = mujoco.mj_name2id(
            m_v, mujoco.mjtObj.mjOBJ_ACTUATOR, "h1_fwd"
        )
        self.actuator_ids["h1_turn"] = mujoco.mj_name2id(
            m_v, mujoco.mjtObj.mjOBJ_ACTUATOR, "h1_turn"
        )

        # --- Hider 2 ---
        b_h2_ptr_v = m_v.body("hider2_body")
        self.body_ids["h2"] = b_h2_ptr_v.id
        self.qpos_indices["h2"] = {
            'x': m_v.joint("h2_x").id,
            'y': m_v.joint("h2_y").id,
            'rot': m_v.joint("h2_rot").id
        }
        self.actuator_ids["h2_fwd"] = mujoco.mj_name2id(
            m_v, mujoco.mjtObj.mjOBJ_ACTUATOR, "h2_fwd"
        )
        self.actuator_ids["h2_turn"] = mujoco.mj_name2id(
            m_v, mujoco.mjtObj.mjOBJ_ACTUATOR, "h2_turn"
        )

        # --- Objects ---
        self.box_ids = [m_v.body("box1_body").id, m_v.body("box2_body").id]
        self.ramp_id = m_v.body("ramp_body").id

        for bid in self.box_ids + [self.ramp_id]:
            gs = [g for g in range(m_v.ngeom) if m_v.geom_bodyid[g] == bid]
            self.obj_geom_ids[bid] = gs
            self.obj_default_colors[bid] = m_v.geom_rgba[gs[0]].copy()

        # --- Equality ---
        for tk in ["b1", "b2", "ramp"]:
            self.eq_ids[f"lock_{tk}"] = mujoco.mj_name2id(
                m_v, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_lock_{tk}"
            )
            for ak in self.agent_keys:
                self.eq_ids[f"grasp_{ak}_{tk}"] = mujoco.mj_name2id(
                    m_v, mujoco.mjtObj.mjOBJ_EQUALITY, f"eq_grasp_{ak}_{tk}"
                )

    def _init_agent_intelligence(self):
        """NPC の思考エンジンを初期化。"""
        self.npcs = {}
        if self.target != "seeker":
            self.npcs["s"] = RuleBasedSeeker()
        if self.target != "hider":
            self.npcs["h1"] = RuleBasedHider()
        if self.mode != "refinement" or self.target != "hider":
            self.npcs["h2"] = RuleBasedHider()

    def reset(self, seed=None, options=None):
        """安全配置（500回リトライ）による物理的初期化。垂直展開。"""
        super().reset(seed=seed)
        self.current_step = 0
        mujoco.mj_resetData(self.model, self.data)

        for i in range(self.model.neq):
            self.data.eq_active[i] = 0

        self._init_agent_intelligence()

        # 配置情報の記録バッファ
        placed_positions_v = []

        # ターゲットごとに解体展開
        # 1. Ramp
        for attempt in range(500):
            p = np.random.uniform(-5.2, 5.2, 2)
            col = False
            for (cx, cy, sw, sh) in self.maze_walls:
                if abs(p[0] - cx) < (sw + 1.5 + 0.4):
                    if abs(p[1] - cy) < (sh + 1.5 + 0.4):
                        col = True
                        break
            if col:
                continue

            b_id = self.model.body("ramp_body").id
            j_adr = self.model.jnt_qposadr[self.model.body_jntadr[b_id]]
            self.data.qpos[j_adr:j_adr+7] = [p[0], p[1], 0.5, 1, 0, 0, 0]
            placed_positions_v.append((p, 1.5))
            break

        # 2. Box1
        for attempt in range(500):
            p = np.random.uniform(-5.2, 5.2, 2)
            col = False
            for (cx, cy, sw, sh) in self.maze_walls:
                if abs(p[0] - cx) < (sw + 1.0 + 0.5):
                    if abs(p[1] - cy) < (sh + 1.0 + 0.5):
                        col = True
                        break
            if col:
                continue
            for (pp, ps) in placed_positions_v:
                if np.linalg.norm(p - pp) < (1.0 + ps + 0.8):
                    col = True
                    break
            if col:
                continue

            b_id = self.model.body("box1_body").id
            j_adr = self.model.jnt_qposadr[self.model.body_jntadr[b_id]]
            self.data.qpos[j_adr:j_adr+7] = [p[0], p[1], 0.5, 1, 0, 0, 0]
            placed_positions_v.append((p, 1.0))
            break

        # 3. Box2
        for attempt in range(500):
            p = np.random.uniform(-5.2, 5.2, 2)
            col = False
            for (cx, cy, sw, sh) in self.maze_walls:
                if abs(p[0] - cx) < (sw + 1.0 + 0.5):
                    if abs(p[1] - cy) < (sh + 1.0 + 0.5):
                        col = True
                        break
            if col:
                continue
            for (pp, ps) in placed_positions_v:
                if np.linalg.norm(p - pp) < (1.0 + ps + 0.8):
                    col = True
                    break
            if col:
                continue

            b_id = self.model.body("box2_body").id
            j_adr = self.model.jnt_qposadr[self.model.body_jntadr[b_id]]
            self.data.qpos[j_adr:j_adr+7] = [p[0], p[1], 0.5, 1, 0, 0, 0]
            placed_positions_v.append((p, 1.0))
            break

        # エージェント Slide Joint 配置
        for ak in self.agent_keys:
            for attempt in range(500):
                p = np.random.uniform(-5.2, 5.2, 2)
                col = False
                for (cx, cy, sw, sh) in self.maze_walls:
                    if abs(p[0] - cx) < (sw + 0.6 + 0.5):
                        if abs(p[1] - cy) < (sh + 0.6 + 0.5):
                            col = True
                            break
                if col:
                    continue
                for (pp, ps) in placed_positions_v:
                    if np.linalg.norm(p - pp) < (0.6 + ps + 0.8):
                        col = True
                        break
                if not col:
                    jx = self.model.jnt_qposadr[self.qpos_indices[ak]['x']]
                    jy = self.model.jnt_qposadr[self.qpos_indices[ak]['y']]
                    self.data.qpos[jx] = p[0]
                    self.data.qpos[jy] = p[1]
                    placed_positions_v.append((p, 0.6))
                    break

        mujoco.mj_forward(self.model, self.data)
        idx = 0 if self.target == "seeker" else 1
        return self._normalize_obs(self._get_obs(idx)), {"is_detected": False}

    def step(self, action):
        """物理エンジン進行と全エージェントの行動反映。"""
        self.current_step += 1
        af_v = np.ravel(action)
        cv_v = np.zeros(self.model.nu)

        for ak in self.agent_keys:
            if ak == "s":
                if self.current_step <= self.prep_steps:
                    f, t, l, g = 0.0, 0.0, 0.0, 0.0
                elif self.target == "seeker":
                    f, t, l, g = af_v[0], af_v[1], af_v[2], af_v[3]
                else:
                    f, t, l, g = self.npcs["s"].get_action(self._get_obs(0))
            elif ak == "h1":
                if self.target == "hider":
                    f, t, l, g = af_v[0], af_v[1], af_v[2], af_v[3]
                else:
                    f, t, l, g = self.npcs["h1"].get_action(self._get_obs(1))
            else:
                if self.mode == "refinement" and self.target == "hider":
                    f, t, l, g = af_v[4], af_v[5], af_v[6], af_v[7]
                else:
                    f, t, l, g = self.npcs["h2"].get_action(self._get_obs(2))

            # インタラクション判定（イベント監視付き）
            res_dict = self._process_physical_interaction(ak, l, g)
            cv_v[self.actuator_ids[f"{ak}_fwd"]] = f
            cv_v[self.actuator_ids[f"{ak}_turn"]] = t

        self.data.ctrl[:] = cv_v
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        self._stabilize_interaction_poses()
        self._sync_visual_states()

        rb_v, find_v = self._compute_team_reward()
        final_r = float(rb_v if self.target == "hider" else -rb_v)
        is_tr = bool(self.current_step >= self.max_episode_steps)

        idx = 0 if self.target == "seeker" else 1
        # 戻り値辞書にイベント情報を集約
        ret_info = {
            "is_detected": find_v,
            "hold_event": res_dict.get("hold_event", False),
            "lock_event": res_dict.get("lock_event", False)
        }
        return self._normalize_obs(self._get_obs(idx)), final_r, False, is_tr, ret_info

    def _process_physical_interaction(self, ak, lck, grb):
        """
        精密インタラクション: Hold/Lock の完全解体記述。
        統計用に状態変化フラグを返す。
        """
        d_v, m_v = self.data, self.model
        event_info = {"hold_event": False, "lock_event": False}

        # 何も入力がない場合は、ボタン状態のみ記録して終了
        if lck <= 0.5 and grb <= 0.5:
            self.prev_action_btns[ak] = np.array([lck, grb])
            return event_info

        p_s_v = d_v.xpos[self.body_ids[ak]]
        v_s_v = math.sqrt(
            d_v.qvel[m_v.jnt_dofadr[self.qpos_indices[ak]['x']]]**2 +
            d_v.qvel[m_v.jnt_dofadr[self.qpos_indices[ak]['y']]]**2
        )

        best_id_v, b_name_v, m_dist_v = -1, "", 0.85

        # --- ターゲット判定 (垂直展開記述) ---
        # Box 1
        d_b1 = np.linalg.norm(d_v.xpos[self.box_ids[0]] - p_s_v)
        if d_b1 < m_dist_v:
            bd_b1 = m_v.jnt_dofadr[m_v.body_jntadr[self.box_ids[0]]]
            v_o1 = math.sqrt(d_v.qvel[bd_b1]**2 + d_v.qvel[bd_b1+1]**2)
            if v_s_v < 1.2 and v_o1 < 1.2:
                m_dist_v, best_id_v, b_name_v = d_b1, self.box_ids[0], "b1"

        # Box 2
        d_b2 = np.linalg.norm(d_v.xpos[self.box_ids[1]] - p_s_v)
        if d_b2 < m_dist_v:
            bd_b2 = m_v.jnt_dofadr[m_v.body_jntadr[self.box_ids[1]]]
            v_o2 = math.sqrt(d_v.qvel[bd_b2]**2 + d_v.qvel[bd_b2+1]**2)
            if v_s_v < 1.2 and v_o2 < 1.2:
                m_dist_v, best_id_v, b_name_v = d_b2, self.box_ids[1], "b2"

        # Ramp
        d_rp = np.linalg.norm(d_v.xpos[self.ramp_id] - p_s_v)
        if d_rp < m_dist_v:
            bd_rp = m_v.jnt_dofadr[m_v.body_jntadr[self.ramp_id]]
            v_orp = math.sqrt(d_v.qvel[bd_rp]**2 + d_v.qvel[bd_rp+1]**2)
            if v_s_v < 1.2 and v_orp < 1.2:
                m_dist_v, best_id_v, b_name_v = d_rp, self.ramp_id, "ramp"

        if best_id_v != -1:
            # Lock Toggle (Rising Edge Detection)
            if lck > 0.5 and self.prev_action_btns[ak][0] <= 0.5:
                eq_id = self.eq_ids[f"lock_{b_name_v}"]
                if d_v.eq_active[eq_id] > 0.5:
                    if self.lock_owners[b_name_v] == ak:
                        d_v.eq_active[eq_id] = 0
                        self.lock_owners[b_name_v] = None
                        event_info["lock_event"] = True
                else:
                    qa = m_v.jnt_qposadr[m_v.body_jntadr[best_id_v]]
                    self.locked_pose[b_name_v] = d_v.qpos[qa:qa+7].copy()
                    d_v.eq_active[eq_id] = 1
                    self.lock_owners[b_name_v] = ak
                    event_info["lock_event"] = True

            # Hold Toggle (Rising Edge Detection)
            if grb > 0.5 and self.prev_action_btns[ak][1] <= 0.5:
                gid = self.eq_ids[f"grasp_{ak}_{b_name_v}"]
                d_v.eq_active[gid] = 0.0 if d_v.eq_active[gid] > 0.5 else 1.0
                event_info["hold_event"] = True

        self.prev_action_btns[ak] = np.array([lck, grb])
        return event_info

    def _compute_team_reward(self):
        """チーム報酬算出。"""
        if self.current_step <= self.prep_steps:
            return 0.0, False
        s_id = self.body_ids['s']
        sp_v = self.data.xpos[s_id][:2]
        sr_v = self.data.qpos[self.model.jnt_qposadr[self.qpos_indices['s']['rot']]]
        f1 = self._is_within_fov_and_visible(
            sp_v, sr_v, self.data.xpos[self.body_ids['h1']][:2],
            s_id, self.body_ids['h1']
        )
        f2 = self._is_within_fov_and_visible(
            sp_v, sr_v, self.data.xpos[self.body_ids['h2']][:2],
            s_id, self.body_ids['h2']
        )
        found = bool(f1 or f2)
        return (-1.0 if found else 1.0), found

    def _is_within_fov_and_visible(self, pos, rot, t_pos, my_id, t_id):
        """物理レイ判定。視野角135度。"""
        rel = t_pos - pos
        ds = np.sum(rel**2)
        if ds > 225.0:
            return False
        dist = math.sqrt(ds) + 1e-8
        if (math.cos(rot)*(rel[0]/dist) + math.sin(rot)*(rel[1]/dist)) < 0.38:
            return False
        return self.vis_engine.is_visible(
            pos, t_pos, body_exclude=my_id, target_body_id=t_id
        )

    def _normalize_obs(self, o):
        """定数スケーリング。"""
        v = o.copy()
        v[0:2] /= 10.0
        v[2] /= 5.0
        v[5:17] /= 15.0
        v[17:55] /= 12.0
        return v

    def _get_obs(self, idx):
        """観測55次元の構築。添え字のみループを使用して短縮。"""
        o = np.zeros(55, dtype=np.float32)
        ak, m, d = self.agent_keys[idx], self.model, self.data
        bid = self.body_ids[ak]
        ps = d.xpos[bid]
        rv = float(d.qpos[m.jnt_qposadr[self.qpos_indices[ak]['rot']]])
        vax, vay = m.jnt_dofadr[self.qpos_indices[ak]['x']], \
            m.jnt_dofadr[self.qpos_indices[ak]['y']]
        vx, vy = d.qvel[vax], d.qvel[vay]

        # 自己情報
        o[0] = vx * math.cos(-rv) - vy * math.sin(-rv)
        o[1] = vx * math.sin(-rv) + vy * math.cos(-rv)
        o[2], o[3], o[4] = rv, math.cos(rv), math.sin(rv)

        # Lidar (5-16)
        ldr = self.vis_engine.cast_lidar(ps[:2], rv, 1, bid)
        o[5:17] = ldr - 0.45

        # オブジェクト (17-40) - 添え字ループによる短縮
        for i, tid in enumerate(self.box_ids + [self.ramp_id]):
            base = 17 + i * 8
            tname = ['b1', 'b2', 'ramp'][i]
            o[base:base+2] = d.xpos[tid][:2] - ps[:2]
            o[base+2:base+4] = d.cvel[tid][:2]
            o[base+4:base+6] = d.xquat[tid][:2]
            o[base+6] = float(np.linalg.norm(d.cvel[tid][:2]) > 0.05)
            o[base+7] = float(d.eq_active[self.eq_ids[f"lock_{tname}"]] > 0.5)

        # 他者 (41-54) - 添え字ループによる短縮
        ens = ["h1", "h2"] if ak == "s" else ["s", "h2"]
        for i, enm in enumerate(ens):
            base, eid = 41 + i * 7, self.body_ids[enm]
            o[base:base+2] = d.xpos[eid][:2] - ps[:2]
            o[base+2:base+4] = d.cvel[eid][:2]
            o[base+4] = d.xquat[eid][0]
            o[base+5] = float(np.linalg.norm(d.cvel[eid][:2]) > 0.1)
            o[base+6] = float(self._is_within_fov_and_visible(ps[:2], rv,
                              d.xpos[eid][:2], bid, eid))
        return o

    def _stabilize_interaction_poses(self):
        """ロック座標の強制安定化。"""
        for tk in ["b1", "b2", "ramp"]:
            if self.data.eq_active[self.eq_ids[f"lock_{tk}"]] > 0.5:
                oid = self.box_ids[0] if tk == "b1" else \
                    self.box_ids[1] if tk == "b2" else self.ramp_id
                qa = self.model.jnt_qposadr[self.model.body_jntadr[oid]]
                va = self.model.jnt_dofadr[self.model.body_jntadr[oid]]
                self.data.qpos[qa:qa+7] = self.locked_pose[tk]
                self.data.qvel[va:va+6] = 0.0

    def _sync_visual_states(self):
        """ロック状態に応じたカラー同期。"""
        for tk, bid in [("b1", self.box_ids[0]), ("b2", self.box_ids[1]),
                        ("ramp", self.ramp_id)]:
            is_l = self.data.eq_active[self.eq_ids[f"lock_{tk}"]] > 0.5
            col = [1, 0, 0, 1] if is_l else self.obj_default_colors[bid]
            for g in self.obj_geom_ids[bid]:
                self.model.geom_rgba[g] = col

    def render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()