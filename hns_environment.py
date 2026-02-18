# hns_environment.py
# 演習第25回：MuJoCo環境定義と53次元観測構築ロジックの完全復刻版
#
# 【修正内容】
# 1. reset メソッドで seed を受け取り、self.np_random を初期化するように修正。
# 2. オブジェクト配置に self.np_random を使用し、再現性を確保。
# 3. 53次元観測マッピングのロジックを最後まで記述。

import os
import tempfile
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from visibility_engine import VisibilityEngine

# ==========================================
# 0. XML 定義 (手作業によるコピーをそのまま維持)
# ==========================================
HNS_XML = """
<mujoco>
    <!-- 重力とシミュレーションステップ時間の設定 -->
    <option gravity="0 0 -9.81" timestep="0.005"/>
    
    <!-- ビジュアル設定: ヘッドライトの調整 -->
    <visual>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    </visual>
    
    <asset>
        <!-- 床のグリッドテクスチャ -->
        <texture name="grid_tex" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid_mat" texture="grid_tex" texrepeat="1 1" reflectance="0.2"/>
        
        <!-- スロープ用のメッシュ定義 (三角柱のような形状) -->
        <mesh name="ramp_mesh" 
              vertex="-0.6666 -0.5 0.0   0.6666 -0.5 0.0   0.6666 -0.5 1.0   -0.6666 0.5 0.0   0.6666 0.5 0.0   0.6666 0.5 1.0" 
              face="0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0"/>
    </asset>

    <worldbody>
        <!-- 照明の位置と方向 -->
        <light pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        
        <!-- 床平面 -->
        <geom name="floor" type="plane" size="6 6 0.1" material="grid_mat" friction="1.0 0.05 0.0001" solref="0.04 1"/>
        
        <!-- 外壁: 北(N), 南(S), 東(E), 西(W) -->
        <geom name="wall_n" type="box" size="6.2 0.1 4.0" pos="0 6.1 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="wall_s" type="box" size="6.2 0.1 4.0" pos="0 -6.1 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="wall_e" type="box" size="0.1 6 4.0" pos="6.1 0 4.0" rgba="0.7 0.7 0.7 0.3"/>
        <geom name="wall_w" type="box" size="0.1 6 4.0" pos="-6.1 0 4.0" rgba="0.7 0.7 0.7 0.3"/>
        
        <!-- 内部の迷路壁 (Hiderが隠れるための構造物) -->
        <geom name="maze_w0" type="box" size="1.5 0.2 0.5" pos="3.0 1.5 0.5" rgba="0.0 0.7 0.7 1"/>
        <geom name="maze_w1" type="box" size="1.5 0.2 0.5" pos="-3.0 -1.5 0.5" rgba="0.0 0.7 0.7 1"/>
        <geom name="maze_w2" type="box" size="0.2 1.5 0.5" pos="0 -3.0 0.5" rgba="0.0 0.7 0.7 1"/>
        <geom name="maze_w3" type="box" size="0.2 1.5 0.5" pos="0 3.0 0.5" rgba="0.0 0.7 0.7 1"/>
        
        <!-- スロープ物体 (移動可能だが重い) -->
        <body name="ramp_body" pos="0 0 0">
            <!-- 慣性モーメントの設定 -->
            <inertial pos="0.3 0 0.25" mass="50" diaginertia="10 10 20"/>
            <!-- 自由関節: 自由に動ける -->
            <joint type="free" name="ramp_joint" damping="500.0"/>
            <!-- 見た目のメッシュ -->
            <geom type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0" rgba="0 1 0 1"/>
            <!-- 物理的な当たり判定用ジオメトリ -->
            <geom name="ramp_slope_surface" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.3" friction="1.2 0.01 0"/>
            <geom name="ramp_back_panel" type="box" size="0.02 0.5 0.5" pos="0.6666 0 0.5" rgba="0 1 0 0.3"/>
            <!-- 重り: 安定させるため -->
            <geom name="ramp_inner_weight" type="box" size="0.3333 0.5 0.25" pos="0.3333 0 0.25" rgba="0 1 0 0.3" mass="30" solimp="0.95 0.99 0.001"/> 
        </body>
        
        <!-- 移動可能な箱 1 -->
        <body name="box1_body" pos="2 -2 0.5">
            <joint name="box1_joint" type="free" damping="100.0"/>
            <geom name="box1_geom" type="box" size="0.6 0.6 0.5" rgba="0.6 0.4 0.2 1" mass="100" solref="0.02 1" condim="3" friction="1.0 0.005 0.0001"/>
        </body>

        <!-- 移動可能な箱 2 -->
        <body name="box2_body" pos="-2 2 0.5">
            <joint name="box2_joint" type="free" damping="100.0"/>
            <geom name="box2_geom" type="box" size="0.6 0.6 0.5" rgba="0.7 0.5 0.3 1" mass="100" solref="0.02 1" condim="3" friction="1.0 0.005 0.0001"/>
        </body>
        
        <!-- Seeker (鬼) エージェント -->
        <body name="seeker_anchor" pos="0 0 0.5">
            <!-- X, Y方向へのスライド移動と、Z軸回転の関節 -->
            <joint name="s_x" type="slide" axis="1 0 0" damping="40"/>
            <joint name="s_y" type="slide" axis="0 1 0" damping="40"/>
            <!-- Z方向は少しだけ動ける(浮き上がり防止) -->
            <joint name="s_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2" damping="20"/>
            <joint name="s_rot" type="hinge" axis="0 0 1" damping="50" armature="3.0"/>
            
            <body name="seeker_body">
                <!-- 推力発生位置 -->
                <site name="seeker_thrust_site" pos="0 0 0"/>
                <!-- 状態表示用ラベルサイト -->
                <site name="site_s_label" pos="0 0 1.8" type="sphere" size="0.01" rgba="0 0 0 0"/>
                
                <!-- 底部: 摩擦のある球体 -->
                <geom name="seeker_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="1.2 0.01 0"/>
                <!-- 本体: 赤いカプセル -->
                <geom name="seeker_capsule" type="capsule" size="0.3 0.2" rgba="0.9 0.1 0.1 1" mass="5"/>
                <!-- 鼻: 向きを示す -->
                <geom name="seeker_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <!-- 尻尾 -->
                <geom name="seeker_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.9 0.1 0.1 1" contype="0" conaffinity="0"/>
            </body>
        </body>
        
        <!-- Hider 1 (子1) エージェント -->
        <body name="hider1_anchor" pos="0 0 0.5">
            <joint name="h1_x" type="slide" axis="1 0 0" damping="40"/>
            <joint name="h1_y" type="slide" axis="0 1 0" damping="40"/>
            <joint name="h1_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2" damping="20"/>
            <joint name="h1_rot" type="hinge" axis="0 0 1" damping="50" armature="3.0"/>
            <body name="hider1_body">
                <site name="hider1_thrust_site" pos="0 0 0"/>
                <site name="site_h1_label" pos="0 0 1.2" type="sphere" size="0.01" rgba="0 0 0 0"/>
                
                <geom name="hider1_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="1.2 0.01 0"/>
                <!-- 本体: 青いカプセル -->
                <geom name="hider1_capsule" type="capsule" size="0.3 0.2" rgba="0.1 0.1 0.9 1" mass="5"/>
                <geom name="hider1_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <geom name="hider1_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.1 0.1 0.9 1" contype="0" conaffinity="0"/>
            </body>
        </body>

        <!-- Hider 2 (子2) エージェント -->
        <body name="hider2_anchor" pos="0 0 0.5">
            <joint name="h2_x" type="slide" axis="1 0 0" damping="40"/>
            <joint name="h2_y" type="slide" axis="0 1 0" damping="40"/>
            <joint name="h2_z" type="slide" axis="0 0 1" limited="true" range="-0.05 1.2" damping="20"/>
            <joint name="h2_rot" type="hinge" axis="0 0 1" damping="50" armature="3.0"/>
            <body name="hider2_body">
                <site name="hider2_thrust_site" pos="0 0 0"/>
                <site name="site_h2_label" pos="0 0 1.5" type="sphere" size="0.01" rgba="0 0 0 0"/>
                
                <geom name="hider2_btm" type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="1.2 0.01 0"/>
                <!-- 本体: 水色のカプセル -->
                <geom name="hider2_capsule" type="capsule" size="0.3 0.2" rgba="0.1 0.6 0.9 1" mass="5"/>
                <geom name="hider2_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <geom name="hider2_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.1 0.6 0.9 1" contype="0" conaffinity="0"/>
            </body>
        </body>
    </worldbody>

    <!-- 等価拘束: 物体を掴む/ロックする機能の実装 -->
    <equality>
        <!-- Hider1がBox1/Box2を掴むための溶接拘束 -->
        <weld name="eq_grasp1_b1" body1="hider1_body" body2="box1_body" active="false" relpose="0 0 0 1 0 0 0" solref="0.08 1" solimp="0.9 0.95 0.001"/>
        <weld name="eq_grasp1_b2" body1="hider1_body" body2="box2_body" active="false" relpose="0 0 0 1 0 0 0" solref="0.08 1" solimp="0.9 0.95 0.001"/>
        
        <!-- Hider2がBox1/Box2を掴むための溶接拘束 -->
        <weld name="eq_grasp2_b1" body1="hider2_body" body2="box1_body" active="false" relpose="0 0 0 1 0 0 0" solref="0.08 1" solimp="0.9 0.95 0.001"/>
        <weld name="eq_grasp2_b2" body1="hider2_body" body2="box2_body" active="false" relpose="0 0 0 1 0 0 0" solref="0.08 1" solimp="0.9 0.95 0.001"/>
        
        <!-- Box1/Box2を空間に固定（ロック）するための溶接拘束 -->
        <weld name="eq_lock_b1" body1="world" body2="box1_body" active="false" solref="0.02 1" solimp="0.95 0.99 0.001"/>
        <weld name="eq_lock_b2" body1="world" body2="box2_body" active="false" solref="0.02 1" solimp="0.95 0.99 0.001"/>
    </equality>

    <!-- アクチュエータ: エージェントの動きを制御 -->
    <actuator>
        <!-- Seekerの移動と回転 -->
        <general name="s_fwd" site="seeker_thrust_site" gear="1 0 0 0 0 0" gainprm="9000" ctrlrange="-1 1"/>
        <general name="s_turn" joint="s_rot" gear="0.6" gainprm="500" ctrlrange="-1 1"/>
        
        <!-- Hider1の移動と回転 -->
        <general name="h1_fwd" site="hider1_thrust_site" gear="1 0 0 0 0 0" gainprm="9000" ctrlrange="-1 1"/>
        <general name="h1_turn" joint="h1_rot" gear="0.6" gainprm="500" ctrlrange="-1 1"/>
        
        <!-- Hider2の移動と回転 -->
        <general name="h2_fwd" site="hider2_thrust_site" gear="1 0 0 0 0 0" gainprm="9000" ctrlrange="-1 1"/>
        <general name="h2_turn" joint="h2_rot" gear="0.6" gainprm="500" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""

class TeamCosEnv(gym.Env):
    """
    かくれんぼ強化学習環境。
    手作業でコピーいただいた XML 設定を維持し、物理構造に基づく
    絶対座標配置ロジックと53次元観測を実装。
    """
    def __init__(self, lidar_mode=1):
        super().__init__()
        
        # モデルとデータのロード
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(HNS_XML)
            xml_path = f.name
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        os.unlink(xml_path)
        
        # 演算エンジンの同期
        self.vis_engine = VisibilityEngine(self.model, self.data)
        self.lidar_mode = lidar_mode
        
        # 観測・アクション空間の定義
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32)
        # 各エージェント[推力, 予備, 回転]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # 内部マッピング管理 (XMLの構造に準拠)
        self.agent_bodies = ["seeker_body", "hider1_body", "hider2_body"]
        self.obs_buffer = np.zeros(53, dtype=np.float32)
        self.obs_memo = {}
        self.prev_pos = np.zeros((3, 2))

    def _get_obs(self, agent_id):
        """引き継ぎ資料準拠：53次元観測の構築"""
        if agent_id in self.obs_memo:
            return self.obs_memo[agent_id]
        
        d = self.data
        m = self.model
        
        # 物理インデックス
        # ramp(7), box1(7), box2(7) -> 21 qpos offset
        q_off = 21 
        q_idx = q_off + (agent_id * 4) 
        v_off = 18 # freejoint (6*3) velocity offset
        v_idx = v_off + (agent_id * 4)
        
        # 物理エンジンから絶対座標と速度を取得
        agent_body_id = m.body(self.agent_bodies[agent_id]).id
        pos = d.xpos[agent_body_id][:2]
        vel = d.cvel[agent_body_id][:2]
        rot = d.qpos[q_idx + 3]

        self.obs_buffer.fill(0.0)

        # --- [00-04] Self ---
        self.obs_buffer[0] = vel[0]
        self.obs_buffer[1] = vel[1]
        self.obs_buffer[2] = rot
        self.obs_buffer[3] = np.cos(rot)
        self.obs_buffer[4] = np.sin(rot)
        
        # --- [05-16] Lidar ---
        self.obs_buffer[5:17] = self.vis_engine.cast_lidar(pos, mode=self.lidar_mode, body_exclude=agent_body_id)
        
        # --- [17-32] Box1, Box2 (8次元x2) ---
        objs = ["box1_body", "box2_body"]
        for i, name in enumerate(objs):
            offset = 17 + (i * 8)
            obj_id = m.body(name).id
            obj_pos = d.xpos[obj_id][:2]
            
            if self.vis_engine.is_visible(pos, obj_pos, body_exclude=agent_body_id):
                # 相対座標
                self.obs_buffer[offset] = obj_pos[0] - pos[0]
                self.obs_buffer[offset + 1] = obj_pos[1] - pos[1]
                # 相対速度
                obj_vel = d.cvel[obj_id][0:2]
                self.obs_buffer[offset + 2] = obj_vel[0] - vel[0]
                self.obs_buffer[offset + 3] = obj_vel[1] - vel[1]
                
                # 箱(freejoint) の回転取得
                q_b_start = 7 + (i * 7) + 3 
                q_b = d.qpos[q_b_start : q_b_start + 4]
                r_b = np.arctan2(2.0*(q_b[0]*q_b[3]+q_b[1]*q_b[2]), 1.0-2.0*(q_b[2]*q_b[2]+q_b[3]*q_b[3]))
                self.obs_buffer[offset + 4] = np.cos(r_b)
                self.obs_buffer[offset + 5] = np.sin(r_b)
                self.obs_buffer[offset + 7] = 1.0 

        # --- [33-39] Ramp (7次元) ---
        ramp_id = m.body("ramp_body").id
        ramp_pos = d.xpos[ramp_id][:2]
        if self.vis_engine.is_visible(pos, ramp_pos, body_exclude=agent_body_id):
            self.obs_buffer[33] = ramp_pos[0] - pos[0]
            self.obs_buffer[34] = ramp_pos[1] - pos[1]
            r_vel = d.cvel[ramp_id][0:2]
            self.obs_buffer[35] = r_vel[0] - vel[0]
            self.obs_buffer[36] = r_vel[1] - vel[1]
            # Ramp回転
            q_r = d.qpos[3:7]
            r_r = np.arctan2(2.0*(q_r[0]*q_r[3]+q_r[1]*q_r[2]), 1.0-2.0*(q_r[2]*q_r[2]+q_r[3]*q_r[3]))
            self.obs_buffer[37] = np.cos(r_r)
            self.obs_buffer[38] = np.sin(r_r)
            self.obs_buffer[39] = 1.0

        # --- [40-44] Enemy ---
        enemy_id = 0 if agent_id != 0 else 1 
        enemy_body_id = m.body(self.agent_bodies[enemy_id]).id
        enemy_pos = d.xpos[enemy_body_id][:2]
        if self.vis_engine.is_visible(pos, enemy_pos, body_exclude=agent_body_id):
            self.obs_buffer[40] = enemy_pos[0] - pos[0]
            self.obs_buffer[41] = enemy_pos[1] - pos[1]
            e_vel = d.cvel[enemy_body_id][0:2]
            self.obs_buffer[42] = e_vel[0] - vel[0]
            self.obs_buffer[43] = e_vel[1] - vel[1]
            self.obs_buffer[44] = 1.0

        # --- [45-51] Partner ---
        if agent_id != 0: 
            partner_id = 2 if agent_id == 1 else 1
            partner_body_id = m.body(self.agent_bodies[partner_id]).id
            partner_pos = d.xpos[partner_body_id][:2]
            if self.vis_engine.is_visible(pos, partner_pos, body_exclude=agent_body_id):
                self.obs_buffer[45] = partner_pos[0] - pos[0]
                self.obs_buffer[46] = partner_pos[1] - pos[1]
                p_vel = d.cvel[partner_body_id][0:2]
                self.obs_buffer[47] = p_vel[0] - vel[0]
                self.obs_buffer[48] = p_vel[1] - vel[1]
                # パートナーの角度
                r_p = d.qpos[21 + (partner_id * 4) + 3]
                self.obs_buffer[49] = np.cos(r_p)
                self.obs_buffer[50] = np.sin(r_p)
                self.obs_buffer[51] = 1.0

        res = self.obs_buffer.copy()
        self.obs_memo[agent_id] = res
        return res

    def step(self, action):
        """物理進行と報酬計算"""
        h1_body_id = self.model.body("hider1_body").id
        self.prev_pos[1] = self.data.xpos[h1_body_id][:2].copy()

        for i in range(3):
            # Actuator 分配
            self.data.ctrl[i * 2] = action[0]     
            self.data.ctrl[i * 2 + 1] = action[2] 
            
        mujoco.mj_step(self.model, self.data, 5)
        self.obs_memo.clear()
        
        obs = self._get_obs(1) 
        reward = self._compute_reward()
        done = self.data.time > 20.0
        
        return obs, reward, done, False, {}

    def _compute_reward(self):
        """生存・視界報酬ロジック"""
        reward = 0.05
        h1_pos = self.data.xpos[self.model.body("hider1_body").id][:2]
        s_pos = self.data.xpos[self.model.body("seeker_body").id][:2]
        
        is_visible = self.vis_engine.is_visible(
            s_pos, 
            h1_pos, 
            body_exclude=self.model.body("seeker_body").id
        )
        
        if is_visible:
            reward -= 0.1 
            
        return reward

    def reset(self, seed=None, options=None):
        """ 
        再現性を確保したリセットロジック
        """
        # gym.Env の仕様に従い、self.np_random を初期化
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.obs_memo.clear()
        
        # 静的障害物座標
        static_obstacles = [
            np.array([3.0, 1.5, 1.2]), np.array([-3.0, -1.5, 1.2]),
            np.array([0.0, -3.0, 1.2]), np.array([0.0, 3.0, 1.2]),
        ]
        
        # 配置オブジェクト定義 [body_name, qpos_idx, is_free, safety_radius]
        objects_to_place = [
            ["ramp_body", 0, True, 2.0],
            ["box1_body", 7, True, 1.5],
            ["box2_body", 14, True, 1.5],
            ["seeker_anchor", 21, False, 1.5],
            ["hider1_anchor", 25, False, 1.5],
            ["hider2_anchor", 29, False, 1.5],
        ]
        
        placed_positions = []
        
        for name, q_idx, is_free, radius in objects_to_place:
            placed = False
            body_id = self.model.body(name).id
            xml_pos = self.model.body_pos[body_id][:2]
            
            while not placed:
                # self.np_random を使用
                test_pos = self.np_random.uniform(-5.2, 5.2, size=2)
                
                # 壁チェック
                is_safe = True
                for obs in static_obstacles:
                    if np.linalg.norm(test_pos - obs[:2]) < (radius + 0.2):
                        is_safe = False
                        break
                if not is_safe: continue
                    
                # 他オブジェクトチェック
                for prev_pos, prev_radius in placed_positions:
                    if np.linalg.norm(test_pos - prev_pos) < (radius + prev_radius):
                        is_safe = False
                        break
                
                if is_safe:
                    if is_free:
                        self.data.qpos[q_idx : q_idx + 2] = test_pos - xml_pos
                        self.data.qpos[q_idx + 2] = 0.5 
                        # クォータニオン回転のランダム化
                        angle = self.np_random.uniform(0, 2 * np.pi)
                        self.data.qpos[q_idx+3 : q_idx+7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
                    else:
                        self.data.qpos[q_idx : q_idx + 2] = test_pos - xml_pos
                        self.data.qpos[q_idx + 2] = 0.0 
                        self.data.qpos[q_idx + 3] = self.np_random.uniform(-np.pi, np.pi)
                    
                    placed_positions.append((test_pos, radius))
                    placed = True
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(1), {}