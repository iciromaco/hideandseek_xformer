import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard

xml = """
<mujoco>
    <option gravity="0 0 -9.81" timestep="0.005"/>
    <visual>
        <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    </visual>

    <asset>
        <texture name="grid_tex" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid_tex" texrepeat="1 1" reflectance="0.2"/>
        <mesh name="slope_mesh" vertex="-0.6666 -0.5 0 0.6666 -0.5 0 0.6666 -0.5 1 -0.6666 0.5 0 0.6666 0.5 0 0.6666 0.5 1"  face=" 0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0"/>
    </asset>

    <worldbody>
        <light pos="0 0 10" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
        <geom name="floor" type="plane" size="6 6 0.1" material="grid" friction="1.0 0.005 0.0001"/>

        <!-- 四方の壁  -->
        <geom name="wall_north" type="box" size="6.2 0.1 1.0" pos="0 6.1 1" rgba="0.7 0.7 0.7 1" friction="0.5"/>
        <geom name="wall_south" type="box" size="6.2 0.1 1.0" pos="0 -6.1 1" rgba="0.7 0.7 0.7 1" friction="0.5"/>
        <geom name="wall_east" type="box" size="0.1 6 1.0" pos="6.1 0 1" rgba="0.7 0.7 0.7 1" friction="0.5"/>
        <geom name="wall_west" type="box" size="0.1 6 1.0" pos="-6.1 0 1" rgba="0.7 0.7 0.7 1" friction="0.5"/>

        <!-- 内部壁  -->
        <geom name="wall_0" type="box" size="1.8 0.1 0.5" pos="3 0 0.5" rgba="0.0 0.7 0.7 1" friction="0.5"/>
        <geom name="wall_1" type="box" size="2 0.1 0.5" pos="-2 0 0.5" rgba="0.0 0.7 0.7 1" friction="0.5"/>
        <geom name="wall_2" type="box" size="4 0.1 0.5" pos="0 -3 0.5" rgba="0.0 0.7 0.7 1" friction="0.5"/>
        <geom name="wall_3" type="box" size="0.1 2.9 0.5" pos="-0.1 3 0.5" rgba="0.0 0.7 0.7 1" friction="0.5"/>

        <body name="ramp_0_" pos="2 -4 0">
          <inertial pos="0.3 0 0.25" mass="50" diaginertia="1 1 1"/>
          <joint type="free" damping="200.0"/>
          <geom type="mesh" mesh="slope_mesh" contype="0" conaffinity="0" rgba="0 1 0 1"/>
          <geom name="ramp_0_slope" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.3" friction="0.5 0 0"/>
          <geom name="ramp0_back_panel" type="box" size="0.02 0.5 0.5" pos="0.6666 0 0.5" rgba="0 1 0 0.3"/>
          <geom name="ramp_0_inner_box" type="box" size="0.3333 0.5 0.25" pos="0.3333 0 0.25" rgba="0 1 0 0.3" mass="30" solimp="0.95 0.99 0.001"/> 
        </body>

        <body name="seeker_0_anchor" pos="0 -4 0.5">
            <inertial pos="0 0 0" mass="10" diaginertia="5 5 10"/>
            <joint name="seeker_0_x" type="slide" axis="1 0 0" damping="1" pos="0 0 0"/>
            <joint name="seeker_0_y" type="slide" axis="0 1 0" damping="1" pos="0 0 0"/>
            <joint name="seeker_0_up" type="slide" axis="0 0 1" damping="1" pos="0 0 0"/>
            <joint name="seeker_0_rot" type="hinge" axis="0 0 1" damping="20" armature="1.5" pos="0 0 0"/>
            <body name="seeker_0" pos="0 0 0">               
                <site name="seeker_0_thruster" pos="0 0 0"/>
                <geom name="seeker_0_bottom"  type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="0.5 0 0.001"/>
                <geom name="seeker_0_body" type="capsule" size="0.3 0.2" rgba="0.9 0.1 0.1 1" mass="5" friction="0.5 0 0.001"/>
                <geom name="seeker_0_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <geom name="seeker_0_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.9 0.1 0.1 1" contype="0" conaffinity="0"/>
            </body>
        </body>

        <body name="hider_0_anchor" pos="5 5 0.5" euler="0 0 180.0">
            <inertial pos="0 0 0" mass="25" diaginertia="5 5 5"/>
            <joint name="hider_0_x" type="slide" axis="1 0 0" damping="1" pos="0 0 0"/>
            <joint name="hider_0_y" type="slide" axis="0 1 0" damping="1" pos="0 0 0"/>
            <joint name="hider_0_up" type="slide" axis="0 0 1" damping="1" pos="0 0 0"/>
            <joint name="hider_0_rot" type="hinge" axis="0 0 1" damping="20" armature="1.5" pos="0 0 0"/>
            <body name="hider_0" pos="0 0 0">               
                <site name="hider_0_thruster" pos="0 0 0"/>
                <geom name="hider_0_bottom"  type="sphere" size="0.4" pos="0 0 -0.1" mass="15" friction="0.5 0 0.001"/>
                <geom name="hider_0_body" type="capsule" size="0.3 0.2" rgba="0.1 0.1 0.9 1" mass="5" friction="0.5 0 0.001"/>
                <geom name="hider_0_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <geom name="hider_0_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="0.1 0.1 0.9 1" contype="0" conaffinity="0"/>
            </body>
        </body>

        <body name="box_0" pos="3.5 -4 0.5">
            <joint type="free" damping="200.0"/>
            <geom type="box" size="0.5 0.5 0.5" rgba="0.6 0.4 0.2 1" mass="20"/>
        </body>

    </worldbody>
    
    <actuator>
        <general name="m_fwd" site="seeker_0_thruster" gear="1 0 0 0 0 0" dyntype="none" gainprm="1000" biastype="none" ctrlrange="-1 1"/>
        <general name="m_rot" joint="seeker_0_rot" gear="0.5" dyntype="none" gainprm="500" biastype="none" ctrlrange="-1 1"/>
        <general name="m_up"  joint="seeker_0_up"  gear="1" dyntype="none" gainprm="500" biastype="none" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
key_states = {"up": False, "down": False, "left": False, "right": False}

# 自分のジオメトリIDを事前に取得
# my_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'seeker_0_body','seeker_0_back')]
my_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'seeker_0_bottom')]
ramp_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ramp_0_slope')

def on_press(key):
    try:
        k = key.char if hasattr(key, 'char') else key.name
        if k in ['up', 'down', 'left', 'right']: key_states[k] = True
    except: pass
def on_release(key):
    try:
        k = key.char if hasattr(key, 'char') else key.name
        if k in ['up', 'down', 'left', 'right']: key_states[k] = False
    except: pass

def main():
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    # 接地バッファの初期化
    contact_counter = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 初期視点を「引き」に設定（マウス操作も可能）
        viewer.cam.azimuth = 90    # 方位角（横方向の回転）
        viewer.cam.elevation = -60 # 仰角（上下の角度）
        viewer.cam.distance = 25.0 # 引きの距離（ここを大きくすると遠くなる）
        viewer.cam.lookat = np.array([0.0, 0.0, 0.0]) # 視点の中心位置

        while viewer.is_running():
            step_start = time.time()
            
            # 1. 状態取得 (z座標だけ取得)
            current_z = data.qpos[model.joint('seeker_0_up').qposadr[0]]

            # 2. キー入力
            fwd = 0.0; side = 0.0; rot = 0.0
            if key_states["up"]:    fwd += 1.0
            if key_states["down"]:  fwd -= 1.0
            if key_states["left"]:  rot += 0.5
            if key_states["right"]: rot -= 0.5

            # 3. 接地判定 (IDベース)
            is_touching_now = False
            is_on_ramp_geom = False
            for i in range(data.ncon):
                c = data.contact[i]
                if (c.geom1 in my_geom_ids or c.geom2 in my_geom_ids) and c.dist < 0.02:
                    is_touching_now = True
                    if c.geom1 == ramp_geom_id or c.geom2 == ramp_geom_id:
                        is_on_ramp_geom = True
                    break
            
            if is_touching_now: contact_counter = 10
            else: contact_counter = max(0, contact_counter - 1)
            has_contact = (contact_counter > 0)

            # --- 4. 推力の決定（ベクトル分解版） ---
            ctrl_fwd = 0.0
            ctrl_up  = 0.0
            
            # 5.0 以前に計算された fwd, rot を使用
            if has_contact and fwd != 0:
                # 機体(seeker_0)の回転行列を取得
                body_id = model.body('seeker_0').id
                # data.xmat は 9要素の配列なので 3x3 に変換
                body_xmat = data.xmat[body_id].reshape(3, 3)
                
                # 機体の正面ベクトル(Local X軸)がワールド座標でどこを向いているか
                forward_vec = body_xmat[:, 0] 
                
                # 前方向の傾き（Z成分が sin(θ) に相当）
                # 斜面を登っている時は pitch_sin が正になる
                pitch_sin = forward_vec[2]
                # 水平成分の比率 (cos(θ))
                pitch_cos = np.sqrt(max(0, 1 - pitch_sin**2))

                # ギア比の決定
                gear_ratio = 3.0 if (is_on_ramp_geom or current_z > 0.1) else 1.0
                total_force = fwd * gear_ratio

                # 推力の分解
                # 地面を押すのではなく、斜面に沿って進むためのベクトル配分
                ctrl_fwd = total_force * pitch_cos
                ctrl_up  = total_force * pitch_sin

                # 勾配が急な場合の重力補償（滑り落ち防止の微調整）
                if is_on_ramp_geom:
                    ctrl_up += 0.2  # 常に少しだけ浮かせる力を足して摩擦を軽減
            else:
                ctrl_fwd = 0.0
                ctrl_up = 0.0

            # 5. アクチュエータ適用
            # 順番: m_fwd, m_side, m_up, m_rot
            data.ctrl[model.actuator('m_fwd').id] = ctrl_fwd
            data.ctrl[model.actuator('m_rot').id] = rot
            data.ctrl[model.actuator('m_up').id]  = ctrl_up

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, 0.005 - (time.time() - step_start)))

if __name__ == "__main__":
    main()