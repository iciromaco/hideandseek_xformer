#!/usr/bin/env python3
"""
XMLビルド & ヒューマンモード専用ツール
XMLの物理設定を手動でリファインするための最小限のプログラム

使用方法:
    python xml_refinement_tool.py [オプション]

操作方法 (ビューア起動時):
    d: 前方へ移動
    a: 左回転
    w: 後退
    s: 右回転
    q: 上昇
    e: 下降
    スペース: ビューア停止/再開
"""

import sys
import math
import time
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import argparse


# =============================================================================
# 最小限の環境設定クラス
# =============================================================================
class MinimalEnvConfig:
    """XMLビルダーに必要な最小限の環境パラメータ"""
    
    # アリーナ設定
    ARENA_HALF = 6.0
    
    # エージェント設定
    AGENT_MASS = 1.0
    AGENT_Z_POS = 0.1
    AGENT_Z_MIN = 0.0
    AGENT_Z_MAX = 1.0
    AGENT_DAMPING_XY = 0.12
    AGENT_DAMPING_Z = 1.0
    AGENT_DAMPING_ROT = 0.35
    AGENT_ACTUATOR_FWD = "1800 0 0"
    AGENT_TURN_GAIN = "75 0 0"
    
    # オブジェクト設定
    BOX_MASS = 0.5
    BOX_JOINT_DAMPING = 0.001
    RAMP_MASS = 0.5
    RAMP_JOINT_DAMPING = 0.001
    RAMP_INNER_WEIGHT_MASS = 0.1
    
    def __init__(self, n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1):
        self.n_seekers = n_seekers
        self.n_hiders = n_hiders
        self.n_boxes = n_boxes
        self.n_ramps = n_ramps
        
        # エージェントキー
        if self.n_seekers == 1:
            self.seeker_keys = ["s"]
        else:
            self.seeker_keys = [f"s{i}" for i in range(1, self.n_seekers + 1)]
        self.hider_keys = [f"h{i}" for i in range(1, self.n_hiders + 1)]
        self.agent_keys = self.seeker_keys + self.hider_keys


# =============================================================================
# XMLビルダー（元の実装から最小化）
# =============================================================================
class XMLBuilder:
    """MuJoCoアリーナのXML生成クラス"""
    
    def __init__(self, env_config):
        self.env = env_config
    
    def _euler_z_to_quat(self, yaw):
        """Z軸周りのオイラー角をクォータニオンに変換"""
        w, z = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        return f"{w:.6f} 0 0 {z:.6f}"
    
    def build_xml(self):
        """動的XMLを生成"""
        e = self.env

        def _linspace_positions(center_x, center_y, spacing, count):
            if count <= 0:
                return []
            if count == 1:
                return [[center_x, center_y]]
            start = -0.5 * spacing * (count - 1)
            return [[center_x, center_y + start + spacing * i] for i in range(count)]

        def _safe_points(points, count):
            if count <= 0:
                return []
            if count <= len(points):
                return [list(p) for p in points[:count]]
            # 足りない場合は最後の点をわずかにずらして追加
            out = [list(p) for p in points]
            base_x, base_y = points[-1]
            for i in range(count - len(points)):
                out.append([base_x - 0.6 * (i + 1), base_y])
            return out

        # 初期重なりを避けるため、チーム/オブジェクトごとにスポーン領域を分離する
        seeker_pos = _linspace_positions(-4.2, 0.0, 1.1, e.n_seekers)
        hider_pos = _linspace_positions(4.2, 0.0, 1.1, e.n_hiders)
        box_pos = _safe_points(
            [
                (3.8, -4.2),
                (2.2, -4.2),
                (0.6, -4.2),
                (3.8, -2.7),
                (2.2, -2.7),
            ],
            e.n_boxes,
        )
        ramp_pos = _safe_points(
            [
                (-3.8, 4.2),
                (-2.2, 4.2),
                (-0.6, 4.2),
                (-3.8, 2.7),
                (-2.2, 2.7),
            ],
            e.n_ramps,
        )

        arena = self._xml_static_scene()
        ramps = "".join(
            self._xml_ramp(i, ramp_pos[i - 1], 0.0)
            for i in range(1, e.n_ramps + 1)
        )
        boxes = "".join(
            self._xml_box(i, box_pos[i - 1], 0.0)
            for i in range(1, e.n_boxes + 1)
        )
        
        # エージェント：シーカーは赤、ハイダーは青系
        seekers = "".join(
            self._xml_agent(ak, seeker_pos[i], 0.0, (0.9, 0.1, 0.1))
            for i, ak in enumerate(e.seeker_keys)
        )
        hiders = "".join(
            self._xml_agent(
                ak, hider_pos[i], math.pi,
                (0.1, 0.1, 0.9) if (i % 2 == 0) else (0.1, 0.6, 0.9),
            )
            for i, ak in enumerate(e.hider_keys)
        )
        acts = "".join(self._xml_actuators(ak) for ak in e.agent_keys)
        equality = self._xml_equality()
        
        return f"""<?xml version="1.0" ?>
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>
    <mesh name="ramp_mesh" 
          vertex="-0.6666 -0.5 0.0 0.6666 -0.5 0.0 0.6666 -0.5 1.0 -0.6666 0.5 0.0 0.6666 0.5 0.0 0.6666 0.5 1.0"
          face="0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0"/>
  </asset>
  <worldbody>
    <light pos="0 0 12" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="0 13 13" euler="2.35 0 -3.14" mode="fixed" />
    {arena} {ramps} {boxes} {seekers} {hiders}
  </worldbody>
  <actuator>{acts}</actuator>
  {equality}
</mujoco>"""
    
    def _xml_static_scene(self):
        """床と壁のXML"""
        e = self.env
        s = e.ARENA_HALF
        attr = 'friction="0.05 0.05 0.05" solref="0.01 1" solimp="0.95 0.99 0.001"'
        return f"""
    <geom name="floor" type="plane" size="{s} {s} 0.1" material="grid" friction="1.1 0.15 0.003"/>
    <geom name="wall_n" type="box" size="{s+0.15} 0.1 2.0" pos="0 6.1 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_s" type="box" size="{s+0.15} 0.1 2.0" pos="0 -6.1 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_e" type="box" size="0.1 {s} 2.0" pos="6.1 0 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_w" type="box" size="0.1 {s} 2.0" pos="-6.1 0 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="maze_w0" type="box" size="1.5 0.2 0.5" pos="3.0 1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w1" type="box" size="1.5 0.2 0.5" pos="-3.0 -1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w2" type="box" size="0.2 1.5 0.5" pos="0.0 -3.0 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w3" type="box" size="0.2 1.5 0.5" pos="0.0 3.0 0.5" rgba="0 0.7 0.7 1" {attr}/>
"""
    
    def _xml_ramp(self, i, xy, rot):
        """ランプのXML"""
        e = self.env
        q = self._euler_z_to_quat(rot)
        return f"""
    <body name="ramp{i}_body" pos="{xy[0]} {xy[1]} 0" quat="{q}">
        <inertial pos="0.3 0 0.25" mass="{e.RAMP_MASS}" diaginertia="10 10 20"/>
        <joint type="free" name="ramp{i}_joint" damping="{e.RAMP_JOINT_DAMPING}"/>
        <geom name="ramp{i}_geom" type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0" rgba="0 1 0 1"/>
        <geom name="ramp{i}_slope_surface" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.3" friction="1.35 0.22 0.00001"/>
        <geom name="ramp{i}_back_panel" type="box" size="0.02 0.5 0.5" pos="0.6666 0 0.5" rgba="0 1 0 0.3" friction="1.35 0.22 0.01"/>
        <geom name="ramp{i}_inner_weight" type="box" size="0.3333 0.5 0.25" pos="0.3333 0 0.25" rgba="0 1 0 0.3" mass="{e.RAMP_INNER_WEIGHT_MASS}" solimp="0.95 0.99 0.001" friction="1.35 0.22 0.01"/>
    </body>"""
    
    def _xml_box(self, i, xy, rot):
        """箱のXML"""
        e = self.env
        q = self._euler_z_to_quat(rot)
        return f"""
    <body name="box{i}_body" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
        <joint name="box{i}_joint" type="free" damping="{e.BOX_JOINT_DAMPING}"/>
        <geom name="box{i}_geom" type="box" size="0.6 0.6 0.5" mass="{e.BOX_MASS}" rgba="0.75 0.55 0.3 1" friction="1.2 0.08 0.003"/>
    </body>"""
    
    def _xml_agent(self, pre, xy, rot, color):
        """エージェント（シーカー/ハイダー）のXML"""
        e = self.env
        q = self._euler_z_to_quat(rot)
        r, g, b = color
        limit_z = f"{e.AGENT_Z_MIN-e.AGENT_Z_POS} {e.AGENT_Z_MAX-e.AGENT_Z_POS}"
        return f"""
    <body name="{pre}_anchor" pos="{xy[0]} {xy[1]} {e.AGENT_Z_POS}" quat="{q}">
        <joint name="{pre}_x" type="slide" axis="1 0 0" damping="{e.AGENT_DAMPING_XY}"/>
        <joint name="{pre}_y" type="slide" axis="0 1 0" damping="{e.AGENT_DAMPING_XY}"/>
        <joint name="{pre}_z" type="slide" axis="0 0 1" damping="{e.AGENT_DAMPING_Z}" limited="true" range="{limit_z}"/>
        <joint name="{pre}_rot" type="hinge" axis="0 0 1" damping="{e.AGENT_DAMPING_ROT}"/>
        <body name="{pre}_body">
            <site name="{pre}_thrust" pos="0 0 0"/>
            <geom name="{pre}_btm" type="sphere" pos="0 0 -0.1" size="0.4" mass="{e.AGENT_MASS}" friction="0.05 0.01 0"/>
            <geom name="{pre}_capsule" type="capsule" size="0.3 0.2" pos="0 0 0" rgba="{r} {g} {b} 1" mass="70" friction="0.05 0.01 0"/>
            <geom name="{pre}_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
            <geom name="{pre}_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="{r} {g} {b} 1" contype="0" conaffinity="0"/>
        </body>
    </body>"""
    
    def _xml_actuators(self, pre):
        """アクチュエータのXML"""
        e = self.env
        return f"""
    <general name="{pre}_fwd" site="{pre}_thrust" gear="1 0 0 0 0 0" gainprm="{e.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{pre}_turn" joint="{pre}_rot" gear="0.5" gainprm="{e.AGENT_TURN_GAIN}" ctrlrange="-1 1"/>
"""
    
    def _xml_equality(self):
        """等式制約（グラスプ）のXML"""
        e = self.env
        grasp_solref = "0.08 1"
        grasp_solimp = "0.9 0.95 0.001"
        parts = ["  <equality>"]
        
        for ak in e.agent_keys:
            for i in range(1, e.n_boxes + 1):
                parts.append(
                    f'    <weld name="eq_grasp_{ak}_box{i}" body1="{ak}_body" body2="box{i}_body" '
                    f'active="false" relpose="0 0 0 1 0 0 0" solref="{grasp_solref}" solimp="{grasp_solimp}"/>'
                )
            for i in range(1, e.n_ramps + 1):
                parts.append(
                    f'    <weld name="eq_grasp_{ak}_ramp{i}" body1="{ak}_body" body2="ramp{i}_body" '
                    f'active="false" relpose="0 0 0 1 0 0 0" solref="{grasp_solref}" solimp="{grasp_solimp}"/>'
                )
        parts.append("  </equality>")
        return "\n".join(parts)


# =============================================================================
# ビューアコントロール（MuJoCoネイティブ key_callback 使用）
# =============================================================================
# GLFW キーコード（矢印キー）
_GLFW_KEY_UP    = 265
_GLFW_KEY_DOWN  = 264
_GLFW_KEY_LEFT  = 263
_GLFW_KEY_RIGHT = 262


class HumanController:
    """MuJoCoビューアのkey_callbackでエージェントを制御（矢印キー）"""

    def __init__(self, model, learnable_key="s"):
        self.model = model
        self.learnable_key = learnable_key
        self.fwd_actuator_id  = model.actuator(f"{learnable_key}_fwd").id
        self.turn_actuator_id = model.actuator(f"{learnable_key}_turn").id
        self.ctrl = np.zeros(model.nu)
        # キー押下イベントは key-down のみ拾えるため、短時間パルスとして扱う
        self._pulse_sec = 0.65
        self._t_up = -1e9
        self._t_down = -1e9
        self._t_left = -1e9
        self._t_right = -1e9

    def key_callback(self, keycode):
        """launch_passive の key_callback として渡す（パルス方式）"""
        now = time.perf_counter()
        if keycode == _GLFW_KEY_UP:
            self._t_up = now
        elif keycode == _GLFW_KEY_DOWN:
            self._t_down = now
        elif keycode == _GLFW_KEY_LEFT:
            self._t_left = now
        elif keycode == _GLFW_KEY_RIGHT:
            self._t_right = now

    def update_control(self):
        now = time.perf_counter()

        up_active = (now - self._t_up) <= self._pulse_sec
        down_active = (now - self._t_down) <= self._pulse_sec
        left_active = (now - self._t_left) <= self._pulse_sec
        right_active = (now - self._t_right) <= self._pulse_sec

        fwd = 0.0
        if up_active and not down_active:
            fwd = 1.0
        elif down_active and not up_active:
            fwd = -1.0

        turn = 0.0
        if left_active and not right_active:
            turn = 1.0
        elif right_active and not left_active:
            turn = -1.0

        self.ctrl[self.fwd_actuator_id] = fwd
        self.ctrl[self.turn_actuator_id] = turn
        return self.ctrl


# =============================================================================
# メイン実行クラス
# =============================================================================
class XMLRefinementApp:
    """XML設定の手動リファインツール"""
    
    def __init__(self, n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1,
                 learnable_agent="s", interactive=True):
        self.n_seekers = n_seekers
        self.n_hiders = n_hiders
        self.n_boxes = n_boxes
        self.n_ramps = n_ramps
        self.learnable_agent = learnable_agent
        self.interactive = interactive
        
        # 環境設定とXMLビルダーを初期化
        self.env_config = MinimalEnvConfig(
            n_seekers=n_seekers,
            n_hiders=n_hiders,
            n_boxes=n_boxes,
            n_ramps=n_ramps,
        )
        self.builder = XMLBuilder(self.env_config)
        
        # MuJoCoモデルを生成
        xml_string = self.builder.build_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.controller = None
    
    def run(self, duration=30.0):
        """シミュレーション実行"""
        # main28 の action_repeat=10 相当: 1描画フレームあたり10物理ステップ
        action_repeat = 10
        frame_dt = float(self.model.opt.timestep) * action_repeat

        controller = None
        key_cb = None

        if self.interactive:
            controller = HumanController(self.model, self.learnable_agent)
            key_cb = controller.key_callback

        with mujoco.viewer.launch_passive(
            self.model, self.data,
            key_callback=key_cb,
        ) as viewer:
            # カメラ設定
            with viewer.lock():
                viewer.cam.lookat[:] = [0, 0, 0.8]
                viewer.cam.distance = 18.0
                viewer.cam.elevation = -35.0
                viewer.cam.azimuth = 90.0

            try:
                while viewer.is_running():
                    loop_start = time.perf_counter()

                    if controller:
                        self.data.ctrl[:] = controller.update_control()

                    for _ in range(action_repeat):
                        mujoco.mj_step(self.model, self.data)

                    viewer.sync()

                    # 目視しやすいよう実時間を frame_dt に合わせる
                    elapsed = time.perf_counter() - loop_start
                    remain = frame_dt - elapsed
                    if remain > 0.0:
                        time.sleep(remain)

            except KeyboardInterrupt:
                print("\n中断しました。")
    
    def save_xml(self, filepath):
        """生成したXMLを保存"""
        xml_string = self.builder.build_xml()
        with open(filepath, 'w') as f:
            f.write(xml_string)
        print(f"XMLを保存しました: {filepath}")
    
    def print_xml(self):
        """XMLをコンソールに出力"""
        xml_string = self.builder.build_xml()
        print(xml_string)
    
    def get_xml_string(self):
        """XMLを文字列で取得"""
        return self.builder.build_xml()


# =============================================================================
# メイン処理
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="XMLビルド & ヒューマンモード専用ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
操作方法 (ビューア起動時):
  ↑: 前進
  ↓: 後退
  ←: 左回転
  →: 右回転
    ※短いパルス入力（長押しで連続入力）
  CTRL+C: 終了

オプション:
  --duration: シミュレーション実行時間 (秒)
  --no-interactive: キーボード入力を無効化
  --save-xml FILE: XMLをファイルに保存
  --print-xml: XMLをコンソール出力
"""
    )
    parser.add_argument("--n-seekers", type=int, default=1, help="シーカー数")
    parser.add_argument("--n-hiders", type=int, default=2, help="ハイダー数")
    parser.add_argument("--n-boxes", type=int, default=2, help="箱の数")
    parser.add_argument("--n-ramps", type=int, default=1, help="ランプの数")
    parser.add_argument("--duration", type=float, default=60.0, help="実行時間（秒）")
    parser.add_argument("--learnable-agent", type=str, default="s", help="制御対象エージェント")
    parser.add_argument("--no-interactive", action="store_true", help="キーボード入力なし")
    parser.add_argument("--save-xml", type=str, help="XMLの保存先")
    parser.add_argument("--print-xml", action="store_true", help="XMLをコンソール出力")
    
    args = parser.parse_args()
    
    app = XMLRefinementApp(
        n_seekers=args.n_seekers,
        n_hiders=args.n_hiders,
        n_boxes=args.n_boxes,
        n_ramps=args.n_ramps,
        learnable_agent=args.learnable_agent,
        interactive=not args.no_interactive,
    )
    
    if args.print_xml:
        app.print_xml()
    
    if args.save_xml:
        app.save_xml(args.save_xml)
    
    if not args.print_xml and not args.save_xml:
        # ビューアを起動して実行
        print("ビューアを起動中...")
        print(f"操作方法: ↑=前進, ↓=後退, ←=左回転, →=右回転 (パルス), CTRL+C=終了")
        app.run(duration=args.duration)


if __name__ == "__main__":
    try:
        from pynput import keyboard as _pynput_keyboard
    except ImportError:
        _pynput_keyboard = None
        print("警告: pynputがインストールされていません。キーボード入力は無効です。")
        print("  インストール: pip install pynput")
    
    main()
