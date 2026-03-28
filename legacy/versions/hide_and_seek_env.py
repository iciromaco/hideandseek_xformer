"""
hide_and_seek_env.py
====================
MuJoCo Hide-and-Seek 環境

配置指定の方針:
  各グループ (seekers / hiders / boxes / ramps) ごとに
  PlacementSpec のリストを渡せる。
    - None        → rejection-sampling でランダム安全配置
    - (x, y, rot) → 指定座標・指定向き (rad) に固定配置

  リストの長さは n_* と一致させる。省略時は全エントリを None として扱う。

使い方:
    python hide_and_seek_env.py
    python hide_and_seek_env.py --seekers 2 --hiders 3 --boxes 3 --ramps 2 --seed 0
"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

# ---------------------------------------------------------------------------
# 型エイリアス
#   None          → ランダム安全配置
#   (x, y, rot)   → 指定位置・指定向き [rad]
# ---------------------------------------------------------------------------
PlacementSpec = Optional[Tuple[float, float, float]]


# ---------------------------------------------------------------------------
# 内部管理用: 配置済み物体の記録
# ---------------------------------------------------------------------------
@dataclass
class _PlacedItem:
    name: str
    pos: np.ndarray  # (x, y)
    radius: float


# ---------------------------------------------------------------------------
# メイン環境クラス
# ---------------------------------------------------------------------------
class HideAndSeekEnv:
    """
    動的に XML を生成し MuJoCo モデルをビルドする環境クラス。

    Parameters
    ----------
    n_seekers, n_hiders, n_boxes, n_ramps : int
        各グループの物体数。
    seeker_specs, hider_specs, box_specs, ramp_specs : list of PlacementSpec
        各エントリが None → ランダム、(x, y, rot) → 固定。
        リストが None または短い場合は残りを None で補完する。
    seed : int, optional
        ランダム配置の乱数シード。
    """

    # ---- アリーナ設定 --------------------------------------------------------
    ARENA_HALF = 6.0  # 外壁内側 (m)
    SAFE_HALF = 5.0  # ランダム配置の範囲
    PLACE_MARGIN = 0.25  # 物体間の最小隙間 (m)
    MAX_TRIES = 3000  # rejection-sampling の最大試行回数

    # ---- 配置衝突半径 --------------------------------------------------------
    R_AGENT = 0.55
    R_BOX = 0.95
    R_RAMP = 1.30

    # ---- 迷路壁 (cx, cy, half_x, half_y) ------------------------------------
    MAZE_WALLS: List[Tuple[float, float, float, float]] = [
        (3.0, 1.5, 1.5, 0.2),
        (-3.0, -1.5, 1.5, 0.2),
        (0.0, -3.0, 0.2, 1.5),
        (0.0, 3.0, 0.2, 1.5),
    ]

    # ---- エージェント色 ------------------------------------------------------
    SEEKER_COLORS = [
        (0.9, 0.1, 0.1),
        (0.9, 0.5, 0.1),
        (0.8, 0.8, 0.1),
    ]
    HIDER_COLORS = [
        (0.1, 0.1, 0.9),
        (0.1, 0.6, 0.9),
        (0.1, 0.9, 0.5),
    ]

    # ---- 物理パラメータ -------------------------------------------------------
    # エージェント質量 16 kg、アリーナ 12 m 基準
    # 終端速度   = FWD / DAMPING_XY  = 90 / 30 = 3.0 m/s
    # 起動加速度 = FWD / mass        = 90 / 16 = 5.6 m/s²
    # 終端角速度 = (TURN × gear) / DAMPING_ROT = (120×0.5)/30 = 2.0 rad/s
    # Box 摩擦力  0.20×40×9.81 = 78.5 N < 90 N → 強く押せばゆっくり動く
    # Ramp 摩擦力 0.20×55×9.81 = 108 N > 90 N → 1体では動かない

    AGENT_MASS_BODY = 4.0
    AGENT_MASS_BOTTOM = 12.0
    AGENT_FRICTION = (0.35, 0.01, 0.001)
    AGENT_CAPSULE_FRICTION = (0.35, 0.01, 0.001)
    AGENT_DAMPING_XY = 26.0
    AGENT_DAMPING_Z = 15.0
    AGENT_DAMPING_ROT = 30.0
    AGENT_ACTUATOR_FWD = 1000
    AGENT_ACTUATOR_TURN = 120

    FLOOR_FRICTION = (0.02, 0.05, 0.001)

    BOX_MASS = 40.0
    BOX_DAMPING = 20.0
    BOX_FRICTION = (0.20, 0.005, 0.001)

    RAMP_MASS_BASE = 60.0
    RAMP_DAMPING = 65.0
    RAMP_SLOPE_FRICTION = (2.00, 0.02, 0.001)
    RAMP_BASE_FRICTION = (0.95, 0.02, 0.001)

    # =========================================================================
    def __init__(
        self,
        n_seekers: int = 1,
        n_hiders: int = 2,
        n_boxes: int = 2,
        n_ramps: int = 1,
        seeker_specs: Optional[List[PlacementSpec]] = None,
        hider_specs: Optional[List[PlacementSpec]] = None,
        box_specs: Optional[List[PlacementSpec]] = None,
        ramp_specs: Optional[List[PlacementSpec]] = None,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            np.random.seed(seed)

        self.n_seekers = n_seekers
        self.n_hiders = n_hiders
        self.n_boxes = n_boxes
        self.n_ramps = n_ramps

        # specs を n_* の長さに揃える（短い・None は None 補完）
        self._seeker_specs = self._normalize_specs(seeker_specs, n_seekers)
        self._hider_specs = self._normalize_specs(hider_specs, n_hiders)
        self._box_specs = self._normalize_specs(box_specs, n_boxes)
        self._ramp_specs = self._normalize_specs(ramp_specs, n_ramps)

        self._build()

    # =========================================================================
    # 公開ユーティリティ: 指定配置 spec を作るヘルパー
    # =========================================================================
    @staticmethod
    def make_spec(x: float, y: float, rot_deg: float = 0.0) -> PlacementSpec:
        """度単位の向きで PlacementSpec を作る便利メソッド。"""
        return (x, y, np.deg2rad(rot_deg))

    # =========================================================================
    # 配置生成
    # =========================================================================
    @staticmethod
    def _normalize_specs(specs: Optional[List[PlacementSpec]], n: int) -> List[PlacementSpec]:
        """specs を長さ n のリストに正規化する。不足分は None で補完。"""
        if specs is None:
            return [None] * n
        result = list(specs)
        while len(result) < n:
            result.append(None)
        return result[:n]

    def _generate_placements(self) -> dict:
        """
        全グループの配置リスト [(xy, rot), ...] を生成する。
        spec が None のエントリは rejection-sampling で安全配置する。
        spec が (x, y, rot) のエントリは指定値を使う（干渉チェックはしない）。
        両者とも配置済みリストに登録し、後続の random 配置から除外する。
        """
        placed: List[_PlacedItem] = []

        def resolve_group(specs: List[PlacementSpec], radius: float, tag: str) -> List[Tuple[np.ndarray, float]]:
            results = []
            for i, spec in enumerate(specs):
                if spec is not None:
                    # ---- 指定配置 ----------------------------------------
                    x, y, rot = spec
                    xy = np.array([x, y], dtype=float)
                else:
                    # ---- ランダム安全配置 --------------------------------
                    xy = self._place_random(placed, radius)
                    rot = np.random.uniform(0.0, 2.0 * np.pi)

                placed.append(_PlacedItem(f"{tag}_{i}", xy, radius))
                results.append((xy, rot))
            return results

        return dict(
            seekers=resolve_group(self._seeker_specs, self.R_AGENT, "seeker"),
            hiders=resolve_group(self._hider_specs, self.R_AGENT, "hider"),
            boxes=resolve_group(self._box_specs, self.R_BOX, "box"),
            ramps=resolve_group(self._ramp_specs, self.R_RAMP, "ramp"),
        )

    def _place_random(self, placed: List[_PlacedItem], radius: float) -> np.ndarray:
        """Rejection sampling で干渉しない (x,y) を返す。"""
        for _ in range(self.MAX_TRIES):
            xy = np.random.uniform(-self.SAFE_HALF, self.SAFE_HALF, 2)
            if self._overlaps_maze_wall(xy, radius):
                continue
            if any(np.linalg.norm(xy - p.pos) < radius + p.radius + self.PLACE_MARGIN for p in placed):
                continue
            return xy
        raise RuntimeError(f"ランダム配置に失敗 ({self.MAX_TRIES} 試行)。" "物体数を減らすか SAFE_HALF / PLACE_MARGIN を調整してください。")

    def _overlaps_maze_wall(self, xy: np.ndarray, radius: float) -> bool:
        for cx, cy, hx, hy in self.MAZE_WALLS:
            dx = max(0.0, abs(xy[0] - cx) - hx)
            dy = max(0.0, abs(xy[1] - cy) - hy)
            if np.hypot(dx, dy) < radius + self.PLACE_MARGIN:
                return True
        return False

    # =========================================================================
    # ビルド / リセット
    # =========================================================================
    def _build(self) -> None:
        """配置を決定して XML をコンパイルし model / data を生成する。"""
        placements = self._generate_placements()
        xml_str = self._build_xml(**placements)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, seed: Optional[int] = None) -> None:
        """
        シーンを再生成して初期化する。
        specs は __init__ で与えたものを再利用する。
        """
        if seed is not None:
            np.random.seed(seed)
        self._build()

    # =========================================================================
    # XML 生成
    # =========================================================================
    def _build_xml(
        self,
        seekers: List[Tuple[np.ndarray, float]],
        hiders: List[Tuple[np.ndarray, float]],
        boxes: List[Tuple[np.ndarray, float]],
        ramps: List[Tuple[np.ndarray, float]],
    ) -> str:
        wb, eq, ac = [], [], []

        wb.append(self._xml_static_scene())

        for i, (xy, rot) in enumerate(ramps):
            wb.append(self._xml_ramp(i, xy, rot))
            eq.append(self._xml_lock_eq("ramp", i))

        for i, (xy, rot) in enumerate(boxes):
            wb.append(self._xml_box(i, xy, rot))
            eq.append(self._xml_lock_eq("box", i))

        for i, (xy, rot) in enumerate(seekers):
            c = self.SEEKER_COLORS[i % len(self.SEEKER_COLORS)]
            wb.append(self._xml_agent("seeker", i, xy, rot, c))
            ac.append(self._xml_actuators("seeker", i))

        for i, (xy, rot) in enumerate(hiders):
            c = self.HIDER_COLORS[i % len(self.HIDER_COLORS)]
            wb.append(self._xml_agent("hider", i, xy, rot, c))
            ac.append(self._xml_actuators("hider", i))

        for tag, n_ag in [("seeker", self.n_seekers), ("hider", self.n_hiders)]:
            for ai in range(n_ag):
                for bi in range(self.n_boxes):
                    eq.append(self._xml_grasp_eq(tag, ai, "box", bi))
                for ri in range(self.n_ramps):
                    eq.append(self._xml_grasp_eq(tag, ai, "ramp", ri))

        return f"""
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
  </visual>
  <asset>
    <texture name="grid_tex" type="2d" builtin="checker"
             rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid_mat" texture="grid_tex" texrepeat="1 1" reflectance="0.2"/>
    <mesh name="ramp_mesh"
          vertex="-0.6666 -0.5 0.0  0.6666 -0.5 0.0  0.6666 -0.5 1.0
                  -0.6666  0.5 0.0  0.6666  0.5 0.0  0.6666  0.5 1.0"
          face="0 1 2  3 5 4  0 3 4  0 4 1  1 4 5  1 5 2  2 5 3  2 3 0"/>
  </asset>
  <worldbody>
    <light pos="0 0 12" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {''.join(wb)}
  </worldbody>
  <equality>
    {''.join(eq)}
  </equality>
  <actuator>
    {''.join(ac)}
  </actuator>
</mujoco>
"""

    # ---- 静的シーン ----------------------------------------------------------
    def _xml_static_scene(self) -> str:
        s = self.ARENA_HALF
        flr = " ".join(str(v) for v in self.FLOOR_FRICTION)
        outer_wall_half_z = 2.0
        outer_wall_center_z = outer_wall_half_z
        inner_wall_half_z = 0.5
        inner_wall_center_z = inner_wall_half_z
        return f"""
    <geom name="floor" type="plane" size="{s} {s} 0.1"
          material="grid_mat" friction="{flr}" solref="0.02 1"/>
    <geom name="wall_n" type="box" size="{s+0.15} 0.1 {outer_wall_half_z}"
          pos="0  {s+0.1} {outer_wall_center_z}" rgba="0.65 0.65 0.65 0.35"/>
    <geom name="wall_s" type="box" size="{s+0.15} 0.1 {outer_wall_half_z}"
          pos="0 -{s+0.1} {outer_wall_center_z}" rgba="0.65 0.65 0.65 0.35"/>
    <geom name="wall_e" type="box" size="0.1 {s} {outer_wall_half_z}"
          pos=" {s+0.1} 0 {outer_wall_center_z}" rgba="0.65 0.65 0.65 0.35"/>
    <geom name="wall_w" type="box" size="0.1 {s} {outer_wall_half_z}"
          pos="-{s+0.1} 0 {outer_wall_center_z}" rgba="0.65 0.65 0.65 0.35"/>
    <geom name="maze_w0" type="box" size="1.5 0.2 {inner_wall_half_z}"
          pos=" 3.0  1.5 {inner_wall_center_z}" rgba="0.0 0.7 0.7 1"/>
    <geom name="maze_w1" type="box" size="1.5 0.2 {inner_wall_half_z}"
          pos="-3.0 -1.5 {inner_wall_center_z}" rgba="0.0 0.7 0.7 1"/>
    <geom name="maze_w2" type="box" size="0.2 1.5 {inner_wall_half_z}"
          pos=" 0.0 -3.0 {inner_wall_center_z}" rgba="0.0 0.7 0.7 1"/>
    <geom name="maze_w3" type="box" size="0.2 1.5 {inner_wall_half_z}"
          pos=" 0.0  3.0 {inner_wall_center_z}" rgba="0.0 0.7 0.7 1"/>
"""

    # ---- Ramp ----------------------------------------------------------------
    def _xml_ramp(self, i: int, xy: np.ndarray, rot: float) -> str:
        quat = _euler_z_to_quat(rot)
        ramp_slope_friction = " ".join(str(v) for v in self.RAMP_SLOPE_FRICTION)
        ramp_base_friction = " ".join(str(v) for v in self.RAMP_BASE_FRICTION)
        return f"""
    <body name="ramp_{i}_body" pos="{xy[0]:.3f} {xy[1]:.3f} 0.0" quat="{quat}">
      <inertial pos="0.30 0 0.33" mass="{self.RAMP_MASS_BASE}" diaginertia="6 6 12"/>
      <joint type="free" name="ramp_{i}_joint" damping="{self.RAMP_DAMPING}"/>
      <geom type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0"
            rgba="0.2 0.85 0.2 0.9"/>
      <geom name="ramp_{i}_slope" type="box" size="0.833 0.5 0.02"
            pos="0 0 0.516" euler="0 -36.87 0"
        rgba="0.2 0.85 0.2 0.4" friction="{ramp_slope_friction}" mass="0"/>
      <geom name="ramp_{i}_back" type="box" size="0.02 0.5 0.5"
            pos="0.6666 0 0.5" rgba="0.2 0.85 0.2 0.4" mass="0"/>
      <geom name="ramp_{i}_base" type="box" size="0.333 0.5 0.25"
            pos="0.333 0 0.25" rgba="0.2 0.85 0.2 0.4"
        friction="{ramp_base_friction}" mass="{self.RAMP_MASS_BASE}"/>
    </body>
"""

    # ---- Box -----------------------------------------------------------------
    def _xml_box(self, i: int, xy: np.ndarray, rot: float) -> str:
        quat = _euler_z_to_quat(rot)
        return f"""
    <body name="box_{i}_body" pos="{xy[0]:.3f} {xy[1]:.3f} 0.5" quat="{quat}">
      <joint name="box_{i}_joint" type="free" damping="{self.BOX_DAMPING}"/>
      <geom name="box_{i}_geom" type="box" size="0.6 0.6 0.5"
            rgba="0.75 0.55 0.30 1" mass="{self.BOX_MASS}"
            friction="{' '.join(str(f) for f in self.BOX_FRICTION)}"
            solref="0.02 1" condim="4"/>
    </body>
"""

    # ---- Agent ---------------------------------------------------------------
    def _xml_agent(
        self,
        tag: str,
        i: int,
        xy: np.ndarray,
        rot: float,
        color: Tuple[float, float, float],
    ) -> str:
        r, g, b = color
        quat = _euler_z_to_quat(rot)
        af = " ".join(str(f) for f in self.AGENT_FRICTION)
        acf = " ".join(str(f) for f in self.AGENT_CAPSULE_FRICTION)
        return f"""
    <body name="{tag}_{i}_anchor" pos="{xy[0]:.3f} {xy[1]:.3f} 0.45" quat="{quat}">
      <joint name="{tag}_{i}_x"   type="slide" axis="1 0 0"
             damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{tag}_{i}_y"   type="slide" axis="0 1 0"
             damping="{self.AGENT_DAMPING_XY}"/>
      <joint name="{tag}_{i}_z"   type="slide" axis="0 0 1"
             limited="true" range="-0.1 1.0" damping="{self.AGENT_DAMPING_Z}"/>
      <joint name="{tag}_{i}_rot" type="hinge" axis="0 0 1"
             damping="{self.AGENT_DAMPING_ROT}" armature="2.0"/>
      <body name="{tag}_{i}_body">
        <site name="{tag}_{i}_thrust" pos="0 0 0"/>
        <geom name="{tag}_{i}_btm" type="sphere" size="0.35" pos="0 0 -0.1"
              mass="{self.AGENT_MASS_BOTTOM}" friction="{af}"/>
        <geom name="{tag}_{i}_capsule" type="capsule" size="0.28 0.18"
              rgba="{r} {g} {b} 1" mass="{self.AGENT_MASS_BODY}"
              friction="{acf}" contype="0" conaffinity="0"/>
        <geom name="{tag}_{i}_nose" type="capsule"
              fromto="0 0 0.18 0.28 0 0.18" size="0.09"
              rgba="1 1 1 1" contype="0" conaffinity="0"/>
        <geom name="{tag}_{i}_tail" type="capsule"
              fromto="0 0 0 -0.40 0 -0.25" size="0.045"
              rgba="{r} {g} {b} 1" contype="0" conaffinity="0"/>
      </body>
    </body>
"""

    # ---- Actuators -----------------------------------------------------------
    def _xml_actuators(self, tag: str, i: int) -> str:
        return f"""
    <general name="{tag}_{i}_fwd"
             site="{tag}_{i}_thrust" gear="1 0 0 0 0 0"
             gainprm="{self.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{tag}_{i}_turn"
             joint="{tag}_{i}_rot" gear="0.5"
             gainprm="{self.AGENT_ACTUATOR_TURN}" ctrlrange="-1 1"/>
"""

    # ---- Equality ------------------------------------------------------------
    def _xml_grasp_eq(self, agent_tag: str, ai: int, obj_tag: str, oi: int) -> str:
        name = f"grasp_{agent_tag}_{ai}_{obj_tag}_{oi}"
        body1 = f"{agent_tag}_{ai}_body"
        body2 = f"{obj_tag}_{oi}_body"
        return f'<weld name="{name}" body1="{body1}" body2="{body2}" ' f'active="false" solref="0.06 1" solimp="0.90 0.95 0.001"/>\n    '

    def _xml_lock_eq(self, obj_tag: str, oi: int) -> str:
        name = f"lock_{obj_tag}_{oi}"
        body2 = f"{obj_tag}_{oi}_body"
        return f'<weld name="{name}" body1="world" body2="{body2}" ' f'active="false" solref="0.02 1" solimp="0.95 0.99 0.001"/>\n    '

    # =========================================================================
    # シミュレーション制御
    # =========================================================================
    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    def run_viewer(
        self,
        lookat: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 18.0,
        elevation: float = -55.0,
        azimuth: float = 135.0,
    ) -> None:
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.lookat[:] = lookat
            viewer.cam.distance = distance
            viewer.cam.elevation = elevation
            viewer.cam.azimuth = azimuth

            viewer.sync()
            time.sleep(0.05)

            sim_time = 0.0
            wall_start = time.perf_counter()
            while viewer.is_running():
                wall_now = time.perf_counter()
                target = wall_now - wall_start
                while sim_time < target:
                    self.step()
                    sim_time += self.model.opt.timestep
                viewer.sync()


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------
def _euler_z_to_quat(yaw: float) -> str:
    half = yaw / 2.0
    w, z = np.cos(half), np.sin(half)
    return f"{w:.6f} 0.000000 0.000000 {z:.6f}"


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="Hide-and-Seek MuJoCo Viewer")
    p.add_argument("--seekers", type=int, default=1)
    p.add_argument("--hiders", type=int, default=2)
    p.add_argument("--boxes", type=int, default=2)
    p.add_argument("--ramps", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"[HideAndSeek] Seekers={args.seekers}, Hiders={args.hiders}, " f"Boxes={args.boxes}, Ramps={args.ramps}, Seed={args.seed}")
    env = HideAndSeekEnv(
        n_seekers=args.seekers,
        n_hiders=args.hiders,
        n_boxes=args.boxes,
        n_ramps=args.ramps,
        seed=args.seed,
    )
    print("[HideAndSeek] Viewer を起動します。ウィンドウを閉じると終了します。")
    env.run_viewer()
