"""
# test_lidar_validation.py
Lidar Raycast Methods Validation
3つのレイキャスト方法の出力を比較するプログラム
"""

import os
import sys
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import mujoco

# main18_optimization をインポートしてbase_configとして使用
current_script_abs_path_val = os.path.abspath(__file__)
current_script_parent_dir_val = os.path.dirname(current_script_abs_path_val)
search_dir_path_pointer_idx = current_script_parent_dir_val

for _ in range(5):
    potential_base_config_file_path = os.path.join(search_dir_path_pointer_idx, "main18_optimization.py")
    if os.path.exists(potential_base_config_file_path):
        if search_dir_path_pointer_idx not in sys.path:
            sys.path.insert(0, search_dir_path_pointer_idx)
        break
    search_dir_path_pointer_idx = os.path.dirname(search_dir_path_pointer_idx)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

try:
    import main18_optimization as base_config
except ImportError as e:
    print(f"Error importing main18_optimization: {e}")
    print(f"sys.path: {sys.path}")
    raise

from main23_sightmap_optimized import (
    VisibilityEngine, cast_ray_direct_numba, cast_ray_numba,
    LIDAR_MAX_DIST
)

# ============================================
# 初期化
# ============================================
xml_string = base_config.XML_CONTENT
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# 環境をリセット
mujoco.mj_resetData(model, data)

# ============================================
# 検証用の設定
# ============================================

# 視点を迷路内部に設定（障害物の外側）
viewpoint = np.array([-2.0, -.0], dtype=np.float32)  # 部屋の内部に移動
yaw = np.pi / 3  # 60度
SEEKER_BODY_ID = -1  # seekerも検出対象に含める
EXCLUDE_BODY_ID = -1  # テスト用: 何も除外しない

# エージェントの位置を設定（検証用）
agent_positions = {
    'seeker_body': np.array([3.0, 3.0]),
    'hider1_body': np.array([1.0, 1.0]),
    'hider2_body': np.array([5.0, 5.0])
}

# Debug prints removed
# print("[DEBUG] Agent positions (before visibility_engine):")
# for name, pos in agent_positions.items():
#     print(f"  {name}: {pos}")

# visibility engineの初期化
visibility_engine = VisibilityEngine(model, data, epsilon=0.1, max_steps=15, max_dist=LIDAR_MAX_DIST)

# Body IDの設定
s0_body = model.body("seeker_body").id
h1_body = model.body("hider1_body").id
h2_body = model.body("hider2_body").id
box1_body = model.body("box1_body").id
box2_body = model.body("box2_body").id
ramp_body = model.body("ramp_body").id

visibility_engine.set_bodies(s0_body, h1_body, h2_body, box1_body, box2_body, ramp_body)

# 手動で動的オブジェクト位置を設定
visibility_engine.dynamic_positions[0] = agent_positions['seeker_body']
visibility_engine.dynamic_positions[1] = agent_positions['hider1_body']
visibility_engine.dynamic_positions[2] = agent_positions['hider2_body']
visibility_engine.dynamic_positions[3] = np.array([2.0, -2.0])  # box1
visibility_engine.dynamic_positions[4] = np.array([-2.0, 2.0])  # box2
visibility_engine.dynamic_positions[5] = np.array([0.0, 0.0])   # ramp

# Debug prints removed
# print("\n[DEBUG] visibility_engine positions AFTER manual set:")
# for i in range(6):
#     print(f"  Object {i}: {visibility_engine.dynamic_positions[i]}")

# 壁を線分として定義（厚みなし、SDF と同じ扱い）
# 外壁：±6 の4本の線分
# 内壁：各マップ壁の長辺のみ
# XML定義：
#   maze_w0: pos=(3.0, 1.5), size=(1.5, 0.2)  → 横長 x=[1.5,4.5], y=[1.3,1.7]
#   maze_w1: pos=(-3.0, -1.5), size=(1.5, 0.2) → 横長 x=[-4.5,-1.5], y=[-1.7,-1.3]
#   maze_w2: pos=(0.0, -3.0), size=(0.2, 1.5) → 縦長 x=[-0.1,0.1], y=[-4.5,-1.5]
#   maze_w3: pos=(0.0, 3.0), size=(0.2, 1.5)  → 縦長 x=[-0.1,0.1], y=[1.5,4.5]
wall_segments = [
    # 外壁 4本
    ((-6.0, 6.0), (6.0, 6.0)),    # 北壁
    ((-6.0, -6.0), (6.0, -6.0)),  # 南壁
    ((6.0, -6.0), (6.0, 6.0)),    # 東壁
    ((-6.0, -6.0), (-6.0, 6.0)),  # 西壁
    # 内壁 8本（4壁の上下/左右の辺）
    # maze_w0: 横長 x=[1.5,4.5], y=[1.3,1.7] → 上下の辺
    ((1.5, 1.7), (4.5, 1.7)),     # maze_w0 上辺
    ((1.5, 1.3), (4.5, 1.3)),     # maze_w0 下辺
    # maze_w1: 横長 x=[-4.5,-1.5], y=[-1.7,-1.3] → 上下の辺
    ((-4.5, -1.3), (-1.5, -1.3)), # maze_w1 上辺
    ((-4.5, -1.7), (-1.5, -1.7)), # maze_w1 下辺
    # maze_w2: 縦長 x=[-0.1,0.1], y=[-4.5,-1.5] → 左右の辺
    ((-0.1, -4.5), (-0.1, -1.5)), # maze_w2 左辺
    ((0.1, -4.5), (0.1, -1.5)),   # maze_w2 右辺
    # maze_w3: 縦長 x=[-0.1,0.1], y=[1.5,4.5] → 左右の辺
    ((-0.1, 1.5), (-0.1, 4.5)),   # maze_w3 左辺
    ((0.1, 1.5), (0.1, 4.5))      # maze_w3 右辺
]

visibility_engine.static_walls = wall_segments

# wall_segments を NumPy 配列に変換（cast_ray_direct_numba 用）
wall_segments_array = np.zeros((len(wall_segments), 4), dtype=np.float32)
for i, (p1, p2) in enumerate(wall_segments):
    wall_segments_array[i] = [p1[0], p1[1], p2[0], p2[1]]

# Visibility Engine の状態を確認
print("\n[VISIBILITY ENGINE DEBUG]")
print(f"static_walls type: {type(visibility_engine.static_walls)}")
if visibility_engine.static_walls is not None:
    print(f"static_walls length: {len(visibility_engine.static_walls)}")
    if len(visibility_engine.static_walls) > 0:
        print(f"First 3 walls: {visibility_engine.static_walls[:3]}")

print(f"\ndynamic_positions shape: {visibility_engine.dynamic_positions.shape}")
print(f"dynamic_positions: {visibility_engine.dynamic_positions}")
print(f"dynamic_radii: {visibility_engine.dynamic_radii}")
print(f"num_dynamic_objects: {visibility_engine.num_dynamic_objects}")

# SDF距離場の読み込み
try:
    from pathlib import Path
    import pickle
    cache_file = Path("sdf_distance_field.pkl")
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            data_dict = pickle.load(f)
            visibility_engine.sdf_field = data_dict.get("sdf_field")
            print(f"SDF field loaded: {visibility_engine.sdf_field.shape if visibility_engine.sdf_field is not None else 'None'}")
except Exception as e:
    print(f"Warning: Could not load SDF field: {e}")

# ============================================
# MuJoCo Viewer で環境を表示
# ============================================
print("\nLaunching MuJoCo Viewer...")
print("Close the viewer window to continue with validation.")

# エージェント位置を data に反映
data.qpos[0:3] = [agent_positions['seeker_body'][0], agent_positions['seeker_body'][1], 0.5]   # seeker
data.qpos[3:6] = [agent_positions['hider1_body'][0], agent_positions['hider1_body'][1], 0.5]   # hider1
data.qpos[6:9] = [agent_positions['hider2_body'][0], agent_positions['hider2_body'][1], 0.5]   # hider2

mujoco.mj_forward(model, data)

try:
    # 新しいMuJoCoバージョンのViewer API（macOSではmjpython必須）
    viewer = mujoco.viewer.launch_passive(model, data)
    print("Viewer is running. Interact the window or close it to continue.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
    viewer.close()
except RuntimeError as e:
    # mjpython環境ではないためスキップ
    print(f"Note: MuJoCo Viewer skipped ({str(e)[:50]}...)")
    print("MuJoCo Viewer APIが異なります。スキップします.")

print("Viewer closed. Continuing with validation...\n")

# ============================================
# SDF フィールドのデバッグ（視点周辺の値を確認）
# ============================================
print("[SDF DEBUG]")
print(f"Viewpoint: {viewpoint}")
if visibility_engine.sdf_field is not None:
    grid_size = visibility_engine.sdf_field.shape
    cell_size = 12.0 / (grid_size[0] - 1)
    
    # グリッド座標を計算
    grid_x = (viewpoint[0] + 6.0) / cell_size
    grid_y = (viewpoint[1] + 6.0) / cell_size
    
    print(f"Grid coordinates: ({grid_x:.2f}, {grid_y:.2f})")
    print(f"Grid size: {grid_size}")
    print(f"Cell size: {cell_size:.4f}")
    
    # 最近傍のグリッドポイントでの SDF 値
    x = int(grid_x + 0.5)
    y = int(grid_y + 0.5)
    if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
        sdf_val = visibility_engine.sdf_field[x, y]
        print(f"SDF value at viewpoint: {sdf_val:.4f}")
        print(f"  -> Positive (outside): {sdf_val > 0}, Negative (inside): {sdf_val < 0}")
    
print()

# ============================================
# 検証用の設定（続き）
# ============================================

# Lidar設定（main18と同じ）
surround = np.linspace(0, 2*np.pi, 8, endpoint=False)  # 45度ごとの8方向
front = np.linspace(-np.pi/6, np.pi/6, 5)  # 前方±30度の5方向
lidar_angles = np.unique(np.concatenate([surround, front]))
n_beams = len(lidar_angles)

# cos/sinを事前計算
lidar_angle_cos = np.cos(lidar_angles)
lidar_angle_sin = np.sin(lidar_angles)

# ビーム方向を計算（回転）
cy = np.cos(yaw)
sy = np.sin(yaw)
beam_cos = lidar_angle_cos * cy - lidar_angle_sin * sy
beam_sin = lidar_angle_cos * sy + lidar_angle_sin * cy

# ============================================
# 3つの方法でLidar計測
# ============================================

print("Computing Lidar distances with 3 methods...")
print(f"Viewpoint: {viewpoint}, Yaw: {yaw:.3f} rad")
print()

# デバッグ情報: 壁と動的オブジェクトの確認
# デバッグ情報: 壁と動的オブジェクトの確認
print("=" * 80)
print("ENVIRONMENT DEBUG INFO:")
print("=" * 80)
print(f"Number of wall segments: {len(visibility_engine.static_walls) if visibility_engine.static_walls else 0}")
if visibility_engine.static_walls:
    print(f"All wall segments:")
    for i, seg in enumerate(visibility_engine.static_walls):
        print(f"  Segment {i}: ({seg[0][0]:.2f}, {seg[0][1]:.2f}) -> ({seg[1][0]:.2f}, {seg[1][1]:.2f})")

print(f"\nNumber of dynamic objects: {visibility_engine.num_dynamic_objects}")
print(f"Dynamic radii: {visibility_engine.dynamic_radii[:visibility_engine.num_dynamic_objects]}")
if visibility_engine.num_dynamic_objects > 0:
    print(f"Dynamic objects:")
    for i in range(visibility_engine.num_dynamic_objects):
        pos = visibility_engine.dynamic_positions[i]
        radius = visibility_engine.dynamic_radii[i]
        print(f"  Object {i}: pos=({pos[0]:.2f}, {pos[1]:.2f}), radius={radius:.2f}")

print(f"\nSDF Field loaded: {visibility_engine.sdf_field is not None}")
if visibility_engine.sdf_field is not None:
    print(f"SDF Field shape: {visibility_engine.sdf_field.shape}")
print("=" * 80)
print()

distances_direct = []
distances_sphere = []
distances_mujoco = []

# ビーム方向の確認（最初の3ビーム）
print("\n[BEAM DIRECTION DEBUG]")
print(f"Lidar angles (degrees): {np.degrees(lidar_angles)}")
for i in range(min(3, n_beams)):
    dir_mag = np.sqrt(beam_cos[i]**2 + beam_sin[i]**2)
    beam_angle = np.arctan2(beam_sin[i], beam_cos[i])
    print(f"Beam {i}: direction=({beam_cos[i]:.4f}, {beam_sin[i]:.4f}), magnitude={dir_mag:.4f}, angle={np.degrees(beam_angle):.2f}°")

# Box1 が実際にどこにあるか確認
print("\n[BOX1 POSITION CHECK]")
box1_geom = None
for geom_id in range(model.ngeom):
    geom = model.geom(geom_id)
    if geom.name == "box1_geom":
        box1_geom = geom
        print(f"box1_geom found:")
        print(f"  Body ID: {geom.bodyid}")
        print(f"  Body name: {model.body(int(geom.bodyid)).name}")
        print(f"  Geom position (relative to body): {geom.pos}")
        print(f"  Geom size: {geom.size}")
        print(f"  Geom type: {geom.type}")
        # 実際のワールド座標でのBox1位置
        body = model.body(int(geom.bodyid))
        world_pos = body.pos + geom.pos
        print(f"  Body world position: {body.pos}")
        print(f"  Geom world position: {world_pos}")
        break

# Beam 1 の詳細位置情報（紫の箱をチェック）
print("\n[BEAM 1 ANALYSIS]")
print(f"Viewpoint: {viewpoint}")
print(f"Beam 1 direction (normalized): ({beam_cos[1]:.4f}, {beam_sin[1]:.4f})")
print(f"Box1 (purple): pos=(2.0, -2.0), radius=0.85")
# Beam 1 が箱方向を向いているか確認
box1_from_view = np.array([2.0 - viewpoint[0], -2.0 - viewpoint[1]])
angle_to_box = np.arctan2(box1_from_view[1], box1_from_view[0])
beam_angle = np.arctan2(beam_sin[1], beam_cos[1])
angle_diff = np.abs(np.degrees(angle_to_box - beam_angle))
if angle_diff > 180:
    angle_diff = 360 - angle_diff
print(f"Angle to Box1: {np.degrees(angle_to_box):.2f}°")
print(f"Beam 1 angle: {np.degrees(beam_angle):.2f}°")
print(f"Difference: {angle_diff:.2f}°")

# Box1 に向かうビームを特定
print("\n[WHICH BEAM HITS BOX1?]")
for i in range(n_beams):
    beam_angle = np.arctan2(beam_sin[i], beam_cos[i])
    angle_diff = np.abs(np.degrees(angle_to_box - beam_angle))
    if angle_diff > 180:
        angle_diff = 360 - angle_diff
    if angle_diff < 15:  # 15度以内
        dist_to_box = np.linalg.norm(box1_from_view)
        print(f"Beam {i}: angle={np.degrees(beam_angle):.2f}°, dist_to_box_center={dist_to_box:.3f}m, angle_diff={angle_diff:.2f}°")

# Beam 10 の詳細出力
print("\n[BEAM 10 DETAILED ANALYSIS]")
print(f"Beam 10 direction: ({beam_cos[10]:.4f}, {beam_sin[10]:.4f})")
print(f"Direction angle: {np.degrees(np.arctan2(beam_sin[10], beam_cos[10])):.2f}°")
print(f"Box1 center: (2.0, -2.0), radius: 0.85")
print(f"Expected hit distance (to box surface): ~{np.linalg.norm(box1_from_view) - 0.85:.3f}m")
print()


# 各ビームについて計測
for i in range(n_beams):
    direction = np.array([beam_cos[i], beam_sin[i], 0.0], dtype=np.float64)
    
    # Method 1: Direct Intersection
    dist_dir, _ = cast_ray_direct_numba(
        viewpoint[0], viewpoint[1],
        direction[0], direction[1],
        visibility_engine.dynamic_positions,
        visibility_engine.dynamic_radii,
        visibility_engine.dynamic_body_ids_array,
        visibility_engine.num_dynamic_objects,
        wall_segments_array,
        len(wall_segments),
        SEEKER_BODY_ID, EXCLUDE_BODY_ID,
        LIDAR_MAX_DIST
    )
    
    # DEBUG: Beam 11と6の詳細を確認
    if i == 11:
        print(f"\n[BEAM 11 DEBUG]")
        print(f"Direction: ({direction[0]:.4f}, {direction[1]:.4f})")
        print(f"Direct result: {dist_dir:.3f}m")
        print(f"Dynamic objects:")
        for j in range(visibility_engine.num_dynamic_objects):
            bid = visibility_engine.dynamic_body_ids_array[j]
            pos = visibility_engine.dynamic_positions[j]
            rad = visibility_engine.dynamic_radii[j]
            from main23_sightmap_optimized import ray_circle_intersection_numba
            dist = ray_circle_intersection_numba(
                viewpoint[0], viewpoint[1], direction[0], direction[1],
                pos[0], pos[1], rad
            )
            if dist < 2.0:  # 近いものだけ表示
                print(f"  Object {j}: pos=({pos[0]:.1f},{pos[1]:.1f}), r={rad:.2f}, dist={dist:.3f}m")
    
    if i == 6:
        print(f"\n[BEAM 6 DEBUG]")
        print(f"Direction: ({direction[0]:.4f}, {direction[1]:.4f})")
        print(f"Direct result: {dist_dir:.3f}m, Sphere will be computed next")
    
    distances_direct.append(dist_dir)
    
    # Method 2: Sphere Tracing
    # SDF場がない場合はスキップ
    if visibility_engine.sdf_field is not None:
        grid_size = visibility_engine.sdf_field.shape
        cell_size = 12.0 / (grid_size[0] - 1)
        globals()['_CELL_SIZE'] = cell_size  # グローバルに保存（プロット用）
        dist_sph, hit = cast_ray_numba(
            viewpoint[0], viewpoint[1],
            direction[0], direction[1],
            visibility_engine.sdf_field, grid_size[0], grid_size[1], cell_size,
            visibility_engine.dynamic_positions, visibility_engine.dynamic_radii,
            visibility_engine.dynamic_body_ids_array,
            visibility_engine.num_dynamic_objects,
            SEEKER_BODY_ID, EXCLUDE_BODY_ID,
            cell_size, 300, LIDAR_MAX_DIST  # epsilon = cell_size (セルサイズ以上必須)
        )
    else:
        dist_sph = LIDAR_MAX_DIST
    distances_sphere.append(dist_sph)
    
    # DEBUG: Beam 6のSphere Tracingの詳細軌跡を記録
    if i == 6 and visibility_engine.sdf_field is not None:
        print(f"\n[BEAM 6 SPHERE TRACING DEBUG]")
        print(f"Direction: ({direction[0]:.4f}, {direction[1]:.4f})")
        
        # Sphere Tracingを手動で実行（デバッグ出力付き）
        grid_size = visibility_engine.sdf_field.shape
        cell_size = 12.0 / (grid_size[0] - 1)
        
        # 軌跡を記録
        trajectory = []
        positions_x = []
        positions_y = []
        sdf_values = []
        
        curr_x = viewpoint[0]
        curr_y = viewpoint[1]
        total_d = 0.0
        epsilon = cell_size  # ★SDF グリッドセルサイズに設定（重要: 閾値は cell_size 以上必須）
        max_steps = 300
        
        print(f"Grid size: {grid_size}, Cell size: {cell_size:.6f}")
        print(f"Epsilon threshold: {epsilon:.6f}m")
        print(f"Start pos: ({curr_x:.3f}, {curr_y:.3f})")
        
        from main23_sightmap_optimized import sdf_lookup_numba
        import math
        
        for step in range(max_steps):
            # SDF値を取得
            d_static = sdf_lookup_numba(curr_x, curr_y, visibility_engine.sdf_field, grid_size[0], grid_size[1], cell_size)
            
            # 動的SDF
            d_dynamic = 1e6
            for j in range(visibility_engine.num_dynamic_objects):
                bid = visibility_engine.dynamic_body_ids_array[j]
                if bid != -1:
                    dx = visibility_engine.dynamic_positions[j, 0] - curr_x
                    dy = visibility_engine.dynamic_positions[j, 1] - curr_y
                    dist = math.sqrt(dx * dx + dy * dy) - visibility_engine.dynamic_radii[j]
                    if dist < d_dynamic:
                        d_dynamic = dist
            
            d = min(d_static, d_dynamic)
            
            positions_x.append(curr_x)
            positions_y.append(curr_y)
            sdf_values.append(d)
            
            if step < 10 or step % 30 == 0:  # 最初の10ステップと30ステップごとに表示
                print(f"  Step {step}: pos=({curr_x:.3f}, {curr_y:.3f}), d={d:.6f}, total_d={total_d:.3f}m")
            
            # 負の距離 → 障害物内部
            if d < epsilon:
                print(f"  Step {step}: COLLISION! d={d:.6f} < epsilon={epsilon}")
                break
            
            total_d += d
            dir_mag = math.sqrt(direction[0]**2 + direction[1]**2)
            dir_norm_x = direction[0] / dir_mag
            dir_norm_y = direction[1] / dir_mag
            
            curr_x += dir_norm_x * d
            curr_y += dir_norm_y * d
            
            if total_d > LIDAR_MAX_DIST:
                print(f"  Step {step}: EXCEEDED MAX_DIST ({total_d:.3f}m > {LIDAR_MAX_DIST}m)")
                break
        
        print(f"Final: total_d={total_d:.3f}m, final_pos=({curr_x:.3f}, {curr_y:.3f})")
        
        # 軌跡をプロット用に保存
        globals()['beam6_trajectory'] = {
            'positions_x': positions_x,
            'positions_y': positions_y,
            'sdf_values': sdf_values
        }
    else:
        dist_sph = LIDAR_MAX_DIST
    
    # Method 3: MuJoCo mj_ray
    raycast_from = np.array([viewpoint[0], viewpoint[1], 0.5], dtype=np.float64)
    dir_mag = np.sqrt(direction[0]**2 + direction[1]**2)
    if dir_mag > 1e-10:
        raycast_dir = np.array([direction[0] / dir_mag, direction[1] / dir_mag, 0.0], dtype=np.float64)
    else:
        raycast_dir = np.array([direction[0], direction[1], 0.0], dtype=np.float64)
    
    raycast_geomid = np.array([-1], dtype=np.int32)
    
    dist_muj = mujoco.mj_ray(
        model, data, 
        raycast_from, 
        raycast_dir, 
        None, 1, SEEKER_BODY_ID,
        raycast_geomid
    )
    distances_mujoco.append(dist_muj if dist_muj >= 0 else LIDAR_MAX_DIST)
    
    # デバッグ: すべてのビームでヒット情報を記録
    if raycast_geomid[0] >= 0:
        hit_geom = model.geom(raycast_geomid[0])
        body_id = int(hit_geom.bodyid)
        hit_body = model.body(body_id).name
        
        # レイの終端を計算
        end_point = raycast_from + raycast_dir * dist_muj
        
        # Beam 10 の詳細ログ
        if i == 10:
            print(f"[BEAM 10 MuJoCo Debug]")
            print(f"  Ray from: {raycast_from}, dir: {raycast_dir}")
            print(f"  Exclude body ID: {SEEKER_BODY_ID}")
            print(f"  Hit: {hit_body} (body_id={body_id}), geom: {hit_geom.name}")
            print(f"  Hit endpoint: {end_point}")
            print(f"  Distance: {dist_muj:.3f}m")
            
            # Box1の body_id を確認
            box1_body_id = model.body("box1_body").id
            print(f"  Box1 body ID: {box1_body_id}")
            print(f"  Box1 is being excluded?: {box1_body_id == SEEKER_BODY_ID}")
            
            # モデルの全ジオメトリをリスト
            print(f"  All geoms in order:")
            for gid in range(model.ngeom):
                g = model.geom(gid)
                b = model.body(int(g.bodyid)).name
                if "box" in b or "box" in g.name:
                    print(f"    Geom {gid}: {g.name} (body: {b})")
            
            # Box1 に向かうダイレクトレイテストを実行（除外なし）
            print(f"  Testing mj_ray with no exclusion:")
            geomid_test = np.array([-1], dtype=np.int32)
            dist_test = mujoco.mj_ray(
                model, data, 
                raycast_from, 
                raycast_dir, 
                None, 1, -1,  # exclude_body = -1 (なし)
                geomid_test
            )
            if geomid_test[0] >= 0:
                test_geom = model.geom(geomid_test[0])
                test_body = model.body(int(test_geom.bodyid)).name
                print(f"    No exclusion: Hit {test_body} ({test_geom.name}) at {dist_test:.3f}m")
            else:
                print(f"    No exclusion: No hit")


        
        # 通常のログ
        print(f"Beam {i}: Ray hit {hit_body} (geom: {hit_geom.name}) at distance {dist_muj:.3f}")





distances_direct = np.array(distances_direct)
distances_sphere = np.array(distances_sphere)
distances_mujoco = np.array(distances_mujoco)

# ============================================
# 結果の表示
# ============================================

print("Lidar Distances Comparison:")
print("-" * 80)
print(f"{'Beam':<6} {'Direct':<12} {'Sphere':<12} {'MuJoCo':<12} {'Max Diff':<12}")
print("-" * 80)

max_diffs = []
for i in range(n_beams):
    d = distances_direct[i]
    s = distances_sphere[i]
    m = distances_mujoco[i]
    diff = max(abs(d - s), abs(d - m), abs(s - m))
    max_diffs.append(diff)
    print(f"{i:<6} {d:<12.3f} {s:<12.3f} {m:<12.3f} {diff:<12.3f}")

print("-" * 80)
print(f"Average distances: Direct={distances_direct.mean():.3f}, Sphere={distances_sphere.mean():.3f}, MuJoCo={distances_mujoco.mean():.3f}")
print(f"Max difference: {max(max_diffs):.3f}")

print()

# ============================================
# 可視化
# ============================================

# グリッドプロット用の関数
def plot_environment_with_rays(ax, distances, method_name):
    ax.set_title(f'{method_name}\n(Avg: {distances.mean():.2f}m)', fontsize=11, fontweight='bold')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    
    # 環境描画（簡略版）
    # 外壁
    ax.plot([-6, 6], [6, 6], 'k-', linewidth=2)  # North
    ax.plot([-6, 6], [-6, -6], 'k-', linewidth=2)  # South
    ax.plot([6, 6], [-6, 6], 'k-', linewidth=2)  # East
    ax.plot([-6, -6], [-6, 6], 'k-', linewidth=2)  # West
    
    # 内壁（矩形として描画）
    # maze_w0: x∈[1.5,4.5], y∈[1.3,1.7]
    rect = plt.Rectangle((1.5, 1.3), 3.0, 0.4, fill=False, edgecolor='cyan', linewidth=1.5)
    ax.add_patch(rect)
    # maze_w1: x∈[-4.5,-1.5], y∈[-1.7,-1.3]
    rect = plt.Rectangle((-4.5, -1.7), 3.0, 0.4, fill=False, edgecolor='cyan', linewidth=1.5)
    ax.add_patch(rect)
    # maze_w2: x∈[-0.2,0.2], y∈[-4.5,-1.5]
    rect = plt.Rectangle((-0.2, -4.5), 0.4, 3.0, fill=False, edgecolor='cyan', linewidth=1.5)
    ax.add_patch(rect)
    # maze_w3: x∈[-0.2,0.2], y∈[1.5,4.5]
    rect = plt.Rectangle((-0.2, 1.5), 0.4, 3.0, fill=False, edgecolor='cyan', linewidth=1.5)
    ax.add_patch(rect)
    
    # 動的オブジェクト
    circle = plt.Circle((3.0, 3.0), 0.4, color='red', alpha=0.5)
    ax.add_patch(circle)
    circle = plt.Circle((1.0, 1.0), 0.4, color='green', alpha=0.5)
    ax.add_patch(circle)
    circle = plt.Circle((5.0, 5.0), 0.4, color='orange', alpha=0.5)
    ax.add_patch(circle)
    circle = plt.Circle((2.0, -2.0), 0.85, color='purple', alpha=0.5)
    ax.add_patch(circle)
    circle = plt.Circle((-2.0, 2.0), 0.85, color='brown', alpha=0.5)
    ax.add_patch(circle)
    
    # Ramp (0, 0), radius=0.84
    circle = plt.Circle((0.0, 0.0), 0.84, color='gray', alpha=0.5)
    ax.add_patch(circle)
    ax.text(0.0, 0.0, 'Ramp', fontsize=8, ha='center', va='center', color='black')
    
    # 視点と光線
    ax.plot(viewpoint[0], viewpoint[1], 'ko', markersize=12, zorder=100)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_beams))
    for i in range(n_beams):
        dist = distances[i]
        end_x = viewpoint[0] + beam_cos[i] * min(dist, LIDAR_MAX_DIST)
        end_y = viewpoint[1] + beam_sin[i] * min(dist, LIDAR_MAX_DIST)
        ax.plot([viewpoint[0], end_x], [viewpoint[1], end_y], color=colors[i], linewidth=1, alpha=0.6)
        ax.plot(end_x, end_y, 'o', color=colors[i], markersize=6)
        # ビーム番号を表示
        ax.text(end_x, end_y, str(i), fontsize=8, ha='center', va='center', 
                color='white', weight='bold', bbox=dict(boxstyle='circle', facecolor=colors[i], edgecolor='black', linewidth=0.5))
    
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

fig = plt.figure(figsize=(18, 12))

# Subplot 1-3: 3つの方法の環境図
ax1 = plt.subplot(2, 3, 1)
plot_environment_with_rays(ax1, distances_direct, "Direct Intersection")

ax2 = plt.subplot(2, 3, 2)
plot_environment_with_rays(ax2, distances_sphere, "Sphere Tracing")

ax3 = plt.subplot(2, 3, 3)
plot_environment_with_rays(ax3, distances_mujoco, "MuJoCo Ray")

# Subplot 4: 距離比較
ax4 = plt.subplot(2, 3, 4)
x = np.arange(n_beams)
width = 0.25
ax4.bar(x - width, distances_direct, width, label='Direct', alpha=0.8)
ax4.bar(x, distances_sphere, width, label='Sphere', alpha=0.8)
ax4.bar(x + width, distances_mujoco, width, label='MuJoCo', alpha=0.8)
ax4.axhline(y=LIDAR_MAX_DIST, color='r', linestyle='--', alpha=0.5)
ax4.set_xlabel('Beam Index')
ax4.set_ylabel('Distance (m)')
ax4.set_title('Distance Comparison')
ax4.set_xticks(x)
ax4.set_xticklabels([str(i) for i in range(n_beams)], fontsize=8)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Subplot 5: 差分比較
ax5 = plt.subplot(2, 3, 5)
diff_ds = np.abs(distances_direct - distances_sphere)
diff_dm = np.abs(distances_direct - distances_mujoco)
diff_sm = np.abs(distances_sphere - distances_mujoco)
ax5.plot(x, diff_ds, 'o-', label='|Direct-Sphere|', linewidth=2)
ax5.plot(x, diff_dm, 's-', label='|Direct-MuJoCo|', linewidth=2)
ax5.plot(x, diff_sm, '^-', label='|Sphere-MuJoCo|', linewidth=2)
ax5.set_xlabel('Beam Index')
ax5.set_ylabel('Distance Difference (m)')
ax5.set_title('Pairwise Differences')
ax5.set_xticks(x)
ax5.set_xticklabels([str(i) for i in range(n_beams)], fontsize=8)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Subplot 6: 統計情報
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
stats_text = f"""
METHOD STATISTICS

Direct Intersection:
  Mean: {distances_direct.mean():.3f}m
  Std:  {distances_direct.std():.3f}m
  Min:  {distances_direct.min():.3f}m
  Max:  {distances_direct.max():.3f}m

Sphere Tracing:
  Mean: {distances_sphere.mean():.3f}m
  Std:  {distances_sphere.std():.3f}m
  Min:  {distances_sphere.min():.3f}m
  Max:  {distances_sphere.max():.3f}m

MuJoCo Ray:
  Mean: {distances_mujoco.mean():.3f}m
  Std:  {distances_mujoco.std():.3f}m
  Min:  {distances_mujoco.min():.3f}m
  Max:  {distances_mujoco.max():.3f}m

Mean Differences:
  |D-S|: {np.abs(distances_direct - distances_sphere).mean():.3f}m
  |D-M|: {np.abs(distances_direct - distances_mujoco).mean():.3f}m
  |S-M|: {np.abs(distances_sphere - distances_mujoco).mean():.3f}m
"""
ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('lidar_validation_3methods.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to 'lidar_validation_3methods.png'")

# ビーム6の軌跡をプロット
if 'beam6_trajectory' in globals():
    print("\n[BEAM 6 TRAJECTORY PLOT]")
    fig_beam6 = plt.figure(figsize=(14, 6))
    
    traj = globals()['beam6_trajectory']
    pos_x = traj['positions_x']
    pos_y = traj['positions_y']
    sdf_val = traj['sdf_values']
    
    # 軌跡プロット
    ax_traj = plt.subplot(1, 2, 1)
    ax_traj.set_title('Beam 6 Sphere Tracing Trajectory', fontsize=12, fontweight='bold')
    
    # 環境描画
    ax_traj.plot([-6, 6], [6, 6], 'k-', linewidth=2)
    ax_traj.plot([-6, 6], [-6, -6], 'k-', linewidth=2)
    ax_traj.plot([6, 6], [-6, 6], 'k-', linewidth=2)
    ax_traj.plot([-6, -6], [-6, 6], 'k-', linewidth=2)
    
    # 内壁
    rect = plt.Rectangle((1.5, 1.3), 3.0, 0.4, fill=False, edgecolor='cyan', linewidth=1.5)
    ax_traj.add_patch(rect)
    rect = plt.Rectangle((-4.5, -1.7), 3.0, 0.4, fill=False, edgecolor='cyan', linewidth=1.5)
    ax_traj.add_patch(rect)
    rect = plt.Rectangle((-0.2, -4.5), 0.4, 3.0, fill=False, edgecolor='cyan', linewidth=1.5)
    ax_traj.add_patch(rect)
    rect = plt.Rectangle((-0.2, 1.5), 0.4, 3.0, fill=False, edgecolor='cyan', linewidth=1.5)
    ax_traj.add_patch(rect)
    
    # 軌跡をプロット（色はSDF値で段階的に）
    if len(pos_x) > 1:
        ax_traj.plot(pos_x, pos_y, 'b-', linewidth=2, alpha=0.6, label='Trajectory')
        ax_traj.plot(pos_x[0], pos_y[0], 'go', markersize=10, label='Start')
        ax_traj.plot(pos_x[-1], pos_y[-1], 'ro', markersize=10, label='End')
        
        # ステップポイント（10ステップごと）
        for i in range(0, len(pos_x), max(1, len(pos_x)//20)):
            ax_traj.plot(pos_x[i], pos_y[i], 'b.', markersize=3, alpha=0.5)
    
    # Beam6の終点（Direct Intersection）
    beam6_end_x = viewpoint[0] + beam_cos[6] * distances_direct[6]
    beam6_end_y = viewpoint[1] + beam_sin[6] * distances_direct[6]
    ax_traj.plot([viewpoint[0], beam6_end_x], [viewpoint[1], beam6_end_y], 'r--', linewidth=1.5, alpha=0.5, label='Direct Intersection')
    
    ax_traj.set_xlim(-7, 7)
    ax_traj.set_ylim(-7, 7)
    ax_traj.set_aspect('equal')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend()
    ax_traj.set_xlabel('X-coordinate')
    ax_traj.set_ylabel('Y-coordinate')
    
    # SDF値の推移
    ax_sdf = plt.subplot(1, 2, 2)
    ax_sdf.set_title('SDF Values Along Trajectory', fontsize=12, fontweight='bold')
    ax_sdf.plot(range(len(sdf_val)), sdf_val, 'b-', linewidth=1.5, alpha=0.7)
    
    # epsilonラインをプロット（cell_sizeから取得）
    epsilon_val = globals().get('_CELL_SIZE', 0.020374)
    ax_sdf.axhline(y=epsilon_val, color='r', linestyle='--', linewidth=2, label=f'epsilon={epsilon_val:.6f}m (cell_size)')
    ax_sdf.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax_sdf.set_xlabel('Step')
    ax_sdf.set_ylabel('SDF Distance (m)')
    ax_sdf.grid(True, alpha=0.3)
    ax_sdf.legend()
    
    plt.tight_layout()
    plt.savefig('beam6_trajectory.png', dpi=150, bbox_inches='tight')
    print("✓ Beam 6 trajectory plot saved to 'beam6_trajectory.png'")
    plt.close(fig_beam6)
plt.close()

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print(f"""
Sphere Tracing vs MuJoCo Raycast:

Sphere Tracing Statistics:
  Mean: {distances_sphere.mean():.3f}m
  Std:  {distances_sphere.std():.3f}m
  Min:  {distances_sphere.min():.3f}m
  Max:  {distances_sphere.max():.3f}m

MuJoCo Ray Statistics:
  Mean: {distances_mujoco.mean():.3f}m
  Std:  {distances_mujoco.std():.3f}m
  Min:  {distances_mujoco.min():.3f}m
  Max:  {distances_mujoco.max():.3f}m

Mean Difference (|Sphere - MuJoCo|): {np.abs(distances_sphere - distances_mujoco).mean():.3f}m
""")
print("="*80)
print("\nValidation complete!")


