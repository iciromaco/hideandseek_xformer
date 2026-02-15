# visibilityMap.py
import xml.etree.ElementTree as ET
import pickle
import numpy as np
import time
from pathlib import Path
from typing import List, Tuple, Dict

# MuJoCoライブラリは動的障害物の可視性チェックで必要
# このセルではインポートせず、動的障害物関数が定義されるセルでインポート
# import mujoco 
# from scipy.spatial import ConvexHull 

XML = """
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

# ==========================================
# 設定
# ==========================================
ENV_BOUNDS = 5.9        # 環境範囲 (±6)
CELL_SIZE = 0.2         # セルサイズ
MAZE_PREFIX = "maze_"   # 内壁の接頭語

# ==========================================
# 幾何学関数
# ==========================================
def ccw(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
    """時計回り判定"""
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def segments_intersect(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> bool:
    """線分AB と CD が交差するかどうか"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def rect_to_segments(x_min: float, y_min: float, x_max: float, y_max: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """矩形を4本の線分に分解"""
    return [
        ((x_min, y_min), (x_max, y_min)),  # 下辺
        ((x_max, y_min), (x_max, y_max)),  # 右辺
        ((x_max, y_max), (x_min, y_max)),  # 上辺
        ((x_min, y_max), (x_min, y_min)),  # 左辺
    ]

def is_visible(point1: np.ndarray, point2: np.ndarray, wall_segments: List) -> bool:
    """2点が互いに見えるか判定（中の壁のみチェック）"""
    p1 = np.array(point1, dtype=np.float32)
    p2 = np.array(point2, dtype=np.float32)

    for wall_segment in wall_segments:
        seg_start = np.array(wall_segment[0], dtype=np.float32)
        seg_end = np.array(wall_segment[1], dtype=np.float32)
        if segments_intersect(p1, p2, seg_start, seg_end):
            return False
    return True

# ==========================================
# XMLパース関数
# ==========================================
def extract_maze_walls_from_xml(xml_source: str, from_string: bool = False) -> List[Dict]:
    """
    XMLから maze_ で始まる geom 要素を抽出
    xml_source: XMLファイルパスまたはXML文字列
    from_string: xml_sourceがXML文字列の場合True

    戻り値:
        [(name, pos, size), ...] のリスト
    """
    if from_string:
        root = ET.fromstring(xml_source)
    else:
        tree = ET.parse(xml_source)
        root = tree.getroot()

    maze_walls = []

    # すべての geom 要素を探索
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")

        # maze_ で始まる要素だけ抽出
        if name.startswith(MAZE_PREFIX):
            pos_str = geom.get("pos", "0 0 0")
            size_str = geom.get("size", "0 0 0")
            geom_type = geom.get("type", "")

            # 3次元の値を取得
            pos = list(map(float, pos_str.split()))
            size = list(map(float, size_str.split()))

            maze_walls.append({
                'name': name,
                'pos': pos,
                'size': size,
                'type': geom_type
            })
            print(f"Found: {name}, pos={pos}, size={size}, type={geom_type}")

    return maze_walls

def walls_to_segments(maze_walls: List[Dict]) -> List:
    """
    MuJoCoの矩形壁情報をlineSegmentリストに変換
    MuJoCoでは size は半幅、pos は中心座標
    2D投影として x-y 平面を使用
    """
    wall_segments = []

    for wall in maze_walls:
        pos = wall['pos']
        size = wall['size']

        # 中心座標
        cx, cy = pos[0], pos[1]

        # MuJoCoの size は半幅なので、実際の座標は pos ± size
        x_min = cx - size[0]
        x_max = cx + size[0]
        y_min = cy - size[1]
        y_max = cy + size[1]

        print(f"  Wall {wall['name']}: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

        segments = rect_to_segments(x_min, y_min, x_max, y_max)
        wall_segments.extend(segments)

    return wall_segments

# ==========================================
# セルグリッド生成
# ==========================================
def create_cell_grid(bounds: float, cell_size: float) -> Tuple[np.ndarray, Dict]:
    """
    セルグリッドを生成

    Args:
        bounds: 環境範囲 (±bounds)
        cell_size: セルサイズ

    戻り値:
        cell_centers: セル中心座標 (N, 2)
        metadata: メタデータ辞書
    """
    x_cells = np.arange(-bounds, bounds, cell_size)
    y_cells = np.arange(-bounds, bounds, cell_size)

    cell_centers = []
    for x in x_cells:
        for y in y_cells:
            cell_centers.append([x + cell_size/2, y + cell_size/2])

    cell_centers = np.array(cell_centers, dtype=np.float32)
    num_cells = len(cell_centers)

    metadata = {
        'bounds': bounds,
        'cell_size': cell_size,
        'num_cells': num_cells,
        'x_range': [x_cells[0], x_cells[-1]],
        'y_range': [y_cells[0], y_cells[-1]],
    }

    print(f"nCell Grid Info:")
    print(f"  Total cells: {num_cells}")
    print(f"  Grid: {len(x_cells)} x {len(y_cells)}")
    print(f"  Cell size: {cell_size}")

    return cell_centers, metadata

# ==========================================
# キャッシュ構築
# ==========================================
def build_visibility_cache(cell_centers: np.ndarray, wall_segments: List) -> Dict:
    """
    セル間の可視性をすべて計算してキャッシュ化

    戻り値:
        {(i, j): bool, ...}
    """
    num_cells = len(cell_centers)
    cache = {}

    print(f"nBuilding visibility cache...")
    print(f"  Total cell pairs: {num_cells * (num_cells - 1) // 2}")

    count = 0
    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            p1 = cell_centers[i]
            p2 = cell_centers[j]
            cache[(i, j)] = is_visible(p1, p2, wall_segments)
            count += 1

        # 進捗表示（10%刻み）
        if (i + 1) % max(1, num_cells // 10) == 0:
            print(f"  Progress: {(i + 1) / num_cells * 100:.1f}%")

    print(f"Cache built: {count} pairs")
    return cache

# ==========================================
# セーブ・ロード
# ==========================================
def save_cache(cache: Dict, cell_centers: np.ndarray, metadata: Dict, output_file: str):
    """キャッシュを pickle で保存"""
    data = {
        'cache': cache,
        'cell_centers': cell_centers,
        'metadata': metadata,
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    file_size = Path(output_file).stat().st_size / 1024  # KB
    print(f"n✓ Cache saved to: {output_file}")
    print(f"  File size: {file_size:.1f} KB")

def load_cache(cache_file: str) -> Tuple[Dict, np.ndarray, Dict]:
    """キャッシュを pickle から読み込み"""
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)

    return data['cache'], data['cell_centers'], data['metadata']

# ==========================================
# メイン処理
# ==========================================
def main(xml_source: str, output_cache_file: str = "visibility_cache.pkl", is_xml_string: bool = False):
    """
    可視性キャッシュを構築・保存

    Args:
        xml_source: 入力XMLファイルパスまたはXML文字列
        output_cache_file: 出力キャッシュファイルパス
        is_xml_string: xml_sourceがXML文字列の場合True
    """
    print("=" * 60)
    print("Visibility Cache Builder")
    print("=" * 60)

    # 1. XMLからmaze壁を抽出
    if is_xml_string:
        print(f"n[1] Extracting maze walls from XML string...")
    else:
        print(f"n[1] Extracting maze walls from XML file: {xml_source}")
    maze_walls = extract_maze_walls_from_xml(xml_source, from_string=is_xml_string)
    print(f"  Found {len(maze_walls)} maze walls")

    if len(maze_walls) == 0:
        print("  WARNING: No maze walls found!")

    # 2. 壁を線分に変換
    print(f"n[2] Converting walls to line segments")
    wall_segments = walls_to_segments(maze_walls)
    print(f"  Total segments: {len(wall_segments)}")

    # 3. セルグリッドを生成
    print(f"n[3] Creating cell grid")
    cell_centers, metadata = create_cell_grid(ENV_BOUNDS, CELL_SIZE)

    # 4. 可視性キャッシュを構築
    print(f"n[4] Computing visibility")
    visibility_cache = build_visibility_cache(cell_centers, wall_segments)

    # 5. 保存
    print(f"n[5] Saving cache")
    save_cache(visibility_cache, cell_centers, metadata, output_cache_file)

    print("n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return visibility_cache, cell_centers, metadata

# ==========================================
# テスト・ユーティリティ
# ==========================================
def test_loaded_cache(cache_file: str, cell_centers: np.ndarray = None):
    """保存されたキャッシュをテストロード"""
    print(f"n[Test] Loading cache from: {cache_file}")
    cache, loaded_cell_centers, metadata = load_cache(cache_file)

    print(f"  Loaded {len(cache)} pairs")
    print(f"  Metadata: {metadata}")
    print(f"  Cell centers shape: {loaded_cell_centers.shape}")

    # ランダムなペアを確認
    if len(cache) > 0:
        sample_key = list(cache.keys())[0]
        print(f"  Sample pair {sample_key}: visible={cache[sample_key]}")

    return cache, loaded_cell_centers, metadata

# MuJoCoとSciPyのインストールとインポート
!pip install mujoco
from scipy.spatial import ConvexHull
import mujoco # `mujoco`を再度インポート（セルの独立性を保つため）

# --- Helper function: get_2d_geom_segments (8頂点凸包による正確な2D投影) ---
def get_2d_geom_segments(model: mujoco.MjModel, data: mujoco.MjData, geom_names: List[str]) -> List[Dict]:
    """
    MuJoCoのmj_modelとmj_dataを使用して、指定されたgeomの現在のワールド座標における
    2D線分を抽出するヘルパー関数。
    ジオムの8つの3D頂点をワールド座標に変換し、XY平面に投影した後、
    その2D点の凸包（Convex Hull）を計算して2D線分を抽出する。

    Args:
        model: mj_modelオブジェクト。
        data: mj_dataオブジェクト。
        geom_names: 2D線分を抽出するgeomの名前のリスト。

    Returns:
        各geomに関する情報（名前、中心、凸包線分のリスト）を含む辞書のリスト。
        各線分は ((x1, y1), (x2, y2)) の形式。
    """
    structured_dynamic_segments = []

    for geom_name in geom_names:
        try:
            geom_id = model.geom(geom_name).id
        except KeyError:
            print(f"Warning: Geom '{geom_name}' not found. Skipping.")
            continue

        sx, sy, sz = model.geom_size[geom_id]
        geom_xpos = data.geom_xpos[geom_id]
        geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)

        local_corners_3d = np.array([
            [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz], # bottom face
            [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]     # top face
        ])

        world_corners_2d = []
        for lc in local_corners_3d:
            wc_3d = geom_xpos + np.dot(geom_xmat, lc)
            world_corners_2d.append([wc_3d[0], wc_3d[1]])
        
        world_corners_2d_np = np.array(world_corners_2d)

        geom_segments = []
        try:
            unique_world_corners_2d_np = np.unique(world_corners_2d_np, axis=0)

            if len(unique_world_corners_2d_np) >= 3:
                hull = ConvexHull(unique_world_corners_2d_np)
                for simplex in hull.simplices:
                    p1 = tuple(unique_world_corners_2d_np[simplex[0]])
                    p2 = tuple(unique_world_corners_2d_np[simplex[1]])
                    geom_segments.append((p1, p2))
            elif len(unique_world_corners_2d_np) == 2:
                p1 = tuple(unique_world_corners_2d_np[0])
                p2 = tuple(unique_world_corners_2d_np[1])
                geom_segments.append((p1, p2))
            elif len(unique_world_corners_2d_np) == 1:
                geom_segments = []

        except Exception as e:
            print(f"Warning: Could not compute convex hull for geom '{geom_name}': {e}")
            x_coords = world_corners_2d_np[:, 0]
            y_coords = world_corners_2d_np[:, 1]
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            geom_segments = [
                ((min_x, min_y), (max_x, min_y)),
                ((max_x, min_y), (max_x, max_y)),
                ((max_x, max_y), (min_x, max_y)),
                ((min_x, max_y), (min_x, min_y))
            ]
        
        structured_dynamic_segments.append({
            'name': geom_name,
            'center_2d': np.array([geom_xpos[0], geom_xpos[1]]),
            'segments': geom_segments
        })

    return structured_dynamic_segments


# --- is_agent_visible function (静的壁と動的障害物を考慮) ---
def is_agent_visible(
    point_a: np.ndarray,
    point_b: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wall_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    cache: Dict[Tuple[int, int], bool],
    cell_centers: np.ndarray
) -> bool:
    """
    Agent AからAgent Bが見えるかどうかを判定する関数。
    静的壁と動的障害物の両方による遮蔽を考慮する。

    Args:
        point_a: Agent Aの2D座標 (np.ndarray)。
        point_b: Agent Bの2D座標 (np.ndarray)。
        model: MuJoCoのmj_modelオブジェクト。
        data: MuJoCoのmj_dataオブジェクト (mujoco.mj_forwardが実行済みであることを想定)。
        wall_segments: 静的な迷路壁の線分リスト。
        cache: 静的な壁に対する可視性キャッシュ。
        cell_centers: 可視性キャッシュで使用されるセル中心の配列。

    Returns:
        Agent AからAgent Bが見える場合はTrue、見えない場合はFalse。
    """

    # 1. 静的壁のチェック (キャッシュを使用)
    idx_a = find_nearest_cell_index(point_a, cell_centers)
    idx_b = find_nearest_cell_index(point_b, cell_centers)

    cache_key = tuple(sorted((idx_a, idx_b)))

    if not cache.get(cache_key, False) or not is_visible(point_a, point_b, wall_segments):
        return False

    # 2. 動的障害物の取得とソート
    target_dynamic_geoms = ["box1_geom", "box2_geom", "ramp_slope_surface"]
    dynamic_obstacles_info = get_2d_geom_segments(model, data, target_dynamic_geoms)

    for obstacle in dynamic_obstacles_info:
        obstacle['distance'] = np.linalg.norm(obstacle['center_2d'] - point_a)

    sorted_dynamic_obstacles = sorted(dynamic_obstacles_info, key=lambda x: x['distance'])

    # 3. 動的障害物のチェック
    for obstacle_info in sorted_dynamic_obstacles:
        for segment in obstacle_info['segments']:
            seg_start_np = np.array(segment[0], dtype=np.float32)
            seg_end_np = np.array(segment[1], dtype=np.float32)
            if segments_intersect(point_a, point_b, seg_start_np, seg_end_np):
                return False # 動的障害物によって視界が遮られた

    # 4. 可視性の確定
    return True

# --- テストコード ---
print("Defining function: is_agent_visible and its helper get_2d_geom_segments (with Convex Hull logic)")

# MuJoCoモデルとデータの初期化
model = mujoco.MjModel.from_xml_string(XML_CONTENT)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

if __name__ == "__main__":
    # XML_CONTENT変数が前のセルで定義されていることを想定
    xml_input_source = XML_CONTENT # XML_CONTENTはノートブックの状態から利用可能
    is_input_string = True
    output_file = "visibility_map.pkl" # デフォルトの出力ファイル名
	cache, cell_centers, metadata, wall_segments = main_build_cache(XML output_file, is_xml_string=True)

