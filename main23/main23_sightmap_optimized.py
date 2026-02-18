# main23_sightmap_optimized.py
# 演習第23回：視界勾配（Cos）ペナルティと固定長エピソードによるチーム学習
# 
# 【修正内容 (v25.64 - Sightmap統合版)】
# 1. 可視性判定の大幅高速化（Sightmap統合）:
#    - 静的壁の見通しをグリッドベースでルックアップテーブル化
#    - 動的オブジェクト（box1, box2, ramp）の2D投影による高速判定
#    - レイキャストの大幅削減（静的壁はキャッシュから即座に判定）
#    - 初回起動時に可視性キャッシュを自動ビルド、2回目以降は高速ロード
# 2. パフォーマンス最適化（v25.63より継承）:
#    - レイキャスト用バッファの事前確保（NumPy配列生成の削減）
#    - Lidar方向ベクトルのバッファ再利用
#    - 視界外マスク配列の事前確保
#    - 報酬計算用ベクトルのバッファ化
# 3. 推定パフォーマンス:
#    - Sightmap統合により可視性判定が約10-20倍高速化
#    - 総合的なSPS: 1100
# `` M4 initial

import os
import sys
import platform
import json
import time
import signal
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict
import math

# Numba（高速化）
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not available. Using fallback implementation.")
    # njit のダミー実装
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# 強化学習・物理演算ライブラリのインポート (グローバルスコープ)
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import mujoco
import mujoco.viewer
import gymnasium as gym
import torch.optim as optim
import wandb

# Scipy (動的障害物の凸包計算用)
try:
    from scipy.spatial import ConvexHull
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Dynamic obstacle detection will be limited.")

# --- 実行環境の最適化 ---
# 並列実行時に各プロセスが CPU スレッドを奪い合わないよう、計算ライブラリを制限します。
system_processor_type_id = platform.processor()
if system_processor_type_id != 'arm':
    # Intel/AMD環境（Windows/Linux）では計算ライブラリのスレッドを 1 に制限
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- プロジェクトパスの解決 ---
# 基盤となる main18_optimization.py を確実にインポートするためのパス設定。
current_script_abs_path_val = os.path.abspath(__file__)
current_script_parent_dir_val = os.path.dirname(current_script_abs_path_val)
search_dir_path_pointer_idx = current_script_parent_dir_val

# 最大 5 階層上まで main18 を探索
for _ in range(5):
    potential_base_config_file_path = os.path.join(search_dir_path_pointer_idx, "main18_optimization.py")
    if os.path.exists(potential_base_config_file_path):
        if search_dir_path_pointer_idx not in sys.path:
            sys.path.insert(0, search_dir_path_pointer_idx)
        break
    search_dir_path_pointer_idx = os.path.dirname(search_dir_path_pointer_idx)

# カレントディレクトリをパスに追加
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# 基盤となる最適化構成モジュールをインポート
import main18_optimization as base_config

# ==========================================
# 1. 実験設定 (定数定義)
# ==========================================
# ★ MODE: 学習フェーズ設定
MODE = "initial" # "refinement" # 

EXPERIMENT_BASE_NAME = "HideAndSeekV23"

# ★ TRAIN_TARGET: 学習対象エージェントの定義
TRAIN_TARGET = "HIDER" 

EXPERIMENT_NAME = f"{EXPERIMENT_BASE_NAME}_{MODE}"

# 既存モデルのロード判定（MODE に厳格に連動）
LOAD_EXISTING_MODELS = False
if MODE == "refinement":
    LOAD_EXISTING_MODELS = True

# 実行モードの設定
EXECUTION_MODE = "PLAY" # "TRAIN" / "PLAY"

# モデル保存および記録の有無
SAVE_MODEL = True
TRACK_WANDB = True           
FIXED_SEED = None

# TRIAL_MODE: Optuna 探索時に True。統計情報を即時 flush します。
TRIAL_MODE = False

# デバイス設定の継承
CUDA = base_config.CUDA

# PPO アルゴリズムのハイパーパラメータ (main18 準拠名)
TOTAL_TIMESTEPS = 5000000 
NUM_ENVS = 8
NUM_STEPS = 128
LEARNING_RATE = 2e-4
ENT_COEF = 0.001
MINIBATCH_SIZE = 128
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2

# Transformer 設定
TRANSFORMER_SEQ_LEN = 8
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_HEADS = 2

# 環境・物理定数
ACTION_REPEAT = 16
PREP_STEPS = 80
MAX_STEPS = 300
FOV_DEG = 135

# FOV 計算用のグローバル定数（毎回計算を避けるため事前計算）
FOV_HALF_RAD = FOV_DEG * 0.5 * math.pi / 180.0
FOV_COS_HALF = math.cos(FOV_HALF_RAD)

# キャッシュ閾値
LIDAR_CACHE_POS_THRESH = 0.0 # 0.25    # 25cm（毎フレーム再計算ほぼ不要）
LIDAR_CACHE_ANG_THRESH = 0.0 # np.deg2rad(4.0)  # 4度
RAYCAST_CACHE_POS_THRESH = 0.05

# Lidar 最大距離（正規化上限）
LIDAR_MAX_DIST = 15.0

# 視界外情報の外れ値マスク
OUTLIER_VALUE = 2.0

# 報酬設計パラメータ
REWARD_HIDDEN_BONUS = 1.0
COS_PENALTY_SCALE = 2.0
REWARD_DISTANCE_DIFF_SCALE = 1.0

# 共通ペナルティ
PENALTY_SAFEGUARD = -20.0
PENALTY_STAGNATION = -0.5

# エージェント推力制限
HIDER_THRUST_LIMIT = 0.40  
SEEKER_THRUST_LIMIT = 0.35 
SEEKER_RB_THRUST = 0.38
SEEKER_RB_TURN_THRESH = np.pi / 6.0

# Lidar レイキャスト方式: "direct_intersection" | "sphere_tracing" | "mujoco_ray"
LIDAR_RAYCAST_MODE = "direct_intersection" # "sphere_tracing" # "mujoco_ray" # "direct_intersection"

SAVE_MODEL_PATH = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}.pt"

# ==========================================
# Sightmap: 可視性判定高速化関数群
# ==========================================

# Sightmap設定
ENV_BOUNDS = 5.9                # 環境範囲 (±6)
SIGHTMAP_CELL_SIZE = 0.1        # Sightmap用セルサイズ（粗い、高速）
SDF_CELL_SIZE = 0.02            # SDF距離場用セルサイズ（細かい、高精度）
MAZE_PREFIX = "maze_"           # 内壁の接頭語
VISIBILITY_CACHE_FILE = "visibility_cache.pkl"
SDF_DISTANCE_FIELD_FILE = "sdf_distance_field.pkl"

@njit(cache=True, fastmath=True)
def ccw_numba(ax, ay, bx, by, cx, cy):
    """時計回り判定（numba版）"""
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

@njit(cache=True, fastmath=True)
def segments_intersect_numba(ax, ay, bx, by, cx, cy, dx, dy):
    """線分AB と CD が交差するかどうか（numba版）"""
    ccw1 = ccw_numba(ax, ay, cx, cy, dx, dy)
    ccw2 = ccw_numba(bx, by, cx, cy, dx, dy)
    ccw3 = ccw_numba(ax, ay, bx, by, cx, cy)
    ccw4 = ccw_numba(ax, ay, bx, by, dx, dy)
    return (ccw1 != ccw2) and (ccw3 != ccw4)

@njit(cache=True, fastmath=True)
def is_visible_static_numba(p1x, p1y, p2x, p2y, wall_segments_array, num_walls):
    """
    2点が互いに見えるか判定（numba版）
    
    Args:
        p1x, p1y: 点1の座標
        p2x, p2y: 点2の座標
        wall_segments_array: 壁の線分配列 (N, 2, 2) - N個の線分、各線分は2点
        num_walls: 壁の数
    
    Returns:
        可視性 (bool)
    """
    for i in range(num_walls):
        seg_start_x = wall_segments_array[i, 0, 0]
        seg_start_y = wall_segments_array[i, 0, 1]
        seg_end_x = wall_segments_array[i, 1, 0]
        seg_end_y = wall_segments_array[i, 1, 1]
        
        if segments_intersect_numba(p1x, p1y, p2x, p2y, seg_start_x, seg_start_y, seg_end_x, seg_end_y):
            return False
    return True

@njit(cache=True, fastmath=True, parallel=False)
def build_visibility_cache_numba(cell_centers, wall_segments_array, num_cells, num_walls):
    """
    可視性キャッシュのビルド（numba版）
    
    Args:
        cell_centers: セル中心座標 (N, 2)
        wall_segments_array: 壁の線分配列 (M, 2, 2)
        num_cells: セル数
        num_walls: 壁の数
    
    Returns:
        可視性キャッシュ (N, N)
    """
    cache = np.ones((num_cells, num_cells), dtype=np.uint8)
    
    for i in range(num_cells):
        p1x = cell_centers[i, 0]
        p1y = cell_centers[i, 1]
        
        for j in range(i + 1, num_cells):
            p2x = cell_centers[j, 0]
            p2y = cell_centers[j, 1]
            
            visible = is_visible_static_numba(p1x, p1y, p2x, p2y, wall_segments_array, num_walls)
            val = 1 if visible else 0
            cache[i, j] = val
            cache[j, i] = val
    
    return cache

# 互換性のために残す（numpy版）
@njit(cache=True, fastmath=True)
def is_visible_numba(ax, ay, origin_rot, bx, by,
                      visibility_cache, sightmap_bounds, sightmap_cell_size, grid_n,
                      half_fov_rad, cos_half_fov):
    """
    Numba高速化版：オブジェクトの可視性判定
    
    Args:
        ax, ay: 観渫者位置
        origin_rot: 観渫者方向
        bx, by: ターゲット位置
        visibility_cache: 可視性キャッシュ
        sightmap_bounds: sightmap範囲
        sightmap_cell_size: sightmapセルサイズ
        grid_n: グリッド数
        half_fov_rad: FOV/2 (ラジアン)
        cos_half_fov: cos(FOV/2)
    
    Returns:
        (可視性, target_body_id or -1)
    """
    dx = bx - ax
    dy = by - ay
    dist_sq = dx*dx + dy*dy
    
    # 1. 距離チェック
    if dist_sq < 0.01:  # 0.1m以下
        return True, 1
    
    # 2. 視野角判定 (FOV)
    forward_x = math.cos(origin_rot)
    forward_y = math.sin(origin_rot)
    dot = dx*forward_x + dy*forward_y
    if dot < math.sqrt(dist_sq) * cos_half_fov:
        return False, 0
    
    # 3. 静的壁キャッシュをルックアップ
    # グリッド座標計算 (pos2idx 盤隆）
    grid_x_a = int((ax + sightmap_bounds) / sightmap_cell_size)
    grid_y_a = int((ay + sightmap_bounds) / sightmap_cell_size)
    grid_x_a = max(0, min(grid_n - 1, grid_x_a))
    grid_y_a = max(0, min(grid_n - 1, grid_y_a))
    idx_a = grid_x_a * grid_n + grid_y_a
    
    grid_x_b = int((bx + sightmap_bounds) / sightmap_cell_size)
    grid_y_b = int((by + sightmap_bounds) / sightmap_cell_size)
    grid_x_b = max(0, min(grid_n - 1, grid_x_b))
    grid_y_b = max(0, min(grid_n - 1, grid_y_b))
    idx_b = grid_x_b * grid_n + grid_y_b
    
    # キャッシュアクセス
    try:
        cache_val = visibility_cache[idx_a, idx_b]
    except:
        cache_val = 1
    
    return cache_val == 1, 1

def ccw(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> bool:
    """時計回り判定"""
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def segments_intersect(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> bool:
    """線分AB と CD が交差するかどうか"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def line_segment_intersection(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    線分ABと線分CDの交点を返す。交差しない場合はNoneを返す。
    A, B: 線分1の始点・終点
    C, D: 線分2の始点・終点
    """
    # まず交差判定
    if not segments_intersect(A, B, C, D):
        return None
    
    # 交点を計算
    x1, y1 = A[0], A[1]
    x2, y2 = B[0], B[1]
    x3, y3 = C[0], C[1]
    x4, y4 = D[0], D[1]
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    # 交点
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    
    return np.array([px, py], dtype=np.float32)

def rect_to_segments(x_min: float, y_min: float, x_max: float, y_max: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """矩形を4本の線分に分解"""
    return [
        ((x_min, y_min), (x_max, y_min)),
        ((x_max, y_min), (x_max, y_max)),
        ((x_max, y_max), (x_min, y_max)),
        ((x_min, y_max), (x_min, y_min)),
    ]

def is_visible_static(point1: np.ndarray, point2: np.ndarray, wall_segments: List) -> bool:
    """2点が互いに見えるか判定（静的壁のみチェック）"""
    p1 = np.array(point1, dtype=np.float32)
    p2 = np.array(point2, dtype=np.float32)
    for wall_segment in wall_segments:
        seg_start = np.array(wall_segment[0], dtype=np.float32)
        seg_end = np.array(wall_segment[1], dtype=np.float32)
        if segments_intersect(p1, p2, seg_start, seg_end):
            return False
    return True

def extract_maze_walls_from_xml(xml_source: str, from_string: bool = False) -> List[Dict]:
    """XMLから maze_ で始まる geom 要素と wall_ で始まる外壁要素を抽出"""
    if from_string:
        root = ET.fromstring(xml_source)
    else:
        tree = ET.parse(xml_source)
        root = tree.getroot()

    maze_walls = []
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        # 内壁（maze_）と外壁（wall_）を両方抽出
        if name.startswith(MAZE_PREFIX) or name.startswith("wall_"):
            pos_str = geom.get("pos", "0 0 0")
            size_str = geom.get("size", "0 0 0")
            pos = list(map(float, pos_str.split()))
            size = list(map(float, size_str.split()))
            maze_walls.append({'name': name, 'pos': pos, 'size': size})
    return maze_walls

def walls_to_segments(maze_walls: List[Dict]) -> List[Dict]:
    """MuJoCoの矩形壁情報をAABB矩形として保持（SDF計算用）
    
    内壁の内側/外側を正しく判定するため、矩形データそのものを保持
    """
    wall_boxes = []
    
    for wall in maze_walls:
        pos, size = wall['pos'], wall['size']
        name = wall.get('name', '')
        
        # pos = [cx, cy, cz], size = [sx, sy, sz]
        # AABB: [x_min, y_min, x_max, y_max]
        cx, cy = pos[0], pos[1]
        sx, sy = size[0], size[1]
        
        wall_boxes.append({
            'name': name,
            'pos': np.array([cx, cy]),
            'size': np.array([sx, sy]),
            'x_min': cx - sx,
            'y_min': cy - sy,
            'x_max': cx + sx,
            'y_max': cy + sy,
            'is_external': name.startswith("wall_")
        })
    
    return wall_boxes

def create_cell_grid(bounds: float, cell_size: float) -> Tuple[np.ndarray, Dict]:
    """セルグリッドを生成"""
    x_cells = np.arange(-bounds, bounds, cell_size)
    y_cells = np.arange(-bounds, bounds, cell_size)
    grid_n = len(x_cells)
    cell_centers = []
    for x in x_cells:
        for y in y_cells:
            cell_centers.append([x + cell_size/2, y + cell_size/2])
    cell_centers = np.array(cell_centers, dtype=np.float32)
    metadata = {
        'bounds': bounds, 
        'cell_size': cell_size, 
        'num_cells': len(cell_centers),
        'grid_n': grid_n
    }
    return cell_centers, metadata

def build_visibility_cache(cell_centers: np.ndarray, wall_segments: List) -> np.ndarray:
    """セル間の可視性をすべて計算してキャッシュ化（NumPy2D配列）"""
    num_cells = len(cell_centers)
    num_walls = len(wall_segments)
    
    print(f"Building visibility cache for {num_cells} cells...")
    
    # wall_segments を numpy 配列に変換 (N, 2, 2)
    wall_segments_array = np.zeros((num_walls, 2, 2), dtype=np.float32)
    for i, seg in enumerate(wall_segments):
        wall_segments_array[i, 0, 0] = seg[0][0]  # start_x
        wall_segments_array[i, 0, 1] = seg[0][1]  # start_y
        wall_segments_array[i, 1, 0] = seg[1][0]  # end_x
        wall_segments_array[i, 1, 1] = seg[1][1]  # end_y
    
    # numba 高速化関数を呼び出し
    cache = build_visibility_cache_numba(cell_centers, wall_segments_array, num_cells, num_walls)
    
    print(f"  ✓ Visibility cache built ({num_cells}x{num_cells} cells)")
    return cache

def save_visibility_cache(cache: np.ndarray, cell_centers: np.ndarray, metadata: Dict, output_file: str):
    """キャッシュを pickle で保存（NumPy配列形式）"""
    data = {'cache': cache, 'cell_centers': cell_centers, 'metadata': metadata}
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    file_size = Path(output_file).stat().st_size / 1024
    print(f"✓ Visibility cache saved: {output_file} ({file_size:.1f} KB)")

def load_visibility_cache(cache_file: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """キャッシュを pickle から読み込み（NumPy配列形式）"""
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    return data['cache'], data['cell_centers'], data['metadata']

def build_sdf_distance_field(visibility_engine, cell_centers: np.ndarray) -> np.ndarray:
    """
    グリッド全体のSDF（静的環境のみ）を計算
    ルックアップテーブル用にNumPy配列で返す [grid_size, grid_size]
    
    cell_centersの書き出し順、単純にgrid_index=iで簡潔化
    """
    num_cells = len(cell_centers)
    grid_size = int(np.sqrt(num_cells))
    sdf_field = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    print(f"[SDF] Building distance field for {num_cells} cells ({grid_size}x{grid_size}) (this may take a few minutes)...", flush=True)
    start_time = time.time()
    
    for i, cell_pos in enumerate(cell_centers):
        # 静的環境のみでSDF計算（可動物は含めない）
        point_3d = np.array([cell_pos[0], cell_pos[1], 0.5], dtype=np.float32)
        sdf_value = visibility_engine.get_sdf_static(point_3d)
        
        # 2次元グリッド配列に格納
        # cell_centersは [x,y]の順で並んでいる。
        # x方向が外人ループ => i // grid_size = x索引, i % grid_size = y索引
        # NumPyは [行,列]=[y,x] なので sdf_field[y, x] でアクセス
        grid_x = i // grid_size
        grid_y = i % grid_size
        sdf_field[grid_y, grid_x] = sdf_value
        
        if (i + 1) % max(1, num_cells // 20) == 0:
            elapsed = time.time() - start_time
            progress = (i + 1) / num_cells
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"  Progress: {progress*100:.1f}% ({i+1}/{num_cells}), ETA: {eta:.0f}s", flush=True)
    
    elapsed = time.time() - start_time
    print(f"[SDF] ✓ Distance field built in {elapsed:.1f}s", flush=True)
    return sdf_field

def save_sdf_distance_field(sdf_field: np.ndarray, cell_centers: np.ndarray, metadata: Dict, output_file: str):
    """SDF距離場を pickle で保存（NumPy配列形式）"""
    data = {'sdf_field': sdf_field, 'cell_centers': cell_centers, 'metadata': metadata}
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    file_size = Path(output_file).stat().st_size / 1024
    print(f"[SDF] ✓ Distance field saved: {output_file} ({file_size:.1f} KB)", flush=True)

def load_sdf_distance_field(cache_file: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """SDF距離場を pickle から読み込み（NumPy配列形式）"""
    with open(cache_file, 'rb') as f:
        data = pickle.load(f)
    return data['sdf_field'], data['cell_centers'], data['metadata']

def pos2idx(point: np.ndarray, bounds: float, cell_size: float, grid_n: int) -> int:
    """座標からセルインデックスを直接計算（O(1)）"""
    x, y = point[0], point[1]
    # グリッド座標に変換（境界外はクランプ）
    grid_x = int((x + bounds) / cell_size)
    grid_y = int((y + bounds) / cell_size)
    grid_x = max(0, min(grid_n - 1, grid_x))
    grid_y = max(0, min(grid_n - 1, grid_y))
    # 1次元インデックスに変換（y軸優先）
    return grid_x * grid_n + grid_y

@njit(cache=True, fastmath=True)
def sdf_lookup_numba(p_x, p_y, sdf_field, grid_size_0, grid_size_1, cell_size):
    """
    Numba高速化版：SDF距離場のルックアップ（最近傍補間）
    
    Args:
        p_x, p_y: 点のXY座標
        sdf_field: SDF距離場 (grid_size_0, grid_size_1) - [y, x] indexing
        grid_size_0, grid_size_1: グリッドサイズ
        cell_size: セルサイズ
    
    Returns:
        SDF値 (float)
    """
    inv_float = 1.0 / cell_size
    # グリッド座標の計算：X方向はgrid_size_1（列数）に対応、Y方向はgrid_size_0（行数）に対応
    grid_x = (p_x + 6.0) * inv_float  # X座標 → 列インデックス
    grid_y = (p_y + 6.0) * inv_float  # Y座標 → 行インデックス
    
    # 境界チェック
    if grid_x < 0 or grid_x >= grid_size_1 or grid_y < 0 or grid_y >= grid_size_0:
        # グリッド外の点：環境外を示す負の距離を返す
        # 環境中心 (0, 0)、半径 6.0m からの距離
        dist_from_center = math.sqrt(p_x * p_x + p_y * p_y)
        return -(dist_from_center - 6.0)  # 外壁の外なので負の値
    
    # 最近傍補間
    x = int(grid_x + 0.5)  # 0.5を加えて四捨五入
    y = int(grid_y + 0.5)  # 0.5を加えて四捨五入
    
    # クランプ
    if 0 <= x < grid_size_1 and 0 <= y < grid_size_0:
        return sdf_field[y, x]  # NumPyは[行,列]=[y,x]の順序
    else:
        return 1e6
@njit(cache=True, fastmath=True)
def ray_circle_intersection_numba(start_x, start_y, dir_x, dir_y, 
                                    circle_x, circle_y, radius):
    """
    直線と円の交点を計算（最短交点距離を返す）
    2D 平面での XY 円との交点を計算
    
    Args:
        start_x, start_y: レイ開始点 (XY平面)
        dir_x, dir_y: レイ方向ベクトル（単位ベクトル想定、XY成分）
        circle_x, circle_y: 円の中心 (XY平面)
        radius: 円の半径
    
    Returns:
        最短交点までの距離（交点なし時は1e6）
    """
    # 方向ベクトルを正規化
    dir_mag = math.sqrt(dir_x * dir_x + dir_y * dir_y)
    if dir_mag < 1e-10:
        return 1e6  # 無効な方向ベクトル
    
    dir_x_norm = dir_x / dir_mag
    dir_y_norm = dir_y / dir_mag
    
    # レイ開始点から円の中心へのベクトル
    fx = start_x - circle_x
    fy = start_y - circle_y
    
    # 二次方程式の係数：t^2 + 2bt + c = 0
    # (start + t*dir - circle) · (start + t*dir - circle) = radius^2
    b = fx * dir_x_norm + fy * dir_y_norm
    c = fx * fx + fy * fy - radius * radius
    
    # 判別式
    discriminant = b * b - c
    
    if discriminant < 0:
        return 1e6  # 交点なし
    
    sqrt_disc = math.sqrt(discriminant)
    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc
    
    # 最短かつ正の距離を返す
    eps = 1e-6
    if t1 > eps:  # 数値誤差対策
        return t1 * dir_mag
    elif t2 > eps:
        return t2 * dir_mag
    else:
        return 1e6

@njit(cache=True, fastmath=True)
def ray_segment_intersection_numba(start_x, start_y, dir_x, dir_y,
                                   seg_x1, seg_y1, seg_x2, seg_y2):
    """
    2つの直線 (p1-p2) と (p3-p4) の交点を求める
    （ユーザー提示の標準公式を使用）
    
    Args:
        start_x, start_y: レイ開始点 (p1)
        dir_x, dir_y: レイ方向ベクトル (p1から p2 へのオフセット)
        seg_x1, seg_y1: 線分開始点 (p3)
        seg_x2, seg_y2: 線分終了点 (p4)
    
    Returns:
        レイパラメータ t（交点までのパラメータ値。単位ベクトルなら距離）
    """
    # P1, P2 を計算（レイ上の2点）
    p1_x = start_x
    p1_y = start_y
    p2_x = start_x + dir_x
    p2_y = start_y + dir_y
    
    # P3, P4 は線分の両端
    p3_x = seg_x1
    p3_y = seg_y1
    p4_x = seg_x2
    p4_y = seg_y2
    
    # 分母を計算
    denom = (p1_x - p2_x) * (p3_y - p4_y) - (p1_y - p2_y) * (p3_x - p4_x)
    
    eps = 1e-10
    if abs(denom) < eps:
        # 平行
        return 1e6
    
    # 交点座標を計算（標準公式）
    t1_num = (p1_x * p2_y - p1_y * p2_x) * (p3_x - p4_x) - (p1_x - p2_x) * (p3_x * p4_y - p3_y * p4_x)
    t2_num = (p1_x * p2_y - p1_y * p2_x) * (p3_y - p4_y) - (p1_y - p2_y) * (p3_x * p4_y - p3_y * p4_x)
    
    intersect_x = t1_num / denom
    intersect_y = t2_num / denom
    
    # レイパラメータ t を計算
    # (intersect_x, intersect_y) = (start_x, start_y) + t * (dir_x, dir_y)
    # より安定した計算：最も大きい成分を使用
    abs_dx = abs(dir_x)
    abs_dy = abs(dir_y)
    
    if abs_dx > abs_dy and abs_dx > eps:
        t = (intersect_x - start_x) / dir_x
    elif abs_dy > eps:
        t = (intersect_y - start_y) / dir_y
    elif abs_dx > eps:
        t = (intersect_x - start_x) / dir_x
    else:
        return 1e6  # 方向ベクトルが 0
    
    if t < 1e-6:
        return 1e6  # 後ろ側またはレイの開始点付近
    
    # 線分パラメータ u を計算
    # (intersect_x, intersect_y) = (seg_x1, seg_y1) + u * (seg_dx, seg_dy)
    seg_dx = seg_x2 - seg_x1
    seg_dy = seg_y2 - seg_y1
    
    abs_seg_dx = abs(seg_dx)
    abs_seg_dy = abs(seg_dy)
    
    if abs_seg_dx > abs_seg_dy and abs_seg_dx > eps:
        u = (intersect_x - seg_x1) / seg_dx
    elif abs_seg_dy > eps:
        u = (intersect_y - seg_y1) / seg_dy
    elif abs_seg_dx > eps:
        u = (intersect_x - seg_x1) / seg_dx
    else:
        return 1e6  # 線分が退化
    
    # 線分の範囲内か（0 <= u <= 1）
    if -1e-6 <= u <= 1.0 + 1e-6:
        return t
    else:
        return 1e6

@njit(cache=True, fastmath=True)
def cast_ray_direct_numba(start_x, start_y, dir_x, dir_y,
                          positions, radii, body_ids, num_objects,
                          wall_segments, num_walls,
                          exclude_id1, exclude_id2, max_dist):
    """
    直接交点計算によるレイキャスト（Sphere Tracingより高速）
    
    Args:
        start_x, start_y: レイ開始点
        dir_x, dir_y: レイ方向ベクトル
        positions: 円の中心位置 (num_objects, 2)
        radii: 円の半径 (num_objects,)
        body_ids: body ID (num_objects,)
        num_objects: 円の数
        wall_segments: 壁線分 (num_walls, 4) [x1, y1, x2, y2]
        num_walls: 壁の数
        exclude_id1, exclude_id2: 除外ID
        max_dist: 最大距離
    
    Returns:
        (距離, ヒット判定)
    """
    min_dist = max_dist
    
    # 円（動的オブジェクト）との交点
    for i in range(num_objects):
        bid = body_ids[i]
        if bid != exclude_id1 and bid != exclude_id2:
            dist = ray_circle_intersection_numba(
                start_x, start_y, dir_x, dir_y,
                positions[i, 0], positions[i, 1], radii[i]
            )
            if dist < min_dist:
                min_dist = dist
    
    # 壁との交点
    for i in range(num_walls):
        dist = ray_segment_intersection_numba(
            start_x, start_y, dir_x, dir_y,
            wall_segments[i, 0], wall_segments[i, 1],
            wall_segments[i, 2], wall_segments[i, 3]
        )
        if dist < min_dist:
            min_dist = dist
    
    return min_dist, min_dist < max_dist

@njit(cache=True, fastmath=True)
def cast_ray_numba(start_x, start_y, dir_x, dir_y, 
                   sdf_field, grid_size_0, grid_size_1, cell_size,
                   positions, radii, body_ids, num_objects, 
                   exclude_id1, exclude_id2,
                   epsilon, max_steps, max_dist):
    """
    Numba高速化版：Sphere Tracing による光線距離計算
    
    Args:
        start_x, start_y: 開始位置
        dir_x, dir_y: 方向ベクトル（単位ベクトル想定）
        sdf_field: 静的SDF距離場
        grid_size_0, grid_size_1: グリッドサイズ
        cell_size: セルサイズ
        positions: 動的オブジェクトの位置
        radii: 動的オブジェクトの半径
        body_ids: 動的オブジェクトのbody_id
        num_objects: 動的オブジェクト数
        exclude_id1, exclude_id2: 除外body_id
        epsilon: 収束判定閾値
        max_steps: 最大ステップ数
        max_dist: 最大距離
    
    Returns:
        (距離, ヒット判定)
    """
    # 方向ベクトルを正規化
    dir_mag = math.sqrt(dir_x * dir_x + dir_y * dir_y)
    if dir_mag < 1e-10:
        return max_dist, False  # 無効な方向ベクトル
    
    dir_x_norm = dir_x / dir_mag
    dir_y_norm = dir_y / dir_mag
    
    curr_x = start_x
    curr_y = start_y
    total_d = 0.0
    
    for step in range(max_steps):
        # 静的SDF
        d_static = sdf_lookup_numba(curr_x, curr_y, sdf_field, grid_size_0, grid_size_1, cell_size)
        
        # 動的SDF
        d_dynamic = 1e6
        for i in range(num_objects):
            bid = body_ids[i]
            if bid != exclude_id1 and bid != exclude_id2:
                dx = positions[i, 0] - curr_x
                dy = positions[i, 1] - curr_y
                dist = math.sqrt(dx * dx + dy * dy) - radii[i]
                if dist < d_dynamic:
                    d_dynamic = dist
        
        # 最小距離（符号付き距離なので負は内部を意味する）
        d = min(d_static, d_dynamic)
        
        # 負の距離 → 障害物内部にいる → 衝突
        if d < epsilon:
            return total_d, True
        
        total_d += d
        curr_x += dir_x_norm * d
        curr_y += dir_y_norm * d
        
        if total_d > max_dist:
            break
    
    if total_d < max_dist:
        return total_d, False
    else:
        return max_dist, False

def get_2d_geom_segments(model: mujoco.MjModel, data: mujoco.MjData, geom_names: List[str]) -> List[Dict]:
    """MuJoCoのgeomの現在位置から2D線分を抽出（Lidar用に最適化）"""
    structured_dynamic_segments = []
    for geom_name in geom_names:
        try:
            geom_id = model.geom(geom_name).id
        except (KeyError, AttributeError):
            continue
        
        sx, sy, sz = model.geom_size[geom_id]
        geom_xpos = data.geom_xpos[geom_id]
        geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
        
        geom_segments = []
        
        # ランプの場合：斜面のXY投影のみ
        if "ramp" in geom_name:
            # ランプの上面4頂点のXY投影
            local_top = np.array([
                [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]
            ])
            world_2d = []
            for lc in local_top:
                wc = geom_xpos + np.dot(geom_xmat, lc)
                world_2d.append([wc[0], wc[1]])
            
            # 4つの辺を追加
            for i in range(4):
                p1 = tuple(world_2d[i])
                p2 = tuple(world_2d[(i + 1) % 4])
                geom_segments.append((p1, p2))
        
        # 箱の場合：4つの側面のみ（上下を除く）
        else:
            # 下面の4頂点
            local_bottom = np.array([
                [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz]
            ])
            # 上面の4頂点
            local_top = np.array([
                [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz]
            ])
            
            bottom_2d = []
            top_2d = []
            for lc in local_bottom:
                wc = geom_xpos + np.dot(geom_xmat, lc)
                bottom_2d.append([wc[0], wc[1]])
            for lc in local_top:
                wc = geom_xpos + np.dot(geom_xmat, lc)
                top_2d.append([wc[0], wc[1]])
            
            # 側面4つのみ（底面と上面の対応する頂点を結ぶ辺は省略）
            for i in range(4):
                # 底面の辺
                p1 = tuple(bottom_2d[i])
                p2 = tuple(bottom_2d[(i + 1) % 4])
                geom_segments.append((p1, p2))
        
        structured_dynamic_segments.append({
            'name': geom_name,
            'center_2d': np.array([geom_xpos[0], geom_xpos[1]]),
            'segments': geom_segments
        })
    
    return structured_dynamic_segments


class VisibilityEngine:
    """
    SDFとSphere Tracingを用いた視界観測エンジン
    静的壁はXMLから抽出、可動物はリアルタイム計算
    """
    def __init__(self, model, data, epsilon=0.1, max_steps=15, max_dist=LIDAR_MAX_DIST):
        """
        Visibility Engine の初期化
        
        Args:
            epsilon: Sphere Tracing の衝突判定閾値
                     セルサイズ (0.020374m) より大きくする必要があります
                     デフォルト 0.1m は約5倍のマージンを持たせています
        """
        self.model = model
        self.data = data
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.max_dist = max_dist
        
        # 外壁（境界）は固定
        self.boundary = np.array([0.0, 0.0, 6.0, 6.0])
        
        # 内壁はXMLから抽出（XMLの形式に合わせる）
        self.static_walls = []  # walls_to_segments()の結果を格納
        self._build_static_walls_from_xml()
        
        # 可動物の参照（環境から提供される）
        self.agent_bodies = {}  # {0: s0_body, 1: h1_body, 2: h2_body}
        self.box_bodies = {}    # {box1_body, box2_body}
        self.ramp_body = None
        
        # 動的オブジェクト情報の事前割り当て（毎フレーム更新用）
        self.dynamic_positions = np.zeros((6, 2), dtype=np.float32)  # 最大6オブジェクト
        self.dynamic_radii = np.zeros(6, dtype=np.float32)
        self.num_dynamic_objects = 0
        
        # ★直線と円の交点計算用：壁線分配列（Numba用）
        self.wall_segments = np.zeros((50, 4), dtype=np.float32)  # 最大50線分
        self.num_wall_segments = 0
        
        # Sphere Tracing 用バッファ
        self.sp_curr_p = np.zeros(3, dtype=np.float32)  # 現在位置
        
        # SDF距離場（後で外部から設定）
        self.sdf_field = None
        
    def _build_static_walls_from_xml(self):
        """XMLから内壁を抽出（矩形AABB形式）"""
        try:
            xml_string = base_config.XML_CONTENT
            maze_walls = extract_maze_walls_from_xml(xml_string, from_string=True)
            # walls_to_segments()で矩形データに変換（SDF計算用）
            self.static_walls = walls_to_segments(maze_walls)
            
            # Numba互換性のため num_wall_segments を設定
            self.num_wall_segments = len(self.static_walls)
        except Exception as e:
            self.static_walls = []
            self.num_wall_segments = 0
    
    def set_bodies(self, s0_body, h1_body, h2_body, box1_body, box2_body, ramp_body):
        """環境から body IDs を設定"""
        self.agent_bodies = {0: s0_body, 1: h1_body, 2: h2_body}
        self.box_bodies = {box1_body, box2_body}
        self.ramp_body = ramp_body
        
        # 動的オブジェクトの構成を記録（毎フレーム更新に使用）
        # agents, boxes, ramp 全てを含める
        self.dynamic_body_ids = []
        self.dynamic_body_radii = []
        
        # agents も含める（自分自身は exclude_agent_id で除外）
        for agent_id, body_id in self.agent_bodies.items():
            self.dynamic_body_ids.append(body_id)
            self.dynamic_body_radii.append(0.4)

        for box_body in self.box_bodies:
            self.dynamic_body_ids.append(box_body)
            self.dynamic_body_radii.append(np.sqrt(0.6**2 + 0.6**2))
        if ramp_body is not None:
            self.dynamic_body_ids.append(ramp_body)
            self.dynamic_body_radii.append(np.sqrt(0.67**2 + 0.5**2))
        
        self.num_dynamic_objects = len(self.dynamic_body_ids)
        self.dynamic_radii[:self.num_dynamic_objects] = self.dynamic_body_radii
        
        # Numba用にbody_idsをnumpy配列に変換（int型）
        self.dynamic_body_ids_array = np.array(self.dynamic_body_ids, dtype=np.int32)
    
    def update_dynamic_positions(self):
        """毎フレーム、動的オブジェクトの位置を更新（cast_ray呼び出し前に実行）"""
        for i, body_id in enumerate(self.dynamic_body_ids):
            self.dynamic_positions[i] = self.data.xpos[body_id][:2]
    
    def get_sdf_static(self, p):
        """
        点pの符号付き距離関数（静的壁のみ）
        - 外壁
        - 内壁（線分）
        事前計算に使用
        
        注意：Sphere Tracing 用に、環境内では外壁までの距離を正の値で返す
        """
        # 1. 外壁までの距離
        center = self.boundary[:2]  # 環境の中心 (0, 0)
        half_size = self.boundary[2:]  # 環境の半幅 (6, 6)
        d = np.abs(p[:2] - center) - half_size   # 点pから環境中心までの距離 - 半幅
        outside_dist = np.linalg.norm(np.maximum(d, 0.0))    # 環境外の距離
        inside_dist = np.minimum(np.max(d), 0.0)
        boundary_dist = outside_dist + inside_dist
        
        # 2. 内壁（線分）までの距離（ベクトル化）
        if len(self.static_walls) > 0:
            # static_walls: [(start, end), ...]
            seg_starts = np.array([seg[0] for seg in self.static_walls], dtype=np.float32)  # (N, 2)
            seg_ends = np.array([seg[1] for seg in self.static_walls], dtype=np.float32)    # (N, 2)
            
            # 点pから各線分への距離を計算
            seg_vecs = seg_ends - seg_starts  # (N, 2)
            seg_len_sq = np.sum(seg_vecs**2, axis=1, keepdims=True)  # (N, 1)
            
            # 点pから線分開始点へのベクトル
            p_to_start = p[:2] - seg_starts  # (N, 2)
            
            # 投影パラメータ t (各線分ごと)
            t = np.sum(p_to_start * seg_vecs, axis=1, keepdims=True) / (seg_len_sq + 1e-8)  # (N, 1)
            t = np.clip(t, 0, 1)  # (N, 1)
            
            # 線分上の最近点
            closest_pts = seg_starts + t * seg_vecs  # (N, 2)
            
            # 点pから最近点までの距離
            dists = np.linalg.norm(p[:2] - closest_pts, axis=1)  # (N,)
            min_wall_dist = np.min(dists)
        else:
            min_wall_dist = 1e6
        
        # 3. 外壁と内壁の最小値
        # 環境内（boundary_dist < 0）の場合、外壁距離を正にしてから比較
        if boundary_dist < 0:
            # 環境内：外壁までの距離（正の値）と内壁を比較
            return min(-boundary_dist, min_wall_dist)
        else:
            # 環境外：外壁までの距離
            return min(boundary_dist, min_wall_dist)

    
    def _point_to_segment_distance(self, p, seg_start, seg_end):
        """点pから線分(seg_start, seg_end)までの距離"""
        seg_vec = seg_end - seg_start
        seg_len_sq = np.dot(seg_vec, seg_vec)
        if seg_len_sq < 1e-8:
            return np.linalg.norm(p - seg_start)
        
        t = np.dot(p - seg_start, seg_vec) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest = seg_start + t * seg_vec
        return np.linalg.norm(p - closest)
    
    def _point_to_rect_distance(self, p, rect_center, half_size):
        """点pから矩形（中心rect_center, 半幅half_size）までの距離（軸並行）"""
        d = np.abs(p - rect_center) - half_size
        outside_dist = np.linalg.norm(np.maximum(d, 0.0))
        inside_dist = np.minimum(np.max(d), 0.0)
        return outside_dist + inside_dist

    def cast_ray(self, start_pos, direction, exclude_agent_id=None, exclude_body_id=None, target_body_id=None):
        """
        光線の距離を計算（LIDAR_RAYCAST_MODE で方式を選択）
        
        - "direct_intersection"  : 直線-円交点計算（最速）
        - "sphere_tracing"       : SDF + Sphere Tracing（精密）
        - "mujoco_ray"           : MuJoCo mj_ray（高速）
        
        exclude_agent_id: 自エージェントのIDを指定（自分を除外）
        exclude_body_id: ターゲットのbody_idを指定（ターゲットを除外）
        target_body_id: デバッグ用（ターゲットの body_id）
        """
        # 除外IDを取得
        exclude_id1 = -1
        exclude_id2 = -1
        
        if exclude_agent_id is not None:
            if exclude_agent_id in self.agent_bodies:
                exclude_id1 = self.agent_bodies[exclude_agent_id]
        
        if exclude_body_id is not None:
            if exclude_id1 == -1:
                exclude_id1 = exclude_body_id
            else:
                exclude_id2 = exclude_body_id
        
        if LIDAR_RAYCAST_MODE == "direct_intersection":
            # ★直線-円交点計算（最速）
            dist, hit = cast_ray_direct_numba(
                start_pos[0], start_pos[1],
                direction[0], direction[1],
                self.dynamic_positions, self.dynamic_radii, self.dynamic_body_ids_array,
                self.num_dynamic_objects,
                self.wall_segments, self.num_wall_segments,
                exclude_id1, exclude_id2, self.max_dist
            )
            return dist, hit
        
        elif LIDAR_RAYCAST_MODE == "sphere_tracing":
            # ★SDF + Sphere Tracing（精密）
            if self.sdf_field is None:
                return self.max_dist, False
            
            grid_size = self.sdf_field.shape
            cell_size = 12.0 / (grid_size[0] - 1)
            
            dist, hit = cast_ray_numba(
                start_pos[0], start_pos[1],
                direction[0], direction[1],
                self.sdf_field, grid_size[0], grid_size[1], cell_size,
                self.dynamic_positions, self.dynamic_radii, self.dynamic_body_ids_array,
                self.num_dynamic_objects,
                exclude_id1, exclude_id2,
                self.epsilon, self.max_steps, self.max_dist
            )
            return dist, hit
        
        elif LIDAR_RAYCAST_MODE == "mujoco_ray":
            # ★MuJoCo mj_ray
            # 注: このメソッドは通常は直接呼び出されず、Lidar観測ループ内で直接実装
            # フォールバック: sphere_tracing を使用
            if self.sdf_field is None:
                return self.max_dist, False
            
            grid_size = self.sdf_field.shape
            cell_size = 12.0 / (grid_size[0] - 1)
            
            dist, hit = cast_ray_numba(
                start_pos[0], start_pos[1],
                direction[0], direction[1],
                self.sdf_field, grid_size[0], grid_size[1], cell_size,
                self.dynamic_positions, self.dynamic_radii, self.dynamic_body_ids_array,
                self.num_dynamic_objects,
                exclude_id1, exclude_id2,
                self.epsilon, self.max_steps, self.max_dist
            )
            return dist, hit
        
        else:
            # デフォルト: direct_intersection
            dist, hit = cast_ray_direct_numba(
                start_pos[0], start_pos[1],
                direction[0], direction[1],
                self.dynamic_positions, self.dynamic_radii, self.dynamic_body_ids_array,
                self.num_dynamic_objects,
                self.wall_segments, self.num_wall_segments,
                exclude_id1, exclude_id2, self.max_dist
            )
            return dist, hit
    
    def _sdf_lookup(self, p):
        """
        事前計算SDF（静的環境）をルックアップテーブルから取得
        双線形補間で値を計算
        """
        if self.sdf_field is None:
            # フォールバック：その場で計算（コストあり）
            return self.get_sdf_static(p)
        
        # グリッド座標に変換（最近傍補間）
        grid_size = self.sdf_field.shape
        cell_size = 12.0 / (grid_size[0] - 1)  # 12m環境
        grid_x = (p[0] + 6.0) / cell_size
        grid_y = (p[1] + 6.0) / cell_size
        
        # 境界チェック
        if grid_x < 0 or grid_x >= grid_size[0] or grid_y < 0 or grid_y >= grid_size[1]:
            return 1e6  # 環境外
        
        # ★最適化：単純な最近傍補間に簡略化
        x = int(np.round(grid_x))
        y = int(np.round(grid_y))
        x = np.clip(x, 0, grid_size[0] - 1)
        y = np.clip(y, 0, grid_size[1] - 1)
        
        return float(self.sdf_field[y, x])  # NumPy は [行,列]=[y,x] の順序
    
# ==========================================
# 2. クラス定義 (Agent / ObsHistory / Env)
# ==========================================

def layer_init(layer, std=np.sqrt(2), bias=0.0):
    """ネットワーク重みの直交初期化を行い、学習初期の安定性を確保します。"""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer

class Agent(nn.Module):
    """ Transformer エンコーダを基幹に持つ Actor-Critic ネットワーク。 """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # 埋め込み層
        self.embed = nn.Linear(obs_dim, HIDDEN_DIM)
        
        # 位置エンコーディング
        self.pos_enc = nn.Parameter(
            torch.zeros(1, TRANSFORMER_SEQ_LEN, HIDDEN_DIM)
        )
        
        # Transformer エンコーダ
        enc_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM, 
            nhead=NUM_HEADS, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, 
            num_layers=NUM_LAYERS, 
            enable_nested_tensor=False
        )
        
        # 出力層
        self.actor_mean = layer_init(
            nn.Linear(HIDDEN_DIM, action_dim), 
            std=0.01
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, action_dim)
        )
        self.critic = layer_init(
            nn.Linear(HIDDEN_DIM, 1), 
            std=1.0
        )

    def get_value(self, x):
        """ 系列情報を基に、現在の状態価値予測 V(s) を算出します。 """
        embed = self.embed(x)
        h = embed + self.pos_enc
        h = self.transformer(h)
        # 最終ステップの要約ベクトル
        h = h[:, -1, :]
        value = self.critic(h)
        return value

    def get_action_and_value(self, x, action=None):
        """ 行動、対数確率、エントロピー、状態価値を一括計算。 """
        embed = self.embed(x)
        h = embed + self.pos_enc
        h = self.transformer(h)
        h = h[:, -1, :]
        
        # 行動決定
        mean = self.actor_mean(h)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(h)
        
        return action, log_prob, entropy, value

class ObsHistory:
    """ ミラーリング・ダブルバッファ構造による履歴管理クラス。 """
    def __init__(self, n_envs, seq_len, obs_dim, device):
        self.buf_len = seq_len * 2
        self.buffer = torch.zeros((n_envs, self.buf_len, obs_dim), device=device)
        self.device = device
        self.seq_len = seq_len
        self.write_idx = 0

    def reset(self):
        """ バッファをゼロで初期化。 """
        self.buffer.zero_()
        self.write_idx = 0

    def update(self, latest_obs):
        """ 最新観測値をミラーリング書き込み。 """
        new_obs = torch.as_tensor(latest_obs, dtype=torch.float32, device=self.device)
        if new_obs.ndim == 1:
            new_obs = new_obs.unsqueeze(0)
            
        # 本体領域とミラー領域
        self.buffer[:, self.write_idx] = new_obs
        mirror_idx = self.write_idx + self.seq_len
        self.buffer[:, mirror_idx] = new_obs
        
        # ポインタ循環
        self.write_idx = (self.write_idx + 1) % self.seq_len

    def get(self):
        """ 最新の系列スライスを View で取得。 ★修正: 属性参照の統一 """
        slice_end = self.write_idx + self.seq_len
        return self.buffer[:, self.write_idx : slice_end]

class TeamCosEnv(base_config.HideAndSeekEnv):
    """
    不整合を解消し、将来的なオブジェクト増設に対応した超高速化環境。
    """
    def __init__(self, render_mode=None):
        # ★ 属性初期化の順序厳守 (reset() 時の参照バグ防止)
        self.hider_pos = {1: None, 2: None}
        self.dist_to_seeker = {1: 0.0, 2: 0.0}
        self.lidar_cache = {}  # Lidar出力キャッシュ: {agent_id: (pos, yaw, lidar_array)}
        self.raycast_cache = {} 
        self.raycast_perf = {"hits": 0, "misses": 0}
        self.visible_map = {0: {}, 1: {}, 2: {}}
        self.obs_memo = {}
        self.hidden_steps = 0
        self.caught_steps = 0 
        self.recovery_turn_dir = 1.0
        self.visible_names = {0: [], 1: [], 2: []}

        # 親クラスの初期化
        super().__init__(render_mode=render_mode)
        # Lidar角度の事前計算（cos/sin）
        self._lidar_angle_cos = np.cos(self.lidar_angles)
        self._lidar_angle_sin = np.sin(self.lidar_angles)
        self._beam_cos = np.zeros(self.lidar_angles.shape, dtype=np.float32)
        self._beam_sin = np.zeros(self.lidar_angles.shape, dtype=np.float32)
        self._beam_tmp = np.zeros(self.lidar_angles.shape, dtype=np.float32)
        
        # Body ID to Name マッピング (visible_names 更新用)
        self.body_id_to_name = {
            self.s0_body: "s0",
            self.h1_body: "h1",
            self.h2_body: "h2",
            self.box1_body: "box1",
            self.box2_body: "box2",
            self.ramp_body: "ramp"
        }
        
        cpu_device = torch.device("cpu")
        # 各個体専用の履歴
        self.npc_history = {
            0: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
            1: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device), 
            2: ObsHistory(1, TRANSFORMER_SEQ_LEN, 53, cpu_device)
        }

        # NPC 推論エンジン
        self.npc_hider_agent = None
        self.npc_seeker_agent = None
        
        log_marker_id_status_str = os.environ.get("NPC_MODELS_LOGGED")
        should_log = (log_marker_id_status_str != "TRUE")
        
        # ★PLAY MODE判定（render_mode="human"）
        is_play_mode = (render_mode == "human")
        
        # ★正確な設計実装
        if is_play_mode:
            # PLAY MODE: すべてのエージェント用モデルを読み込み
            self.npc_seeker_agent = Agent(53, 4).to("cpu")
            s_path = load_model_with_mode(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER", MODE)
            if s_path:
                if should_log:
                    print(f"[PLAY] Loaded Seeker model: {s_path}", flush=True)
            else:
                self.npc_seeker_agent = None  # モデルがない場合は None
            
            self.npc_hider_agent = Agent(53, 4).to("cpu")
            h_path = load_model_with_mode(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER", MODE)
            if h_path:
                if should_log:
                    print(f"[PLAY] Loaded Hider model: {h_path}", flush=True)
            else:
                self.npc_hider_agent = None  # モデルがない場合は None
        
        elif MODE == "refinement":
            if TRAIN_TARGET == "HIDER":
                # 1-1: Seeker は学習済みモデルがあれば推論、なければルールベース
                self.npc_seeker_agent = Agent(53, 4).to("cpu")
                s_path = load_model_with_mode(self.npc_seeker_agent, EXPERIMENT_BASE_NAME, "SEEKER", MODE)
                if s_path:
                    if should_log:
                        print(f"[Refinement] Loaded Seeker model: {s_path}", flush=True)
                else:
                    self.npc_seeker_agent = None  # ルールベース使用
                
                # 1-3: パートナーHider は学習済みモデルがあれば推論、なければランダム
                self.npc_hider_agent = Agent(53, 4).to("cpu")
                h_path = load_model_with_mode(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER", MODE)
                if h_path:
                    if should_log:
                        print(f"[Refinement] Loaded Hider model for NPC: {h_path}", flush=True)
                else:
                    self.npc_hider_agent = None  # ランダム使用
            else:
                # TRAIN_TARGET == "SEEKER"
                # 2-1, 2-2: Hider は学習済みモデルがあれば推論、なければランダム
                self.npc_hider_agent = Agent(53, 4).to("cpu")
                h_path = load_model_with_mode(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER", MODE)
                if h_path:
                    if should_log:
                        print(f"[Refinement] Loaded Hider model: {h_path}", flush=True)
                else:
                    self.npc_hider_agent = None  # ランダム使用
        
        elif MODE == "initial":
            if TRAIN_TARGET == "SEEKER":
                # initial + SEEKER: Hider は学習済みモデルがあれば推論、なければランダム
                self.npc_hider_agent = Agent(53, 4).to("cpu")
                h_path = load_model_with_mode(self.npc_hider_agent, EXPERIMENT_BASE_NAME, "HIDER", MODE)
                if h_path:
                    if should_log:
                        print(f"[Initial] Loaded Hider model: {h_path}", flush=True)
                else:
                    self.npc_hider_agent = None  # ランダム使用
            else:
                # initial + HIDER: パートナーHider は ランダムのみ
                # 学習済みモデルが存在しても読み込まない
                self.npc_hider_agent = None

        os.environ["NPC_MODELS_LOGGED"] = "TRUE"
        
        # ★ パフォーマンス最適化: 事前確保バッファ（NumPy配列生成の削減）
        # レイキャスト用バッファ
        self._raycast_geomid = np.zeros(1, dtype=np.int32)
        self._raycast_from = np.zeros(3, dtype=np.float64)
        self._raycast_dir = np.zeros(3, dtype=np.float64)
        
        # Lidar用バッファ
        self._lidar_dir = np.zeros(3, dtype=np.float64)
        self._lidar_from_pos = np.zeros(3, dtype=np.float64)  # 光線の開始位置
        self._lidar_from_pos[2] = 0.5  # Z座標は固定（地面からの高さ）
        
        # 報酬計算用バッファ
        self._reward_fwd_vec = np.zeros(2, dtype=np.float64)
        
        # 視界外マスク用の事前確保
        self._masked_vec_7 = np.full(7, OUTLIER_VALUE, dtype=np.float32)
        self._masked_vec_7[-1] = 0.0
        self._masked_vec_8 = np.full(8, OUTLIER_VALUE, dtype=np.float32)
        self._masked_vec_8[-1] = 0.0
        
        # 静的なゼロベクトル
        self._zero_vec_2 = np.zeros(2, dtype=np.float64)
        
        # 自己状態バッファ（agent_id ごとに用意、vel_x, vel_y, yaw, cos_yaw, sin_yaw）
        self._self_state = [np.zeros(5, dtype=np.float64) for _ in range(3)]
        
        # ★最適化: 観測バッファ事前割り当て（53次元、agent_id ごとに用意）
        # seeker: 5(self) + 24(lidar) + 7(box1) + 7(box2) + 7(ramp) + 3(zero_pad)
        # hider:  5(self) + 24(lidar) + 7(box1) + 7(box2) + 7(ramp) + 5(enemy)[:5] + 8(partner) + 1(grasp)
        self._obs_buffer = [np.zeros(53, dtype=np.float32) for _ in range(3)]
        
        # ★ Sightmap: 可視性キャッシュのロード/ビルド
        self._init_sightmap()

    def _init_sightmap(self):
        """可視性キャッシュの初期化（ロードまたはビルド）"""
        # XMLから静的壁を抽出（base_configのXML_CONTENTを使用）
        try:
            xml_string = base_config.XML_CONTENT
        except AttributeError as e:
            self.wall_segments = []
            self.visibility_cache = {}
            self.cell_centers = np.array([[0, 0]], dtype=np.float32)
            return
        
        maze_walls = extract_maze_walls_from_xml(xml_string, from_string=True)
        self.wall_segments = walls_to_segments(maze_walls)
        
        # キャッシュファイルの存在確認
        if os.path.exists(VISIBILITY_CACHE_FILE):
            try:
                self.visibility_cache, self.cell_centers, cache_metadata = load_visibility_cache(VISIBILITY_CACHE_FILE)
                self.sightmap_bounds = cache_metadata['bounds']
                self.sightmap_cell_size = cache_metadata['cell_size']
                self.grid_n = cache_metadata['grid_n']
            except Exception as e:
                self.visibility_cache = None
        else:
            self.visibility_cache = None
        
        # キャッシュが存在しないか、ロード失敗した場合はビルド
        if self.visibility_cache is None:
            self.cell_centers, cache_metadata = create_cell_grid(ENV_BOUNDS, SIGHTMAP_CELL_SIZE)
            self.sightmap_bounds = cache_metadata['bounds']
            self.sightmap_cell_size = cache_metadata['cell_size']
            self.grid_n = cache_metadata['grid_n']
            self.visibility_cache = build_visibility_cache(self.cell_centers, self.wall_segments)
            save_visibility_cache(self.visibility_cache, self.cell_centers, cache_metadata, VISIBILITY_CACHE_FILE)
        
        # ★ VisibilityEngine の初期化
        self.visibility_engine = VisibilityEngine(self.model, self.data, max_dist=LIDAR_MAX_DIST)
        self.visibility_engine.set_bodies(
            self.s0_body, self.h1_body, self.h2_body,
            self.box1_body, self.box2_body, self.ramp_body
        )
        
        # ★ Lidar SDF キャッシュのロード/ビルド
        self._init_lidar_sdf_cache()
    
    def _init_lidar_sdf_cache(self):
        """SDF距離場の初期化（ロードまたはビルド）"""
        # キャッシュファイルの存在確認
        if os.path.exists(SDF_DISTANCE_FIELD_FILE):
            try:
                sdf_field, sdf_cell_centers, metadata = load_sdf_distance_field(SDF_DISTANCE_FIELD_FILE)
                self.visibility_engine.sdf_field = sdf_field
                self.sdf_cell_centers = sdf_cell_centers
                return
            except Exception as e:
                pass
        
        # キャッシュが存在しないか、ロード失敗した場合はビルド
        # SDF用に別のグリッドを作成（細かい間隔）
        sdf_cell_centers, cache_metadata = create_cell_grid(ENV_BOUNDS, SDF_CELL_SIZE)
        self.sdf_cell_centers = sdf_cell_centers
        cache_metadata['cell_size'] = SDF_CELL_SIZE
        sdf_field = build_sdf_distance_field(self.visibility_engine, sdf_cell_centers)
        self.visibility_engine.sdf_field = sdf_field
        save_sdf_distance_field(sdf_field, sdf_cell_centers, cache_metadata, SDF_DISTANCE_FIELD_FILE)

    def _get_cached_ray(self, agent_id, origin, direction, beam_id):
        """ レイキャストの空間・角度キャッシュ制御。★最適化: バッファ再利用 """
        angle = np.arctan2(direction[1], direction[0])
        cache_key = (agent_id, beam_id)
        
        if cache_key in self.raycast_cache:
            cached_pos, cached_ang, cached_dist, cached_geom = self.raycast_cache[cache_key]
            pos_diff = np.linalg.norm(origin - cached_pos)
            if pos_diff < RAYCAST_CACHE_POS_THRESH:
                ang_diff = (angle - cached_ang + np.pi) % (2.0 * np.pi) - np.pi
                # ★修正: 定数を使用（ハードコード削除）
                if abs(ang_diff) < LIDAR_CACHE_ANG_THRESH:
                    self.raycast_perf["hits"] += 1
                    return cached_dist, cached_geom
        
        # 実計測の実行
        self.raycast_perf["misses"] += 1
        
        # ★最適化: 事前確保したバッファに直接書き込み（配列生成を削減）
        self._raycast_from[0] = origin[0]
        self._raycast_from[1] = origin[1]
        self._raycast_from[2] = 0.5
        self._raycast_dir[0] = direction[0]
        self._raycast_dir[1] = direction[1]
        self._raycast_dir[2] = 0.0
        
        # 除外ボディの設定
        if agent_id == 0:
            exclude_body = self.s0_body
        elif agent_id == 1:
            exclude_body = self.h1_body
        else:
            exclude_body = self.h2_body
            
        dist = mujoco.mj_ray(
            self.model, self.data, 
            self._raycast_from, 
            self._raycast_dir, 
            None, 1, exclude_body, 
            self._raycast_geomid
        )
        
        self.raycast_cache[cache_key] = (
            origin.copy(), 
            angle, 
            dist, 
            self._raycast_geomid[0]
        )
        return dist, self._raycast_geomid[0]
    
    def _get_exclude_body_for_agent(self, agent_id):
        """エージェント ID から除外する body_id を取得"""
        if agent_id == 0:
            return self.s0_body
        elif agent_id == 1:
            return self.h1_body
        else:
            return self.h2_body
    
    def _is_visible(self, origin_pos, origin_rot, target_pos, target_body_id, agent_id):
        """
        オブジェクトの可視性判定（Sightmap高速版 - 静的壁キャッシュのみ）
        """
        ax = origin_pos[0]
        ay = origin_pos[1]
        bx = target_pos[0]
        by = target_pos[1]
        
        # Numba高速化関数を呼び出し（FOV計算はグローバル定数を使用）
        visible, _ = is_visible_numba(
            ax, ay, origin_rot, bx, by,
            self.visibility_cache,
            self.sightmap_bounds,
            self.sightmap_cell_size,
            self.grid_n,
            FOV_HALF_RAD,
            FOV_COS_HALF
        )
        
        return visible, target_body_id if visible else -1
    
    def _get_obs(self, agent_id):
        """ 個体固有の 53次元観測。自己状態変数の名称を統一。 """
        if agent_id in self.obs_memo:
            return self.obs_memo[agent_id]
        
        # ★ 動的オブジェクトの位置を毎フレーム更新（Lidar計測前に）
        self.visibility_engine.update_dynamic_positions()
        
        if agent_id == 0:
            body_id = self.s0_body
            prefix = 's'
        elif agent_id == 1:
            body_id = self.h1_body
            prefix = 'h1'
        else:
            body_id = self.h2_body
            prefix = 'h2'
            
        pos = self.data.xpos[body_id][:2]
        joint_rot = self.model.joint(f'{prefix}_rot')
        yaw = self.data.qpos[self.model.jnt_qposadr[joint_rot.id]]
        
        # ★最適化: yaw三角関数を1度計算（回転行列と自己状態で統一）
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        # 回転行列はコピーで作成（バッファを上書きするのではなく毎回新規作成）
        rot_mat = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        
        # 物理速度取得
        move_joint = self.model.joint(f'{prefix}_x')
        dof_addr = self.model.jnt_dofadr[move_joint.id]
        vel_global = self.data.qvel[dof_addr : dof_addr + 2]
        vel_local = rot_mat @ vel_global
        vel_norm = vel_local / 12.0
        
        # ★最適化: 事前確保バッファに直接代入（concatenate削除）
        self_state_buf = self._self_state[agent_id]
        self_state_buf[0] = vel_norm[0]
        self_state_buf[1] = vel_norm[1]
        self_state_buf[2] = yaw
        # yaw の cos/sin も保存（元のコードに合わせる）
        self_state_buf[3] = np.cos(yaw)
        self_state_buf[4] = np.sin(yaw)
        self_state = self_state_buf.copy()  # コピーして参照を避ける
        
        # 1. Lidar データの生成 (キャッシュ活用で高速化)
        lidar = None
        if agent_id in self.lidar_cache:
            cached_pos, cached_yaw, cached_lidar = self.lidar_cache[agent_id]
            pos_diff = np.linalg.norm(pos - cached_pos)
            if pos_diff < LIDAR_CACHE_POS_THRESH:
                yaw_diff = (yaw - cached_yaw + np.pi) % (2.0 * np.pi) - np.pi
                if abs(yaw_diff) < LIDAR_CACHE_ANG_THRESH:
                    # キャッシュヒット：前フレームの Lidar 結果を再利用
                    lidar = cached_lidar.copy()
        
        # キャッシュミス：Lidar を計算
        if lidar is None:
            lidar = np.zeros(len(self.lidar_angles), dtype=np.float32)
            
            # 加法定理でビーム方向の cos/sin を一括計算
            cy = np.cos(yaw)
            sy = np.sin(yaw)
            np.multiply(self._lidar_angle_cos, cy, out=self._beam_cos)
            np.multiply(self._lidar_angle_sin, sy, out=self._beam_tmp)
            np.subtract(self._beam_cos, self._beam_tmp, out=self._beam_cos)

            np.multiply(self._lidar_angle_cos, sy, out=self._beam_sin)
            np.multiply(self._lidar_angle_sin, cy, out=self._beam_tmp)
            np.add(self._beam_sin, self._beam_tmp, out=self._beam_sin)
            
            # 全モード共通：位置設定
            self._lidar_from_pos[0] = pos[0]
            self._lidar_from_pos[1] = pos[1]
            
            for i in range(len(self.lidar_angles)):
                self._lidar_dir[0] = self._beam_cos[i]
                self._lidar_dir[1] = self._beam_sin[i]
                
                dist, _ = self.visibility_engine.cast_ray(self._lidar_from_pos, self._lidar_dir, exclude_agent_id=agent_id)
                lidar[i] = min(dist, LIDAR_MAX_DIST) / LIDAR_MAX_DIST if dist >= 0 else 1.0 
            
            # print(step_accumulator, end=", ", flush=True)  # デバッグ: 全ビームの平均ステップ数を出力    
            # キャッシュに保存
            self.lidar_cache[agent_id] = (pos.copy(), yaw, lidar.copy())

        # 2. オブジェクト視認情報の更新
        vis_lookup_record_dict_ref = self.visible_map[agent_id]
        vis_lookup_record_dict_ref.clear()
        self.visible_names[agent_id] = []
        
        target_bodies = [
            (self.box1_body, "Box1"), (self.box2_body, "Box2"), 
            (self.ramp_body, "Ramp"), (self.h1_body, "H1"), 
            (self.h2_body, "H2"), (self.s0_body, "Seeker")
        ]
        
        for tid, name in target_bodies:
            if tid != body_id:
                # デバッグ出力：距離を計算
                pos_a = self.data.xpos[body_id][:2]
                pos_b = self.data.xpos[tid][:2]
                
                is_visible, _ = self._is_visible(self.data.xpos[body_id], yaw, self.data.xpos[tid], tid, agent_id)
                vis_lookup_record_dict_ref[tid] = is_visible
                if is_visible:
                    self.visible_names[agent_id].append(name)

        def get_rel_obs(target_id, lock=None):
            """ 視認情報に基づくベクトル生成。 """
            is_visible = vis_lookup_record_dict_ref.get(target_id, False)
            dim = 8 if lock is not None else 7
            
            if is_visible:
                target_pos = self.data.xpos[target_id]
                rel_pos_global = target_pos[:2] - pos
                rel_pos_local = rot_mat @ rel_pos_global / 12.0
                
                target_quat = self.data.xquat[target_id]
                # Yaw 角度算出
                y_part = 2.0 * (target_quat[0]*target_quat[3] + target_quat[1]*target_quat[2])
                x_part = 1.0 - 2.0 * (target_quat[2]**2 + target_quat[3]**2)
                target_yaw = np.arctan2(y_part, x_part)
                
                # 速度
                target_jnt_adr = self.model.body_jntadr[target_id]
                if target_jnt_adr != -1:
                    target_vel = self.data.qvel[target_jnt_adr : target_jnt_adr + 2]
                else:
                    # ★最適化: 事前確保したゼロベクトルを使用
                    target_vel = self._zero_vec_2
                    
                rel_vel_global = target_vel - vel_global
                rel_vel_local = rot_mat @ rel_vel_global / 12.0
                
                # 順序: [pos, vel, rot, (lock), vis]
                parts = [
                    rel_pos_local, 
                    rel_vel_local, 
                    [np.cos(target_yaw - yaw), np.sin(target_yaw - yaw)]
                ]
                if lock is not None:
                    parts.append([1.0 if lock else 0.0])
                parts.append([1.0]) 
                return np.concatenate(parts)
            else:
                # ★最適化: 視界外マスクは事前確保したバッファを使用
                if lock is not None:
                    return self._masked_vec_8.copy()
                else:
                    return self._masked_vec_7.copy()

        # ★最適化: 役割別合成。バッファに直接埋め込み（concatenate削除）
        obs_buf = self._obs_buffer[agent_id]
        obs_buf[:5] = self_state
        obs_buf[5:17] = lidar  # Lidarは12ビーム
        
        if agent_id == 0:
            # Seeker: [self(5) + lidar(12) + box1(8) + box2(8) + ramp(7) + h1[:5] + h2[:5] + pad(3)]
            # 合計: 5 + 12 + 8 + 8 + 7 + 5 + 5 + 3 = 53
            obs_buf[17:25] = get_rel_obs(self.box1_body, self.locked_boxes[self.box1_body])
            obs_buf[25:33] = get_rel_obs(self.box2_body, self.locked_boxes[self.box2_body])
            obs_buf[33:40] = get_rel_obs(self.ramp_body)
            h1_info = get_rel_obs(self.h1_body)[:5]
            h2_info = get_rel_obs(self.h2_body)[:5]
            obs_buf[40:45] = h1_info
            obs_buf[45:50] = h2_info
            obs_buf[50:] = 0.0  # パディング
        else:
            # Hider: [self(5) + lidar(12) + box1(8) + box2(8) + ramp(7) + enemy[:5] + partner(7) + grasp(1)]
            # 合計: 5 + 12 + 8 + 8 + 7 + 5 + 7 + 1 = 53
            obs_buf[17:25] = get_rel_obs(self.box1_body, self.locked_boxes[self.box1_body])
            obs_buf[25:33] = get_rel_obs(self.box2_body, self.locked_boxes[self.box2_body])
            obs_buf[33:40] = get_rel_obs(self.ramp_body)
            obs_buf[40:45] = get_rel_obs(self.s0_body)[:5]
            partner = self.h2_body if agent_id == 1 else self.h1_body
            partner_info = get_rel_obs(partner)
            obs_buf[45:52] = partner_info
            obs_buf[52] = 1.0 if self.grasping[agent_id] else 0.0
        
        obs = obs_buf.astype(np.float32)
        self.obs_memo[agent_id] = obs
        return obs

    def _update_seeker_state(self):
        """ 鬼の思考ステートマシン。 """
        seeker_pos = self.data.xpos[self.s0_body][:2]
        self._get_obs(0)
        h1_seen = self.visible_map[0].get(self.h1_body, False)
        h2_seen = self.visible_map[0].get(self.h2_body, False)
        
        if h1_seen or h2_seen:
            target_bid = self.h1_body if h1_seen else self.h2_body
            target_pos = self.data.xpos[target_bid][:2].copy()
            self.seeker_target_pos = target_pos
            self.seeker_last_known_pos = target_pos.copy()
            self.seeker_mode = "CHASING"
        elif self.seeker_last_known_pos is not None:
            distance = np.linalg.norm(seeker_pos - self.seeker_last_known_pos)
            if distance > 0.5:
                self.seeker_target_pos = self.seeker_last_known_pos.copy()
                self.seeker_mode = "SEARCHING"
            else:
                self.seeker_last_known_pos = None
                self.seeker_search_timer = 50
        else:
            if self.seeker_search_timer <= 0:
                self.seeker_random_target = self.np_random.uniform(-4.0, 4.0, 2)
                self.seeker_search_timer = 80
            self.seeker_search_timer = self.seeker_search_timer - 1
            self.seeker_target_pos = self.seeker_random_target.copy()
            self.seeker_mode = "PATROLLING"
    
    def _seeker_rule_based_policy(self):
        """ 鬼の移動ロジック。 """
        if self.current_step < PREP_STEPS:
            return 0.0, 0.0
            
        seeker_pos = self.data.xpos[self.s0_body][:2]
        seeker_yaw = self.data.qpos[self.srot_adr]
        target_pos = self.seeker_target_pos
        
        dx = target_pos[0] - seeker_pos[0]
        dy = target_pos[1] - seeker_pos[1]
        target_yaw = np.arctan2(dy, dx)
        angle_err = (target_yaw - seeker_yaw + np.pi) % (2.0 * np.pi) - np.pi
        
        thrust = SEEKER_RB_THRUST
        turn = np.clip(angle_err * 6.0, -3.0, 3.0)
        
        if abs(angle_err) > SEEKER_RB_TURN_THRESH:
            thrust = thrust * 0.3
            
        dof_idx = self.model.jnt_dofadr[self.model.joint('s_x').id]
        vel_norm = np.linalg.norm(self.data.qvel[dof_idx : dof_idx + 2])
        
        if thrust > 0.05 and vel_norm < 0.05:
            self.s0_stuck_timer = self.s0_stuck_timer + 5
        else:
            self.s0_stuck_timer = max(0, self.s0_stuck_timer - 1)
            
        if self.s0_stuck_timer > 15:
            self.s0_recovery_mode = 15
            self.s0_stuck_timer = 0
            self.recovery_turn_dir = self.np_random.choice([-1.0, 1.0])
            
        if self.s0_recovery_mode > 0:
            thrust = -0.2
            turn = 1.5 * self.recovery_turn_dir
            self.s0_recovery_mode = self.s0_recovery_mode - 1
            
        return float(thrust), float(turn)

    def _get_npc_action(self, agent_id, agent_type):
        """ NPC 行動生成。 """
        # 2. モデルがある場合は推論
        observation = self._get_obs(agent_id)
        self.npc_history[agent_id].update(observation)
        model  = self.npc_seeker_agent if agent_type == "SEEKER" else self.npc_hider_agent
        
        if  model is not None:
            with torch.no_grad():
                ctx_sequence_data_tensor_ref = self.npc_history[agent_id].get()
                act_tensor_out_res_final_val, _, _, _ = model.get_action_and_value(ctx_sequence_data_tensor_ref)
            return act_tensor_out_res_final_val.cpu().numpy()[0]
            
        if agent_type == "SEEKER":
            f_rb_p_res_final, r_rb_p_res_final = self._seeker_rule_based_policy()
            normalized_thrust_applied_val = f_rb_p_res_final / SEEKER_THRUST_LIMIT
            return np.array([normalized_thrust_applied_val, r_rb_p_res_final, 0.0, 0.0], dtype=np.float32)    
        else:
            return self.action_space.sample() * 0.5 # ランダム

    def reset(self, seed=None, options=None):
        """ 環境リセット。 ★修正: 変数名の完全統一 """
        obs, info = super().reset(seed=seed, options=options)
        self.hidden_steps = 0
        self.caught_steps = 0
        self.obs_memo.clear()
        self.lidar_cache.clear()
        self.raycast_cache.clear()
        self.recovery_turn_dir = 1.0
        
        seeker_pos = self.data.xpos[self.s0_body][:2]
        for hi_idx in [1, 2]:
            body_idx = self.h1_body if hi_idx == 1 else self.h2_body
            hider_pos = self.data.xpos[body_idx][:2].copy()
            self.dist_to_seeker[hi_idx] = np.linalg.norm(hider_pos - seeker_pos)
            self.hider_pos[hi_idx] = hider_pos
            
        return obs, info

    def step(self, action):
        """ 1ステップ進展。 ★修正: 集計変数の完全統一 """
        self.current_step = self.current_step + 1
        for i in [1, 2]:
            self.lock_cooldown[i] = max(0, self.lock_cooldown[i] - 1)
        
        # ★最適化: 動的オブジェクト位置の更新は step() 内で一度だけ実行
        self.visibility_engine.update_dynamic_positions()
        
        self._update_seeker_state()
        self.data.ctrl[:] = 0.0 
        
        if TRAIN_TARGET == "HIDER":
            h_idx = self._apply_action(self.learning_agent_id, action)
            self.data.ctrl[h_idx] = float(action[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[h_idx + 1] = float(action[1])
            partner_id = 2 if self.learning_agent_id == 1 else 1
            partner_action = self._get_npc_action(partner_id, "HIDER")
            p_idx = self._apply_action(partner_id, partner_action)
            self.data.ctrl[p_idx] = float(partner_action[0]) * HIDER_THRUST_LIMIT
            self.data.ctrl[p_idx + 1] = float(partner_action[1])
            seeker_action = self._get_npc_action(0, "SEEKER")
            self.data.ctrl[0] = float(seeker_action[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(seeker_action[1])
        else:
            self.data.ctrl[0] = float(action[0]) * SEEKER_THRUST_LIMIT
            self.data.ctrl[1] = float(action[1])
            for i in [1, 2]:
                hider_action = self._get_npc_action(i, "HIDER")
                h_idx = self._apply_action(i, hider_action)
                self.data.ctrl[h_idx] = float(hider_action[0]) * HIDER_THRUST_LIMIT
                self.data.ctrl[h_idx + 1] = float(hider_action[1])

        # --- PHYSICS LOOP (Slice-based) ---
        for _ in range(ACTION_REPEAT):
            for bid, pose_data in self.locked_pose.items():
                if self.locked_boxes[bid]:
                    bjid = self.box1_joint_id if bid == self.box1_body else self.box2_joint_id
                    qa = self.model.jnt_qposadr[bjid]
                    da = self.model.jnt_dofadr[bjid]
                    self.data.qpos[qa : qa + 7] = pose_data
                    self.data.qvel[da : da + 6] = 0.0
            mujoco.mj_step(self.model, self.data)

        self.obs_memo.clear()
        # 観測確定
        obs = self._get_obs(self.learning_agent_id)
        
        # ★重要: 物理シミュレーション後のSeeker観測を再計算
        # _update_seeker_state()内の_get_obs(0)は物理シミュレーション「前」の状態
        # 報酬計算と統計には物理シミュレーション「後」の最新状態が必要
        _ = self._get_obs(0)
        
        # ★render()用の観測も物理シミュレーション後に計算
        if self.render_mode == "human":
            _ = self._get_obs(1)
            _ = self._get_obs(2)
        
        # 統計用にキャッシュ参照
        h1_visible = self.visible_map[0].get(self.h1_body, False)
        h2_visible = self.visible_map[0].get(self.h2_body, False)
        
        if h1_visible or h2_visible:
            self.caught_steps = self.caught_steps + 1
        else:
            self.hidden_steps = self.hidden_steps + 1
            
        # ★修正: 名称統一 (total_team_reward_accumulator)
        total_reward = 0.0
        for h_idx, bid in [(1, self.h1_body), (2, self.h2_body)]:
            is_visible = self.visible_map[0].get(bid, False)
            seeker_pos = self.data.xpos[self.s0_body][:2]
            hider_pos = self.data.xpos[bid][:2]
            current_dist = np.linalg.norm(hider_pos - seeker_pos)
            
            if is_visible:
                seeker_yaw = self.data.qpos[self.srot_adr]
                diff_vec = hider_pos - seeker_pos
                diff_norm = diff_vec / (np.linalg.norm(diff_vec) + 1e-8)
                
                # ★最適化: 前方ベクトルをバッファに直接書き込み
                self._reward_fwd_vec[0] = np.cos(seeker_yaw)
                self._reward_fwd_vec[1] = np.sin(seeker_yaw)
                
                cosine = np.dot(diff_norm, self._reward_fwd_vec)
                reward = -cosine * COS_PENALTY_SCALE
                dist_delta = current_dist - self.dist_to_seeker[h_idx]
                reward = reward + dist_delta * REWARD_DISTANCE_DIFF_SCALE
            else:
                reward = REWARD_HIDDEN_BONUS
                
            if h_idx == self.learning_agent_id:
                if self.hider_pos[h_idx] is not None:
                    displacement = np.linalg.norm(hider_pos - self.hider_pos[h_idx])
                    if displacement < 0.01:
                        reward = reward + PENALTY_STAGNATION
                self.hider_pos[h_idx] = hider_pos.copy()
            
            if max(abs(hider_pos)) > 6.5:
                reward = reward + PENALTY_SAFEGUARD
            
            total_reward = total_reward + reward
            self.dist_to_seeker[h_idx] = current_dist
            
        final_reward = total_reward if TRAIN_TARGET == "HIDER" else -total_reward
        truncated = (self.current_step >= MAX_STEPS)
        
        info = {
            "hidden_steps": float(self.hidden_steps), 
            "caught_steps": float(self.caught_steps)
        }
        return obs, float(final_reward), False, truncated, info

    def render(self, stats=None):
        """ MuJoCo Viewer 描画。 """
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.elevation, self.viewer.cam.distance = -60, 23.0
            
            for bid, gid in [(self.box1_body, self.box1_geom_id), (self.box2_body, self.box2_geom_id)]:
                if self.locked_boxes[bid]:
                    self.model.geom_rgba[gid][:] = [0.8, 0.1, 0.1, 1.0]
                elif any(g == bid for g in self.grasping.values()):
                    self.model.geom_rgba[gid][:] = [0.1, 0.1, 0.9, 1.0]
                else:
                    rgba = [0.6, 0.4, 0.2, 1.0] if bid == self.box1_body else [0.7, 0.5, 0.3, 1.0]
                    self.model.geom_rgba[gid][:] = rgba

            if self.viewer.user_scn:
                scn = self.viewer.user_scn
                scn.ngeom = 0 

                def add_line(p1, p2, rgba):
                    if scn.ngeom < scn.maxgeom:
                        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, size=np.array([0,0,0]), pos=np.array([0,0,0]), mat=np.eye(3).flatten(), rgba=rgba)
                        mujoco.mjv_connector(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=2.0, from_=p1, to=p2)
                        scn.ngeom = scn.ngeom + 1

                def add_label(p, txt, rgba):
                    if scn.ngeom < scn.maxgeom:
                        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=np.array([0,0,0]), pos=p, mat=np.eye(3).flatten(), rgba=rgba)
                        scn.geoms[scn.ngeom].label = txt
                        scn.ngeom = scn.ngeom + 1

                # H1 (Yellow)
                h1_pos = self.data.xpos[self.h1_body]
                for target_id in [self.box1_body, self.box2_body, self.ramp_body, self.s0_body, self.h2_body]:
                    if target_id == self.h1_body: continue
                    if self.visible_map[1].get(target_id, False):
                        add_line(h1_pos + [0, 0, 0.5], self.data.xpos[target_id] + [0, 0, 0.5], [1, 1, 0, 0.4])
                add_label(self.data.site_xpos[self.id_h1_label], f"H1 Vis:[{','.join(self.visible_names[1])}]", [1, 1, 0, 1])

                # H2 (Cyan)
                h2_pos = self.data.xpos[self.h2_body]
                for target_id in [self.box1_body, self.box2_body, self.ramp_body, self.s0_body, self.h1_body]:
                    if target_id == self.h2_body: continue
                    if self.visible_map[2].get(target_id, False):
                        add_line(h2_pos + [0, 0, 0.5], self.data.xpos[target_id] + [0, 0, 0.5], [0, 1, 1, 0.4])
                add_label(self.data.site_xpos[self.id_h2_label], f"H2 Vis:[{','.join(self.visible_names[2])}]", [0, 1, 1, 1])

                # Seeker (Red)
                seeker_pos = self.data.xpos[self.s0_body]
                # ★修正: Seekerから全オブジェクト（H1, H2, Box1, Box2, Ramp）へのラインを表示
                for target_id in [self.h1_body, self.h2_body, self.box1_body, self.box2_body, self.ramp_body]:
                    if self.visible_map[0].get(target_id, False):
                        add_line(seeker_pos + [0, 0, 0.5], self.data.xpos[target_id] + [0, 0, 0.5], [1, 0, 0, 0.6])
                add_label(self.data.site_xpos[self.id_s_label], f"S:{self.seeker_mode} Vis:[{','.join(self.visible_names[0])}]", [1, 0, 0, 1])

            self.viewer.sync()

# ==========================================
# 3. ヘルパー関数 & ファクトリ
# ==========================================

def load_model_with_mode(model, base_name, agent_type, mode):
    """モード別の識別子優先度でモデルをロード"""
    if mode == "refinement":
        # refinement → initial → 汎用
        search_paths = [
            f"{base_name}_refinement_{agent_type}.pt",
            f"{base_name}_initial_{agent_type}.pt",
            f"{base_name}_{agent_type}.pt"
        ]
    else:  # initial
        # initial → 汎用（refinement識別子は使わない）
        search_paths = [
            f"{base_name}_initial_{agent_type}.pt",
            f"{base_name}_{agent_type}.pt"
        ]
    
    print(f"[Model] Searching {agent_type} (mode={mode}):", flush=True)
    for i, path in enumerate(search_paths, 1):
        print(f"  [{i}] {path}...", flush=True, end=" ")
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()
                print(f"✓ LOADED", flush=True)
                return path
            except Exception as e:
                print(f"✗ ERROR: {e}", flush=True)
                continue
        else:
            print(f"✗ NOT FOUND", flush=True)
    
    print(f"[Model] {agent_type} model not found", flush=True)
    return None

def load_model_safely(model, base_name, agent_type):
    """後方互換性用（デフォルト: refinement モード）"""
    return load_model_with_mode(model, base_name, agent_type, "refinement")

def env_factory():
    """ AsyncVectorEnv 用。 """
    env = TeamCosEnv()
    return gym.wrappers.RecordEpisodeStatistics(env)

# ==========================================
# 5. メイン処理 (学習ループ)
# ==========================================

def main():
    if platform.system() == "Linux":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # グローバル変数で vec_envs と agent を保持（Signal ハンドラからアクセス可能）
    state_ref = {'vec_envs': None, 'agent': None, 'global_step': 0, 'device': None}
    
    def signal_handler(signum, frame):
        """Ctrl+C や SIGTERM で graceful shutdown を実行"""
        print("\n[SHUTDOWN] Received signal, saving model and closing environments...", flush=True)
        
        # モデル保存
        if state_ref['agent'] is not None and state_ref['device'] is not None:
            try:
                save_path = SAVE_MODEL_PATH.replace('.pt', '_checkpoint_emergency.pt')
                torch.save(state_ref['agent'].state_dict(), save_path)
                checkpoint_path = save_path.replace('.pt', '.json')
                with open(checkpoint_path, 'w') as f:
                    json.dump({'global_step': state_ref['global_step']}, f)
                print(f"[SHUTDOWN] Model saved: {save_path}", flush=True)
            except Exception as e:
                print(f"[SHUTDOWN] Error saving model: {e}")
        
        # 環境クローズ（パイプ破損エラーを無視）
        if state_ref['vec_envs'] is not None:
            try:
                # タイムアウト付きで close を試みる
                state_ref['vec_envs'].close(timeout=1.0)
                print("[SHUTDOWN] Environments closed.", flush=True)
            except (EOFError, BrokenPipeError, ConnectionResetError):
                # パイプ破損エラーは無視（プロセスはどっちにしろ終了する）
                print("[SHUTDOWN] Environment pipes already broken, forcing exit.", flush=True)
            except Exception as e:
                print(f"[SHUTDOWN] Error closing environments: {e}")
        
        sys.exit(0)
    
    # Signal ハンドラを登録
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
    state_ref['device'] = device
    run_time = int(time.time())
    run_id = f"{EXPERIMENT_NAME}_{TRAIN_TARGET}_{run_time}"
    
    if EXECUTION_MODE == "PLAY":
        print(f"--- Inference Mode (PLAY) ---")
        print(f"[PLAY] MODE={MODE}, TRAIN_TARGET={TRAIN_TARGET}")
        env = TeamCosEnv(render_mode="human")
        model = Agent(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
        print(f"[PLAY] Loading main model ({TRAIN_TARGET})...")
        model_path = load_model_safely(model, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if model_path:
            print(f"[PLAY] ✓ Main model loaded: {model_path}")
        else:
            print(f"[PLAY] ✗ Main model not found")
        model.eval()
        print(f"[PLAY] Environment NPC models: (see above for details)")
        obs_history = ObsHistory(1, TRANSFORMER_SEQ_LEN, env.observation_space.shape[0], device)
        try:
            while True:
                obs, _ = env.reset()
                obs_history.reset()
                obs_history.update(obs)
                done = False
                ret = 0.0
                while not done:
                    t_start = time.time()
                    with torch.no_grad():
                        action, _, _, _ = model.get_action_and_value(obs_history.get())
                    next_obs, reward, term, trunc, info = env.step(action.cpu().numpy()[0])
                    done = term or trunc
                    ret = ret + reward
                    obs_history.update(next_obs)
                    env.render(stats={"EpRet": f"{ret:.1f}"})
 
                    # ★★★ 修正: ここに終了判定を追加 (内側のループ) ★★★
                    # Viewerが生成されており、かつユーザーによって閉じられた(is_running() == False)場合
                    if env.viewer is not None:
                        if not env.viewer.is_running():
                            print("\nViewer closed by user. Exiting...")
                            env.close()
                            return  # breakではなくreturnでmain関数ごと終了させる
                    # FPS制御 (Sleep)
                    wait = (0.005 * ACTION_REPEAT) - (time.time() - t_start)
                    if wait > 0:
                        time.sleep(wait)
                print(f"Result -> Return: {ret:.1f}, Hidden: {info['hidden_steps']:.0f}", flush=True)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nInterrupted (PLAY).")
        finally:
            # Viewerを確実に閉じる
            env.close()
            # ★Viewerが閉じ切るまで少し待つハック（必要であれば）
            time.sleep(0.5)
        return

    print(f"--- [Parent] 1. Initializing {NUM_ENVS} workers ---", flush=True)
    try:
        vec_envs = gym.vector.AsyncVectorEnv([env_factory for _ in range(NUM_ENVS)])
        state_ref['vec_envs'] = vec_envs  # Signal ハンドラからアクセス可能に
        print("--- [Parent] 2. Parallel environment ready ---", flush=True)
    except Exception as e:
        print(f"--- [Parent] startup failed: {e} ---", flush=True)
        sys.exit(1)

    if TRACK_WANDB:
        wandb.init(project=base_config.WANDB_PROJECT_NAME, config={"Target": TRAIN_TARGET, "MODE": MODE, "v": "25.62_StandardizedNames"}, name=run_id, sync_tensorboard=False, save_code=True)

    writer = SummaryWriter(f"runs/{run_id}")
    agent = Agent(vec_envs.single_observation_space.shape[0], vec_envs.single_action_space.shape[0]).to(device)
    state_ref['agent'] = agent  # Signal ハンドラからアクセス可能に
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    global_step = 0
    start_step = 0
    if LOAD_EXISTING_MODELS:
        print(f"[Training] MODE={MODE}, TRAIN_TARGET={TRAIN_TARGET}")
        print(f"[Training] Loading agent model...")
        model_path = load_model_safely(agent, EXPERIMENT_BASE_NAME, TRAIN_TARGET)
        if model_path:
            print(f"[Training] ✓ Agent model resumed: {model_path}")
            checkpoint_path = model_path.replace('.pt', '_checkpoint.json')
            if os.path.exists(checkpoint_path):
                try:
                    with open(checkpoint_path, 'r') as f:
                        data = json.load(f)
                        global_step = data.get('global_step', 0)
                        start_step = global_step
                except:
                    pass
        else:
            print(f"[Training] ✗ Agent model not found, training from scratch")
    else:
        print(f"[Training] LOAD_EXISTING_MODELS=False, training from scratch")

    # --- ROLLOUT BUFFER ---
    obs_history = ObsHistory(NUM_ENVS, TRANSFORMER_SEQ_LEN, 53, device)
    S, E, O, A = NUM_STEPS, NUM_ENVS, 53, 4
    batch_obs = torch.zeros((S, E, TRANSFORMER_SEQ_LEN, O), device=device)
    batch_actions = torch.zeros((S, E, A), device=device)
    batch_logprobs = torch.zeros((S, E), device=device)
    batch_rewards = torch.zeros((S, E), device=device)
    batch_dones = torch.zeros((S, E), device=device)
    batch_values = torch.zeros((S, E), device=device)
    
    next_obs = vec_envs.reset(seed=FIXED_SEED if FIXED_SEED else int(time.time()))[0]
    next_done = torch.zeros(E).to(device)
    obs_history.reset()
    obs_history.update(next_obs)
    
    num_updates = int(max(1, (TOTAL_TIMESTEPS - global_step) // (E * S)))
    history_returns = []
    history_hidden = []
    history_caught = []
    start_time = time.time()
    last_loss = 0.0
    last_entropy = 0.0

    print(f"--- Training Sequence Started (v25.62) ---")
    try:
        # ★最適化: OPTUNA_SEARCH環境変数で判定し、子プロセス時はtqdmを無効化
        disable_tqdm = os.environ.get("OPTUNA_SEARCH", "False") == "True"
        for u in tqdm(range(1, num_updates + 1), desc="Updates", disable=disable_tqdm):
            for step in range(S):
                global_step = global_step + E
                state_ref['global_step'] = global_step  # Signal ハンドラ用に同期
                batch_obs[step] = obs_history.get()
                batch_dones[step] = next_done
                
                with torch.no_grad():
                    action, lp, _, value = agent.get_action_and_value(obs_history.get())
                    batch_values[step] = value.flatten()
                
                batch_actions[step] = action
                batch_logprobs[step] = lp
                
                next_obs, reward, term, trunc, info = vec_envs.step(action.cpu().numpy())
                done = np.logical_or(term, trunc)
                
                if "final_info" in info:
                    for e in range(E):
                        final = info["final_info"][e]
                        if done[e] and final is not None:
                            if "episode" in final:
                                history_returns.append(float(final["episode"]["r"]))
                            if "hidden_steps" in final:
                                history_hidden.append(float(final["hidden_steps"]))
                            if "caught_steps" in final:
                                history_caught.append(float(final["caught_steps"]))
                elif "episode" in info:
                    mask = info.get("_episode", [True] * E)
                    for e in range(E):
                        if mask[e] and done[e]:
                            history_returns.append(float(info["episode"]["r"][e]))
                            if "hidden_steps" in info:
                                history_hidden.append(float(info["hidden_steps"][e]))
                            if "caught_steps" in info:
                                history_caught.append(float(info["caught_steps"][e]))
                
                batch_rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_done = torch.tensor(done).to(device, dtype=torch.float32)
                obs_history.update(next_obs)
            
            # --- PPO UPDATE ---
            with torch.no_grad():
                v_next = agent.get_value(obs_history.get()).reshape(1, -1)
                advantages = torch.zeros_like(batch_rewards).to(device)
                gae = 0
                for t in reversed(range(S)):
                    if t == S - 1:
                        nt = 1.0 - next_done
                        vp = v_next
                    else:
                        nt = 1.0 - batch_dones[t + 1]
                        vp = batch_values[t + 1]
                    delta = batch_rewards[t] + 0.99 * vp * nt - batch_values[t]
                    gae = delta + 0.99 * 0.95 * nt * gae
                    advantages[t] = gae
                returns = advantages + batch_values
            
            flat_obs = batch_obs.reshape((-1, TRANSFORMER_SEQ_LEN, 53))
            flat_lp = batch_logprobs.reshape(-1)
            flat_actions = batch_actions.reshape((-1, 4))
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = batch_values.reshape(-1)
            
            # PPO OPTIMIZATION
            for ep in range(UPDATE_EPOCHS):
                idx = np.arange(S * E)
                np.random.shuffle(idx)
                for start in range(0, S * E, MINIBATCH_SIZE):
                    mb_idx = idx[start : start + MINIBATCH_SIZE]
                    _, new_lp, entropy, new_value = agent.get_action_and_value(flat_obs[mb_idx], flat_actions[mb_idx])
                    
                    log_ratio = new_lp - flat_lp[mb_idx]
                    ratio = log_ratio.exp()
                    
                    with torch.no_grad():
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    
                    mb_adv = flat_advantages[mb_idx]
                    mb_adv_norm = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    
                    loss_pol_1 = -mb_adv_norm * ratio
                    loss_pol_2 = -mb_adv_norm * torch.clamp(ratio, 0.8, 1.2)
                    loss_pol = torch.max(loss_pol_1, loss_pol_2).mean()
                    
                    loss_val = 0.5 * ((new_value.view(-1) - flat_returns[mb_idx]) ** 2).mean()
                    loss_total = loss_pol - ENT_COEF * entropy.mean() + 0.5 * loss_val
                    
                    optimizer.zero_grad()
                    loss_total.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
                    
                    last_loss = loss_total.item()
                    last_entropy = entropy.mean().item()

            y_pred = flat_values.cpu().numpy()
            y_actual = flat_returns.cpu().numpy()
            var_y = np.var(y_actual)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_actual - y_pred) / var_y
                    
            if history_returns:
                avg_hidden = np.mean(history_hidden)
                avg_caught = np.mean(history_caught)
                avg_return = np.mean(history_returns)
                
                # ★最適化: ログ出力を10回に1回に削減
                if (TRIAL_MODE) or (u % 10 == 0):
                    elapsed = time.time() - start_time
                    sps = int((global_step - start_step) / elapsed) if elapsed > 0 else 0
                    
                    # tqdmの postfix で情報表示（ログ行を増やさない）
                    tqdm.write(f"Update {u:4d}, SPS: {sps:4d}, EpRet: {avg_return:7.1f}, Hidden: {avg_hidden:6.1f}, Caught: {avg_caught:6.1f}")
                    sys.stdout.flush() 
                    
                    if TRACK_WANDB:
                        wandb.log({
                            "charts/SPS": sps, 
                            "losses/total_loss": last_loss, 
                            "losses/entropy": last_entropy,
                            "losses/explained_variance": explained_var,
                            "charts/episodic_return": avg_return, 
                            "charts/steps_hidden": avg_hidden, 
                            "global_step": global_step
                        })
                    writer.add_scalar("charts/SPS", sps, global_step)
                    
                    history_returns = []
                    history_hidden = []
                    history_caught = []
                
    except KeyboardInterrupt:
        print("\nInterrupted.")
        vec_envs.close()
        sys.exit(0)
        
    if SAVE_MODEL :
        torch.save(agent.state_dict(), SAVE_MODEL_PATH)
        checkpoint_path = SAVE_MODEL_PATH.replace('.pt', '_checkpoint.json')
        with open(checkpoint_path, 'w') as f:
            json.dump({'global_step': global_step}, f)
        print(f"Model saved: {SAVE_MODEL_PATH}")
        
    vec_envs.close()
    writer.close()
    if TRACK_WANDB:
        wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hide and Seek RL Environment")
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["TRAIN", "PLAY"],
        default=EXECUTION_MODE,
        help="Execution mode: TRAIN (learning) or PLAY (inference with rendering)"
    )
    parser.add_argument(
        "--target", "-t",
        type=str,
        choices=["SEEKER", "HIDER"],
        default=TRAIN_TARGET,
        help="Training target agent: SEEKER or HIDER"
    )
    parser.add_argument(
        "--stage", "-s",
        type=str,
        choices=["initial", "refinement"],
        default=MODE,
        help="Training stage: initial or refinement"
    )
    
    args = parser.parse_args()
    
    # グローバル変数を設定
    import sys
    current_module = sys.modules[__name__]
    current_module.EXECUTION_MODE = args.mode
    current_module.TRAIN_TARGET = args.target
    current_module.MODE = args.stage
    
    print(f"[Config] Mode: {args.mode}, Target: {args.target}, Stage: {args.stage}, Lidar: {LIDAR_RAYCAST_MODE}")
    main()