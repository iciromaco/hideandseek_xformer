# visibility_engine.py
# 演習第25回：Numba高速化とMuJoCo型安全性を両立した演算エンジン
# 
# 【修正】循環参照を防ぐため hns_environment からのインポートを削除しました。
# 物理環境の XML には一切手を加えず、Python 側の呼び出し精度を調整することで
# mj_ray の型エラー (TypeError) を解決しています。

import numpy as np
import mujoco
from numba import njit

# ==========================================
# Numbaによる数学計算コア (JITコンパイル対象)
# ==========================================

@njit(cache=True)
def compute_box_sdf_core(p, boxes):
    """
    矩形集合に対する最短距離(SDF)を算出します。
    """
    min_dist = 1e10
    for i in range(boxes.shape[0]):
        # 中心相対座標への変換と、絶対値化による対称化
        dx = abs(p[0] - boxes[i, 0]) - boxes[i, 2]
        dy = abs(p[1] - boxes[i, 1]) - boxes[i, 3]
        
        # 外側距離（正）と内側距離（負）を統合
        outer_dist = np.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2)
        inner_dist = min(max(dx, dy), 0.0)
        
        d = outer_dist + inner_dist
        if d < min_dist:
            min_dist = d
    return min_dist

@njit(cache=True)
def get_ray_intersect_dist(p_start, p_dir, a, b):
    """
    2D平面上での半直線と線分(a-b)の交差距離をベクトル演算で算出します。
    """
    v1 = p_start - a
    v2 = b - a
    v3 = np.array([-p_dir[1], p_dir[0]], dtype=np.float32)
    
    dot = v2[0] * v3[0] + v2[1] * v3[1]
    if abs(dot) < 1e-8:
        return 1e10 # 平行

    t1 = (v2[0] * v1[1] - v2[1] * v1[0]) / dot
    t2 = (v1[0] * v3[0] + v1[1] * v3[1]) / dot
    
    if t1 >= 0.0 and 0.0 <= t2 <= 1.0:
        return t1
    return 1e10

@njit(cache=True)
def geometric_lidar_scan(pos, directions, walls, boxes, max_dist=10.0):
    """
    線分交差判定(2D Intersection)を用いた超高速Lidarスキャン。
    """
    results = np.full(12, max_dist, dtype=np.float32)
    for i in range(12):
        p_dir = directions[i]
        min_d = max_dist
        
        for j in range(walls.shape[0]):
            d = get_ray_intersect_dist(pos, p_dir, walls[j, 0], walls[j, 1])
            if d < min_d: d = min_d
            
        for j in range(boxes.shape[0]):
            bx, by, hw, hh = boxes[j]
            corners = [
                np.array([bx-hw, by-hh]), np.array([bx+hw, by-hh]),
                np.array([bx+hw, by+hh]), np.array([bx-hw, by+hh])
            ]
            for k in range(4):
                d = get_ray_intersect_dist(pos, p_dir, corners[k], corners[(k+1)%4])
                if d < min_d: min_d = d
        results[i] = min_d
    return results

# ==========================================
# VisibilityEngine クラス (MuJoCo連携層)
# ==========================================
class VisibilityEngine:
    """
    AIの視覚情報を管理するエンジンクラス。
    """
    def __init__(self, m, d):
        self.m = m
        self.d = d
        self._extract_geoms()
        # Lidarの12方向（30度間隔）を事前計算
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        self.lidar_dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
        
        # ⚠️ mjpython 型エラー解消の鍵: 書き込み可能なIDバッファを事前確保
        self._geomid_out = np.zeros(1, dtype=np.int32)

    def _extract_geoms(self):
        """モデルから幾何情報を抽出。内壁 maze_w も自動で抽出対象となります。"""
        walls = []
        self.box_ids = []
        self.box_data = []
        
        for i in range(self.m.ngeom):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name is None:
                continue
                
            p = self.m.geom_pos[i][:2]
            s = self.m.geom_size[i][:2]
            
            # 名称から静的・動的を判別
            if "wall" in name or "border" in name or "maze" in name:
                if s[0] > s[1]:
                    walls.append([[p[0]-s[0], p[1]], [p[0]+s[0], p[1]]])
                else:
                    walls.append([[p[0], p[1]-s[1]], [p[0], p[1]+s[1]]])
            if "box" in name or "ramp" in name:
                self.box_ids.append(i)
                self.box_data.append([p[0], p[1], s[0], s[1]])
                
        self.walls = np.array(walls, dtype=np.float32)
        self.box_data = np.array(self.box_data, dtype=np.float32)

    def update_positions(self):
        """動的オブジェクト位置の更新"""
        for i, gid in enumerate(self.box_ids):
            self.box_data[i, :2] = self.d.geom_xpos[gid][:2]

    def is_visible(self, p1, p2, body_exclude=-1):
        """
        2点間が遮蔽されていないかを判定。
        TypeError対策：引数を厳密な精度と連続メモリで渡します。
        """
        diff = p2 - p1
        dist = np.linalg.norm(diff)
        if dist < 0.01:
            return True
        ray_dir = diff / (dist + 1e-8)
        
        # ⚠️ 修正：メモリ連続な float64 配列として定義
        pnt = np.ascontiguousarray([p1[0], p1[1], 0.1], dtype=np.float64)
        vec = np.ascontiguousarray([ray_dir[0], ray_dir[1], 0.0], dtype=np.float64)
        
        # ⚠️ 修正：geomid に int32 バッファ配列を指定
        hit_dist = mujoco.mj_ray(
            self.m, 
            self.d, 
            pnt, 
            vec, 
            None, 
            1, 
            int(body_exclude), 
            self._geomid_out
        )
        return hit_dist < 0 or hit_dist > dist

    def cast_lidar(self, pos, mode=1, body_exclude=-1):
        self.update_positions()
        if mode == 0:
            res = np.zeros(12, dtype=np.float32)
            p_origin = np.ascontiguousarray([pos[0], pos[1], 0.1], dtype=np.float64)
            for i in range(12):
                v_dir = np.ascontiguousarray([self.lidar_dirs[i,0], self.lidar_dirs[i,1], 0.0], dtype=np.float64)
                d = mujoco.mj_ray(self.m, self.d, p_origin, v_dir, None, 1, int(body_exclude), self._geomid_out)
                res[i] = d if d >= 0 else 10.0
            return res
        return geometric_lidar_scan(pos.astype(np.float32), self.lidar_dirs, self.walls, self.box_data)