# visibility_engine.py
# 演習第25回：ハイブリッド・高速演算エンジン（mj_ray引数修正・is_visible最適化版）
# 
# 修正ポイント:
# 1. mj_ray 型安全化: TypeError を回避するため、geomid 引数に np.int32 配列を渡すように修正。
# 2. 幾何判定の正確化: _intersect_ray_box_jit 内の Y 軸判定ロジックのタイポを修正。
# 3. is_visible 高速化: 30度内積フィルタ + 距離フィルタ + Sightmap バイパスを統合。

import numpy as np
import mujoco
import pickle
import time
import math
from pathlib import Path
from numba import njit

# --- JIT 加速された要素計算関数群 ---

@njit(cache=True)
def _intersect_ray_box_jit(origin_x, origin_y, dir_x, dir_y, box_x, box_y, box_sx, box_sy, b_cos, b_sin, max_dist):
    """回転した矩形(OBB)とのレイ交差判定 (スカラー高速版)"""
    dx, dy = origin_x - box_x, origin_y - box_y
    local_p_x = dx * b_cos + dy * b_sin
    local_p_y = -dx * b_sin + dy * b_cos
    local_v_x = dir_x * b_cos + dir_y * b_sin
    local_v_y = -dir_x * b_sin + dir_y * b_cos
    
    t_min, t_max = -1e10, max_dist
    
    # X軸方向判定
    if abs(local_v_x) < 1e-10:
        if abs(local_p_x) > box_sx: return max_dist
    else:
        inv_dx = 1.0 / local_v_x
        t1 = (-box_sx - local_p_x) * inv_dx
        t2 = (box_sx - local_p_x) * inv_dx
        if t1 > t2: t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)

    # Y軸方向判定
    if abs(local_v_y) < 1e-10:
        if abs(local_p_y) > box_sy: return max_dist
    else:
        inv_dy = 1.0 / local_v_y
        t1 = (-box_sy - local_p_y) * inv_dy
        t2 = (box_sy - local_p_y) * inv_dy
        if t1 > t2: t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
            
    if t_max >= t_min and t_max >= 0:
        res = t_min if t_min >= 0 else t_max
        return res if res <= max_dist else max_dist
    return max_dist

@njit(cache=True)
def _intersect_ray_circle_jit(ox, oy, dx, dy, cx, cy, r, max_dist):
    """レイと円の交差判定 (スカラー高速版)"""
    ocx, ocy = ox - cx, oy - cy
    a = dx*dx + dy*dy
    b = 2.0 * (ocx * dx + ocy * dy)
    c = (ocx*ocx + ocy*ocy) - r*r
    h = b*b - 4*a*c
    if h < 0: return max_dist
    t = (-b - math.sqrt(h)) / (2.0 * a)
    return t if 0 <= t <= max_dist else max_dist

# --- JIT 加速されたメインロジック ---

@njit(cache=True)
def _compute_geometric_lidar_optimized_jit(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin, base_angles, walls, boxes, circles, max_dist):
    """Mode 1: 方向計算を内包した高速幾何判定"""
    res = np.empty(12, dtype=np.float32)
    limit_angle = 0.5236 # 30 deg
    heading = math.atan2(h_sin, h_cos)
    for i in range(12):
        d_min = max_dist
        vx = base_cos[i] * h_cos - base_sin[i] * h_sin
        vy = base_sin[i] * h_cos + base_cos[i] * h_sin
        ang = base_angles[i] + heading
        
        # 1. 静的な壁
        for j in range(len(walls)):
            d = _intersect_ray_box_jit(pos_x, pos_y, vx, vy, walls[j,0], walls[j,1], walls[j,2], walls[j,3], 1.0, 0.0, max_dist)
            if d < d_min: d_min = d
            
        # 2. 動的ボックス (OBB)
        for j in range(len(boxes)):
            bx, by, bsx, bsy, bcos, bsin = boxes[j,0], boxes[j,1], boxes[j,2], boxes[j,3], boxes[j,5], boxes[j,6]
            rel_x, rel_y = bx - pos_x, by - pos_y
            obj_ang = math.atan2(rel_y, rel_x)
            diff = abs(obj_ang - ang)
            if diff > 3.14159: diff = 6.28318 - diff
            if diff > limit_angle and bsx < 0.65: continue
            d = _intersect_ray_box_jit(pos_x, pos_y, vx, vy, bx, by, bsx, bsy, bcos, bsin, max_dist)
            if d < d_min: d_min = d
            
        # 3. エージェント
        for j in range(len(circles)):
            cx, cy, cr = circles[j,0], circles[j,1], circles[j,2]
            rel_x, rel_y = cx - pos_x, cy - pos_y
            obj_ang = math.atan2(rel_y, rel_x)
            diff = abs(obj_ang - ang)
            if diff > 3.14159: diff = 6.28318 - diff
            if diff > limit_angle: continue
            d = _intersect_ray_circle_jit(pos_x, pos_y, vx, vy, cx, cy, cr, max_dist)
            if d < d_min: d_min = d
        res[i] = d_min
    return res

@njit(cache=True)
def _compute_sphere_tracing_optimized_jit(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin, base_angles, sdf_map, bounds, cell_size, n_side, boxes, circles, max_dist):
    """Mode 2: 方向計算を内包した高速SDF走査"""
    res = np.empty(12, dtype=np.float32)
    epsilon = 0.005; max_steps = 40; limit_angle = 0.5236
    heading = math.atan2(h_sin, h_cos)
    for i in range(12):
        vx = base_cos[i] * h_cos - base_sin[i] * h_sin
        vy = base_sin[i] * h_cos + base_cos[i] * h_sin
        ang = base_angles[i] + heading
        t = 0.0
        
        active_boxes_idx = []
        for j in range(len(boxes)):
            if boxes[j,2] > 0.65: active_boxes_idx.append(j)
            else:
                rel_x, rel_y = boxes[j,0] - pos_x, boxes[j,1] - pos_y
                obj_ang = math.atan2(rel_y, rel_x)
                diff = abs(obj_ang - ang)
                if diff > 3.14159: diff = 6.28318 - diff
                if diff <= limit_angle: active_boxes_idx.append(j)
        
        for _ in range(max_steps):
            cx, cy = pos_x + vx * t, pos_y + vy * t
            fx, fy = (cx + bounds) / cell_size, (bounds - cy) / cell_size
            ix, iy = int(fx), int(fy)
            
            if ix < 0 or ix >= n_side - 1 or iy < 0 or iy >= n_side - 1: d_min = 0.0
            else:
                wx, wy = fx - ix, fy - iy
                d00, d10, d01, d11 = sdf_map[iy,ix], sdf_map[iy,ix+1], sdf_map[iy+1,ix], sdf_map[iy+1,ix+1]
                d_min = (d00 * (1.0 - wx) + d10 * wx) * (1.0 - wy) + (d01 * (1.0 - wx) + d11 * wx) * wy
            
            for idx in active_boxes_idx:
                bx, by, bsx, bsy, bcos, bsin = boxes[idx,0], boxes[idx,1], boxes[idx,2], boxes[idx,3], boxes[idx,5], boxes[idx,6]
                dx, dy = cx - bx, cy - by
                lx, ly = abs(dx * bcos + dy * bsin), abs(-dx * bsin + dy * bcos)
                qx, qy = lx - bsx, ly - bsy
                d_obj = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2) + min(max(qx, qy), 0.0)
                if d_obj < d_min: d_min = d_obj
                
            for j in range(len(circles)):
                dist_c = math.sqrt((cx - circles[j,0])**2 + (cy - circles[j,1])**2) - circles[j,2]
                if dist_c < d_min: d_min = dist_c
            
            if d_min < epsilon: break
            t += d_min
            if t >= max_dist: t = max_dist; break
        res[i] = t
    return res

class VisibilityEngine:
    def __init__(self, m, d, layout_name="Maze"):
        self.m, self.d, self.max_dist = m, d, 20.0
        mujoco.mj_forward(self.m, self.d)
        self.cache_path = Path(f"hns_sightmap_v25_cache_{layout_name}.pkl")
        self.walls_np = self._extract_static_walls()
        self.dynamic_bodies = self._prepare_dynamic_objects()
        self.cache = self._ensure_cache(layout_name)
        
        self.angles_deg = np.array([0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180], dtype=np.float32)
        self.angles_rad = np.deg2rad(self.angles_deg).astype(np.float32)
        self.base_cos, self.base_sin = np.cos(self.angles_rad).astype(np.float32), np.sin(self.angles_rad).astype(np.float32)
        
        # mj_ray 用の書き込み可能バッファ (TypeError 回避用)
        self._geomid_out = np.zeros(1, dtype=np.int32)

    def _extract_static_walls(self):
        walls = []
        for i in range(self.m.ngeom):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and any(k in name for k in ["wall", "maze", "border"]):
                walls.append([self.d.geom_xpos[i][0], self.d.geom_xpos[i][1], self.m.geom_size[i][0], self.m.geom_size[i][1]])
        return np.array(walls, dtype=np.float32)

    def _prepare_dynamic_objects(self):
        dyn = {"boxes": [], "agents": []}
        for b_name in ["box1_body", "box2_body", "ramp_body"]:
            try:
                bid = self.m.body(b_name).id
                g_start, g_num = self.m.body_geomadr[bid], self.m.body_geomnum[bid]
                for g_idx in range(g_start, g_start + g_num): dyn["boxes"].append((bid, g_idx))
            except: pass
        for a_name in ["hider1_body", "hider2_body", "seeker_body"]:
            try: bid = self.m.body(a_name).id; dyn["agents"].append(bid)
            except: pass
        return dyn

    def _ensure_cache(self, layout_name):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    cache = pickle.load(f)
                if cache and "sdf_map" in cache: return cache
            except: pass
        print(f"🔄 SDF Map を新規生成します...")
        # (自動生成ロジックが必要な場合はここに実装)
        return None 

    def _get_dynamic_objects_np(self, body_exclude):
        boxes_list, circles_list = [], []
        for bid, gid in self.dynamic_bodies["boxes"]:
            q = self.d.xquat[bid]
            yaw = math.atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]))
            boxes_list.append([self.d.geom_xpos[gid][0], self.d.geom_xpos[gid][1], self.m.geom_size[gid][0], self.m.geom_size[gid][1], yaw, math.cos(yaw), math.sin(yaw)])
        for bid in self.dynamic_bodies["agents"]:
            if bid != body_exclude: circles_list.append([self.d.xpos[bid][0], self.d.xpos[bid][1], 0.4])
        return np.array(boxes_list, dtype=np.float32) if boxes_list else np.empty((0,7), dtype=np.float32), \
               np.array(circles_list, dtype=np.float32) if circles_list else np.empty((0,3), dtype=np.float32)

    def cast_lidar(self, pos, heading=0.0, mode=1, body_exclude=-1):
        h_cos, h_sin = math.cos(heading), math.sin(heading)
        
        if mode == 0:
            res = np.zeros(12, dtype=np.float32)
            p_origin = np.ascontiguousarray([pos[0], pos[1], 0.5], dtype=np.float64)
            for i in range(12):
                vx, vy = self.base_cos[i] * h_cos - self.base_sin[i] * h_sin, self.base_sin[i] * h_cos + self.base_cos[i] * h_sin
                v_dir = np.ascontiguousarray([vx, vy, 0.0], dtype=np.float64)
                # 修正: geomid 引数に None ではなく np.int32 配列を渡す
                d_val = mujoco.mj_ray(self.m, self.d, p_origin, v_dir, None, 1, int(body_exclude), self._geomid_out)
                res[i] = d_val if (0 <= d_val <= self.max_dist) else self.max_dist
            return res

        boxes_np, circles_np = self._get_dynamic_objects_np(body_exclude)
        if mode == 2 and self.cache is not None:
            return _compute_sphere_tracing_optimized_jit(pos[0], pos[1], h_cos, h_sin, self.base_cos, self.base_sin, self.angles_rad, self.cache["sdf_map"], self.cache["bounds"], self.cache["cell_size"], self.cache["n_side"], boxes_np, circles_np, self.max_dist)
        return _compute_geometric_lidar_optimized_jit(pos[0], pos[1], h_cos, h_sin, self.base_cos, self.base_sin, self.angles_rad, self.walls_np, boxes_np, circles_np, self.max_dist)

    def is_visible(self, p1, p2, body_exclude=-1):
        """p1 から p2 への可視性を 30度内積フィルタ + Sightmap で高速判定"""
        diff = p2 - p1; dist = np.sqrt(np.sum(diff**2))
        if dist < 0.05: return True
        
        # 1. 静的な壁のバイパス (Sightmap 利用)
        if self.cache and "vis_matrix" in self.cache:
            idx1 = self._pos_to_vis_idx(p1); idx2 = self._pos_to_vis_idx(p2)
            if idx1 is not None and idx2 is not None and not self.cache["vis_matrix"][idx1, idx2]:
                return False

        vx, vy = diff[0]/dist, diff[1]/dist
        boxes_np, circles_np = self._get_dynamic_objects_np(body_exclude)
        
        # 2. 静的な壁との幾何交差 (Sightmapがない場合のフォールバック)
        if not self.cache or "vis_matrix" not in self.cache:
            for j in range(len(self.walls_np)):
                if _intersect_ray_box_jit(p1[0], p1[1], vx, vy, self.walls_np[j,0], self.walls_np[j,1], self.walls_np[j,2], self.walls_np[j,3], 1.0, 0.0, dist) < dist - 0.01:
                    return False

        # 3. 動的オブジェクトとの交差 (30度フィルタ + 距離フィルタ)
        for j in range(len(boxes_np)):
            bx, by = boxes_np[j, 0], boxes_np[j, 1]
            rel_x, rel_y = bx - p1[0], by - p1[1]
            d_obj = math.sqrt(rel_x**2 + rel_y**2)
            if d_obj > dist + 0.6: continue 
            dot = (vx * rel_x + vy * rel_y) / (d_obj + 1e-8)
            if dot < 0.87: continue 
            if _intersect_ray_box_jit(p1[0], p1[1], vx, vy, bx, by, boxes_np[j,2], boxes_np[j,3], boxes_np[j,5], boxes_np[j,6], dist) < dist - 0.01:
                return False

        for j in range(len(circles_np)):
            cx, cy = circles_np[j, 0], circles_np[j, 1]
            rel_x, rel_y = cx - p1[0], cy - p1[1]
            d_obj = math.sqrt(rel_x**2 + rel_y**2)
            if d_obj > dist + 0.4: continue
            dot = (vx * rel_x + vy * rel_y) / (d_obj + 1e-8)
            if dot < 0.87: continue
            if _intersect_ray_circle_jit(p1[0], p1[1], vx, vy, cx, cy, circles_np[j,2], dist) < dist - 0.01:
                return False
            
        return True

    def _pos_to_vis_idx(self, p):
        if not self.cache or "vis_n_side" not in self.cache: return None
        c = self.cache; bounds, cell_size = c["bounds"], c["vis_cell_size"]
        ix, iy = int((p[0] + bounds) / cell_size), int((bounds - p[1]) / cell_size)
        if 0 <= ix < c["vis_n_side"] and 0 <= iy < c["vis_n_side"]: return iy * c["vis_n_side"] + ix
        return None