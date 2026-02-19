# visibility_engine.py
# 演習第25回：Lidar/Visibility 完全統合・エージェント形状最適化(btm限定)版
# 
# 修正点:
# 1. 計算対象の極小化: 装飾パーツ(nose, tail, capsule)を計算から除外。
#    seeker_btm, hider1_btm, hider2_btm のみを Lidar/視界判定の対象とする。
# 2. パフォーマンス向上: エージェントあたりの判定 geom 数を 1 つに絞り、幾何演算を高速化。
# 3. 物理真理の維持: 物理的な最大体積である _btm (Sphere) を基準とすることで、実用上の精度を確保。
# 4. デバッグ出力の整理: 計算対象(Active)と装飾(Decorative)を明確に区別。

import numpy as np
import mujoco
import pickle
import math
from pathlib import Path
from numba import njit

# --- JIT 加速された演算コア (数値的安定性・全オブジェクト対応版) ---

@njit(cache=True)
def _intersect_ray_aabb_robust_jit(ox, oy, dx, dy, bx, by, bsx, bsy, max_dist):
    """Slab法による軸並行矩形との交差判定。最短の正の交点(t > 0)を返す。"""
    t_min, t_max = -1e10, 1e10; eps = 1e-12
    if abs(dx) < eps:
        if abs(ox - bx) > bsx: return max_dist
    else:
        inv_d = 1.0 / dx
        t1, t2 = (bx - bsx - ox) * inv_d, (bx + bsx - ox) * inv_d
        t_min, t_max = max(t_min, min(t1, t2)), min(t_max, max(t1, t2))
    if abs(dy) < eps:
        if abs(oy - by) > bsy: return max_dist
    else:
        inv_d = 1.0 / dy
        t1, t2 = (by - bsy - oy) * inv_d, (by + bsy - oy) * inv_d
        t_min, t_max = max(t_min, min(t1, t2)), min(t_max, max(t1, t2))
    
    if t_max > t_min and t_max > 1e-6:
        hit_t = t_min if t_min > 1e-6 else t_max
        return hit_t if hit_t < max_dist else max_dist
    return max_dist

@njit(cache=True)
def _compute_geometric_lidar_core_jit(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin, 
                                     walls, g_xpos, g_size, box_g_ids,
                                     b_quat, box_b_ids, a_xpos, ag_b_ids,
                                     exclude_idx, max_dist):
    """Mode 1: 装飾を除外した _btm 基準の超高速幾何判定コア"""
    res = np.empty(12, dtype=np.float32)
    n_boxes = len(box_g_ids)
    cos_y, sin_y = np.empty(n_boxes, dtype=np.float32), np.empty(n_boxes, dtype=np.float32)
    for k in range(n_boxes):
        q = b_quat[box_b_ids[k]]
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
        cos_y[k], sin_y[k] = math.cos(yaw), math.sin(yaw)

    for i in range(12):
        d_min = max_dist
        vx, vy = base_cos[i]*h_cos - base_sin[i]*h_sin, base_sin[i]*h_cos + base_cos[i]*h_sin
        
        # 1. 静的な壁
        for j in range(len(walls)):
            d = _intersect_ray_aabb_robust_jit(pos_x, pos_y, vx, vy, walls[j,0], walls[j,1], walls[j,2], walls[j,3], d_min)
            if d < d_min: d_min = d
            
        # 2. 動的な箱/ランプ
        for k in range(n_boxes):
            gid = box_g_ids[k]; bc, bs = cos_y[k], sin_y[k]
            bx, by = g_xpos[gid, 0], g_xpos[gid, 1]
            bsx, bsy = g_size[gid, 0], g_size[gid, 1]
            rx, ry = pos_x - bx, pos_y - by
            lx, ly = rx*bc + ry*bs, -rx*bs + ry*bc; lvx, lvy = vx*bc + vy*bs, -vx*bs + vy*bc
            d = _intersect_ray_aabb_robust_jit(lx, ly, lvx, lvy, 0.0, 0.0, bsx, bsy, d_min)
            if d < d_min: d_min = d
            
        # 3. 他エージェント (_btm ボディの中心座標を利用した円判定)
        for k in range(len(ag_b_ids)):
            if k == exclude_idx: continue
            ax, ay = a_xpos[ag_b_ids[k], 0], a_xpos[ag_b_ids[k], 1]
            ox, oy = pos_x - ax, pos_y - ay
            a = vx*vx + vy*vy; b = 2.0 * (ox*vx + oy*vy); c = (ox*ox + oy*oy) - 0.2025 # r=0.45m
            h = b*b - 4*a*c
            if h >= 0:
                sqrt_h = math.sqrt(h); t1, t2 = (-b - sqrt_h) / (2.0 * a), (-b + sqrt_h) / (2.0 * a)
                t = t1 if t1 > 1e-6 else t2
                if 1e-6 < t < d_min: d_min = t
        res[i] = d_min
    return res

@njit(cache=True)
def _compute_sphere_tracing_core_jit(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin,
                                    sdf_map, bounds, cell_size, n_side,
                                    g_xpos, g_size, box_g_ids, b_quat, box_b_ids,
                                    a_xpos, ag_b_ids, exclude_idx, max_dist):
    """Mode 2: _btm のみを合成対象とした高速 Sphere Tracing"""
    res = np.empty(12, dtype=np.float32)
    epsilon = 0.005; max_steps = 64; n_boxes = len(box_g_ids)
    for i in range(12):
        vx, vy = base_cos[i]*h_cos - base_sin[i]*h_sin, base_sin[i]*h_cos + base_cos[i]*h_sin
        t = 0.0
        for _ in range(max_steps):
            cx, cy = pos_x + vx * t, pos_y + vy * t
            gx, gy = (cx + bounds) / cell_size, (bounds - cy) / cell_size
            ix, iy = int(gx), int(gy)
            if 0 <= ix < n_side - 1 and 0 <= iy < n_side - 1:
                wx, wy = gx - ix, gy - iy
                d00, d10, d01, d11 = sdf_map[iy,ix], sdf_map[iy,ix+1], sdf_map[iy+1,ix], sdf_map[iy+1,ix+1]
                d_min = (d00*(1.0-wx) + d10*wx)*(1.0-wy) + (d01*(1.0-wx) + d11*wx)*wy
            else: d_min = 0.0
            for k in range(n_boxes):
                gid, bid = box_g_ids[k], box_b_ids[k]; q = b_quat[bid]
                yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
                bc, bs = math.cos(yaw), math.sin(yaw)
                lx, ly = abs((cx-g_xpos[gid,0])*bc + (cy-g_xpos[gid,1])*bs), abs(-(cx-g_xpos[gid,0])*bs + (cy-g_xpos[gid,1])*bc)
                qx, qy = lx-g_size[gid,0], ly-g_size[gid,1]
                d_obj = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2) + min(max(qx, qy), 0.0)
                if d_obj < d_min: d_min = d_obj
            for k in range(len(ag_b_ids)):
                if k == exclude_idx: continue
                d_a = math.sqrt((cx-a_xpos[ag_b_ids[k],0])**2 + (cy-a_xpos[ag_b_ids[k],1])**2) - 0.45
                if d_a < d_min: d_min = d_a
            if d_min < epsilon: break
            t += max(d_min, epsilon)
            if t >= max_dist: t = max_dist; break
        res[i] = t
    return res

class VisibilityEngine:
    def __init__(self, m, d, layout_name="Maze"):
        self.m, self.d, self.max_dist = m, d, 20.0
        mujoco.mj_forward(m, d)
        self.cache_path = Path(f"hns_sightmap_v25_cache_{layout_name}.pkl")
        if not self.cache_path.exists():
            from generate_sightmap import generate_layout_cache
            generate_layout_cache(layout_name)
        with open(self.cache_path, "rb") as f: self.cache = pickle.load(f)
        
        self.meta_geoms = [] # 装飾パーツおよび視覚サイトの保持リスト
        self.walls_np = self._extract_static_walls()
        self._setup_indices()
        
        self.angles_deg = np.array([0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180], dtype=np.float32)
        self.base_cos, self.base_sin = np.cos(np.deg2rad(self.angles_deg)).astype(np.float32), np.sin(np.deg2rad(self.angles_deg)).astype(np.float32)
        self._geomid_out = np.zeros(1, dtype=np.int32)
        self.debug_print_objects()

    def debug_print_objects(self):
        print("\n" + "="*75)
        print("🔍 Lidar Optimization Audit (Decorative Parts Excluded)")
        print("="*75)
        print(f"Static Walls: {len(self.walls_np)}")
        print(f"Dynamic Box Geoms: {len(self.idx_box_geom)}")
        print(f"Active Agent Targets (_btm only): {len(self.idx_agent_body)}")
        print(f"Decorative Meta Geoms (Visual only): {len(self.meta_geoms)}")
        print("="*75 + "\n")

    def _extract_static_walls(self):
        walls = []
        for i in range(self.m.ngeom):
            name = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
            bid = self.m.geom_bodyid[i]
            if bid == 0:
                if any(k in name for k in ["wall", "maze", "border"]):
                    if "floor" in name or "ground" in name: continue
                    walls.append([self.d.geom_xpos[i][0], self.d.geom_xpos[i][1], self.m.geom_size[i][0], self.m.geom_size[i][1]])
                else:
                    self.meta_geoms.append({"name": name, "pos": self.d.geom_xpos[i][:2].copy(), "size": self.m.geom_size[i][:2].copy()})
        return np.array(walls, dtype=np.float32) if walls else np.empty((0,4), dtype=np.float32)

    def _setup_indices(self):
        box_g, box_b, ag_b = [], [], []
        # エージェント衝突の主ターゲット
        target_btm_names = ["seeker_btm", "hider1_btm", "hider2_btm"]
        
        for i in range(1, self.m.nbody):
            b_name = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "").lower()
            
            # 1. 動的オブジェクト
            if any(k in b_name for k in ["box", "ramp"]):
                for g in range(self.m.body_geomadr[i], self.m.body_geomadr[i] + self.m.body_geomnum[i]):
                    g_name = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
                    if "ramp" in b_name and "slope_surface" not in g_name:
                        self.meta_geoms.append({"name": g_name, "pos": self.d.geom_xpos[g][:2].copy(), "size": self.m.geom_size[g][:2].copy()})
                        continue
                    if self.m.geom_contype[g] > 0: box_g.append(g); box_b.append(i)
            
            # 2. エージェントの抽出 (btm geom を持つボディのみを計算対象とする)
            else:
                has_btm = False
                for g in range(self.m.body_geomadr[i], self.m.body_geomadr[i] + self.m.body_geomnum[i]):
                    g_name = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
                    if any(tn in g_name for tn in target_btm_names):
                        has_btm = True
                        break
                
                if has_btm:
                    ag_b.append(i)
                    # そのボディに属する他の geom (nose, tail, capsule) をメタリストへ送り、計算から除外
                    for g in range(self.m.body_geomadr[i], self.m.body_geomadr[i] + self.m.body_geomnum[i]):
                        g_name = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
                        if not any(tn in g_name for tn in target_btm_names):
                            self.meta_geoms.append({"name": g_name, "pos": self.d.geom_xpos[g][:2].copy(), "size": self.m.geom_size[g][:2].copy()})

        self.idx_box_geom, self.idx_box_body, self.idx_agent_body = np.array(box_g, dtype=np.int32), np.array(box_b, dtype=np.int32), np.array(ag_b, dtype=np.int32)

    def cast_lidar(self, pos, heading=0.0, mode=1, body_exclude=-1):
        h_cos, h_sin = math.cos(heading), math.sin(heading)
        if mode == 0:
            res = np.full(12, self.max_dist, dtype=np.float32); p_orig = np.array([pos[0], pos[1], 0.45], dtype=np.float64)
            for i in range(12):
                vx, vy = self.base_cos[i]*h_cos - self.base_sin[i]*h_sin, self.base_sin[i]*h_cos + self.base_cos[i]*h_sin
                d_hit = mujoco.mj_ray(self.m, self.d, p_orig, np.array([vx, vy, 0.0]), None, 1, int(body_exclude), self._geomid_out)
                if 0 <= d_hit <= self.max_dist: res[i] = d_hit
            return res
        
        exclude_idx = -1
        for k, bid in enumerate(self.idx_agent_body):
            if bid == body_exclude: exclude_idx = k
        g_xpos, g_size, b_quat, a_xpos = self.d.geom_xpos, self.m.geom_size, self.d.xquat, self.d.xpos
        if mode == 2 and "sdf_map" in self.cache:
            return _compute_sphere_tracing_core_jit(pos[0], pos[1], h_cos, h_sin, self.base_cos, self.base_sin, self.cache["sdf_map"], self.cache["bounds"], self.cache["cell_size"], self.cache["n_side"], g_xpos, g_size, self.idx_box_geom, b_quat, self.idx_box_body, a_xpos, self.idx_agent_body, exclude_idx, self.max_dist)
        return _compute_geometric_lidar_core_jit(pos[0], pos[1], h_cos, h_sin, self.base_cos, self.base_sin, self.walls_np, g_xpos, g_size, self.idx_box_geom, b_quat, self.idx_box_body, a_xpos, self.idx_agent_body, exclude_idx, self.max_dist)

    def is_visible(self, p1, p2, body_exclude=-1):
        diff = p2 - p1; dist = np.sqrt(np.sum(diff**2))
        if dist < 0.01: return True
        vx, vy = diff[0]/dist, diff[1]/dist
        for j in range(len(self.walls_np)):
            if _intersect_ray_aabb_robust_jit(p1[0], p1[1], vx, vy, self.walls_np[j,0], self.walls_np[j,1], self.walls_np[j,2], self.walls_np[j,3], dist) < dist - 0.01: return False
        for k in range(len(self.idx_box_geom)):
            gid, bid = self.idx_box_geom[k], self.idx_box_body[k]
            if bid == body_exclude: continue
            q = self.d.xquat[bid]; yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
            bc, bs = math.cos(yaw), math.sin(yaw)
            rx, ry = p1[0]-self.d.geom_xpos[gid,0], p1[1]-self.d.geom_xpos[gid,1]
            lx, ly = rx*bc+ry*bs, -rx*bs+ry*bc; lvx, lvy = vx*bc+vy*bs, -vx*bs+vy*bc
            if _intersect_ray_aabb_robust_jit(lx, ly, lvx, lvy, 0.0, 0.0, self.m.geom_size[gid,0], self.m.geom_size[gid,1], dist) < dist - 0.01: return False
        target_body_id = -1
        for k in range(len(self.idx_agent_body)):
            bid = self.idx_agent_body[k]
            if np.linalg.norm(p2 - self.d.xpos[bid][:2]) < 0.01: target_body_id = bid
        for k in range(len(self.idx_agent_body)):
            bid = self.idx_agent_body[k]
            if bid == body_exclude or bid == target_body_id: continue
            ax, ay = self.d.xpos[bid][0], self.d.xpos[bid][1]
            ox, oy = p1[0] - ax, p1[1] - ay
            b = 2.0 * (ox * vx + oy * vy); c = (ox*ox + oy*oy) - 0.2025; h = b*b - 4*c
            if h >= 0:
                sqrt_h = math.sqrt(h); t1, t2 = (-b - sqrt_h) / 2.0, (-b + sqrt_h) / 2.0
                t = t1 if t1 > 1e-6 else t2
                if 1e-6 < t < dist - 0.01: return False
        return True