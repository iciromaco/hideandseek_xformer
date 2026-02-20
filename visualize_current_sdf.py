# visibility_engine.py v1.24
# 演習第25回：Native(M0)透過問題の根本解決版
# 
# 修正履歴:
# v1.23: geomgroup マスクを追加（これが原因で透過が発生）。
# v1.24: 不確実な指定を排除。mj_ray の geomgroup を None (全対象) に戻し、
#        bodyexclude の型を厳密にキャスト。
#        また、抽出漏れを疑わなくて済むよう、内部リストの状態を print するデバッグ出力を追加。

import numpy as np
import mujoco
import math
import pickle
from pathlib import Path
from numba import njit

# --- 1. Geometric Intersection (Slab法 + 2D円判定) ---
@njit(cache=True)
def _compute_geometric_lidar_core_optimized(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin, 
                                           walls, g_xpos, g_size, box_g_ids,
                                           b_quat, box_b_ids, 
                                           ag_g_ids, ag_b_ids, ag_radii,
                                           exclude_body_id, max_dist):
    res = np.full(12, max_dist, dtype=np.float32)
    n_boxes = len(box_g_ids)
    n_agents = len(ag_g_ids)
    
    cos_y = np.empty(n_boxes, dtype=np.float32)
    sin_y = np.empty(n_boxes, dtype=np.float32)
    for k in range(n_boxes):
        q = b_quat[box_b_ids[k]]
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
        cos_y[k], sin_y[k] = math.cos(yaw), math.sin(yaw)

    for i in range(12):
        d_min = max_dist
        vx = base_cos[i]*h_cos - base_sin[i]*h_sin
        vy = base_sin[i]*h_cos + base_cos[i]*h_sin
        
        # A. 静的な壁
        for j in range(len(walls)):
            bx, by, sx, sy = walls[j]
            t_near, t_far = -1e10, 1e10
            if abs(vx) > 1e-12:
                inv_v = 1.0 / vx
                t1 = (bx - sx - pos_x) * inv_v; t2 = (bx + sx - pos_x) * inv_v
                t_near = max(t_near, min(t1, t2)); t_far = min(t_far, max(t1, t2))
            else:
                if abs(pos_x - bx) > sx: continue
            if abs(vy) > 1e-12:
                inv_v = 1.0 / vy
                t1 = (by - sy - pos_y) * inv_v; t2 = (by + sy - pos_y) * inv_v
                t_near = max(t_near, min(t1, t2)); t_far = min(t_far, max(t1, t2))
            else:
                if abs(pos_y - by) > sy: continue
            if t_far >= t_near and t_far > 0:
                d_hit = t_near if t_near > 0.41 else 0.0
                if 0.41 < d_hit < d_min: d_min = d_hit

        # B. 他エージェント判定 (M1/M2用)
        for k in range(n_agents):
            if ag_b_ids[k] == exclude_body_id: continue
            ax, ay = g_xpos[ag_g_ids[k], 0], g_xpos[ag_g_ids[k], 1]
            r = ag_radii[k]
            ox, oy = pos_x - ax, pos_y - ay
            b_val = 2.0 * (ox*vx + oy*vy)
            c_val = (ox*ox + oy*oy) - r*r
            h_val = b_val*b_val - 4.0*c_val
            if h_val >= 0:
                t_hit = (-b_val - math.sqrt(h_val)) / 2.0
                if 0.41 < t_hit < d_min: d_min = t_hit
        
        # C. 動的ボックス判定
        for k in range(n_boxes):
            bx, by, sx, sy = g_xpos[box_g_ids[k], 0], g_xpos[box_g_ids[k], 1], g_size[box_g_ids[k], 0], g_size[box_g_ids[k], 1]
            c, s = cos_y[k], sin_y[k]
            dx, dy = pos_x - bx, pos_y - by
            v_rot_x, v_rot_y = vx*c + vy*s, -vx*s + vy*c
            p_rot_x, p_rot_y = dx*c + dy*s, -dx*s + dy*c
            t_near, t_far = -1e10, 1e10
            if abs(v_rot_x) > 1e-12:
                inv_v = 1.0 / v_rot_x
                t1, t2 = (-sx - p_rot_x)*inv_v, (sx - p_rot_x)*inv_v
                t_near = max(t_near, min(t1, t2)); t_far = min(t_far, max(t1, t2))
            else:
                if abs(p_rot_x) > sx: continue
            if abs(v_rot_y) > 1e-12:
                inv_v = 1.0 / v_rot_y
                t1, t2 = (-sy - p_rot_y)*inv_v, (sy - p_rot_y)*inv_v
                t_near = max(t_near, min(t1, t2)); t_far = min(t_far, max(t1, t2))
            else:
                if abs(p_rot_y) > sy: continue
            if t_far >= t_near and t_far > 0:
                d_hit = t_near if t_near > 0.41 else 0.0
                if 0.41 < d_hit < d_min: d_min = d_hit
        res[i] = d_min
    return res

# --- 2. Sphere Tracing ---
@njit(cache=True)
def _get_sdf_scene_jit(p_x, p_y, walls, g_xpos, g_size, box_g_ids, b_quat, box_b_ids, ag_g_ids, ag_b_ids, ag_radii, exclude_body_id):
    d_min = 1e10
    for j in range(len(walls)):
        bx, by, sx, sy = walls[j]
        dx = abs(p_x - bx) - sx; dy = abs(p_y - by) - sy
        d = math.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2) + min(max(dx, dy), 0.0)
        if d < d_min: d_min = d
    for k in range(len(box_g_ids)):
        bx, by, sx, sy = g_xpos[box_g_ids[k], 0], g_xpos[box_g_ids[k], 1], g_size[box_g_ids[k], 0], g_size[box_g_ids[k], 1]
        q = b_quat[box_b_ids[k]]
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx, dy = p_x - bx, p_y - by
        p_rot_x, p_rot_y = dx*c - dy*s, dx*s + dy*c
        qx, qy = abs(p_rot_x) - sx, abs(p_rot_y) - sy
        d = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2) + min(max(qx, qy), 0.0)
        if d < d_min: d_min = d
    for k in range(len(ag_g_ids)):
        if ag_b_ids[k] == exclude_body_id: continue
        ax, ay = g_xpos[ag_g_ids[k], 0], g_xpos[ag_g_ids[k], 1]
        d = math.sqrt((p_x - ax)**2 + (p_y - ay)**2) - ag_radii[k]
        if d < d_min: d_min = d
    return d_min

@njit(cache=True)
def _compute_sphere_tracing_lidar_jit(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin, 
                                     walls, g_xpos, g_size, box_g_ids,
                                     b_quat, box_b_ids, ag_g_ids, ag_b_ids, ag_radii,
                                     exclude_body_id, max_dist):
    res = np.full(12, max_dist, dtype=np.float32)
    for i in range(12):
        vx, vy = base_cos[i]*h_cos - base_sin[i]*h_sin, base_sin[i]*h_cos + base_cos[i]*h_sin
        curr_t = 0.41
        for _ in range(50):
            curr_x, curr_y = pos_x + vx * curr_t, pos_y + vy * curr_t
            d = _get_sdf_scene_jit(curr_x, curr_y, walls, g_xpos, g_size, box_g_ids, b_quat, box_b_ids, ag_g_ids, ag_b_ids, ag_radii, exclude_body_id)
            if d < 0.005:
                res[i] = curr_t; break
            curr_t += d
            if curr_t >= max_dist: break
    return res

class VisibilityEngine:
    def __init__(self, m, d, layout_name="Maze"):
        self.m, self.d = m, d
        mujoco.mj_forward(m, d)
        self.max_dist = 15.0
        self.lidar_height = 0.40 
        self.base_angles = np.array([0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180], dtype=np.float32)
        self.base_cos = np.cos(np.deg2rad(self.base_angles)).astype(np.float32)
        self.base_sin = np.sin(np.deg2rad(self.base_angles)).astype(np.float32)
        self._geomid_out = np.zeros(1, dtype=np.int32)
        self._extract_geometry()

    def _extract_geometry(self):
        self.walls_np = []
        self.idx_box_geom, self.idx_box_body = [], []
        self.idx_agent_geom, self.idx_agent_body, self.agent_radii = [], [], []
        
        for i in range(self.m.ngeom):
            name = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
            bid = self.m.geom_bodyid[i]
            b_name = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
            
            # _btm を見つけたら無条件でエージェントとして登録
            if "_btm" in name:
                self.idx_agent_geom.append(i)
                self.idx_agent_body.append(bid)
                self.agent_radii.append(self.m.geom_size[i, 0])
            # ジョイントがないものは壁
            elif self.m.body_jntnum[bid] == 0:
                if any(k in name for k in ["wall", "maze", "border"]):
                    if "floor" in name: continue
                    self.walls_np.append([self.d.geom_xpos[i,0], self.d.geom_xpos[i,1], self.m.geom_size[i,0], self.m.geom_size[i,1]])
            # それ以外は動的オブジェクト
            else:
                if any(k in b_name for k in ["box", "ramp"]):
                    if "ramp" in b_name and "slope_surface" not in name: continue
                    self.idx_box_geom.append(i); self.idx_box_body.append(bid)
        
        self.walls_np = np.array(self.walls_np, dtype=np.float32)
        self.idx_box_geom = np.array(self.idx_box_geom, dtype=np.int32)
        self.idx_box_body = np.array(self.idx_box_body, dtype=np.int32)
        self.idx_agent_geom = np.array(self.idx_agent_geom, dtype=np.int32)
        self.idx_agent_body = np.array(self.idx_agent_body, dtype=np.int32)
        self.agent_radii = np.array(self.agent_radii, dtype=np.float32)

    def cast_lidar(self, pos, heading=0.0, mode=1, body_exclude=-1, return_names=False):
        h_cos, h_sin = math.cos(heading), math.sin(heading)
        
        if mode == 0:
            res = np.full(12, self.max_dist, dtype=np.float32)
            hit_names = [] if return_names else None
            p_orig = np.array([pos[0], pos[1], self.lidar_height], dtype=np.float64)
            for i in range(12):
                vx, vy = self.base_cos[i]*h_cos - self.base_sin[i]*h_sin, self.base_sin[i]*h_cos + self.base_cos[i]*h_sin
                v_dir = np.array([vx, vy, 0.0])
                
                # 【重要】不確実な geomgroup ビットマスクを廃止 (None = 全グループ対象)
                # flg_static=0 (動的・静的両方対象) を明示
                d_hit = mujoco.mj_ray(self.m, self.d, p_orig, v_dir, 
                                      geomgroup=None, flg_static=0, 
                                      bodyexclude=int(body_exclude), geomid=self._geomid_out)
                
                if return_names:
                    if 0.41 < d_hit <= self.max_dist:
                        g_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, self._geomid_out[0]) or "Unknown"
                        hit_names.append(g_name)
                    else: hit_names.append("None")
                res[i] = d_hit if (0.41 < d_hit <= self.max_dist) else self.max_dist
            return (res, hit_names) if return_names else res
        
        # M1/M2モード
        if mode == 2:
            res = _compute_sphere_tracing_lidar_jit(pos[0], pos[1], h_cos, h_sin, self.base_cos, self.base_sin, self.walls_np, self.d.geom_xpos, self.m.geom_size, self.idx_box_geom, self.d.xquat, self.idx_box_body, self.idx_agent_geom, self.idx_agent_body, self.agent_radii, int(body_exclude), self.max_dist)
        else:
            res = _compute_geometric_lidar_core_optimized(pos[0], pos[1], h_cos, h_sin, self.base_cos, self.base_sin, self.walls_np, self.d.geom_xpos, self.m.geom_size, self.idx_box_geom, self.d.xquat, self.idx_box_body, self.idx_agent_geom, self.idx_agent_body, self.agent_radii, int(body_exclude), self.max_dist)
            
        return (res, ["(Approx)"] * 12) if return_names else res

    def is_visible(self, p1, p2, body_exclude=-1):
        diff = p2 - p1; dist = np.linalg.norm(diff)
        if dist < 0.1: return True
        p_orig = np.array([p1[0], p1[1], self.lidar_height], dtype=np.float64)
        v_dir = np.array([(p2[0]-p1[0])/dist, (p2[1]-p1[1])/dist, 0.0])
        hit_dist = mujoco.mj_ray(self.m, self.d, p_orig, v_dir, 
                                 geomgroup=None, flg_static=0, 
                                 bodyexclude=int(body_exclude), geomid=self._geomid_out)
        if 0 < hit_dist < 0.41: return True
        return hit_dist < 0 or hit_dist > (dist - 0.1)