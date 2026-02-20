# visibility_engine.py v1.67
# 演習第25回：Lidar 全モード完全同期 ＆ 高速・高精度統合（決定版）
# 
# 修正履歴:
# v1.66: Slab法復元。
# v1.67: ユーザー報告の Native 不全・速度低下・不正確さを解消。
#        1. mj_ray の flg_static=0 により動的な箱やハイダーも Native で検知。
#        2. Geometric(M1) にエージェント・ボックスの交差判定を完全実装。
#        3. Sphere Tracing(M2) の配列生成・座標回転を極限まで最適化。
#        4. 全モードで自己衝突除外 (0.41m) を論理的に統一。

import numpy as np
import mujoco
import math
from numba import njit

@njit(cache=True)
def _get_sdf_core(p, walls_xpos, walls_size, ag_pos, ag_radii, box_pos, box_size, box_cos, box_sin, exclude_idx):
    """SDF計算：配列生成を伴わない純粋演算"""
    d_min = 15.0
    # 1. 壁 (AABB)
    for i in range(len(walls_xpos)):
        bx, by, sx, sy = walls_xpos[i,0], walls_xpos[i,1], walls_size[i,0], walls_size[i,1]
        qx, qy = abs(p[0]-bx)-sx, abs(p[1]-by)-sy
        d = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2) + min(max(qx, qy), 0.0)
        if d < d_min: d_min = d
    # 2. 他エージェント (Circle)
    for i in range(len(ag_pos)):
        if i == exclude_idx: continue
        d = math.sqrt((p[0]-ag_pos[i,0])**2 + (p[1]-ag_pos[i,1])**2) - ag_radii[i]
        if d < d_min: d_min = d
    # 3. 箱 (OBB)
    for i in range(len(box_pos)):
        c, s = box_cos[i], box_sin[i]
        dx, dy = p[0]-box_pos[i,0], p[1]-box_pos[i,1]
        rx, ry = dx*c + dy*s, -dx*s + dy*c
        qx, qy = abs(rx)-box_size[i,0], abs(ry)-box_size[i,1]
        d = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2) + min(max(qx, qy), 0.0)
        if d < d_min: d_min = d
    return d_min

@njit(cache=True)
def _compute_lidar_optimized(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin, 
                             walls_xpos, walls_size, ag_pos, ag_radii, ag_b_ids, 
                             box_pos, box_size, box_cos, box_sin, 
                             mode, exclude_body_id, max_dist):
    """3モードを一本化した超高速 JIT コア"""
    res = np.full(12, max_dist, dtype=np.float32)
    ag_idx = -1
    for i in range(len(ag_b_ids)):
        if ag_b_ids[i] == exclude_body_id: ag_idx = i; break

    for i in range(12):
        vx = base_cos[i]*h_cos - base_sin[i]*h_sin
        vy = base_sin[i]*h_cos + base_cos[i]*h_sin
        
        if mode == 1: # Geometric Intersection (Slab & Circle)
            d_min = max_dist
            # A. 壁 (AABB)
            for j in range(len(walls_xpos)):
                bx, by, sx, sy = walls_xpos[j,0], walls_xpos[j,1], walls_size[j,0], walls_size[j,1]
                t_near, t_far = -1e10, 1e10
                if abs(vx) > 1e-12:
                    inv_v = 1.0 / vx
                    t1 = (bx - sx - pos_x) * inv_v; t2 = (bx + sx - pos_x) * inv_v
                    t_near = max(t_near, min(t1, t2)); t_far = min(t_far, max(t1, t2))
                elif abs(pos_x - bx) > sx: continue
                if abs(vy) > 1e-12:
                    inv_v = 1.0 / vy
                    t1 = (by - sy - pos_y) * inv_v; t2 = (by + sy - pos_y) * inv_v
                    t_near = max(t_near, min(t1, t2)); t_far = min(t_far, max(t1, t2))
                elif abs(pos_y - by) > sy: continue
                if t_far >= t_near and t_far > 0:
                    d_hit = t_near if t_near > 0 else t_far
                    if 0.41 < d_hit < d_min: d_min = d_hit
            # B. エージェント (Circle)
            for k in range(len(ag_pos)):
                if ag_b_ids[k] == exclude_body_id: continue
                ox, oy = pos_x - ag_pos[k,0], pos_y - ag_pos[k,1]
                b = 2.0*(ox*vx + oy*vy); c = (ox*ox + oy*oy) - ag_radii[k]**2
                h = b*b - 4.0*c
                if h >= 0:
                    t = (-b - math.sqrt(h))/2.0
                    if 0.41 < t < d_min: d_min = t
            # C. ボックス (OBB Slab)
            for k in range(len(box_pos)):
                c, s = box_cos[k], box_sin[k]
                dx, dy = pos_x - box_pos[k,0], pos_y - box_pos[k,1]
                v_rx, v_ry = vx*c + vy*s, -vx*s + vy*c
                p_rx, p_ry = dx*c + dy*s, -dx*s + dy*c
                t_near, t_far = -1e10, 1e10
                sx, sy = box_size[k,0], box_size[k,1]
                if abs(v_rx) > 1e-12:
                    inv_v = 1.0 / v_rx; t1 = (-sx - p_rx) * inv_v; t2 = (sx - p_rx) * inv_v
                    t_near = max(t_near, min(t1, t2)); t_far = min(t_far, max(t1, t2))
                elif abs(p_rx) > sx: continue
                if abs(v_ry) > 1e-12:
                    inv_v = 1.0 / v_ry; t1 = (-sy - p_ry) * inv_v; t2 = (sy - p_ry) * inv_v
                    t_near = max(t_near, min(t1, t2)); t_far = min(t_far, max(t1, t2))
                elif abs(p_ry) > sy: continue
                if t_far >= t_near and t_far > 0:
                    d_hit = t_near if t_near > 0 else t_far
                    if 0.41 < d_hit < d_min: d_min = d_hit
            res[i] = d_min

        elif mode == 2: # Sphere Tracing
            curr_t = 0.41
            for _ in range(35):
                p_step = np.array([pos_x + vx*curr_t, pos_y + vy*curr_t])
                d = _get_sdf_core(p_step, walls_xpos, walls_size, ag_pos, ag_radii, box_pos, box_size, box_cos, box_sin, ag_idx)
                if d < 0.005: res[i] = curr_t; break
                curr_t += d
                if curr_t >= max_dist: break
    return res

class VisibilityEngine:
    def __init__(self, m, d):
        self.m, self.d = m, d; self.max_dist = 15.0
        self.base_angles = np.array([0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180], dtype=np.float32)
        self.base_cos, self.base_sin = np.cos(np.deg2rad(self.base_angles)), np.sin(np.deg2rad(self.base_angles))
        self._geomid_out = np.zeros(1, dtype=np.int32)
        self.group_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8) 
        self._extract_indices()

    def _extract_indices(self):
        self.idx_wall_geom, self.idx_box_geom, self.idx_box_body = [], [], []
        self.idx_agent_geom, self.idx_agent_body, self.agent_radii = [], [], []
        for i in range(self.m.ngeom):
            if self.m.geom_group[i] != 0: continue
            bid = self.m.geom_bodyid[i]
            name = (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
            if "_btm" in name:
                self.idx_agent_geom.append(i); self.idx_agent_body.append(bid); self.agent_radii.append(self.m.geom_size[i,0])
            elif self.m.body_jntnum[bid] == 0:
                if any(k in name for k in ["wall", "maze", "border"]) and "floor" not in name: self.idx_wall_geom.append(i)
            else:
                self.idx_box_geom.append(i); self.idx_box_body.append(bid)
        self.idx_wall_geom = np.array(self.idx_wall_geom, dtype=np.int32)
        self.idx_box_geom = np.array(self.idx_box_geom, dtype=np.int32); self.idx_box_body = np.array(self.idx_box_body, dtype=np.int32)
        self.idx_agent_geom = np.array(self.idx_agent_geom, dtype=np.int32); self.idx_agent_body = np.array(self.idx_agent_body, dtype=np.int32)
        self.agent_radii = np.array(self.agent_radii, dtype=np.float32)

    def cast_lidar(self, pos, heading=0.0, mode=1, body_exclude=-1):
        h_c, h_s = math.cos(heading), math.sin(heading)
        if mode == 0:
            res = np.full(12, self.max_dist, dtype=np.float32); p_orig = np.array([pos[0], pos[1], 0.4], dtype=np.float64)
            for i in range(12):
                vx, vy = self.base_cos[i]*h_c - self.base_sin[i]*h_s, self.base_sin[i]*h_c + self.base_cos[i]*h_s
                # flg_static=0 に是正。これで全ジオメトリを検知。
                d_hit = mujoco.mj_ray(self.m, self.d, p_orig, np.array([vx, vy, 0.0]), None, 0, int(body_exclude), self._geomid_out)
                res[i] = d_hit if d_hit > 0.41 else self.max_dist
            return res
        
        box_cos = np.empty(len(self.idx_box_body), dtype=np.float32)
        box_sin = np.empty(len(self.idx_box_body), dtype=np.float32)
        for k, bid in enumerate(self.idx_box_body):
            q = self.d.xquat[bid]
            yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
            box_cos[k], box_sin[k] = math.cos(yaw), math.sin(yaw)

        return _compute_lidar_optimized(
            pos[0], pos[1], h_c, h_s, self.base_cos, self.base_sin,
            self.d.geom_xpos[self.idx_wall_geom], self.m.geom_size[self.idx_wall_geom],
            self.d.geom_xpos[self.idx_agent_geom, :2], self.agent_radii, self.idx_agent_body,
            self.d.geom_xpos[self.idx_box_geom, :2], self.m.geom_size[self.idx_box_geom, :2],
            box_cos, box_sin, mode, int(body_exclude), self.max_dist
        )