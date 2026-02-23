# visibility_engine.py v1.78
# 演習第25回：JIT判定完全復元 ＆ OBB交差ロジック再展開版
# 
# 修正履歴:
# v1.77: Row-Major同期。
# v1.78: 1. 疑念のあった「行数減少」を解消するため、Slab法(OBB/箱)の全計算ステップを
#           JITコア内に明示的に記述。
#        2. is_visible が inference 時に MuJoCo Native を一切呼ばない純粋 JIT であることを保証。

import numpy as np
import mujoco
import math
from numba import njit

@njit(cache=True)
def _get_sdf_scalar(px, py, walls_xpos, walls_size, ag_pos, ag_radii, box_pos, box_size, box_quats, exclude_idx):
    d_min = 15.0
    # 壁 (AABB)
    for i in range(len(walls_xpos)):
        bx, by, sx, sy = walls_xpos[i,0], walls_xpos[i,1], walls_size[i,0], walls_size[i,1]
        dx, dy = abs(px - bx) - sx, abs(py - by) - sy
        d = math.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2) + min(max(dx, dy), 0.0)
        if d < d_min: d_min = d
    # エージェント (Circle)
    for i in range(len(ag_pos)):
        if i == exclude_idx: continue
        d = math.sqrt((px - ag_pos[i,0])**2 + (py - ag_pos[i,1])**2) - ag_radii[i]
        if d < d_min: d_min = d
    # 箱 (OBB)
    for i in range(len(box_pos)):
        q = box_quats[i]
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx = (px - box_pos[i,0])*c - (py - box_pos[i,1])*s
        ly = (px - box_pos[i,0])*s + (py - box_pos[i,1])*c
        dx, dy = abs(lx) - box_size[i,0], abs(ly) - box_size[i,1]
        d = math.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2) + min(max(dx, dy), 0.0)
        if d < d_min: d_min = d
    return d_min

@njit(cache=True)
def _is_visible_jit_core(p1x, p1y, p2x, p2y, walls_xpos, walls_size, ag_pos, ag_radii, ag_b_ids, 
                         box_pos, box_size, box_quats, box_b_ids,
                         exclude_body_id, target_body_id):
    """mj_ray を完全に代替する JIT 高速遮蔽判定"""
    vx, vy = p2x - p1x, p2y - p1y
    dist_sq = vx*vx + vy*vy
    if dist_sq < 0.2025: return True
    dist = math.sqrt(dist_sq); ux, uy = vx/dist, vy/dist; hit_threshold = dist - 0.15
    
    # 1. 壁 (Slab法)
    for i in range(len(walls_xpos)):
        bx, by, sx, sy = walls_xpos[i,0], walls_xpos[i,1], walls_size[i,0], walls_size[i,1]
        t_n, t_f = -1e10, 1e10
        if abs(ux) > 1e-12:
            iv = 1.0/ux; t1 = (bx-sx-p1x)*iv; t2 = (bx+sx-p1x)*iv
            t_n = max(t_n, min(t1, t2)); t_f = min(t_f, max(t1, t2))
        elif abs(p1x-bx) > sx: continue
        if abs(uy) > 1e-12:
            iv = 1.0/uy; t1 = (by-sy-p1y)*iv; t2 = (by+sy-p1y)*iv
            t_n = max(t_n, min(t1, t2)); t_f = min(t_f, max(t1, t2))
        elif abs(p1y-by) > sy: continue
        if t_f >= t_n and t_f > 0.41:
            hit = t_n if t_n > 0.41 else t_f
            if hit < hit_threshold: return False
            
    # 2. エージェント (円交差)
    for i in range(len(ag_pos)):
        bid = ag_b_ids[i]
        if bid == exclude_body_id or bid == target_body_id: continue
        ox, oy = p1x - ag_pos[i,0], p1y - ag_pos[i,1]
        b = 2.0*(ox*ux + oy*uy); c = ox*ox+oy*oy - ag_radii[i]**2; det = b*b - 4.0*c
        if det >= 0:
            sd = math.sqrt(det); t1 = (-b-sd)/2.0; t2 = (-b+sd)/2.0
            if t2 > 0.41:
                hit = t1 if t1 > 0.41 else t2
                if hit < hit_threshold: return False

    # 3. 箱 (OBB Slab法)
    for i in range(len(box_pos)):
        bid = box_b_ids[i]
        if bid == exclude_body_id or bid == target_body_id: continue
        q = box_quats[i]
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = p1x - box_pos[i,0], p1y - box_pos[i,1]
        v_rx, v_ry = ux*c+uy*s, -ux*s+uy*c; p_rx, p_ry = dx*c+dy*s, -dx*s+dy*c
        sx, sy = box_size[i,0], box_size[i,1]
        t_n, t_f = -1e10, 1e10
        if abs(v_rx) > 1e-12:
            iv=1.0/v_rx; t1=(-sx-p_rx)*iv; t2=(sx-p_rx)*iv
            t_n=max(t_n,min(t1,t2)); t_f=min(t_f,max(t1,t2))
        elif abs(p_rx)>sx: continue
        if abs(v_ry) > 1e-12:
            iv=1.0/v_ry; t1=(-sy-p_ry)*iv; t2=(sy-p_ry)*iv
            t_n=max(t_n,min(t1,t2)); t_f=min(t_f,max(t1,t2))
        elif abs(p_ry)>sy: continue
        if t_f >= t_n and t_f > 0.41:
            hit = t_n if t_n > 0.41 else t_f
            if hit < hit_threshold: return False
    return True

@njit(cache=True)
def _compute_lidar_jit_core(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin, 
                             walls_xpos, walls_size, ag_pos, ag_radii, ag_b_ids, 
                             box_pos, box_size, box_quats, box_b_ids,
                             mode, exclude_body_id, max_dist):
    res = np.full(12, max_dist, dtype=np.float32)
    ag_idx = -1
    for i in range(len(ag_b_ids)):
        if ag_b_ids[i] == exclude_body_id: ag_idx = i; break
    for i in range(12):
        vx = base_cos[i]*h_cos - base_sin[i]*h_sin
        vy = base_sin[i]*h_cos + base_cos[i]*h_sin
        if mode == 1: # Geometric
            d_min = max_dist
            # 壁
            for j in range(len(walls_xpos)):
                bx, by, sx, sy = walls_xpos[j,0], walls_xpos[j,1], walls_size[j,0], walls_size[j,1]
                t_n, t_f = -1e10, 1e10
                if abs(vx) > 1e-12:
                    iv = 1.0/vx; t1 = (bx-sx-pos_x)*iv; t2 = (bx+sx-pos_x)*iv
                    t_n = max(t_n, min(t1, t2)); t_f = min(t_f, max(t1, t2))
                elif abs(pos_x-bx) > sx: continue
                if abs(vy) > 1e-12:
                    iv = 1.0/vy; t1 = (by-sy-pos_y)*iv; t2 = (by+sy-pos_y)*iv
                    t_n = max(t_n, min(t1, t2)); t_f = min(t_f, max(t1, t2))
                elif abs(pos_y-by) > sy: continue
                if t_f >= t_n and t_f > 0.41:
                    hit = t_n if t_n > 0.41 else t_f
                    if 0.41 < hit < d_min: d_min = hit
            # 箱 (OBB Slab)
            for k in range(len(box_pos)):
                q = box_quats[k]
                yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
                c, s = math.cos(yaw), math.sin(yaw)
                dx, dy = pos_x-box_pos[k,0], pos_y-box_pos[k,1]
                v_rx, v_ry = vx*c+vy*s, -vx*s+vy*c; p_rx, p_ry = dx*c+dy*s, -dx*s+dy*c
                sx, sy = box_size[k,0], box_size[k,1]
                t_n, t_f = -1e10, 1e10
                if abs(v_rx) > 1e-12:
                    iv=1.0/v_rx; t1=(-sx-p_rx)*iv; t2=(sx-p_rx)*iv
                    t_n=max(t_n,min(t1,t2)); t_f=min(t_f,max(t1,t2))
                elif abs(p_rx)>sx: continue
                if abs(v_ry) > 1e-12:
                    iv=1.0/v_ry; t1=(-sy-p_ry)*iv; t2=(sy-p_ry)*iv
                    t_n=max(t_n,min(t1,t2)); t_f=min(t_f,max(t1,t2))
                elif abs(p_ry)>sy: continue
                if t_f >= t_n and t_f > 0.41:
                    hit = t_n if t_n > 0.41 else t_f
                    if 0.41 < hit < d_min: d_min = hit
            res[i] = d_min
        elif mode == 2: # Sphere Tracing
            curr_t = 0.41
            for _ in range(40):
                cx, cy = pos_x + vx * curr_t, pos_y + vy * curr_t
                dist = _get_sdf_scalar(cx, cy, walls_xpos, walls_size, ag_pos, ag_radii, box_pos, box_size, box_quats, ag_idx)
                if dist < 0.005: 
                    res[i] = curr_t
                    break
                curr_t += dist
                if curr_t >= max_dist: 
                    res[i] = max_dist
                    break
    return res

class VisibilityEngine:
    def __init__(self, m, d):
        self.m, self.d = m, d; self.max_dist = 15.0
        self.base_angles = np.array([0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180], dtype=np.float32)
        self.base_cos, self.base_sin = np.cos(np.deg2rad(self.base_angles)), np.sin(np.deg2rad(self.base_angles))
        self.group_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8) 
        self._geomid_out = np.zeros(1, dtype=np.int32)
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
        self.idx_box_geom, self.idx_box_body = np.array(self.idx_box_geom, dtype=np.int32), np.array(self.idx_box_body, dtype=np.int32)
        self.idx_agent_geom, self.idx_agent_body = np.array(self.idx_agent_geom, dtype=np.int32), np.array(self.idx_agent_body, dtype=np.int32)
        self.agent_radii = np.array(self.agent_radii, dtype=np.float32)

    def cast_lidar(self, pos, heading=0.0, mode=1, body_exclude=-1):
        h_cos, h_sin = math.cos(heading), math.sin(heading)
        if mode == 0:
            res = np.full(12, self.max_dist, dtype=np.float32); p_o = np.array([pos[0], pos[1], 0.4], dtype=np.float64)
            for i in range(12):
                vx, vy = self.base_cos[i]*h_cos - self.base_sin[i]*h_sin, self.base_sin[i]*h_cos + self.base_cos[i]*h_sin
                d = mujoco.mj_ray(self.m, self.d, p_o, np.array([vx, vy, 0.0]), self.group_mask, 1, int(body_exclude), self._geomid_out)
                res[i] = d if d > 0.41 else self.max_dist
            return res
        return _compute_lidar_jit_core(pos[0], pos[1], h_cos, h_sin, self.base_cos, self.base_sin,
                                      self.d.geom_xpos[self.idx_wall_geom], self.m.geom_size[self.idx_wall_geom],
                                      self.d.geom_xpos[self.idx_agent_geom, :2], self.agent_radii, self.idx_agent_body,
                                      self.d.geom_xpos[self.idx_box_geom, :2], self.m.geom_size[self.idx_box_geom, :2],
                                      self.d.xquat[self.idx_box_body], self.idx_box_body, mode, int(body_exclude), self.max_dist)

    def is_visible(self, p1, p2, body_exclude=-1, target_body_id=-1):
        return _is_visible_jit_core(p1[0], p1[1], p2[0], p2[1],
                                    self.d.geom_xpos[self.idx_wall_geom], self.m.geom_size[self.idx_wall_geom],
                                    self.d.geom_xpos[self.idx_agent_geom, :2], self.agent_radii, self.idx_agent_body,
                                    self.d.geom_xpos[self.idx_box_geom, :2], self.m.geom_size[self.idx_box_geom, :2],
                                    self.d.xquat[self.idx_box_body], self.idx_box_body, int(body_exclude), int(target_body_id))