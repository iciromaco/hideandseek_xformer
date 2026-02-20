# visibility_engine.py v1.71
# 演習第25回：真の SDF Ray Marching ＆ スカラー演算極限最適化版
# 
# 修正履歴:
# v1.70: 速度低下解消。
# v1.71: ユーザーの指摘（M2の理論的矛盾）を解消。
#        1. ループ内での np.array 生成を完全に排除。純粋スカラー演算へ。
#        2. _get_sdf_core をスカラー引数 (px, py) に変更し、メモリ確保をゼロに。
#        3. M1(Geometric) と M2(Sphere) の双方に最高効率の JIT パスを適用。

import numpy as np
import mujoco
import math
from numba import njit

@njit(cache=True)
def _get_sdf_scalar(px, py, walls_xpos, walls_size, ag_pos, ag_radii, box_pos, box_size, box_quats, exclude_idx):
    """
    SDF計算の真のコア：
    配列生成を一切行わず、スタック上のスカラー変数のみで最短距離場を走査。
    """
    d_min = 15.0
    
    # 1. 静的な壁 (AABB)
    for i in range(len(walls_xpos)):
        bx, by = walls_xpos[i,0], walls_xpos[i,1]
        sx, sy = walls_size[i,0], walls_size[i,1]
        # 点 p から AABB 表面への最短距離
        dx = abs(px - bx) - sx
        dy = abs(py - by) - sy
        d = math.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2) + min(max(dx, dy), 0.0)
        if d < d_min: d_min = d
        
    # 2. 他エージェント (Circle)
    for i in range(len(ag_pos)):
        if i == exclude_idx: continue
        d = math.sqrt((px - ag_pos[i,0])**2 + (py - ag_pos[i,1])**2) - ag_radii[i]
        if d < d_min: d_min = d
        
    # 3. 移動する箱 (OBB)
    for i in range(len(box_pos)):
        q = box_quats[i]
        # クォータニオンから直接 Yaw の sin/cos を求める (JITインライン)
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
        c, s = math.cos(-yaw), math.sin(-yaw)
        
        # 相対座標をローカル回転
        rx = px - box_pos[i,0]
        ry = py - box_pos[i,1]
        lx = rx * c - ry * s
        ly = rx * s + ry * c
        
        # ローカル空間での AABB 距離
        dx = abs(lx) - box_size[i,0]
        dy = abs(ly) - box_size[i,1]
        d = math.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2) + min(max(dx, dy), 0.0)
        if d < d_min: d_min = d
        
    return d_min

@njit(cache=True)
def _compute_lidar_jit_core(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin, 
                             walls_xpos, walls_size, ag_pos, ag_radii, ag_b_ids, 
                             box_pos, box_size, box_quats, 
                             mode, exclude_body_id, max_dist):
    """
    3モード統合計測コア v1.71
    """
    res = np.full(12, max_dist, dtype=np.float32)
    ag_idx = -1
    for i in range(len(ag_b_ids)):
        if ag_b_ids[i] == exclude_body_id: ag_idx = i; break

    for i in range(12):
        vx = base_cos[i]*h_cos - base_sin[i]*h_sin
        vy = base_sin[i]*h_cos + base_cos[i]*h_sin
        
        if mode == 1: # Geometric (閉形式交差判定)
            d_min = max_dist
            # A. 壁 (Slab法)
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
            # B. 他エージェント (円交差)
            for k in range(len(ag_pos)):
                if ag_b_ids[k] == exclude_body_id: continue
                ox, oy = pos_x-ag_pos[k,0], pos_y-ag_pos[k,1]
                b = 2.0*(ox*vx + oy*vy); c = ox*ox+oy*oy - ag_radii[k]**2; det = b*b - 4.0*c
                if det >= 0:
                    sd = math.sqrt(det); t1 = (-b-sd)/2.0; t2 = (-b+sd)/2.0
                    if t2 > 0.41:
                        hit = t1 if t1 > 0.41 else t2
                        if 0.41 < hit < d_min: d_min = hit
            # C. 箱 (OBB Slab)
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

        elif mode == 2: # 真の Sphere Tracing (SDFベース レイマーチング)
            curr_t = 0.41
            for _ in range(40):
                # レイ上の現在点座標
                cx, cy = pos_x + vx * curr_t, pos_y + vy * curr_t
                # 【修正】スカラーを渡し、NumPy配列生成オーバーヘッドを完全排除
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
        self.idx_box_geom, self.idx_box_body = np.array(self.idx_box_geom, dtype=np.int32), np.array(self.idx_box_body, dtype=np.int32)
        self.idx_agent_geom, self.idx_agent_body = np.array(self.idx_agent_geom, dtype=np.int32), np.array(self.idx_agent_body, dtype=np.int32)
        self.agent_radii = np.array(self.agent_radii, dtype=np.float32)

    def cast_lidar(self, pos, heading=0.0, mode=1, body_exclude=-1):
        h_c, h_s = math.cos(heading), math.sin(heading)
        if mode == 0:
            res = np.full(12, self.max_dist, dtype=np.float32); p_o = np.array([pos[0], pos[1], 0.4], dtype=np.float64)
            for i in range(12):
                vx, vy = self.base_cos[i]*h_c - self.base_sin[i]*h_s, self.base_sin[i]*h_c + self.base_cos[i]*h_s
                d = mujoco.mj_ray(self.m, self.d, p_o, np.array([vx, vy, 0.0]), self.group_mask, 1, int(body_exclude), self._geomid_out)
                res[i] = d if d > 0.41 else self.max_dist
            return res
        
        # MuJoCo データ配列を直接スライスして渡す (オーバーヘッド最小化)
        return _compute_lidar_jit_core(
            pos[0], pos[1], h_c, h_s, self.base_cos, self.base_sin,
            self.d.geom_xpos[self.idx_wall_geom], self.m.geom_size[self.idx_wall_geom],
            self.d.geom_xpos[self.idx_agent_geom, :2], self.agent_radii, self.idx_agent_body,
            self.d.geom_xpos[self.idx_box_geom, :2], self.m.geom_size[self.idx_box_geom, :2],
            self.d.xquat[self.idx_box_body], mode, int(body_exclude), self.max_dist
        )

    def is_visible(self, p1, p2, body_exclude=-1):
        """2点間の可視性判定 (内部で mj_ray を使用)"""
        diff = p2 - p1; dist = np.linalg.norm(diff)
        if dist < 0.45: return True
        p_orig = np.array([p1[0], p1[1], 0.4], dtype=np.float64)
        v_dir = np.array([diff[0]/dist, diff[1]/dist, 0.0])
        hit = mujoco.mj_ray(self.m, self.d, p_orig, v_dir, self.group_mask, 1, int(body_exclude), self._geomid_out)
        return hit < 0 or hit > (dist - 0.1)