# visibility_engine.py v2.00
# 演習第26回：完全数学実装 ＆ 全オブジェクト遮蔽判定（Lidar/is_visible同期）極限最適化版
# 
# 遵守事項:
# 1. 省略・簡略化の完全禁止: is_visible および cast_lidar の判定数式をすべて詳細に記述。
# 2. 物理的整合性: 0.45m以内(自己表面)を透過する「次点探査」を全オブジェクト・全モードに適用。
# 3. 高速性維持: Numba JIT によるスカラー演算を徹底。ループ内でのメモリ確保をゼロに抑止。
# 4. 全天候型検知: 静的な壁 (AABB), 動的なエージェント (Circle), 動的な箱/ランプ (OBB) を全て個別に計算。

import numpy as np
import mujoco
import math
from numba import njit

@njit(cache=True)
def _get_sdf_scalar(px, py, walls_xpos, walls_size, ag_pos, ag_radii, ag_b_ids, box_pos, box_size, box_quats, box_b_ids, ex1, ex2, ignore_id):
    """SDF演算コア：指定IDを除外して最短距離を算出（純粋スカラー演算）"""
    d_min = 15.0
    # 1. 静的な壁 (AABB)
    for i in range(len(walls_xpos)):
        dx = abs(px - walls_xpos[i, 0]) - walls_size[i, 0]
        dy = abs(py - walls_xpos[i, 1]) - walls_size[i, 1]
        # 外側距離と内側距離を統合した AABB SDF
        d = math.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2) + min(max(dx, dy), 0.0)
        if d < d_min:
            d_min = d
            
    # 2. 他エージェント (Circle)
    for i in range(len(ag_pos)):
        bid = ag_b_ids[i]
        # 自己、または透過対象(ignore_id)をスキップ
        if bid == ex1 or bid == ex2 or bid == ignore_id:
            continue
        d = math.sqrt((px - ag_pos[i, 0])**2 + (py - ag_pos[i, 1])**2) - ag_radii[i]
        if d < d_min:
            d_min = d
            
    # 3. 箱 / Ramp (OBB)
    for i in range(len(box_pos)):
        bid = box_b_ids[i]
        if bid == ex1 or bid == ex2 or bid == ignore_id:
            continue
        # クォータニオンから Yaw を直接復元
        q = box_quats[i]
        yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
        cs, sn = math.cos(-yaw), math.sin(-yaw)
        # 点 p を箱のローカル座標系へ回転投影
        rx, ry = px - box_pos[i, 0], py - box_pos[i, 1]
        lx = rx * cs - ry * sn
        ly = rx * sn + ry * cs
        # ローカル AABB 空間での SDF
        dx = abs(lx) - box_size[i, 0]
        dy = abs(ly) - box_size[i, 1]
        d = math.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2) + min(max(dx, dy), 0.0)
        if d < d_min:
            d_min = d
            
    return d_min

@njit(cache=True)
def _compute_lidar_jit_core(pos_x, pos_y, h_cos, h_sin, base_cos, base_sin, 
                             walls_xpos, walls_size, ag_pos, ag_radii, ag_b_ids, 
                             box_pos, box_size, box_quats, box_b_ids,
                             mode, exclude_body_id, ignore_id, max_dist):
    """Lidar 演算コア：全軸のスラブ判定と円交差判定を完全に記述"""
    res = np.full(12, max_dist, dtype=np.float32)
    SKIP_THRESHOLD = 0.45 

    for i in range(12):
        # 計測方向ベクトルの算出
        vx = base_cos[i] * h_cos - base_sin[i] * h_sin
        vy = base_sin[i] * h_cos + base_cos[i] * h_sin
        
        if mode == 1: # Geometric
            d_min = max_dist
            
            # A. 壁判定 (AABB Slab法)
            for j in range(len(walls_xpos)):
                bx, by, sx, sy = walls_xpos[j,0], walls_xpos[j,1], walls_size[j,0], walls_size[j,1]
                t_n, t_f = -1e10, 1e10
                if abs(vx) > 1e-12:
                    iv = 1.0 / vx
                    t1 = (bx - sx - pos_x) * iv
                    t2 = (bx + sx - pos_x) * iv
                    t_n = max(t_n, min(t1, t2))
                    t_f = min(t_f, max(t1, t2))
                elif abs(pos_x - bx) > sx: continue
                if abs(vy) > 1e-12:
                    iv = 1.0 / vy
                    t1 = (by - sy - pos_y) * iv
                    t2 = (by + sy - pos_y) * iv
                    t_n = max(t_n, min(t1, t2))
                    t_f = min(t_f, max(t1, t2))
                elif abs(pos_y - by) > sy: continue
                if t_f >= t_n and t_f > SKIP_THRESHOLD:
                    hit = t_n if t_n > SKIP_THRESHOLD else t_f
                    if hit < d_min: d_min = hit
            
            # B. 他エージェント判定 (Circle 交差判定)
            for k in range(len(ag_pos)):
                if ag_b_ids[k] == exclude_body_id or ag_b_ids[k] == ignore_id:
                    continue
                ox, oy = pos_x - ag_pos[k,0], pos_y - ag_pos[k,1]
                b = 2.0 * (ox * vx + oy * vy)
                c = ox * ox + oy * oy - ag_radii[k]**2
                det = b * b - 4.0 * c
                if det >= 0:
                    sd = math.sqrt(det)
                    t1 = (-b - sd) / 2.0
                    t2 = (-b + sd) / 2.0
                    if t2 > SKIP_THRESHOLD:
                        hit = t1 if t1 > SKIP_THRESHOLD else t2
                        if hit < d_min: d_min = hit

            # C. 箱 / Ramp 判定 (OBB Slab法)
            for k in range(len(box_pos)):
                if box_b_ids[k] == exclude_body_id or box_b_ids[k] == ignore_id:
                    continue
                q = box_quats[k]
                yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
                cs, sn = math.cos(yaw), math.sin(yaw)
                dx, dy = pos_x - box_pos[k,0], pos_y - box_pos[k,1]
                vrx, vry = vx * cs + vy * sn, -vx * sn + vy * cs
                prx, pry = dx * cs + dy * sn, -dx * sn + dy * cs
                sx, sy = box_size[k,0], box_size[k,1]
                tn_l, tf_l = -1e10, 1e10
                if abs(vrx) > 1e-12:
                    iv = 1.0 / vrx; t1, t2 = (-sx - prx) * iv, (sx - prx) * iv
                    tn_l, tf_l = max(tn_l, min(t1, t2)), min(tf_l, max(t1, t2))
                elif abs(prx) > sx: continue
                if abs(vry) > 1e-12:
                    iv = 1.0 / vry; t1, t2 = (-sy - pry) * iv, (sy - pry) * iv
                    tn_l, tf_l = max(tn_l, min(t1, t2)), min(tf_l, max(t1, t2))
                elif abs(pry) > sy: continue
                if tf_l >= tn_l and tf_l > SKIP_THRESHOLD:
                    hit = tn_l if tn_l > SKIP_THRESHOLD else tf_l
                    if hit < d_min: d_min = hit
            res[i] = d_min

        elif mode == 2: # Sphere Tracing
            curr_t = SKIP_THRESHOLD
            for _ in range(40):
                cx, cy = pos_x + vx * curr_t, pos_y + vy * curr_t
                dist = _get_sdf_scalar(cx, cy, walls_xpos, walls_size, ag_pos, ag_radii, ag_b_ids, box_pos, box_size, box_quats, box_b_ids, exclude_body_id, -1, ignore_id)
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
                if any(k in name for k in ["wall", "maze", "border"]): self.idx_wall_geom.append(i)
            else:
                self.idx_box_geom.append(i); self.idx_box_body.append(bid)
        
        self.idx_wall_geom = np.array(self.idx_wall_geom, dtype=np.int32)
        self.idx_box_geom, self.idx_box_body = np.array(self.idx_box_geom, dtype=np.int32), np.array(self.idx_box_body, dtype=np.int32)
        self.idx_agent_geom, self.idx_agent_body = np.array(self.idx_agent_geom, dtype=np.int32), np.array(self.idx_agent_body, dtype=np.int32)
        self.agent_radii = np.array(self.agent_radii, dtype=np.float32)

    def cast_lidar(self, pos, heading=0.0, mode=1, body_exclude=-1, ignore_body_id=-1):
        h_c, h_s = math.cos(heading), math.sin(heading)
        if mode == 0:
            res = np.full(12, self.max_dist, dtype=np.float32); p_o = np.array([pos[0], pos[1], 0.4], dtype=np.float64)
            for i in range(12):
                v_dir = np.array([self.base_cos[i]*h_c - self.base_sin[i]*h_s, self.base_sin[i]*h_c + self.base_cos[i]*h_s, 0.0])
                total_d, curr_p = 0.0, p_o.copy()
                for _ in range(3):
                    d = mujoco.mj_ray(self.m, self.d, curr_p, v_dir, self.group_mask, 1, int(body_exclude), self._geomid_out)
                    if d < 0: total_d = self.max_dist; break
                    hit_abs = total_d + d
                    if hit_abs < 0.45 or self.m.geom_bodyid[self._geomid_out[0]] == ignore_body_id:
                        curr_p += v_dir * (d + 0.001); total_d += d + 0.001; continue
                    else: total_d = hit_abs; break
                res[i] = total_d
            return res
        return _compute_lidar_jit_core(pos[0], pos[1], h_c, h_s, self.base_cos, self.base_sin, self.d.geom_xpos[self.idx_wall_geom], self.m.geom_size[self.idx_wall_geom], self.d.geom_xpos[self.idx_agent_geom, :2], self.agent_radii, self.idx_agent_body, self.d.geom_xpos[self.idx_box_geom, :2], self.m.geom_size[self.idx_box_geom, :2], self.d.xquat[self.idx_box_body], self.idx_box_body, mode, int(body_exclude), int(ignore_body_id), self.max_dist)

    def is_visible(self, p1, p2, mode=1, body_exclude=-1, target_body_id=-1):
        """Lidarと同じ 0.45m スキップ ＆ 全オブジェクト判定を完全に記述"""
        diff = p2[:2] - p1[:2]; dist = np.linalg.norm(diff)
        if dist < 0.46: return True
        vx, vy = diff[0]/dist, diff[1]/dist
        
        if mode == 0: # Native
            p_orig = np.array([p1[0], p1[1], 0.4], dtype=np.float64); v_dir = np.array([vx, vy, 0.0])
            hit = mujoco.mj_ray(self.m, self.d, p_orig, v_dir, self.group_mask, 1, int(body_exclude), self._geomid_out)
            if hit < 0: return True
            return self.m.geom_bodyid[self._geomid_out[0]] == target_body_id or hit > (dist - 0.1)

        elif mode == 1: # Geometric (完全展開版)
            # A. 壁判定 (Slab)
            for j in range(len(self.idx_wall_geom)):
                bx, by = self.d.geom_xpos[self.idx_wall_geom[j]][:2]; sx, sy = self.m.geom_size[self.idx_wall_geom[j]][:2]
                tn, tf = -1e10, 1e10
                if abs(vx) > 1e-12:
                    iv=1.0/vx; t1=(bx-sx-p1[0])*iv; t2=(bx+sx-p1[0])*iv; tn=max(tn,min(t1,t2)); tf=min(tf,max(t1,t2))
                elif abs(p1[0]-bx)>sx: continue
                if abs(vy) > 1e-12:
                    iv=1.0/vy; t1=(by-sy-p1[1])*iv; t2=(by+sy-p1[1])*iv; tn=max(tn,min(t1,t2)); tf=min(tf,max(t1,t2))
                elif abs(p1[1]-by)>sy: continue
                if tf >= tn and 0.45 < tn < dist - 0.05: return False
            
            # B. 他エージェント判定 (Circle)
            for k in range(len(self.idx_agent_geom)):
                bid = self.idx_agent_body[k]
                if bid == body_exclude or bid == target_body_id: continue
                ox, oy = p1[0] - self.d.geom_xpos[self.idx_agent_geom[k], 0], p1[1] - self.d.geom_xpos[self.idx_agent_geom[k], 1]
                rad = self.agent_radii[k]; b = 2.0*(ox*vx+oy*vy); c = ox*ox+oy*oy - rad**2; det = b*b-4.0*c
                if det >= 0:
                    sd = math.sqrt(det); t1 = (-b-sd)/2.0; t2 = (-b+sd)/2.0
                    if 0.45 < t2 < dist - 0.05: return False

            # C. 箱判定 (OBB Slab)
            for k in range(len(self.idx_box_geom)):
                bid = self.idx_box_body[k]
                if bid == body_exclude or bid == target_body_id: continue
                q = self.d.xquat[bid]; yaw = math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))
                cs, sn = math.cos(yaw), math.sin(yaw)
                dx, dy = p1[0]-self.d.geom_xpos[self.idx_box_geom[k],0], p1[1]-self.d.geom_xpos[self.idx_box_geom[k],1]
                vrx, vry = vx*cs+vy*sn, -vx*sn+vy*cs; prx, pry = dx*cs+dy*sn, -dx*sn+dy*cs
                sx, sy = self.m.geom_size[self.idx_box_geom[k],0], self.m.geom_size[self.idx_box_geom[k],1]
                tn_l, tf_l = -1e10, 1e10
                if abs(vrx)>1e-12:
                    iv=1.0/vrx; t1, t2 = (-sx-prx)*iv, (sx-prx)*iv; tn_l, tf_l = max(tn_l,min(t1,t2)), min(tf_l,max(t1,t2))
                elif abs(prx)>sx: continue
                if abs(vry)>1e-12:
                    iv=1.0/vry; t1, t2 = (-sy-pry)*iv, (sy-pry)*iv; tn_l, tf_l = max(tn_l,min(t1,t2)), min(tf_l,max(t1,t2))
                elif abs(pry)>sy: continue
                if tf_l >= tn_l and 0.45 < tn_l < dist - 0.05: return False
            return True

        elif mode == 2: # Sphere Tracing
            curr_t = 0.46
            for _ in range(30):
                cx, cy = p1[0] + vx * curr_t, p1[1] + vy * curr_t
                d_sdf = _get_sdf_scalar(cx, cy, self.d.geom_xpos[self.idx_wall_geom], self.m.geom_size[self.idx_wall_geom], self.d.geom_xpos[self.idx_agent_geom, :2], self.agent_radii, self.idx_agent_body, self.d.geom_xpos[self.idx_box_geom, :2], self.m.geom_size[self.idx_box_geom, :2], self.d.xquat[self.idx_box_body], self.idx_box_body, int(body_exclude), int(target_body_id), -1)
                if d_sdf < 0.005: return False
                curr_t += d_sdf
                if curr_t >= dist - 0.05: return True
            return True