# visibility_engine.py v2.03
# 演習第26回：【不連続性根絶版】近距離での15mジャンプを廃止し、物理的誠実さを復元
#
# 修正内容:
# 1. 跳ね上がりバグの解消: レイが極至近距離でヒットした際に max_dist を返していたロジックを削除。
#    - これにより、エージェントが壁に密着しても正確に 0.45m 付近の値を返し続けます。
# 2. SKIP_THRESHOLDの適正化: 数値安定性のための 0.001m まで縮小し、実質的に全距離を有効化。
# 3. 1行1命令の遵守: Numba JIT 内の全ステップを詳細に記述。

import math

import mujoco
import numpy as np
from numba import njit

SDF_CELL_SIZE = 0.02  # 2cm セルサイズで静的SDFグリッドを構築


def _compute_wall_sdf_grid_jit(min_x, min_y, cell_size, n_x, n_y, walls_xpos, walls_size, max_dist):
    # walls_xpos: (N,2), walls_size: (N,2)
    grid = np.empty((n_y, n_x), dtype=np.float32)
    obstacles = np.concatenate([walls_xpos, walls_size], axis=1)  # (N,4)
    debug_printed = False
    for iy in range(n_y):
        py = min_y + iy * cell_size
        for ix in range(n_x):
            px = min_x + ix * cell_size
            p = np.array([px, py])
            d = np.abs(p - obstacles[:, :2]) - obstacles[:, 2:]
            outside = np.any(d > 0, axis=1)
            dist_out = np.linalg.norm(np.maximum(d, 0.0), axis=1)
            dist_in = -np.min(np.abs(d), axis=1)
            sdf_vals = np.where(outside, dist_out, dist_in)
            d_min = np.min(sdf_vals)
            if d_min < max_dist:
                grid[iy, ix] = d_min
            else:
                grid[iy, ix] = max_dist
            # --- デバッグ出力: SDF値が-0.1以下となった最初のグリッド点 ---
            if not debug_printed and d_min < -0.2:
                wall_idx = np.argmin(sdf_vals)
                wall_pos = obstacles[wall_idx, :2]
                wall_size = obstacles[wall_idx, 2:]
                print(
                    f"[SDF DEBUG] mode=static_grid, grid=({ix},{iy}), world=({px:.3f},{py:.3f}), SDF={d_min:.4f}, wall_center=({wall_pos[0]:.3f},{wall_pos[1]:.3f}), wall_size=({wall_size[0]:.3f},{wall_size[1]:.3f})"
                )
                debug_printed = True
    return grid


@njit(cache=True)
def _sample_sdf_grid_bilinear(px, py, sdf_grid, min_x, min_y, cell_size, max_dist):
    n_y = sdf_grid.shape[0]
    n_x = sdf_grid.shape[1]
    gx = (px - min_x) / cell_size
    gy = (py - min_y) / cell_size

    if gx < 0.0 or gy < 0.0 or gx > (n_x - 1) or gy > (n_y - 1):
        return max_dist

    x0 = int(math.floor(gx))
    y0 = int(math.floor(gy))
    x1 = x0 + 1
    y1 = y0 + 1
    if x1 >= n_x:
        x1 = n_x - 1
    if y1 >= n_y:
        y1 = n_y - 1

    tx = gx - x0
    ty = gy - y0

    v00 = float(sdf_grid[y0, x0])
    v10 = float(sdf_grid[y0, x1])
    v01 = float(sdf_grid[y1, x0])
    v11 = float(sdf_grid[y1, x1])

    v0 = v00 * (1.0 - tx) + v10 * tx
    v1 = v01 * (1.0 - tx) + v11 * tx
    return v0 * (1.0 - ty) + v1 * ty


@njit(cache=True)
def _get_dynamic_sdf_scalar(
    px,
    py,
    ag_pos,
    ag_radii,
    ag_b_ids,
    box_pos,
    box_size,
    box_quats,
    box_b_ids,
    ex1,
    ignore_id,
    max_dist,
):
    d_min = max_dist

    num_ags = len(ag_pos)
    for i in range(num_ags):
        bid = ag_b_ids[i]
        if bid == ex1:
            continue
        if bid == ignore_id:
            continue
        dist_c = math.sqrt((px - ag_pos[i, 0]) ** 2 + (py - ag_pos[i, 1]) ** 2)
        d_agent = dist_c - ag_radii[i]
        if d_agent < d_min:
            d_min = d_agent

    num_boxes = len(box_pos)
    for i in range(num_boxes):
        bid = box_b_ids[i]
        if bid == ex1:
            continue
        if bid == ignore_id:
            continue

        q = box_quats[i]
        y_sq = q[2] * q[2]
        z_sq = q[3] * q[3]
        term1 = 2.0 * (q[0] * q[3] + q[1] * q[2])
        term2 = 1.0 - 2.0 * (y_sq + z_sq)
        yaw = math.atan2(term1, term2)

        cs = math.cos(-yaw)
        sn = math.sin(-yaw)

        rx = px - box_pos[i, 0]
        ry = py - box_pos[i, 1]
        lx = rx * cs - ry * sn
        ly = rx * sn + ry * cs

        dx_b = abs(lx) - box_size[i, 0]
        dy_b = abs(ly) - box_size[i, 1]
        d_obb = math.sqrt(max(dx_b, 0.0) ** 2 + max(dy_b, 0.0) ** 2) + min(max(dx_b, dy_b), 0.0)
        if d_obb < d_min:
            d_min = d_obb

    return d_min


@njit(cache=True)
def _get_sdf_scalar(
    px,
    py,
    walls_xpos,
    walls_size,
    ag_pos,
    ag_radii,
    ag_b_ids,
    box_pos,
    box_size,
    box_quats,
    box_b_ids,
    ex1,
    ex2,
    ignore_id,
):
    """SDF演算コア：スカラー変数による最短距離算出"""
    d_min = 15.0

    # 1. 静的な壁 (AABB)
    num_walls = len(walls_xpos)
    for i in range(num_walls):
        dx = abs(px - walls_xpos[i, 0]) - walls_size[i, 0]
        dy = abs(py - walls_xpos[i, 1]) - walls_size[i, 1]

        d_box = math.sqrt(max(dx, 0.0) ** 2 + max(dy, 0.0) ** 2) + min(max(dx, dy), 0.0)
        if d_box < d_min:
            d_min = d_box

    # 2. 他エージェント (Circle)
    num_ags = len(ag_pos)
    for i in range(num_ags):
        bid = ag_b_ids[i]
        if bid == ex1:
            continue
        if bid == ex2:
            continue
        if bid == ignore_id:
            continue

        dist_c = math.sqrt((px - ag_pos[i, 0]) ** 2 + (py - ag_pos[i, 1]) ** 2)
        d_agent = dist_c - ag_radii[i]
        if d_agent < d_min:
            d_min = d_agent

    # 3. 箱 / Ramp (OBB)
    num_boxes = len(box_pos)
    for i in range(num_boxes):
        bid = box_b_ids[i]
        if bid == ex1:
            continue
        if bid == ex2:
            continue
        if bid == ignore_id:
            continue

        q = box_quats[i]
        y_sq = q[2] * q[2]
        z_sq = q[3] * q[3]
        term1 = 2.0 * (q[0] * q[3] + q[1] * q[2])
        term2 = 1.0 - 2.0 * (y_sq + z_sq)
        yaw = math.atan2(term1, term2)

        cs = math.cos(-yaw)
        sn = math.sin(-yaw)

        rx = px - box_pos[i, 0]
        ry = py - box_pos[i, 1]

        lx = rx * cs - ry * sn
        ly = rx * sn + ry * cs

        dx_b = abs(lx) - box_size[i, 0]
        dy_b = abs(ly) - box_size[i, 1]

        d_obb = math.sqrt(max(dx_b, 0.0) ** 2 + max(dy_b, 0.0) ** 2) + min(max(dx_b, dy_b), 0.0)
        if d_obb < d_min:
            d_min = d_obb

    return d_min


@njit(cache=True)
def _compute_lidar_jit_core(
    pos_x,
    pos_y,
    h_cos,
    h_sin,
    base_cos,
    base_sin,
    walls_xpos,
    walls_size,
    ag_pos,
    ag_radii,
    ag_b_ids,
    box_pos,
    box_size,
    box_quats,
    box_b_ids,
    mode,
    exclude_body_id,
    ignore_id,
    max_dist,
    static_sdf_grid,
    static_min_x,
    static_min_y,
    static_cell_size,
):
    """Lidar 演算コア：近距離死角を完全に排除"""
    res = np.full(12, max_dist, dtype=np.float32)

    # 💡 0.45m 以前の跳ね上がりを許さないため、最小マージン(1mm)のみを設定
    SAFE_MARGIN = 0.001

    for i in range(12):
        vx = base_cos[i] * h_cos - base_sin[i] * h_sin
        vy = base_sin[i] * h_cos + base_cos[i] * h_sin

        if mode == 1:  # Geometric Intersection
            d_hit_min = max_dist

            # A. 壁
            for j in range(len(walls_xpos)):
                bx = walls_xpos[j, 0]
                by = walls_xpos[j, 1]
                sx = walls_size[j, 0]
                sy = walls_size[j, 1]
                tn = -1e10
                tf = 1e10

                if abs(vx) > 1e-12:
                    inv_vx = 1.0 / vx
                    t1 = (bx - sx - pos_x) * inv_vx
                    t2 = (bx + sx - pos_x) * inv_vx
                    tn = max(tn, min(t1, t2))
                    tf = min(tf, max(t1, t2))
                elif abs(pos_x - bx) > sx:
                    continue

                if abs(vy) > 1e-12:
                    inv_vy = 1.0 / vy
                    t1 = (by - sy - pos_y) * inv_vy
                    t2 = (by + sy - pos_y) * inv_vy
                    tn = max(tn, min(t1, t2))
                    tf = min(tf, max(t1, t2))
                elif abs(pos_y - by) > sy:
                    continue

                if tf >= tn:
                    # 💡 修正：小さな tn も有効なヒットとして扱う
                    if tf > SAFE_MARGIN:
                        hit_t = tn
                        if tn <= SAFE_MARGIN:
                            hit_t = tf
                        if hit_t < d_hit_min:
                            d_hit_min = hit_t

            # B. 他エージェント
            for k in range(len(ag_pos)):
                if ag_b_ids[k] == exclude_body_id:
                    continue
                if ag_b_ids[k] == ignore_id:
                    continue

                ox = pos_x - ag_pos[k, 0]
                oy = pos_y - ag_pos[k, 1]
                radius = ag_radii[k]

                b_val = 2.0 * (ox * vx + oy * vy)
                c_val = ox * ox + oy * oy - radius**2
                det = b_val * b_val - 4.0 * c_val

                if det >= 0:
                    sqrt_det = math.sqrt(det)
                    t1_c = (-b_val - sqrt_det) / 2.0
                    t2_c = (-b_val + sqrt_det) / 2.0
                    if t2_c > SAFE_MARGIN:
                        hit_t_c = t1_c
                        if t1_c <= SAFE_MARGIN:
                            hit_t_c = t2_c
                        if hit_t_c < d_hit_min:
                            d_hit_min = hit_t_c

            # C. 箱 / Ramp
            for k in range(len(box_pos)):
                if box_b_ids[k] == exclude_body_id:
                    continue
                if box_b_ids[k] == ignore_id:
                    continue

                q_b = box_quats[k]
                y_b_sq = q_b[2] * q_b[2]
                z_b_sq = q_b[3] * q_b[3]
                y_term1 = 2.0 * (q_b[0] * q_b[3] + q_b[1] * q_b[2])
                y_term2 = 1.0 - 2.0 * (y_b_sq + z_b_sq)
                yaw_b = math.atan2(y_term1, y_term2)

                cs_b = math.cos(yaw_b)
                sn_b = math.sin(yaw_b)
                dx_rel = pos_x - box_pos[k, 0]
                dy_rel = pos_y - box_pos[k, 1]
                vrx = vx * cs_b + vy * sn_b
                vry = -vx * sn_b + vy * cs_b
                prx = dx_rel * cs_b + dy_rel * sn_b
                pry = -dx_rel * sn_b + dy_rel * cs_b
                sx_b = box_size[k, 0]
                sy_b = box_size[k, 1]
                tn_l = -1e10
                tf_l = 1e10

                if abs(vrx) > 1e-12:
                    inv_vrx = 1.0 / vrx
                    t1_l = (-sx_b - prx) * inv_vrx
                    t2_l = (sx_b - prx) * inv_vrx
                    tn_l = max(tn_l, min(t1_l, t2_l))
                    tf_l = min(tf_l, max(t1_l, t2_l))
                elif abs(prx) > sx_b:
                    continue
                if abs(vry) > 1e-12:
                    inv_vry = 1.0 / vry
                    t1_l = (-sy_b - pry) * inv_vry
                    t2_l = (sy_b - pry) * inv_vry
                    tn_l = max(tn_l, min(t1_l, t2_l))
                    tf_l = min(tf_l, max(t1_l, t2_l))
                elif abs(pry) > sy_b:
                    continue
                if tf_l >= tn_l:
                    if tf_l > SAFE_MARGIN:
                        hit_t_l = tn_l
                        if tn_l <= SAFE_MARGIN:
                            hit_t_l = tf_l
                        if hit_t_l < d_hit_min:
                            d_hit_min = hit_t_l

            res[i] = d_hit_min

        elif mode == 2:  # Sphere Tracing
            curr_t = SAFE_MARGIN
            for _step in range(40):
                cx = pos_x + vx * curr_t
                cy = pos_y + vy * curr_t
                dist_sdf = _get_sdf_scalar(
                    cx,
                    cy,
                    walls_xpos,
                    walls_size,
                    ag_pos,
                    ag_radii,
                    ag_b_ids,
                    box_pos,
                    box_size,
                    box_quats,
                    box_b_ids,
                    exclude_body_id,
                    -1,
                    ignore_id,
                )
                if dist_sdf < 0.005:
                    res[i] = curr_t
                    break
                curr_t = curr_t + dist_sdf
                if curr_t >= max_dist:
                    res[i] = max_dist
                    break
        elif mode == 4:  # Hybrid Sphere Tracing (precomputed static SDF + dynamic correction)
            curr_t = SAFE_MARGIN
            for _step in range(40):
                cx = pos_x + vx * curr_t
                cy = pos_y + vy * curr_t
                dist_static = _sample_sdf_grid_bilinear(
                    cx,
                    cy,
                    static_sdf_grid,
                    static_min_x,
                    static_min_y,
                    static_cell_size,
                    max_dist,
                )
                dist_dynamic = _get_dynamic_sdf_scalar(
                    cx,
                    cy,
                    ag_pos,
                    ag_radii,
                    ag_b_ids,
                    box_pos,
                    box_size,
                    box_quats,
                    box_b_ids,
                    exclude_body_id,
                    ignore_id,
                    max_dist,
                )
                dist_sdf = dist_static
                if dist_dynamic < dist_sdf:
                    dist_sdf = dist_dynamic
                if dist_sdf < 0.005:
                    res[i] = curr_t
                    break
                curr_t = curr_t + dist_sdf
                if curr_t >= max_dist:
                    res[i] = max_dist
                    break
    return res


class VisibilityEngine:
    def wall_distance(self, px, py):
        """
        px, py: 環境のx, y座標（Mujocoワールド座標系）
        壁までの最短距離（SDFグリッドによるバイリニア補間）を返す。
        """
        self._ensure_mode4_sdf()
        return _sample_sdf_grid_bilinear(
            px,
            py,
            self._mode4_sdf_grid,
            self._mode4_min_x,
            self._mode4_min_y,
            self._mode4_cell_size,
            self.max_dist,
        )

    def __init__(self, m, d, mode4_sdf_cell_size=SDF_CELL_SIZE):
        self.m = m
        self.d = d
        self.max_dist = 15.0
        self.mode4_sdf_cell_size = float(mode4_sdf_cell_size)
        if self.mode4_sdf_cell_size <= 0.0:
            self.mode4_sdf_cell_size = 0.05
        angles = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
        self.base_angles = np.array(angles, dtype=np.float32)
        self.base_cos = np.cos(np.deg2rad(self.base_angles))
        self.base_sin = np.sin(np.deg2rad(self.base_angles))
        self.group_mask = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self._geomid_out = np.zeros(1, dtype=np.int32)
        self._mode4_sdf_grid = np.empty((1, 1), dtype=np.float32)
        self._mode4_min_x = -self.max_dist
        self._mode4_min_y = -self.max_dist
        self._mode4_cell_size = self.mode4_sdf_cell_size
        self._mode4_ready = False
        self._extract_indices()

    def _ensure_mode4_sdf(self):
        if self._mode4_ready:
            return
        import os
        import pickle

        walls = self.d.geom_xpos[self.idx_wall_geom]
        wall_sizes = self.m.geom_size[self.idx_wall_geom]

        if len(walls) == 0:
            min_x = -self.max_dist
            max_x = self.max_dist
            min_y = -self.max_dist
            max_y = self.max_dist
        else:
            min_x = float(np.min(walls[:, 0] - wall_sizes[:, 0]) - 0.5)
            max_x = float(np.max(walls[:, 0] + wall_sizes[:, 0]) + 0.5)
            min_y = float(np.min(walls[:, 1] - wall_sizes[:, 1]) - 0.5)
            max_y = float(np.max(walls[:, 1] + wall_sizes[:, 1]) + 0.5)

        cell = self.mode4_sdf_cell_size
        n_x = max(2, int(math.ceil((max_x - min_x) / cell)) + 1)
        n_y = max(2, int(math.ceil((max_y - min_y) / cell)) + 1)

        # ファイル名を壁配置・サイズ・cell_sizeで一意に決定
        wall_hash = hash(
            (
                tuple(np.round(walls.flatten(), 4)),
                tuple(np.round(wall_sizes.flatten(), 4)),
                round(cell, 4),
            )
        )
        sdf_dir = os.path.join(os.path.dirname(__file__), "../envs")
        os.makedirs(sdf_dir, exist_ok=True)
        sdf_path = os.path.join(sdf_dir, f"sdfgrid_mode4_{wall_hash}.pkl")

        if os.path.exists(sdf_path):
            with open(sdf_path, "rb") as f:
                data = pickle.load(f)
            self._mode4_sdf_grid = data["grid"]
            self._mode4_min_x = data["min_x"]
            self._mode4_min_y = data["min_y"]
            self._mode4_cell_size = data["cell_size"]
        else:
            self._mode4_sdf_grid = _compute_wall_sdf_grid_jit(
                float(min_x),
                float(min_y),
                float(cell),
                int(n_x),
                int(n_y),
                self.d.geom_xpos[self.idx_wall_geom, :2],
                self.m.geom_size[self.idx_wall_geom, :2],
                float(self.max_dist),
            )
            self._mode4_min_x = float(min_x)
            self._mode4_min_y = float(min_y)
            self._mode4_cell_size = float(cell)
            # 保存
            with open(sdf_path, "wb") as f:
                pickle.dump(
                    {
                        "grid": self._mode4_sdf_grid,
                        "min_x": self._mode4_min_x,
                        "min_y": self._mode4_min_y,
                        "cell_size": self._mode4_cell_size,
                    },
                    f,
                )
        self._mode4_ready = True

    def _extract_indices(self):
        self.idx_wall_geom = []
        self.idx_box_geom = []
        self.idx_box_body = []
        self.idx_agent_geom = []
        self.idx_agent_body = []
        self.agent_radii = []
        num_geoms = self.m.ngeom
        for i in range(num_geoms):
            if self.m.geom_group[i] != 0:
                continue
            bid = self.m.geom_bodyid[i]
            geom_name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            name_lower = geom_name.lower()
            if "_btm" in name_lower:
                self.idx_agent_geom.append(i)
                self.idx_agent_body.append(bid)
                radius = self.m.geom_size[i, 0]
                self.agent_radii.append(radius)
            elif self.m.body_jntnum[bid] == 0:
                if any(key in name_lower for key in ["wall", "maze", "border"]):
                    self.idx_wall_geom.append(i)
            else:
                self.idx_box_geom.append(i)
                self.idx_box_body.append(bid)
        self.idx_wall_geom = np.array(self.idx_wall_geom, dtype=np.int32)
        self.idx_box_geom = np.array(self.idx_box_geom, dtype=np.int32)
        self.idx_box_body = np.array(self.idx_box_body, dtype=np.int32)
        self.idx_agent_geom = np.array(self.idx_agent_geom, dtype=np.int32)
        self.idx_agent_body = np.array(self.idx_agent_body, dtype=np.int32)
        self.agent_radii = np.array(self.agent_radii, dtype=np.float32)

    def cast_lidar(self, pos, heading=0.0, mode=1, body_exclude=-1, ignore_body_id=-1):
        h_cos = math.cos(heading)
        h_sin = math.sin(heading)
        if mode == 0:
            res_native = np.full(12, self.max_dist, dtype=np.float32)
            p_origin = np.array([pos[0], pos[1], 0.4], dtype=np.float64)
            for i in range(12):
                vx = self.base_cos[i] * h_cos - self.base_sin[i] * h_sin
                vy = self.base_sin[i] * h_cos + self.base_cos[i] * h_sin
                v_dir = np.array([vx, vy, 0.0])
                dist = mujoco.mj_ray(
                    self.m,
                    self.d,
                    p_origin,
                    v_dir,
                    self.group_mask,
                    1,
                    int(body_exclude),
                    self._geomid_out,
                )
                # 💡 修正：ヒットがあればそのまま返す（15mへのジャンプを削除）
                if dist >= 0:
                    res_native[i] = dist
            return res_native
        if mode == 4:
            self._ensure_mode4_sdf()
        return _compute_lidar_jit_core(
            pos[0],
            pos[1],
            h_cos,
            h_sin,
            self.base_cos,
            self.base_sin,
            self.d.geom_xpos[self.idx_wall_geom],
            self.m.geom_size[self.idx_wall_geom],
            self.d.geom_xpos[self.idx_agent_geom, :2],
            self.agent_radii,
            self.idx_agent_body,
            self.d.geom_xpos[self.idx_box_geom, :2],
            self.m.geom_size[self.idx_box_geom, :2],
            self.d.xquat[self.idx_box_body],
            self.idx_box_body,
            mode,
            int(body_exclude),
            int(ignore_body_id),
            self.max_dist,
            self._mode4_sdf_grid,
            float(self._mode4_min_x),
            float(self._mode4_min_y),
            float(self._mode4_cell_size),
        )

    def is_visible(self, p1, p2, mode=1, body_exclude=-1, target_body_id=-1):
        diff_v = p2[:2] - p1[:2]
        dist_full = np.linalg.norm(diff_v)
        if dist_full < 0.41:
            return True
        vx = diff_v[0] / dist_full
        vy = diff_v[1] / dist_full
        if mode == 0:
            p_start = np.array([p1[0], p1[1], 0.4], dtype=np.float64)
            v_ray = np.array([vx, vy, 0.0])
            hit = mujoco.mj_ray(
                self.m,
                self.d,
                p_start,
                v_ray,
                self.group_mask,
                1,
                int(body_exclude),
                self._geomid_out,
            )
            if hit < 0:
                return True
            hit_body = self.m.geom_bodyid[self._geomid_out[0]]
            if hit_body == target_body_id:
                return True
            return hit > (dist_full - 0.1)
        elif mode == 1:
            MIN_T = 0.01
            num_w = len(self.idx_wall_geom)
            for j in range(num_w):
                geom_pos = self.d.geom_xpos[self.idx_wall_geom[j]]
                geom_size = self.m.geom_size[self.idx_wall_geom[j]]
                bx, by = geom_pos[0], geom_pos[1]
                sx, sy = geom_size[0], geom_size[1]
                tn, tf = -1e10, 1e10
                if abs(vx) > 1e-12:
                    iv = 1.0 / vx
                    t1 = (bx - sx - p1[0]) * iv
                    t2 = (bx + sx - p1[0]) * iv
                    tn = max(tn, min(t1, t2))
                    tf = min(tf, max(t1, t2))
                elif abs(p1[0] - bx) > sx:
                    continue
                if abs(vy) > 1e-12:
                    iv = 1.0 / vy
                    t1 = (by - sy - p1[1]) * iv
                    t2 = (by + sy - p1[1]) * iv
                    tn = max(tn, min(t1, t2))
                    tf = min(tf, max(t1, t2))
                elif abs(p1[1] - by) > sy:
                    continue
                if tf >= tn:
                    if MIN_T < tn < dist_full - 0.05:
                        return False
            num_a = len(self.idx_agent_geom)
            for k in range(num_a):
                bid = self.idx_agent_body[k]
                if bid == body_exclude or bid == target_body_id:
                    continue
                ag_g_pos = self.d.geom_xpos[self.idx_agent_geom[k]]
                ox, oy = p1[0] - ag_g_pos[0], p1[1] - ag_g_pos[1]
                radius = self.agent_radii[k]
                b_val = 2.0 * (ox * vx + oy * vy)
                c_val = ox * ox + oy * oy - radius**2
                det = b_val * b_val - 4.0 * c_val
                if det >= 0:
                    sd = math.sqrt(det)
                    t2_val = (-b_val + sd) / 2.0
                    if MIN_T < t2_val < dist_full - 0.05:
                        return False
            num_b = len(self.idx_box_geom)
            for k in range(num_b):
                bid = self.idx_box_body[k]
                if bid == body_exclude or bid == target_body_id:
                    continue
                q_box = self.d.xquat[bid]
                y_box_sq, z_box_sq = q_box[2] ** 2, q_box[3] ** 2
                term1, term2 = (
                    2.0 * (q_box[0] * q_box[3] + q_box[1] * q_box[2]),
                    1.0 - 2.0 * (y_box_sq + z_box_sq),
                )
                yaw_box = math.atan2(term1, term2)
                cs_box, sn_box = math.cos(yaw_box), math.sin(yaw_box)
                g_pos, g_size = (
                    self.d.geom_xpos[self.idx_box_geom[k]],
                    self.m.geom_size[self.idx_box_geom[k]],
                )
                dx, dy = p1[0] - g_pos[0], p1[1] - g_pos[1]
                vrx, vry = vx * cs_box + vy * sn_box, -vx * sn_box + vy * cs_box
                prx, pry = dx * cs_box + dy * sn_box, -dx * sn_box + dy * cs_box
                sx_box, sy_box = g_size[0], g_size[1]
                tn_l, tf_l = -1e10, 1e10
                if abs(vrx) > 1e-12:
                    iv = 1.0 / vrx
                    t1_l, t2_l = (-sx_box - prx) * iv, (sx_box - prx) * iv
                    tn_l, tf_l = max(tn_l, min(t1_l, t2_l)), min(tf_l, max(t1_l, t2_l))
                elif abs(prx) > sx_box:
                    continue
                if abs(vry) > 1e-12:
                    iv = 1.0 / vry
                    t1_l, t2_l = (-sy_box - pry) * iv, (sy_box - pry) * iv
                    tn_l, tf_l = max(tn_l, min(t1_l, t2_l)), min(tf_l, max(t1_l, t2_l))
                elif abs(pry) > sy_box:
                    continue
                if tf_l >= tn_l:
                    if MIN_T < tn_l < dist_full - 0.05:
                        return False
            return True
        return True
