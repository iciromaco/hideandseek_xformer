import math
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=False, seed=123)
_ = env.reset()
act_dim = env.action_space.shape[0]
# advance a few steps
for i in range(10):
    a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
    env.step(a)

viewer = "s"
target = "h1"
if viewer not in env.agent_keys or target not in env.agent_keys:
    print("agents not present")
    sys.exit(0)

v_bid = env.body_ids[viewer]
t_bid = env.body_ids[target]
v_pos = env.data.xpos[v_bid][:2]
t_pos = env.data.xpos[t_bid][:2]
print("viewer pos", v_pos, "target pos", t_pos)

p1 = np.array([v_pos[0], v_pos[1], 0.4])
p2 = np.array([t_pos[0], t_pos[1], 0.4])
print("engine visible?", env.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))

# Now replicate mode==1 loop from visibility_engine and print wall test details
from src.core.visibility_engine import VisibilityEngine

ve = env.vis_engine

diff_v = p2[:2] - p1[:2]
dist_full = np.linalg.norm(diff_v)
if dist_full < 0.41:
    print("dist < 0.41 -> visible")
else:
    vx = diff_v[0] / dist_full
    vy = diff_v[1] / dist_full
    MIN_T = 0.01
    print("checking walls...")
    for j in range(len(ve.idx_wall_geom)):
        idx = ve.idx_wall_geom[j]
        geom_pos = ve.d.geom_xpos[idx]
        geom_size = ve.m.geom_size[idx]
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
            print(f"wall {j} skipped by x check")
            continue
        if abs(vy) > 1e-12:
            iv = 1.0 / vy
            t1 = (by - sy - p1[1]) * iv
            t2 = (by + sy - p1[1]) * iv
            tn = max(tn, min(t1, t2))
            tf = min(tf, max(t1, t2))
        elif abs(p1[1] - by) > sy:
            print(f"wall {j} skipped by y check")
            continue
        print(f"wall {j}: bx,by=({bx:.3f},{by:.3f}) sx,sy=({sx:.3f},{sy:.3f}) tn={tn:.4f} tf={tf:.4f} dist_full={dist_full:.4f}")
        if tf >= tn:
            print("  tf>=tn true")
            if MIN_T < tn < dist_full - 0.05:
                print("  -> blocked by wall (MIN_T < tn < dist_full - 0.05)")
            else:
                print("  -> passes wall test (tn not in blocking range)")

# Also list agent geoms checks
print("\nchecking agent bodies...")
for k in range(len(ve.idx_agent_geom)):
    bid = ve.idx_agent_body[k]
    if bid == v_bid or bid == t_bid:
        continue
    ag_g_pos = ve.d.geom_xpos[ve.idx_agent_geom[k]]
    ox, oy = p1[0] - ag_g_pos[0], p1[1] - ag_g_pos[1]
    radius = ve.agent_radii[k]
    b_val = 2.0 * (ox * vx + oy * vy)
    c_val = ox * ox + oy * oy - radius**2
    det = b_val * b_val - 4.0 * c_val
    print(f"agent geom {k}: bid={bid} agpos=({ag_g_pos[0]:.3f},{ag_g_pos[1]:.3f}) radius={radius:.3f} det={det:.4f}")
    if det >= 0:
        sqrt_det = math.sqrt(det)
        t1_c = (-b_val - sqrt_det) / 2.0
        t2_c = (-b_val + sqrt_det) / 2.0
        print(f"  t1_c={t1_c:.4f} t2_c={t2_c:.4f} dist_full={dist_full:.4f}")
        if t2_c > MIN_T:
            print("  potential circle hit; blocking if appropriate")

# box/ramp checks
print("\nchecking box/ramp geoms...")
for k in range(len(ve.idx_box_geom)):
    bid = ve.idx_box_body[k]
    if bid == v_bid or bid == t_bid:
        continue
    q_box = ve.d.xquat[bid]
    y_box_sq, z_box_sq = q_box[2] ** 2, q_box[3] ** 2
    term1 = 2.0 * (q_box[0] * q_box[3] + q_box[1] * q_box[2])
    term2 = 1.0 - 2.0 * (y_box_sq + z_box_sq)
    yaw_box = math.atan2(term1, term2)
    cs_box, sn_box = math.cos(yaw_box), math.sin(yaw_box)
    g_pos = ve.d.geom_xpos[ve.idx_box_geom[k]]
    g_size = ve.m.geom_size[ve.idx_box_geom[k]]
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
        print(f"box {k} skipped by prx check")
        continue
    if abs(vry) > 1e-12:
        iv = 1.0 / vry
        t1_l, t2_l = (-sy_box - pry) * iv, (sy_box - pry) * iv
        tn_l, tf_l = max(tn_l, min(t1_l, t2_l)), min(tf_l, max(t1_l, t2_l))
    elif abs(pry) > sy_box:
        print(f"box {k} skipped by pry check")
        continue
    print(f"box {k}: g_pos=({g_pos[0]:.3f},{g_pos[1]:.3f}) g_size=({g_size[0]:.3f},{g_size[1]:.3f}) tn_l={tn_l:.4f} tf_l={tf_l:.4f}")
    if tf_l >= tn_l:
        if MIN_T < tn_l < dist_full - 0.05:
            print("  -> blocked by box/ramp")
        else:
            print("  -> passes box/ramp test")
