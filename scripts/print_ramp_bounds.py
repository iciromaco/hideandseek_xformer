import sys

sys.path.insert(0, ".")
import numpy as np

from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=True, n_ramps=1, render_mode=None)
# init
res = env.reset()
try:
    rid = env.ramp_ids[0]
except Exception as e:
    print("NO_RAMP_IDS", e)
    raise
up = env._ramp_uphill_dir(rid)
try:
    lx_min, lx_max, ly_thresh = env._compute_ramp_lx_bounds(rid, up)
except Exception as e:
    print("COMPUTE_FAILED", e)
    raise
rpos = env.data.xpos[rid][:2]
print("rkey=", f"ramp1")
print("rpos=", rpos.tolist())
print("up=", up.tolist())
print("lx_min=", lx_min)
print("lx_max=", lx_max)
print("ly_thresh=", ly_thresh)
print("projected_top=", lx_max)

# Also print geom projections for inspection
try:
    geom_ids = env.obj_geom_ids.get("ramp1", [])
    print("geom_ids=", geom_ids)
    for gid in geom_ids:
        try:
            gpos = env.data.geom_xpos[gid][:2]
        except Exception:
            gpos = env.data.xpos[rid][:2]
        rel = gpos - rpos
        lx = float(np.dot(rel, up))
        side = np.array([-up[1], up[0]], dtype=np.float32)
        ly = float(np.dot(rel, side))
        print(f"geom {gid}: gpos={gpos.tolist()} lx={lx:.6f} ly={ly:.6f}")
except Exception as e:
    print("GEOM_PRINT_FAILED", e)

env.close()
