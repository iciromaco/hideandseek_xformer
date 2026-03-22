import math
import sys
import time

import numpy as np

# ensure project root is importable
sys.path.insert(0, ".")

try:
    from src.envs.hns_environment import TeamCosEnv
except Exception as e:
    print("IMPORT_ERROR", e)
    raise

env = TeamCosEnv(debug_mode=True, n_ramps=1, render_mode=None)
# optionally increase ramp boost for testing
TEST_RAMP_BOOST = 1.0  # set to float to override, e.g. 1.0
if TEST_RAMP_BOOST is not None:
    env.RAMP_BOOST_FWD = float(TEST_RAMP_BOOST)

# reset (handle gym / gymnasium return signatures)
res = env.reset()
if isinstance(res, tuple) and len(res) == 2:
    obs, info = res
else:
    obs = res
    info = {}

print("RESET_INFO_KEYS=", sorted(list(info.keys())))

# place learnable agent onto the first ramp (override qpos then forward)
try:
    ak = env.learnable_agent_key
    rid = env.ramp_ids[0]
    rpos = env.data.xpos[rid].copy()
    # qpos address for agent joints
    # use qpos_indices mapping for addresses
    qz_jid = env.qpos_indices[ak]["z"]
    qz_adr = env.model.jnt_qposadr[qz_jid]
    qx_jid = env.qpos_indices[ak]["x"]
    qx_adr = env.model.jnt_qposadr[qx_jid]
    qy_jid = env.qpos_indices[ak]["y"]
    qy_adr = env.model.jnt_qposadr[qy_jid]
    qrot_jid = env.qpos_indices[ak]["rot"]
    qrot_adr = env.model.jnt_qposadr[qrot_jid]
    # set planar x,y to ramp position and z slide to place on ramp surface
    env.data.qpos[qx_adr] = float(rpos[0])
    env.data.qpos[qy_adr] = float(rpos[1])
    # set slide joint so body_z = anchor_z(0.5) + qz -> choose qz=0.65
    env.data.qpos[qz_adr] = 0.65
    # set rotation to face uphill
    try:
        up = env._ramp_uphill_dir(rid)
        yaw = math.atan2(float(up[1]), float(up[0]))
        env.data.qpos[qrot_adr] = float(yaw)
    except Exception:
        pass
    # zero velocities
    env.data.qvel[:] = 0.0
    env.data.forward()
    print(f"[PLACED_ON_RAMP] ak={ak} ramp_pos={rpos} qz={env.data.qpos[qz_adr]:.3f} body_z={env.data.xpos[env.body_ids[ak]][2]:.3f}")
except Exception as e:
    print("PLACEMENT_OVERRIDE_FAILED", e)

# lock the ramp to prevent it moving during the climb test
try:
    rkey = "ramp1"
    if rkey in env.object_state:
        env.object_state[rkey]["mode"] = "locked"
        env.object_state[rkey]["owner"] = None
        env._cache_planar_object_pose()
        print(f"[RAMP_LOCKED] {rkey}")
    else:
        print(f"[RAMP_LOCKED_SKIPPED] {rkey} not in object_state")
except Exception as e:
    print("RAMP_LOCK_FAILED", e)

# we'll print per-ramp lx/ly/facing after a few steps (agent has settled)
DIAG_STEP = 6
max_steps = 120
for step in range(max_steps):
    # diagnostic dump at DIAG_STEP
    if step == DIAG_STEP:
        try:
            ak = env.learnable_agent_key
            apos = env.data.xpos[env.body_ids[ak]][:2]
            arot = env.data.qpos[env.model.jnt_qposadr[env.qpos_indices[ak]["rot"]]]
            afwd = np.array([np.cos(arot), np.sin(arot)], dtype=np.float32)
            s = []
            for i, rid in enumerate(env.ramp_ids, start=1):
                up = env._ramp_uphill_dir(rid)
                side = np.array([-up[1], up[0]], dtype=np.float32)
                rpos = env.data.xpos[rid][:2]
                rel = apos - rpos
                lx = float(np.dot(rel, up))
                ly = float(np.dot(rel, side))
                facing = float(np.dot(afwd, up))
                s.append(f"ramp{i}(lx={lx:.3f},ly={ly:.3f},f={facing:.3f})")
            print(f"[RAMP_CANDIDATES_STEP{DIAG_STEP}] " + ", ".join(s))
        except Exception as e:
            print("RAMP_CANDIDATES_DEBUG_FAILED", e)
    # try full forward to test boost sufficiency
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = 1.0
    out = env.step(action)
    # gymnasium: obs, reward, terminated, truncated, info
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    else:
        # older gym: obs, reward, done, info
        obs, reward, done, info = out
    dbg_z = info.get("dbg_agent_z")
    dbg_vz = info.get("dbg_agent_vz")
    dbg_prog = info.get("dbg_ramp_progress")
    dbg_boost = info.get("dbg_ramp_boost")
    dbg_climb = info.get("dbg_ramp_climbing")
    dbg_top = info.get("dbg_ramp_reached_top")
    print(f"step={step} dbg_agent_z={dbg_z} dbg_agent_vz={dbg_vz} dbg_ramp_progress={dbg_prog} dbg_ramp_boost={dbg_boost} climbing={dbg_climb} top={dbg_top}")
    if done:
        print("EPISODE_DONE at step", step)
        break

env.close()
print("RUN_COMPLETE")
