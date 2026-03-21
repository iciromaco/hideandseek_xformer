#!/usr/bin/env python3
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.envs.hns_environment import TeamCosEnv

# Optional flag to disable boxes/ramps for a clean ground spawn test
no_objects = "--no-objects" in sys.argv
# If requested, remove boxes/ramps when creating the env for a clean ground test
# (environment now allows n_ramps==0)
n_boxes = 0 if no_objects else 2
n_ramps = 0 if no_objects else 1
if no_objects:
    print("debug_ctrl_inspect: running with --no-objects (no boxes/ramps)")
env = TeamCosEnv(
    mode="initial",
    target="seeker",
    n_seekers=1,
    n_hiders=2,
    n_boxes=n_boxes,
    n_ramps=n_ramps,
    render_mode=None,
)
# Safe-ground respawn: retry reset until agent z is below threshold (avoid forcing to origin)
MAX_RESPAWN_TRIES = 8
GROUND_Z_THRESH = 0.75  # accept spawns with agent_z < 0.75 (boxes at 0.5 + agent offset ~0.5 -> 1.0)
obs, info = env.reset()
ak = None
for try_i in range(MAX_RESPAWN_TRIES):
    ak = env.seeker_keys[0]
    bid = env.body_ids[ak]
    agent_z = float(env.data.xpos[bid][2])
    if agent_z < GROUND_Z_THRESH:
        break
    # otherwise try resetting again to get a flatter spawn
    obs, info = env.reset()
else:
    # reached max tries; keep current spawn but warn
    print(f"warning: safe-ground respawn: gave up after {MAX_RESPAWN_TRIES} tries, agent_z={agent_z}")
# ensure ak/bid are set (if not set above)
if "ak" not in locals() or ak is None:
    ak = env.seeker_keys[0]
    bid = env.body_ids[ak]
act_ids = env.actuator_ids
print("actuator_ids:", act_ids)
print("override_learnable_policy:", env.override_learnable_policy)
print("_inference_models keys:", list(env._inference_models.keys()))
print("learnable_agent_key:", env.learnable_agent_key)
print("initial pos:", env.data.xpos[bid].copy())

s_fwd_idx = act_ids.get("s_fwd")
s_turn_idx = act_ids.get("s_turn")
ctrl_len = env.action_space.shape[0]
print("seeker actuator indices:", s_fwd_idx, s_turn_idx, "ctrl_len:", ctrl_len)

# Warmup: step env.prep_steps no-op actions so that prep period ends
warmup_steps = getattr(env, "prep_steps", 80)
if warmup_steps > 0:
    print(f"warming up for {warmup_steps} prep steps (no-op actions)")
    zero_a = np.zeros(ctrl_len, dtype=np.float32)
    for _ in range(warmup_steps):
        obs, _, _, _, _ = env.step(zero_a)
    print("post-warmup current_step:", env.current_step)

for val, name in [(1.0, "forward"), (-1.0, "backward")]:
    a = np.zeros(ctrl_len, dtype=np.float32)
    if s_fwd_idx is not None:
        a[s_fwd_idx] = val
    obs, r, _, done, info = env.step(a)
    ctrl = env.data.ctrl.copy()
    try:
        vel = env._body_speed_xy(bid)
    except Exception as e:
        # provide extra debug info when body velocity indices are invalid
        try:
            vadr = env.model.jnt_dofadr[env.model.body_jntadr[bid]]
            qlen = env.data.qvel.shape[0]
            if vadr < qlen:
                qslice = env.data.qvel[vadr : min(vadr + 2, qlen)].copy()
            else:
                qslice = env.data.qvel[vadr:].copy()
        except Exception:
            vadr = None
            qlen = len(env.data.qvel)
            qslice = env.data.qvel.copy()
        print(
            "_body_speed_xy error:",
            e,
            "vadr=",
            vadr,
            "qvel_len=",
            qlen,
            "qvel_slice=",
            qslice,
        )
        vel = float("nan")
    print(
        f"sent {name} ({val}) -> action_indices_set: {[i for i in range(len(a)) if a[i]!=0]} ctrl[first 12]={ctrl[:12]} vel_xy={vel if not np.isnan(vel) else 'nan'} pos={env.data.xpos[bid].copy()} reward={r}"
    )

env.close()
