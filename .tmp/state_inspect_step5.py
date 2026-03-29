import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import mujoco

from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=False, seed=123)
_ = env.reset()
act_dim = env.action_space.shape[0]

# step until 5
for s in range(6):
    a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
    # process actions
    cv, stats = env._process_agent_actions(a)
    env.data.ctrl[:] = cv
    print("--- before capture prephysics step", s, "---")
    print("frame_vis_cache keys sample:", list(env._frame_vis_cache.items())[:5])
    env._capture_prephysics_prev_vis()
    print("after capture_prephysics_prev_vis prev_vis:", {k: v for k, v in list(env._prev_vis.items())})
    print("frame_vis_cache sample after capture:", list(env._frame_vis_cache.items())[:5])
    for _ in range(env.action_repeat):
        mujoco.mj_step(env.model, env.data)
    env._apply_object_constraints()
    env._stabilize_agent_vertical_motion()
    mujoco.mj_forward(env.model, env.data)
    print("--- after forward, before populate ---")
    print("frame_vis_cache sample:", list(env._frame_vis_cache.items())[:5])
    env._populate_prev_vis_and_being_hit()
    print("after populate prev_vis:", {k: v for k, v in list(env._prev_vis.items())})
    print("frame_vis_cache sample after populate:", list(env._frame_vis_cache.items())[:5])
    # inspect pair s,h1
    pair = ("s", "h1")
    v_bid = env.body_ids[pair[0]]
    t_bid = env.body_ids[pair[1]]
    v_pos = env.data.xpos[v_bid][:2]
    t_pos = env.data.xpos[t_bid][:2]
    p1 = np.array([v_pos[0], v_pos[1], 0.4])
    p2 = np.array([t_pos[0], t_pos[1], 0.4])
    eng = bool(env.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))
    print("engine says", eng)
    try:
        rot_idx = env.model.jnt_qposadr[env.qpos_indices[pair[0]]["rot"]]
        rot = float(env.data.qpos[rot_idx])
    except Exception:
        rot = 0.0
    isvis = env._is_vis(v_pos, rot, t_pos, v_bid, t_bid)
    print("_is_vis says", isvis)
    i = env.agent_keys.index(pair[0])
    obs = env._get_obs(i)
    ens = [k for k in env.agent_keys if k != pair[0]]
    ens.sort(key=lambda k: (0 if k.startswith("s" if pair[0].startswith("h") else "h") else 1, k))
    j = ens.index(pair[1])
    en_idx = env.idx.OTHERS[j]
    print("obs visible val", float(obs[en_idx.VISIBLE]))
print("done")
