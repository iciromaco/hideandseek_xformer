import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import mujoco

from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=False, seed=123)
_ = env.reset()
act_dim = env.action_space.shape[0]

# find a step index where mismatch happened earlier: 5
step_to_inspect = 5
for s in range(step_to_inspect + 1):
    a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
    # process action
    cv, stats = env._process_agent_actions(a)
    env.data.ctrl[:] = cv
    print(f"--- step {s} after process_agent_actions ---")
    print("prev_vis keys sample:", list(env._prev_vis.items())[:5])
    # capture prephysics
    env._capture_prephysics_prev_vis()
    print("after capture_prephysics_prev_vis sample:", list(env._prev_vis.items())[:5])
    for _ in range(env.action_repeat):
        mujoco.mj_step(env.model, env.data)
    env._apply_object_constraints()
    env._stabilize_agent_vertical_motion()
    mujoco.mj_forward(env.model, env.data)
    print("after forward, before populate _prev_vis")
    print("prev_vis sample:", list(env._prev_vis.items())[:5])
    env._populate_prev_vis_and_being_hit()
    print("after populate_prev_vis sample:", list(env._prev_vis.items())[:10])
    # compute reward
    _ = env._compute_team_reward()
    # compute next obs for first agent
    obs0 = env._get_obs(0)
    print("obs0 visible entries:", [(k, obs0[env.idx.OTHERS[i].VISIBLE]) for i, k in enumerate(env.agent_keys[: len(env.idx.OTHERS)])])
print("done")
