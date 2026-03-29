import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=False, seed=123)
_ = env.reset()
act_dim = env.action_space.shape[0]

mismatches = []
for step in range(200):
    a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
    res = env.step(a)
    # after step, observations computed and cached as _cached_obs? but compute per-agent
    for i, ak in enumerate(env.agent_keys):
        obs = env._get_obs(i)
        # for each other
        for j, enm in enumerate(env.agent_keys[: len(env.idx.OTHERS)]):
            en_idx = env.idx.OTHERS[j]
            obs_vis = bool(obs[en_idx.VISIBLE] > 0.5)
            try:
                v_bid = env.body_ids[ak]
                t_bid = env.body_ids[enm]
                v_pos = env.data.xpos[v_bid][:2]
                t_pos = env.data.xpos[t_bid][:2]
                p1 = np.array([v_pos[0], v_pos[1], 0.4])
                p2 = np.array([t_pos[0], t_pos[1], 0.4])
                eng_vis = bool(env.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))
            except Exception:
                eng_vis = None
            if eng_vis is False and obs_vis is True:
                mismatches.append((step, ak, enm, v_pos.tolist(), t_pos.tolist()))
    if len(mismatches) > 20:
        break
print("found mismatches count:", len(mismatches))
for m in mismatches[:20]:
    print(m)
