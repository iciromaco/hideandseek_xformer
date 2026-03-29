import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=False, seed=123)
_ = env.reset()
act_dim = env.action_space.shape[0]
pair = ("s", "h1")
for step in range(120):
    a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
    res = env.step(a)
    pre = env._prev_vis.get(pair, None)
    i = env.agent_keys.index(pair[0])
    ens = [k for k in env.agent_keys if k != pair[0]]
    ens.sort(key=lambda k: (0 if k.startswith("s" if pair[0].startswith("h") else "h") else 1, k))
    j = ens.index(pair[1])
    en_idx = env.idx.OTHERS[j]
    obs = env._get_obs(i)
    obs_vis = bool(obs[en_idx.VISIBLE] > 0.5)
    v_bid = env.body_ids[pair[0]]
    t_bid = env.body_ids[pair[1]]
    p1 = np.array([env.data.xpos[v_bid][0], env.data.xpos[v_bid][1], 0.4])
    p2 = np.array([env.data.xpos[t_bid][0], env.data.xpos[t_bid][1], 0.4])
    eng = bool(env.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))
    print(step, pair, "prev_vis", pre, "obs_vis", obs_vis, "engine", eng)
    if not eng and obs_vis:
        print("MISMATCH at step", step)
        print("ens ordering", ens)
        print("prev_vis entries:", {k: env._prev_vis.get((pair[0], k), None) for k in ens})
        break
