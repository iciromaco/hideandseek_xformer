import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=False, seed=123)
_ = env.reset()
act_dim = env.action_space.shape[0]

mismatches = []
stats = {"obs_true_engine_false": 0, "obs_false_engine_true": 0, "agree": 0}
for step in range(200):
    a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
    res = env.step(a)
    for i, ak in enumerate(env.agent_keys):
        obs = env._get_obs(i)
        # build ens ordering same as in _get_obs
        ens = [k for k in env.agent_keys if k != ak]
        ens.sort(key=lambda k: (0 if k.startswith("s" if ak.startswith("h") else "h") else 1, k))
        for j, enm in enumerate(ens[: len(env.idx.OTHERS)]):
            en_idx = env.idx.OTHERS[j]
            obs_vis = bool(obs[en_idx.VISIBLE] > 0.5)
            try:
                v_bid = env.body_ids[ak]
                t_bid = env.body_ids[enm]
                p1 = np.array([env.data.xpos[v_bid][0], env.data.xpos[v_bid][1], 0.4])
                p2 = np.array([env.data.xpos[t_bid][0], env.data.xpos[t_bid][1], 0.4])
                eng_vis = bool(env.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))
            except Exception:
                eng_vis = None
            if isinstance(eng_vis, bool):
                if obs_vis and not eng_vis:
                    stats["obs_true_engine_false"] += 1
                    if len(mismatches) < 20:
                        mismatches.append((step, ak, enm, env.data.xpos[v_bid][:2].tolist(), env.data.xpos[t_bid][:2].tolist()))
                elif not obs_vis and eng_vis:
                    stats["obs_false_engine_true"] += 1
                else:
                    stats["agree"] += 1
print("stats:", stats)
print("examples:", mismatches)
