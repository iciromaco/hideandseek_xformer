import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=False, seed=123)
_ = env.reset()
act_dim = env.action_space.shape[0]

pair = ("h1", "s")
for step in range(25):
    a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
    res = env.step(a)
    pre = None
    post = None
    try:
        pre = env._prev_vis.get(pair, None)
    except Exception:
        pre = None
    # compute obs for ak
    try:
        i = env.agent_keys.index(pair[0])
        obs = env._get_obs(i)
        # reproduce `ens` ordering from _get_obs
        ens = [k for k in env.agent_keys if k != pair[0]]
        ens.sort(key=lambda k: (0 if k.startswith("s" if pair[0].startswith("h") else "h") else 1, k))
        j = ens.index(pair[1])
        en_idx = env.idx.OTHERS[j]
        obs_vis = bool(obs[en_idx.VISIBLE] > 0.5)
    except Exception as e:
        obs_vis = f"err:{e}"
    try:
        v_bid = env.body_ids[pair[0]]
        t_bid = env.body_ids[pair[1]]
        v_pos = env.data.xpos[v_bid][:2]
        t_pos = env.data.xpos[t_bid][:2]
        p1 = np.array([v_pos[0], v_pos[1], 0.4])
        p2 = np.array([t_pos[0], t_pos[1], 0.4])
        eng = bool(env.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))
    except Exception as e:
        eng = f"err:{e}"
    try:
        rot_idx = env.model.jnt_qposadr[env.qpos_indices[pair[0]]["rot"]]
        rot = float(env.data.qpos[rot_idx])
    except Exception:
        rot = 0.0
    try:
        isvis_cached = env._is_vis(env.data.xpos[v_bid][:2], rot, env.data.xpos[t_bid][:2], v_bid, t_bid)
    except Exception as e:
        isvis_cached = f"err:{e}"

    print(step, "pair", pair, "prev_vis", pre, "obs_vis", obs_vis, "engine", eng)
    print("  _is_vis (cached) before _get_obs:", isvis_cached)
    if step == 5:
        # print detailed per-ens info
        ens = [k for k in env.agent_keys if k != pair[0]]
        ens.sort(key=lambda k: (0 if k.startswith("s" if pair[0].startswith("h") else "h") else 1, k))
        print("ens ordering for", pair[0], ens)
        obs_i = env._get_obs(env.agent_keys.index(pair[0]))
        for i_en, enm in enumerate(ens):
            pv = env._prev_vis.get((pair[0], enm), None)
            try:
                v_bid = env.body_ids[pair[0]]
                t_bid = env.body_ids[enm]
                p1 = np.array([env.data.xpos[v_bid][0], env.data.xpos[v_bid][1], 0.4])
                p2 = np.array([env.data.xpos[t_bid][0], env.data.xpos[t_bid][1], 0.4])
                eng_local = bool(env.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))
            except Exception:
                eng_local = None
            en_idx = env.idx.OTHERS[i_en]
            print("  enm", enm, "prev_vis", pv, "engine", eng_local, "obs_visible_val", float(obs_i[en_idx.VISIBLE]))
        break
