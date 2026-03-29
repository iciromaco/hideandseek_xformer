import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(debug_mode=False, seed=123)
_ = env.reset()
act_dim = env.action_space.shape[0]
a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
# build ctrl
cv, stats = env._process_agent_actions(a)
env.data.ctrl[:] = cv
# capture prephysics
env._capture_prephysics_prev_vis()
pre = dict(env._prev_vis)
# one physics step
mujoco = __import__("mujoco")
for _ in range(env.action_repeat):
    mujoco.mj_step(env.model, env.data)
# forward
mujoco.mj_forward(env.model, env.data)
# populate post
env._populate_prev_vis_and_being_hit()
post = dict(env._prev_vis)
# compare for some pairs
pairs = []
for v in env.agent_keys:
    for t in env.agent_keys:
        if v == t:
            continue
        pv = pre.get((v, t), None)
        po = post.get((v, t), None)
        pairs.append((v, t, pv, po))

for e in pairs[:10]:
    print(e)

# compare post to vis_engine on current positions
mism = 0
for v in env.agent_keys:
    for t in env.agent_keys:
        if v == t:
            continue
        p_bid = env.body_ids[v]
        t_bid = env.body_ids[t]
        vpos = env.data.xpos[p_bid][:2]
        tpos = env.data.xpos[t_bid][:2]
        p1 = np.array([vpos[0], vpos[1], 0.4])
        p2 = np.array([tpos[0], tpos[1], 0.4])
        try:
            eng = bool(env.vis_engine.is_visible(p1, p2, body_exclude=p_bid, target_body_id=t_bid))
        except Exception as e:
            eng = None
        if post.get((v, t), None) != eng:
            mism += 1
            print("mismatch", v, t, post.get((v, t), None), eng)
print("total mismatches post vs engine:", mism)
