import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.envs.hns_environment import TeamCosEnv


def main(seed=123, steps=100):
    env = TeamCosEnv(debug_mode=False, seed=seed)
    _ = env.reset()
    act_dim = env.action_space.shape[0]
    mismatches = []
    stats = {"pre_true_post_false": 0, "pre_false_post_true": 0, "agree": 0}
    for i in range(steps):
        a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
        # capture prephysics snapshot manually
        env._capture_prephysics_prev_vis()
        pre_snapshot = dict(env._prev_vis)

        # call step (which will capture prephysics internally and then
        # populate post-physics prev_vis as implemented in step())
        res = env.step(a)

        # now read the prev_vis stored after step (should be post-physics)
        post_snapshot = dict(env._prev_vis)

        # compute true post visibility directly from engine for verification
        for viewer in env.agent_keys:
            for target in env.agent_keys:
                if viewer == target:
                    continue
                pre = pre_snapshot.get((viewer, target), None)
                post = post_snapshot.get((viewer, target), None)
                try:
                    v_bid = env.body_ids[viewer]
                    t_bid = env.body_ids[target]
                    v_pos = env.data.xpos[v_bid][:2]
                    t_pos = env.data.xpos[t_bid][:2]
                    p1 = np.array([v_pos[0], v_pos[1], 0.4])
                    p2 = np.array([t_pos[0], t_pos[1], 0.4])
                    engine_post = bool(env.vis_engine.is_visible(p1, p2, body_exclude=v_bid, target_body_id=t_bid))
                except Exception:
                    engine_post = None
                if isinstance(pre, bool) and isinstance(post, bool):
                    if pre and not post:
                        stats["pre_true_post_false"] += 1
                        if len(mismatches) < 10:
                            mismatches.append((i, viewer, target, pre, post, v_pos.tolist(), t_pos.tolist()))
                    elif not pre and post:
                        stats["pre_false_post_true"] += 1
                        if len(mismatches) < 10:
                            mismatches.append((i, viewer, target, pre, post, v_pos.tolist(), t_pos.tolist()))
                    else:
                        stats["agree"] += 1
    print("stats:", stats)
    print("examples:", mismatches)


if __name__ == "__main__":
    main()
