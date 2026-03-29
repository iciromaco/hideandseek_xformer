import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())
from src.envs.hns_environment import TeamCosEnv


def main(seed=123, trials=200):
    env = TeamCosEnv(debug_mode=False, seed=seed)
    _ = env.reset()
    # step a few times to populate dynamics
    act_dim = env.action_space.shape[0]
    for _ in range(3):
        a = np.random.uniform(-1, 1, size=(act_dim,)).astype(np.float32)
        env.step(a)

    agents = list(env.body_ids.keys())
    objs = list(env.obj_body_map.items())  # list of (tk, bid)

    mismatches = []
    stats = {"agree_true": 0, "agree_false": 0, "v1_true_v2_false": 0, "v1_false_v2_true": 0}

    for i in range(trials):
        ak = np.random.choice(agents)
        tk, bid = objs[np.random.randint(len(objs))]
        aid = env.body_ids[ak]
        apos = env.data.xpos[aid][:2].copy()
        tpos = env.data.xpos[bid][:2].copy()
        try:
            rot = float(env.data.qpos[env.model.jnt_qposadr[env.qpos_indices[ak]["rot"]]])
        except Exception:
            rot = 0.0

        try:
            v1 = bool(env.vis_engine.is_visible(np.array([apos[0], apos[1], 0.4]), np.array([tpos[0], tpos[1], 0.4]), body_exclude=aid, target_body_id=bid))
        except Exception as e:
            v1 = f"err:{e}"
        try:
            v2 = bool(env._is_vis(apos, rot, tpos, aid, bid))
        except Exception as e:
            v2 = f"err:{e}"

        if isinstance(v1, bool) and isinstance(v2, bool):
            if v1 and v2:
                stats["agree_true"] += 1
            elif not v1 and not v2:
                stats["agree_false"] += 1
            elif v1 and not v2:
                stats["v1_true_v2_false"] += 1
                if len(mismatches) < 10:
                    mismatches.append((ak, tk, apos.tolist(), tpos.tolist(), rot, v1, v2))
            elif not v1 and v2:
                stats["v1_false_v2_true"] += 1
                if len(mismatches) < 10:
                    mismatches.append((ak, tk, apos.tolist(), tpos.tolist(), rot, v1, v2))
        else:
            if len(mismatches) < 10:
                mismatches.append((ak, tk, apos.tolist(), tpos.tolist(), rot, v1, v2))

    print("Comparison results:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"mismatch_examples (<=10): {len(mismatches)}")
    for ex in mismatches:
        print(ex)


if __name__ == "__main__":
    main()
