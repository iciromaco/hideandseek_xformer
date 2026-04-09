#!/usr/bin/env python3
import json
import os
import math
import numpy as np
from datetime import datetime

from src.envs.hns28_environment import TeamCosEnv

OUT_DIR = "logs/reward_mismatches"
os.makedirs(OUT_DIR, exist_ok=True)


def capture_pre(env):
    pre_obs_by_agent = {}
    pre_state_by_agent = {}
    for ak in env.agent_keys:
        try:
            bid = env.body_ids[ak]
            pre_obs_by_agent[bid] = env._get_obs(env.agent_keys.index(ak)).copy()
            pre_state_by_agent[bid] = {
                "pos": env.data.xpos[bid][:2].copy(),
                "rot": float(env.data.qpos[env.model.jnt_qposadr[env.qpos_indices[ak]["rot"]]]),
            }
        except Exception:
            pre_obs_by_agent[env.agent_keys.index(ak)] = env._get_obs(env.agent_keys.index(ak)).copy()
            pre_state_by_agent[env.agent_keys.index(ak)] = None
    return pre_obs_by_agent, pre_state_by_agent


def analyze_frame(env):
    # return per-pair diagnostics
    pairs = []
    for sk in env.seeker_keys:
        sid = env.body_ids[sk]
        obs_post = env._get_obs(env.agent_keys.index(sk))
        for hk in env.hider_keys:
            hid = env.body_ids[hk]
            # find en_idx mapping for this seeker
            ens = env._ens_orderings.get(sk, [k for k in env.agent_keys if k != sk])
            en_idx = None
            for i, e in enumerate(ens[: len(env.idx.OTHERS)]):
                if e == hk:
                    en_idx = env.idx.OTHERS[i]
                    break
            obs_vis = None if en_idx is None else float(obs_post[en_idx.VISIBLE]) > 0.5
            # state vis
            state_vis = bool(env._cached_vis.get((sid, hid), False))
            # distances
            # state dist (world coords)
            hpos = env.data.xpos[hid][:2]
            spos = env.data.xpos[sid][:2]
            state_dist = float(math.hypot(hpos[0]-spos[0], hpos[1]-spos[1]))
            # obs dist (agent-frame rel)
            obs_rel_x = None
            obs_rel_y = None
            obs_dist = None
            if en_idx is not None:
                obs_rel_x = float(obs_post[en_idx.REL_X])
                obs_rel_y = float(obs_post[en_idx.REL_Y])
                obs_dist = float(math.hypot(obs_rel_x, obs_rel_y))
            pairs.append({
                "seeker": sk,
                "hider": hk,
                "sid": int(sid),
                "hid": int(hid),
                "state_vis": bool(state_vis),
                "obs_vis": (None if obs_vis is None else bool(obs_vis)),
                "state_dist": state_dist,
                "obs_rel_x": obs_rel_x,
                "obs_rel_y": obs_rel_y,
                "obs_dist": obs_dist,
            })
    return pairs


def main():
    env = TeamCosEnv()
    env.reset()
    # warmup
    while env.current_step <= env.prep_steps:
        env.step(np.zeros(4, dtype=np.float32))
    N = 50
    mismatches = []
    out_file = os.path.join(OUT_DIR, f"mismatch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    with open(out_file, 'w') as fh:
        for step in range(N):
            pre_obs, pre_state = capture_pre(env)
            env.step(np.zeros(4, dtype=np.float32))
            try:
                s_res = env._compute_team_reward_state(pre_state_by_agent=pre_state)
            except Exception as e:
                s_res = ("ERR", str(e))
            # observational implementation removed; use state result for comparison
            o_res = s_res

            equal = False
            if isinstance(s_res, tuple) and isinstance(o_res, tuple) and len(s_res) == 3 and len(o_res) == 3:
                diffs = [abs(float(s_res[i]) - float(o_res[i])) for i in range(3)]
                equal = not any(d > 1e-6 for d in diffs)
            if not equal:
                pairs = analyze_frame(env)
                rec = {
                    "step": env.current_step,
                    "s_res": s_res,
                    "o_res": o_res,
                    "pairs": pairs,
                }
                fh.write(json.dumps(rec) + "\n")
                mismatches.append(rec)
    print("Wrote", len(mismatches), "mismatches to", out_file)
    if mismatches:
        print("Sample:", json.dumps(mismatches[0], indent=2))

if __name__ == '__main__':
    main()
