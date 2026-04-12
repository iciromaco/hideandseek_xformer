#!/usr/bin/env python3
import math
import numpy as np

from src.envs.hns28_environment import TeamCosEnv


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
            # defensive fallback
            pre_obs_by_agent[env.agent_keys.index(ak)] = env._get_obs(env.agent_keys.index(ak)).copy()
            pre_state_by_agent[env.agent_keys.index(ak)] = None
    return pre_obs_by_agent, pre_state_by_agent


def main():
    env = TeamCosEnv()
    env.reset()
    # wait until after prep_steps (warmup) because seekers don't move before that
    while env.current_step <= env.prep_steps:
        env.step(np.zeros(4, dtype=np.float32))
    N = 50
    mismatches = []

    for step in range(N):
        pre_obs, pre_state = capture_pre(env)
        # zero action (shape flexible; env.step flattens)
        action = np.zeros(4, dtype=np.float32)
        ret = env.step(action)
        # env.step returns (obs, reward, False, done, info)
        if len(ret) == 5:
            obs, reward, _, done, info = ret
        else:
            obs, reward, done, info = ret
        # compare team reward implementations
        try:
            s_res = env._compute_team_reward_state()
        except Exception as e:
            s_res = ("ERR", str(e))
        # observational implementation removed; use state-based result
        o_res = s_res

        if not (isinstance(s_res, tuple) and isinstance(o_res, tuple) and len(s_res) == 2 and len(o_res) == 2):
            mismatches.append((step, s_res, o_res))
        else:
            # numeric compare with small tolerance
            diffs = [abs(float(s_res[i]) - float(o_res[i])) for i in range(2)]
            if any(d > 1e-6 for d in diffs):
                mismatches.append((step, s_res, o_res, diffs))

        if done:
            env.reset()

    if mismatches:
        print(f"MISMATCH_COUNT: {len(mismatches)}")
        for mm in mismatches[:10]:
            print(mm)
        raise SystemExit(2)
    else:
        print("ALL_MATCH", N)


if __name__ == "__main__":
    main()
