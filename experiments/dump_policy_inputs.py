#!/usr/bin/env python3
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.envs.hns_environment import TeamCosEnv

LOG = {
    'per_agent_norm_obs': {},
    'per_agent_seq': {}
}

class LoggerPolicy:
    def __init__(self, env, agent_key):
        self.env = env
        self.agent_key = agent_key
        LOG['per_agent_norm_obs'].setdefault(agent_key, [])
        LOG['per_agent_seq'].setdefault(agent_key, [])
    def __call__(self, norm_obs):
        # record normalized obs and the seq that would be passed to a model
        LOG['per_agent_norm_obs'][self.agent_key].append(norm_obs.tolist())
        try:
            seq = self.env._get_policy_history_seq(self.agent_key, self.env._inference_seq_lens.get(self.agent_key, 8), norm_obs)
            LOG['per_agent_seq'][self.agent_key].append(seq.tolist())
        except Exception as e:
            LOG['per_agent_seq'][self.agent_key].append(str(e))
        # return zeros action
        return np.zeros(4, dtype=np.float32)


def main():
    env = TeamCosEnv()
    env.debug_mode = True
    # ensure learnable agent is seeker for this test
    env.learnable_agent_key = env.seeker_keys[0]
    env.learnable_agent_index = env.agent_keys.index(env.learnable_agent_key)
    # set inference_seq_lens for agents
    for ak in env.agent_keys:
        env._inference_seq_lens[ak] = 8
    # install LoggerPolicy for all agents
    for ak in env.agent_keys:
        env.inference_policies[ak] = LoggerPolicy(env, ak)
    # run some steps
    obs, info = env.reset()
    steps = 200
    for t in range(steps):
        # action doesn't matter because policies are in env for non-learnable; for learnable we force override
        env.override_learnable_policy = True
        a = np.zeros(4, dtype=np.float32)
        out = env.step(a)
        # step returns either 4- or 5-tuple
        # we only need to continue
    # write logs
    with open('experiments/policy_input_dump.json','w') as f:
        json.dump(LOG, f)
    print('wrote experiments/policy_input_dump.json')

if __name__=='__main__':
    main()
