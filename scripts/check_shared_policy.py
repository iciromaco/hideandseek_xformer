"""
Runtime check: print per-agent policy object ids, history, and RNG summaries.
Run: python3 scripts/check_shared_policy.py
"""

import os
import sys
# make project root importable when running scripts directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from src.envs.hns_environment import TeamCosEnv

np.random.seed(12345)
torch.manual_seed(12345)


def short_bytes(x, n=8):
    return tuple(int(b) for b in x[:n])


def main():
    env = TeamCosEnv(debug_mode=True, n_seekers=1, n_hiders=2)

    print("agent_keys:", env.agent_keys)
    print("policy_adapter shared_team_policy:", env.shared_team_policy)
    print("shared_policy_model id:", id(env.shared_policy_model) if env.shared_policy_model is not None else None)

    # RNG summaries
    np_state = np.random.get_state()[1]
    torch_state = torch.get_rng_state()
    print("numpy rng sample:", tuple(np_state[:6]))
    print("torch rng first_bytes:", short_bytes(torch_state.numpy()))

    for ak in env.agent_keys:
        npc = env.npcs.get(ak)
        print(f"\n--- {ak} ---")
        print("npc id:", id(npc), "type:", type(npc))
        # policy histories: print all keys that match this agent
        hist_keys = [k for k in getattr(env, '_policy_histories', {}) if k[0] == ak]
        if hist_keys:
            for hk in hist_keys:
                h = env._policy_histories[hk]
                print("hist key:", hk, "buffer_shape:", h['buffer'].shape, "ptr:", h.get('ptr'))
        else:
            print("no policy history for agent")
        print("has_inference_model:", ak in getattr(env, '_inference_models', {}), "model id:", id(env._inference_models.get(ak)) if ak in env._inference_models else None)
        print("is_shared_enabled:", env.policy_adapter.is_shared_enabled_for_agent(ak))

        # produce a normalized obs and query adapter directly to see fallback/source
        try:
            idx = env.agent_keys.index(ak)
            raw_obs = env._get_obs(idx)
            norm_obs = env._normalize_obs(raw_obs)
            act = env.policy_adapter.get_action(ak, norm_obs)
            print("sample action:", act)
        except Exception as e:
            print("get_action error:", e)

    # quick env step with zero action for learnable agent to exercise update hooks
    try:
        learnable_idx = env.agent_keys.index(env.learnable_agent_key)
        zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        obs, info = env.reset()
        print("\nreset obs shape:", obs.shape if hasattr(obs, 'shape') else type(obs))
        step_out = env.step(zero_action)
        print("step returned types:", tuple(type(x) for x in step_out))
    except Exception as e:
        print("env step error:", e)


if __name__ == '__main__':
    main()
