#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from src.envs.hns28_environment import TeamCosEnv
from src.models.ppo_transformer_v2 import AgentV2

CKPT = 'checkpoints/keepGoodpkl/HNS_V27_seeker_s1_h2_b2_r1.pt'

def load_model_for_env(env, ckpt_path):
    try:
        obj = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print('ERR load:', e)
        return None
    # if module
    if isinstance(obj, torch.nn.Module):
        m = obj
        m.eval()
        return m
    if isinstance(obj, dict):
        # try common keys
        if 'model_state_dict' in obj:
            state = obj['model_state_dict']
        elif 'state_dict' in obj:
            state = obj['state_dict']
        else:
            state = obj
        obs_dim = int(getattr(env.idx, 'total_dim', 256))
        action_dim = 4
        hidden_dim = 256
        seq_len = 8
        try:
            m = AgentV2(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, seq_len=seq_len)
            m.load_state_dict(state)
            m.eval()
            return m
        except Exception as e:
            print('ERR build/load state:', e)
            return None
    print('Unknown checkpoint format:', type(obj))
    return None


def main():
    env = TeamCosEnv(dist_bonus_weight=1.0, team_reward_gain=1.0)
    print('Trying checkpoint:', CKPT)
    model = load_model_for_env(env, CKPT)
    if model is None:
        print('Failed to load model from checkpoint')
        return
    env._inference_models[env.learnable_agent_key] = model
    env.override_learnable_policy = True
    print('Model loaded and registered for', env.learnable_agent_key)
    obs = env.reset()
    for step in range(100):
        ret = env.step(np.zeros(4))
        if len(ret) == 5:
            obs, rew, _, done, info = ret
        else:
            obs, rew, done, info = ret
        print(step, 'reward', float(rew), 'applied_fwd', info.get('applied_forward'), 'applied_turn', info.get('dbg_last_ctrl_t'))
        if done:
            break

if __name__ == '__main__':
    import numpy as np
    main()
