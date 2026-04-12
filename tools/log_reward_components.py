#!/usr/bin/env python3
import sys
import os
import numpy as np

# Ensure project root is on sys.path so `src` can be imported when
# running scripts from the tools/ directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.envs.hns28_environment import TeamCosEnv
import torch
import glob


def _try_load_checkpoint_for_env(env):
    # search checkpoints directories for a reasonable .pt file
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cand_dirs = [os.path.join(base, 'checkpoints'), os.path.join(base, 'checkpoints', 'keepGoodpkl')]
    files = []
    for d in cand_dirs:
        if os.path.isdir(d):
            files.extend(sorted(glob.glob(os.path.join(d, '*.pt'))))
    if not files:
        return None
    # prefer files mentioning seeker/hider based on env.target
    preferred = None
    key = 'seeker' if env.target == 'seeker' else 'hider'
    for f in files:
        if key in os.path.basename(f).lower():
            preferred = f; break
    ck = preferred or files[0]
    try:
        obj = torch.load(ck, map_location='cpu')
    except Exception:
        return None
    # if it's already a nn.Module, use it directly
    try:
        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, dict):
            # common patterns
            if 'model_state_dict' in obj:
                state = obj['model_state_dict']
            elif 'state_dict' in obj:
                state = obj['state_dict']
            else:
                # if dict appears to be a state_dict (tensor values), assume that
                state = obj

            from src.models.ppo_transformer_v2 import AgentV2
            obs_dim = int(getattr(env.idx, 'total_dim', 256))
            action_dim = 4
            hidden_dim = 128
            seq_len = 8
            model = AgentV2(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, seq_len=seq_len)
            try:
                model.load_state_dict(state)
            except Exception:
                # try loading nested keys
                if 'agent' in state:
                    try:
                        model.load_state_dict(state['agent'])
                    except Exception:
                        return None
                else:
                    return None
        else:
            return None
        model.eval()
        return model
    except Exception:
        return None


def main():
    env = TeamCosEnv(dist_bonus_weight=1.0, team_reward_gain=1.0)
    # try to load a checkpoint and register as inference model
    model = _try_load_checkpoint_for_env(env)
    if model is not None:
        env._inference_models[env.learnable_agent_key] = model
        env.override_learnable_policy = True
        print(f"Loaded checkpoint for {env.learnable_agent_key}")
    else:
        env.override_learnable_policy = False
        print("No usable checkpoint found; running with scripted policies / zero actions")
    env.reset()
    for step in range(100):
        ret = env.step(np.zeros(4))
        if len(ret) == 5:
            obs, rew, _, done, info = ret
        else:
            obs, rew, done, info = ret
        print(step, "reward", float(rew),
              "dist_contrib", getattr(env, "_last_team_dist_contrib", None),
              "base_contrib", getattr(env, "_last_team_base_contrib", None),
              "min_seen", getattr(env, "_last_team_dist", None),
              "applied_fwd", info.get("applied_forward", None),
              "applied_turn", info.get("dbg_last_ctrl_t", None),
              "applied_fwd_dbg", info.get("dbg_last_ctrl_f", None))
        if done:
            env.reset()


if __name__ == '__main__':
    main()
