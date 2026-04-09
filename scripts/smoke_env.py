import sys
sys.path.insert(0, '.')
from src.envs.hns_environment import TeamCosEnv
import numpy as np
from models.ppo_transformer_v2 import AgentV2
import torch

print("Creating TeamCosEnv (render_mode=None) ...")
env = TeamCosEnv(render_mode=None)
print("Created TeamCosEnv")

# Test shared-team policy API: construct a minimal AgentV2 and inject its state
try:
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    print(f"Creating test AgentV2 obs_dim={obs_dim} act_dim={act_dim}")
    test_agent = AgentV2(obs_dim, act_dim, 128, 8)
    state = {k: v.detach().cpu().clone() for k, v in test_agent.state_dict().items()}
    print("Applying shared team policy state via env.policy_adapter or env.set_shared_team_policy_state(...)")
    ok = False
    try:
        if hasattr(env, 'policy_adapter'):
            ok = env.policy_adapter.set_shared_team_policy_state(state, seq_len=8, hidden_dim=128)
        else:
            ok = env.set_shared_team_policy_state(state, seq_len=8, hidden_dim=128)
    except Exception as e:
        print("set_shared_team_policy_state raised:", e)
    print("set_shared_team_policy_state returned:", bool(ok))
except Exception as e:
    print("Shared policy injection test skipped (error):", e)

print("Resetting environment...")
reset_ret = env.reset()
print("reset returned type:", type(reset_ret))
try:
    if isinstance(reset_ret, tuple):
        print("reset tuple length:", len(reset_ret))
    else:
        print("reset repr:", reset_ret)
except Exception:
    pass

action = env.action_space.sample()
print("Sampled action:", action)

step_ret = env.step(action)
print("step() returned:", type(step_ret))
# Try to pretty-print common step return shapes
try:
    if isinstance(step_ret, tuple):
        print("step tuple length:", len(step_ret))
        for i, v in enumerate(step_ret):
            print(f"  [{i}] type={type(v)} repr={v if i<2 else '...'}")
    else:
        print(step_ret)
except Exception as e:
    print("Error printing step return:", e)

print("Done")
