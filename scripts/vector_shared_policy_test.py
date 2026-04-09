import sys
sys.path.insert(0, '.')
from functools import partial
import numpy as np
import torch
from models.ppo_transformer_v2 import AgentV2
from main27_train_final import build_env
import gymnasium as gym

NUM_ENVS = 4
SEQ_LEN = 8
HIDDEN_DIM = 128

def main():
    print(f"Creating {NUM_ENVS} AsyncVectorEnv instances...")
    raw_common = __import__('tomllib').load(open('configs/hparams_main27.toml','rb')).get('runtime', {}).get('common', {})
    # Map runtime.common naming (num_seekers etc.) to ENV_CONFIG naming expected by TeamCosEnv
    env_config_snapshot = {
        'n_seekers': raw_common.get('num_seekers', 1),
        'n_hiders': raw_common.get('num_hiders', 2),
        'n_boxes': raw_common.get('num_boxes', 2),
        'n_ramps': raw_common.get('num_ramps', 1),
        'mode4_sdf_cell_size': raw_common.get('mode4_sdf_cell_size', 0.05),
        'show_turn_lines': raw_common.get('show_turn_lines', False),
        'debug_log_interval_steps': raw_common.get('debug_log_interval_steps', 200),
        'action_repeat': raw_common.get('action_repeat', 10),
    }

    vec_envs = gym.vector.AsyncVectorEnv([
        partial(build_env, 'refinement', 'hider', env_config_snapshot, None) for _ in range(NUM_ENVS)
    ])
    print('Vector env created')

    # Create a test AgentV2 and state
    # Get obs/act from a reference env via call
    ref = build_env('refinement', 'hider', env_config_snapshot, None)
    obs_dim = ref.observation_space.shape[0]
    act_dim = ref.action_space.shape[0]
    print(f"ref obs_dim={obs_dim}, act_dim={act_dim}")
    agent = AgentV2(obs_dim, act_dim, HIDDEN_DIM, SEQ_LEN)
    state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}

    print('Calling vec_envs.call("set_shared_team_policy_state", state, SEQ_LEN, HIDDEN_DIM)')
    try:
        results = vec_envs.call('set_shared_team_policy_state', state, SEQ_LEN, HIDDEN_DIM)
        print('Results:', results)
        enabled = sum(1 for r in results if bool(r))
        print(f'Enabled on {enabled}/{len(results)} envs')
    except Exception as e:
        print('vec_envs.call failed:', e)

    print('Cleanup')
    try:
        vec_envs.close()
    except Exception:
        pass
    ref.close()
    print('Done')


if __name__ == '__main__':
    main()
