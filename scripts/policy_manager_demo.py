import sys
sys.path.insert(0, '.')
from src.policy_manager import PolicyManager
from main27_train_final import build_env
from models.ppo_transformer_v2 import AgentV2

print('PolicyManager demo: build env and apply a test state as shared')
raw_common = __import__('tomllib').load(open('configs/hparams_main27.toml','rb')).get('runtime', {}).get('common', {})
config = {
	'n_seekers': raw_common.get('num_seekers', 1),
	'n_hiders': raw_common.get('num_hiders', 2),
	'n_boxes': raw_common.get('num_boxes', 2),
	'n_ramps': raw_common.get('num_ramps', 1),
}
env = build_env('refinement', 'hider', config, None)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
agent = AgentV2(obs_dim, act_dim, 128, 8)
state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
pm = PolicyManager(seq_len=8, hidden_dim=128)
res = pm.apply_to_envs(env, None, [], state, shared=True)
print('apply_to_envs returned:', res)
env.close()
print('Done')
