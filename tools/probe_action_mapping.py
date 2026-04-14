import sys, os, time
sys.path.append(os.path.abspath('src'))
from envs.hns28_environment import TeamCosEnv

print('Creating env...')
env = TeamCosEnv(mode='refinement', target='seeker', n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1, action_repeat=8)
obs, info = env.reset()
print('Reset done')
# advance past preparation period so learner's actions are applied
prep = env.prep_steps + 1
print(f'Advancing {prep} prep steps...')
for i in range(prep):
    env.step([0.0, 0.0, 0.0, 0.0])

tests = [([1.0,0.0,0.0,0.0], 'full_forward'), ([-1.0,0.0,0.0,0.0], 'full_backward'), ([0.0,1.0,0.0,0.0], 'full_left'), ([0.0,-1.0,0.0,0.0], 'full_right')]
for act, name in tests:
    print('\n--- Test:', name)
    next_obs, reward, term, done, info = env.step(act)
    print('applied_forward:', info.get('applied_forward'))
    print('dbg_last_ctrl_f:', info.get('dbg_last_ctrl_f'))
    print('dbg_last_ctrl_t:', info.get('dbg_last_ctrl_t'))
    print('agent_vx:', info.get('agent_vx'), 'agent_vy:', info.get('agent_vy'), 'agent_vz:', info.get('agent_vz'))
    print('min_world_dist:', info.get('min_world_dist'))
    print('reward:', reward)

print('\nDone probe')
env.close()
