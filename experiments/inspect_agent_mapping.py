import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.envs.hns_environment import TeamCosEnv
from experiments.utils import prepare_env


def main():
    env = TeamCosEnv(debug_mode=False, target="seeker")
    env = prepare_env(env, action_repeat=None, place_far=True)

    print('agent_keys:', env.agent_keys)
    print('learnable_agent_key:', env.learnable_agent_key)
    try:
        print('body_ids keys:', list(env.body_ids.keys()))
    except Exception:
        print('no body_ids')
    try:
        print('qpos_indices keys:', list(getattr(env, 'qpos_indices', {}).keys()))
    except Exception:
        pass

    # list model body names for quick lookup
    try:
        names = [env.model.body(i).name for i in range(env.model.nbody)]
        print('model body count:', env.model.nbody)
        print('some model bodies:', names[:40])
    except Exception as e:
        print('could not list model bodies:', e)

    # show mapping of agents -> body id and current xvelp
    for k in env.agent_keys:
        try:
            bid = env.body_ids[k]
            xpos = env.data.xpos[bid].tolist()
            xvelp = env.data.xvelp[bid].tolist()
            print(f'{k}: body_id={bid}, xpos={xpos}, xvelp={xvelp}')
        except Exception as e:
            print(f'{k}: error {e}')

    # step a few times and print info/applied_forward and vx for each agent
    print('\n--- stepping 10 steps ---')
    a = env.action_space.sample() * 0.0
    for i in range(10):
        obs, rew, term, trunc, info = env.step(a)
        print(f'step {i}, applied_forward={info.get("applied_forward")}, info_keys={list(info.keys())}')
        for k in env.agent_keys:
            try:
                bid = env.body_ids[k]
                vx = float(env.data.xvelp[bid][0])
                print(f'  {k} vx={vx}')
            except Exception as e:
                print(f'  {k} vx error {e}')
    print('done')

if __name__ == '__main__':
    main()
