import sys
import traceback
from pprint import pprint
import numpy as np

sys.path.insert(0, '.')

try:
    from src.envs.hns28_environment import TeamCosEnv
except Exception as e:
    print('IMPORT_ERROR', e)
    traceback.print_exc()
    raise


def build_pre_dicts(env):
    pre_obs_by_agent = {}
    pre_state_by_agent = {}
    for i, ak in enumerate(env.agent_keys):
        try:
            bid = env._resolve_body_id(ak)
        except Exception:
            bid = i
        try:
            obs = env._get_obs(i)
        except Exception:
            obs = None
        try:
            pos = env._body_pos_copy(bid)
        except Exception:
            pos = None
        try:
            rot = env._agent_rot(ak)
        except Exception:
            rot = None
        if obs is not None:
            pre_obs_by_agent[bid] = obs.copy()
        if pos is not None or rot is not None:
            pre_state_by_agent[bid] = {'pos': pos, 'rot': rot}
    return pre_obs_by_agent, pre_state_by_agent


def main():
    try:
        env = TeamCosEnv(debug_mode=False, render_mode=None, action_repeat=1)
    except Exception as e:
        print('ENV_INIT_ERROR', e)
        traceback.print_exc()
        return

    try:
        # reset returns (obs, info)
        obs, info = env.reset()
    except Exception as e:
        print('RESET_ERROR', e)
        traceback.print_exc()
        return

    # advance until after prep_steps (prep_steps == 80 by default)
    steps_to_run = env.prep_steps + 5
    print(f"Stepping {steps_to_run} steps (prep_steps={env.prep_steps}) to exit preparation period")
    for i in range(steps_to_run):
        try:
            out = env.step(np.zeros(4))
        except Exception as e:
            print(f'STEP_ERROR at step {i}:', e)
            traceback.print_exc()
            return

    # build pre dicts similar to step()
    pre_obs_by_agent, pre_state_by_agent = build_pre_dicts(env)

    print('pre_obs_by_agent keys:', list(pre_obs_by_agent.keys())[:10])
    print('pre_state_by_agent keys:', list(pre_state_by_agent.keys())[:10])

    try:
        r_state, f_state = env._compute_team_reward_state(pre_state_by_agent=pre_state_by_agent)
        print('_compute_team_reward_state ->', r_state, f_state)
    except Exception as e:
        print('_compute_team_reward_state ERROR', e)
        traceback.print_exc()


if __name__ == '__main__':
    main()
