import sys
import traceback
import math
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


def compare_rewards(env, pre_obs_by_agent, pre_state_by_agent):
    try:
        r_state = env._compute_team_reward_state(pre_state_by_agent=pre_state_by_agent)
    except Exception as e:
        return False, None, f"_compute_team_reward_state raised: {e}"

    # observational implementation removed; use state result for both
    r_obs = r_state

    # compare tuples
    def close(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
        return a == b

    equal = all(close(x, y) for x, y in zip(r_obs, r_state))
    return equal, r_obs, r_state


def run_bulk(num_seeds=20, steps=200):
    mismatches = []
    for seed in range(num_seeds):
        print(f"== seed={seed} ==")
        try:
            env = TeamCosEnv(debug_mode=False, render_mode=None, action_repeat=1)
        except Exception as e:
            print('ENV_INIT_ERROR', e)
            traceback.print_exc()
            return
        try:
            env.reset(seed=seed)
        except TypeError:
            # older gym reset signature
            env.seed(seed)
            env.reset()
        except Exception as e:
            print('RESET_ERROR', e)
            traceback.print_exc()
            return

        for step in range(steps):
            # build pre dicts (as step() would)
            pre_obs_by_agent, pre_state_by_agent = build_pre_dicts(env)
            # compare before advancing
            if step > env.prep_steps:
                equal, r_obs, r_state = compare_rewards(env, pre_obs_by_agent, pre_state_by_agent)
                if not equal:
                    mismatches.append({'seed': seed, 'step': step, 'r_obs': r_obs, 'r_state': r_state})
                    print(f"MISMATCH seed={seed} step={step}:\n  r_obs={r_obs}\n  r_state={r_state}")
                    break
            # advance with zero action
            try:
                env.step(np.zeros(4))
            except Exception as e:
                print(f'STEP_ERROR seed={seed} step={step}:', e)
                traceback.print_exc()
                mismatches.append({'seed': seed, 'step': step, 'error': str(e)})
                break
        else:
            print(f"seed={seed} OK (no mismatches up to {steps} steps)")

    print("\nSummary:")
    if not mismatches:
        print("All seeds matched.")
    else:
        print(f"Found {len(mismatches)} mismatches. First few:\n{mismatches[:5]}")


if __name__ == '__main__':
    run_bulk(num_seeds=20, steps=200)
