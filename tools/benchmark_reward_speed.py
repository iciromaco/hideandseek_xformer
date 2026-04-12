import sys
import time
import traceback

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


def benchmark(n_iters=1000):
    env = TeamCosEnv(debug_mode=False, render_mode=None, action_repeat=1)
    # reset and step past preparation period
    env.reset()
    for _ in range(env.prep_steps + 1):
        env.step([0.0, 0.0, 0.0, 0.0])

    pre_obs_by_agent, pre_state_by_agent = build_pre_dicts(env)

    # warmup state-based
    for _ in range(10):
        env._compute_team_reward_state()

    # benchmark state-based only (observational implementation removed)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        env._compute_team_reward_state()
    t1 = time.perf_counter()
    state_time = t1 - t0

    print(f"iters={n_iters}")
    print(f"state_total={state_time:.6f}s state_per_call={state_time/n_iters*1e6:.2f}us")


if __name__ == '__main__':
    try:
        benchmark(n_iters=1000)
    except Exception as e:
        print('BENCH_ERROR', e)
        traceback.print_exc()
