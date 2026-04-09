from src.envs.hns_environment import TeamCosEnv


def main():
    env = TeamCosEnv(debug_mode=True)
    env.dump_actuator_map()
    env.set_debug_agent('h1')
    env.set_override_learnable_policy(False)  # 学習対象を外部action経路に戻す
    obs, info = env.reset()
    for i in range(30):
        # 学習対象に非ゼロアクションを渡す（fwd, turn, lock, grab）
        obs, r, done, dd, info = env.step([0.6, 0.2, 0.0, 0.0])
        # removed debug STEP_OUT print
        if done:
            break
    print('TEST_EXTERNAL_ACTION_DONE')


if __name__ == '__main__':
    main()
