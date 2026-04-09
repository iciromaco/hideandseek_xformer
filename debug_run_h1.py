from src.envs.hns_environment import TeamCosEnv


def main():
    env = TeamCosEnv(debug_mode=True)
    env.dump_actuator_map()
    env.set_debug_agent('h1')
    # 学習対象もルールベースで動かす（モデル/外部actionではなく adapter 経由）
    env.set_override_learnable_policy(True)
    obs, info = env.reset()
    for i in range(30):
        obs, r, done, dd, info = env.step([0.0, 0.0, 0.0, 0.0])
        # removed debug STEP_OUT print
        if done:
            break
    print('DEBUG_RUN_DONE')


if __name__ == '__main__':
    main()
