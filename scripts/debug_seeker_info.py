from src.envs.hns_environment import TeamCosEnv

if __name__ == "__main__":
    env = TeamCosEnv()
    env.reset()
    for i in range(5):
        action = env.action_space.sample()
        obs, r, done, info = env.step(action)
        if i == 0:
            print(
                "step0 info sample:",
                {
                    k: info[k]
                    for k in [
                        "dbg_learnable_hider_seen",
                        "is_detected",
                        "wall_distance",
                        "agent_vz",
                    ]
                    if k in info
                },
            )
        if done:
            break
    env.close()
