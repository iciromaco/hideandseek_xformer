from envs.hns_environment import TeamCosEnv

e = TeamCosEnv(mode="initial", target="seeker", n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1, render_mode=None)
print("TeamCosEnv constructed OK")
e.close()
