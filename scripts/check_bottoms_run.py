import sys
sys.path.insert(0, '.')
from src.envs.hns_environment import TeamCosEnv

env = TeamCosEnv(render_mode=None)
obs, info = env.reset()
print('AGENT_Z_MIN, R_AGENT =', env.AGENT_Z_MIN, env.R_AGENT)
for ak in env.agent_keys:
    gid = env.model.geom(f"{ak}_btm").id
    center_z = float(env.data.geom_xpos[gid][2])
    radius = float(env.model.geom_size[gid][0])
    bottom_z = center_z - radius
    print(f"{ak}: geom_id={gid}, center_z={center_z:.6f}, radius={radius:.6f}, bottom_z={bottom_z:.6f}")
