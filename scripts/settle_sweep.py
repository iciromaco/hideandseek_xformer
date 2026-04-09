import sys
sys.path.insert(0, '.')
from src.envs.hns_environment import TeamCosEnv
import mujoco

vals = [0.48, 0.49, 0.50, 0.51, 0.52]
steps = 60
lift = 0.08

for v in vals:
    TeamCosEnv.AGENT_Z_MIN = v
    env = TeamCosEnv(render_mode=None)
    env.reset()
    # prepare addresses
    qaddrs = {}
    vaddrs = {}
    for ak in env.agent_keys:
        j = env.model.joint(f"{ak}_z").id
        qaddrs[ak] = env.model.jnt_qposadr[j]
        vaddrs[ak] = env.model.jnt_dofadr[j]
    # lift
    for ak in env.agent_keys:
        env.data.qpos[qaddrs[ak]] += lift
        env.data.qvel[vaddrs[ak]] = 0.0
    mujoco.mj_forward(env.model, env.data)
    # step
    for t in range(steps):
        mujoco.mj_step(env.model, env.data)
    print(f"AGENT_Z_MIN={v:.3f}")
    for ak in env.agent_keys:
        qz = float(env.data.qpos[qaddrs[ak]])
        body_z = float(env.data.xpos[env.body_ids[ak]][2])
        print(f"  {ak}: qpos_z={qz:.6f}, body_z={body_z:.6f}")
    print("")
