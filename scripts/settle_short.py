import sys
sys.path.insert(0, '.')
from src.envs.hns_environment import TeamCosEnv
import mujoco
import numpy as np

env = TeamCosEnv(render_mode=None)
env.reset()
print('AGENT_Z_MIN, R_AGENT =', env.AGENT_Z_MIN, env.R_AGENT)

lift = 0.08
steps = 60
report_at = {0,1,2,5,10,20,steps}

# collect joint addresses
jaddrs = {}
qaddrs = {}
vaddrs = {}
for ak in env.agent_keys:
    j = env.model.joint(f"{ak}_z").id
    jaddrs[ak] = j
    qaddrs[ak] = env.model.jnt_qposadr[j]
    vaddrs[ak] = env.model.jnt_dofadr[j]

print('before lift:')
for ak in env.agent_keys:
    qz = float(env.data.qpos[qaddrs[ak]])
    body_z = float(env.data.xpos[env.body_ids[ak]][2])
    print(f"{ak}: qpos_z={qz:.6f}, body_z={body_z:.6f}")

# apply lift
for ak in env.agent_keys:
    env.data.qpos[qaddrs[ak]] += lift
    env.data.qvel[vaddrs[ak]] = 0.0

# forward so derived quantities update
mujoco.mj_forward(env.model, env.data)

print('\nafter lift, before stepping:')
for ak in env.agent_keys:
    qz = float(env.data.qpos[qaddrs[ak]])
    body_z = float(env.data.xpos[env.body_ids[ak]][2])
    print(f"{ak}: qpos_z={qz:.6f}, body_z={body_z:.6f}")

print('\nstepping...')
for t in range(1, steps+1):
    mujoco.mj_step(env.model, env.data)
    if t in report_at:
        print(f"step={t}")
        for ak in env.agent_keys:
            qz = float(env.data.qpos[qaddrs[ak]])
            body_z = float(env.data.xpos[env.body_ids[ak]][2])
            vz = float(env.data.qvel[vaddrs[ak]])
            print(f"  {ak}: qpos_z={qz:.6f}, body_z={body_z:.6f}, vz={vz:.6f}")

print('\nfinal (after stepping):')
for ak in env.agent_keys:
    qz = float(env.data.qpos[qaddrs[ak]])
    body_z = float(env.data.xpos[env.body_ids[ak]][2])
    print(f"{ak}: qpos_z={qz:.6f}, body_z={body_z:.6f}")
