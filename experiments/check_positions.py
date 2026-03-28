import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, "src"))

from envs.hns_environment import TeamCosEnv
from experiments.utils import prepare_env

env = TeamCosEnv(debug_mode=False, target="seeker")
# before reset (initial model qpos from constructor)
try:
    print("INITIAL box xpos:", [env.data.xpos[b][:3].tolist() for b in env.box_ids])
    print("INITIAL ramp xpos:", [env.data.xpos[r][:3].tolist() for r in env.ramp_ids])
    print(
        "INITIAL agents:",
        [(k, env.data.xpos[env.body_ids[k]][:3].tolist()) for k in env.agent_keys],
    )
    print("MODEL body names:", [env.model.body(i).name for i in range(env.model.nbody)])
except Exception as e:
    print("initial read error", e)

env.reset()
print("\nAFTER reset:")
print("box xpos:", [env.data.xpos[b][:3].tolist() for b in env.box_ids])
print("ramp xpos:", [env.data.xpos[r][:3].tolist() for r in env.ramp_ids])
print(
    "agents:",
    [(k, env.data.xpos[env.body_ids[k]][:3].tolist()) for k in env.agent_keys],
)
print("qpos_indices keys:", list(getattr(env, "qpos_indices", {}).keys()))
print(
    "qpos_indices sample:",
    {k: getattr(env, "qpos_indices", {}).get(k) for k in env.agent_keys},
)

# apply prepare_env to move objects far
env = prepare_env(env, place_far=True)
print("\nAFTER prepare_env:")
print("box xpos:", [env.data.xpos[b][:3].tolist() for b in env.box_ids])
print("ramp xpos:", [env.data.xpos[r][:3].tolist() for r in env.ramp_ids])
print(
    "agents:",
    [(k, env.data.xpos[env.body_ids[k]][:3].tolist()) for k in env.agent_keys],
)

# Also print qpos for first box and ramp
if env.box_ids:
    bid = env.box_ids[0]
    adr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
    print("\nbox[0] qpos slice:", env.data.qpos[adr : adr + 7].tolist())
if env.ramp_ids:
    rid = env.ramp_ids[0]
    adr = env.model.jnt_qposadr[env.model.body_jntadr[rid]]
    print("ramp[0] qpos slice:", env.data.qpos[adr : adr + 7].tolist())
print("\nagent qpos joint values:")
for ak in env.agent_keys:
    try:
        qmap = env.qpos_indices[ak]
        jx = qmap["x"]
        adr = env.model.jnt_qposadr[jx]
        print(ak, env.data.qpos[adr : adr + 4].tolist())
    except Exception as e:
        print(ak, "qpos read error", e)
