import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.envs.hns_environment import TeamCosEnv
from experiments.utils import prepare_env
try:
    import tomllib as _tomlib
except Exception:
    try:
        import tomli as _tomlib
    except Exception:
        _tomlib = None

def _load_action_repeat_from_config():
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'hparams_main27.toml')
    if _tomlib is None:
        return None
    try:
        with open(cfg_path, 'rb') as f:
            cfg = _tomlib.load(f)
        return int(cfg.get('runtime', {}).get('common', {}).get('action_repeat', 0) or 0)
    except Exception:
        return None

env = TeamCosEnv(debug_mode=False, target='seeker')
ar = _load_action_repeat_from_config()
if ar:
    env = prepare_env(env, action_repeat=ar, place_far=True)
else:
    env = prepare_env(env, action_repeat=None, place_far=True)

# warm
env.step(env.action_space.sample())
wait_steps = getattr(env, 'prep_steps', 80)
for _ in range(wait_steps):
    env.step(env.action_space.sample() * 0.0)

# run up to 300 steps
a = np.array([0.25, 0.0, 0.0, 0.0], dtype=np.float32)
for i in range(320):
    obs, rew, term, trunc, info = env.step(a)
    if i == 288:
        ak = env.learnable_agent_key
        bid = env.body_ids[ak]
        pos = env.data.xpos[bid].tolist()
        try:
            vel = env.data.xvelp[bid].tolist()
        except Exception:
            # fall back to joint qvel mapping
            try:
                dofadr = env.model.jnt_dofadr[env.model.joint(f"{ak}_x").id]
                vx = float(env.data.qvel[dofadr])
                vy = float(env.data.qvel[dofadr+1])
                vel = [vx, vy, 0.0]
            except Exception:
                vel = [None, None, None]
        qj = env.qpos_indices[ak]['rot']
        qadr = env.model.jnt_qposadr[qj]
        yaw = float(env.data.qpos[qadr])
        ctrl = env.data.ctrl[env.actuator_ids[f"{ak}_fwd"]]
        print('step', i)
        print('agent pos', pos)
        print('agent vel (xvelp)', vel)
        print('qvel slice', env.data.qvel[env.model.jnt_dofadr[env.model.joint(f"{ak}_x").id]:env.model.jnt_dofadr[env.model.joint(f"{ak}_x").id]+6].tolist())
        print('yaw', yaw)
        print('ctrl', ctrl)
        print('actuator_force', env.data.actuator_force.tolist())
        print('xfrc_applied', env.data.xfrc_applied[bid].tolist())
        print('cfrc_body', env.data.cfrc_body[bid].tolist())
        break
