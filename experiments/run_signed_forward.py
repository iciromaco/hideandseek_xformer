import os
import sys
import json
import math
import numpy as np

# Ensure project root is on PYTHONPATH so `import src...` works when running from
# the experiments/ folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.hns_environment import TeamCosEnv
from experiments.utils import prepare_env

# load toml config safely
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
        return 0
    try:
        with open(cfg_path, 'rb') as f:
            cfg = _tomlib.load(f)
        return int(cfg.get('runtime', {}).get('common', {}).get('action_repeat', 0) or 0)
    except Exception:
        return 0


def main(steps=1000, out_path='experiments/signed_forward.json'):
    env = TeamCosEnv(debug_mode=False, target="seeker")
    ar = _load_action_repeat_from_config()
    if ar:
        env = prepare_env(env, action_repeat=ar, place_far=False)
    else:
        env = prepare_env(env, action_repeat=None, place_far=False)
    # warm-step to init internal state
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    records = []
    for i in range(steps):
        a = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(a)

        # world-frame vx, vy: read directly from body linear velocity (xvelp)
        try:
            learnable = env.learnable_agent_key
            body_id = env.body_ids[learnable]
            v = env.data.xvelp[body_id]
            vx = float(v[0])
            vy = float(v[1])
        except Exception:
            vx = info.get('agent_vx', 0.0)
            vy = info.get('agent_vy', 0.0)

        # try to get yaw (heading) from qpos via the rot joint qpos adr
        yaw = None
        try:
            learnable = env.learnable_agent_key
            rot_jid = env.qpos_indices[learnable]['rot']
            qpos_adr = env.model.jnt_qposadr[rot_jid]
            yaw = float(env.data.qpos[qpos_adr])
        except Exception:
            yaw = None

        if yaw is None:
            # fallback: try info (if env exposes it)
            yaw = info.get('agent_yaw', 0.0)

        # signed forward velocity in agent frame: forward = (cos yaw, sin yaw)
        sfwd = float(vx * math.cos(yaw) + vy * math.sin(yaw))

        records.append({
            'step': i,
            'action': a.tolist(),
            'vx': float(vx),
            'vy': float(vy),
            'yaw': float(yaw),
            'signed_forward': sfwd,
            'applied_forward': float(info.get('applied_forward', 0.0)),
        })

        if term or trunc:
            env.reset()

    # aggregate stats
    sf = np.array([r['signed_forward'] for r in records], dtype=np.float32)
    arr = np.array([r['action'] for r in records], dtype=np.float32)
    stats = {
        'steps': int(len(records)),
        'signed_forward_mean': float(sf.mean()),
        'signed_forward_std': float(sf.std()),
        'action_mean': arr.mean(axis=0).tolist(),
        'action_std': arr.std(axis=0).tolist(),
    }

    out = {'stats': stats, 'records_sample': records[:200]}
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved', out_path)
    print('signed_forward mean/std:', stats['signed_forward_mean'], stats['signed_forward_std'])
    print('action mean:', stats['action_mean'])


if __name__ == '__main__':
    main()
