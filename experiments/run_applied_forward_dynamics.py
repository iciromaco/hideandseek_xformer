import os
import sys
import json
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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


def main(steps=1000, out_path='experiments/applied_forward_dynamics.json'):
    env = TeamCosEnv(debug_mode=False, target="seeker")
    ar = _load_action_repeat_from_config()
    if ar:
        env = prepare_env(env, action_repeat=ar, place_far=False)
    else:
        env = prepare_env(env, action_repeat=None, place_far=False)
    # warm-step
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    records = []
    for i in range(steps):
        # read vx before applying action directly from env internals
        try:
            learnable = env.learnable_agent_key
            body_id = env.body_ids[learnable]
            vx_before = float(env.data.xvelp[body_id][0])
        except Exception:
            vx_before = float(info.get('agent_vx', 0.0))

        a = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(a)

        # read vx after step
        try:
            learnable = env.learnable_agent_key
            body_id = env.body_ids[learnable]
            vx_after = float(env.data.xvelp[body_id][0])
        except Exception:
            vx_after = float(info.get('agent_vx', 0.0))

        applied = float(info.get('applied_forward', 0.0))
        delta_vx = vx_after - vx_before

        records.append({
            'step': i,
            'action0': float(a[0]),
            'applied_forward': applied,
            'vx_before': float(vx_before),
            'vx_after': float(vx_after),
            'delta_vx': float(delta_vx),
        })

        if term or trunc:
            env.reset()

    arr = np.array([r['applied_forward'] for r in records], dtype=np.float32)
    dv = np.array([r['delta_vx'] for r in records], dtype=np.float32)
    a0 = np.array([r['action0'] for r in records], dtype=np.float32)

    # linear regression applied_forward -> delta_vx
    coef = np.polyfit(arr, dv, 1)
    pred = np.polyval(coef, arr)
    corr = float(np.corrcoef(arr, dv)[0,1]) if arr.size>1 else 0.0

    stats = {
        'steps': int(len(records)),
        'applied_to_delta_coef': coef.tolist(),
        'applied_delta_corr': corr,
        'mean_applied': float(arr.mean()),
        'mean_delta_vx': float(dv.mean()),
        'mean_action0': float(a0.mean()),
    }
    out = {'stats': stats, 'records_sample': records[:200]}
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Saved', out_path)
    print('applied->delta coef:', stats['applied_to_delta_coef'], 'corr:', stats['applied_delta_corr'])


if __name__ == '__main__':
    main()
