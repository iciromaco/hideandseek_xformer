import os
import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        return 0
    try:
        with open(cfg_path, 'rb') as f:
            cfg = _tomlib.load(f)
        return int(cfg.get('runtime', {}).get('common', {}).get('action_repeat', 0) or 0)
    except Exception:
        return 0


def run_block(env, hold_action, steps=200):
    a = np.array([hold_action, 0.0, 0.0, 0.0], dtype=np.float32)
    records = []
    for i in range(steps):
        obs, reward, term, trunc, info = env.step(a)
        try:
            bid = env.body_ids[env.learnable_agent_key]
            v = env.data.xvelp[bid]
            vx = float(v[0]); vy = float(v[1])
        except Exception:
            vx = float(info.get('agent_vx', 0.0)); vy = float(info.get('agent_vy', 0.0))
        yaw = 0.0
        try:
            rot_jid = env.qpos_indices[env.learnable_agent_key]['rot']
            qadr = env.model.jnt_qposadr[rot_jid]
            yaw = float(env.data.qpos[qadr])
        except Exception:
            yaw = float(info.get('agent_yaw', 0.0) or 0.0)
        signed = float(vx * math.cos(yaw) + vy * math.sin(yaw))
        applied = float(info.get('applied_forward', 0.0))
        records.append({'step': i, 'applied': applied, 'vx': vx, 'vy': vy, 'signed_forward': signed})
        if term or trunc:
            env.reset()
    return records


def main(out_json='experiments/step_response_sweep.json', out_png='experiments/plots/step_response_sweep.png'):
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    holds = [0.25, 0.5, 1.0, -0.25, -0.5, -1.0]
    per_block = {}

    env = TeamCosEnv(debug_mode=False, target="seeker")
    try:
        env.prep_steps = 0
    except Exception:
        pass

    ar = _load_action_repeat_from_config()
    if ar:
        env = prepare_env(env, action_repeat=ar, place_far=True)
    else:
        env = prepare_env(env, action_repeat=None, place_far=True)

    # warm
    env.step(env.action_space.sample())

    for h in holds:
        env.reset()
        # short wait
        for _ in range(10):
            env.step(env.action_space.sample() * 0.0)
        recs = run_block(env, h, steps=200)
        per_block[str(h)] = recs

    # aggregate stats
    stats = {}
    for h, recs in per_block.items():
        arr_signed = np.array([r['signed_forward'] for r in recs], dtype=np.float32)
        steady_last = min(50, len(arr_signed)//2)
        stats[h] = {'mean_last': float(arr_signed[-steady_last:].mean()), 'std_last': float(arr_signed[-steady_last:].std())}

    out = {'stats': stats, 'blocks': per_block}
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

    # plot
    plt.figure(figsize=(10,6))
    for h, recs in per_block.items():
        t = np.arange(len(recs)) + float(h)*0  # keep index
        plt.plot(t, [r['signed_forward'] for r in recs], label=f'h={h}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print('Saved', out_json)


if __name__ == '__main__':
    main()
