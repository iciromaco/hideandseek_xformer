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

# load toml safely
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


def main(steps=400, hold_action=0.25, out_json='experiments/step_response_nowalls.json', out_png='experiments/plots/step_response_nowalls.png'):
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    env = TeamCosEnv(debug_mode=False, target="seeker")

    ar = _load_action_repeat_from_config()
    if ar:
        env = prepare_env(env, action_repeat=ar, place_far=True)
    else:
        env = prepare_env(env, action_repeat=None, place_far=True)

    # warm
    env.step(env.action_space.sample())

    # wait until prep_steps elapse (Viewer uses prep_steps=80)
    wait_steps = getattr(env, 'prep_steps', 80)
    for _ in range(wait_steps):
        env.step(env.action_space.sample() * 0.0)

    records = []
    a = np.array([hold_action, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(steps):
        obs, reward, term, trunc, info = env.step(a)
        # Prefer body linear velocity (xvelp) to match recorder logic; fall back to joint qvel then info
        vx, vy = 0.0, 0.0
        try:
            name = env.learnable_agent_key
            bid = env.body_ids.get(name)
            if bid is not None:
                vv = env.data.xvelp[bid]
                vx = float(vv[0]); vy = float(vv[1])
        except Exception:
            try:
                qmap = getattr(env, 'qpos_indices', {})
                name = env.learnable_agent_key
                if name in qmap:
                    jx = qmap[name].get('x')
                    if jx is not None:
                        dof_adr = int(env.model.jnt_dofadr[jx])
                        vx = float(env.data.qvel[dof_adr]); vy = float(env.data.qvel[dof_adr+1])
            except Exception:
                vx = float(info.get('agent_vx', 0.0)); vy = float(info.get('agent_vy', 0.0))

        # compute yaw (qpos) when available, otherwise use info
        yaw = 0.0
        try:
            rot_jid = env.qpos_indices[env.learnable_agent_key]['rot']
            qadr = env.model.jnt_qposadr[rot_jid]
            yaw = float(env.data.qpos[qadr])
        except Exception:
            yaw = float(info.get('agent_yaw', 0.0) or 0.0)

        signed = float(vx * math.cos(yaw) + vy * math.sin(yaw))

        # prefer the actual ctrl value written to the actuator (signed); fallback to info
        applied = None
        try:
            aid = env.actuator_ids.get(f"{env.learnable_agent_key}_fwd")
            if aid is not None:
                applied = float(env.data.ctrl[aid])
        except Exception:
            applied = None
        if applied is None:
            applied = float(info.get('applied_forward', 0.0))

        records.append({'step': i, 'applied': applied, 'vx': vx, 'vy': vy, 'signed_forward': signed})
        if term or trunc:
            env.reset()

    arr_signed = np.array([r['signed_forward'] for r in records], dtype=np.float32)
    arr_applied = np.array([r['applied'] for r in records], dtype=np.float32)

    steady_last = min(100, steps//2)
    steady_mean = float(arr_signed[-steady_last:].mean())
    steady_std = float(arr_signed[-steady_last:].std())

    stats = {
        'steps': steps,
        'hold_action': float(hold_action),
        'steady_mean_signed_forward': steady_mean,
        'steady_std_signed_forward': steady_std,
        'applied_mean': float(arr_applied.mean()),
    }

    out = {'stats': stats, 'records': records}
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2)

    t = np.arange(len(records))
    plt.figure(figsize=(8,4))
    plt.plot(t, [r['signed_forward'] for r in records], label='signed_forward')
    plt.plot(t, [r['applied'] for r in records], label='applied_forward', alpha=0.7)
    plt.xlabel('step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print('Saved', out_json)
    print('Steady mean signed_forward (last {}): {:.4f} std {:.4f}'.format(steady_last, steady_mean, steady_std))


if __name__ == '__main__':
    main()
