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


def main(steps=500, hold_action=1.0, out_json='experiments/step_response.json', out_png='experiments/plots/step_response.png',
         stop_thresh=0.02, stop_frames=5, max_wait_steps=500):
    os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    env = TeamCosEnv(debug_mode=False)
    # avoid prep behavior
    try:
        env.prep_steps = 0
    except Exception:
        pass

    # warm
    env.step(env.action_space.sample())

    # wait until agent body linear speed is below threshold for stop_frames consecutive steps
    learnable = env.learnable_agent_key
    body_id = env.body_ids[learnable]
    stable = 0
    waited = 0
    while stable < stop_frames and waited < max_wait_steps:
        # apply zero action while waiting
        obs, reward, term, trunc, info = env.step(env.action_space.sample() * 0.0)
        try:
            v = env.data.xvelp[body_id]
        except AttributeError:
            try:
                cv = env.data.cvel[body_id]
                v = np.array([float(cv[3]), float(cv[4]), float(cv[5])])
            except Exception:
                v = np.array([float(info.get('agent_vx', 0.0)), float(info.get('agent_vy', 0.0)), 0.0])
        speed = float((v[0]**2 + v[1]**2)**0.5)
        if speed <= stop_thresh:
            stable += 1
        else:
            stable = 0
        waited += 1

    if waited >= max_wait_steps:
        print(f'Warning: waited {max_wait_steps} steps but did not reach stable stop threshold')

    records = []
    a = np.array([hold_action, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(steps):
        obs, reward, term, trunc, info = env.step(a)
        # body velocities
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

    # plot time series
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
