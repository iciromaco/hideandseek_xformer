import os
import sys
import json
import numpy as np

# Ensure project root is on PYTHONPATH so `import src...` works when running from
# the experiments/ folder.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.hns_environment import TeamCosEnv

def main(steps=1000, out_path='experiments/action_dist.json'):
    env = TeamCosEnv(debug_mode=False)
    obs, reward, term, trunc, info = env.step(env.action_space.sample())  # warm-step to init
    actions = []
    speeds = []
    angs = []
    for i in range(steps):
        a = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(a)
        actions.append(a.tolist())
        # collect vx, vy from info if present, else try to read from env
        vx = info.get('agent_vx', None)
        vy = info.get('agent_vy', None)
        if vx is None or vy is None:
            try:
                # attempt to read directly from env internal state for learnable agent
                lid = env.learnable_agent_key
                jz = env.qpos_indices[lid]['z']
                vadr = env.model.jnt_dofadr[env.model.joint(f"{lid}_x").id]
                vx = float(env.data.qvel[vadr])
                vy = float(env.data.qvel[vadr + 1])
            except Exception:
                vx, vy = 0.0, 0.0
        speeds.append(float((vx * vx + vy * vy) ** 0.5))
        # angular yaw velocity: jnt_dofadr for rot joint
        try:
            rot_jid = env.qpos_indices[env.learnable_agent_key]['rot']
            rot_dof = env.model.jnt_dofadr[rot_jid]
            angs.append(float(env.data.qvel[rot_dof]))
        except Exception:
            angs.append(0.0)
        if term or trunc:
            env.reset()
    arr = np.array(actions, dtype=np.float32)
    arr = np.array(actions, dtype=np.float32)
    sp = np.array(speeds, dtype=np.float32)
    ag = np.array(angs, dtype=np.float32)
    stats = {
        'steps': int(arr.shape[0]),
        'action_mean': arr.mean(axis=0).tolist(),
        'action_std': arr.std(axis=0).tolist(),
        'action_min': arr.min(axis=0).tolist(),
        'action_max': arr.max(axis=0).tolist(),
        'speed_mean': float(sp.mean()),
        'speed_std': float(sp.std()),
        'speed_min': float(sp.min()),
        'speed_max': float(sp.max()),
        'ang_mean': float(ag.mean()),
        'ang_std': float(ag.std()),
        'ang_min': float(ag.min()),
        'ang_max': float(ag.max()),
        'histograms': [],
    }
    bins = 21
    for d in range(arr.shape[1]):
        hist, edges = np.histogram(arr[:, d], bins=bins, range=(-1.0, 1.0))
        stats['histograms'].append({
            'counts': hist.tolist(),
            'edges': edges.tolist(),
        })
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print('Saved', out_path)
    print('Action mean:', stats['action_mean'])
    print('Action std :', stats['action_std'])
    print('Speed mean :', stats['speed_mean'], ' std:', stats['speed_std'])
    print('Ang mean   :', stats['ang_mean'], ' std:', stats['ang_std'])

if __name__ == '__main__':
    main()
