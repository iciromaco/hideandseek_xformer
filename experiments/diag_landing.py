import os
import sys
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.envs.hns_environment import TeamCosEnv
from experiments.utils import prepare_env


def main(steps=40, out='experiments/diag_landing.json'):
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    env = TeamCosEnv(debug_mode=False, target="seeker")
    try:
        env.prep_steps = 0
    except Exception:
        pass
    env = prepare_env(env, action_repeat=None, place_far=True)

    # warm
    env.step(env.action_space.sample())

    recs = []
    for i in range(steps):
        a = np.array([0.25, 0.0, 0.0, 0.0], dtype=np.float32)
        obs, rew, term, trunc, info = env.step(a)
        learn = env.learnable_agent_key
        bid = env.body_ids[learn]
        # positions and velocities
        try:
            xpos = env.data.xpos[bid].tolist()
        except Exception:
            xpos = [float(info.get('agent_x', 0.0)), float(info.get('agent_y', 0.0)), float(info.get('agent_z', 0.0))]
        try:
            v = env.data.xvelp[bid].tolist()
        except Exception:
            v = [float(info.get('agent_vx', 0.0)), float(info.get('agent_vy', 0.0)), float(info.get('agent_vz', 0.0))]
        # yaw
        try:
            rot_jid = env.qpos_indices[learn]['rot']
            qadr = env.model.jnt_qposadr[rot_jid]
            yaw = float(env.data.qpos[qadr])
        except Exception:
            yaw = float(info.get('agent_yaw', 0.0) or 0.0)

        # contact count and contact force on body if available
        ncon = int(getattr(env.data, 'ncon', 0) or 0)
        cfrc = None
        try:
            cfrc = env.data.cfrc_body[bid].tolist()
        except Exception:
            cfrc = None

        recs.append({'step': i, 'applied': float(info.get('applied_forward', 0.0)), 'xpos': xpos, 'vel': v, 'yaw': yaw, 'ncon': ncon, 'cfrc_body': cfrc})
        if term or trunc:
            env.reset()

    with open(out, 'w') as f:
        json.dump({'records': recs}, f, indent=2)
    print('Saved', out)


if __name__ == '__main__':
    main()
