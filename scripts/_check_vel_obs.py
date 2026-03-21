# Check VEL_X/VEL_Y observation correctness
import sys, os
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.envs.hns_environment import TeamCosEnv
import math

env = TeamCosEnv(mode='initial', target='seeker', n_seekers=1, n_hiders=2, n_boxes=0, n_ramps=0, render_mode=None)
obs, info = env.reset()
print('reset info:', info)

for f in range(8):
    # apply a forward command to the learnable agent to induce motion
    act = [0.6, 0.0, 0.0, 0.0]
    obs, r, _, done, info = env.step(act)
    print(f'FRAME {f} reward={r:.4f}')
    for i, ak in enumerate(env.agent_keys):
        idx = env.idx
        si = idx.SELF
        # get observation for agent i
        o = env._get_obs(i)
        # own velocity (obs) in agent-local frame
        self_vx = float(o[si.VEL_X])
        self_vy = float(o[si.VEL_Y])
        print(f'  Agent {ak} SELF VEL obs: vx={self_vx:.4f} vy={self_vy:.4f}')
        # check others entries filled
        ens = [k for k in env.agent_keys if k != ak]
        if ak.startswith('s'):
            ens.sort(key=lambda k: (0 if k.startswith('h') else 1, k))
        else:
            ens.sort(key=lambda k: (0 if k.startswith('s') else 1, k))
        for j, enm in enumerate(ens[:len(idx.OTHERS)]):
            en_idx = idx.OTHERS[j]
            vis = bool(o[en_idx.VISIBLE])
            vel_obs_x = float(o[en_idx.VEL_X])
            vel_obs_y = float(o[en_idx.VEL_Y])
            # compute ground-truth vel in observer-local frame from qvel
            vadr = env.model.jnt_dofadr[env.qpos_indices[enm]['x']]
            qlen = env.data.qvel.shape[0]
            vx_world = float(env.data.qvel[vadr]) if vadr < qlen else 0.0
            vy_world = float(env.data.qvel[vadr+1]) if (vadr+1) < qlen else 0.0
            # rotate into observer frame (uses same cos_r/sin_r as _get_obs)
            rv = float(env.data.qpos[env.model.jnt_qposadr[env.qpos_indices[ak]['rot']]])
            cos_r = math.cos(-rv); sin_r = math.sin(-rv)
            vx_rel = vx_world * cos_r - vy_world * sin_r
            vy_rel = vx_world * sin_r + vy_world * cos_r
            print(f'    Other {enm} vis={vis} obs_vel=({vel_obs_x:.4f},{vel_obs_y:.4f}) calc_vel=({vx_rel:.4f},{vy_rel:.4f})')
    print('')

env.close()
print('done')
