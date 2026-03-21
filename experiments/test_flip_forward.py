import os
import sys
import json
import torch
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from envs.hns_environment import TeamCosEnv

OUT = 'experiments/model_debug_seeker_flip.jsonl'
os.makedirs('experiments', exist_ok=True)

# load env and checkpoint similar to debug_learned_seeker
def find_and_load_checkpoint():
    import glob, re
    candidates = sorted(glob.glob(os.path.join('checkpoints', 'HNS_V27_seeker*.pt')))
    if not candidates:
        return None, None
    path = candidates[-1]
    state = torch.load(path, map_location='cpu')
    return path, state

path, state = find_and_load_checkpoint()
if path is None:
    print('No seeker checkpoint found in checkpoints/, aborting')
    raise SystemExit(1)
print('Using checkpoint:', path)

# infer cfg from filename
inferred_cfg = None
try:
    import re
    m = re.search(r'HNS_V27_seeker_s(\d+)_h(\d+)_b(\d+)_r(\d+)\.pt', os.path.basename(path))
    if m:
        inferred_cfg = {
            'n_seekers': int(m.group(1)),
            'n_hiders': int(m.group(2)),
            'n_boxes': int(m.group(3)),
            'n_ramps': int(m.group(4)),
        }
        print('Inferred env config:', inferred_cfg)
except Exception:
    inferred_cfg = None

if inferred_cfg is not None:
    env = TeamCosEnv(mode='initial', target='seeker', n_seekers=inferred_cfg.get('n_seekers',1), n_hiders=inferred_cfg.get('n_hiders',1), n_boxes=inferred_cfg.get('n_boxes',0), n_ramps=inferred_cfg.get('n_ramps',0), debug_mode=True)
else:
    env = TeamCosEnv(mode='initial', target='seeker', n_seekers=1, n_hiders=1, n_boxes=0, n_ramps=0, debug_mode=True)

# Prepare model inside env inference (but we will not use env.override; we'll call model ourselves)
model_state = state
if isinstance(state, dict) and 'model_state_dict' in state:
    model_state = state['model_state_dict']

# infer sizes
hidden_dim = 128
seq_len = 8
try:
    for k in model_state.keys():
        if k.endswith('obs_encoder.0.weight'):
            w = model_state[k]
            hidden_dim = int(w.shape[0])
            break
except Exception:
    pass
try:
    if isinstance(state, dict) and 'seq_len' in state:
        seq_len = int(state['seq_len'])
    elif isinstance(state, dict) and 'args' in state and isinstance(state['args'], dict) and 'seq_len' in state['args']:
        seq_len = int(state['args']['seq_len'])
except Exception:
    pass

try:
    env.set_inference_policy_state([env.learnable_agent_key], model_state, seq_len=seq_len, hidden_dim=hidden_dim)
    # ensure env will accept external action for learnable agent
    env.set_override_learnable_policy(False)
    env.set_model_policy_deterministic(True)
    print('Loaded model into env (for inference calls).')
except Exception as e:
    print('Failed to set inference model in env:', e)
    raise

# Run short episode where we query the model, flip forward sign, and pass as external action
with open(OUT, 'w') as fo:
    obs, info = env.reset()
    done = False
    step = 0
    while not done and step < 500:
        # get normalized obs and model output
        norm = env._normalize_obs(env._get_obs(env.learnable_agent_index))
        seq = env._get_policy_history_seq(env.learnable_agent_key, env._inference_seq_lens.get(env.learnable_agent_key, seq_len), norm)
        import torch
        seq_t = torch.as_tensor(seq[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            m = env._inference_models[env.learnable_agent_key]
            if env.model_policy_deterministic and hasattr(m, 'get_deterministic_action_and_value'):
                arr = m.get_deterministic_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
            else:
                arr = m.get_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
        # flip forward sign
        f, t, lck, grb = float(arr[0]), float(arr[1]) if arr.shape[0]>1 else 0.0, float(arr[2]) if arr.shape[0]>2 else 0.0, float(arr[3]) if arr.shape[0]>3 else 0.0
        f = -f
        a = np.array([f, t, lck, grb], dtype=np.float32)
        next_obs, base_r, term, trunc, info = env.step(a)

        raw_obs = env._get_obs(env.learnable_agent_index)
        rec = {
            'step': step,
            'info': {k: info.get(k) for k in ['dbg_learnable_hider_seen','is_detected','wall_distance','agent_vx','agent_vy','dbg_last_ctrl_f','dbg_last_ctrl_t'] if k in info},
            'raw_obs_rel': {
                'rel_x': float(raw_obs[env.idx.OTHERS[0].REL_X]) if hasattr(env.idx.OTHERS[0],'REL_X') else None,
                'rel_y': float(raw_obs[env.idx.OTHERS[0].REL_Y]) if hasattr(env.idx.OTHERS[0],'REL_Y') else None,
                'visible': float(raw_obs[env.idx.OTHERS[0].VISIBLE]) if hasattr(env.idx.OTHERS[0],'VISIBLE') else None,
            },
            'norm_obs_sample': norm.tolist(),
            'model_action': arr.tolist(),
            'applied_ctrl': info.get('applied_forward', None),
            'base_reward': float(base_r),
        }
        # record only when visible (same criterion as debug script)
        vis_flag = rec['raw_obs_rel']['visible'] if rec['raw_obs_rel']['visible'] is not None else info.get('dbg_learnable_hider_seen', False)
        if vis_flag:
            fo.write(json.dumps(rec) + '\n')

        step += 1
        done = term or trunc

print('Wrote flipped logs to', OUT)
