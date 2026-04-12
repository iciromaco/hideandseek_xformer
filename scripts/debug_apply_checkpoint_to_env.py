#!/usr/bin/env python3
"""Load checkpoint into env via set_inference_policy_state, warm up, then
compare model output vs env.data.ctrl and resulting vel/pos/reward.

Usage:
  uv run mjpython scripts/debug_apply_checkpoint_to_env.py [--no-objects] [--ckpt PATH]
"""
import sys, os, torch, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import numpy as np
from src.envs.hns_environment import TeamCosEnv

def main():
    no_objects = '--no-objects' in sys.argv
    ckpt_arg = None
    for i, a in enumerate(sys.argv):
        if a == '--ckpt' and i + 1 < len(sys.argv):
            ckpt_arg = sys.argv[i+1]

    n_boxes = 0 if no_objects else 2
    n_ramps = 0 if no_objects else 1

    env = TeamCosEnv(mode='initial', target='seeker', n_seekers=1, n_hiders=2, n_boxes=n_boxes, n_ramps=n_ramps, render_mode=None)
    ak = env.seeker_keys[0]
    bid = env.body_ids[ak]
    ctrl_len = env.action_space.shape[0]
    print('env created. actuator_ids:', env.actuator_ids)

    # checkpoint path default same naming as other scripts
    if ckpt_arg is None:
        path = os.path.join('checkpoints', f"HNS_V27_seeker_s{env.n_seekers}_h{env.n_hiders}_b{env.n_boxes}_r{env.n_ramps}.pt")
    else:
        path = ckpt_arg
    print('requested checkpoint path:', path)
    if not os.path.exists(path):
        # fallback: search for any matching seeker checkpoint in checkpoints/
        ck_dir = os.path.join(os.getcwd(), 'checkpoints')
        found = []
        if os.path.isdir(ck_dir):
            for fn in os.listdir(ck_dir):
                if fn.startswith('HNS_V27_seeker_') and fn.endswith('.pt'):
                    found.append(os.path.join(ck_dir, fn))
        if found:
            # pick newest
            found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            path = found[0]
            print('requested checkpoint not found; falling back to', path)
        else:
            print('checkpoint not found and no fallback available; aborting')
            env.close();
            return

    state_dict = torch.load(path, map_location='cpu')
    # If checkpoint was trained with different env (obs dim), try to recreate env
    # by parsing b{n}_r{m} from filename pattern.
    import re
    m = re.search(r"_b(\d+)_r(\d+)\.pt$", path)
    if m:
        ck_b = int(m.group(1))
        ck_r = int(m.group(2))
        if ck_b != env.n_boxes or ck_r != env.n_ramps:
            print(f"checkpoint expects b={ck_b}, r={ck_r} but env has b={env.n_boxes}, r={env.n_ramps}; recreating env to match checkpoint")
            env.close()
            # recreate env with matching object counts
            env = TeamCosEnv(mode='initial', target='seeker', n_seekers=1, n_hiders=2, n_boxes=ck_b, n_ramps=ck_r, render_mode=None)
            ak = env.seeker_keys[0]
            bid = env.body_ids[ak]
            ctrl_len = env.action_space.shape[0]
            print('recreated env. actuator_ids:', env.actuator_ids)
    # set into env
    # apply into env via policy_adapter when available
    if hasattr(env, 'policy_adapter'):
        ok = env.policy_adapter.set_inference_policy_state([ak], state_dict, seq_len=16, hidden_dim=256)
    else:
        ok = env.set_inference_policy_state([ak], state_dict, seq_len=16, hidden_dim=256)
    print('set_inference_policy_state ok=', ok)

    # prime policy history for learnable agent
    obs, _ = env.reset()
    norm_obs = env._normalize_obs(env._get_obs(env.learnable_agent_index))
    seq_len = int(env._inference_seq_lens.get(ak, 16))
    # prime via policy_adapter when present
    if hasattr(env, 'policy_adapter'):
        env.policy_adapter._prime_policy_history(ak, seq_len, norm_obs)
    else:
        env._prime_policy_history(ak, seq_len, norm_obs)

    # warmup
    warmup = getattr(env, 'prep_steps', 80)
    print('warming up for', warmup)
    zero_a = np.zeros(ctrl_len, dtype=np.float32)
    for _ in range(warmup):
        env.step(zero_a)

    # get model directly
    # try to obtain model via adapter or env
    model = None
    if hasattr(env, 'policy_adapter'):
        # adapter stores models into env._inference_models for compatibility
        model = env._inference_models.get(ak)
    else:
        model = env._inference_models.get(ak)
    if model is None:
        print('no inference model found in env._inference_models')
    else:
        if hasattr(env, 'policy_adapter'):
            hist = env.policy_adapter._get_policy_history_seq(ak, seq_len, env._normalize_obs(env._get_obs(env.learnable_agent_index)))
        else:
            hist = env._get_policy_history_seq(ak, seq_len, env._normalize_obs(env._get_obs(env.learnable_agent_index)))
        inp = torch.as_tensor(hist[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            out = model.get_action_and_value(inp)[0].cpu().numpy().reshape(-1)
        print('model output (first 8):', out[:8])

    # run multiple frames comparing model->env behavior
    n_steps = 32
    print(f'Running {n_steps} model-driven environment steps (override learnable policy) and dumping observations.')
    obs_hist_list = []
    model_out_list = []
    ctrl_list = []
    vel_list = []
    pos_list = []
    reward_list = []
    info_list = []
    # reward components
    team_rb_list = []
    team_seen_bool_list = []
    team_learnable_hider_seen_list = []
    team_min_seeker_dist_list = []

    for i in range(n_steps):
        norm_obs = env._normalize_obs(env._get_obs(env.learnable_agent_index))
        hist = env._get_policy_history_seq(ak, seq_len, norm_obs)
        inp = torch.as_tensor(hist[None, :, :], dtype=torch.float32)
        with torch.no_grad():
            out = model.get_action_and_value(inp)[0].cpu().numpy().reshape(-1)

        # step environment (use model via override)
        if hasattr(env, 'policy_adapter'):
            env.policy_adapter.set_override_learnable_policy(True)
        else:
            env.set_override_learnable_policy(True)
        obs, reward, _, done, info = env.step(zero_a)
        ctrl = env.data.ctrl.copy()
        vel = env._body_speed_xy(bid)
        pos = env.data.xpos[bid].copy()

        obs_hist_list.append(hist.copy())
        model_out_list.append(out.copy())
        ctrl_list.append(ctrl.copy())
        vel_list.append(float(vel))
        pos_list.append(pos.copy())
        reward_list.append(float(reward))
        info_list.append(info)

        # compute/collect reward components by calling env helper and by local metrics
        try:
            if hasattr(env, '_compute_team_reward_state'):
                rb, seen_bool = env._compute_team_reward_state()
            else:
                rb, seen_bool = env._compute_team_reward()
            learnable_seen = False
        except Exception:
            rb, seen_bool = float('nan'), False
            learnable_seen = False

        # compute min_seeker_dist and seen_count locally to capture numeric count
        min_seeker_dist = 9999.0
        seen_count = 0
        for hk in env.hider_keys:
            hid = env.body_ids[hk]
            hpos = env.data.xpos[hid][:2]
            this_seen = False
            for sk in env.seeker_keys:
                sid = env.body_ids[sk]
                spos = env.data.xpos[sid][:2]
                dx = float(hpos[0] - spos[0]); dy = float(hpos[1] - spos[1])
                dist = (dx*dx + dy*dy) ** 0.5
                min_seeker_dist = min(min_seeker_dist, dist)
                # use env._is_vis to match environment visibility logic
                try:
                    if env._is_vis(spos, env.data.qpos[env.model.jnt_qposadr[env.qpos_indices[sk]['rot']]], hpos, sid, hid):
                        this_seen = True
                        break
                except Exception:
                    # fallback: assume not seen
                    pass
            if this_seen:
                seen_count += 1

        team_rb_list.append(float(rb))
        team_seen_bool_list.append(bool(seen_bool))
        team_learnable_hider_seen_list.append(False)
        team_min_seeker_dist_list.append(float(min_seeker_dist if min_seeker_dist < 9998.0 else float('nan')))

        print(f'FRAME {i}: model_out[:4]={out[:4]} -> env.ctrl[first4]={ctrl[:4]} vel={vel:.3f} reward={reward:.3f}')

    # save dump
    dump_path = 'debug_model_env_dump_ext.npz'
    np.savez_compressed(
        dump_path,
        obs_hist=np.asarray(obs_hist_list),
        model_out=np.asarray(model_out_list),
        ctrl=np.asarray(ctrl_list),
        vel=np.asarray(vel_list),
        pos=np.asarray(pos_list),
        reward=np.asarray(reward_list),
        info=np.asarray(info_list),
        team_rb=np.asarray(team_rb_list),
        team_seen_bool=np.asarray(team_seen_bool_list),
        
        team_learnable_hider_seen=np.asarray(team_learnable_hider_seen_list),
        team_min_seeker_dist=np.asarray(team_min_seeker_dist_list),
    )
    print('wrote extended dump to', dump_path)
    if hasattr(env, 'policy_adapter'):
        env.policy_adapter.set_override_learnable_policy(False)
    else:
        env.set_override_learnable_policy(False)

    env.close()

if __name__ == '__main__':
    main()
