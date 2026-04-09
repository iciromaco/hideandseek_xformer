#!/usr/bin/env python3
import argparse
import os
import json
import torch
import numpy as np
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.hns28_environment import TeamCosEnv


def find_checkpoint(target, n_seekers, n_hiders, n_boxes, n_ramps):
    ck_name = f"HNS_V27_{target}_s{n_seekers}_h{n_hiders}_b{n_boxes}_r{n_ramps}.pt"
    ck_path = os.path.join(os.getcwd(), "checkpoints", ck_name)
    if os.path.exists(ck_path):
        return ck_path
    # fallback to newest matching
    ck_dir = os.path.join(os.getcwd(), "checkpoints")
    found = []
    if os.path.isdir(ck_dir):
        for fn in os.listdir(ck_dir):
            if fn.startswith(f"HNS_V27_{target}_") and fn.endswith('.pt'):
                found.append(os.path.join(ck_dir, fn))
    if found:
        found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return found[0]
    return None


def evaluate(args):
    val = float(args.dist_bonus_scale)
    target = args.target
    ck = find_checkpoint(target, args.n_seekers, args.n_hiders, args.n_boxes, args.n_ramps)
    if ck is None:
        print(f"No checkpoint found for target={target}")
        return 1

    print(f"Using checkpoint: {ck}")

    env = TeamCosEnv(mode='initial', target=target, n_seekers=args.n_seekers, n_hiders=args.n_hiders, n_boxes=args.n_boxes, n_ramps=args.n_ramps, render_mode=None, dist_bonus_scale=val)
    try:
        state = torch.load(ck, map_location='cpu')
    except Exception as e:
        print('Failed to load checkpoint:', e)
        env.close()
        return 1

    ak = env.learnable_agent_key
    try:
        if hasattr(env, 'policy_adapter'):
            env.policy_adapter.set_inference_policy_state([ak], state, seq_len=16, hidden_dim=256)
        else:
            env.set_inference_policy_state([ak], state, seq_len=16, hidden_dim=256)
    except Exception:
        pass

    try:
        norm_obs = env._normalize_obs(env._get_obs(env.learnable_agent_index))
        if hasattr(env, 'policy_adapter'):
            env.policy_adapter._prime_policy_history(ak, 16, norm_obs)
        else:
            env._prime_policy_history(ak, 16, norm_obs)
    except Exception:
        pass

    try:
        if hasattr(env, 'policy_adapter'):
            env.policy_adapter.set_override_learnable_policy(True)
        else:
            env.set_override_learnable_policy(True)
    except Exception:
        pass

    seen_count = 0
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_seen = False
        # warmup
        for _ in range(getattr(env, 'prep_steps', 80)):
            env.step(np.zeros(4, dtype=np.float32))
        while not done:
            _, _, term, trunc, info = env.step(np.zeros(4, dtype=np.float32))
            done = bool(term or trunc)
            try:
                if ak.startswith('s'):
                    obs_learn = env._get_obs(env.agent_keys.index(ak))
                    ens = env._ens_orderings.get(ak, [k for k in env.agent_keys if k != ak])
                    for i, e in enumerate(ens[: len(env.idx.OTHERS)]):
                        if not e.startswith('h'):
                            continue
                        en_idx = env.idx.OTHERS[i]
                        if float(obs_learn[en_idx.VISIBLE]) > 0.5:
                            ep_seen = True
                            break
            except Exception:
                pass

        if ak.startswith('s'):
            if ep_seen:
                seen_count += 1
        else:
            if info.get('dbg_learnable_hider_seen', False):
                seen_count += 1

    rate = float(seen_count) / float(max(1, args.episodes))
    print(f"Evaluation learnable_seen_rate={rate:.6f} (episodes={args.episodes})")

    if args.upload_wandb:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project or 'hideandseek-xformer', entity=args.wandb_entity or None, reinit=True)
            wandb.log({'eval/learnable_seen_rate': rate, 'eval/dist_bonus_scale': val})
            run.finish()
            print('Uploaded eval result to wandb')
        except Exception as e:
            print('Failed to upload to wandb:', e)

    env.close()
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dist-bonus-scale', type=float, required=True)
    p.add_argument('--episodes', type=int, default=30)
    p.add_argument('--target', type=str, default='seeker')
    p.add_argument('--n-seekers', type=int, default=1)
    p.add_argument('--n-hiders', type=int, default=2)
    p.add_argument('--n-boxes', type=int, default=2)
    p.add_argument('--n-ramps', type=int, default=1)
    p.add_argument('--upload-wandb', dest='upload_wandb', action='store_true', default=True)
    p.add_argument('--no-upload-wandb', dest='upload_wandb', action='store_false')
    p.add_argument('--wandb-project', type=str, default=None)
    p.add_argument('--wandb-entity', type=str, default=None)
    args = p.parse_args()
    return evaluate(args)


if __name__ == '__main__':
    raise SystemExit(main())
