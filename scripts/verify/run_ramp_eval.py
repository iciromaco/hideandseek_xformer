#!/usr/bin/env python3
"""Headless ramp climb evaluation script.

Usage:
  python3 scripts/verify/run_ramp_eval.py --mode debug --target seeker --episodes 3 --max-steps 220 --device cpu
"""

import os
import sys
import argparse
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

import torch

import main27_new as m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="debug")
    p.add_argument("--target", default="seeker", choices=["seeker", "hider"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    # build a temp env to get dims
    env = m.build_env(args.mode, args.target, m.ENV_CONFIG, render_mode=None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    try:
        env.close()
    except Exception:
        pass

    agent = m.AgentV2(obs_dim, act_dim, m.HIDDEN_DIM, m.SEQ_LEN).to(device)
    model_path = args.model_path if args.model_path is not None else m.model_path_for_config(args.target, m.ENV_CONFIG)
    if os.path.exists(model_path):
        try:
            agent.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model: {model_path}")
        except Exception as exc:
            print(f"Failed to load model ({model_path}): {exc}")
    else:
        print(f"No model found at: {model_path} (running with random init)")

    # Use a deterministic placement: place agent under first ramp facing uphill
    env = m.build_env(args.mode, args.target, m.ENV_CONFIG, render_mode=None)
    try:
        obs, _ = env.reset()
        if env.ramp_ids:
            rid = env.ramp_ids[0]
            try:
                rpos = env.data.xpos[rid][:2].copy()
                up = env._ramp_uphill_dir(rid)
                offset_along = -0.6
                place_xy = rpos + up * offset_along
                ak = env.learnable_agent_key
                jx = env.qpos_indices[ak]["x"]
                jy = env.qpos_indices[ak]["y"]
                jz = env.qpos_indices[ak]["z"]
                jr = env.qpos_indices[ak]["rot"]
                qx_adr = env.model.jnt_qposadr[jx]
                qy_adr = env.model.jnt_qposadr[jy]
                qz_adr = env.model.jnt_qposadr[jz]
                qr_adr = env.model.jnt_qposadr[jr]
                ramp_body_z = float(env.data.xpos[rid][2])
                place_z = ramp_body_z + 1.0
                env.data.qpos[qx_adr] = float(place_xy[0])
                env.data.qpos[qy_adr] = float(place_xy[1])
                env.data.qpos[qz_adr] = float(place_z)
                rot = float(__import__('math').atan2(up[1], up[0]))
                env.data.qpos[qr_adr] = rot
                try:
                    import mujoco

                    mujoco.mj_forward(env.model, env.data)
                except Exception:
                    pass
            except Exception:
                pass
    finally:
        try:
            env.close()
        except Exception:
            pass

    result = m.evaluate_fixed_ramp_climb(args.mode, args.target, agent, device, episodes=args.episodes, max_steps=args.max_steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
