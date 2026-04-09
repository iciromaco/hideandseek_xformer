#!/usr/bin/env python3
"""軽量化したランプビューワスクリプト（構文検証済み）

このファイルは内部で `main27_train_final` を読み込み、ビューワを起動します。
"""

import argparse
import os
import sys
import math
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="debug")
    p.add_argument("--target", default="seeker", choices=["seeker", "hider"])
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-path", default=None)
    return p.parse_args()


def main():
    # import heavy project modules inside main to avoid top-level import errors during tooling
    import main27_train_final as m

    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    # infer dims by building a temporary env
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

    env = m.build_env(args.mode, args.target, m.ENV_CONFIG, render_mode="human")
    obs, _ = env.reset()

    # best-effort: position agent near first ramp if available
    try:
        if getattr(env, "ramp_ids", None):
            rid = env.ramp_ids[0]
            rpos = env.data.xpos[rid][:2].copy()
            up = env._ramp_uphill_dir(rid)
            place_xy = rpos + up * -0.6
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
            env.data.qpos[qx_adr] = float(place_xy[0])
            env.data.qpos[qy_adr] = float(place_xy[1])
            env.data.qpos[qz_adr] = ramp_body_z + 1.0
            env.data.qpos[qr_adr] = float(math.atan2(up[1], up[0]))
            try:
                import mujoco

                mujoco.mj_forward(env.model, env.data)
            except Exception:
                pass
    except Exception:
        print("ランプ配置はスキップされました（例外発生）")

    history = m.ObsHistory(1, m.SEQ_LEN, env.observation_space.shape[0], device)
    for ep in range(args.episodes):
        try:
            cur_obs = env._cached_obs if hasattr(env, "_cached_obs") and env._cached_obs is not None else obs
            history.prime_single(cur_obs)
        except Exception:
            try:
                o, _ = env.reset()
                history.prime_single(o)
            except Exception:
                pass

        for step in range(args.max_steps):
            with torch.no_grad():
                seq = history.get()
                out = agent.get_action_and_value(seq)
                action = out[0].cpu().numpy().reshape(-1)
            next_obs, _, term, trun, info = env.step(action)
            history.update(next_obs)
            env.render()
            time.sleep(0.025)
            if term or trun:
                break

    print("Episode(s) finished — viewer will remain open until you interrupt (Ctrl-C).")
    try:
        while True:
            try:
                env.render()
            except Exception:
                pass
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Closing viewer (KeyboardInterrupt)")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
