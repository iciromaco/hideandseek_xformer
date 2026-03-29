#!/usr/bin/env python3
"""簡易ランナー: main27_train_final.evaluate_fixed_ramp_climb を呼び出す

使い方例:
  python scripts/verify/run_eval_fixed_ramp.py --mode debug --target seeker --episodes 3
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

# ルートの src を見つけられるように main と同じパス調整
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

import main27_train_final as m


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="debug")
    p.add_argument("--target", default="seeker", choices=["seeker", "hider"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=220)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-path", default=None)
    p.add_argument("--verbose", action="store_true", help="print per-step info")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    # 環境を一度構築して obs_dim/act_dim を取得（環境の初期化は軽量）
    env = m.build_env(args.mode, args.target, m.ENV_CONFIG, render_mode=None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    try:
        env.close()
    except Exception:
        pass

    agent = m.AgentV2(obs_dim, act_dim, m.HIDDEN_DIM, m.SEQ_LEN).to(device)

    # モデルがあれば読み込む
    model_path = args.model_path if args.model_path is not None else m.model_path_for_config(args.target, m.ENV_CONFIG)
    if os.path.exists(model_path):
        try:
            agent.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model: {model_path}")
        except Exception as exc:
            print(f"Failed to load model ({model_path}): {exc}")
    else:
        print(f"No model found at: {model_path} (running with random init)")

    if not args.verbose:
        res = m.evaluate_fixed_ramp_climb(args.mode, args.target, agent, device, episodes=args.episodes, max_steps=args.max_steps)
        print("evaluate_fixed_ramp_climb =>", res)
        return

    # verbose mode: run episodes step-by-step and print dbg keys when present
    env = m.build_env(args.mode, args.target, m.ENV_CONFIG, render_mode=None)
    history = m.ObsHistory(1, m.SEQ_LEN, env.observation_space.shape[0], device)
    for ep in range(args.episodes):
        obs, _ = env.reset()
        # place learnable agent under first ramp facing uphill (deterministic test pose)
        try:
            if env.ramp_ids:
                rid = env.ramp_ids[0]
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
                # place agent on top of ramp: ramp body z + estimated slope offset
                ramp_body_z = float(env.data.xpos[rid][2])
                # start a bit higher so the agent drops onto the ramp and settles
                place_z = ramp_body_z + 1.0
                env.data.qpos[qx_adr] = float(place_xy[0])
                env.data.qpos[qy_adr] = float(place_xy[1])
                env.data.qpos[qz_adr] = float(place_z)
                rot = float(math.atan2(up[1], up[0]))
                env.data.qpos[qr_adr] = rot
                try:
                    import mujoco

                    mujoco.mj_forward(env.model, env.data)
                except Exception:
                    pass
                # print diagnostic values per ramp
                afwd = np.array([math.cos(rot), math.sin(rot)], dtype=np.float32)
                print(f"[PLACED] agent={ak} place_xy={place_xy} rot={rot:.3f}")
                for i, rid in enumerate(env.ramp_ids, start=1):
                    rkey = f"ramp{i}"
                    rpos = env.data.xpos[rid][:2]
                    up = env._ramp_uphill_dir(rid)
                    side = np.array([-up[1], up[0]], dtype=np.float32)
                    rel = env.data.xpos[env.body_ids[ak]][:2] - rpos
                    lx = float(np.dot(rel, up))
                    ly = float(np.dot(rel, side))
                    facing = float(np.dot(afwd, up))
                    print(f"[RAMP_DBG] {rkey} lx={lx:.3f} ly={ly:.3f} facing={facing:.3f} rel={rel} up={up} side={side} afwd={afwd}")
                try:
                    print("_ramp_boost_gain=", float(env._ramp_boost_gain(env.learnable_agent_key)))
                except Exception:
                    pass
        except Exception:
            pass
        history.prime_single(obs)
        for step in range(args.max_steps):
            with torch.no_grad():
                seq = history.get()
                out = agent.get_action_and_value(seq)
                action = out[0].cpu().numpy().reshape(-1)
            next_obs, _, term, trun, info = env.step(action)
            history.update(next_obs)
            # print dbg values
            # debug: print full info on first step to ensure keys exist
            if step == 0:
                print(f"[DEBUG_INFO_KEYS] ep={ep} step={step} keys={list(info.keys())}")
                try:
                    print(info)
                except Exception:
                    pass
                # additional diagnostics: qpos.z, anchor/body xpos, geom bottom z
                try:
                    ak = env.learnable_agent_key
                    jz = env.qpos_indices[ak]["z"]
                    qz_adr = env.model.jnt_qposadr[jz]
                    qz_val = float(env.data.qpos[qz_adr])
                    anchor_bid = env.model.body(f"{ak}_anchor").id
                    body_bid = env.body_ids[ak]
                    anchor_z = float(env.data.xpos[anchor_bid][2])
                    body_z = float(env.data.xpos[body_bid][2])
                    print(f"[POS_DEBUG] qpos.z={qz_val} anchor_z={anchor_z} body_z={body_z}")
                    # geom bottoms
                    gids = env.agent_geom_ids.get(ak, [])
                    bottoms = []
                    for gid in gids:
                        try:
                            gpos_z = float(env.data.geom_xpos[gid][2])
                            gsize0 = float(env.model.geom_size[gid][0])
                            bottom_z = gpos_z - gsize0
                            bottoms.append((gid, bottom_z))
                        except Exception:
                            pass
                    if bottoms:
                        s = ", ".join(f"gid={g[0]} bottom_z={g[1]:.3f}" for g in bottoms)
                        print(f"[GEOM_BOTTOMS] {s}")
                except Exception:
                    pass
            # always print agent height and vertical velocity; prefer neutral keys
            dbg_boost = info.get("dbg_ramp_boost")
            dbg_h = info.get("agent_world_z", info.get("dbg_agent_z", info.get("dbg_ramp_height")))
            dbg_vz = info.get("agent_world_vz", info.get("dbg_agent_vz", info.get("agent_vz")))
            dbg_p = info.get("ramp_progress", info.get("dbg_ramp_progress"))
            print(f"ep={ep} step={step} agent_world_z={dbg_h} agent_world_vz={dbg_vz} ramp_progress={dbg_p} dbg_ramp_boost={dbg_boost}")
            if term or trun:
                break
    env.close()


if __name__ == "__main__":
    main()
