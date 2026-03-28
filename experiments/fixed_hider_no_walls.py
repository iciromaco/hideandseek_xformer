import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from envs.hns_environment import TeamCosEnv
from experiments.utils import prepare_env


def find_latest_seeker_checkpoint():
    import glob

    candidates = sorted(glob.glob(os.path.join("checkpoints", "HNS_V27_seeker*.pt")))
    if not candidates:
        return None
    return candidates[-1]


def set_hider_pos_fixed(env, hider_key, x, y, z=0.5):
    try:
        bid = env.model.body(f"{hider_key}_anchor").id
        qadr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
        # quaternion w,x,y,z for no rotation
        env.data.qpos[qadr : qadr + 7] = [
            float(x),
            float(y),
            float(z),
            1.0,
            0.0,
            0.0,
            0.0,
        ]
    except Exception:
        # fallback: try qpos_indices mapping if available
        try:
            qmap = env.qpos_indices[hider_key]
            env.data.qpos[env.model.jnt_qposadr[qmap["x"]]] = float(x)
            env.data.qpos[env.model.jnt_qposadr[qmap["y"]]] = float(y)
            env.data.qpos[env.model.jnt_qposadr[qmap["z"]]] = float(z)
        except Exception:
            pass


def main(steps=400, show_progress=True):
    # build env and remove inner walls via prepare_env(place_far=True) edit
    env = TeamCosEnv(debug_mode=False, target="seeker")
    env = prepare_env(env, action_repeat=None, place_far=True)

    # try to load latest seeker checkpoint if present
    ckpt = find_latest_seeker_checkpoint()
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu")
        model_state = state.get("model_state_dict", state) if isinstance(state, dict) else state
        try:
            # try default load first
            env.set_inference_policy_state([env.learnable_agent_key], model_state, seq_len=8, hidden_dim=128)
            env.set_model_policy_deterministic(True)
            # enable env-side override so the loaded inference policy is used
            env.set_override_learnable_policy(True)
            print("Loaded seeker checkpoint:", ckpt)
        except Exception as e:
            # attempt to infer hidden_dim (and seq_len) from checkpoint and retry
            inferred_hidden = None
            inferred_seq = None
            try:
                if isinstance(model_state, dict):
                    # look for obs_encoder weight to infer hidden dim
                    for k, v in model_state.items():
                        if k.endswith("obs_encoder.0.weight") or k.endswith("obs_encoder.0.weight"):
                            inferred_hidden = int(v.shape[0])
                            break
                    # generic fallback: any obs_encoder.*.weight
                    if inferred_hidden is None:
                        for k, v in model_state.items():
                            if k.startswith("obs_encoder") and k.endswith(".weight") and v.ndim == 2:
                                inferred_hidden = int(v.shape[0])
                                break
                # try top-level metadata
                if isinstance(state, dict):
                    if inferred_seq is None:
                        inferred_seq = state.get("seq_len", None)
                    if inferred_hidden is None:
                        inferred_hidden = state.get("hidden_dim", None)
            except Exception:
                pass

            if inferred_hidden is not None:
                try:
                    seq_try = int(inferred_seq) if inferred_seq is not None else 8
                    env.set_inference_policy_state(
                        [env.learnable_agent_key],
                        model_state,
                        seq_len=seq_try,
                        hidden_dim=int(inferred_hidden),
                    )
                    env.set_model_policy_deterministic(True)
                    env.set_override_learnable_policy(True)
                    print(
                        "Loaded seeker checkpoint with inferred hidden_dim:",
                        inferred_hidden,
                        "seq_len:",
                        seq_try,
                    )
                except Exception as e2:
                    print("Failed to load checkpoint after inferring dims:", e2)
            else:
                print("Failed to load checkpoint into env:", e)
    else:
        print("No seeker checkpoint found; running without learned model")

    # reset and place hider at fixed location relative to seeker
    obs, info = env.reset()
    # get seeker pos
    s_key = env.learnable_agent_key
    s_bid = env.body_ids[s_key]
    s_pos = env.data.xpos[s_bid][:2].copy()

    # place hider at 1.5m ahead of seeker in world frame
    # compute forward unit from seeker yaw
    try:
        rot_jid = env.qpos_indices[s_key]["rot"]
        qadr = env.model.jnt_qposadr[rot_jid]
        yaw = float(env.data.qpos[qadr])
    except Exception:
        yaw = 0.0
    hx = float(s_pos[0] + 1.5 * np.cos(yaw))
    hy = float(s_pos[1] + 1.5 * np.sin(yaw))
    h_key = env.hider_keys[0]
    set_hider_pos_fixed(env, h_key, hx, hy, z=0.5)
    # forward kinematics
    try:
        import mujoco

        mujoco.mj_forward(env.model, env.data)
    except Exception:
        pass

    # run steps, but after each env.step teleport hider back to fixed pos
    records = []
    for i in range(steps):
        # step with no external override (use model if loaded)
        obs, r, term, done, info = env.step(env.action_space.sample() * 0.0)

        # re-teleport hider to fixed position so it cannot move
        set_hider_pos_fixed(env, h_key, hx, hy, z=0.5)
        try:
            mujoco.mj_forward(env.model, env.data)
        except Exception:
            pass

        # measure approach: vector from seeker to hider and agent velocity
        sbid = env.body_ids[s_key]
        hbid = env.body_ids[h_key]
        spos = env.data.xpos[sbid][:2]
        hpos = env.data.xpos[hbid][:2]
        rel = np.array([hpos[0] - spos[0], hpos[1] - spos[1]], dtype=np.float32)
        dist = float(np.linalg.norm(rel))
        # prefer body linear velocity if available, fall back to joint qvel mapping
        try:
            vv = env.data.xvelp[sbid]
            vx = float(vv[0])
            vy = float(vv[1])
        except Exception:
            try:
                qmap = getattr(env, "qpos_indices", {})
                jx = qmap[s_key].get("x")
                if jx is not None:
                    dof_adr = int(env.model.jnt_dofadr[jx])
                    vx = float(env.data.qvel[dof_adr])
                    vy = float(env.data.qvel[dof_adr + 1])
                else:
                    vx = 0.0
                    vy = 0.0
            except Exception:
                vx = 0.0
                vy = 0.0
        speed = float(np.linalg.norm([vx, vy]))
        dot = float((vx * rel[0] + vy * rel[1]) / (dist + 1e-8))

        # collect contact forces if available
        try:
            xfrc = env.data.xfrc_applied[sbid].tolist()
        except Exception:
            try:
                xfrc = env.data.cfrc_ext[sbid].tolist()
            except Exception:
                xfrc = None

        # collect contacts touching this body
        contacts = []
        try:
            nc = env.data.ncon
            for ci in range(int(nc)):
                c = env.data.contact[ci]
                if int(c.geom1) in getattr(env, "obj_geom_ids", {}).values() or int(c.geom2) in getattr(env, "obj_geom_ids", {}).values():
                    pass
                # include contact if either body matches seeker body id
                if int(c.bi) == sbid or int(c.bj) == sbid:
                    contacts.append(
                        {
                            "geom1": int(c.geom1),
                            "geom2": int(c.geom2),
                            "frame": [
                                float(c.frame[0]),
                                float(c.frame[1]),
                                float(c.frame[2]),
                            ],
                        }
                    )
        except Exception:
            contacts = []

        rec = {
            "step": i,
            "dist": dist,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "toward_dot": dot,
            "applied_forward_model": info.get("applied_forward_model"),
            "applied_forward": info.get("applied_forward"),
            "dbg_last_ctrl_f": info.get("dbg_last_ctrl_f"),
            "dbg_last_ctrl_t": info.get("dbg_last_ctrl_t"),
            "agent_vx": info.get("agent_vx"),
            "agent_vy": info.get("agent_vy"),
            "xfrc_applied": xfrc,
            "contacts": contacts,
            "info_keys": list(info.keys()),
        }
        records.append(rec)

        if show_progress and i % 25 == 0:
            print(
                f"step={i} dist={dist:.3f} speed={speed:.3f} dot={dot:.3f} applied={rec['applied_forward']:.3f} last_ctrl={rec['dbg_last_ctrl_f']:.3f} vx={rec['agent_vx']:.3f} xfrc={rec['xfrc_applied']}"
            )

        if done:
            break

    # summarize
    arr = np.array([r["toward_dot"] for r in records], dtype=np.float32)
    mean_dot = float(arr.mean()) if arr.size else float("nan")
    print("Mean toward_dot (positive -> moving toward hider):", mean_dot)
    outp = {"summary": {"mean_toward_dot": mean_dot}, "records": records}
    with open("experiments/fixed_hider_no_walls.json", "w") as f:
        json.dump(outp, f, indent=2)
    print("Wrote experiments/fixed_hider_no_walls.json")


if __name__ == "__main__":
    main(steps=400)
