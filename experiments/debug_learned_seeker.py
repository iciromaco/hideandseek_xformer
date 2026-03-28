import json
import os
import sys

import numpy as np
import torch

# sys.path拡張（src配下の独自モジュール用）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from envs.hns_environment import TeamCosEnv


def model_path_for_config(target, config):
    fname = f"HNS_V27_{target}_s{config.get('n_seekers', 1)}_h{config.get('n_hiders', 2)}_b{config.get('n_boxes', 2)}_r{config.get('n_ramps', 1)}.pt"
    return os.path.join("checkpoints", fname)


OUT = "experiments/model_debug_seeker.jsonl"
os.makedirs("experiments", exist_ok=True)

# locate seeker model if exists (and infer env config from filename)
model_path = model_path_for_config("seeker", {"n_seekers": 1, "n_hiders": 1, "n_boxes": 0, "n_ramps": 0})
state = None
inferred_cfg = None
if os.path.exists(model_path):
    try:
        state = torch.load(model_path, map_location="cpu")
        print("Loaded seeker model:", model_path)
        inferred_path = model_path
    except Exception as e:
        print("Failed to load model:", e)
else:
    # fallback: search for any available seeker checkpoint in checkpoints/
    try:
        import glob
        import re

        candidates = sorted(glob.glob(os.path.join("checkpoints", "HNS_V27_seeker*.pt")))
        if candidates:
            model_path = candidates[-1]
            inferred_path = model_path
            try:
                state = torch.load(model_path, map_location="cpu")
                print("Loaded seeker model (fallback):", model_path)
            except Exception as e:
                print("Failed to load fallback model:", e)
        else:
            print("Model not found at", model_path)
    except Exception:
        print("Model not found at", model_path)

# if we found a checkpoint, try to infer environment config from filename
if state is not None:
    try:
        import re

        m = re.search(
            r"HNS_V27_seeker_s(\d+)_h(\d+)_b(\d+)_r(\d+)\.pt",
            os.path.basename(inferred_path),
        )
        if m:
            inferred_cfg = {
                "n_seekers": int(m.group(1)),
                "n_hiders": int(m.group(2)),
                "n_boxes": int(m.group(3)),
                "n_ramps": int(m.group(4)),
            }
            print("Inferred env config from filename:", inferred_cfg)
    except Exception:
        inferred_cfg = None

# build env (use inferred config if available so obs_dim matches checkpoint)
if inferred_cfg is not None:
    env = TeamCosEnv(
        mode="initial",
        target="seeker",
        n_seekers=inferred_cfg.get("n_seekers", 1),
        n_hiders=inferred_cfg.get("n_hiders", 1),
        n_boxes=inferred_cfg.get("n_boxes", 0),
        n_ramps=inferred_cfg.get("n_ramps", 0),
        debug_mode=True,
    )
else:
    env = TeamCosEnv(
        mode="initial",
        target="seeker",
        n_seekers=1,
        n_hiders=1,
        n_boxes=0,
        n_ramps=0,
        debug_mode=True,
    )

# if checkpoint loaded, try to extract model_state_dict and matching hidden/seq sizes
if state is not None:
    model_state = state
    # common checkpoint wrapper key
    if isinstance(state, dict) and "model_state_dict" in state:
        model_state = state["model_state_dict"]

    # infer hidden dim from obs_encoder weight if possible
    hidden_dim = 128
    seq_len = 8
    try:
        for k in model_state.keys():
            if k.endswith("obs_encoder.0.weight") or k.endswith("obs_encoder.0.weight"):
                w = model_state[k]
                hidden_dim = int(w.shape[0])
                break
    except Exception:
        pass

    try:
        # some checkpoints may store seq_len metadata
        if isinstance(state, dict) and "seq_len" in state:
            seq_len = int(state["seq_len"])
        elif isinstance(state, dict) and "args" in state and isinstance(state["args"], dict) and "seq_len" in state["args"]:
            seq_len = int(state["args"]["seq_len"])
    except Exception:
        pass

    try:
        env.set_inference_policy_state(
            [env.learnable_agent_key],
            model_state,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
        )
        env.set_override_learnable_policy(True)
        env.set_model_policy_deterministic(True)
        print(f"Applied checkpoint to env inference model (hidden_dim={hidden_dim}, seq_len={seq_len})")
    except Exception as e:
        print("Failed to apply checkpoint to env model:", e)
else:
    print("No model applied; enable override to use rulebase")

# run one episode and log steps where hider is visible to learner
with open(OUT, "w") as fo:
    obs, info = env.reset()
    done = False
    step = 0
    while not done and step < 500:
        # prepare action: if model applied via override, pass zeros (external action ignored)
        a = np.zeros(env.action_space.shape, dtype=np.float32)
        next_obs, base_r, term, trunc, info = env.step(a)
        # raw obs for learner
        raw_obs = env._get_obs(env.learnable_agent_index)
        norm_obs = env._normalize_obs(raw_obs)
        # log when learner 'sees' hider or when visible flag in obs
        vis_flag = False
        try:
            en_idx = env.idx.OTHERS[0]
            vis_flag = bool(raw_obs[en_idx.VISIBLE] > 0.5) if hasattr(en_idx, "VISIBLE") else False
        except Exception:
            # fallback to info
            vis_flag = bool(info.get("dbg_learnable_hider_seen", False))
        # also log if info dbg flag true
        if vis_flag or info.get("dbg_learnable_hider_seen", False):
            # get model action output if model present
            model_act = None
            if env._inference_models.get(env.learnable_agent_key) is not None:
                norm = env._normalize_obs(env._get_obs(env.learnable_agent_index))
                seq = env._get_policy_history_seq(
                    env.learnable_agent_key,
                    env._inference_seq_lens.get(env.learnable_agent_key, 8),
                    norm,
                )
                import torch

                seq_t = torch.as_tensor(seq[None, :, :], dtype=torch.float32)
                with torch.no_grad():
                    m = env._inference_models[env.learnable_agent_key]
                    if env.model_policy_deterministic and hasattr(m, "get_deterministic_action_and_value"):
                        arr = m.get_deterministic_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
                    else:
                        arr = m.get_action_and_value(seq_t)[0].cpu().numpy().reshape(-1)
                    model_act = arr.tolist()
            rec = {
                "step": step,
                "info": {
                    k: info.get(k)
                    for k in [
                        "dbg_learnable_hider_seen",
                        "is_detected",
                        "wall_distance",
                        "agent_vx",
                        "agent_vy",
                        "dbg_last_ctrl_f",
                        "dbg_last_ctrl_t",
                    ]
                    if k in info
                },
                "raw_obs_rel": {
                    "rel_x": (float(raw_obs[env.idx.OTHERS[0].REL_X]) if hasattr(env.idx.OTHERS[0], "REL_X") else None),
                    "rel_y": (float(raw_obs[env.idx.OTHERS[0].REL_Y]) if hasattr(env.idx.OTHERS[0], "REL_Y") else None),
                    "visible": (float(raw_obs[env.idx.OTHERS[0].VISIBLE]) if hasattr(env.idx.OTHERS[0], "VISIBLE") else None),
                },
                "norm_obs_sample": norm_obs.tolist(),
                "model_action": model_act,
                "applied_ctrl": info.get("applied_forward", None),
                "base_reward": float(base_r),
            }
            fo.write(json.dumps(rec) + "\n")
        step += 1
        done = term or trunc
print("Wrote logs to", OUT)
