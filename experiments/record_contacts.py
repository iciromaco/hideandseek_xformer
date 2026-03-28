import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from experiments.utils import prepare_env
from src.envs.hns_environment import TeamCosEnv

try:
    import tomllib as _tomlib
except Exception:
    try:
        import tomli as _tomlib
    except Exception:
        _tomlib = None


def _load_action_repeat_from_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hparams_main27.toml")
    if _tomlib is None:
        return None
    try:
        with open(cfg_path, "rb") as f:
            cfg = _tomlib.load(f)
        return int(cfg.get("runtime", {}).get("common", {}).get("action_repeat", 0) or 0)
    except Exception:
        return None


def dump_contacts(env, step):
    out = []
    ncon = int(env.data.ncon)
    for i in range(ncon):
        c = env.data.contact[i]
        try:
            g1 = env.model.geom(c.geom1).name
            g2 = env.model.geom(c.geom2).name
        except Exception:
            g1 = int(c.geom1)
            g2 = int(c.geom2)
        out.append(
            {
                "i": i,
                "geom1": g1,
                "geom2": g2,
                "pos": list(c.pos),
                "frame": list(c.frame),
            }
        )
    return out


def dump_cfrc(env, body_name):
    try:
        bid = env.model.body(body_name).id
    except Exception:
        return None
    # Prefer MuJoCo's applied spatial force (signed) if available, otherwise fall back
    # to cfrc arrays. xfrc_applied (world-frame spatial force) is signed and
    # generally more useful than cfrc_body which in some builds appears zero.
    try:
        arr = env.data.xfrc_applied[bid].tolist()
        return arr
    except Exception:
        pass
    try:
        arr = env.data.cfrc_ext[bid].tolist()
        return arr
    except Exception:
        try:
            arr = env.data.cfrc_body[bid].tolist()
            return arr
        except Exception:
            return None


def main(steps=500, out_json="experiments/contact_trace.json"):
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    env = TeamCosEnv(debug_mode=False, target="seeker")
    ar = _load_action_repeat_from_config()
    if ar:
        env = prepare_env(env, action_repeat=ar, place_far=True)
    else:
        env = prepare_env(env, action_repeat=None, place_far=True)

    # warm
    env.step(env.action_space.sample())

    wait_steps = getattr(env, "prep_steps", 80)
    for _ in range(wait_steps):
        env.step(env.action_space.sample() * 0.0)

    records = []
    a = np.array([0.25, 0.0, 0.0, 0.0], dtype=np.float32)
    for i in range(steps):
        obs, rew, term, trunc, info = env.step(a)
        # basic kinematics
        # Prefer body linear velocity (xvelp) to match how batch runner measures vx.
        v = [float(info.get("agent_vx", 0.0)), float(info.get("agent_vy", 0.0)), 0.0]
        qv = None
        try:
            name = env.learnable_agent_key
            bid = env.body_ids.get(name)
            if bid is not None:
                vv = env.data.xvelp[bid]
                v = [float(vv[0]), float(vv[1]), 0.0]
        except Exception:
            # fall back to joint-based qvel if xvelp unavailable
            try:
                qmap = getattr(env, "qpos_indices", {})
                name = env.learnable_agent_key
                if name in qmap:
                    jx = qmap[name].get("x")
                    if jx is not None:
                        dof_adr = int(env.model.jnt_dofadr[jx])
                        vx = float(env.data.qvel[dof_adr])
                        vy = float(env.data.qvel[dof_adr + 1])
                        v = [vx, vy, 0.0]
                        qv = env.data.qvel[dof_adr : dof_adr + 6].tolist()
            except Exception:
                pass
        # contacts and forces
        contacts = dump_contacts(env, i)
        # keep cfrc dump but use anchor/body if present
        cfrc = dump_cfrc(env, f"{env.learnable_agent_key}_body")
        # also capture applied joint/generalized forces for the agent
        qfrc_agent = None
        xfrc_agent = None
        try:
            # joint-space applied forces/torques (signed)
            qmap = getattr(env, "qpos_indices", {})
            name = env.learnable_agent_key
            if name in qmap:
                jx = qmap[name].get("x")
                if jx is not None:
                    dof_adr = int(env.model.jnt_dofadr[jx])
                    qlen = env.data.qfrc_applied.shape[0]
                    # grab up to 6 entries from joint dof adr
                    end = min(dof_adr + 6, qlen)
                    qfrc_agent = env.data.qfrc_applied[dof_adr:end].tolist()
        except Exception:
            qfrc_agent = None
        try:
            bid = env.model.body(f"{env.learnable_agent_key}_body").id
            xfrc_agent = env.data.xfrc_applied[bid].tolist()
        except Exception:
            xfrc_agent = None
        actuator_forces = None
        try:
            actuator_forces = env.data.actuator_force.tolist()
        except Exception:
            try:
                actuator_forces = env.data.actuator_length[:].tolist()
            except Exception:
                actuator_forces = None
        # prefer the actual ctrl value written to the actuator (signed)
        applied_val = None
        try:
            aid = env.actuator_ids.get(f"{env.learnable_agent_key}_fwd")
            if aid is not None:
                applied_val = float(env.data.ctrl[aid])
        except Exception:
            applied_val = None

        # compute signed_forward same as batch runner: vx*cos(yaw)+vy*sin(yaw)
        signed = None
        try:
            yaw = 0.0
            rot_jid = env.qpos_indices[env.learnable_agent_key]["rot"]
            qadr = env.model.jnt_qposadr[rot_jid]
            yaw = float(env.data.qpos[qadr])
            signed = float(v[0] * np.cos(yaw) + v[1] * np.sin(yaw))
        except Exception:
            try:
                signed = float(info.get("agent_vx", 0.0))
            except Exception:
                signed = None

        records.append(
            {
                "step": i,
                "applied": (float(applied_val) if applied_val is not None else float(info.get("applied_forward", 0.0))),
                "vx": float(v[0]),
                "vy": float(v[1]),
                "signed_forward": signed,
                "qvel_slice": qv,
                "qfrc_applied": qfrc_agent,
                "xfrc_applied": xfrc_agent,
                "contacts": contacts,
                "cfrc_body": cfrc,
                "actuator_forces": actuator_forces,
            }
        )
        if term or trunc:
            env.reset()
    out = {"records": records}
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", out_json)


if __name__ == "__main__":
    main()
