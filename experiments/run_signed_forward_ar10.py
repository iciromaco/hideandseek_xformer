"""Run signed-forward probe with action_repeat=10 and compute correlations.
Writes JSON to experiments/signed_forward_ar10.json and prints correlations.
"""
import json
import numpy as np
import os
import sys

# ensure project root on PYTHONPATH when running from experiments/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.hns_environment import TeamCosEnv


def run(steps=1000, out_path="experiments/signed_forward_ar10.json"):
    env = TeamCosEnv()
    # Force action_repeat to 10 to match main27
    try:
        env.action_repeat = 10
    except Exception:
        pass

    obs = env.reset()
    # warm-step to init internal state
    try:
        env.step(env.action_space.sample())
    except Exception:
        pass
    records = []
    for t in range(steps):
        action = env.action_space.sample()
        step_out = env.step(action)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs, rew, term, trunc, info = step_out
            done = bool(term or trunc)
        else:
            obs, rew, done, info = step_out
        # compute world-frame vx, vy from body linear velocity
        try:
            learnable = env.learnable_agent_key
            body_id = env.body_ids[learnable]
            v = env.data.xvelp[body_id]
            vx = float(v[0])
            vy = float(v[1])
        except Exception:
            vx = float(info.get('agent_vx', 0.0))
            vy = float(info.get('agent_vy', 0.0))

        # get yaw
        yaw = None
        try:
            learnable = env.learnable_agent_key
            rot_jid = env.qpos_indices[learnable]['rot']
            qpos_adr = env.model.jnt_qposadr[rot_jid]
            yaw = float(env.data.qpos[qpos_adr])
        except Exception:
            yaw = info.get('agent_yaw', 0.0)

        sfwd = float(vx * np.cos(yaw) + vy * np.sin(yaw))
        applied = info.get("applied_forward")
        records.append({
            "t": t,
            "action0": float(action[0]),
            "applied_forward": None if applied is None else float(applied),
            "signed_forward": sfwd,
        })
        if done:
            obs = env.reset()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f)

    arr = [r for r in records if r["signed_forward"] is not None]
    if len(arr) == 0:
        print("No signed_forward values recorded.")
        return

    a0 = np.array([r["action0"] for r in arr])
    applied = np.array([r["applied_forward"] if r["applied_forward"] is not None else np.nan for r in arr])
    sf = np.array([r["signed_forward"] for r in arr])

    # drop NaNs in applied for applied vs signed_forward correlation
    mask = ~np.isnan(applied)
    if mask.sum() > 0:
        corr_applied = np.corrcoef(applied[mask], sf[mask])[0, 1]
    else:
        corr_applied = None

    corr_a0 = np.corrcoef(a0, sf)[0, 1]

    print(f"Wrote: {out_path}")
    print(f"N samples: {len(arr)}")
    print(f"corr(action0, signed_forward) = {corr_a0:.6f}")
    if corr_applied is not None:
        print(f"corr(applied_forward, signed_forward) = {corr_applied:.6f}")
    else:
        print("No applied_forward samples to correlate.")


if __name__ == "__main__":
    run()
