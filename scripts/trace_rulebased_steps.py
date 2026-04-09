"""
Trace RuleBased agents over multiple steps to observe actions and internal timers.
Run:
  cd /Users/dan/Desktop/Semi/hideandseek_xformer
  python3 scripts/trace_rulebased_steps.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
from src.envs.hns_environment import TeamCosEnv


def main(steps=200):
    env = TeamCosEnv(debug_mode=True, n_seekers=1, n_hiders=2)
    obs, info = env.reset()
    print("Reset obs shape:", obs.shape)

    for step in range(steps):
        print(f"\n=== step {step} ===")
        for ak in env.agent_keys:
            npc = env.npcs.get(ak)
            # extract a few internal attrs if present
            attrs = {}
            for a in ('wander_timer', 'reflex_timer', 'stuck_counter', 'interact_cooldown', 'interact_focus_steps'):
                attrs[a] = getattr(npc, a, None)
            raw_obs = env._get_obs(env.agent_keys.index(ak))
            norm_obs = env._normalize_obs(raw_obs)
            try:
                act = env.policy_adapter.get_action(ak, norm_obs)
            except Exception as e:
                act = f"ERROR: {e}"
            # also print world position and visibility towards the learnable agent for diagnosis
            body_id = env.body_ids[ak]
            pos = env.data.xpos[body_id][:2].copy()
            vis_to_learnable = env._is_vis(pos, env.data.qpos[env.model.jnt_qposadr[env.qpos_indices[ak]['rot']]], env.data.xpos[env.body_ids[env.learnable_agent_key]][:2], env.body_ids[ak], env.body_ids[env.learnable_agent_key])
            print(f"{ak}: pos={pos} attrs={attrs} visible_to_learnable={bool(vis_to_learnable)} action={act}")

        # step environment with zero action for the learnable agent to advance sim
        zero_action = np.zeros(env.action_space.shape[0], dtype=np.float32)
        out = env.step(zero_action)
        # (obs, reward, term, trunc, info)
        obs = out[0] if isinstance(out, tuple) else out
        time.sleep(0.01)

if __name__ == '__main__':
    main(steps=120)
