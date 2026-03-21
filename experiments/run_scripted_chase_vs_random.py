#!/usr/bin/env python3
import json
import numpy as np
from statistics import mean
import sys
import pathlib

# ensure project root is on sys.path so `src` imports work when run directly
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# project imports
from src.envs.hns_environment import TeamCosEnv

# Simple scripted seeker policy: for each action, point toward nearest hider and full forward
# Action layout assumed: [fwd, turn, lock, grab] (from earlier logs dbg_last_ctrl_f/t)


def scripted_policy(obs, env):
    # find own index (learnable_agent_key is set to seeker in earlier runs)
    # use env.data to get positions: rely on env internals
    # fallback: random small action if missing
    try:
        # assume single controlled agent is learnable_agent_key
        sk = env.learnable_agent_key
        # get seeker body pos and rotation
        sid = env.body_ids[sk]
        spos = env.data.xpos[sid][:2]
        srot = env.data.qpos[env.model.jnt_qposadr[env.qpos_indices[sk]['rot']]]
        # find nearest hider
        nearest = None
        ndist = 1e9
        for hk in env.hider_keys:
            hid = env.body_ids[hk]
            hpos = env.data.xpos[hid][:2]
            dx = float(hpos[0]-spos[0]); dy = float(hpos[1]-spos[1])
            d = (dx*dx+dy*dy)**0.5
            if d<ndist:
                ndist=d; nearest=(dx,dy)
        if nearest is None:
            return env.action_space.sample()
        dx,dy = nearest
        # desired heading
        desired = np.arctan2(dy, dx)
        heading = srot
        # turn = signed difference clipped
        diff = desired - heading
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        turn = float(np.clip(diff, -1.0, 1.0))
        fwd = 1.0
        # lock/grab zeros
        return np.array([fwd, turn, 0.0, 0.0], dtype=np.float32)
    except Exception:
        return env.action_space.sample()


def random_policy(obs, env):
    return env.action_space.sample()


def run_episode(env, policy, steps=500):
    obs = env.reset()
    total = 0.0
    for t in range(steps):
        a = policy(obs, env)
        obs, r, done, truncated, info = env.step(a)
        total += float(r)
        if done:
            break
    return total


def main():
    N=20
    steps=500
    env = TeamCosEnv()
    env.debug_mode = False
    env.learnable_agent_key = env.seeker_keys[0]  # ensure seeker is learnable

    scripted_scores = []
    random_scores = []

    for i in range(N):
        scripted_scores.append(run_episode(env, scripted_policy, steps=steps))
        random_scores.append(run_episode(env, random_policy, steps=steps))

    out = {
        'scripted_mean': mean(scripted_scores),
        'scripted_all': scripted_scores,
        'random_mean': mean(random_scores),
        'random_all': random_scores,
    }
    print(json.dumps(out, indent=2))
    with open('experiments/scripted_vs_random.json','w') as f:
        json.dump(out, f, indent=2)

if __name__=='__main__':
    main()
