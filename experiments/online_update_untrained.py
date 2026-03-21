import os
import sys
import json
import time
import numpy as np
import torch
from PIL import Image
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from envs.hns_environment import TeamCosEnv
from experiments.utils import prepare_env
from models.ppo_transformer_v2 import AgentV2


def set_hider_pos_fixed(env, hider_key, x, y, z=0.5):
    try:
        bid = env.model.body(f"{hider_key}_anchor").id
        qadr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
        env.data.qpos[qadr:qadr+7] = [float(x), float(y), float(z), 1.0, 0.0, 0.0, 0.0]
        print(f"[set_hider_pos_fixed] qpos書き換え直後 hider xpos={env.data.xpos[bid]}")
    except Exception:
        try:
            qmap = env.qpos_indices[hider_key]
            env.data.qpos[env.model.jnt_qposadr[qmap['x']]] = float(x)
            env.data.qpos[env.model.jnt_qposadr[qmap['y']]] = float(y)
            env.data.qpos[env.model.jnt_qposadr[qmap['z']]] = float(z)
            bid = env.model.body(f"{hider_key}_anchor").id
            print(f"[set_hider_pos_fixed] qpos書き換え直後 hider xpos={env.data.xpos[bid]}")
        except Exception:
            pass
    # attempt to zero velocities for the hider's joint(s) to avoid explosive contact responses
    try:
        # prefer body anchor joint
        bid = env.model.body(f"{hider_key}_anchor").id
        jnt = env.model.body_jntadr[bid]
        if jnt >= 0:
            dofadr = env.model.jnt_dofadr[jnt]
            ndof = env.model.jnt_dofnum[jnt]
            if ndof > 0:
                env.data.qvel[dofadr:dofadr+ndof] = 0.0
    except Exception:
        try:
            # fallback: zero any DOFs referenced in qpos_indices for this key
            qmap = env.qpos_indices.get(hider_key, {})
            for k, idx in qmap.items():
                try:
                    # idx might be a qpos index or a joint index; try to map to dofadr
                    if isinstance(idx, int):
                        # find joint that owns this qpos adr
                        for j in range(env.model.njnt):
                            qadr_j = env.model.jnt_qposadr[j]
                            if qadr_j <= idx < qadr_j + env.model.jnt_qposadr[j+1] if j+1 < env.model.njnt else qadr_j <= idx:
                                da = env.model.jnt_dofadr[j]
                                dn = env.model.jnt_dofnum[j]
                                if dn > 0:
                                    env.data.qvel[da:da+dn] = 0.0
                                    break
                except Exception:
                    continue
        except Exception:
            pass
    try:
        import mujoco
        # Ensure velocities/forces are zeroed to avoid explosive contact responses
        try:
            env.data.qvel[:] = 0.0
        except Exception:
            pass
        try:
            env.data.qacc[:] = 0.0
        except Exception:
            pass
        try:
            env.data.qfrc_applied[:] = 0.0
        except Exception:
            pass
        try:
            env.data.qacc_warmstart[:] = 0.0
        except Exception:
            pass
        mujoco.mj_forward(env.model, env.data)
        # mj_forward直後のxposをprint
        try:
            bid = env.model.body(f"{hider_key}_anchor").id
            print(f"[set_hider_pos_fixed] mj_forward直後 hider xpos={env.data.xpos[bid]}")
        except Exception:
            pass
    except Exception:
        pass


def make_seq_input(norm_obs, seq_len):
    arr = np.tile(np.asarray(norm_obs, dtype=np.float32), (seq_len, 1))
    return torch.from_numpy(arr[np.newaxis, ...])  # shape (1, seq_len, obs_dim)


def evaluate_policy(agent, env, seq_len=8, steps=80, render=False):
    obs, info = env.reset()
    # place hider fixed in front
    s_key = env.learnable_agent_key
    s_bid = env.body_ids[s_key]
    s_pos = env.data.xpos[s_bid][:2].copy()
    # determine seeker yaw using environment's qpos `rot` first (matches _is_vis)
    try:
        rot_jid = env.qpos_indices[s_key]['rot']
        qadr = env.model.jnt_qposadr[rot_jid]
        yaw = float(env.data.qpos[qadr])
    except Exception:
        # fall back to body quaternion or rotation matrix if qpos not available
        try:
            q = env.data.xquat[s_bid]
            q0, q1, q2, q3 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            yaw = float(np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3)))
        except Exception:
            try:
                mat = np.asarray(env.data.xmat[s_bid]).reshape(3, 3)
                fx, fy = float(mat[0, 0]), float(mat[1, 0])
                yaw = float(np.arctan2(fy, fx))
            except Exception:
                yaw = 0.0
    hx = float(s_pos[0] + 1.5 * np.cos(yaw))
    hy = float(s_pos[1] + 1.5 * np.sin(yaw))
    h_key = env.hider_keys[0]
    # place slightly higher to avoid penetration impulses
    set_hider_pos_fixed(env, h_key, hx, hy, z=0.8)
    try:
        import mujoco
        mujoco.mj_forward(env.model, env.data)
    except Exception:
        pass

    # Diagnostic: print seeker/hider pose and visibility
    try:
        s_pos = env.data.xpos[s_bid][:2].copy()
        mat = np.asarray(env.data.xmat[s_bid]).reshape(3, 3)
        fx, fy = float(mat[0, 0]), float(mat[1, 0])
        yaw_xmat = float(np.arctan2(fy, fx))
    except Exception:
        s_pos = env.data.xpos[s_bid][:2].copy()
        try:
            rot_jid = env.qpos_indices[s_key]['rot']
            qadr = env.model.jnt_qposadr[rot_jid]
            yaw_xmat = float(env.data.qpos[qadr])
        except Exception:
            yaw_xmat = 0.0
    # compute rot from qpos if available and frontness used by _is_vis
    try:
        rot_jid = env.qpos_indices[s_key]['rot']
        qadr = env.model.jnt_qposadr[rot_jid]
        rot_qpos = float(env.data.qpos[qadr])
    except Exception:
        rot_qpos = yaw_xmat
    rel = np.array([hx - s_pos[0], hy - s_pos[1]], dtype=np.float32)
    dist_diag = float(np.linalg.norm(rel))
    frontness = float(np.cos(rot_qpos) * (rel[0] / (dist_diag + 1e-8)) + np.sin(rot_qpos) * (rel[1] / (dist_diag + 1e-8)))
    print(f'[DIAG] seeker_pos={s_pos} yaw_xmat={yaw_xmat:.3f} rot_qpos={rot_qpos:.3f} frontness={frontness:.3f} hider_pos={[hx,hy]} dist={dist_diag:.3f}')
    try:
        vis = env.vis_engine.is_visible(np.asarray([s_pos[0], s_pos[1], 0.4]), np.asarray([hx, hy, 0.4]), mode=1, body_exclude=env.body_ids.get(s_key, -1), target_body_id=env.body_ids.get(h_key, -1))
        print(f'[DIAG] vis_engine.is_visible -> {vis}  info.is_detected (after reset) = {info.get("is_detected")}')
    except Exception as e:
        print('[DIAG] vis check failed:', e)

    records = []
    for i in range(steps):
        norm_obs = env._normalize_obs(env._get_obs(env.learnable_agent_index))
        x = make_seq_input(norm_obs, seq_len)
        with torch.no_grad():
            a_mean, _, _, _ = agent.get_deterministic_action_and_value(x)
        act = a_mean.cpu().numpy().ravel()
        full_act = np.zeros(env.action_space.shape[0], dtype=np.float32)
        full_act[: len(act)] = act
        obs, r, term, done, info = env.step(full_act)
        # compute toward dot
        sbid = env.body_ids[s_key]
        hbid = env.body_ids[h_key]
        spos = env.data.xpos[sbid][:2]
        hpos = env.data.xpos[hbid][:2]
        rel = np.array([hpos[0]-spos[0], hpos[1]-spos[1]], dtype=np.float32)
        dist = float(np.linalg.norm(rel))
        try:
            vadr = env.model.jnt_dofadr[env.model.body_jntadr[sbid]]
            vx = float(env.data.qvel[vadr])
            vy = float(env.data.qvel[vadr+1])
        except Exception:
            vx = float(info.get('agent_vx', 0.0)); vy = float(info.get('agent_vy', 0.0))
        dot = float((vx*rel[0] + vy*rel[1]) / (dist + 1e-8))
        records.append(dot)
        if done:
            break
    arr = np.array(records, dtype=np.float32)
    return float(np.nanmean(arr)) if arr.size else float('nan')


def main():
    seq_len = 8
    hidden_dim = 128
    env = TeamCosEnv(debug_mode=True, target='seeker')
    env = prepare_env(env, action_repeat=None, place_far=True)
    # Try passive render; if that doesn't show, allow forcing a blocking viewer
    try:
        env.render()
    except Exception as e:
        import traceback
        print('[RENDER ERROR] initial render failed:', e)
        traceback.print_exc()
        # attempt offscreen save for debugging when interactive viewer fails
        try:
            import mujoco
            renderer = mujoco.Renderer(device=0)
            scene = mujoco.Scene(env.model, env.data)
            renderer.render(scene)
            rgb, depth = renderer.read_pixels()
            img = Image.fromarray(rgb)
            img.save('experiments/online_update_initial_frame.png')
            print('Wrote experiments/online_update_initial_frame.png')
        except Exception as _e:
            print('[RENDER ERROR] offscreen initial save failed:', _e)
            traceback.print_exc()

    if os.environ.get('BLOCK_VIEWER', '0') == '1':
        try:
            import mujoco
            print('[VIEWER] Launching blocking viewer (BLOCK_VIEWER=1)')
            mujoco.viewer.launch(env.model, env.data)
        except Exception as e:
            import traceback
            print('[VIEWER ERROR] blocking launch failed:', e)
            traceback.print_exc()
            # fallback: try offscreen render and save a frame for inspection
            try:
                try:
                    renderer = mujoco.Renderer(device=0)
                except TypeError:
                    renderer = mujoco.Renderer()
                scene = mujoco.Scene(env.model, env.data)
                renderer.render(scene)
                rgb, depth = renderer.read_pixels()
                img = Image.fromarray(rgb)
                img.save('experiments/online_update_block_viewer_frame.png')
                print('Wrote experiments/online_update_block_viewer_frame.png')
                try:
                    renderer.close()
                except Exception:
                    pass
            except Exception as _e:
                print('[VIEWER ERROR] offscreen fallback failed:', _e)
                traceback.print_exc()
    # reduce warmup so initial drop/convergence is allowed but not too long
    env.prep_steps = 20

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = AgentV2(obs_dim, act_dim, hidden_dim, seq_len)
    agent.train()
    optim = torch.optim.Adam(agent.parameters(), lr=3e-4)

    # ensure we DO NOT include warmup transitions
    env_pre_steps = env.prep_steps
    print('env.prep_steps =', env_pre_steps)

    # initial evaluation
    pre_score = evaluate_policy(agent, env, seq_len=seq_len, steps=80)
    print('Pre-update mean toward_dot:', pre_score)

    # run until we see the hider (after warmup) and collect a short rollout
    rollout_len = 16
    obs, info = env.reset()
    # set fixed hider like eval
    s_key = env.learnable_agent_key
    s_bid = env.body_ids[s_key]
    s_pos = env.data.xpos[s_bid][:2].copy()
    # prefer body anchor quaternion for yaw (handles nested bodies reliably)
    # prefer qpos `rot` so placement matches env's visibility frontness
    try:
        rot_jid = env.qpos_indices[s_key]['rot']
        qadr = env.model.jnt_qposadr[rot_jid]
        yaw = float(env.data.qpos[qadr])
    except Exception:
        try:
            q = env.data.xquat[s_bid]
            q0, q1, q2, q3 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            yaw = float(np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3)))
        except Exception:
            try:
                mat = np.asarray(env.data.xmat[s_bid]).reshape(3, 3)
                fx, fy = float(mat[0, 0]), float(mat[1, 0])
                yaw = float(np.arctan2(fy, fx))
            except Exception:
                yaw = 0.0
    # place hider slightly closer than 1.5m but avoid overlap
    HIDER_DIST = 1.2
    hx = float(s_pos[0] + HIDER_DIST * np.cos(yaw))
    hy = float(s_pos[1] + HIDER_DIST * np.sin(yaw))
    h_key = env.hider_keys[0]
    # place slightly higher to reduce contact explosions on teleport
    set_hider_pos_fixed(env, h_key, hx, hy, z=0.8)
    try:
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        try:
            env.render()
        except Exception as e:
            import traceback
            print('[RENDER ERROR] forward render failed:', e)
            traceback.print_exc()
    except Exception as e:
        import traceback
        print('[RENDER ERROR] mj_forward failed:', e)
        traceback.print_exc()
    # Diagnostic: print seeker/hider pose and visibility before rollout
    try:
        s_pos = env.data.xpos[s_bid][:2].copy()
        mat = np.asarray(env.data.xmat[s_bid]).reshape(3, 3)
        fx, fy = float(mat[0, 0]), float(mat[1, 0])
        yaw_xmat = float(np.arctan2(fy, fx))
    except Exception:
        try:
            rot_jid = env.qpos_indices[s_key]['rot']
            qadr = env.model.jnt_qposadr[rot_jid]
            yaw_xmat = float(env.data.qpos[qadr])
        except Exception:
            yaw_xmat = 0.0
    try:
        rot_jid = env.qpos_indices[s_key]['rot']
        qadr = env.model.jnt_qposadr[rot_jid]
        rot_qpos = float(env.data.qpos[qadr])
    except Exception:
        rot_qpos = yaw_xmat
    rel2 = np.array([hx - s_pos[0], hy - s_pos[1]], dtype=np.float32)
    dist2 = float(np.linalg.norm(rel2))
    frontness2 = float(np.cos(rot_qpos) * (rel2[0] / (dist2 + 1e-8)) + np.sin(rot_qpos) * (rel2[1] / (dist2 + 1e-8)))
    print(f'[DIAG] (rollout-start) seeker_pos={s_pos} yaw_xmat={yaw_xmat:.3f} rot_qpos={rot_qpos:.3f} frontness={frontness2:.3f} hider_pos={[hx,hy]} dist={dist2:.3f}')
    try:
        vis2 = env.vis_engine.is_visible(np.asarray([s_pos[0], s_pos[1], 0.4]), np.asarray([hx, hy, 0.4]), mode=1, body_exclude=env.body_ids.get(s_key, -1), target_body_id=env.body_ids.get(h_key, -1))
        print(f'[DIAG] vis_engine.is_visible (rollout-start) -> {vis2}')
    except Exception as e:
        print('[DIAG] vis check failed (rollout-start):', e)

    collected = []
    steps = 0
    seen = False
    while steps < 1000 and len(collected) < rollout_len:
        norm_obs = env._normalize_obs(env._get_obs(env.learnable_agent_index))
        x = make_seq_input(norm_obs, seq_len)
        a, logp, _, value = agent.get_action_and_value(x)
        act = a.detach().cpu().numpy().ravel()
        full_act = np.zeros(env.action_space.shape[0], dtype=np.float32)
        full_act[: len(act)] = act
        obs, r, term, done, info = env.step(full_act)
        steps += 1
        # per-step diagnostic
        try:
            s_pos = env.data.xpos[s_bid][:2].copy()
            h_pos = env.data.xpos[env.body_ids[h_key]][:2].copy()
            try:
                rot_jid = env.qpos_indices[s_key]['rot']
                qadr = env.model.jnt_qposadr[rot_jid]
                rot_qpos = float(env.data.qpos[qadr])
            except Exception:
                rot_qpos = 0.0
            rels = h_pos - s_pos
            distc = float(np.linalg.norm(rels))
            front = float(np.cos(rot_qpos) * (rels[0] / (distc + 1e-8)) + np.sin(rot_qpos) * (rels[1] / (distc + 1e-8)))
            try:
                vis_now = env.vis_engine.is_visible(np.asarray([s_pos[0], s_pos[1], 0.4]), np.asarray([h_pos[0], h_pos[1], 0.4]), mode=1, body_exclude=env.body_ids.get(s_key, -1), target_body_id=env.body_ids.get(h_key, -1))
            except Exception:
                vis_now = None
            if steps <= 60 or info.get('is_detected', False):
                print(f'[STEP DIAG] step={steps} in_prep={info.get("in_prep")} is_detected={info.get("is_detected")} vis={vis_now} front={front:.3f} dist={distc:.3f}')
        except Exception:
            pass
        # re-teleport hider each step to keep it fixed
        try:
            set_hider_pos_fixed(env, h_key, hx, hy, z=0.5)
            import mujoco
            mujoco.mj_forward(env.model, env.data)
            try:
                env.render()
            except Exception as e:
                import traceback
                print('[RENDER ERROR] teleport render failed:', e)
                traceback.print_exc()
                # attempt to save a debug frame
                try:
                    try:
                        renderer = mujoco.Renderer(env.model, device=0)
                    except TypeError:
                        renderer = mujoco.Renderer(env.model)
                        rendered = False
                        try:
                            # try old Scene-based API
                            if hasattr(mujoco, 'Scene'):
                                scene = mujoco.Scene(env.model, env.data)
                                renderer.render(scene)
                                rgb, depth = renderer.read_pixels()
                                rendered = True
                        except Exception:
                            pass
                        if not rendered:
                            try:
                                # try legacy renderer.render(model, data)
                                renderer.render(env.model, env.data)
                                rgb, depth = renderer.read_pixels()
                                rendered = True
                            except Exception:
                                pass
                        if not rendered:
                            try:
                                # try newer API: update_scene + render()
                                if hasattr(renderer, 'update_scene') and hasattr(renderer, 'render'):
                                    try:
                                        renderer.update_scene(env.data)
                                    except TypeError:
                                        renderer.update_scene(env.data, -1)
                                    out = renderer.render()
                                    if out is not None:
                                        if out.ndim == 3:
                                            rgb = out
                                        else:
                                            rgb = None
                                        rendered = True
                            except Exception:
                                pass
                        if not rendered:
                            raise RuntimeError('No supported render path on this mujoco build')

                        img = Image.fromarray(rgb)
                        img.save(f'experiments/online_update_step_frame_{steps}.png')
                    print(f'Wrote experiments/online_update_step_frame_{steps}.png')
                    try:
                        renderer.close()
                    except Exception:
                        pass
                except Exception as _e:
                    print('[RENDER ERROR] step offscreen save failed:', _e)
                    traceback.print_exc()
        except Exception as e:
            import traceback
            print('[RENDER ERROR] teleport mj_forward failed:', e)
            traceback.print_exc()
        # skip warmup transitions
        if info.get('in_prep', False):
            continue
        if info.get('is_detected', False):
            seen = True
        if not seen:
            continue
        # store (keep autograd graph for policy/value tensors)
        collected.append({'logp': logp, 'value': value, 'reward': torch.tensor([r], dtype=torch.float32)})
        if len(collected) >= rollout_len:
            break

    if not collected:
        print('No rollout collected (hider not seen after warmup). Exiting.')
        return

    # compute returns (simple 1-step discounted returns)
    rewards = [c['reward'] for c in collected]
    values = [c['value'] for c in collected]
    gamma = 0.99
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = float(r.item()) + gamma * G
        returns.insert(0, torch.tensor([G], dtype=torch.float32))

    # prepare tensors (ensure shapes match)
    vals = torch.cat(values).squeeze(-1)
    rets = torch.cat(returns).squeeze(-1)
    logps = torch.cat([c['logp'] for c in collected]).squeeze(-1)
    advs = (rets.detach() - vals.detach())

    # loss
    policy_loss = -(logps * advs).mean()
    value_loss = torch.nn.functional.mse_loss(vals, rets)
    loss = policy_loss + 0.5 * value_loss

    optim.zero_grad()
    loss.backward()
    optim.step()

    # post-update evaluation
    post_score = evaluate_policy(agent, env, seq_len=seq_len, steps=80)
    print('Post-update mean toward_dot:', post_score)

    out = {'pre_mean_toward_dot': pre_score, 'post_mean_toward_dot': post_score, 'num_rollout': len(collected)}
    with open('experiments/online_update_untrained.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote experiments/online_update_untrained.json')


if __name__ == '__main__':
    main()
