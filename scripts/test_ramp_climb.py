"""Simple ramp-climb test.
Place the ramp at origin, orient agent facing uphill, apply forward control
and report whether agent reaches computed ramp top.
"""

import datetime
import math
import os
import time

import mujoco
import numpy as np

from envs.hns_environment import TeamCosEnv

MAX_STEPS = 2000
FORWARD_ACTION = 1.0
FORWARD_CAP = 0.8  # cap forward action to 80% when forcing inputs
# step range to always log verbose contact + snapshot info (inclusive)
CONTACT_LOG_RANGE = (330, 370)


def main():
    # Enable rendering so Viewer window is available during the test.
    # Keep one hider (environment requires >=1) but move it away after creation.
    env = TeamCosEnv(mode="initial", target="seeker", n_seekers=1, n_hiders=1, n_boxes=0, n_ramps=1, render_mode="human", debug_mode=False)
    contact_log = None
    try:
        # prepare contact log file to capture verbose per-contact data (avoids flooding console)
        logs_dir = os.path.join(os.path.dirname(__file__), "../logs")
        os.makedirs(logs_dir, exist_ok=True)
        contact_log_path = os.path.join(logs_dir, f"ramp_climb_contacts_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.log")
        contact_log = open(contact_log_path, "a", buffering=1)
        contact_log.write(f"# ramp_climb contact log started {datetime.datetime.utcnow().isoformat()}\n")
    except Exception:
        contact_log = None
    # If any hider exists, move the first hider far away to avoid overlap with ramp
    try:
        if getattr(env, "hider_keys", None):
            hk = env.hider_keys[0]
            jx = env.qpos_indices[hk]["x"]
            jy = env.qpos_indices[hk]["y"]
            jz = env.qpos_indices[hk]["z"]
            env.data.qpos[int(env.model.jnt_qposadr[jx])] = 10.0
            env.data.qpos[int(env.model.jnt_qposadr[jy])] = 10.0
            env.data.qpos[int(env.model.jnt_qposadr[jz])] = 0.5
            mujoco.mj_forward(env.model, env.data)
            print(f"Moved hider {hk} to (10.0,10.0)")
    except Exception:
        pass
    try:
        rid = env.ramp_ids[0]
        rkey = "ramp1"

        # Place ramp at origin with the pitch used in XML (-36.87 deg)
        qadr, _ = env._obj_addr(rkey)
        pitch = math.radians(-36.87)
        qw = math.cos(pitch / 2.0)
        qx = 0.0
        qy = math.sin(pitch / 2.0)
        qz = 0.0
        # ramp now uses slide x/y + hinge z (qpos=[x,y,rot]), set position and leave rotation at 0
        env.data.qpos[qadr : qadr + 3] = [0.0, 0.0, 0.0]
        mujoco.mj_forward(env.model, env.data)

        # Lock the ramp from the start so it cannot translate while agent climbs
        try:
            env.object_state[rkey]["mode"] = "locked"
            qadr_lock, _ = env._obj_addr(rkey)
            env.object_state[rkey]["locked_pose"] = env.data.qpos[qadr_lock : qadr_lock + 3].copy()
            print(f"Locked ramp {rkey} at qpos={env.object_state[rkey]['locked_pose']}")
        except Exception:
            pass

        # uphill direction
        up = env._ramp_uphill_dir(rid)
        side = np.array([-up[1], up[0]], dtype=np.float32)
        rpos = env.data.xpos[rid][:2].copy()

        # compute lx bounds and choose a start a bit away from the ramp approach
        lx_min, lx_max, lyt = env._compute_ramp_lx_bounds(rid, up)
        # start slightly behind the approach so BOOST isn't immediately engaged
        start_lx = lx_min - 0.40
        apos2 = rpos + up * start_lx

        # place the learnable agent at this start position, facing uphill
        ak = env.learnable_agent_key
        jx = env.qpos_indices[ak]["x"]
        jy = env.qpos_indices[ak]["y"]
        jz = env.qpos_indices[ak]["z"]
        jr = env.qpos_indices[ak]["rot"]
        adr_x = int(env.model.jnt_qposadr[jx])
        adr_y = int(env.model.jnt_qposadr[jy])
        adr_z = int(env.model.jnt_qposadr[jz])
        adr_r = int(env.model.jnt_qposadr[jr])

        env.data.qpos[adr_x] = float(apos2[0])
        env.data.qpos[adr_y] = float(apos2[1])
        env.data.qpos[adr_z] = 0.5
        agent_yaw = math.atan2(up[1], up[0])
        env.data.qpos[adr_r] = float(agent_yaw)

        mujoco.mj_forward(env.model, env.data)

        top_z = env._compute_ramp_top_z(rid)
        # Detailed ramp/top diagnostics
        geom_ids = env.obj_geom_ids.get(rkey, [])
        geom_sizes = []
        for gid in geom_ids:
            try:
                geom_sizes.append(tuple(env.model.geom_size[gid]))
            except Exception:
                geom_sizes.append(None)
        # compute pitch and vertical half explicitly (duplicate of helper)
        try:
            q = env.data.xquat[rid]
            sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
            if sinp >= 1.0:
                pitch = math.pi / 2
            elif sinp <= -1.0:
                pitch = -math.pi / 2
            else:
                pitch = math.asin(sinp)
        except Exception:
            pitch = None
        geom_info = f"geom_sizes={geom_sizes} pitch={pitch}"
        print(f"Ramp placed at {rpos}, uphill={up}, lx_bounds=({lx_min:.3f},{lx_max:.3f}), start_lx={start_lx:.3f}, top_z={top_z:.3f} | {geom_info}")

        # record starting agent body z to use floor-relative success criterion
        agent_bid = env.body_ids[ak]
        start_agent_z = float(env.data.xpos[agent_bid][2])
        reached = False
        # ensure we don't assume boost is on at t=0
        prev_boost = 0.0
        for step in range(1, MAX_STEPS + 1):
            # Force orientation lock and capped forward thrust for this experiment
            try:
                # enforce agent facing uphill each step (keep qpos rot fixed)
                env.data.qpos[adr_r] = float(agent_yaw)
                mujoco.mj_forward(env.model, env.data)
            except Exception:
                pass

            fwd = min(FORWARD_ACTION, FORWARD_CAP)
            action = np.array([fwd, 0.0, 0.0, 0.0], dtype=np.float32)
            out = env.step(action)
            # env.step returns (obs, reward, False, done, info)
            try:
                obs, reward, _, done, info = out
            except Exception:
                # older API fallback
                obs, reward, done, info = out
            agent_z = float(env.data.xpos[agent_bid][2])

            # diagnostics: compute planar rel pos and ramp-local coords
            up = env._ramp_uphill_dir(rid)
            side = np.array([-up[1], up[0]], dtype=np.float32)
            rel = np.array(env.data.xpos[agent_bid][:2] - rpos)
            lx = float(np.dot(rel, up))
            ly = float(np.dot(rel, side))
            try:
                lx_min, lx_max, ly_thresh = env._compute_ramp_lx_bounds(rid, up)
            except Exception:
                lx_min, lx_max, ly_thresh = -1.15, 0.666, 0.95
            boost = float(info.get("dbg_ramp_boost", -1.0)) if isinstance(info, dict) else -1.0
            prog = info.get("dbg_ramp_progress") if isinstance(info, dict) else None
            # compute agent yaw and ramp uphill angle difference
            try:
                agent_yaw = float(env.data.qpos[adr_r])
            except Exception:
                agent_yaw = float(env.data.qpos[env.model.jnt_qposadr[env.qpos_indices[ak]["rot"]]])
            up_angle = math.atan2(up[1], up[0])
            # signed difference in radians in [-pi, pi]
            diff = (agent_yaw - up_angle + math.pi) % (2 * math.pi) - math.pi
            diff_deg = math.degrees(diff)
            facing_cos = math.cos(agent_yaw) * up[0] + math.sin(agent_yaw) * up[1]

            if step % 20 == 0 or step == 1 or (prev_boost is not None and boost < 0.5 and prev_boost >= 0.5):
                print(f"step={step} z={agent_z:.3f} boost={boost:.2f} prog={prog} lx={lx:.3f} " f"lx_min={lx_min:.3f} lx_max={lx_max:.3f} ly={ly:.3f} ly_thresh={ly_thresh:.3f}")
                print(f"  ORIENT: agent_yaw={agent_yaw:.3f} up_angle={up_angle:.3f} diff_deg={diff_deg:.1f} facing_cos={facing_cos:.3f}")
                # Additional diagnostics when boost just dropped
                if prev_boost is not None and boost < 0.5 and prev_boost >= 0.5:
                    try:
                        # agent planar velocity in world frame
                        aid = agent_bid
                        vadr = env.model.jnt_dofadr[env.model.joint_name2id(f"{ak}_x")]
                    except Exception:
                        vadr = env.model.jnt_dofadr[env.qpos_indices[ak]["x"]] if False else None
                    try:
                        # fallback: compute body linear velocity approximate from qvel entries
                        bv = env.data.cvel[agent_bid] if hasattr(env.data, "cvel") else None
                    except Exception:
                        bv = None
                    try:
                        # body spatial velocity from data.xvel if available
                        world_vel = env.data.xvel[agent_bid] if hasattr(env.data, "xvel") else None
                    except Exception:
                        world_vel = None
                    try:
                        # external contact force on body (6-vector)
                        cfrc = env.data.cfrc_ext[agent_bid] if hasattr(env.data, "cfrc_ext") else None
                    except Exception:
                        cfrc = None
                    try:
                        ncon = int(env.data.ncon)
                    except Exception:
                        ncon = 0
                    print(f"  DIAG: ncon={ncon} world_vel={world_vel} cfrc_ext={cfrc}")
                    # write detailed contact pairs to file (if available) to avoid flooding editor
                    if contact_log is not None:
                        try:
                            for ci in range(min(ncon, 200)):
                                try:
                                    c = env.data.contact[ci]
                                    g1 = int(c.geom1)
                                    g2 = int(c.geom2)
                                    pos = tuple(c.pos) if hasattr(c, "pos") else None
                                    frame = tuple(c.frame[:3]) if hasattr(c, "frame") and len(c.frame) >= 3 else None
                                    try:
                                        b1 = int(env.model.geom_bodyid[g1]) if g1 >= 0 else None
                                    except Exception:
                                        b1 = None
                                    try:
                                        b2 = int(env.model.geom_bodyid[g2]) if g2 >= 0 else None
                                    except Exception:
                                        b2 = None
                                    try:
                                        f1 = tuple(env.data.cfrc_ext[b1][:3]) if (b1 is not None and b1 >= 0) else None
                                    except Exception:
                                        f1 = None
                                    try:
                                        f2 = tuple(env.data.cfrc_ext[b2][:3]) if (b2 is not None and b2 >= 0) else None
                                    except Exception:
                                        f2 = None
                                    contact_log.write(f"{datetime.datetime.utcnow().isoformat()} STEP={step} contact[{ci}] g1={g1} g2={g2} b1={b1} b2={b2} pos={pos} normal={frame} f1={f1} f2={f2}\n")
                                except Exception:
                                    pass
                        except Exception:
                            pass
            # unconditional snapshot & contact dump for configured step window
            if contact_log is not None and CONTACT_LOG_RANGE[0] <= step <= CONTACT_LOG_RANGE[1]:
                try:
                    # snapshot positions and object state
                    agent_xy = tuple(env.data.xpos[agent_bid][:2])
                    ramp_xy = tuple(env.data.xpos[rid][:2])
                    contact_log.write(
                        f"{datetime.datetime.utcnow().isoformat()} STEP={step} SNAPSHOT agent_xy={agent_xy} ramp_xy={ramp_xy} lx={lx:.3f} ly={ly:.3f} up={tuple(up)} object_state={env.object_state.get(rkey)}\n"
                    )
                except Exception:
                    pass
                try:
                    ncon2 = int(env.data.ncon)
                except Exception:
                    ncon2 = 0
                try:
                    for ci in range(min(ncon2, 400)):
                        try:
                            c = env.data.contact[ci]
                            g1 = int(c.geom1)
                            g2 = int(c.geom2)
                            pos = tuple(c.pos) if hasattr(c, "pos") else None
                            frame = tuple(c.frame[:3]) if hasattr(c, "frame") and len(c.frame) >= 3 else None
                            try:
                                b1 = int(env.model.geom_bodyid[g1]) if g1 >= 0 else None
                            except Exception:
                                b1 = None
                            try:
                                b2 = int(env.model.geom_bodyid[g2]) if g2 >= 0 else None
                            except Exception:
                                b2 = None
                            try:
                                f1 = tuple(env.data.cfrc_ext[b1][:3]) if (b1 is not None and b1 >= 0) else None
                            except Exception:
                                f1 = None
                            try:
                                f2 = tuple(env.data.cfrc_ext[b2][:3]) if (b2 is not None and b2 >= 0) else None
                            except Exception:
                                f2 = None
                            contact_log.write(f"{datetime.datetime.utcnow().isoformat()} STEP={step} RANGE_CONTACT[{ci}] g1={g1} g2={g2} b1={b1} b2={b2} pos={pos} normal={frame} f1={f1} f2={f2}\n")
                        except Exception:
                            pass
                except Exception:
                    pass
            prev_boost = boost

            # render to the viewer (if available) so we can observe behavior live
            try:
                env.render()
                time.sleep(0.02)
            except Exception:
                pass

            # success criterion: agent_z increased by >= 1.0m from start (floor-relative)
            if (agent_z - start_agent_z) >= 1.0:
                print(f"SUCCESS: climbed >=1.0m from start at step={step} agent_z={agent_z:.3f} start_z={start_agent_z:.3f}")
                reached = True
                break

            # failure: agent fell below ground significantly
            if agent_z < -0.1:
                print(f"FAIL: agent fell below ground at step={step} z={agent_z:.3f}")
                break

        if not reached:
            print(f"Did not reach top within {MAX_STEPS} steps. final_z={agent_z:.3f}")

    finally:
        try:
            if contact_log is not None:
                contact_log.write(f"# ramp_climb contact log ended {datetime.datetime.utcnow().isoformat()}\n")
                contact_log.close()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    main()
