import re
import numpy as np
import mujoco


def _edit_xml_place_far(xml, box_positions=None, ramp_positions=None, agent_pos_map=None):
    lines = xml.splitlines()
    new_lines = []
    for L in lines:
        outL = L
        # skip walls entirely
        if 'geom name="wall_' in L or 'geom name="maze_w' in L:
            continue
        # box bodies
        if 'body name="box' in L and 'pos="' in L and box_positions is not None:
            m = re.search(r'box(\d+)_body', L)
            if m:
                i = int(m.group(1)) - 1
                if i < len(box_positions):
                    x, y = box_positions[i]
                    outL = re.sub(r'pos="[^"]+"', f'pos="{x} {y} 0.5"', L)
        # ramp bodies
        if 'body name="ramp' in L and 'pos="' in L and ramp_positions is not None:
            m = re.search(r'ramp(\d+)_body', L)
            if m:
                i = int(m.group(1)) - 1
                if i < len(ramp_positions):
                    x, y = ramp_positions[i]
                    outL = re.sub(r'pos="[^"]+"', f'pos="{x} {y} 0"', L)
        # agent anchors
        if '_anchor' in L and 'pos="' in L and agent_pos_map is not None:
            m = re.search(r'body name="([a-zA-Z0-9_]+)_anchor"', L)
            if m:
                name = m.group(1)
                if name in agent_pos_map:
                    x, y = agent_pos_map[name]
                    outL = re.sub(r'pos="[^"]+"', f'pos="{x} {y} 0.5"', L)
        new_lines.append(outL)
    return '\n'.join(new_lines)


def prepare_env(env, action_repeat=None, place_far=True, far_offset=20.0):
    """Prepare env for clean experiments.
    - set env.action_repeat if provided
    - optionally move boxes, ramps and non-learnable agents far away
    - rebuild model/data and re-run structure analysis
    """
    if action_repeat is not None:
        env.action_repeat = int(action_repeat)

    if not place_far:
        return env

    # construct far positions
    box_positions = [(far_offset + 2.0 * i, far_offset) for i in range(env.n_boxes)] if env.n_boxes > 0 else None
    ramp_positions = [(far_offset, far_offset + 4.0 + 2.0 * i) for i in range(env.n_ramps)] if env.n_ramps > 0 else None
    other_agents = [k for k in env.agent_keys if k != env.learnable_agent_key]
    agent_positions = [(far_offset + 10.0, far_offset + 2.0 * i) for i in range(len(other_agents))] if other_agents else None
    agent_pos_map = dict(zip(other_agents, agent_positions)) if agent_positions is not None else None

    # get xml, edit, rebuild
    xml = env._build_dynamic_xml()
    new_xml = _edit_xml_place_far(xml, box_positions=box_positions, ramp_positions=ramp_positions, agent_pos_map=agent_pos_map)
    env.model = mujoco.MjModel.from_xml_string(new_xml)
    env.data = mujoco.MjData(env.model)

    # reinit structure caches and interaction state
    try:
        env._analyze_structure()
    except Exception:
        pass
    try:
        env._init_interaction_state()
    except Exception:
        pass

    # Write qpos for boxes, ramps and other agents so positions take effect
    try:
        # boxes
        if box_positions is not None and getattr(env, 'box_ids', None):
            for i, pos in enumerate(box_positions):
                if i < len(env.box_ids):
                    bid = env.box_ids[i]
                    adr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
                    x, y = pos
                    env.data.qpos[adr:adr+7] = [float(x), float(y), 0.5, 1.0, 0.0, 0.0, 0.0]
        # ramps
        if ramp_positions is not None and getattr(env, 'ramp_ids', None):
            for i, pos in enumerate(ramp_positions):
                if i < len(env.ramp_ids):
                    rid = env.ramp_ids[i]
                    adr = env.model.jnt_qposadr[env.model.body_jntadr[rid]]
                    x, y = pos
                    env.data.qpos[adr:adr+7] = [float(x), float(y), 0.0, 1.0, 0.0, 0.0, 0.0]
        # other agents anchors: ensure every non-learnable agent is placed far
        qmap = getattr(env, 'qpos_indices', {})
        try:
            other_agents = [k for k in env.agent_keys if k != env.learnable_agent_key]
            for i, name in enumerate(other_agents):
                x = far_offset + 10.0 + 4.0 * i
                y = far_offset + 2.0 * i
                # try body-based qpos write first
                try:
                    bid = env.model.body(f"{name}_anchor").id
                    adr = env.model.jnt_qposadr[env.model.body_jntadr[bid]]
                    env.data.qpos[adr:adr+7] = [float(x), float(y), 0.5, 1.0, 0.0, 0.0, 0.0]
                except Exception:
                    # fallback to joint qpos indices
                    if name in qmap:
                        jx = qmap[name]['x']
                        jy = qmap[name]['y']
                        jz = qmap[name]['z']
                        jr = qmap[name]['rot']
                        env.data.qpos[env.model.jnt_qposadr[jx]] = float(x)
                        env.data.qpos[env.model.jnt_qposadr[jy]] = float(y)
                        env.data.qpos[env.model.jnt_qposadr[jz]] = 0.5
                        env.data.qpos[env.model.jnt_qposadr[jr]] = 0.0
        except Exception:
            pass
    except Exception:
        pass

    # forward model to apply qpos -> xpos/xquat
    try:
        mujoco.mj_forward(env.model, env.data)
    except Exception:
        pass

    # try to reinit visibility engine and mark ready
    try:
        env.vis_engine = type(env.vis_engine)(env.model, env.data, mode4_sdf_cell_size=env.mode4_sdf_cell_size)
        env.vis_engine._mode4_ready = True
        md = getattr(env.vis_engine, 'max_dist', 10.0)
        env.vis_engine._mode4_sdf_grid = np.full((2, 2), float(md), dtype=np.float32)
        env.vis_engine._mode4_min_x = -float(md)
        env.vis_engine._mode4_min_y = -float(md)
        env.vis_engine._mode4_cell_size = float(env.mode4_sdf_cell_size)
    except Exception:
        pass

    return env
