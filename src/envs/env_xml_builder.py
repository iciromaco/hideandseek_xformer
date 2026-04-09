import math


class EnvXMLBuilder:
    """Generate Mujoco XML for arena, objects and agents using an environment's constants.

    The builder expects to be constructed with the environment instance and will
    reference constants and helper functions (like `_euler_z_to_quat`) on it.
    """

    def __init__(self, env):
        self.env = env

    def _euler_z_to_quat(self, yaw):
        w, z = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        return f"{w:.6f} 0 0 {z:.6f}"

    def build_dynamic_xml(self):
        e = self.env
        arena = self._xml_static_scene()
        ramps = "".join(self._xml_ramp(i, [0, 0], 0) for i in range(1, e.n_ramps + 1))
        boxes = "".join(self._xml_box(i, [0, 0], 0) for i in range(1, e.n_boxes + 1))
        # Match legacy coloring: seekers red, hiders blue variants
        seekers = "".join(
            self._xml_agent(
                ak,
                [0, 0],
                0,
                (0.9, 0.1, 0.1),
            )
            for ak in e.seeker_keys
        )
        hiders = "".join(
            self._xml_agent(
                ak,
                [0, 0],
                0,
                (0.1, 0.1, 0.9) if (i % 2 == 0) else (0.1, 0.6, 0.9),
            )
            for i, ak in enumerate(e.hider_keys)
        )
        acts = "".join(self._xml_actuators(ak) for ak in e.agent_keys)

        # Build equality (weld) entries for potential grasps/locks. These are
        # created inactive by default and activated at runtime when needed.
        equality = self._xml_equality()

        return f"""
    <mujoco>
  <option gravity="0 0 -9.81" timestep="0.005"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="1 1" reflectance="0.2"/>
    <mesh name="ramp_mesh" 
          vertex="-0.6666 -0.5 0.0 0.6666 -0.5 0.0 0.6666 -0.5 1.0 -0.6666 0.5 0.0 0.6666 0.5 0.0 0.6666 0.5 1.0"
          face="0 1 2 3 5 4 0 3 4 0 4 1 1 4 5 1 5 2 2 5 3 2 3 0"/>
  </asset>
  <worldbody>
    <light pos="0 0 12" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="overview" pos="0 13 13" euler="2.35 0 -3.14" mode="fixed" />
        {arena} {ramps} {boxes} {seekers} {hiders}
  </worldbody>
    <actuator>{acts}</actuator>
    {equality}
</mujoco>"""

    def _xml_static_scene(self):
        e = self.env
        s = e.ARENA_HALF
        attr = 'friction="0.05 0.05 0.05" solref="0.01 1" solimp="0.95 0.99 0.001"'
        return f"""
    <geom name="floor" type="plane" size="{s} {s} 0.1" material="grid" friction="1.1 0.15 0.003"/>
    <geom name="wall_n" type="box" size="{s+0.15} 0.1 2.0" pos="0 6.1 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_s" type="box" size="{s+0.15} 0.1 2.0" pos="0 -6.1 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_e" type="box" size="0.1 {s} 2.0" pos="6.1 0 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="wall_w" type="box" size="0.1 {s} 2.0" pos="-6.1 0 2.0" rgba="0.65 0.65 0.65 0.35" {attr}/>
    <geom name="maze_w0" type="box" size="1.5 0.2 0.5" pos="3.0 1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w1" type="box" size="1.5 0.2 0.5" pos="-3.0 -1.5 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w2" type="box" size="0.2 1.5 0.5" pos="0.0 -3.0 0.5" rgba="0 0.7 0.7 1" {attr}/>
    <geom name="maze_w3" type="box" size="0.2 1.5 0.5" pos="0.0 3.0 0.5" rgba="0 0.7 0.7 1" {attr}/>
"""

    def _xml_ramp(self, i, xy, rot):
        e = self.env
        q = self._euler_z_to_quat(rot)
        return f"""
        <body name="ramp{i}_body" pos="{xy[0]} {xy[1]} 0" quat="{q}">
            <inertial pos="0.3 0 0.25" mass="{e.RAMP_MASS}" diaginertia="10 10 20"/>
            <joint type="free" name="ramp{i}_joint" damping="{e.RAMP_JOINT_DAMPING}"/>
            <geom name="ramp{i}_geom" type="mesh" mesh="ramp_mesh" contype="0" conaffinity="0" rgba="0 1 0 1"/>
            <!-- geom name="ramp{i}_slope_surface" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.3" friction="1.35 0.22 0.01"/ -->
            <geom name="ramp{i}_slope_surface" type="box" size="0.8333 0.5 0.02" pos="0 0 0.516" euler="0 -36.87 0" rgba="0 1 0 0.3" friction="1.35 0.22 0.00001"/>
            <geom name="ramp{i}_back_panel" type="box" size="0.02 0.5 0.5" pos="0.6666 0 0.5" rgba="0 1 0 0.3" friction="1.35 0.22 0.01"/>
            <geom name="ramp{i}_inner_weight" type="box" size="0.3333 0.5 0.25" pos="0.3333 0 0.25" rgba="0 1 0 0.3" mass="{e.RAMP_INNER_WEIGHT_MASS}" solimp="0.95 0.99 0.001" friction="1.35 0.22 0.01"/>
    </body>"""

    def _xml_box(self, i, xy, rot):
        e = self.env
        q = self._euler_z_to_quat(rot)
        return f"""
    <body name="box{i}_body" pos="{xy[0]} {xy[1]} 0.5" quat="{q}">
            <joint name="box{i}_joint" type="free" damping="{e.BOX_JOINT_DAMPING}"/>
            <geom name="box{i}_geom" type="box" size="0.6 0.6 0.5" mass="{e.BOX_MASS}" rgba="0.75 0.55 0.3 1" friction="1.2 0.08 0.003"/>
    </body>"""

    def _xml_agent(self, pre, xy, rot, color):
        e = self.env
        q = self._euler_z_to_quat(rot)
        r, g, b = color
        # anchor の初期高さを環境定数 AGENT_Z_MIN に合わせる（Viewer の Reset と一致させる）
        limit_z = f"{e.AGENT_Z_MIN-e.AGENT_Z_POS} {e.AGENT_Z_MAX-e.AGENT_Z_POS}"
        return f"""
        <body name="{pre}_anchor" pos="{xy[0]} {xy[1]} {e.AGENT_Z_POS}" quat="{q}">
            <joint name="{pre}_x" type="slide" axis="1 0 0" damping="{e.AGENT_DAMPING_XY}"/>
            <joint name="{pre}_y" type="slide" axis="0 1 0" damping="{e.AGENT_DAMPING_XY}"/>
            <joint name="{pre}_z" type="slide" axis="0 0 1" damping="{e.AGENT_DAMPING_Z}" limited="true" range="{limit_z}"/> #　微かに浮かせている
            <joint name="{pre}_rot" type="hinge" axis="0 0 1" damping="{e.AGENT_DAMPING_ROT}"/>
            <body name="{pre}_body">
                <site name="{pre}_thrust" pos="0 0 0"/>
                <!-- bottom sphere used for visibility / lidar and agent radius -->
                <geom name="{pre}_btm" type="sphere" pos="0 0 -0.1" size="0.4" mass="{e.AGENT_MASS}" friction="0.05 0.01 0"/>
                <geom name="{pre}_capsule" type="capsule" size="0.3 0.2" pos="0 0 0" rgba="{r} {g} {b} 1" mass="70" friction="0.05 0.01 0"/>
                <!-- nose/tail geoms (visual, non-colliding) -->
                <geom name="{pre}_nose" type="capsule" fromto="0 0 0.2 0.3 0 0.2" size="0.1" rgba="1 1 1 1" contype="0" conaffinity="0"/>
                <geom name="{pre}_tail" type="capsule" fromto="0 0 0 -0.45 0 -0.3" size="0.05" rgba="{r} {g} {b} 1" contype="0" conaffinity="0"/>
            </body>
    </body>"""

    def _xml_actuators(self, pre):
        e = self.env
        return f"""
    <general name="{pre}_fwd" site="{pre}_thrust" gear="1 0 0 0 0 0" gainprm="{e.AGENT_ACTUATOR_FWD}" ctrlrange="-1 1"/>
    <general name="{pre}_turn" joint="{pre}_rot" gear="0.5" gainprm="{e.AGENT_TURN_GAIN}" ctrlrange="-1 1"/>
"""

    def _xml_equality(self):
        e = self.env
        # solref/solimp tuned for grasp (softer contact)
        grasp_solref = "0.08 1"
        grasp_solimp = "0.9 0.95 0.001"
        lock_solref = "0.02 1"
        lock_solimp = "0.95 0.99 0.001"
        parts = ["  <equality>"]
        # grasps: for each agent and each object (boxes and ramps)
        for ak in e.agent_keys:
            for i in range(1, e.n_boxes + 1):
                parts.append(f'    <weld name="eq_grasp_{ak}_box{i}" body1="{ak}_body" body2="box{i}_body" active="false" relpose="0 0 0 1 0 0 0" solref="{grasp_solref}" solimp="{grasp_solimp}"/>')
            for i in range(1, e.n_ramps + 1):
                parts.append(f'    <weld name="eq_grasp_{ak}_ramp{i}" body1="{ak}_body" body2="ramp{i}_body" active="false" relpose="0 0 0 1 0 0 0" solref="{grasp_solref}" solimp="{grasp_solimp}"/>')
        # locks: world to object
        for i in range(1, e.n_boxes + 1):
            parts.append(f'    <weld name="eq_lock_box{i}" body1="world" body2="box{i}_body" active="false" solref="{lock_solref}" solimp="{lock_solimp}"/>')
        for i in range(1, e.n_ramps + 1):
            parts.append(f'    <weld name="eq_lock_ramp{i}" body1="world" body2="ramp{i}_body" active="false" solref="{lock_solref}" solimp="{lock_solimp}"/>')
        parts.append("  </equality>")
        return "\n".join(parts)

