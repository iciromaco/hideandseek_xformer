# verify_seeker_rule.py v2.91
# 演習第25回：ハイダー純粋ランダム化 ＆ デッドロック解消（物理スタック自律復帰型）
#
# 修正履歴:
# v2.90: 反発ロジック。
# v2.91: ユーザー指摘（ハイダーに知恵は不要）を反映。
#        1. SimpleWanderer から対シーカー逃避ロジックを完全削除。純粋にランダムに彷徨う挙動へ。
#        2. スタック検知（stuck_counter）により、物理的に詰まった時だけ強制的に行動を再抽選する「物理的本能」のみ維持。
#        3. FPS固定・視線描画・視野境界表示はそのまま継続。

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from agents.scripted_agents import RuleBasedSeeker

# プロジェクトルートからのインポート
from envs.hns_environment import TeamCosEnv

# --- 監査・表示設定 ---
ACTION_REPEAT = 2
FOV_DEG = 135
FOV_HALF_RAD = FOV_DEG * 0.5 * math.pi / 180.0
FOV_COS_HALF = math.cos(FOV_HALF_RAD)
TARGET_FPS = 15.0
FRAME_DURATION = 1.0 / TARGET_FPS


class SimpleWanderer:
    """ハイダー：知恵を持たない、純粋なランダム放浪者"""

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.timer = 0
        self.mode = "WANDER"
        self.action = np.array([0.15, 0.0])
        self.last_pos = np.zeros(2)
        self.stuck_counter = 0
        np.random.seed(42 + agent_id)

    def get_action(self, obs, current_pos):
        """
        シーカーの位置を見ず、自身の周囲の空き具合と乱数だけで動く。
        """
        lidar = obs[5:17]
        front_dist = min(lidar[0:3])

        # 1. スタック検知（何かに詰まって動けていないか）
        dist_moved = np.linalg.norm(current_pos - self.last_pos)
        self.last_pos = current_pos.copy()
        if dist_moved < 0.005:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # 2. 壁または障害物への緊急回避 (物理的本能)
        if self.mode == "BRAKE":
            self.timer -= 1
            if self.timer <= 0:
                self.mode = "ESCAPE"
                self.timer = 30
                # 空いている方向を適当に選んで旋回
                best_idx = np.argmax(lidar)
                angles_deg = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
                self.action = np.array([-0.05, np.clip(np.deg2rad(angles_deg[best_idx]), -0.5, 0.5)])
            return np.array([0.0, 0.0])

        if self.mode == "ESCAPE":
            self.timer -= 1
            if self.timer <= 0 or front_dist > 0.8:
                self.mode = "WANDER"
            return self.action

        # 壁が近すぎる、またはスタックしているなら回避フェーズへ
        if front_dist < 0.22 or self.stuck_counter > 40:
            self.mode = "BRAKE"
            self.timer = 8
            return np.array([0.0, 0.0])

        # 3. 通常のランダム巡回
        self.timer -= 1
        if self.timer <= 0:
            self.timer = np.random.randint(80, 150)
            # 常に一定の前進速度を持ち、旋回角だけランダムに決める
            fwd = 0.18
            turn = np.random.uniform(-0.4, 0.4)
            self.action = np.array([fwd, turn])
        return self.action


def can_see_target_visual(engine, viewer_pos, viewer_heading, target_pos, body_exclude, target_body_id):
    diff = target_pos - viewer_pos
    dist = np.linalg.norm(diff)
    if dist > 15.0:
        return False, dist, "RANGE"
    view_dir = np.array([math.cos(viewer_heading), math.sin(viewer_heading)])
    rel_dir = diff / (dist + 1e-8)
    if np.dot(view_dir, rel_dir) < FOV_COS_HALF:
        return False, dist, "FOV"
    if not engine.is_visible(viewer_pos, target_pos, body_exclude=body_exclude, target_body_id=target_body_id):
        return False, dist, "WALL"
    return True, dist, "VISIBLE"


def run_live_monitor():
    print("=" * 120)
    print("🚀 Seeker Auditor v2.91 (Pure Random Hiders Mode)")
    print("=" * 120)
    env = TeamCosEnv()
    seeker_agent = RuleBasedSeeker()
    hider_agents = [SimpleWanderer(i) for i in range(len(env.hiders))]
    m, d = env.model, env.data
    engine = env.vis_engine
    s_key = env.seekers[0]
    s_id = env.body_ids[s_key]
    s_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "seeker_btm")
    h_body_ids = [env.body_ids[hk] for hk in env.hiders]
    h_geom_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"hider{i+1}_btm") for i in range(len(env.hiders))]

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance = 18.0
        step_count = 0
        try:
            while viewer.is_running():
                frame_start = time.perf_counter()
                if step_count % 5000 == 0:
                    env.reset()

                s_pos_world = d.geom_xpos[s_geom_id][:2].copy()
                s_heading = d.qpos[m.jnt_qposadr[env.qpos_indices[s_key]["rot"]]]
                h_positions_2d = [d.geom_xpos[gid][:2].copy() for gid in h_geom_ids]

                # Seeker: ターゲットの ID リストを渡し、視認判定
                obs_s = env._get_obs(0)
                s_action = seeker_agent.get_action(
                    obs_s,
                    engine,
                    s_pos_world,
                    s_heading,
                    h_positions_2d,
                    s_id,
                    h_body_ids,
                )

                # Hiders: シーカーの位置を見せず、自身の座標だけ渡してランダムに動く
                full_action = np.zeros(env.num_agents * 2)
                full_action[0:2] = s_action
                for i in range(len(env.hiders)):
                    full_action[(i + 1) * 2 : (i + 1) * 2 + 2] = hider_agents[i].get_action(env._get_obs(i + 1), h_positions_2d[i])

                for _ in range(ACTION_REPEAT):
                    env.step(full_action)
                    step_count += 1

                if hasattr(viewer, "user_scn"):
                    ctx = viewer.user_scn
                    ctx.ngeom = 0
                    mujoco.mjv_initGeom(
                        ctx.geoms[0],
                        type=mujoco.mjtGeom.mjGEOM_LABEL,
                        size=np.zeros(3),
                        pos=d.geom_xpos[s_geom_id] + np.array([0, 0, 1.2]),
                        mat=np.eye(3).flatten(),
                        rgba=[1, 1, 1, 1],
                    )
                    ctx.geoms[0].label = f"S:{seeker_agent.status}"
                    ctx.ngeom = 1
                    s_center = d.geom_xpos[s_geom_id].copy()
                    s_center[2] = 0.4
                    for side in [-1, 1]:
                        limit_ang = s_heading + (side * FOV_HALF_RAD)
                        end_pt = s_center + np.array([math.cos(limit_ang), math.sin(limit_ang), 0]) * 5.0
                        mujoco.mjv_connector(
                            ctx.geoms[ctx.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_LINE,
                            width=2.0,
                            from_=s_center,
                            to=end_pt,
                        )
                        ctx.geoms[ctx.ngeom].rgba = np.array([1, 1, 1, 0.6])
                        ctx.ngeom += 1
                    for i, h_bid in enumerate(h_body_ids):
                        t_pos = d.geom_xpos[h_geom_ids[i]].copy()
                        t_pos[2] = 0.4
                        vis, _, _ = can_see_target_visual(engine, s_pos_world, s_heading, t_pos[:2], s_id, h_bid)
                        if vis:
                            mujoco.mjv_connector(
                                ctx.geoms[ctx.ngeom],
                                type=mujoco.mjtGeom.mjGEOM_LINE,
                                width=4.0,
                                from_=s_center,
                                to=t_pos,
                            )
                            ctx.geoms[ctx.ngeom].rgba = np.array([1, 0, 0, 1] if i == 0 else [0.2, 0.8, 1, 1])
                            ctx.ngeom += 1

                viewer.sync()
                elapsed = time.perf_counter() - frame_start
                if elapsed < FRAME_DURATION:
                    time.sleep(FRAME_DURATION - elapsed)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    run_live_monitor()
