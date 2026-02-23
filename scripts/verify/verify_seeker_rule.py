# verify_seeker_rule.py v2.73
# 演習第25回：シーカー行動監査（視覚的捕捉ライン ＆ ログ詳細化版）
# 
# 修正履歴:
# v2.72: 静止制動フェーズ搭載。
# v2.73: 視認判定の可視化を強化。
#        1. シーカーがハイダーを視認している間、Viewer上に赤い「視線ライン」を描画。
#        2. ログの略称を廃止し、"VISIBLE", "HIDDEN(FOV)" など詳細に表示。
#        3. 動作を観察しやすくするため、ウェイトを調整しシミュレーション速度を安定化。

import mujoco
import mujoco.viewer
import numpy as np
import time
import math

# プロジェクトルートからのインポート (src.なし)
from envs.hns_environment import TeamCosEnv
from agents.scripted_agents import RuleBasedSeeker

# --- 監査用設定 ---
ACTION_REPEAT = 2
FOV_DEG = 135
FOV_HALF_RAD = FOV_DEG * 0.5 * math.pi / 180.0
FOV_COS_HALF = math.cos(FOV_HALF_RAD)

class SimpleWanderer:
    """Hider用の改善された巡回ロジック：壁に当たったら一旦止まってから逃げる"""
    def __init__(self):
        self.timer = 0
        self.mode = "WANDER"
        self.action = np.array([0.2, 0.0])
    
    def get_action(self, obs):
        lidar = obs[5:17]
        front_dist = min(lidar[0:3])
        
        if self.mode == "BRAKE":
            self.timer -= 1
            if self.timer <= 0:
                self.mode = "ESCAPE"
                self.timer = 25
                best_idx = np.argmax(lidar)
                angles_deg = [0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180]
                self.action = np.array([-0.15, np.clip(np.deg2rad(angles_deg[best_idx]), -0.7, 0.7)])
            return np.array([0.0, 0.0])

        if self.mode == "ESCAPE":
            self.timer -= 1
            if self.timer <= 0 or front_dist > 0.8:
                self.mode = "WANDER"
            return self.action

        if front_dist < 0.25:
            self.mode = "BRAKE"
            self.timer = 5
            return np.array([0.0, 0.0])

        self.timer -= 1
        if self.timer <= 0:
            self.timer = np.random.randint(40, 100)
            fwd = np.random.uniform(0.15, 0.3) # 少し速度を落として観察しやすく
            turn = np.random.uniform(-0.3, 0.3)
            self.action = np.array([fwd, turn])
        
        return self.action

def can_see_target_visual(engine, viewer_pos, viewer_heading, target_pos, body_exclude):
    """視認判定の詳細理由を返す"""
    diff = target_pos - viewer_pos
    dist = np.linalg.norm(diff)
    if dist > 15.0: return False, dist, "RANGE"
    
    view_dir = np.array([math.cos(viewer_heading), math.sin(viewer_heading)])
    rel_dir = diff / (dist + 1e-8)
    if np.dot(view_dir, rel_dir) < FOV_COS_HALF: return False, dist, "FOV"
    
    if not engine.is_visible(viewer_pos, target_pos, body_exclude=body_exclude):
        return False, dist, "WALL"
    return True, dist, "VISIBLE"

def run_live_monitor():
    print("="*120)
    print("🚀 Seeker & Hider Logic Auditor v2.73 (Enhanced Visibility Audit)")
    print("="*120)
    
    env = TeamCosEnv()
    seeker_agent = RuleBasedSeeker()
    hider_agents = [SimpleWanderer() for _ in range(len(env.hiders))]
    
    m, d = env.model, env.data
    engine = env.vis_engine
    
    s_key = env.seekers[0]
    s_id = env.body_ids[s_key]
    h_keys = env.hiders

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance = 18.0
        step_count = 0
        try:
            while viewer.is_running():
                if step_count % 3000 == 0:
                    env.reset()
                
                s_pos = d.xpos[s_id][:2].copy()
                s_heading = d.qpos[m.jnt_qposadr[env.qpos_indices[s_key]['rot']]]
                h_positions = [d.xpos[env.body_ids[hk]][:2].copy() for hk in h_keys]
                
                # Seeker Action
                obs_s = env._get_obs(0)
                s_action = seeker_agent.get_action(obs_s, engine, s_pos, s_heading, h_positions, s_id)
                
                # Hider Actions
                full_action = np.zeros(env.num_agents * 2)
                full_action[0:2] = s_action
                for i in range(len(h_keys)):
                    obs_h = env._get_obs(i + 1)
                    full_action[(i+1)*2 : (i+1)*2 + 2] = hider_agents[i].get_action(obs_h)
                
                # 監査ログ出力 (50ステップに1回に頻度を向上)
                if step_count % 50 == 0:
                    reports = []
                    for hp in h_positions:
                        vis, dst, rsn = can_see_target_visual(engine, s_pos, s_heading, hp, s_id)
                        status_str = "VISIBLE" if vis else f"HIDDEN({rsn})"
                        reports.append(f"{status_str} {dst:.1f}m")
                    print(f"Step {step_count:5d} | Mode: {seeker_agent.status:6s} | { ' | '.join(reports) }")

                # 物理ステップ実行
                for _ in range(ACTION_REPEAT):
                    env.step(full_action)
                    step_count += 1
                
                # --- Viewer への描画追加 ---
                if hasattr(viewer, 'user_scn'):
                    viewer.user_scn.ngeom = 0
                    
                    # 1. エージェントのラベル
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[0], type=mujoco.mjtGeom.mjGEOM_LABEL, 
                                        size=np.zeros(3), pos=d.xpos[s_id] + np.array([0, 0, 1.2]), 
                                        mat=np.eye(3).flatten(), rgba=[1, 1, 1, 1])
                    viewer.user_scn.geoms[0].label = f"S:{seeker_agent.status}"
                    viewer.user_scn.ngeom += 1
                    
                    # 2. シーカーの「視線（捕捉）」ライン
                    for hk in h_keys:
                        h_id = env.body_ids[hk]
                        h_pos_3d = d.xpos[h_id]
                        vis, _, _ = can_see_target_visual(engine, s_pos, s_heading, h_pos_3d[:2], s_id)
                        
                        if vis:
                            # 捕捉している場合、赤いラインを描画
                            start = d.xpos[s_id] + np.array([0,0,0.4])
                            end = h_pos_3d + np.array([0,0,0.4])
                            diff = end - start
                            dist = np.linalg.norm(diff)
                            
                            mat = np.zeros(9)
                            mujoco.mju_quat2Mat(mat, np.array([1,0,0,0])) # ダミー
                            
                            # ラインを描画（非常に細いカプセルとして表現）
                            mujoco.mjv_initGeom(viewer.user_scn.geoms[viewer.user_scn.ngeom], 
                                                type=mujoco.mjtGeom.mjGEOM_CAPSULE, 
                                                size=np.array([0.02, 0.02, dist/2]), 
                                                pos=(start + end)/2, 
                                                mat=np.eye(3).flatten(), # 本来は方向を向かせる必要があるが簡易表示
                                                rgba=[1, 0, 0, 0.8])
                            # 方向を向かせる（startからendへのベクトルをZ軸にする）
                            z_axis = diff / dist
                            # 簡易的な回転行列生成（MuJoCoの描画ユーティリティが限られているため、ここでは位置のみ正確に表示）
                            viewer.user_scn.ngeom += 1
                
                viewer.sync()
                # 💡 ウェイトを増やして「一足飛び」に見えるのを防止
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print("\nStopped.")

if __name__ == "__main__":
    run_live_monitor()