# verify_scripted_match.py v1.2
# 演習第26回：ルールベース最強対決（ライブラリ統合 ＆ 監査環境完遂版）
# 
# 修正履歴:
# v1.1: ランダムシード ＆ カメラ最適化。
# v1.2: scripted_agents.py v1.70 への完全移行。
#       1. 外部インポートの依存を排除し、管理を容易化。
#       2. H1(緑): 賢いHider, H2(青): ランダムHider の対比を明確化。

import mujoco
import mujoco.viewer
import numpy as np
import time
import math

from envs.hns_environment import TeamCosEnv
from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider, SimpleWanderer

# --- 監査設定 ---
ACTION_REPEAT = 2
TARGET_FPS = 15.0
FRAME_DURATION = 1.0 / TARGET_FPS
FOV_HALF_RAD = 135 * 0.5 * math.pi / 180.0
RESET_INTERVAL = 3000

def can_see(engine, p1, h1, p2, b_exclude, t_id):
    diff = p2 - p1; dist = np.linalg.norm(diff)
    if dist > 15.0: return False
    v_dir = np.array([math.cos(h1), math.sin(h1)])
    if np.dot(v_dir, diff/dist) < math.cos(FOV_HALF_RAD): return False
    return engine.is_visible(p1, p2, body_exclude=b_exclude, target_body_id=t_id)

def run_match():
    print("="*120)
    print("🚀 Rule-Based Final Match: Seeker vs Smart Hider vs Random Hider")
    print("="*120)
    
    env = TeamCosEnv(); m, d = env.model, env.data
    s_agent = RuleBasedSeeker()
    h1_smart = RuleBasedHider() 
    h2_random = SimpleWanderer(1) 

    s_id = env.body_ids['s']; h1_id = env.body_ids['h1']; h2_id = env.body_ids['h2']
    s_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "seeker_btm")
    h1_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "hider1_btm")
    h2_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "hider2_btm")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # カメラの初期位置を俯瞰に設定
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        viewer.cam.distance = 18.0
        viewer.cam.elevation = -75.0
        
        step = 0
        while viewer.is_running():
            t_start = time.perf_counter()
            if step % RESET_INTERVAL == 0: 
                new_seed = int(time.time() * 1000) % 100000
                env.reset(seed=new_seed)
            
            s_pos = d.geom_xpos[s_geom][:2].copy(); s_rot = d.qpos[m.jnt_qposadr[env.qpos_indices['s']['rot']]]
            h1_pos = d.geom_xpos[h1_geom][:2].copy(); h1_rot = d.qpos[m.jnt_qposadr[env.qpos_indices['h1']['rot']]]
            h2_pos = d.geom_xpos[h2_geom][:2].copy()
            
            act_s = s_agent.get_action(env._get_obs(0), env.vis_engine, s_pos, s_rot, [h1_pos, h2_pos], s_id, [h1_id, h2_id])
            act_h1 = h1_smart.get_action(env._get_obs(1), h1_pos, h1_rot, s_pos)
            act_h2 = h2_random.get_action(env._get_obs(2), h2_pos)
            
            env.step(np.concatenate([act_s, act_h1, act_h2]))
            
            if hasattr(viewer, 'user_scn'):
                ctx = viewer.user_scn; ctx.ngeom = 0
                for bid, label, col in [(s_id, f"S:{s_agent.status}", [1,0,0,1]), (h1_id, f"H1:{h1_smart.status}", [0,1,0,1]), (h2_id, f"H2:RAND", [0,0.7,1,1])]:
                    mujoco.mjv_initGeom(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LABEL, size=[0,0,0], pos=d.xpos[bid]+[0,0,1.2], mat=np.eye(3).flatten(), rgba=col)
                    ctx.geoms[ctx.ngeom].label = label; ctx.ngeom += 1
                
                s_c = d.geom_xpos[s_geom].copy(); s_c[2]=0.4
                for tid, gid, col in [(h1_id, h1_geom, [1,0,0,1]), (h2_id, h2_geom, [0,0.7,1,1])]:
                    t_c = d.geom_xpos[gid].copy(); t_c[2]=0.4
                    if can_see(env.vis_engine, s_pos, s_rot, t_c[:2], s_id, tid):
                        mujoco.mjv_connector(ctx.geoms[ctx.ngeom], type=mujoco.mjtGeom.mjGEOM_LINE, width=4.0, from_=s_c, to=t_c)
                        ctx.geoms[ctx.ngeom].rgba = col; ctx.ngeom += 1
            
            viewer.sync(); step += 1
            elapsed = time.perf_counter() - t_start
            if elapsed < FRAME_DURATION: time.sleep(FRAME_DURATION - elapsed)

if __name__ == "__main__": run_match()