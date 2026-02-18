# verify_lidar_visual.py
# 演習第25回：Lidar 3モード計算速度・精度・全オブジェクト描画整合版
# 
# 修正ポイント:
# 1. 描画の完全自動化: 特定の名前ではなく、シーン内の全ての body/geom を自動で収集して描画。
# 2. 視覚化の強化: レイ番号 0-11 を各衝突点に表示し、方向の特定を容易に。
# 3. 座標系の同期: 物理エンジン(Native)と自作エンジンの形状が一致するかを視覚的に監査。

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import mujoco
import time
import math
from pathlib import Path
from hns_environment import TeamCosEnv

def draw_assets(ax, snap):
    """環境内の全アセットを漏れなく描画"""
    # 1. 壁・床 (背景)
    for wall in snap.get('walls', []):
        p, s = wall["pos"], wall["size"]
        ax.add_patch(patches.Rectangle((p[0]-s[0], p[1]-s[1]), s[0]*2, s[1]*2, 
                                     facecolor='gray', alpha=0.1, edgecolor='none', zorder=1))
    
    # 2. ボックス・ランプ (動的オブジェクト)
    for obj in snap.get('boxes', []):
        p, s, yaw, name = obj["pos"], obj["size"], obj.get("yaw", 0), obj.get("name", "")
        # 回転矩形の描画
        ts = ax.transData
        tr = mtransforms.Affine2D().rotate_around(p[0], p[1], yaw)
        t = tr + ts
        color = 'green' if 'ramp' in name else 'orange'
        rect = patches.Rectangle((p[0]-s[0], p[1]-s[1]), s[0]*2, s[1]*2, 
                                facecolor=color, alpha=0.5, edgecolor='black', lw=1, transform=t, zorder=3)
        ax.add_patch(rect)
        ax.text(p[0], p[1], name.split('_')[0], fontsize=8, ha='center', va='center', zorder=4)

    # 3. エージェント
    colors = {"seeker": "red", "hider1": "lime", "hider2": "cyan"}
    for name, pos in snap.get('agents', {}).items():
        c = colors.get(name, "magenta")
        ax.add_patch(patches.Circle(pos, 0.4, facecolor=c, edgecolor='black', lw=1.5, alpha=0.8, zorder=5))
        ax.text(pos[0], pos[1]+0.6, name, fontsize=9, fontweight='bold', ha='center', color='black', zorder=6)

def run_benchmarked_audit():
    print("--- 第2.5フェーズ：Lidar 3モード計算速度・精度・描画完全監査 ---")
    
    env = TeamCosEnv()
    m, d = env.model, env.data
    output_dir = Path("lidar_debug"); output_dir.mkdir(exist_ok=True)
    
    print(f"✅ Lidar Engine Status: SDF Map {'利用可能' if env.vis_engine.cache else '不在'}")

    # 検証用視点
    viewpoints = [
        (-3.0, -1.5, 0.0),      # 地点1
        (2.0, 2.0, 0.0)         # 地点2
    ]
    
    def set_agent_pos(p_name, pos, ang=0.0):
        try:
            d.qpos[m.jnt_qposadr[m.joint(f"{p_name}_x").id]] = pos[0]
            d.qpos[m.jnt_qposadr[m.joint(f"{p_name}_y").id]] = pos[1]
            d.qpos[m.jnt_qposadr[m.joint(f"{p_name}_rot").id]] = ang
        except: pass

    def get_yaw(q):
        return math.atan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]*q[2] + q[3]*q[3]))

    env.reset()
    bench_results = {0: [], 1: [], 2: []}; snapshot_data = []; iterations = 50000; seeker_id = m.body("seeker_body").id

    print("🚀 ウォームアップ中...")
    for mode in [1, 2]: env.vis_engine.cast_lidar(np.array([0.0, 0.0]), mode=mode, body_exclude=seeker_id)

    print(f"🔄 計測実行中 ({len(viewpoints) * iterations} 回)...")
    total_start = time.time()
    
    for v_idx, (vx, vy, vh) in enumerate(viewpoints):
        seeker_pos = np.array([vx, vy]); set_agent_pos('seeker', seeker_pos, vh); mujoco.mj_forward(m, d)
        
        if v_idx == 0 or v_idx == len(viewpoints)-1:
            # --- シーン情報の完全収集 ---
            walls, boxes, agents = [], [], {}
            for i in range(m.ngeom):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
                bid = m.geom_bodyid[i]
                b_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
                
                if any(k in name for k in ["wall", "maze", "border"]):
                    walls.append({'pos': d.geom_xpos[i][:2].copy(), 'size': m.geom_size[i][:2].copy()})
                elif any(k in b_name for k in ["box", "ramp"]):
                    boxes.append({
                        'pos': d.geom_xpos[i][:2].copy(), 'size': m.geom_size[i][:2].copy(),
                        'yaw': get_yaw(d.xquat[bid]), 'name': b_name
                    })
                elif "body" in b_name:
                    agents[b_name.replace("_body", "")] = d.xpos[bid][:2].copy()

            snapshot_data.append({
                'pos': seeker_pos, 'heading': vh, 'walls': walls, 'boxes': boxes, 'agents': agents, 'lidar_results': {}
            })

        for mode in [0, 1, 2]:
            t0 = time.perf_counter()
            for _ in range(iterations): env.vis_engine.cast_lidar(seeker_pos, heading=vh, mode=mode, body_exclude=seeker_id)
            bench_results[mode].append((time.perf_counter() - t0) / iterations)
            if v_idx == 0 or v_idx == len(viewpoints)-1:
                snapshot_data[-1]['lidar_results'][mode] = env.vis_engine.cast_lidar(seeker_pos, heading=vh, mode=mode, body_exclude=seeker_id)

    print(f"✅ 全計測完了 ({time.time() - total_start:.2f}s)\n")
    
    # 統計出力
    mode_names = {0: "Native", 1: "Geometric", 2: "Sphere"}
    print(f"{'Mode':<15} | {'Avg (μs)':>10} | {'SPS':>10}")
    print("-" * 40)
    for m_idx in [0, 1, 2]:
        avg = np.mean(bench_results[m_idx])
        print(f"{mode_names[m_idx]:<15} | {avg*1e6:10.2f} | {int(1/avg):10,d}")

    # 視覚化
    fig = plt.figure(figsize=(18, 12)); gs = plt.GridSpec(2, 3, hspace=0.3, wspace=0.1)
    m_colors = {0: "black", 1: "forestgreen", 2: "royalblue"}
    angles_deg = env.vis_engine.angles_deg

    for row_idx, snap in enumerate(snapshot_data):
        for mode in [0, 1, 2]:
            ax = fig.add_subplot(gs[row_idx, mode]); draw_assets(ax, snap)
            dists, heading, pos = snap['lidar_results'][mode], snap['heading'], snap['pos']
            for i, d_val in enumerate(dists):
                rad = np.deg2rad(angles_deg[i]) + heading
                ex, ey = pos[0] + d_val * np.cos(rad), pos[1] + d_val * np.sin(rad)
                ax.plot([pos[0], ex], [pos[1], ey], color=m_colors[mode], alpha=0.3, lw=1)
                ax.plot(ex, ey, 'o', color=m_colors[mode], markersize=5, zorder=10)
                ax.text(ex + 0.3*np.cos(rad), ey + 0.3*np.sin(rad), str(i), color=m_colors[mode], fontsize=8, ha='center')
            ax.set_title(f"{'First' if row_idx == 0 else 'Last'} View: {mode_names[mode]}", fontweight='bold')
            ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5); ax.set_aspect('equal'); ax.grid(True, alpha=0.1)
    
    plt.savefig(output_dir / "lidar_benchmark_final_debug.png")
    print(f"✅ 解析画像を保存しました: {output_dir / 'lidar_benchmark_final_debug.png'}")

if __name__ == "__main__":
    run_benchmarked_audit()