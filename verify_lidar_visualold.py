# verify_lidar_visual.py
# 演習第25回：Lidar 3モード検証スクリプト (デバッグ出力対応版)

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import time
from hns_environment import TeamCosEnv

def main():
    print("🚀 verify_lidar_visual: Starting benchmark...")
    env = TeamCosEnv()
    engine = env.vis_engine
    
    # テスト地点の定義
    test_points = [
        {"pos": np.array([-3.0, 3.0]), "heading": 0.0, "label": "First View"},
        {"pos": np.array([0.0, 1.0]), "heading": np.deg2rad(45), "label": "Last View"}
    ]
    
    modes = [0, 1, 2] # Native, Geometric, Sphere
    mode_names = ["Native", "Geometric", "Sphere"]
    
    fig, axes = plt.subplots(len(test_points), 3, figsize=(18, 12))
    
    for row, pt in enumerate(test_points):
        pos = pt["pos"]
        heading = pt["heading"]
        print(f"\nTesting point {row+1}: {pt['label']} at {pos}")
        
        for mode in modes:
            # パフォーマンス計測
            t_start = time.perf_counter()
            lidar_res = engine.cast_lidar(pos, heading=heading, mode=mode)
            elapsed = (time.perf_counter() - t_start) * 1e6
            
            ax = axes[row, mode]
            ax.set_title(f"{pt['label']}: {mode_names[mode]}")
            
            # 背景（壁）の描画
            for w in engine.walls_np:
                rect = plt.Rectangle((w[0]-w[2], w[1]-w[3]), w[2]*2, w[3]*2, color='gray', alpha=0.1)
                ax.add_patch(rect)
                
            # 動的オブジェクト（デバッグ用）
            for gid, bid in zip(engine.idx_box_geom, engine.idx_box_body):
                p = env.data.geom_xpos[gid]
                s = env.model.geom_size[gid]
                rect = plt.Rectangle((p[0]-s[0], p[1]-s[1]), s[0]*2, s[1]*2, color='orange', alpha=0.4, ec='black')
                ax.add_patch(rect)
                ax.text(p[0], p[1], "box", ha='center', va='center', fontsize=8)

            # レイの描画
            angles_deg = np.array([0, 15, -15, 30, -30, 45, -45, 90, -90, 135, -135, 180])
            for i, d in enumerate(lidar_res):
                angle_rad = np.deg2rad(angles_deg[i]) + heading
                vx, vy = np.cos(angle_rad), np.sin(angle_rad)
                color = 'black' if mode==0 else ('green' if mode==1 else 'blue')
                ax.plot([pos[0], pos[0]+vx*d], [pos[1], pos[1]+vy*d], color=color, alpha=0.3, linewidth=1)
                ax.scatter(pos[0]+vx*d, pos[1]+vy*d, color=color, s=20)
                if d < 19.0:
                    ax.text(pos[0]+vx*d, pos[1]+vy*d, f"{i}", fontsize=9, color=color)

            # 視点
            ax.scatter(pos[0], pos[1], color='red', s=100, label='Origin')
            ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5)
            ax.set_aspect('equal')
            ax.grid(True, which='both', linestyle='--', alpha=0.2)
            
            print(f"  {mode_names[mode]:10} | {elapsed:6.2f} μs | Result[8]: {lidar_res[8]:5.2f}m")

    plt.tight_layout()
    plt.savefig("lidar_benchmark_final_debug.png")
    print("\n✅ Benchmark report saved: lidar_benchmark_final_debug.png")
    plt.show()

if __name__ == "__main__":
    main()