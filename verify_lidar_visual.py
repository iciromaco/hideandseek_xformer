# verify_lidar_visual.py
# 演習第25回：Lidarスキャン結果の視覚化・監査スクリプト
# 
# 異なるシナリオ（壁際、箱の隣など）でLidarを走らせ、
# エージェントが周囲の形状をどう捉えているかをレーダーチャートで出力します。

import pickle
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from pathlib import Path
from hns_environment import TeamCosEnv

def plot_lidar_scan(name, pos, distances, lidar_dirs, output_dir: Path):
    """Lidarの計測結果を極座標プロット（レーダーチャート）で表示"""
    # 12方向の角度 (0, 30, ..., 330度) をラジアンに変換
    # 閉じた円にするために、最初の点を最後に追加
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    
    plot_angles = np.concatenate((angles, [angles[0]]))
    plot_dists = np.concatenate((distances, [distances[0]]))

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    
    # Lidarの計測範囲（最大10m）
    ax.set_ylim(0, 10)
    
    # 計測データのプロット
    ax.plot(plot_angles, plot_dists, 'b-o', linewidth=2, markersize=5, label="Lidar Scan")
    ax.fill(plot_angles, plot_dists, 'b', alpha=0.1)
    
    # 補助線の設定
    ax.set_theta_zero_location('E') # 0度を東（右）に設定
    ax.set_theta_direction(1)      # 反時計回り
    
    plt.title(f"Lidar Scan Analysis: {name}\nPos: ({pos[0]:.2f}, {pos[1]:.2f})")
    plt.legend(loc='upper right')
    
    filename = output_dir / f"lidar_check_{name.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"✅ Lidar可視化画像を保存しました: {filename}")

def run_lidar_audit():
    print("--- Lidar機能・視覚化監査プロセス開始 ---")
    env = TeamCosEnv()
    m = env.model
    d = env.data
    
    output_dir = Path("lidar_debug")
    output_dir.mkdir(exist_ok=True)

    # 監査用シナリオの定義
    scenarios = [
        {"name": "Open Field", "pos": [0.0, 0.0]},           # 中央（何もない）
        {"name": "Near North Wall", "pos": [0.0, 5.5]},      # 北壁ギリギリ
        {"name": "Near Maze Center", "pos": [2.5, 1.5]},     # 迷路の壁(maze_w0)の横
        {"name": "Between Boxes", "pos": [2.0, -1.5]}        # 箱のすぐそば
    ]

    seeker_body_id = m.body("seeker_body").id
    
    for sc in scenarios:
        print(f"📍 シナリオ実行中: {sc['name']}")
        
        env.reset()
        # Seekerを特定の位置に強制配置
        # anchorからの相対位置として設定
        anchor_id = m.body("seeker_anchor").id
        target_pos = np.array(sc['pos'])
        d.qpos[21:23] = target_pos - m.body_pos[anchor_id][:2]
        
        mujoco.mj_forward(m, d)
        
        # Lidar計測 (VisibilityEngineを使用)
        # body_exclude を指定して自分自身に当たらないようにする
        distances = env.vis_engine.cast_lidar(target_pos, body_exclude=seeker_body_id)
        
        # 視覚化
        plot_lidar_scan(sc['name'], target_pos, distances, env.vis_engine.lidar_dirs, output_dir)

    print(f"\n💡 すべてのLidar監査画像が '{output_dir}' 内に出力されました。")

if __name__ == "__main__":
    run_lidar_audit()