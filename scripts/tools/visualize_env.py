# visualize_env.py
# Viewerで環境動作と配置を確認します

import time

import mujoco.viewer
import numpy as np

from envs.hns_environment import TeamCosEnv


def main():
    # 環境の初期化
    # create env with a typical configuration (avoid passing removed args)
    env = TeamCosEnv(mode="initial", target="seeker", n_seekers=1, n_hiders=2, n_boxes=2, n_ramps=1, render_mode=None)

    # 物理エンジンのハンドル取得
    model = env.model
    data = env.data

    print("Viewerを起動しました。")
    print("- 緑: Hider1, 水色: Hider2, 赤: Seeker, 黄: Ramp, 茶: Box")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # カメラの初期位置を設定（環境全体が見える俯瞰視点）
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]  # 注視点: 環境の中心
        viewer.cam.distance = 25.0  # 距離: 25m
        viewer.cam.azimuth = 90.0  # 方位角: 90度
        viewer.cam.elevation = -70.0  # 仰角: -70度（上から見下ろす）

        while viewer.is_running():
            # 1. リセットデバッグ (Spaceキー相当の自動リセット例)
            # 実際には学習ループと同様に env.reset() を呼ぶ
            obs, _ = env.reset()

            # 2. 短時間のシミュレーション実行
            for _ in range(200):
                # ランダムアクション
                action = np.random.uniform(-1, 1, 3)
                env.step(action)

                # 表示更新
                viewer.sync()
                time.sleep(0.01)


if __name__ == "__main__":
    main()
