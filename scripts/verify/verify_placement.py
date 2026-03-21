# verify_placement.py
# 物理環境の第1フェーズ検証：Seed一致テストと衝突監視

import numpy as np
from hns_environment import TeamCosEnv


def verify_phase1():
    print("--- 第1フェーズ検証：初期配置の決定論的テスト ---")
    env = TeamCosEnv()

    # 1. Seed一致テスト
    seed = 42
    obs1, _ = env.reset(seed=seed)
    pos1 = env.data.qpos.copy()

    obs2, _ = env.reset(seed=seed)
    pos2 = env.data.qpos.copy()

    diff = np.max(np.abs(pos1 - pos2))
    if diff < 1e-10:
        print(f"✅ Seed一致テスト成功 (最大誤差: {diff:.2e})")
    else:
        print(f"❌ Seed一致テスト失敗 (最大誤差: {diff:.2e})")

    # 2. 衝突ログの監視
    collisions = []
    for i in range(50):
        env.reset()
        # mj_forward後に接触点数を数える
        collisions.append(env.data.ncon)

    avg_col = np.mean(collisions)
    max_col = np.max(collisions)
    print(f"📊 衝突統計 (50回試行): 平均 {avg_col:.1f}点, 最大 {max_col}点")

    if max_col < 50:  # 異常な反発（弾け飛び）が起きていない目安
        print("✅ 初期配置の安全性確認（深刻な融合なし）")
    else:
        print("⚠️ 警告：一部の配置で衝突数が多い可能性があります")


if __name__ == "__main__":
    verify_phase1()
