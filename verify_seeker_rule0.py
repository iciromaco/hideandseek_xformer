# verify_seeker_rule.py
# 演習第25回：シーカー・ルールベース行動検証（最適化済み高速エンジン搭載版）
# 
# 目的:
# 1. 最適化された VisibilityEngine によるハイダー視認判定の精度確認。
# 2. シーカーの基本的な索敵ルール（発見時追跡 / 未発見時探索）の挙動テスト。
# 3. 動的オブジェクト（箱/スロープ）による遮蔽が正しくルールに反映されるかの検証。

import matplotlib
matplotlib.use('Agg')

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from hns_environment import TeamCosEnv

def run_seeker_rule_test():
    print("🚀 verify_seeker_rule: Testing Seeker Logic with optimized engine...")
    
    # 1. 環境とエンジンの初期化
    env = TeamCosEnv()
    engine = env.vis_engine
    
    # シーカーとハイダーのボディIDを取得
    m = env.model
    seeker_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "seeker_body")
    if seeker_id == -1: # 接尾辞がない場合のフォールバック
        seeker_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "s_body")
        
    hider_ids = []
    for name in ["hider1_body", "hider2_body", "h1_body", "h2_body"]:
        hid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if hid != -1:
            hider_ids.append(hid)

    print(f"  Seeker Body ID: {seeker_id}")
    print(f"  Hider Body IDs: {hider_ids}")

    # シナリオ設定：特定の配置で視界をチェック
    # 状況1: シーカーの目の前に障害物（箱）があり、ハイダーが隠れている
    # 状況2: 視界が開けており、直接視認できる
    
    # 描画の準備
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- テストループ ---
    for step, ax in enumerate(axes):
        # mj_forward で物理状態を最新にする
        mujoco.mj_forward(env.model, env.data)
        
        seeker_pos = env.data.xpos[seeker_id][:2]
        
        ax.set_title(f"Detection Scenario {step+1}")
        
        # 2. 視認判定の実施 (is_visible)
        visible_hiders = []
        for hid in hider_ids:
            hider_pos = env.data.xpos[hid][:2]
            # シーカーからハイダーへの視線が通るか？ (自分自身 seeker_id は除外)
            is_seen = engine.is_visible(seeker_pos, hider_pos, body_exclude=seeker_id)
            
            if is_seen:
                visible_hiders.append(hid)
                color = 'red'
                linestyle = '-'
                label = "VISIBLE"
            else:
                color = 'gray'
                linestyle = '--'
                label = "OCCLUDED"
            
            # 視界線の描画
            ax.plot([seeker_pos[0], hider_pos[0]], [seeker_pos[1], hider_pos[1]], 
                    color=color, linestyle=linestyle, alpha=0.8, linewidth=2)
            ax.text((seeker_pos[0]+hider_pos[0])/2, (seeker_pos[1]+hider_pos[1])/2, 
                    label, color=color, fontweight='bold', fontsize=10)

        # 3. ルールベースの行動決定（ロギングのみ）
        if visible_hiders:
            decision = f"RULE: Pursuit Hider(s) {visible_hiders}"
        else:
            decision = "RULE: Area Search (Explore)"
        print(f"  Scenario {step+1}: {decision}")

        # 4. 背景描画 (物理スタックの可視化)
        # 静的な壁
        for w in engine.walls_np:
            ax.add_patch(plt.Rectangle((w[0]-w[2], w[1]-w[3]), w[2]*2, w[3]*2, color='black', alpha=0.1))
        
        # 動的な箱
        for gid in engine.idx_box_geom:
            p = env.data.geom_xpos[gid]; s = env.model.geom_size[gid]
            ax.add_patch(plt.Rectangle((p[0]-s[0], p[1]-s[1]), s[0]*2, s[1]*2, color='orange', alpha=0.4))

        # エージェント本体
        ax.add_patch(plt.Circle((seeker_pos[0], seeker_pos[1]), 0.45, color='red', label='Seeker'))
        for hid in hider_ids:
            hp = env.data.xpos[hid][:2]
            ax.add_patch(plt.Circle((hp[0], hp[1]), 0.45, color='lime', label='Hider'))

        ax.set_xlim(-6.5, 6.5); ax.set_ylim(-6.5, 6.5)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 次のステップのためにハイダーを移動させる（デバッグ用）
        if step == 0:
            # ハイダーを箱の裏から開けた場所へ強制移動
            for hid in hider_ids:
                # 実際の環境では data.qpos を操作するが、ここでは簡易的に xpos をずらすデモ
                # (本来は物理演算を介すべきだが、視界判定のロジックテストとして)
                env.data.xpos[hid][0] += 3.0 

    plt.tight_layout()
    plt.savefig("seeker_rule_audit.png", dpi=150)
    print("\n✅ Seeker rule audit report saved: seeker_rule_audit.png")

if __name__ == "__main__":
    run_seeker_rule_test()