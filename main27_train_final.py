# main26_train_final.py v5.5
# PEP 8 準拠: 79文字制限、1行1文、デバッグログ強化版

import torch
import math
import numpy as np
import time
import sys
from core.obs_indices import ObsIdx
from envs.hns_environment import TeamCosEnv

# モード設定
TRAIN_MODE = False
USE_VIEWER = True
NPC_ONLY_DEBUG = True

def compute_custom_reward(obs, action, base_reward, idx, target_team="seeker"):
    """
    動的インデックスを利用した報酬シェイピング。
    """
    # 敵が1人でも見えているか
    enemy_visible = any(obs[en.VISIBLE] > 0.5 for en in idx.OTHERS)

    # 最も近い敵との距離
    min_enemy_dist = 999.0
    for en in idx.OTHERS:
        d = math.sqrt(obs[en.REL_X]**2 + obs[en.REL_Y]**2)
        if d < min_enemy_dist:
            min_enemy_dist = d

    # 自己速度
    v_idx_x = idx.SELF.VEL_X
    v_idx_y = idx.SELF.VEL_Y
    speed = math.sqrt(obs[v_idx_x]**2 + obs[v_idx_y]**2)

    bonus = 0.0
    if target_team == "seeker":
        if enemy_visible:
            bonus += 0.05
        # 接近ボーナス
        bonus += 0.02 / (min_enemy_dist + 0.5)
        # 停止ペナルティ
        if speed < 0.05:
            bonus -= 0.01

    # 行動の激しさに対するコスト
    control_cost = -0.005 * np.sum(np.square(action))
    return base_reward + bonus + control_cost


def run_simulation():
    """メインシミュレーションループ。"""
    config = {
        "n_seekers": 1,
        "n_hiders": 2,
        "n_boxes": 2,
        "n_ramps": 1
    }

    # 環境の初期化
    print("Initializing Environment...")
    env = TeamCosEnv(mode="refinement", target="seeker", **config)
    idx = env.idx

    print("--- Start Simulation ---")

    try:
        for episode in range(100):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            step_count = 0
            viewer_dead_count = 0

            # 初期状態を表示
            if USE_VIEWER:
                env.render()

            print(f"Episode {episode} started.", end=" ", flush=True)

            while not done:
                # Viewerが手動で閉じられたかチェック
                if USE_VIEWER and env.viewer:
                    if not env.viewer.is_running():
                        viewer_dead_count += 1
                        if viewer_dead_count >= 10:
                            print("\nViewer window closed by user.")
                            return
                    else:
                        viewer_dead_count = 0

                # 行動決定
                if TRAIN_MODE:
                    # 強化学習推論（未実装時はランダム）
                    action = env.action_space.sample()
                elif NPC_ONLY_DEBUG:
                    # 全員をスクリプトNPCとして動作させる
                    from agents.scripted_agents import (RuleBasedSeeker,
                                                       RuleBasedHider)
                    # ターゲットチーム用のNPCロジックを生成
                    if env.target == "seeker":
                        debug_npc = RuleBasedSeeker()
                    else:
                        debug_npc = RuleBasedHider()
                    
                    # 観測データからアクションを計算
                    action = debug_npc.get_action(obs, idx)
                else:
                    # 静止状態
                    action = np.zeros(env.action_space.shape)

                # 環境ステップ実行
                next_obs, base_r, term, trun, info = env.step(action)
                
                # カスタム報酬の計算
                reward = compute_custom_reward(obs, action, base_r,
                                               idx, env.target)

                # レンダリングと待機
                if USE_VIEWER:
                    env.render()
                    time.sleep(0.02)

                obs = next_obs
                ep_reward += reward
                step_count += 1
                done = bool(term or trun)

                # 進捗ログ（50ステップごと）
                if step_count % 50 == 0:
                    print(".", end="", flush=True)

            print(f" Finished. Steps={step_count}, Reward={ep_reward:.2f}")

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nCritical Error during simulation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environment...")
        env.close()
        sys.exit(0)


if __name__ == "__main__":
    run_simulation()