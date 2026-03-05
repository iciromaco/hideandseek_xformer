# main27_train_final.py v5.8
# PEP 8 準拠: チャタリング監視ログの追加とシミュレーションの安定化

import torch
import math
import numpy as np
import time
import sys
import os
import traceback
from core.obs_indices import ObsIdx
from envs.hns_environment import TeamCosEnv

# モード設定
TRAIN_MODE = False
USE_VIEWER = True
NPC_ONLY_DEBUG = True

# src ディレクトリを検索パスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

def compute_custom_reward(obs, action, base_reward, idx, target_team="seeker"):
    """動的インデックスを利用した報酬シェイピング。"""
    enemy_visible = any(obs[en.VISIBLE] > 0.5 for en in idx.OTHERS)

    min_enemy_dist = 999.0
    for en in idx.OTHERS:
        d = math.sqrt(obs[en.REL_X]**2 + obs[en.REL_Y]**2)
        if d < min_enemy_dist:
            min_enemy_dist = d

    speed = math.sqrt(obs[idx.SELF.VEL_X]**2 + obs[idx.SELF.VEL_Y]**2)

    bonus = 0.0
    if target_team == "seeker":
        if enemy_visible:
            bonus += 0.05
        bonus += 0.02 / (min_enemy_dist + 0.5)
        if speed < 0.01:
            bonus -= 0.02

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

    print("Initializing Environment...")
    try:
        env = TeamCosEnv(mode="refinement", target="seeker", **config)
        idx = env.idx
    except Exception:
        print("\nFailed to initialize environment:")
        traceback.print_exc()
        return

    print("--- Start Simulation ---")
    print(f"Physics Step: {env.model.opt.timestep * 5}s")

    try:
        for episode in range(100):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            step_count = 0
            lock_events = 0
            grab_events = 0
            lock_events_window = 0
            grab_events_window = 0

            target_npc = None
            if NPC_ONLY_DEBUG:
                from agents.scripted_agents import RuleBasedSeeker, RuleBasedHider
                target_npc = RuleBasedSeeker() if env.target == "seeker" else RuleBasedHider()

            if USE_VIEWER:
                env.render()

            while not done:
                if USE_VIEWER and env.viewer:
                    if not env.viewer.is_running():
                        return

                if TRAIN_MODE:
                    action = env.action_space.sample()
                elif NPC_ONLY_DEBUG and target_npc is not None:
                    action = target_npc.get_action(obs, idx)
                else:
                    action = np.zeros(env.action_space.shape)

                next_obs, base_r, term, trun, info = env.step(action)
                reward = compute_custom_reward(obs, action, base_r, idx, env.target)
                lock_events += int(info.get("lock_event", 0))
                grab_events += int(info.get("grab_event", 0))
                lock_events_window += int(info.get("lock_event", 0))
                grab_events_window += int(info.get("grab_event", 0))

                if USE_VIEWER:
                    env.render()
                    time.sleep(0.025)

                obs = next_obs
                ep_reward += reward
                step_count += 1
                done = bool(term or trun)

                if step_count % 100 == 0:
                    # アクション（特にステアリング T）の絶対値を出力してチャタリングを確認
                    print(
                        "Ep "
                        f"{episode} - Step {step_count} | "
                        f"Steer={action[1]:.2f} "
                        f"LockBtnMax={info.get('dbg_lock_btn_max', 0.0):.2f} "
                        f"GrabBtnMax={info.get('dbg_grab_btn_max', 0.0):.2f} "
                        f"LockPressed={int(info.get('dbg_lock_pressed', 0))} "
                        f"GrabPressed={int(info.get('dbg_grab_pressed', 0))} "
                        f"LockTarget={int(info.get('dbg_lock_target', 0))} "
                        f"GrabTarget={int(info.get('dbg_grab_target', 0))} "
                        f"LockEvt={int(info.get('lock_event', 0))} "
                        f"GrabEvt={int(info.get('grab_event', 0))} "
                        f"WinLock={lock_events_window} "
                        f"WinGrab={grab_events_window} "
                        f"CumLock={lock_events} "
                        f"CumGrab={grab_events}"
                    )
                    lock_events_window = 0
                    grab_events_window = 0

            print(
                f"Ep {episode} Finished. Steps={step_count}, Reward={ep_reward:.2f}, "
                f"LockEvents={lock_events}, GrabEvents={grab_events}"
            )

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        print("\nUnexpected error:")
        traceback.print_exc()
        if USE_VIEWER:
            time.sleep(5)
    finally:
        env.close()
        sys.exit(0)


if __name__ == "__main__":
    run_simulation()