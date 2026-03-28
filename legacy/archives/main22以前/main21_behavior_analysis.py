# main21_behavior_analysis.py
# 演習第20回：学習済みモデルの行動診断スクリプト
#
# 【修正(v20.4)】
# - インポート元を main21_layer0_runner.py に変更し、Layer0RunnerEnv を使用。
# - 分析対象を "HideAndSeek_Layer0_Runner" に設定。
# - Layer 0 (Runner) の報酬体系（距離増分、停留ペナルティ、生存報酬）に合わせて集計ロジックを最適化。
#
# 【実行準備】
# uv add matplotlib seaborn

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# --- 必要なライブラリのチェック ---
try:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Error: 必要なライブラリが見つかりません。")
    print("以下のコマンドでインストールしてください:")
    print("uv add matplotlib seaborn")
    sys.exit(1)

# ★修正点: main21 から必要なクラスをインポート
try:
    # クラス名が Layer0RunnerEnv なので、解析コードの互換性のために HideAndSeekEnv としてインポート
    import main21_layer0_runner as target_config
    from main21_layer0_runner import Agent
    from main21_layer0_runner import Layer0RunnerEnv as HideAndSeekEnv
    from main21_layer0_runner import ObsHistory
except ImportError:
    print("Error: main21_layer0_runner.py が見つかりません。")
    exit(1)

# Matplotlib日本語対応 & スタイル
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
sns.set_style("whitegrid")

# ==========================================
# 1. 分析設定
# ==========================================
EXPERIMENT_NAME = "HideAndSeek_Layer0_Runner"
MODEL_PATH = f"{EXPERIMENT_NAME}_HIDER.pt"
NUM_TEST_EPISODES = 20
OUTPUT_DIR = Path("analysis_results_layer0")
OUTPUT_DIR.mkdir(exist_ok=True)

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_diagnostic():
    print(f"--- Behavior Diagnosis: {EXPERIMENT_NAME} ---")

    # 1. モデルのロード
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    # 環境とエージェントの初期化
    env = HideAndSeekEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = Agent(obs_dim, act_dim).to(device)
    agent.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    agent.eval()

    obs_history = ObsHistory(1, target_config.base_config.TRANSFORMER_SEQ_LEN, obs_dim, device)

    # データ蓄積用
    results = {
        "ep_lengths": [],
        "ep_rewards": [],
        "trajectories": [],  # List of (x, y)
        "reward_components": defaultdict(float),
        "capture_events": [],  # List of (x, y) where captured
        "distances": [],  # Distance to seeker over time
    }

    print(f"Running {NUM_TEST_EPISODES} test episodes...")

    for ep in range(NUM_TEST_EPISODES):
        obs, _ = env.reset()
        obs_history.reset()
        obs_history.update(obs)

        done = False
        ep_ret = 0
        ep_len = 0

        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_history.get())

            # Step実行
            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            done = terminated or truncated

            # 軌跡保存 (Hider 1 or 2)
            bid = env.h1_body if env.learning_agent_id == 1 else env.h2_body
            pos = env.data.xpos[bid][:2].copy()
            results["trajectories"].append(pos)

            # Seekerとの距離
            sp = env.data.xpos[env.s0_body][:2]
            dist = np.linalg.norm(pos - sp)
            results["distances"].append(dist)

            # 報酬分解 (簡易版)
            if reward > 0 and not done:
                results["reward_components"]["Survival/Distance"] += reward
            elif reward < -10:
                results["reward_components"]["Capture Penalty"] += reward
                results["capture_events"].append(pos)
            elif reward < 0:
                results["reward_components"]["Stagnation Penalty"] += reward

            ep_ret += reward
            ep_len += 1
            obs_history.update(next_obs)

        results["ep_lengths"].append(ep_len)
        results["ep_rewards"].append(ep_ret)

    # 2. 可視化・分析
    plot_results(results, env)

    # 統計出力
    print("\n--- Diagnostic Statistics ---")
    print(f"  Avg Episode Length: {np.mean(results['ep_lengths']):.1f} / {target_config.base_config.MAX_STEPS}")
    print(f"  Capture Rate:       {len(results['capture_events'])/NUM_TEST_EPISODES*100:.1f}%")
    print(f"  Avg Distance:       {np.mean(results['distances']):.2f}m")
    print(f"  Results saved to:   {OUTPUT_DIR}/")


def plot_results(results, env):
    # A. 軌跡ヒートマップ
    plt.figure(figsize=(10, 8))
    traj = np.array(results["trajectories"])

    # フィールドの外枠
    plt.plot([-6, 6, 6, -6, -6], [-6, -6, 6, 6, -6], "k-", lw=2)

    # 壁の描画
    for w_pos, w_size in env.wall_data:
        rect = patches.Rectangle(
            (w_pos[0] - w_size[0], w_pos[1] - w_size[1]),
            w_size[0] * 2,
            w_size[1] * 2,
            linewidth=1,
            edgecolor="blue",
            facecolor="cyan",
            alpha=0.3,
        )
        plt.gca().add_patch(rect)

    # ヒートマップ
    if len(traj) > 0:
        plt.hexbin(traj[:, 0], traj[:, 1], gridsize=30, cmap="YlOrRd", alpha=0.8)
        plt.colorbar(label="滞在頻度")

    # 捕獲地点
    if results["capture_events"]:
        caps = np.array(results["capture_events"])
        plt.scatter(
            caps[:, 0],
            caps[:, 1],
            marker="x",
            color="black",
            s=100,
            label="Capture Point",
            zorder=5,
        )

    plt.title(f"Hider Trajectory Heatmap ({EXPERIMENT_NAME})")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.xlim(-7, 7)
    plt.ylim(-7, 7)
    plt.legend()
    plt.savefig(OUTPUT_DIR / "trajectory_heatmap.png")
    plt.close()

    # B. 報酬内訳 (パイチャート)
    plt.figure(figsize=(8, 8))
    labels = list(results["reward_components"].keys())
    values = [abs(v) for v in results["reward_components"].values()]
    if sum(values) > 0:
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
        plt.title("Absolute Reward Components Breakdown")
        plt.savefig(OUTPUT_DIR / "reward_breakdown.png")
    plt.close()

    # C. 生存時間の分布
    plt.figure(figsize=(10, 5))
    sns.histplot(results["ep_lengths"], bins=range(0, 301, 10), kde=True, color="green")
    plt.axvline(np.mean(results["ep_lengths"]), color="red", linestyle="--", label="Average")
    plt.title("Episode Length Distribution")
    plt.xlabel("Steps")
    plt.ylabel("Frequency")
    plt.xlim(0, 310)
    plt.legend()
    plt.savefig(OUTPUT_DIR / "survival_distribution.png")
    plt.close()


if __name__ == "__main__":
    run_diagnostic()
