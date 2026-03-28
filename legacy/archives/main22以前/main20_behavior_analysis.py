# main20_behavior_analysis.py
# 演習第20回：学習済みモデルの行動診断スクリプト
#
# 【目的】
# 学習が横ばいになっている原因を特定するため、以下を可視化・分析します：
# 1. Hiderの行動パターン（移動、壁/箱との距離、停滞度）
# 2. エピソードの詳細（捕獲位置、時間推移）
# 3. 報酬の内訳（survival/distance/stagnation の貢献度）
#
# 【修正(v20.2)】
# - ヒートマップから「箱の初期位置」の描画を削除（ランダム配置のため、重ねると画面が埋まる問題を解消）。
# - 壁の定義を XML (main18) の内容に合わせて正確に修正（外壁4枚＋内壁4枚）。
# - ヒートマップの表示範囲を実際のフィールドサイズ (±6m) に修正。
#
# 【実行準備】
# uv add matplotlib seaborn

import os
import sys
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

# Matplotlib日本語対応 & スタイル
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
sns.set_style("whitegrid")

# ==========================================
# 設定
# ==========================================

# 分析対象のモデル
EXPERIMENT_NAME = "HideAndSeek_Transformer"  # main18で学習したモデル名

# 分析パラメータ
NUM_EPISODES = 50  # 分析するエピソード数
MAX_EPISODE_STEPS = 500  # エピソードの最大ステップ数
RENDER_MODE = None  # None or "human" (可視化する場合)

# 報酬パラメータ（main18と一致させる - 推定値または実績値）
REWARD_SURVIVAL = 0.2
REWARD_DISTANCE_COEFF = 0.08
PENALTY_STAGNATION = -0.5

# 環境パラメータ (XMLに基づく)
ARENA_SIZE = 12.0  # ±6.0m (floor size="6 6 0.1")

# 壁の定義: (x, y, width(x軸方向の全幅), height(y軸方向の全幅))
# XMLの geom size は "半幅" なので 2倍する
# box size="6.2 0.1 4.0" -> width=12.4, height=0.2
WALL_POSITIONS = [
    # 外壁
    (0, 6.1, 12.4, 0.2),  # wall_n
    (0, -6.1, 12.4, 0.2),  # wall_s
    (6.1, 0, 0.2, 12.0),  # wall_e
    (-6.1, 0, 0.2, 12.0),  # wall_w
    # 内壁 (Maze)
    # maze_w0: pos="3.0 1.5 0.5", size="1.5 0.2 0.5" -> w=3.0, h=0.4
    (3.0, 1.5, 3.0, 0.4),
    # maze_w1: pos="-3.0 -1.5 0.5", size="1.5 0.2 0.5" -> w=3.0, h=0.4
    (-3.0, -1.5, 3.0, 0.4),
    # maze_w2: pos="0 -3.0 0.5", size="0.2 1.5 0.5" -> w=0.4, h=3.0
    (0, -3.0, 0.4, 3.0),
    # maze_w3: pos="0 3.0 0.5", size="0.2 1.5 0.5" -> w=0.4, h=3.0
    (0, 3.0, 0.4, 3.0),
]

# ==========================================
# 環境のインポート（main18から）
# ==========================================
try:
    import main18_optimization
    from main18_optimization import Agent, HideAndSeekEnv
except ImportError:
    print("Error: main18_optimization.py が見つかりません。")
    print("同じディレクトリに配置してください。")
    sys.exit(1)

# ==========================================
# データ収集
# ==========================================


class BehaviorAnalyzer:
    """行動分析用のデータ収集クラス"""

    def __init__(self, model_path_hider, model_path_seeker=None):
        self.device = torch.device("cpu")

        # 環境作成
        self.env = HideAndSeekEnv(render_mode=RENDER_MODE)

        # モデルロード
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # Hiderモデル
        self.hider_model = Agent(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
        if os.path.exists(model_path_hider):
            checkpoint = torch.load(model_path_hider, map_location=self.device)
            self.hider_model.load_state_dict(checkpoint)
            print(f"✓ Loaded Hider model: {model_path_hider}")
        else:
            print(f"✗ Hider model not found: {model_path_hider}")
            print("  Will use random policy.")
        self.hider_model.eval()

        # Seekerモデル
        self.seeker_model = None
        if model_path_seeker and os.path.exists(model_path_seeker):
            self.seeker_model = Agent(obs_dim=obs_dim, action_dim=action_dim).to(self.device)
            checkpoint = torch.load(model_path_seeker, map_location=self.device)
            self.seeker_model.load_state_dict(checkpoint)
            self.seeker_model.eval()
            print(f"✓ Loaded Seeker model: {model_path_seeker}")
        else:
            print("  Using rule-based Seeker")

        self.episodes_data = []
        self.seq_len = main18_optimization.TRANSFORMER_SEQ_LEN

    def run_episode(self):
        """1エピソードを実行してデータを収集"""
        obs, _ = self.env.reset()

        obs_dim = obs.shape[0]
        hider_obs_history = np.zeros((self.seq_len, obs_dim), dtype=np.float32)

        episode_data = {
            "positions_hider1": [],
            "positions_hider2": [],
            "positions_seeker": [],
            "distances_to_seeker": [],
            "distances_to_walls": [],
            "distances_to_boxes": [],
            "velocities_hider1": [],
            "rewards_survival": [],
            "rewards_distance": [],
            "rewards_stagnation": [],
            "rewards_total": [],
            "actions_hider1": [],
            "captured": False,
            "capture_step": None,
            "episode_length": 0,
        }

        for step in range(MAX_EPISODE_STEPS):
            # Hider行動
            hider_obs_history = np.roll(hider_obs_history, -1, axis=0)
            hider_obs_history[-1] = obs
            obs_seq = torch.FloatTensor(hider_obs_history).unsqueeze(0).to(self.device)

            with torch.no_grad():
                action, _, _, _ = self.hider_model.get_action_and_value(obs_seq)
                hider_action = action.cpu().numpy()[0]

            # ステップ実行
            obs, reward, done, truncated, info = self.env.step(hider_action)

            # 位置情報
            pos_h1 = self.env.data.xpos[self.env.h1_body][:2]
            pos_h2 = self.env.data.xpos[self.env.h2_body][:2]
            pos_s = self.env.data.xpos[self.env.s0_body][:2]

            episode_data["positions_hider1"].append(pos_h1.copy())
            episode_data["positions_hider2"].append(pos_h2.copy())
            episode_data["positions_seeker"].append(pos_s.copy())

            # 距離
            dist_to_seeker = np.linalg.norm(pos_h1 - pos_s)
            episode_data["distances_to_seeker"].append(dist_to_seeker)

            dist_to_walls = self._calc_distance_to_walls(pos_h1)
            episode_data["distances_to_walls"].append(dist_to_walls)

            # 箱との距離（現在の箱位置を使用）
            box1_pos = self.env.data.xpos[self.env.box1_body][:2]
            box2_pos = self.env.data.xpos[self.env.box2_body][:2]
            dist_to_boxes = min(np.linalg.norm(pos_h1 - box1_pos), np.linalg.norm(pos_h1 - box2_pos))
            episode_data["distances_to_boxes"].append(dist_to_boxes)

            # 速度
            dof_h1 = self.env.model.jnt_dofadr[self.env.model.joint("h1_x").id]
            vel_h1 = self.env.data.qvel[dof_h1 : dof_h1 + 2]
            speed = np.linalg.norm(vel_h1)
            episode_data["velocities_hider1"].append(speed)

            # 報酬内訳（推定）
            r_surv = REWARD_SURVIVAL
            r_dist = min(dist_to_seeker, 13.0) * REWARD_DISTANCE_COEFF
            r_stag = PENALTY_STAGNATION if speed < 0.1 else 0.0

            episode_data["rewards_survival"].append(r_surv)
            episode_data["rewards_distance"].append(r_dist)
            episode_data["rewards_stagnation"].append(r_stag)
            episode_data["rewards_total"].append(r_surv + r_dist + r_stag)

            episode_data["actions_hider1"].append(hider_action.copy())

            if done:
                episode_data["captured"] = True
                episode_data["capture_step"] = step
                episode_data["episode_length"] = step + 1
                break
            if truncated:
                episode_data["episode_length"] = step + 1
                break

        return episode_data

    def _calc_distance_to_walls(self, pos):
        """壁までの最短距離を計算"""
        min_dist = float("inf")
        for wx, wy, w, h in WALL_POSITIONS:
            # 矩形との距離
            x_min, x_max = wx - w / 2, wx + w / 2
            y_min, y_max = wy - h / 2, wy + h / 2

            # 点posに最も近い矩形上の点
            closest_x = np.clip(pos[0], x_min, x_max)
            closest_y = np.clip(pos[1], y_min, y_max)

            dist = np.linalg.norm([pos[0] - closest_x, pos[1] - closest_y])
            min_dist = min(min_dist, dist)
        return min_dist

    def collect_data(self, num_episodes):
        print(f"\nCollecting data from {num_episodes} episodes...")
        for ep in range(num_episodes):
            episode_data = self.run_episode()
            self.episodes_data.append(episode_data)
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep+1}/{num_episodes} completed")
        return self.episodes_data


# ==========================================
# 分析・可視化
# ==========================================


class BehaviorVisualizer:
    def __init__(self, episodes_data, output_dir):
        self.episodes_data = episodes_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def generate_all_plots(self):
        print("\nGenerating analysis plots...")
        self.plot_trajectory_heatmap()
        self.plot_distance_evolution()
        self.plot_reward_breakdown()
        self.plot_capture_analysis()
        self.plot_velocity_distribution()
        self.plot_positional_strategy()
        self.generate_summary_statistics()
        print(f"\nAll plots saved to: {self.output_dir}")

    def plot_trajectory_heatmap(self):
        """Hiderの移動軌跡ヒートマップ"""
        fig, ax = plt.subplots(figsize=(10, 10))

        all_positions = []
        for ep_data in self.episodes_data:
            all_positions.extend(ep_data["positions_hider1"])
        all_positions = np.array(all_positions)

        # ヒートマップ
        heatmap, xedges, yedges = np.histogram2d(
            all_positions[:, 0],
            all_positions[:, 1],
            bins=60,
            range=[
                [-ARENA_SIZE / 2, ARENA_SIZE / 2],
                [-ARENA_SIZE / 2, ARENA_SIZE / 2],
            ],
        )
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(
            heatmap.T,
            extent=extent,
            origin="lower",
            cmap="hot",
            interpolation="bilinear",
        )
        plt.colorbar(im, ax=ax, label="Frequency")

        # 壁の描画 (青色枠)
        for wx, wy, w, h in WALL_POSITIONS:
            # 矩形の左下座標
            xy = (wx - w / 2, wy - h / 2)
            rect = patches.Rectangle(xy, w, h, linewidth=1.5, edgecolor="cyan", facecolor="none", alpha=0.8)
            ax.add_patch(rect)

        # 箱の位置は描画しない (ランダム配置のため、重ねると画面が埋まってしまう)

        ax.set_xlim(-ARENA_SIZE / 2, ARENA_SIZE / 2)
        ax.set_ylim(-ARENA_SIZE / 2, ARENA_SIZE / 2)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Hider Movement Heatmap (Fixed Walls shown)")
        ax.grid(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / "trajectory_heatmap.png", dpi=150)
        plt.close()
        print("  ✓ Trajectory heatmap saved")

    def plot_distance_evolution(self):
        """距離の時間推移"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Seekerとの距離
        ax = axes[0, 0]
        for ep_data in self.episodes_data[:10]:
            ax.plot(ep_data["distances_to_seeker"], alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Distance to Seeker")
        ax.set_title("Distance to Seeker (10 episodes)")
        ax.grid(True, alpha=0.3)

        # 壁との距離
        ax = axes[0, 1]
        for ep_data in self.episodes_data[:10]:
            ax.plot(ep_data["distances_to_walls"], alpha=0.5, linewidth=0.8)
        ax.set_title("Distance to Nearest Wall")
        ax.grid(True, alpha=0.3)

        # 箱との距離
        ax = axes[1, 0]
        for ep_data in self.episodes_data[:10]:
            ax.plot(ep_data["distances_to_boxes"], alpha=0.5, linewidth=0.8)
        ax.set_title("Distance to Nearest Box")
        ax.grid(True, alpha=0.3)

        # 平均距離分布
        ax = axes[1, 1]
        avg_dist = [np.mean(ep["distances_to_seeker"]) for ep in self.episodes_data]
        ax.hist(avg_dist, bins=20, alpha=0.7, edgecolor="black")
        ax.axvline(
            np.mean(avg_dist),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(avg_dist):.2f}",
        )
        ax.set_title("Avg Distance to Seeker Distribution")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "distance_evolution.png", dpi=150)
        plt.close()
        print("  ✓ Distance evolution plot saved")

    def plot_reward_breakdown(self):
        """報酬内訳"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        surv = [np.sum(ep["rewards_survival"]) for ep in self.episodes_data]
        dist = [np.sum(ep["rewards_distance"]) for ep in self.episodes_data]
        stag = [np.sum(ep["rewards_stagnation"]) for ep in self.episodes_data]

        # 積み上げ棒 (最初の20エピソード)
        ax = axes[0]
        idx = range(min(20, len(surv)))
        ax.bar(idx, surv[:20], label="Survival", alpha=0.7)
        ax.bar(idx, dist[:20], bottom=surv[:20], label="Distance", alpha=0.7)
        # Stagnationは負なので底を調整
        bottoms = np.array(surv[:20]) + np.array(dist[:20])
        ax.bar(idx, stag[:20], bottom=bottoms, label="Stagnation", alpha=0.7, color="red")

        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward per Episode")
        ax.set_title("Reward Components (First 20 Ep)")
        ax.legend()

        # 平均構成比（円グラフ）
        ax = axes[1]
        avg_surv = np.mean([np.mean(ep["rewards_survival"]) for ep in self.episodes_data])
        avg_dist = np.mean([np.mean(ep["rewards_distance"]) for ep in self.episodes_data])
        avg_stag_abs = abs(np.mean([np.mean(ep["rewards_stagnation"]) for ep in self.episodes_data]))

        ax.pie(
            [avg_surv, avg_dist, avg_stag_abs],
            labels=["Survival", "Distance", "Stagnation(abs)"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["lightblue", "lightgreen", "lightcoral"],
        )
        ax.set_title("Average Reward Composition")

        plt.tight_layout()
        plt.savefig(self.output_dir / "reward_breakdown.png", dpi=150)
        plt.close()
        print("  ✓ Reward breakdown plot saved")

    def plot_capture_analysis(self):
        """捕獲位置とタイミング"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 捕獲位置
        ax = axes[0]
        caps = []
        for ep in self.episodes_data:
            if ep["captured"]:
                caps.append(ep["positions_hider1"][ep["capture_step"]])

        if caps:
            caps = np.array(caps)
            # 壁描画
            for wx, wy, w, h in WALL_POSITIONS:
                ax.add_patch(patches.Rectangle((wx - w / 2, wy - h / 2), w, h, fill=False, edgecolor="cyan"))
            # 捕獲点
            ax.scatter(caps[:, 0], caps[:, 1], c="red", alpha=0.6, marker="x", label="Capture")
            ax.set_xlim(-ARENA_SIZE / 2, ARENA_SIZE / 2)
            ax.set_ylim(-ARENA_SIZE / 2, ARENA_SIZE / 2)
            ax.set_title(f"Capture Locations (n={len(caps)})")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No Captures", ha="center", va="center")

        # エピソード長分布
        ax = axes[1]
        lens = [ep["episode_length"] for ep in self.episodes_data]
        ax.hist(lens, bins=20, color="green", alpha=0.7)
        ax.set_title("Episode Length Distribution")
        ax.set_xlabel("Steps")

        plt.tight_layout()
        plt.savefig(self.output_dir / "capture_analysis.png", dpi=150)
        plt.close()
        print("  ✓ Capture analysis plot saved")

    def plot_velocity_distribution(self):
        """速度分布"""
        fig, ax = plt.subplots(figsize=(8, 5))
        vels = []
        for ep in self.episodes_data:
            vels.extend(ep["velocities_hider1"])

        ax.hist(vels, bins=50, alpha=0.7, color="purple", density=True)
        ax.axvline(0.1, color="red", linestyle="--", label="Stagnation Threshold (0.1)")
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Density")
        ax.set_title("Hider Velocity Distribution")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "velocity_distribution.png", dpi=150)
        plt.close()
        print("  ✓ Velocity distribution plot saved")

    def plot_positional_strategy(self):
        """壁・箱との距離戦略"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 壁
        wall_dists = [np.mean(ep["distances_to_walls"]) for ep in self.episodes_data]
        axes[0].hist(wall_dists, bins=20, color="skyblue", alpha=0.7)
        axes[0].set_title("Avg Distance to Walls")
        axes[0].set_xlabel("Distance (m)")

        # 箱
        box_dists = [np.mean(ep["distances_to_boxes"]) for ep in self.episodes_data]
        axes[1].hist(box_dists, bins=20, color="lightgreen", alpha=0.7)
        axes[1].set_title("Avg Distance to Boxes")
        axes[1].set_xlabel("Distance (m)")

        plt.tight_layout()
        plt.savefig(self.output_dir / "positional_strategy.png", dpi=150)
        plt.close()
        print("  ✓ Positional strategy plot saved")

    def generate_summary_statistics(self):
        """数値統計保存"""
        path = self.output_dir / "summary_statistics.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Experiment: {EXPERIMENT_NAME}¥n")
            f.write(f"Episodes: {len(self.episodes_data)}¥n¥n")

            # Capture Rate
            n_cap = sum(1 for ep in self.episodes_data if ep["captured"])
            f.write(f"Capture Rate: {n_cap/len(self.episodes_data)*100:.1f}% ({n_cap}/{len(self.episodes_data)})¥n")

            # Ep Len
            lens = [ep["episode_length"] for ep in self.episodes_data]
            f.write(f"Avg Ep Length: {np.mean(lens):.1f} ± {np.std(lens):.1f}¥n")

            # Rewards
            tot = [np.sum(ep["rewards_total"]) for ep in self.episodes_data]
            f.write(f"Avg Total Reward: {np.mean(tot):.2f} ± {np.std(tot):.2f}¥n")

            # Stagnation Rate
            stag_rates = []
            for ep in self.episodes_data:
                v = np.array(ep["velocities_hider1"])
                stag_rates.append(np.mean(v < 0.1) * 100)
            f.write(f"Avg Stagnation Rate: {np.mean(stag_rates):.1f}%¥n")

        print("  ✓ Summary statistics saved")


# ==========================================
# メイン処理
# ==========================================


def main():
    print("=" * 70)
    print("Behavior Analysis for HideAndSeek Trained Models (v20.2)")
    print("=" * 70)

    model_path_hider = f"{EXPERIMENT_NAME}_HIDER.pt"
    model_path_seeker = f"{EXPERIMENT_NAME}_SEEKER.pt"

    if not os.path.exists(model_path_hider):
        print(f"\n⚠️  Warning: Hider model not found: {model_path_hider}")
        print("Will analyze with random policy.\n")

    analyzer = BehaviorAnalyzer(model_path_hider, None)  # SeekerはNoneでルールベース
    episodes_data = analyzer.collect_data(NUM_EPISODES)

    visualizer = BehaviorVisualizer(episodes_data, "analysis_results")
    visualizer.generate_all_plots()

    print("\nAnalysis Complete!")


if __name__ == "__main__":
    main()
