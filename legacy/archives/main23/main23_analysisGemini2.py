# main23_analysis.py
# 演習第23回：報酬パラメータと隠蔽性能の多角的回帰分析 (v25.01)
#
# 【修正内容】
# 1. カラム名の完全整合: main23_optuna_search.py の提案名（小文字）に合わせ param_map を修正。
# 2. パス指定の柔軟化: CSVがカレントディレクトリにある場合に対応。
# 3. データのクレンジング: Optunaの試行状態が 'COMPLETE' のもののみを抽出。
# 4. ログスケール変換の堅牢化: 1e-10 を加算して log10(0) エラーを回避。

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression

# 機械学習ライブラリ
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 実験設定
# ==========================================
# ★実行モード設定: 探索スクリプトの MODE と一致させてください
MODE = "initial"  # "initial" または "refinement"

# main23_optuna_search.py の出力先に合わせる
# カレントディレクトリにCSVがある場合は "." を指定
RESULTS_DIR = Path(".")
CSV_PATH = RESULTS_DIR / f"results_{MODE}.csv"
OUTPUT_DIR = Path(f"./analysis_reports_{MODE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze():
    print(f"--- Multi-Model Analysis Phase: {MODE} ---")

    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} が見つかりません。")
        print("先に main23_optuna_search.py を実行して結果を生成してください。")
        return

    # 1. データの読み込み
    df_raw = pd.read_csv(CSV_PATH)

    # 状態が COMPLETE のものだけを抽出（重要：失敗した試行を除外）
    if "state" in df_raw.columns:
        df = df_raw[df_raw["state"] == "COMPLETE"].copy()
    else:
        df = df_raw.copy()

    # OptunaのCSV列名（params_...）を分析用に変換
    # main23_optuna_search.py の trial.suggest_... の名前に厳密に合わせる
    param_map = {
        "params_ent_coef": "Entropy",
        "params_learning_rate": "LR",
        "params_reward_hidden_bonus": "Hidden_Bonus",
        "params_cos_penalty_scale": "Cos_Penalty",
        "value": "Score",
    }

    # 存在するカラムだけをリネーム
    actual_map = {k: v for k, v in param_map.items() if k in df.columns}
    df = df.rename(columns=actual_map)
    df = df.dropna(subset=["Score"])

    if len(df) < 5:
        print(f"データ数が少なすぎるため（現在 {len(df)} 件）、分析を中断します。")
        print("少なくとも 5〜10 回以上の COMPLETE な試行が必要です。")
        return

    print(f"Loaded {len(df)} trials. Best Score: {df['Score'].max():.1f}")

    # 2. 特徴量作成（LR, Entropy は対数スケールに変換）
    # 実際に存在する特徴量のみを使用
    features = [v for k, v in actual_map.items() if v != "Score"]
    X = df[features].copy()

    if "LR" in X.columns:
        X["LR"] = np.log10(X["LR"] + 1e-10)
    if "Entropy" in X.columns:
        X["Entropy"] = np.log10(X["Entropy"] + 1e-10)

    y = df["Score"].values

    # 3. モデルの定義
    models = {
        "LR": Pipeline([("s", StandardScaler()), ("r", LinearRegression())]),
        "GBR": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "GPR": GaussianProcessRegressor(
            kernel=C(1.0, (1e-3, 1e8)) * RBF(length_scale=[1.0] * len(features), length_scale_bounds=(1e-2, 1e8)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-8, 1e5)),
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=42,
        ),
    }

    # 全モデルのレポート内容を蓄積するリスト
    summary_reports = []

    # 4. 各モデルの実行とPDPの描画
    for name, model in models.items():
        print(f"  Training and Analyzing model: {name}...")
        try:
            model.fit(X.values, y)
        except Exception as e:
            print(f"    Model {name} fitting failed: {e}")
            continue

        # タイル状プロット (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        report_lines = []
        report_lines.append(f"\n{'='*75}")
        report_lines.append(f"{f' ANALYSIS REPORT ({name} Model) ':^75}")
        report_lines.append(f"{'='*75}")
        report_lines.append(f"{'Param Name':<15} | {'Best Value':<15} | {'Impact (Steps)':<15}")
        report_lines.append(f"{'-'*15}-|-{'-'*15}-|-{'-'*15}")

        for i, f_name in enumerate(features):
            if i >= len(axes):
                break
            ax = axes[i]

            # 部分依存性 (PDP) の計算
            pdp_out = partial_dependence(model, X.values, [i], kind="average", grid_resolution=100)
            x_raw = pdp_out["grid_values"][0]
            y_pdp = pdp_out["average"][0]

            # 対数を元のスケールに戻す
            is_log = f_name in ["LR", "Entropy"]
            x_plot = 10**x_raw if is_log else x_raw

            # ピーク値（理論上の最適解）の特定
            best_idx = np.argmax(y_pdp)
            best_x = x_plot[best_idx]
            best_y = y_pdp[best_idx]
            impact = np.max(y_pdp) - np.min(y_pdp)

            # 表への追加
            label_fmt = f"{best_x:.2e}" if is_log else f"{best_x:.3f}"
            report_lines.append(f"{f_name:<15} | {label_fmt:<15} | {impact:15.2f}")

            # グラフ描画
            ax.plot(x_plot, y_pdp, color="tab:blue", linewidth=2.5, label="Estimated Trend")
            ax.fill_between(x_plot, y_pdp - 1.0, y_pdp + 1.0, color="tab:blue", alpha=0.1)

            # 最大値のマーキング
            ax.scatter([best_x], [best_y], color="red", s=80, zorder=5, edgecolors="white")
            ax.annotate(
                f"Best: {label_fmt}\n({best_y:.1f} steps)",
                xy=(best_x, best_y),
                xytext=(10, 10),
                textcoords="offset points",
                color="red",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.7),
            )

            ax.set_title(f"{f_name} (via {name})", fontsize=13, fontweight="bold")
            ax.set_ylabel("Estimated Hidden Steps")
            ax.grid(True, which="both", ls="-", alpha=0.4)
            if is_log:
                ax.set_xscale("log")

        plt.suptitle(
            f"Partial Dependence Plots: {name} Regression ({MODE})",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = OUTPUT_DIR / f"pdp_1d_{name}_{MODE}.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        # レポートの結合
        full_report = "\n".join(report_lines)
        print(full_report)
        summary_reports.append(full_report)

    # 5. 全モデルの統合テキストレポート保存
    summary_file = OUTPUT_DIR / f"summary_report_{MODE}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_reports))

    # 6. 相関行列の保存
    plt.figure(figsize=(10, 8))
    cols_to_corr = [v for v in actual_map.values() if v in df.columns]
    sns.heatmap(df[cols_to_corr].corr(), annot=True, cmap="RdBu_r", center=0)
    plt.title(f"Parameter Correlation ({MODE})")
    corr_path = OUTPUT_DIR / f"correlation_{MODE}.png"
    plt.savefig(corr_path)
    plt.close()

    print(f"\n[Done] すべての分析結果が {OUTPUT_DIR} に保存されました。")
    print(f"テキスト形式のサマリー: {summary_file.name}")


if __name__ == "__main__":
    analyze()
