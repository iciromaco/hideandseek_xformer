# 最適化結果の分析 - 形回帰と勾配ブースティング main22_analysisLRPlus.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline  # make_pipeline から変更
from sklearn.preprocessing import StandardScaler

# ==========================================
# 設定
# ==========================================
CSV_PATH = "./optuna_results_layer0/results_refinement.csv"
OUTPUT_DIR = "./optuna_results_layer0/analysis_output_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams["font.family"] = "sans-serif"

# ==========================================
# 1. データ読み込みと前処理
# ==========================================
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df = df[df["state"] == "COMPLETE"].copy()

X_train = np.column_stack(
    [
        np.log10(df["params_ENT_COEF"] + 1e-10),
        np.log10(df["params_LEARNING_RATE"] + 1e-10),
        df["params_PENALTY_STAGNATION_FORCE"],
        df["params_REWARD_DISTANCE_DIFF_SCALE"],
        df["params_REWARD_SURVIVAL_SCALE"],
    ]
)

y_train = df["value"].values

features_info = [
    {"name": "ENT_COEF", "is_log": True},
    {"name": "LEARNING_RATE", "is_log": True},
    {"name": "PENALTY_STAGNATION", "is_log": False},
    {"name": "REWARD_DIST_SCALE", "is_log": False},
    {"name": "REWARD_SURVIVAL", "is_log": False},
]


# ==========================================
# 共通描画・分析関数の定義
# ==========================================
def run_analysis_and_plot(model, model_name, filename_suffix):
    """
    指定されたモデルでPDPを計算し、GP版と同じスタイルでプロットとレポートを作成する関数
    """
    print(f"Analyzing {model_name}...")

    report_lines = []
    report_lines.append(f"{'='*80}")
    report_lines.append(f"{f'PARAMETER ANALYSIS REPORT ({model_name})':^80}")
    report_lines.append(f"{'='*80}")
    importance_label = "Coef" if "Linear" in model_name else "Imp"
    report_lines.append(f"{'Param Name':<25} | {f'{importance_label}':<10} | {'Impact(Len)':<12} | {'Best Value':<15}")
    report_lines.append("-" * 80)

    # 特徴量重要度の取得
    importances = np.zeros(5)

    # モデルの種類に応じて係数または重要度を取得
    if "Linear" in model_name:
        # パイプラインの場合、'regressor' という名前のステップから係数を取得
        if hasattr(model, "named_steps"):
            # 修正箇所: パイプライン構築時に 'regressor' と命名しているためここで取得可能
            coeffs = model.named_steps["regressor"].coef_
        else:
            coeffs = model.coef_
        importances = np.abs(coeffs)
        importances = importances / np.max(importances) * 100

    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importances = importances / np.max(importances) * 100

    # グラフ準備
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    # GP版と同様にリスト抽出用のインデックス
    target_idx = 0

    for i, info in enumerate(features_info):
        name = info["name"]
        is_log = info["is_log"]
        ax = axes[i]

        # --- PDP計算 ---
        pdp_out = partial_dependence(model, X_train, [i], kind="average", grid_resolution=100)

        if "grid_values" in pdp_out:
            x_vals = pdp_out["grid_values"][target_idx]
        else:
            x_vals = pdp_out["values"][target_idx]

        y_vals = pdp_out["average"][target_idx]

        # --- 数値分析 ---
        impact = np.max(y_vals) - np.min(y_vals)
        best_idx = np.argmax(y_vals)
        best_val_log = x_vals[best_idx]

        if is_log:
            best_val_display = 10**best_val_log
            best_val_str = f"{best_val_display:.2e}"
        else:
            best_val_display = best_val_log
            best_val_str = f"{best_val_display:.4f}"

        imp_str = f"{importances[i]:5.1f}"
        impact_str = f"{impact:5.2f}"

        report_lines.append(f"{name:<25} | {imp_str:>10} | {impact_str:>12} | {best_val_str:>15}")

        # --- グラフ描画 ---
        if is_log:
            plot_x = 10**x_vals
        else:
            plot_x = x_vals

        ax.plot(plot_x, y_vals, color="tab:blue", linewidth=2, label="Mean EpLen")
        ax.fill_between(plot_x, y_vals - 1.0, y_vals + 1.0, color="tab:blue", alpha=0.1)

        ax.scatter(
            [best_val_display],
            [np.max(y_vals)],
            color="red",
            zorder=5,
            label=f"Best: {best_val_str}",
        )

        ax.set_title(f"{name}\n(Impact: {impact:.2f})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Expected Episode Length")

        ax.grid(True, which="both", ls="-", alpha=0.5)

        if is_log:
            ax.set_xscale("log")
            ax.set_xlabel("Value (Log Scale)")
        else:
            ax.set_xlabel("Value (Linear Scale)")

        ax.legend(loc="best")

    for j in range(len(features_info), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Partial Dependence Plots ({model_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plot_filename = f"pdp_analysis_chart_{filename_suffix}.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    report_lines.append("-" * 80)
    report_lines.append("\n【用語解説】")
    if "Linear" in model_name:
        report_lines.append("Coef       : 標準化係数の絶対値(相対評価)。線形モデルにおける寄与の大きさ。")
    else:
        report_lines.append("Imp        : 特徴量重要度(Feature Importance)。決定木における分岐への寄与度。")
    report_lines.append("Impact(Len): 影響度。パラメータ変化によるエピソード長の変動最大幅。")
    report_lines.append("Best Value : モデルが予測する最良パラメータ値。")

    report_text = "\n".join(report_lines)
    print(report_text)

    report_filename = f"analysis_report_{filename_suffix}.txt"
    report_path = os.path.join(OUTPUT_DIR, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n[Saved] {plot_filename} and {report_filename}")


# ==========================================
# 2. モデル構築と実行
# ==========================================

# --- A. 線形回帰 (Linear Regression) ---
# 修正点: make_pipeline ではなく Pipeline を使い、ステップ名を明示的に 'regressor' にする
lr_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),  # ここで名前を指定
    ]
)
lr_pipeline.fit(X_train, y_train)
run_analysis_and_plot(lr_pipeline, "Linear Regression", "LR")


# --- B. 勾配ブースティング (Gradient Boosting) ---
gbr_model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
gbr_model.fit(X_train, y_train)
run_analysis_and_plot(gbr_model, "Gradient Boosting", "GBR")

print(f"\nDone. Results are in {OUTPUT_DIR}")
