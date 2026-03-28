# main22_analysis.py
# 最適化結果の多角分析 - 線形回帰、勾配ブースティング、ガウス過程回帰の一括実行
#
# 【修正内容】
# 1. 異常行の特定と報告: 数値変換に失敗した行や NaN を含む行の「試行番号」と「原因カラム」を表示。
# 2. 強制的数値変換: pd.to_numeric(errors='coerce') により、異常値を NaN に置換。
# 3. 欠損値除外の徹底: 異常報告後に該当行を削除し、分析計算の安定性を確保。

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression

# モデル関連のライブラリ
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ==========================================
# 設定
# ==========================================
CSV_PATH = "./optuna_results_layer0/results_refinement.csv"
OUTPUT_DIR = "./optuna_results_layer0/analysis_output_all"

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams["font.family"] = "sans-serif"

# ==========================================
# 1. データ読み込みと異常行の特定
# ==========================================
print(f"Loading data from {CSV_PATH}...")
if not os.path.exists(CSV_PATH):
    print(f"Error: {CSV_PATH} が見つかりません。")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

# state列のクリーニング
if "state" in df.columns:
    df["state"] = df["state"].astype(str).str.strip()

# 分析対象カラムの定義
param_cols = [
    "params_ENT_COEF",
    "params_LEARNING_RATE",
    "params_PENALTY_STAGNATION_FORCE",
    "params_REWARD_DISTANCE_DIFF_SCALE",
    "params_REWARD_SURVIVAL_SCALE",
]
required_cols = param_cols + ["value"]

# --- 異常行の特定セクション ---
print("Checking for invalid (NaN) or non-numeric rows...")
df_complete = df[df["state"] == "COMPLETE"].copy()

# 各行について、数値変換に失敗する箇所があるかチェック
invalid_rows_info = []

for idx, row in df_complete.iterrows():
    bad_columns = []
    trial_num = row["number"] if "number" in df.columns else idx

    for col in required_cols:
        val = row[col]
        # 数値に変換を試みる
        num_val = pd.to_numeric(val, errors="coerce")
        if pd.isna(num_val):
            bad_columns.append(f"{col}(val='{val}')")

    if bad_columns:
        invalid_rows_info.append(f"Trial {trial_num}: " + ", ".join(bad_columns))

# 異常があった場合に表示
if invalid_rows_info:
    print("-" * 50)
    print(f"FOUND {len(invalid_rows_info)} INVALID ROWS:")
    for info in invalid_rows_info:
        print(f"  [X] {info}")
    print("-" * 50)
else:
    print("  -> No problematic rows found in 'COMPLETE' trials.")

# --- データのクレンジング実行 ---
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[df["state"] == "COMPLETE"].dropna(subset=required_cols).copy()

if len(df) == 0:
    print("Error: 有効なデータが1件もありません。")
    sys.exit(1)

print(f"Total valid trials for analysis: {len(df)}")

# ==========================================
# 2. 前処理・特徴量作成
# ==========================================
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
y_mean_global = np.mean(y_train)

features_info = [
    {"name": "ENT_COEF", "is_log": True},
    {"name": "LEARNING_RATE", "is_log": True},
    {"name": "PENALTY_STAGNATION", "is_log": False},
    {"name": "REWARD_DIST_SCALE", "is_log": False},
    {"name": "REWARD_SURVIVAL", "is_log": False},
]


# ==========================================
# 3. 共通分析・描画関数
# ==========================================
def analyze_and_save(model, model_name, file_suffix, importance_scores=None, importance_label="Imp"):
    print(f"Analyzing {model_name}...")

    report_lines = []
    report_lines.append(f"{'='*80}")
    report_lines.append(f"{f'PARAMETER ANALYSIS REPORT ({model_name})':^80}")
    report_lines.append(f"{'='*80}")
    report_lines.append(f"{'Param Name':<25} | {f'{importance_label}':<10} | {'Impact(Len)':<12} | {'Best Value':<15}")
    report_lines.append("-" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    target_idx = 0

    for i, info in enumerate(features_info):
        name = info["name"]
        is_log = info["is_log"]
        ax = axes[i]

        pdp_out = partial_dependence(model, X_train, [i], kind="average", grid_resolution=100)

        if "grid_values" in pdp_out:
            x_vals = pdp_out["grid_values"][target_idx]
        else:
            x_vals = pdp_out["values"][target_idx]
        y_vals = pdp_out["average"][target_idx]

        if np.mean(y_vals) < y_mean_global * 0.5:
            y_vals = y_vals + y_mean_global

        impact = np.max(y_vals) - np.min(y_vals)
        best_idx = np.argmax(y_vals)
        best_val_log = x_vals[best_idx]

        if is_log:
            best_val_display = 10**best_val_log
            best_val_str = f"{best_val_display:.2e}"
            plot_x = 10**x_vals
        else:
            best_val_display = best_val_log
            best_val_str = f"{best_val_display:.4f}"
            plot_x = x_vals

        imp_val = importance_scores[i] if importance_scores is not None else 0.0
        report_lines.append(f"{name:<25} | {imp_val:10.1f} | {impact:12.2f} | {best_val_str:>15}")

        ax.plot(plot_x, y_vals, color="tab:blue", linewidth=2)
        ax.fill_between(plot_x, y_vals - 1.0, y_vals + 1.0, color="tab:blue", alpha=0.1)
        ax.scatter(
            [best_val_display],
            [np.max(y_vals)],
            color="red",
            zorder=5,
            label=f"Best: {best_val_str}",
        )
        ax.set_title(f"{name}\n(Impact: {impact:.2f})", fontsize=12, fontweight="bold")
        ax.set_ylabel("Expected EpLen")
        ax.grid(True, which="both", ls="-", alpha=0.5)
        if is_log:
            ax.set_xscale("log")
        ax.legend(loc="best")

    for j in range(len(features_info), len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"Partial Dependence Plots ({model_name})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"pdp_analysis_chart_{file_suffix}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    report_text = "\n".join(report_lines)
    print(report_text)
    with open(
        os.path.join(OUTPUT_DIR, f"analysis_report_{file_suffix}.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report_text)


# ==========================================
# 4. モデル実行
# ==========================================
# A. 線形回帰
lr_p = Pipeline([("s", StandardScaler()), ("r", LinearRegression())])
lr_p.fit(X_train, y_train)
lr_imp = np.abs(lr_p.named_steps["r"].coef_)
analyze_and_save(lr_p, "Linear Regression", "LR", lr_imp / np.max(lr_imp) * 100, "Coef")

# B. GBR
gbr = GradientBoostingRegressor(n_estimators=300, random_state=42)
gbr.fit(X_train, y_train)
analyze_and_save(
    gbr,
    "Gradient Boosting",
    "GBR",
    gbr.feature_importances_ / np.max(gbr.feature_importances_) * 100,
    "Imp",
)

# C. GPR
kernel = C(1.0, (1e-3, 1e8)) * Matern(length_scale=[1.0] * 5, nu=2.5) + WhiteKernel(noise_level=0.1)
gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=15, random_state=42)
gp.fit(X_train, y_train)
gp_imp = 1.0 / gp.kernel_.k1.k2.length_scale
analyze_and_save(gp, "Gaussian Process", "GPR", gp_imp / np.max(gp_imp) * 100, "Sens(ARD)")

print(f"\nAnalysis complete. Results in: {OUTPUT_DIR}")
