# 演習第21回：最適化結果の分析スクリプト main21_results_analysis.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Hiragino Sans"
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay

# =========================
# 設定
# =========================
CSV_PATH = "./optuna_results_layer0/results_refinement.csv"
FIG_DIR = "figures"
SHOW_FIG = True   # True にすると画面表示される

os.makedirs(FIG_DIR, exist_ok=True)

# =========================
# 1. データ読み込み
# =========================
df = pd.read_csv(CSV_PATH)

# =========================
# 2. 使用する列
# =========================
目的変数 = "value"

元パラメータ = [
    "params_ENT_COEF",
    "params_LEARNING_RATE",
    "params_PENALTY_STAGNATION_FORCE",
    "params_REWARD_DISTANCE_DIFF_SCALE",
    "params_REWARD_SURVIVAL_SCALE",
]

df = df[[目的変数] + 元パラメータ].dropna()

# =========================
# 3. 特徴量変換
# =========================
df["log_ENT_COEF"] = np.log(df["params_ENT_COEF"])
df["log_LEARNING_RATE"] = np.log(df["params_LEARNING_RATE"])

説明変数 = [
    "log_ENT_COEF",
    "log_LEARNING_RATE",
    "params_PENALTY_STAGNATION_FORCE",
    "params_REWARD_DISTANCE_DIFF_SCALE",
    "params_REWARD_SURVIVAL_SCALE",
]

X = df[説明変数]
y = df[目的変数]

# =========================
# 4. 線形回帰（全体傾向）
# =========================
lin_model = Pipeline(
    [
        ("標準化", StandardScaler()),
        ("回帰", LinearRegression()),
    ]
)

lin_model.fit(X, y)

係数 = lin_model.named_steps["回帰"].coef_

coef_df = (
    pd.DataFrame({"変数": 説明変数, "係数": 係数})
    .sort_values("係数")
)

plt.figure()
plt.barh(coef_df["変数"], coef_df["係数"])
plt.title("線形回帰の係数（標準化後）")
plt.xlabel("係数")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/linear_regression_coefficients.png", dpi=150)
if SHOW_FIG:
    plt.show()
plt.close()

# =========================
# 5. 勾配ブースティング回帰
# =========================
gbr = GradientBoostingRegressor(
    random_state=0,
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
)

gbr.fit(X, y)

# ---- 特徴量重要度
importance_df = (
    pd.DataFrame(
        {
            "変数": 説明変数,
            "重要度": gbr.feature_importances_,
        }
    )
    .sort_values("重要度")
)

plt.figure()
plt.barh(importance_df["変数"], importance_df["重要度"])
plt.title("特徴量重要度（勾配ブースティング）")
plt.xlabel("重要度")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/gbr_feature_importance.png", dpi=150)
if SHOW_FIG:
    plt.show()
plt.close()

# =========================
# 6. 部分依存プロット
# =========================
for var in 説明変数:
    fig, ax = plt.subplots()
    PartialDependenceDisplay.from_estimator(
        gbr,
        X,
        [var],
        grid_resolution=50,
        ax=ax,
    )
    ax.set_title(f"{var} の部分依存")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/pdp_{var}.png", dpi=150)
    if SHOW_FIG:
        plt.show()
    plt.close(fig)
