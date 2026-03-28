import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# =====================================
# 設定
# =====================================
CSV_PATH = "./optuna_results_layer0/results_refinement.csv"
FIG_DIR = "figs_full_analysis_v2"
os.makedirs(FIG_DIR, exist_ok=True)

TARGET_COL = "value"

# 元パラメータ（主役）
BASE_PARAMS = [
    "params_ENT_COEF",
    "params_LEARNING_RATE",
    "params_PENALTY_STAGNATION_FORCE",
    "params_REWARD_DISTANCE_DIFF_SCALE",
    "params_REWARD_SURVIVAL_SCALE",
]

# log パラメータ対応表（あるものだけ）
LOG_PARAM_MAP = {
    "params_ENT_COEF": "log_ENT_COEF",
    "params_LEARNING_RATE": "log_LEARNING_RATE",
}

RANDOM_STATE = 0

# =====================================
# 1. データ読み込み
# =====================================
df = pd.read_csv(CSV_PATH)

X = df[BASE_PARAMS]
y = df[TARGET_COL]

# =====================================
# 2. Trial vs value
# =====================================
plt.figure()
plt.plot(y.values)
plt.xlabel("Trial")
plt.ylabel("Objective value")
plt.title("Trial vs Objective value")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_trial_vs_value.png", dpi=150)
plt.close()

# =====================================
# 3. value 分布
# =====================================
plt.figure()
plt.hist(y, bins=30)
plt.xlabel("Objective value")
plt.ylabel("Frequency")
plt.title("Distribution of objective value")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_value_hist.png", dpi=150)
plt.close()

# =====================================
# 4. 線形回帰（標準化係数）
# =====================================
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

lr = LinearRegression()
lr.fit(X_std, y)

coef = pd.Series(lr.coef_, index=BASE_PARAMS)
coef = coef.reindex(coef.abs().sort_values(ascending=False).index)

plt.figure(figsize=(8, 4))
plt.bar(coef.index, coef.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("標準化係数")
plt.title("線形回帰（標準化係数）")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_linear_regression_coeffs.png", dpi=150)
plt.close()

coef.to_csv(f"{FIG_DIR}/linear_regression_coeffs.csv")

# =====================================
# 5. 非線形モデル（RandomForest）
# =====================================
rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X, y)

# =====================================
# 6. PDP（5 元パラメータすべて）
# =====================================
for p in BASE_PARAMS:
    print(f"PDP 作成中: {p}")

    # PDP 用の特徴量
    pdp = partial_dependence(rf, X, features=[p], grid_resolution=50)

    x = pdp["grid_values"][0]
    y_pdp = pdp["average"][0]

    import matplotlib.ticker as mticker

    plt.figure(figsize=(6, 4))

    if p in LOG_PARAM_MAP:
        # x はすでに実値
        plt.plot(x, y_pdp)  # , marker=".")
        plt.xscale("log")

        ax = plt.gca()
        ax.xaxis.set_minor_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1e}"))
        ax.tick_params(axis="x", which="minor", labelsize=8)

        ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=(2, 5)))  # ← ここが刻み
        # ax.xaxis.set_minor_locator(
        #    mticker.LogLocator(base=10, subs=(2, 3, 4, 5, 6, 7, 8, 9))
        # )
        # ax.xaxis.set_minor_formatter(mticker.LogFormatterSciNotation())
        # ax.xaxis.set_minor_locator(mticker.NullLocator())

        plt.xlabel(p.replace("params_", "").lower())

    else:
        plt.plot(x, y_pdp)  # , marker="o")
        plt.xlabel(p.replace("params_", "").lower())

    plt.ylabel("予測 value（部分依存）")
    plt.title(f"PDP of {p}")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_pdp_{p}.png", dpi=150)
    plt.close()

print("全パラメータの解析が完了しました。")
