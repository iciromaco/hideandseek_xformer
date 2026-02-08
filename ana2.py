# ana2_pdp_log.py
# PDP描画用スクリプト（log版モデル読み込みバージョン）
# ana1.py で作成した gp_model_log.pkl を使用します

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

# ────────────────────────────────────────────────
# 1. データ読み込み（X_log を再現するため）
# ────────────────────────────────────────────────
df = pd.read_csv('./optuna_results_layer0/results_refinement.csv')
df = df[df['state'] == 'COMPLETE'].copy()

# log変換した特徴量を作成（訓練時と同じ処理）
X_log = np.column_stack([
    np.log10(df['params_ENT_COEF'] + 1e-10),
    np.log10(df['params_LEARNING_RATE'] + 1e-10),
    df['params_PENALTY_STAGNATION_FORCE'],
    df['params_REWARD_DISTANCE_DIFF_SCALE'],
    df['params_REWARD_SURVIVAL_SCALE']
])

y = df['value'].values  # 参考用（今回は使わないが念のため）

feature_names_log = [
    'log10_ENT_COEF',
    'log10_LEARNING_RATE',
    'PENALTY_STAGNATION_FORCE',
    'REWARD_DISTANCE_DIFF_SCALE',
    'REWARD_SURVIVAL_SCALE'
]

# ────────────────────────────────────────────────
# 2. 訓練済み log版GPモデルを読み込み
# ────────────────────────────────────────────────
pkl_path = './optuna_results_layer0/gp_model_log.pkl'

with open(pkl_path, 'rb') as f:
    gp_log = pickle.load(f)

print("読み込んだモデル:")
print("Kernel:", gp_log.kernel_)
print("Log marginal likelihood:", gp_log.log_marginal_likelihood(gp_log.kernel_.theta))

# ────────────────────────────────────────────────
# 3. Partial Dependence Plot を描画
# ────────────────────────────────────────────────
fig, ax = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Partial Dependence Plots  (log版モデル・実際のvalueスケール)', 
             fontsize=16, y=0.98)

# 5つの特徴量すべてを描画
PartialDependenceDisplay.from_estimator(
    gp_log,
    X_log,
    features=range(5),
    feature_names=feature_names_log,
    kind='average',               # 平均的な部分依存
    n_jobs=-1,
    grid_resolution=60,           # 少し細かく
    percentiles=(0.05, 0.95),     # 極端な外れ値を少しカット
    ax=ax.ravel()[:5]
)

# 使わない6番目のaxesを非表示
if len(ax.ravel()) > 5:
    ax.ravel()[5].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('./optuna_results_layer0/pdp_log_version.png', dpi=300, bbox_inches='tight')
plt.show()

print("PDP画像を保存しました: ./optuna_results_layer0/pdp_log_version.png")