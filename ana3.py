# ana3.py
# 次の提案点を自動生成するコード（Expected Improvement使用）
import pandas as pd
import numpy as np
import pickle
from scipy.stats import norm

# モデル読み込み
with open('./optuna_results_layer0/gp_model.pkl', 'rb') as f:
    gp = pickle.load(f)

# 探索範囲（前回の分析から少し広めに設定）
bounds = np.array([
    [1e-6, 1e-3],      # ENT_COEF
    [1e-5, 1e-3],      # LEARNING_RATE
    [-5.0, 0.0],       # PENALTY_STAGNATION_FORCE
    [0.0, 10.0],       # REWARD_DISTANCE_DIFF_SCALE
    [0.0, 10.0]        # REWARD_SURVIVAL_SCALE
])

# 候補点をランダムに10,000個生成
np.random.seed(42)
n_candidates = 10000
X_cand = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, 5))

# GP予測
mean, std = gp.predict(X_cand, return_std=True)

# Expected Improvement (EI) 計算
best_y = gp.y_train_.max()          # 現在の最高value
xi = 0.01                           # explorationパラメータ（小さめが推奨）
z = (mean - best_y - xi) / std
ei = (mean - best_y - xi) * norm.cdf(z) + std * norm.pdf(z)
ei = np.where(std > 0, ei, 0)

# 上位10候補を表示
top_idx = np.argsort(ei)[-10:][::-1]
print("=== 次に試すべき上位10候補（EIが高い順） ===")
for i, idx in enumerate(top_idx):
    print(f"{i+1:2d}. value予想={mean[idx]:.2f} ± {std[idx]:.2f} | EI={ei[idx]:.4f}")
    print(f"     ENT_COEF={X_cand[idx][0]:.2e}, LR={X_cand[idx][1]:.2e}, "
          f"PEN={X_cand[idx][2]:.3f}, DIST={X_cand[idx][3]:.3f}, SUR={X_cand[idx][4]:.3f}\n")