# ana1.py
# 1. GPモデルを訓練してpickleで保存するコード
# 必要ライブラリ（事前にpip installしておく）
# pip install scikit-learn pandas numpy joblib

import pandas as pd
import numpy as np
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C

# データ読み込み
df = pd.read_csv('./optuna_results_layer0/results_refinement.csv')
df = df[df['state'] == 'COMPLETE'].copy()

param_cols = ['params_ENT_COEF', 'params_LEARNING_RATE', 'params_PENALTY_STAGNATION_FORCE',
              'params_REWARD_DISTANCE_DIFF_SCALE', 'params_REWARD_SURVIVAL_SCALE']

# ★ここが変更点：ENTとLRを log10 に変換
X_log = np.column_stack([
    np.log10(df['params_ENT_COEF'] + 1e-10),          # 0回避のため微小値を足す
    np.log10(df['params_LEARNING_RATE'] + 1e-10),
    df['params_PENALTY_STAGNATION_FORCE'],
    df['params_REWARD_DISTANCE_DIFF_SCALE'],
    df['params_REWARD_SURVIVAL_SCALE']
])

y = df['value'].values

# 特徴量名も log版に変更（PDPで見やすくなる）
feature_names_log = [
    'log10_ENT_COEF',
    'log10_LEARNING_RATE',
    'PENALTY_STAGNATION_FORCE',
    'REWARD_DISTANCE_DIFF_SCALE',
    'REWARD_SURVIVAL_SCALE'
]

# kernel（以前と同じものを使用）
kernel = C(1.0, (1e-3, 1e8)) * Matern(length_scale=[1.0]*5,
                length_scale_bounds=(1e-2, 1e8),        # length_scaleの上界も広げる
                nu=2.5) + WhiteKernel(noise_level=1.0,
                     noise_level_bounds=(1e-2, 1e5))    # noiseの上界を10→1000に
gp_log = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=15,
    normalize_y=True,          # 今回は実際のスケールで見たい場合はFalse
    random_state=42
)

gp_log.fit(X_log, y)

print("Optimized kernel (log版):", gp_log.kernel_)
print("Log marginal likelihood:", gp_log.log_marginal_likelihood(gp_log.kernel_.theta))

# pickle保存（log版として区別）
with open('./optuna_results_layer0/gp_model_log.pkl', 'wb') as f:
    pickle.dump(gp_log, f)

print("log版GPモデルを gp_model_log.pkl に保存しました！")