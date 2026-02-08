import pandas as pd
import numpy as np
import pickle
from sklearn.inspection import partial_dependence

# ────────────────────────────────────────────────
# 設定・読み込み
# ────────────────────────────────────────────────
PKL_PATH = './optuna_results_layer0/gp_model_log.pkl'
CSV_PATH = './optuna_results_layer0/results_refinement.csv'

# データ読み込み（特徴量の範囲特定用）
df = pd.read_csv(CSV_PATH)
df = df[df['state'] == 'COMPLETE'].copy()

# ログ変換等の前処理（ana1.pyと同じロジック）
X_train = np.column_stack([
    np.log10(df['params_ENT_COEF'] + 1e-10),
    np.log10(df['params_LEARNING_RATE'] + 1e-10),
    df['params_PENALTY_STAGNATION_FORCE'],
    df['params_REWARD_DISTANCE_DIFF_SCALE'],
    df['params_REWARD_SURVIVAL_SCALE']
])

feature_names = [
    'ENT_COEF (log)',
    'LEARNING_RATE (log)',
    'PENALTY_STAGNATION',
    'REWARD_DIST_SCALE',
    'REWARD_SURVIVAL'
]

# モデル読み込み
with open(PKL_PATH, 'rb') as f:
    gp_model = pickle.load(f)

# ────────────────────────────────────────────────
# 1. ARD感度分析 (カーネルパラメータの解析)
# ────────────────────────────────────────────────
# カーネル構造: C * Matern + WhiteKernel
# gp_model.kernel_.k1 が Product(C, Matern)
# gp_model.kernel_.k1.k2 が Matern カーネル
matern_kernel = gp_model.kernel_.k1.k2
length_scales = matern_kernel.length_scale

# 感度 (Sensitivity) = 1 / length_scale
# 長さスケールが小さいほど、その次元のわずかな変化で関数値が大きく変わるため「感度が高い」
sensitivities = 1.0 / length_scales
sensitivity_normalized = sensitivities / np.max(sensitivities) * 100  # 最大を100に正規化

# ────────────────────────────────────────────────
# 2. PDPに基づく影響度と最適値の算出
# ────────────────────────────────────────────────
pdp_results = []

print(f"{'='*80}")
print(f"{'PARAMETER ANALYSIS REPORT':^80}")
print(f"{'='*80}")
print(f"{'Param Name':<25} | {'Sens(ARD)':<10} | {'Impact(Rew)':<12} | {'Best Value':<15}")
print("-" * 80)

for i, name in enumerate(feature_names):
    # PDPの計算
    pdp_out = partial_dependence(
        gp_model, X_train, [i], 
        kind='average', grid_resolution=100
    )
    
    # 修正箇所: scikit-learnのバージョン互換性対応
    # pdp_out['grid_values'] はリスト形式 [array([...])] なので  を指定
    # pdp_out['average'] は shape (1, grid_size) なので  を指定
    
    if 'grid_values' in pdp_out:
        x_vals = pdp_out['grid_values'][0]
    else:
        # 古いバージョンのscikit-learn用フォールバック
        x_vals = pdp_out['values'][0]
        
    y_vals = pdp_out['average']
    
    # 影響度 (Impact): 報酬予測値の変動幅 (Max - Min)
    impact = np.max(y_vals) - np.min(y_vals)
    best_idx = np.argmax(y_vals)
    best_val = x_vals[best_idx]
    
    # 整形表示
    sens_str = f"{sensitivity_normalized[i]:5.1f}"
    impact_str = f"{impact:5.2f}"
    
    # ログ変換されたパラメータは元のスケールに戻して表示
    if "(log)" in name:
        best_val_orig = 10**best_val
        best_val_str = f"{best_val_orig:.2e}"
    else:
        best_val_str = f"{best_val:.4f}"
        
    print(f"{name:<25} | {sens_str:>10} | {impact_str:>12} | {best_val_str:>15}")

print("-" * 80)
print("\n【用語解説】")
print("Sens(ARD)   : 感度スコア (0-100)。GPRカーネルの長さスケール逆数。")
print("              数値が高いほど、このパラメータの微小な変化にモデルが敏感に反応しています。")
print("Impact(Rew) : 影響度。パラメータを変化させた時に、平均報酬(Value)が最大でどれだけ変わるか。")
print("              この値が大きいパラメータが、学習成否の鍵を握っています。")
print("Best Value  : モデルが予測する、最も報酬が高くなるパラメータ値。")
print(f"{'='*80}")

# ノイズレベルの報告
white_kernel = gp_model.kernel_.k2
print(f"\n[モデル診断] 推定ノイズレベル (Noise Level): {white_kernel.noise_level:.4f}")
print("※ この値が報酬の変動幅(Impact)より極端に大きい場合、パラメータ探索よりも「運」の要素が強い可能性があります。")
print("   その場合、より多くの試行回数を確保するか、別の手法を検討してください。")