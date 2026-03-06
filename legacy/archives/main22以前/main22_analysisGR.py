# main22_analysisGR.py
# ガウス過程回帰 (GPR) を用いたパラメータ感度・部分依存性分析 (改良版)
# 
# 【特徴】
# - ログスケール対応: LEARNING_RATE 等の対数変換パラメータをグラフ上で正しく表示。
# - ARD感度分析: どのパラメータが報酬に敏感に反応するかを数値化。
# - 最適値予測: モデルに基づいた理論的なベストパラメータを算出。

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.inspection import partial_dependence

# ==========================================
# 設定
# ==========================================
CSV_PATH = './optuna_results_layer0/results_refinement.csv'
OUTPUT_DIR = './optuna_results_layer0/analysis_output'
REPORT_FILENAME = 'analysis_report.txt'
FIGURE_FILENAME = 'pdp_analysis_chart.png'

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 1. データ読み込みと前処理
# ==========================================
print("Loading data...")
df = pd.read_csv(CSV_PATH)
df = df[df['state'] == 'COMPLETE'].copy()

# パラメータ列の定義
X_train = np.column_stack([
    np.log10(df['params_ENT_COEF'] + 1e-10),
    np.log10(df['params_LEARNING_RATE'] + 1e-10),
    df['params_PENALTY_STAGNATION_FORCE'],
    df['params_REWARD_DISTANCE_DIFF_SCALE'],
    df['params_REWARD_SURVIVAL_SCALE']
])

y_train = df['value'].values

features_info = [
    {'name': 'ENT_COEF', 'is_log': True},
    {'name': 'LEARNING_RATE', 'is_log': True},
    {'name': 'PENALTY_STAGNATION', 'is_log': False},
    {'name': 'REWARD_DIST_SCALE', 'is_log': False},
    {'name': 'REWARD_SURVIVAL', 'is_log': False}
]

# ==========================================
# 2. ガウス過程回帰 (GPR) モデルの学習
# ==========================================
print("Training Gaussian Process Model...")

kernel = C(1.0, (1e-3, 1e8)) * Matern(
    length_scale=[1.0] * 5,
    length_scale_bounds=(1e-2, 1e8),
    nu=2.5
) + WhiteKernel(
    noise_level=0.1,
    noise_level_bounds=(1e-2, 1e5)
)

gp_model = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=15,
    normalize_y=True,
    random_state=42
)

gp_model.fit(X_train, y_train)
print("Model trained.")

# ==========================================
# 3. 分析とグラフ描画
# ==========================================
print("Generating Analysis and Plots...")

report_lines = []
report_lines.append(f"{'='*80}")
report_lines.append(f"{'PARAMETER ANALYSIS REPORT (v4)':^80}")
report_lines.append(f"{'='*80}")
report_lines.append(f"{'Param Name':<25} | {'Sens(ARD)':<10} | {'Impact(Len)':<12} | {'Best Value':<15}")
report_lines.append("-" * 80)

# ARD感度計算
matern_kernel = gp_model.kernel_.k1.k2
length_scales = matern_kernel.length_scale
sensitivities = 1.0 / length_scales
sensitivity_normalized = sensitivities / np.max(sensitivities) * 100

# グラフ準備
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

target_idx = 0 

for i, info in enumerate(features_info):
    name = info['name']
    is_log = info['is_log']
    ax = axes[i]

    # --- PDP計算 ---
    pdp_out = partial_dependence(
        gp_model, X_train, [i],
        kind='average', grid_resolution=100
    )

    if 'grid_values' in pdp_out:
        x_vals = pdp_out['grid_values'][target_idx]
    else:
        x_vals = pdp_out['values'][target_idx]
    
    y_vals = pdp_out['average'][target_idx]

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

    sens_str = f"{sensitivity_normalized[i]:5.1f}"
    impact_str = f"{impact:5.2f}"
    
    report_lines.append(f"{name:<25} | {sens_str:>10} | {impact_str:>12} | {best_val_str:>15}")

    # --- グラフ描画 ---
    if is_log:
        plot_x = 10 ** x_vals
    else:
        plot_x = x_vals

    ax.plot(plot_x, y_vals, color='tab:blue', linewidth=2, label='Mean EpLen')
    ax.fill_between(plot_x, y_vals - 1.0, y_vals + 1.0, color='tab:blue', alpha=0.1)
    
    ax.scatter([best_val_display], [np.max(y_vals)], color='red', zorder=5, label=f'Best: {best_val_str}')

    # ★改善点: タイトルに変動幅(Impact)を表示して、グラフの重要度を数値で直感的に分かるようにする
    ax.set_title(f"{name}\n(Impact: {impact:.2f})", fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Expected Episode Length')
    
    # ★変更点: Y軸の範囲統一を削除し、Matplotlibの自動スケーリングに戻す
    # これにより、微細な変化も拡大表示されるが、タイトルのImpact値で重要度を判断する
    # ax.set_ylim(ylim_global) 
    
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    if is_log:
        ax.set_xscale('log')
        ax.set_xlabel('Value (Log Scale)')
    else:
        ax.set_xlabel('Value (Linear Scale)')
        
    ax.legend(loc='best')

# 余ったサブプロットを消す
for j in range(len(features_info), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, FIGURE_FILENAME)
plt.savefig(plot_path, dpi=300, bbox_inches='tight')

# レポート出力
report_lines.append("-" * 80)
report_lines.append("\n【用語解説】")
report_lines.append("Sens(ARD)   : 感度スコア。数値が高いほど、このパラメータの微小な変化に敏感。")
report_lines.append("Impact(Len) : 影響度。このパラメータを変えることで改善が見込める「エピソード長」の幅。")
report_lines.append("              ※ グラフが大きく動いているように見えても、この値が小さければ(例: < 5.0)、学習への影響は軽微です。")
report_lines.append("Best Value  : 最も長い生存時間が予測されるパラメータ値。")
report_lines.append("-" * 80)

white_kernel = gp_model.kernel_.k2
report_lines.append(f"\n[モデル診断] 推定ノイズレベル: {white_kernel.noise_level:.4f}")
report_text = "\n".join(report_lines)

print(report_text)
report_path = os.path.join(OUTPUT_DIR, REPORT_FILENAME)
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"\n[Saved] Analysis report saved to: {report_path}")
print(f"[Saved] Plot saved to: {plot_path}")