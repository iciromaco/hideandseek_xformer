# main22_analysis.py
# 最適化結果の多角分析 - 線形回帰、勾配ブースティング、ガウス過程回帰の一括実行
# 
# 【特徴】
# - 3手法の統合: A.線形トレンド, B.決定木による重要度, C.ガウス過程による感度を一度に算出。
# - 統一指標: 全てのモデルで部分依存性プロット (PDP) を作成し、手法間での結果の差異を比較可能。
# - 自動レポート: analysis_output_all フォルダに画像とテキストレポートを自動生成。
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.inspection import partial_dependence

# ==========================================
# 設定
# ==========================================
CSV_PATH = './optuna_results_layer0/results_refinement.csv'
OUTPUT_DIR = './optuna_results_layer0/analysis_output_all'

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 1. データ読み込みと前処理
# ==========================================
print(f"Loading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
df = df[df['state'] == 'COMPLETE'].copy()

if len(df) < 10:
    print(f"Warning: データ数が少ないため({len(df)}件)、分析精度が低くなる可能性があります。")
   
X_train = np.column_stack([
    np.log10(df['params_ENT_COEF'] + 1e-10),
    np.log10(df['params_LEARNING_RATE'] + 1e-10),
    df['params_PENALTY_STAGNATION_FORCE'],
    df['params_REWARD_DISTANCE_DIFF_SCALE'],
    df['params_REWARD_SURVIVAL_SCALE']
])

y_train = df['value'].values
y_mean_global = np.mean(y_train)  # 全体の平均値を計算（補正用）

features_info = [
    {'name': 'ENT_COEF', 'is_log': True},
    {'name': 'LEARNING_RATE', 'is_log': True},
    {'name': 'PENALTY_STAGNATION', 'is_log': False},
    {'name': 'REWARD_DIST_SCALE', 'is_log': False},
    {'name': 'REWARD_SURVIVAL', 'is_log': False}
]

# ==========================================
# 共通分析・描画関数
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
        name = info['name']
        is_log = info['is_log']
        ax = axes[i]

        # --- PDP計算 ---
        pdp_out = partial_dependence(
            model, X_train, [i],
            kind='average', grid_resolution=100
        )

        if 'grid_values' in pdp_out:
            x_vals = pdp_out['grid_values'][target_idx]
        else:
            x_vals = pdp_out['values'][target_idx]
        
        y_vals = pdp_out['average'][target_idx]

        # ★重要修正: 値のスケールチェックと補正
        # GBRなどでy_valsが0付近（偏差）になっている場合、全体の平均値を足して絶対値に戻す
        if np.mean(y_vals) < y_mean_global * 0.5: 
            y_vals = y_vals + y_mean_global

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

        imp_val = importance_scores[i] if importance_scores is not None else 0.0
        imp_str = f"{imp_val:5.1f}"
        impact_str = f"{impact:5.2f}"
        
        report_lines.append(f"{name:<25} | {imp_str:>10} | {impact_str:>12} | {best_val_str:>15}")

        # --- グラフ描画 ---
        if is_log:
            plot_x = 10 ** x_vals
        else:
            plot_x = x_vals

        ax.plot(plot_x, y_vals, color='tab:blue', linewidth=2, label='Mean EpLen')
        ax.fill_between(plot_x, y_vals - 1.0, y_vals + 1.0, color='tab:blue', alpha=0.1)
        ax.scatter([best_val_display], [np.max(y_vals)], color='red', zorder=5, label=f'Best: {best_val_str}')

        ax.set_title(f"{name}\n(Impact: {impact:.2f})", fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected Episode Length')
        
        # グラフごとの自動スケール（ただしImpactが小さい場合は範囲を狭めすぎないようにする）
        y_center = np.mean(y_vals)
        if impact < 2.0:
            ax.set_ylim(y_center - 2.0, y_center + 2.0)
        else:
            ax.autoscale(enable=True, axis='y')
            # マージンを少しとる
            ymin, ymax = np.min(y_vals), np.max(y_vals)
            margin = (ymax - ymin) * 0.2
            ax.set_ylim(ymin - margin, ymax + margin)

        ax.grid(True, which="both", ls="-", alpha=0.5)
        
        if is_log:
            ax.set_xscale('log')
            ax.set_xlabel('Value (Log Scale)')
        else:
            ax.set_xlabel('Value (Linear Scale)')
            
        ax.legend(loc='best')

    for j in range(len(features_info), len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Partial Dependence Plots ({model_name})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_filename = f'pdp_analysis_chart_{file_suffix}.png'
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    report_lines.append("-" * 80)
    report_lines.append("\n【用語解説】")
    report_lines.append(f"{importance_label:<10} : モデル固有の重要度指標 (0-100正規化済み)。")
    report_lines.append("Impact(Len): 影響度。パラメータ変化によるエピソード長の変動最大幅。")
    report_lines.append("Best Value : モデルが予測する最良パラメータ値。")
    
    if "Gaussian Process" in model_name:
        white_kernel = model.kernel_.k2
        report_lines.append("-" * 80)
        report_lines.append(f"[モデル診断] 推定ノイズレベル: {white_kernel.noise_level:.4f}")

    report_text = "\n".join(report_lines)
    print(report_text)
    
    report_filename = f'analysis_report_{file_suffix}.txt'
    report_path = os.path.join(OUTPUT_DIR, report_filename)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"[Saved] {plot_filename} & {report_filename}\n")


# ==========================================
# 2. 各モデルの構築と分析実行
# ==========================================

# --- A. 線形回帰 ---
print("--- [A] Linear Regression ---")
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
lr_coeffs = np.abs(lr_pipeline.named_steps['regressor'].coef_)
lr_importances = lr_coeffs / np.max(lr_coeffs) * 100
analyze_and_save(lr_pipeline, "Linear Regression", "LR", importance_scores=lr_importances, importance_label="Coef")

# --- B. 勾配ブースティング ---
print("--- [B] Gradient Boosting ---")
gbr_model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
gbr_model.fit(X_train, y_train)
gbr_importances = gbr_model.feature_importances_
gbr_importances = gbr_importances / np.max(gbr_importances) * 100
analyze_and_save(gbr_model, "Gradient Boosting", "GBR", importance_scores=gbr_importances, importance_label="Imp")

# --- C. ガウス過程回帰 ---
print("--- [C] Gaussian Process Regression ---")
kernel = C(1.0, (1e-3, 1e8)) * Matern(length_scale=[1.0]*5, length_scale_bounds=(1e-2, 1e8), nu=2.5) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-2, 1e5))
gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15, normalize_y=True, random_state=42)
gp_model.fit(X_train, y_train)
matern_kernel = gp_model.kernel_.k1.k2
gp_importances = (1.0 / matern_kernel.length_scale) / np.max(1.0 / matern_kernel.length_scale) * 100
analyze_and_save(gp_model, "Gaussian Process", "GPR", importance_scores=gp_importances, importance_label="Sens(ARD)")

print(f"All analyses complete. Check {OUTPUT_DIR}")