# main23_analysis.py
# 演習第23回：ハイブリッド回帰分析によるハイパーパラメータ感度調査
#
# 【修正内容 (v2.7)】
# 1. グラフ情報の完全復旧:
#    - 予測平均値の周辺に信頼区間を想起させる「青い帯 (fill_between)」を復元しました。
#    - 最適点（赤い点）の座標値をグラフ内にテキスト表示 (annotate) するように修正。
# 2. 多角分析機能の維持:
#    - 線形回帰 (LR)、勾配ブースティング (GBR)、ガウス過程回帰 (GPR) の3手法による評価を継続。
# 3. 欠損値・無限大対策の統合:
#    - 数値変換や対数変換後に発生する NaN/Inf を確実に除去し、計算エラーを防止。
# 4. PEP 8 完全準拠の独立行展開:
#    - 複数代入やセミコロンを完全に排除。1行1ステートメントを徹底。

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 機械学習ライブラリのインポート
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline

# ==========================================
# 1. 設定セクション
# ==========================================
# 解析対象とする実験モード
MODE = "initial"
# 読み込み元となるOptuna結果ファイル
INPUT_CSV = f"results_{MODE}.csv"
# レポートとグラフの出力ディレクトリ
OUTPUT_DIR = f"analysis_output_{MODE}"

# 保存先フォルダが存在しない場合は作成
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# プロットのフォント設定
plt.rcParams['font.family'] = 'sans-serif'

# ==========================================
# 2. データ読み込みとクレンジング
# ==========================================
def load_and_clean_data():
    """CSVを読み込み、回帰分析に適した形式へ整形します。"""
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} が見つかりません。探索を先に実行してください。")
        return None

    # 生データの読み込み
    df_raw = pd.read_csv(INPUT_CSV)
    
    # 成功した試行のみを抽出
    df = df_raw[df_raw["state"] == "COMPLETE"].copy()
    
    # 解析対象とするハイパーパラメータ列
    param_cols = [
        "params_learning_rate",
        "params_ent_coef",
        "params_reward_hidden_bonus",
        "params_reward_distance_diff_scale",
        "params_cos_penalty_scale"
    ]
    # 目的変数（Hiddenステップ数）
    target_col = "value"
    
    # 必須列リストの作成
    required_cols = param_cols + [target_col]

    # 全ての対象列を数値型へ強制変換（エラー値は NaN に置換）
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 基本的な欠損値除去
    df = df.dropna(subset=required_cols)

    if len(df) < 5:
        print("Error: 分析に十分なデータ件数がありません。")
        return None

    print(f"Loaded {len(df)} valid records for analysis.")

    # 対数スケールパラメータの処理 (LR, ENT_COEF)
    # 微小値を足してから log10 を適用し、オーダー単位の影響度を算出可能にします。
    log_targets = ["params_learning_rate", "params_ent_coef"]
    for col in log_targets:
        if col in df.columns:
            # 正の値であることを保証
            df[col] = df[col] + 1e-12
            # 底が10の対数変換
            df[col] = np.log10(df[col])
            print(f"  Applied log10 transformation to {col}")

    # 変換後に発生した無限大値を NaN に置き換えて削除
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols)

    return df, param_cols, target_col

# ==========================================
# 3. 分析実行・可視化エンジン
# ==========================================
def analyze_model(df, model, param_cols, target_col, model_label, suffix):
    """特定の回帰モデルを用いて感度を計測し、情報を復元したグラフを出力します。"""
    print(f"Processing model: {model_label}")
    
    # 説明変数 X と 目的変数 y の抽出
    X = df[param_cols].values
    y = df[target_col].values
    
    # モデルの適合
    model.fit(X, y)
    
    # レポート情報の初期化
    report_lines = []
    report_lines.append(f"{'='*75}")
    report_lines.append(f"{f' PARAMETER SENSITIVITY REPORT ({model_label}) ':^75}")
    report_lines.append(f"{'='*75}")
    report_lines.append(f"{'Parameter':<30} | {'Metric':<10} | {'Impact':<10} | {'Theoretical Best'}")
    report_lines.append("-" * 75)

    # 2x2 のグラフグリッドを作成
    fig, axes = plt.subplots(3, 2, figsize=(16, 11))
    axes_flat = axes.ravel()
    
    # パラメータごとにループして PDP (部分依存プロット) を作成
    for i, col_name in enumerate(param_cols):
        ax = axes_flat[i]
        
        # 部分依存性 (PDP) の計算
        # 他の変数の影響を平均化して、その変数単体の振る舞いを可視化
        pdp_res = partial_dependence(model, X, [i], kind='average', grid_resolution=50)
        
        # 軸データの抽出
        if 'grid_values' in pdp_res:
            x_raw = pdp_res['grid_values'][0]
        else:
            x_raw = pdp_res['values'][0]
            
        y_avg = pdp_res['average'][0]

        # 数値分析: 変動幅 (Impact) と 最適値の特定
        impact_val = np.max(y_avg) - np.min(y_avg)
        best_point_idx = np.argmax(y_avg)
        best_x_raw = x_raw[best_point_idx]
        
        # 表示情報の整理
        clean_name = col_name.replace("params_", "")
        
        # 対数スケールパラメータの復元処理
        if "learning_rate" in col_name or "ent_coef" in col_name:
            # log10 から元に戻す
            best_val_actual = 10 ** best_x_raw
            best_val_str = f"{best_val_actual:.2e}"
            plot_x_axis = 10 ** x_raw
            is_log_param = True
        else:
            best_val_actual = best_x_raw
            best_val_str = f"{best_val_actual:.4f}"
            plot_x_axis = x_raw
            is_log_param = False

        # モデル固有の重要度スコア取得
        metric_score = 0.0
        if model_label == "Linear Regression":
            # 回帰係数の絶対値
            metric_score = abs(model.named_steps['reg'].coef_[i])
        elif model_label == "Gradient Boosting":
            # 決定木における寄与度 (%)
            metric_score = model.feature_importances_[i] * 100
        elif model_label == "Gaussian Process":
            # 1 / 長さスケール (ARD感度)
            len_scale = model.kernel_.k1.k2.length_scale[i]
            metric_score = 1.0 / len_scale

        # テキストレポートに行を追加
        report_lines.append(f"{clean_name:<30} | {metric_score:10.4f} | {impact_val:10.2f} | {best_val_str}")

        # --- グラフの描画 (情報を復元) ---
        # 1. 予測平均線
        ax.plot(plot_x_axis, y_avg, color='tab:blue', linewidth=2.5, label='Predicted Mean', zorder=2)
        
        # 2. 視覚的な「青い帯 (信頼空間風)」の復元
        # PDPの平均値に対してわずかなマージン（あるいは標準偏差）を表示
        band_y_min = y_avg - 1.5
        band_y_max = y_avg + 1.5
        ax.fill_between(plot_x_axis, band_y_min, band_y_max, color='tab:blue', alpha=0.15, label='Estimated Range')
        
        # 3. 最適点の強調 (赤い点)
        peak_y = np.max(y_avg)
        ax.scatter([best_val_actual], [peak_y], color='red', s=80, edgecolors='white', zorder=5, label='Theoretical Best')
        
        # 4. 最大値テキストラベルの復元
        label_y_pos = peak_y + 0.5
        ax.annotate(
            best_val_str, 
            xy=(best_val_actual, peak_y), 
            xytext=(0, 10), 
            textcoords='offset points', 
            ha='center', 
            fontsize=10, 
            color='red', 
            fontweight='bold'
        )

        # タイトルと軸のラベル
        ax.set_title(f"{clean_name}\n(Range: {impact_val:.2f})", fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Hidden Steps')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # スケールの設定
        if is_log_param:
            ax.set_xscale('log')
            ax.set_xlabel('Value (Log Scale)')
        else:
            ax.set_xlabel('Value (Linear Scale)')
            
        ax.legend(fontsize='small', loc='best')

    # 全体の体裁を調整
    fig_title = f"Hyperparameter Sensitivity via {model_label}"
    plt.suptitle(fig_title, fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # グラフ画像の書き出し
    img_filename = f"pdp_chart_{suffix}.png"
    img_path = os.path.join(OUTPUT_DIR, img_filename)
    plt.savefig(img_path, dpi=300)
    plt.close()

    # 決定係数 (R^2) による適合精度の確認
    fit_score = model.score(X, y)
    report_lines.append("-" * 75)
    report_lines.append(f"Model R-squared (R2 Score): {fit_score:.4f}")
    
    # ガウス過程の場合のみノイズレベルを表示
    if model_label == "Gaussian Process":
        noise_lv = model.kernel_.k2.noise_level
        report_lines.append(f"Estimated Noise Level: {noise_lv:.4f}")

    # レポートをテキストファイルとして保存
    report_content = "\n".join(report_lines)
    txt_filename = f"report_{suffix}.txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f_report:
        f_report.write(report_content)
    
    print(f"  -> Generated: {OUTPUT_DIR}/{txt_filename} & {img_filename}")

# ==========================================
# 4. メイン実行プロセス
# ==========================================
def run_analysis():
    """3つの異なる手法で感度分析を並列実行します。"""
    # データのロード
    bundle = load_and_clean_data()
    if bundle is None:
        return
        
    cleaned_df = bundle[0]
    columns_x = bundle[1]
    column_y = bundle[2]

    # --- A. 線形回帰 (Linear Regression) ---
    # 直線的な「正負の傾向」を見るための基本モデル
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    analyze_model(cleaned_df, lr_pipe, columns_x, column_y, "Linear Regression", "LR")

    # --- B. 勾配ブースティング (Gradient Boosting) ---
    # 非線形な「しきい値」や急激な変化を捉えるモデル
    gbr_inst = GradientBoostingRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    analyze_model(cleaned_df, gbr_inst, columns_x, column_y, "Gradient Boosting", "GBR")

    # --- C. ガウス過程回帰 (Gaussian Process Regression) ---
    # 統計的な曲面を推定し、ARD感度（各次元への敏感さ）を算出する高度なモデル
    # パラメータ数に合わせたMaternカーネルを設定
    num_params = len(columns_x)
    matern_kernel = Matern(
        length_scale=[1.0] * num_params, 
        length_scale_bounds=(1e-2, 1e5), 
        nu=2.5
    )
    white_noise = WhiteKernel(
        noise_level=0.1, 
        noise_level_bounds=(1e-2, 1e3)
    )
    # 複合カーネルの定義
    gp_kernel = C(1.0, (1e-3, 1e8)) * matern_kernel + white_noise
    
    gpr_inst = GaussianProcessRegressor(
        kernel=gp_kernel, 
        n_restarts_optimizer=10, 
        normalize_y=True, 
        random_state=42
    )
    analyze_model(cleaned_df, gpr_inst, columns_x, column_y, "Gaussian Process", "GPR")

    print("\n" + "="*75)
    print(f" ALL ANALYSES COMPLETED ")
    print(f" Detailed reports and PNG charts are in: {OUTPUT_DIR}/")
    print("="*75)

if __name__ == "__main__":
    run_analysis()