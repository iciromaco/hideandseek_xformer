# main23_analysis.py
# 演習第23回：Optuna探索結果の統計解析（客観的指標対応版）
#
# 【修正内容 v23.2】
# 1. パス同期: main23_optuna_search.py が出力する results_{MODE}.csv に対応。
# 2. 指標同期: 最適化目標が Hidden ステップ数であることを前提に解析。
# 3. 統計解析の強化: 重回帰分析を行い、各パラメータの物理指標（生存）への寄与度を算出。
# 4. 文法規約: セミコロンなし、独立行での記述。

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 解析設定
# ==========================================
# 解析対象のモードを指定 ("initial" または "refinement")
MODE = "initial"
INPUT_CSV = f"results_{MODE}.csv"

# ==========================================
# 2. データ読み込みと前処理
# ==========================================
def run_analysis():
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] {INPUT_CSV} が見つかりません。")
        print("先に main23_optuna_search.py を実行してください。")
        return

    # CSVの読み込み
    df = pd.read_csv(INPUT_CSV)
    
    # 失敗した試行 (value が NaN または 0) の除外
    df = df.dropna(subset=['value'])
    # 物理指標（ステップ数）が0のデータも、学習が全く進んでいないため除外
    df = df[df['value'] > 0]
    
    if len(df) < 5:
        print(f"[Warning] 有効なデータが少なすぎます（現在 {len(df)} 件）。")
        print("解析には最低5件以上の成功試行が必要です。")
        return

    print(f"--- Analysis Report: {INPUT_CSV} ---")
    print(f"Total successful trials: {len(df)}")
    print(f"Best Objective Value (Hidden Steps): {df['value'].max():.2f}")

    # パラメータカラムの抽出 (Optuna は 'params_' という接頭辞をつける)
    param_cols = [c for c in df.columns if c.startswith('params_')]
    clean_param_names = [c.replace('params_', '') for c in param_cols]
    
    # 解析用データの準備
    X = df[param_cols].copy()
    y = df['value'].copy()

    # 指数スケールのパラメータ（LR等）を対数変換して感度を正しく測る
    for col in X.columns:
        if "learning_rate" in col or "ent_coef" in col:
            X[col] = np.log10(X[col] + 1e-12)
            print(f"  Applied log10 to {col}")

    # ==========================================
    # 3. 重回帰分析 (各パラメータの寄与度算出)
    # ==========================================
    # スケーリング（単位の異なるパラメータを比較可能にする）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 回帰モデルの実行
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # 寄与度の集計
    importance = pd.DataFrame({
        'Parameter': clean_param_names,
        'Coefficient': model.coef_
    })
    importance['Abs_Coef'] = importance['Coefficient'].abs()
    importance = importance.sort_values(by='Abs_Coef', ascending=False)

    print("\n[Parameter Influence (Standardized Coefficients)]")
    print("※ 系数の絶対値が大きいほど、そのパラメータが Hidden Steps の増減に強く影響しています。")
    print(importance[['Parameter', 'Coefficient']].to_string(index=False))

    # ==========================================
    # 4. 可視化
    # ==========================================
    sns.set_theme(style="whitegrid")
    
    # フィギュア作成 (2x2 のレイアウト)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Hyperparameter Sensitivity Analysis ({MODE})\nObjective: Steps Hidden (Physical Metric)", fontsize=16)

    # A. 寄与度の棒グラフ
    sns.barplot(data=importance, x='Coefficient', y='Parameter', ax=axes[0, 0], palette='coolwarm')
    axes[0, 0].set_title("Standardized Impact on Hidden Steps")
    axes[0, 0].set_xlabel("Coefficient (Direction & Strength)")

    # B. 学習推移
    axes[0, 1].plot(df['number'], df['value'], marker='o', linestyle='-', color='tab:blue', alpha=0.6)
    # 移動平均の追加
    if len(df) > 5:
        ma = df['value'].rolling(window=5).mean()
        axes[0, 1].plot(df['number'], ma, color='red', linewidth=2, label='MA(5)')
    axes[0, 1].set_title("Optimization History")
    axes[0, 1].set_xlabel("Trial Number")
    axes[0, 1].set_ylabel("Steps Hidden")
    axes[0, 1].legend()

    # C. 特徴的な散布図 (例: Learning Rate)
    lr_col = [c for c in param_cols if "learning_rate" in c]
    if lr_col:
        sns.scatterplot(data=df, x=lr_col[0], y='value', ax=axes[1, 0], hue='value', palette='magma', legend=False)
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_title("Learning Rate vs Hidden Steps")
        axes[1, 0].set_xlabel("Learning Rate (log scale)")
    else:
        axes[1, 0].text(0.5, 0.5, "LR Data Missing", ha='center')

    # D. 特徴的な散布図 (例: Cos Penalty Scale)
    cos_col = [c for c in param_cols if "cos_penalty_scale" in c]
    if cos_col:
        sns.regplot(data=df, x=cos_col[0], y='value', ax=axes[1, 1], scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        axes[1, 1].set_title("Cos Penalty Scale vs Hidden Steps")
        axes[1, 1].set_xlabel("Cos Penalty Scale")
    else:
        axes[1, 1].text(0.5, 0.5, "Cos Scale Data Missing", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存と表示
    plot_path = f"analysis_summary_{MODE}.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSummary plot saved to {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_analysis()