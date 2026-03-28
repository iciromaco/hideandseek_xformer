import os
import sys

import optuna
import optuna.importance

# --- 設定 (main19と同じにする) ---
STUDY_NAME = "HideAndSeek_Reward_Optimization"

# DBファイルを探す
db_dir = "optuna_results"
db_files = [f for f in os.listdir(db_dir) if f.endswith(".db")] if os.path.exists(db_dir) else []

if not db_files:
    print(f"Error: {db_dir} ディレクトリにデータベースファイル(.db)が見つかりません。")
    sys.exit(1)

print(f"Found databases in {db_dir}:")
for i, f in enumerate(db_files):
    print(f"  [{i}] {f}")

# ユーザーに選択させる、または最新を自動選択
if len(db_files) == 1:
    target_db = db_files[0]
else:
    # タイムスタンプ順に新しいものをデフォルトにするなどの処理も可能だが、
    # ここではシンプルに末尾（最新の可能性が高い）を選択、あるいはファイル名で判断
    # ユーザーが実行していたのが 'refinement' ならそれっぽいものを選ぶ
    print("--------------------------------------------------")
    idx = input("Select DB index (default 0): ")
    if idx.strip() == "":
        idx = 0
    else:
        idx = int(idx)
    target_db = db_files[idx]

db_url = f"sqlite:///{db_dir}/{target_db}"
study_name_in_db = target_db.replace(".db", "").replace("_v2", "")  # ファイル名からstudy名を推測

print(f"\nLoading study from {db_url} ...")

try:
    # Studyのロード
    # ※保存時のstudy_nameと一致している必要があるため、load_if_exists=Trueで読み込む
    # study_nameがファイル名と微妙に違う場合があるので、ストレージ内のstudy一覧を取得してみる
    storage = optuna.storages.RDBStorage(url=db_url)
    all_studies = [s.study_name for s in optuna.get_all_study_summaries(storage=storage)]

    if not all_studies:
        print("Error: No studies found in this database.")
        sys.exit(1)

    target_study_name = all_studies[0]  # 基本的に1つのはず
    print(f"Target Study Name: {target_study_name}")

    study = optuna.load_study(study_name=target_study_name, storage=storage)

    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS REPORT")
    print("=" * 70)
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Score: {study.best_value:.4f}")

    print("\nBest Params:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val:.6f}")

    print("\nTop 5 Trials:")
    try:
        trials_df = study.trials_dataframe()
        # 完了した試行のみ抽出
        trials_df = trials_df[trials_df.state == "COMPLETE"]
        trials_df = trials_df.sort_values("value", ascending=False).head(5)

        for idx, (_, row) in enumerate(trials_df.iterrows(), 1):
            print(f"  [{idx}] Trial {int(row['number'])}: {row['value']:.4f}")
            for key in study.best_params.keys():
                p_key = f"params_{key}"
                if p_key in trials_df.columns:
                    print(f"       {key}: {row[p_key]:.6f}")
            print("-" * 30)
    except Exception as e:
        print(f"  (Could not display top trials: {e})")

    # パラメータ重要度
    print("\nParameter Importances:")
    try:
        importances = optuna.importance.get_param_importances(study)
        for key, val in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            print(f"  {key}: {val:.3f}")
    except Exception as e:
        print(f"  (Could not compute importances: {e})")

    # CSV保存
    csv_path = os.path.join(db_dir, f"{target_db}_report.csv")
    study.trials_dataframe().to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")

    # 推奨アクション
    print("\n" + "=" * 70)
    print("RECOMMENDED ACTION for main18_optimization.py")
    print("=" * 70)
    print("Update the variables with:")
    print()
    for key, val in study.best_params.items():
        if key == "lr":
            print(f"LEARNING_RATE = {val:.8f}")
        elif key == "ent_coef":
            print(f"ENT_COEF = {val:.8f}")
        else:
            print(f"{key} = {val:.5f}")

except Exception as e:
    print(f"An error occurred: {e}")
