# main23_optuna_searchKai.py
# 演習第23回：Optunaによるハイパーパラメータ探索 (中間保存・モデル汚染防止版)
#
# 【修正内容 (v2.7)】
# 1. 最適化対象の変更:
#    - 最適化対象: LEARNING_RATE, ENT_COEF, REWARD_HIDDEN_BONUS, REWARD_DISTANCE_DIFF_SCALE
#    - 固定値: COS_PENALTY_SCALE = 2.0 (最適化対象から除外)
# 2. 中間保存ロジック (Save Callback) の維持:
#    - 試行が1回完了するたびに結果を CSV に書き出し、中断時のデータ損失を防止。
# 3. 既存の安全策を維持:
#    - 各試行でのモデル保存禁止 (SAVE_MODEL = False) によるモデル汚染防止。
#    - 試行終了後の作業ディレクトリ自動削除。

import os
import sys
import shutil
import subprocess
import re
import platform
import multiprocessing
import numpy as np
import optuna
from pathlib import Path
from tqdm import tqdm
import time
import threading

# ==========================================
# 1. 実験設定
# ==========================================
# ★実行モードの設定
# "initial":    新規探索 (LOAD_EXISTING_MODELS = False)
# "refinement": 継続探索 (LOAD_EXISTING_MODELS = True)
MODE = "initial" 

# ターゲットとなる学習スクリプト
RUNNER_SCRIPT = "main23_sightmap_optimized.py"

# --- 探索の制御パラメータ ---
# 全体の試行回数
N_TRIALS = 30
# 1試行あたりの制限時間 (秒)
TIMEOUT_PER_TRIAL = 1200 
# 各試行で実行する学習ステップ数
TOTAL_STEPS = 150000

# Optuna データベース名
STUDY_NAME = f"HS_Cos_Opt_{MODE}"
STORAGE_URL = f"sqlite:///optuna_study_{MODE}.db"
# 保存されるCSVファイルパス
CSV_OUTPUT_PATH = f"results_{MODE}.csv"

# 並列実行数 (GPUメモリ競合を避けるため1を推奨)
PARALLEL_JOBS = 1

# 客観的指標として抽出するキーワード (Hiddenステップ数)
METRIC_KEYWORD = "Hidden:"

# ==========================================
# 2. ロギング監視関数
# ==========================================
def monitor_log_file(log_file_path, process, trial_id, pbar_dict):
    """ログファイルから Update 進捗を抽出して dict に格納"""
    try:
        # ファイルが出現するまで待機
        for _ in range(200):
            if os.path.exists(log_file_path):
                break
            time.sleep(0.05)
        
        last_update_num = 0
        while process.poll() is None:
            try:
                with open(log_file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    
                    # Update 行から最新の情報を抽出
                    for line in lines:
                        if "Update" in line:
                            match_num = re.search(r"Update\s+(\d+)", line)
                            match_hidden = re.search(r"Hidden:\s+([\d\.]+)", line)
                            
                            if match_num:
                                update_num = int(match_num.group(1))
                                hidden_val = float(match_hidden.group(1)) if match_hidden else 0.0
                                
                                # dict に更新（最新値のみ保持）
                                if update_num != last_update_num:
                                    pbar_dict['update_num'] = update_num
                                    pbar_dict['hidden'] = hidden_val
                                    last_update_num = update_num
            except (IOError, OSError):
                pass
            
            time.sleep(0.2)
    except Exception:
        pass

# ==========================================
# 3. コールバック関数 (中間保存用)
# ==========================================
def save_best_callback(study, trial):
    """試行が1回終わるたびに結果をCSVに保存する"""
    try:
        df = study.trials_dataframe()
        df.to_csv(CSV_OUTPUT_PATH, index=False)
        # 最良値が更新された場合のみコンソールに通知
        if (study.best_trial.number == trial.number and
                study.best_trial.value is not None):
            pass
            # msg = f"  [Callback] New best: {study.best_trial.value:.2f}."
            # tqdm.write(msg)
    except Exception as e:
        pass
        # tqdm.write(f"  [Warning] Intermediate save: {e}")

# ==========================================
# 4. Optuna 目的関数
# ==========================================
def objective(trial, pbar_trials=None):
    """Optunaの各試行（Trial）で実行されるメインロジック"""
    
    trial_id = trial.number
    # 作業用ディレクトリ名の決定
    trial_dir = f"trial_{MODE}_{trial_id:03d}"
    trial_path = Path(trial_dir)
    
    # --- A. パラメータの提案 ---
    if MODE == "initial":
        # 学習制御パラメータ (初期学習用)
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        ent = trial.suggest_float("ent_coef", 1e-4, 5e-3, log=True)
    else:
        # 学習制御パラメータ (微調整用)
        lr = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        ent = trial.suggest_float("ent_coef", 5e-5, 1e-3, log=True)
        
    # 報酬設計パラメータ (最適化対象)
    hidden_bonus = trial.suggest_float("reward_hidden_bonus", 0.5, 3.0)
    dist_diff_scale = trial.suggest_float("reward_distance_diff_scale", 0.5, 5.0)
    
    # 固定値設定 (最適化対象外)
    cos_scale = 2.0
    
    # --- B. 環境構築 ---
    if not trial_path.exists():
        os.makedirs(trial_dir)
        
    with open(RUNNER_SCRIPT, "r", encoding="utf-8") as f:
        script_content = f.read()
        
    # --- C. スクリプトの動的書き換え (完全展開) ---
    # 1. 学習率の置換
    script_content = re.sub(r"LEARNING_RATE = .*", f"LEARNING_RATE = {lr}", script_content)
    # 2. エントロピー係数の置換
    script_content = re.sub(r"ENT_COEF = .*", f"ENT_COEF = {ent}", script_content)
    # 3. 隠蔽報酬の置換
    script_content = re.sub(r"REWARD_HIDDEN_BONUS = .*", f"REWARD_HIDDEN_BONUS = {hidden_bonus}", script_content)
    # 4. 距離増分報酬の置換 (新規追加)
    script_content = re.sub(r"REWARD_DISTANCE_DIFF_SCALE = .*", f"REWARD_DISTANCE_DIFF_SCALE = {dist_diff_scale}", script_content)
    # 5. 視界ペナルティの置換 (固定値として上書き)
    script_content = re.sub(r"COS_PENALTY_SCALE = .*", f"COS_PENALTY_SCALE = {cos_scale}", script_content)
    
    # 6. モデルロード設定の反映
    load_flag = "True" if MODE == "refinement" else "False"
    script_content = re.sub(r"LOAD_EXISTING_MODELS = .*", f"LOAD_EXISTING_MODELS = {load_flag}", script_content)

    # 7. モデル保存の強制禁止 (探索時のドリフト防止)
    script_content = re.sub(r"SAVE_MODEL = .*", "SAVE_MODEL = False", script_content)
    # 8. 評価モード（高頻度ログ）有効化
    script_content = re.sub(r"TRIAL_MODE = .*", "TRIAL_MODE = True", script_content)
    # 9. WandB無効化
    script_content = re.sub(r"TRACK_WANDB = .*", "TRACK_WANDB = True", script_content)
    # 10. 学習ステップ数の反映
    script_content = re.sub(r"TOTAL_TIMESTEPS = .*", f"TOTAL_TIMESTEPS = {TOTAL_STEPS}", script_content)
    
    # 書き換えたスクリプトを一時ファイルとして保存
    tmp_script_path = trial_path / "runner.py"
    # パラメータ値を表示
    print(f"  LR={lr:.1e} ENT={ent:.1e} HB={hidden_bonus:.2f} DS={dist_diff_scale:.2f}", flush=True)
    with open(tmp_script_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    # --- D. サブプロセスの実行準備 ---
    current_env = os.environ.copy()
    current_env["PYTHONPATH"] = os.getcwd() + os.pathsep + current_env.get("PYTHONPATH", "")
    current_env["PYTHONUNBUFFERED"] = "1"
    # ★最適化: 子プロセスの tqdm を無効化
    current_env["OPTUNA_SEARCH"] = "True"
    
    # tqdm.write(f"[Optuna] Trial {trial_id} ({MODE}) started")
    
    # メトリクスファイルのパス（子プロセスがここに結果を書く）
    # metrics_file = trial_path / "trial_metrics.json"
    
    # ★最適化: 子プロセスのログをファイルに落とす（親プロセスのtqdm TTY検出を邪魔しない）
    log_file_path = trial_path / "trial.log"
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        # 学習の実行
        process = subprocess.Popen(
            [sys.executable, str(tmp_script_path)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=current_env
        )
        
        # ★ログファイルをリアルタイムで監視するスレッドを起動
        pbar_dict = {'update_num': 0, 'hidden': 0.0}
        monitor_thread = threading.Thread(
            target=monitor_log_file,
            args=(str(log_file_path), process, trial_id, pbar_dict),
            daemon=True
        )
        monitor_thread.start()
        
        # 親側のトレースファイルを書いておく（デバッグ用）
        try:
            with open(trial_path / "optuna_parent_launched.txt", "w", encoding="utf-8") as f:
                f.write(f"launched: pid={process.pid}\n")
        except Exception:
            pass
        
        # ★Update進捗を Trial バーの postfix に表示
        last_shown = 0
        while process.poll() is None:
            current = pbar_dict.get('update_num', 0)
            hidden = pbar_dict.get('hidden', 0.0)
            
            if pbar_trials is not None:
                info = f"T{trial_id} LR={lr:.1e} ENT={ent:.1e} HB={hidden_bonus:.2f} DS={dist_diff_scale:.2f} | U={current}/150 H={hidden:.1f}"
                pbar_trials.set_postfix_str(info)
                last_shown = current
            
            time.sleep(0.5)
        
        # プロセス完了を待機（タイムアウト付き）
        try:
            process.wait(timeout=TIMEOUT_PER_TRIAL)
        except subprocess.TimeoutExpired:
            # tqdm.write(f"[Optuna] Trial {trial_id} timeout ({TIMEOUT_PER_TRIAL}s), killing...")
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
                # tqdm.write(f"[Optuna] Force kill failed")
        
        if process.returncode != 0 and process.returncode is not None:
            pass
            # msg = f"[Trial {trial_id:3d}] FAILED (exit code: {process.returncode})"
            # tqdm.write(msg)
        
        # モニタリングスレッドの終了を待つ
        monitor_thread.join(timeout=2)
    
    # ★最適化: ログファイルからメトリクスを抽出
    collected_metrics = []
    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Hidden:" in line:
                    match = re.search(r"Hidden:\s*(-?[\d\.]+)", line)
                    if match:
                        val = float(match.group(1))
                        collected_metrics.append(val)
    except Exception as e:
        pass
        # tqdm.write(f"[Warning] Failed to read metrics from {log_file_path}: {e}")

    # --- F. スコアの算出 ---
    # 学習後半の安定した隠蔽ステップ数（最後の5回分）の平均をスコアとする
    if len(collected_metrics) > 0:
        sample_size = min(len(collected_metrics), 5)
        final_score = np.mean(collected_metrics[-sample_size:])
        # tqdm.write(f"[Trial {trial_id:3d}] COMPLETED: score={final_score:7.2f} ({len(collected_metrics)} samples)")
    else:
        final_score = 0.0
        # tqdm.write(f"[Trial {trial_id:3d}] COMPLETED: score={final_score:7.2f} (no metrics)")

    # --- G. 自動クリーンアップ：一時フォルダを削除 ---
    if trial_path.exists():
        try:
            shutil.rmtree(trial_dir)
        except Exception as e:
            pass
            # tqdm.write(f"[Warning] Failed to clean up {trial_dir}: {e}")

    return float(final_score)

# ==========================================
# 5. メイン処理
# ==========================================
def main():
    # Linux環境のハング対策
    if platform.system() == "Linux":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    # Optuna Study の作成またはロード
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,
        direction="maximize",
        load_if_exists=True
    )
    
    print(f"Starting Hyperparameter Optimization (v2.8)...")
    print(f"  Mode:           {MODE}")
    print(f"  Target Script:  {RUNNER_SCRIPT}")
    print(f"  Total Trials:   {N_TRIALS}")
    print(f"  Auto-save:      {CSV_OUTPUT_PATH} (Every trial)")
    print("-" * 50)
    print(f"\n[Optuna] Starting trial loop...")
    
    # ★Trial進捗をtqdmで表示（dynamic_ncolsで端末幅に対応） 
    with tqdm(total=N_TRIALS, desc="T", position=0, leave=True, ncols=120,dynamic_ncols=True) as pbar_trials:
        for trial_num in range(N_TRIALS):
            trial = study.ask()
            # Trial開始を簡潔に表示（全体進捗付き）
            # print(f"\n[T{trial.number}/{N_TRIALS}]", flush=True)
            value = objective(trial, pbar_trials)  # pbar_trialsを渡す
            study.tell(trial, value)
            save_best_callback(study, trial)
            # Trial完了を簡潔に表示（ベスト情報付き）
            pbar_trials.update(1)
            best_info = f" | Best: T{study.best_trial.number}={study.best_value:.1f}" if study.best_value is not None else ""
            print(f"[T{trial.number}] Done: {value:.1f}{best_info}", flush=True)
    
    print("\n" + "="*50)
    print(f" OPTIMIZATION RESULTS ({MODE}) ")
    print("="*50)
    print(f"Best Trial:  {study.best_trial.number}")
    print(f"Best Value:  {study.best_value:.2f} steps hidden")
    print("-" * 50)

    # 最終的なCSV保存
    try:
        df = study.trials_dataframe()
        df.to_csv(CSV_OUTPUT_PATH, index=False)
        print(f"Final results saved to {CSV_OUTPUT_PATH}")
    except Exception as e:
        print(f"Failed to save final CSV: {e}")

if __name__ == "__main__":
    main()