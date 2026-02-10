# main23_optuna_search.py
# 演習第23回：Cos勾配報酬・チーム学習のためのパラメータ自動最適化 (v23.99)
#
# 【修正内容】
# - 同期バグへの対策: 子スクリプト (main23_runner_cos.py) の統計収集同期修正版を使用。
# - デバッグ機能の超強化: 
#   - ログの全行を走査し、正規表現がヒットした箇所をリアルタイムに表示。
#   - Hidden スコアが 0.0 の場合でも、なぜ 0.0 なのか（出力自体が 0 なのか、パースミスか）を明示。
# - 堅牢なパース: "Hidden: 123.4" だけでなく、多様な出力形式を許容。
# - 【NEW】解析スクリプトとの互換性: 保存ファイル名を results_{MODE}.csv に変更し、解析スクリプトから直接読み込めるように修正。

import os
import sys
import time
import subprocess
import re
import platform
from pathlib import Path
import numpy as np

try:
    import optuna
except ImportError:
    print("Please run: uv add optuna pandas")
    sys.exit(1)

# ==========================================
# 1. 実験設定
# ==========================================
TARGET_SCRIPT = "main23_runner_cos.py"
STUDY_NAME = "Layer23_TeamStrategy_Optimization"

MODE = "refinement" # "initial" 
N_TRIALS = 50           
TIMEOUT_PER_TRIAL = 1200 
TOTAL_STEPS = 100000    

def get_train_target():
    if not os.path.exists(TARGET_SCRIPT):
        return "HIDER"
    with open(TARGET_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'TRAIN_TARGET\s*=\s*"(HIDER|SEEKER)"', content)
    return match.group(1) if match else "HIDER"

def create_trial_script(trial, output_dir, mode="initial"):
    if not os.path.exists(TARGET_SCRIPT):
        raise FileNotFoundError(f"{TARGET_SCRIPT} not found.")

    reward_hidden = trial.suggest_float("REWARD_HIDDEN_BONUS", 0.1, 2.0)
    cos_penalty_scale = trial.suggest_float("COS_PENALTY_SCALE", 0.5, 5.0)
    lr = trial.suggest_float("LEARNING_RATE", 5e-5, 5e-4, log=True)
    ent_coef = trial.suggest_float("ENT_COEF", 1e-4, 5e-3, log=True)

    with open(TARGET_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    replacements = {
        "REWARD_HIDDEN_BONUS": f"{reward_hidden:.5f}",
        "COS_PENALTY_SCALE": f"{cos_penalty_scale:.5f}",
        "LEARNING_RATE": f"{lr:.8f}",
        "ENT_COEF": f"{ent_coef:.8f}",
        "TOTAL_TIMESTEPS": str(int(TOTAL_STEPS)),
        "EXECUTION_MODE": '"TRAIN"',
        "SAVE_MODEL": "False", 
        "TRACK_WANDB": "True", 
        "FIXED_SEED": "1",
        "TRIAL_MODE": "True"
    }
    
    replacements["LOAD_EXISTING_MODELS"] = "False" if mode == "initial" else "True"

    for key, val in replacements.items():
        pattern = fr"^(\s*{key}\s*=\s*)(.+?)(?=\s*(#.*|;.*|$))"
        content = re.sub(pattern, fr"\g<1>{val}", content, flags=re.MULTILINE)

    content = re.sub(
        r'EXPERIMENT_NAME = ".*?"', 
        f'EXPERIMENT_NAME = "{STUDY_NAME}_{mode}_{trial.number}"', 
        content
    )

    script_path = output_dir / f"trial_{trial.number}.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    return script_path

def objective(trial):
    target = get_train_target()
    trial_dir = Path("optuna_results_layer23") / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    script_path = create_trial_script(trial, trial_dir, mode=MODE)
    
    print(f"\n[Trial {trial.number}] START (Target: {target})")
    
    try:
        env_vars = os.environ.copy()
        env_vars["PYTHONUNBUFFERED"] = "1"
        if platform.processor() != 'arm':
            for k in ["OMP", "MKL", "OPENBLAS", "VECLIB", "NUMEXPR"]:
                env_vars[f"{k}_NUM_THREADS"] = "1"
        
        result = subprocess.run(
            [sys.executable, str(script_path)], 
            capture_output=True, 
            timeout=TIMEOUT_PER_TRIAL, 
            text=True, 
            encoding="utf-8", 
            errors="replace", 
            env=env_vars
        )
        
        # 指標の抽出 (Hidden または Caught)
        metrics = []
        metric_name = "Hidden" if target == "HIDER" else "Caught"
        # ログ全体を走査
        pattern = fr'{metric_name}:\s*([-+]?\d*\.?\d+|nan)'
        
        raw_log_lines = result.stdout.strip().split('\n')
        found_count = 0
        
        for line in raw_log_lines:
            match = re.search(pattern, line)
            if match:
                val_str = match.group(1).lower()
                val = 0.0 if val_str == 'nan' else float(val_str)
                metrics.append(val)
                found_count += 1

        # --- デバッグ出力セクション ---
        print(f"  --- Debug Report (Trial {trial.number}) ---")
        print(f"    Total Logs Scanned: {len(raw_log_lines)}")
        print(f"    Matches Found: {found_count}")
        if found_count > 0:
            print(f"    Last 3 Values: {metrics[-3:]}")
        else:
            print(f"    [!] No '{metric_name}' tag detected. Checking raw output tail:")
            for line in raw_log_lines[-5:]:
                print(f"        {line}")
        
        if result.returncode != 0:
            print(f"    [!] Subprocess failed (code {result.returncode})")
            if result.stderr:
                print(f"    [!] STDERR: {result.stderr.strip().splitlines()[-1]}")
        print("  -------------------------------------------")

        if not metrics:
            return 0.0
            
        final_score = sum(metrics[-5:]) / min(len(metrics), 5)
        print(f"  -> Trial Finished. Final Score: {final_score:.1f}")
        return final_score

    except Exception as e:
        print(f"  -> Exception: {e}")
        return 0.0
    finally:
        if os.path.exists(script_path):
            try:
                os.remove(script_path)
            except:
                pass

def main():
    target = get_train_target()
    print("="*70)
    print(f"Optuna Optimization for Layer 23 (Target: {target})")
    print(f"Objective: Maximize {'Stealth' if target=='HIDER' else 'Capture'} Efficiency")
    print("="*70)
    
    Path("optuna_results_layer23").mkdir(exist_ok=True)
    
    db_path = f"sqlite:///optuna_results_layer23/{STUDY_NAME}.db"
    study = optuna.create_study(
        study_name=f"{STUDY_NAME}_{target}_{MODE}", 
        direction="maximize", 
        storage=db_path, 
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\nOPTIMIZATION FINISHED")
    if study.trials:
        print(f"Best Value: {study.best_value:.2f}")
        print("Best Params:")
        for key, val in study.best_params.items():
            print(f"  {key}: {val}")
            
    try:
        df = study.trials_dataframe()
        # ★修正: 解析スクリプトが期待するファイル名 (results_initial.csv) に合わせて保存
        csv_path = Path("optuna_results_layer23") / f"results_{MODE}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
    except Exception as e:
        print(f"Failed to save CSV: {e}")

if __name__ == "__main__":
    main()