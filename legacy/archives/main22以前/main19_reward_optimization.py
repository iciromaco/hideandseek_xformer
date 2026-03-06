# main22_reward_optimization.py

# uv add optuna pandas

import os
import sys
import time
import subprocess
import re
from pathlib import Path
import shutil

# --- 必要なライブラリのチェック ---
try:
    import optuna
    import pandas as pd
except ImportError:
    print("Error: 必要なライブラリが見つかりません。")
    print("以下のコマンドでインストールしてください:")
    print("uv add optuna pandas")
    sys.exit(1)

# ==========================================
# 1. 実験設定
# ==========================================
TARGET_SCRIPT = "main18_optimization.py"
STUDY_NAME = "HideAndSeek_Reward_Optimization_v3"

# ★重要: モード選択
# MODE = "initial"
MODE = "refinement"

PRETRAINED_EXPERIMENT_NAME = "HideAndSeek_Transformer"

# 実験設定
N_TRIALS = 50           # 探索空間が広がったため少し多めに
TIMEOUT_PER_TRIAL = 900 # 15分
TOTAL_STEPS = 100000    # 10万ステップで傾向を見る

def create_trial_script(trial, output_dir, mode="initial"):
    """Optunaの試行パラメータを埋め込んだ一時スクリプトを作成"""
    if not os.path.exists(TARGET_SCRIPT):
        raise FileNotFoundError(f"{TARGET_SCRIPT} が見つかりません。")

    # --- 探索空間の定義 (v3.0) ---
    
    # 報酬: 生存を重視し、距離は廃止
    reward_survival = trial.suggest_float("REWARD_SURVIVAL", 5.0, 20.0) # 上限を5.0に拡大
    reward_distance = trial.suggest_float("REWARD_DISTANCE_COEFF", 0.5, 10.0)
    penalty_stagnation = trial.suggest_float("PENALTY_STAGNATION", -5.0, -1.5)
    
    # 学習制御: ここが重要
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 5e-2, log=True)

    # --- スクリプトの書き換え ---
    with open(TARGET_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    # パラメータ置換
    replacements = {
        "REWARD_SURVIVAL": f"{reward_survival:.5f}",
        "REWARD_DISTANCE_COEFF": f"{reward_distance:.5f}",
        "PENALTY_STAGNATION": f"{penalty_stagnation:.5f}",
        "LEARNING_RATE": f"{lr:.8f}",
        "ENT_COEF": f"{ent_coef:.8f}",
        "TOTAL_TIMESTEPS": str(int(TOTAL_STEPS)),
        "TRACK_WANDB": "False",
        "USE_VIEWER": "False",
    }

    # モード設定
    if mode == "initial":
        replacements["LOAD_EXISTING_MODELS"] = "False"
        replacements["FIXED_SEED"] = "1"
        experiment_name = f"{STUDY_NAME}_Trial{trial.number}"
        
    elif mode == "refinement":
        replacements["LOAD_EXISTING_MODELS"] = "True"
        replacements["FIXED_SEED"] = "1" 
        experiment_name = PRETRAINED_EXPERIMENT_NAME
        
        # モデル上書き防止
        content = content.replace("torch.save(", "# torch.save(")
        # チェックポイント保存防止
        content = re.sub(
            r"json\.dump\({'global_step': global_step}, f\)",
            "pass # json.dump skipped",
            content
        )
        # 継続学習時でもStepは0から評価
        content = re.sub(
            r"global_step\s*=\s*checkpoint_data\.get\('global_step',\s*0\)",
            "global_step = 0 # Forced reset",
            content
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 文字列置換実行
    for key, val in replacements.items():
        pattern = fr"^(\s*{key}\s*=\s*)(.+?)(?=\s*(#.*)?$)"
        if not re.search(pattern, content, re.MULTILINE):
            pass 
        replacement = fr"\g<1>{val}"
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # 実験名
    content = re.sub(
        r'EXPERIMENT_NAME = ".*?"',
        f'EXPERIMENT_NAME = "{experiment_name}"',
        content
    )
    
    # バッチサイズ計算パッチ
    content = re.sub(
        r"num_updates\s*=\s*TOTAL_TIMESTEPS\s*(//|/)\s*\(NUM_ENVS\s*\*\s*NUM_STEPS\)",
        "num_updates = int(TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS))",
        content
    )

    script_path = output_dir / f"trial_{trial.number}.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)

    return script_path

def objective(trial):
    """目的関数: エピソード長(EpLen)最大化"""
    trial_dir = Path("optuna_results") / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = create_trial_script(trial, trial_dir, mode=MODE)
    
    print(f"\n[Trial {trial.number}]")
    print(f"  SURVIVAL={trial.params['REWARD_SURVIVAL']:.4f}, PENALTY={trial.params['PENALTY_STAGNATION']:.4f}")
    print(f"  LR={trial.params['lr']:.2e}, ENT={trial.params['ent_coef']:.2e}")
    
    try:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            timeout=TIMEOUT_PER_TRIAL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        
        if result.returncode != 0:
            print(f"  -> Error in trial {trial.number}")
            err_msg = result.stderr.split('\n')[-5:]
            for line in err_msg:
                if line.strip(): print(f"    {line}")
            return 0.0

        # EpLen抽出
        ep_lens = []
        pattern = r'EpLen:\s*([-+]?\d*\.?\d+)'
        
        for line in result.stdout.split("\n"):
            match = re.search(pattern, line)
            if match:
                try:
                    ep_lens.append(float(match.group(1)))
                except ValueError: pass
        
        if not ep_lens:
            print(f"  -> No EpLen found.")
            return 0.0
        
        # 後半の平均を採用
        final_score = sum(ep_lens[-5:]) / min(len(ep_lens), 5)
        print(f"  -> Score (EpLen): {final_score:.1f}")
        
        return final_score

    except subprocess.TimeoutExpired:
        print(f"  -> Timeout!")
        return 0.0
    except Exception as e:
        print(f"  -> Exception: {e}")
        return 0.0
    finally:
        if os.path.exists(script_path):
            try: os.remove(script_path)
            except: pass

def main():
    print("="*70)
    print("Optuna Hyperparameter Optimization (Survival Only)")
    print("Objective: Maximize Average Episode Length")
    print("="*70)

    Path("optuna_results").mkdir(exist_ok=True)
    db_name = f"{STUDY_NAME}_{MODE}.db"
    
    study = optuna.create_study(
        study_name=f"{STUDY_NAME}_{MODE}",
        direction="maximize",
        storage=f"sqlite:///optuna_results/{db_name}",
        load_if_exists=True
    )

    print(f"Optimization started... (Mode: {MODE})")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*70)
    print("OPTIMIZATION FINISHED")
    print("="*70)
    print(f"Best Score (EpLen): {study.best_value:.2f}")
    print("\nBest Params:")
    for key, val in study.best_params.items():
        if "lr" in key or "ent" in key:
            print(f"  {key}: {val:.8f}")
        else:
            print(f"  {key}: {val:.5f}")
    
    print(f"  REWARD_DISTANCE_COEFF: 0.0 (Fixed)")

    # CSV保存
    try:
        df = study.trials_dataframe()
        csv_path = Path("optuna_results") / f"{STUDY_NAME}_{MODE}_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")
    except: pass

if __name__ == "__main__":
    main()