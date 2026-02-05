# main19_reward_optimization.py
# 演習第19回：Optunaによる報酬パラメータの自動最適化 (v2.10 - Syntax Fix)
#
# 【修正点】
# 1. Refinementモードでの `IndentationError` を修正。
#    - `json.dump` をコメントアウトすると `with` ブロックが空になりエラーになっていたため、
#      `pass` を挿入して構文エラーを回避しました。
# 2. その他、デバッグ機能は維持。

import os
import sys
import time
import subprocess
import re
from pathlib import Path
import shutil
import statistics

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
STUDY_NAME = "HideAndSeek_Reward_Optimization"

# ★ モード設定 (Refinementでエラーが出ているとのことなのでRefinementに設定)
MODE = "refinement"
# MODE = "initial"

PRETRAINED_EXPERIMENT_NAME = "HideAndSeek_Transformer"

# 実験設定
N_TRIALS = 50
TIMEOUT_PER_TRIAL = 1200 
TOTAL_STEPS = 150000

def create_trial_env(trial, output_dir, mode="initial"):
    if not os.path.exists(TARGET_SCRIPT):
        raise FileNotFoundError(f"{TARGET_SCRIPT} が見つかりません。")

    with open(TARGET_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    # パラメータ探索範囲
    reward_survival = trial.suggest_float("REWARD_SURVIVAL", 0.0, 0.5)
    reward_distance = trial.suggest_float("REWARD_DISTANCE_COEFF", 0.01, 0.2)
    penalty_stagnation = trial.suggest_float("PENALTY_STAGNATION", -1.0, -0.1)
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00005, 0.001, log=True)

    # 置換リスト
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

    # ログ出力コードの注入
    log_pattern = r'(Loss: \{loss\.item\(\):\.3f\},)'
    if re.search(log_pattern, content):
        content = re.sub(log_pattern, r'\1 Entropy: {entropy.mean().item():.3f},', content)

    # パラメータ置換
    for key, val in replacements.items():
        pattern = fr"^(\s*{key}\s*=\s*)(.+?)(?=\s*(#.*)?$)"
        if not re.search(pattern, content, re.MULTILINE):
            print(f"[DEBUG] Warning: Regex pattern for {key} not found in script!")
        content = re.sub(pattern, fr"\g<1>{val}", content, flags=re.MULTILINE)

    # バッチサイズ計算パッチ
    content = re.sub(
        r"num_updates\s*=\s*TOTAL_TIMESTEPS\s*(//|/)\s*\(NUM_ENVS\s*\*\s*NUM_STEPS\)",
        "num_updates = int(TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS))",
        content
    )

    # モード別設定
    experiment_name = f"{STUDY_NAME}_Trial{trial.number}"
    if mode == "initial":
        content = re.sub(r'LOAD_EXISTING_MODELS\s*=\s*True', 'LOAD_EXISTING_MODELS = False', content)
        content = re.sub(r'FIXED_SEED\s*=\s*None', 'FIXED_SEED = 1', content)
    elif mode == "refinement":
        experiment_name = PRETRAINED_EXPERIMENT_NAME 
        content = re.sub(r'LOAD_EXISTING_MODELS\s*=\s*False', 'LOAD_EXISTING_MODELS = True', content)
        content = re.sub(r'FIXED_SEED\s*=\s*None', 'FIXED_SEED = 1', content)
        
        # 保存無効化: Syntax Errorを防ぐため pass を入れる
        content = content.replace("torch.save(", "pass; # torch.save(")
        content = content.replace("json.dump(", "pass; # json.dump(")

        # モデルファイルのコピー（.jsonはコピーしない）
        for agent_type in ["HIDER", "SEEKER"]:
            model_file = f"{PRETRAINED_EXPERIMENT_NAME}_{agent_type}.pt"
            if os.path.exists(model_file):
                shutil.copy(model_file, output_dir / model_file)
            # else:
            #    print(f"[DEBUG] Warning: Model file {model_file} not found in current dir.")

    # EXPERIMENT_NAME 置換
    content = re.sub(r'EXPERIMENT_NAME = ".*?"', f'EXPERIMENT_NAME = "{experiment_name}"', content)

    # スクリプト保存
    script_path = output_dir / f"trial_{trial.number}.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)

    return script_path

def objective(trial):
    trial_dir = Path("optuna_temp") / f"trial_{trial.number}"
    if trial_dir.exists(): shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        script_path = create_trial_env(trial, trial_dir, mode=MODE)
        
        # --- デバッグ: 最初の試行だけ、生成されたスクリプトの中身を確認 ---
        if trial.number == 0:
            print("\n" + "="*30)
            print("[DEBUG] CHECKING GENERATED SCRIPT (Head)")
            print("="*30)
            with open(script_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # パラメータ定義付近を表示
                for i, line in enumerate(lines[:150]):
                    if any(k in line for k in ["TOTAL_TIMESTEPS", "LEARNING_RATE", "ENT_COEF", "LOAD_EXISTING_MODELS"]):
                        print(f"{i+1}: {line.strip()}")
            print("="*30 + "\n")

        print(f"\n[Trial {trial.number}] Executing...")
        
        # 実行
        result = subprocess.run(
            [sys.executable, str(script_path.name)],
            cwd=str(trial_dir),
            capture_output=True,
            timeout=TIMEOUT_PER_TRIAL,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        # エラー確認
        if result.returncode != 0:
            print(f"  [ERROR] Trial {trial.number} FAILED with Exit Code {result.returncode}")
            print("  --- STDERR (FULL) ---")
            print(result.stderr)
            print("  ---------------------")
            return float("-inf")

        # ログ解析
        ep_lens = []
        
        # 出力ログの確認（デバッグ）
        # print("  [DEBUG] STDOUT Head:")
        # print('\n'.join(result.stdout.split('\n')[:5]))

        for line in result.stdout.split("\n"):
            m_l = re.search(r'EpLen:\s*([-+]?\d*\.?\d+)', line)
            if m_l: ep_lens.append(float(m_l.group(1)))
        
        if not ep_lens:
            print(f"  [WARNING] No EpLen data found. Did the training start?")
            print("  --- STDOUT (Tail 20 lines) ---")
            print('\n'.join(result.stdout.split('\n')[-20:]))
            print("  ------------------------------")
            return float("-inf")
        
        score = sum(ep_lens[-max(1, int(len(ep_lens)*0.2)):]) / max(1, int(len(ep_lens)*0.2))
        print(f"  -> Score: {score:.4f}")
        return score

    except subprocess.TimeoutExpired:
        print(f"  [ERROR] Timeout!")
        return float("-inf")
    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
        return float("-inf")
    finally:
        if trial_dir.exists():
            try: shutil.rmtree(trial_dir)
            except: pass

def main():
    print("="*70)
    print("Optuna Optimization (v2.10 Syntax Fix)")
    print("="*70)
    
    if not os.path.exists(TARGET_SCRIPT):
        print(f"Error: {TARGET_SCRIPT} not found.")
        return

    Path("optuna_results").mkdir(exist_ok=True)
    study = optuna.create_study(direction="maximize")
    
    # 探索実行
    study.optimize(objective, n_trials=N_TRIALS)

if __name__ == "__main__":
    main()