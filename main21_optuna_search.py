# main21_optuna_search.py
# 演習第21回：Layer 0 (逃走) 専用のパラメータ自動最適化
#
# 【修正(v21.52) - ログ出力の抑制と安定化】
# - ログ表示のオフ:
#   - リアルタイム出力を廃止し、subprocess.run による静かな実行に変更。
#   - tqdm 等の進捗表示がメインターミナルを汚さないように修正。
# - Windows環境の高速化対策を維持:
#   - OMP_NUM_THREADS=1 等を設定し、CPUコアの奪い合いを防止。
# - 安全な置換ロジックの継続:
#   - start_global_step の定義不備（NameError）対策を維持。
#
# 【実行準備】
# uv add optuna pandas

import os
import sys
import time
import subprocess
import re
import platform
from pathlib import Path

try:
    import optuna
except ImportError:
    sys.exit("Please run: uv add optuna pandas")

# ==========================================
# 1. 実験設定
# ==========================================
TARGET_SCRIPT = "main21_layer0_runner.py"
STUDY_NAME = "Layer0_Runner_Optimization"

# ★モード設定 ("initial": 新規学習からの最適化 / "refinement": 既存モデルを微調整)
MODE = "initial" 

N_TRIALS = 30           
TIMEOUT_PER_TRIAL = 900 # 15分
TOTAL_STEPS = 100000    

def create_trial_script(trial, output_dir, mode="initial"):
    if not os.path.exists(TARGET_SCRIPT):
        raise FileNotFoundError(f"{TARGET_SCRIPT} が見つかりません。")

    # --- 探索空間 ---
    reward_survival = trial.suggest_float("REWARD_SURVIVAL_SCALE", 0.1, 10.0)
    penalty_stagnation = trial.suggest_float("PENALTY_STAGNATION_FORCE", -5.0, -0.1)
    lr = trial.suggest_float("LEARNING_RATE", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ENT_COEF", 1e-5, 1e-2, log=True)

    # --- スクリプトの読み込み ---
    with open(TARGET_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. 基本パラメータ・フラグの置換
    replacements = {
        "REWARD_SURVIVAL_SCALE": f"{reward_survival:.5f}",
        "PENALTY_STAGNATION_FORCE": f"{penalty_stagnation:.5f}",
        "LEARNING_RATE": f"{lr:.8f}",
        "ENT_COEF": f"{ent_coef:.8f}",
        "TOTAL_TIMESTEPS": str(int(TOTAL_STEPS)),
        "EXECUTION_MODE": '"TRAIN"',
        "SAVE_MODEL": "False",      # 各Trialでの上書きを禁止
        "TRACK_WANDB": "True",      # WandB 記録を有効化
        "FIXED_SEED": "1",          # 試行間の条件を同一にする
    }

    # モードに応じたロード設定
    replacements["LOAD_EXISTING_MODELS"] = "False" if mode == "initial" else "True"

    # 置換実行
    for key, val in replacements.items():
        pattern = fr"^(\s*{key}\s*=\s*)(.+?)(?=\s*(#.*|;.*|$))"
        content = re.sub(pattern, fr"\g<1>{val}", content, flags=re.MULTILINE)

    # 2. 実験名の変更
    content = re.sub(
        r'EXPERIMENT_NAME = ".*?"',
        f'EXPERIMENT_NAME = "{STUDY_NAME}_{mode}_{trial.number}"',
        content
    )

    # 3. 評価指標リセットパッチ
    # global_step と start_global_step を確実に 0 にリセット
    content = re.sub(
        r"global_step\s*=\s*data\.get\('global_step',\s*0\)",
        "global_step = 0",
        content
    )
    content = re.sub(
        r"start_global_step\s*=\s*(global_step|\d+)",
        "start_global_step = 0",
        content
    )

    # 4. チェックポイント保存ブロックの安全な除去
    checkpoint_block_pattern = re.compile(
        r"(\s*)try:.*?with open\(checkpoint_path, 'w'\).*?except:.*?pass",
        re.DOTALL
    )
    content = checkpoint_block_pattern.sub(r"\1pass # Checkpoint disabled", content)

    script_path = output_dir / f"trial_{trial.number}.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)

    return script_path

def objective(trial):
    """目的関数: エピソード長(EpLen)の最大化 (静音実行版)"""
    trial_dir = Path("optuna_results_layer0") / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = create_trial_script(trial, trial_dir, mode=MODE)
    
    print(f"\n[Trial {trial.number}] START (MODE={MODE})")
    print(f"  Params: SURVIVAL={trial.params['REWARD_SURVIVAL_SCALE']:.4f}, PENALTY={trial.params['PENALTY_STAGNATION_FORCE']:.4f}, LR={trial.params['LEARNING_RATE']:.2e}")
    
    try:
        # 環境変数の設定 (Windows/Intel環境での高速化対策を継承)
        env = os.environ.copy()
        if platform.processor() != 'arm':
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["VECLIB_MAXIMUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
        
        # 学習実行 (capture_output=True によりターミナルには何も表示されない)
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
            print(f"  -> Trial Failed with exit code {result.returncode}")
            # エラー時のみ詳細を出す
            print("="*30 + " [STDERR] " + "="*30)
            print(result.stderr[-500:]) # 末尾の500文字程度を表示
            return 0.0

        # 出力から EpLen を抽出
        ep_lens = []
        pattern = r'EpLen:\s*([-+]?\d*\.?\d+)'
        for line in result.stdout.split("\n"):
            match = re.search(pattern, line)
            if match:
                try: ep_lens.append(float(match.group(1)))
                except: pass

        if not ep_lens:
            print("  -> No EpLen found in trial output.")
            return 0.0
        
        final_score = sum(ep_lens[-5:]) / min(len(ep_lens), 5)
        print(f"  -> Trial Finished. Final Score (EpLen): {final_score:.1f}")
        return final_score

    except subprocess.TimeoutExpired:
        print("  -> Timeout (900s reached)")
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
    print(f"Optuna Optimization for Layer 0 (Runner) - MODE: {MODE}")
    print(f"Target: {TARGET_SCRIPT}")
    print("="*70)

    Path("optuna_results_layer0").mkdir(exist_ok=True)
    
    study = optuna.create_study(
        study_name=f"{STUDY_NAME}_{MODE}",
        direction="maximize",
        storage=f"sqlite:///optuna_results_layer0/{STUDY_NAME}_{MODE}.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*70)
    print("OPTIMIZATION FINISHED")
    print("="*70)
    if study.trials:
        print(f"Best EpLen: {study.best_value:.2f}")
        print("Best Params:")
        for key, val in study.best_params.items():
            print(f"  {key}: {val}")

    try:
        df = study.trials_dataframe()
        df.to_csv(Path("optuna_results_layer0") / f"results_{MODE}.csv", index=False)
    except: pass

if __name__ == "__main__":
    main()