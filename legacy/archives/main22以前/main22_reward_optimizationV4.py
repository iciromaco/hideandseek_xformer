# main22_reward_optimizationV4.py

import os
import platform
import re
import subprocess
import sys
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

# MODE = "initial"
MODE = "refinement"
N_TRIALS = 30
TIMEOUT_PER_TRIAL = 900
TOTAL_STEPS = 100000


def create_trial_script(trial, output_dir, mode="initial"):
    if not os.path.exists(TARGET_SCRIPT):
        raise FileNotFoundError(f"{TARGET_SCRIPT} が見つかりません。")

    # --- 探索空間の定義 ---
    # 視界内限定になったため、報酬スケールを大きめに設定可能に
    # reward_survival = trial.suggest_float("REWARD_SURVIVAL_SCALE", 0.5, 10.0)
    # reward_dist_diff = trial.suggest_float("REWARD_DISTANCE_DIFF_SCALE", 0.5, 10.0)
    # penalty_stagnation = trial.suggest_float("PENALTY_STAGNATION_FORCE", -5.0, -0.1)
    # lr = trial.suggest_float("LEARNING_RATE", 5e-5, 5e-4, log=True) # 範囲をやや絞る
    # ent_coef = trial.suggest_float("ENT_COEF", 1e-5, 5e-3, log=True)

    # --- 探索空間の定義（V4提案） ---

    # 1. 生存報酬：最適値(16-18)が判明したため、範囲を維持しつつ精度を高める
    # Previous: 5.0 ~ 20.0 -> New: 10.0 ~ 22.0 (最適値を中心に)
    reward_survival = trial.suggest_float("REWARD_SURVIVAL_SCALE", 10.0, 22.0)

    # 2. 学習率：上限(1e-3)に張り付いているため、さらに引き上げる
    # Previous: 1e-4 ~ 1e-3 -> New: 5e-4 ~ 5e-3 (log=True)
    lr = trial.suggest_float("LEARNING_RATE", 5e-4, 5e-3, log=True)

    # 3. エントロピー：GBRの影響度が最大。低値を強く推奨しているため高領域を捨てる
    # Previous: 1e-5 ~ 5e-3 -> New: 1e-6 ~ 1e-4 (log=True)
    ent_coef = trial.suggest_float("ENT_COEF", 1e-6, 1e-4, log=True)

    # 4. 停滞ペナルティ：-1.7付近も視野に入れるため上限を緩和
    # Previous: -5.0 ~ -1.5 -> New: -4.0 ~ -1.0
    penalty_stagnation = trial.suggest_float("PENALTY_STAGNATION_FORCE", -4.0, -1.0)

    # 5. 距離報酬：大きな変更なし
    reward_dist_diff = trial.suggest_float("REWARD_DISTANCE_DIFF_SCALE", 0.5, 10.0)

    with open(TARGET_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()

    replacements = {
        "REWARD_SURVIVAL_SCALE": f"{reward_survival:.5f}",
        "REWARD_DISTANCE_DIFF_SCALE": f"{reward_dist_diff:.5f}",
        "PENALTY_STAGNATION_FORCE": f"{penalty_stagnation:.5f}",
        "LEARNING_RATE": f"{lr:.8f}",
        "ENT_COEF": f"{ent_coef:.8f}",
        "TOTAL_TIMESTEPS": str(int(TOTAL_STEPS)),
        "EXECUTION_MODE": '"TRAIN"',
        "SAVE_MODEL": "False",
        "TRACK_WANDB": "True",
        "FIXED_SEED": "1",
    }

    replacements["LOAD_EXISTING_MODELS"] = "False" if mode == "initial" else "True"

    for key, val in replacements.items():
        pattern = rf"^(\s*{key}\s*=\s*)(.+?)(?=\s*(#.*|;.*|$))"
        content = re.sub(pattern, rf"\g<1>{val}", content, flags=re.MULTILINE)

    content = re.sub(
        r'EXPERIMENT_NAME = ".*?"',
        f'EXPERIMENT_NAME = "{STUDY_NAME}_{mode}_{trial.number}"',
        content,
    )
    content = re.sub(r"global_step\s*=\s*data\.get\('global_step',\s*0\)", "global_step = 0", content)
    content = re.sub(r"start_global_step\s*=\s*(global_step|\d+)", "start_global_step = 0", content)

    checkpoint_block_pattern = re.compile(r"(\s*)try:.*?with open\(checkpoint_path, 'w'\).*?except:.*?pass", re.DOTALL)
    content = checkpoint_block_pattern.sub(r"\1pass # Checkpoint disabled", content)

    script_path = output_dir / f"trial_{trial.number}.py"
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(content)
    return script_path


def objective(trial):
    trial_dir = Path("optuna_results_layer0") / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    script_path = create_trial_script(trial, trial_dir, mode=MODE)

    print(f"\n[Trial {trial.number}] START (MODE={MODE})")
    print(f"  Params: SURVIVAL={trial.params['REWARD_SURVIVAL_SCALE']:.3f}, DIST_DIFF={trial.params['REWARD_DISTANCE_DIFF_SCALE']:.3f}")

    try:
        env = os.environ.copy()
        if platform.processor() != "arm":
            for k in ["OMP", "MKL", "OPENBLAS", "VECLIB", "NUMEXPR"]:
                env[f"{k}_NUM_THREADS"] = "1"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            timeout=TIMEOUT_PER_TRIAL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        if result.returncode != 0:
            print(f"  -> Trial Failed with exit code {result.returncode}")
            return 0.0

        ep_lens = []
        pattern = r"EpLen:\s*([-+]?\d*\.?\d+)"
        for line in result.stdout.split("\n"):
            match = re.search(pattern, line)
            if match:
                try:
                    ep_lens.append(float(match.group(1)))
                except:
                    pass

        if not ep_lens:
            return 0.0
        final_score = sum(ep_lens[-5:]) / min(len(ep_lens), 5)
        print(f"  -> Trial Finished. Final Score (EpLen): {final_score:.1f}")
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
    print("=" * 70)
    print(f"Optuna Optimization for Layer 0 (Runner) - MODE: {MODE}")
    print("Objective: Maximize EpLen with Visibility-Gated Distance Reward")
    print("=" * 70)
    Path("optuna_results_layer0").mkdir(exist_ok=True)
    study = optuna.create_study(
        study_name=f"{STUDY_NAME}_{MODE}",
        direction="maximize",
        storage=f"sqlite:///optuna_results_layer0/{STUDY_NAME}_{MODE}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)
    print("\n" + "=" * 70 + "\nOPTIMIZATION FINISHED\n" + "=" * 70)
    if study.trials:
        print(f"Best EpLen: {study.best_value:.2f}\nBest Params:")
        for key, val in study.best_params.items():
            print(f"  {key}: {val}")
    try:
        df = study.trials_dataframe()
        df.to_csv(Path("optuna_results_layer0") / f"results_{MODE}.csv", index=False)
    except:
        pass


if __name__ == "__main__":
    main()
