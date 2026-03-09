import os
import shutil
import subprocess
import re
import optuna

TOML_PATH = "configs/hparams_main27.toml"
BACKUP_PATH = "configs/hparams_main27.toml.bak.optuna"
TARGET_SCRIPT = "main27_train_final.py"
STUDY_NAME = "optuna_wall_avoid"
TIMEOUT_PER_TRIAL = 1200  # 秒

def set_toml_param_in_section(section, key, value):
    with open(TOML_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    in_section = False
    section_header = f"[{section}]\n"
    for i, line in enumerate(lines):
        if line.strip().startswith("[") and line.strip().endswith("]"):
            in_section = (line == section_header)
        if in_section and line.strip().startswith(f"{key} ="):
            lines[i] = f"{key} = {value}\n"
            break
    else:
        # セクションがなければ末尾に追加
        if not any(l == section_header for l in lines):
            lines.append(f"\n{section_header}")
        lines.append(f"{key} = {value}\n")
    with open(TOML_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

def restore_toml():
    if os.path.exists(BACKUP_PATH):
        shutil.copy(BACKUP_PATH, TOML_PATH)

def objective(trial):
    penalty = trial.suggest_float(
        "rw_wall_avoid_penalty", 0.01, 0.5
    )
    print(f"[Trial {trial.number}] penalty={penalty:.4f} 開始")
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(TOML_PATH, BACKUP_PATH)
    set_toml_param_in_section(
        "runtime.optuna", "rw_wall_avoid_penalty", penalty
    )
    try:
        print(
            f"[Trial {trial.number}] サブプロセス実行: uv run python {TARGET_SCRIPT} --profile optuna"
        )
        result = subprocess.run(
            ["uv", "run", "python", TARGET_SCRIPT, "--profile", "optuna"],
            capture_output=True, text=True, timeout=TIMEOUT_PER_TRIAL
        )
        print(
            f"[Trial {trial.number}] サブプロセス終了 (returncode={{result.returncode}})"
        )
        hide_rates = []
        for line in result.stdout.splitlines():
            m = re.search(r"HideRate=([0-9.]+)", line)
            if m:
                hide_rates.append(float(m.group(1)))
        if hide_rates:
            result_val = max(hide_rates[-5:])
            print(f"[Trial {trial.number}] HideRate={result_val:.4f}")
            return result_val
        else:
            print(f"[Trial {trial.number}] HideRate=0.0 (no data)")
            return 0.0
    except Exception as e:
        print(f"[Trial {trial.number}] Trial failed: {e}")
        return 0.0
    finally:
        restore_toml()

def main():
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=f"sqlite:///optuna_wall_avoid.db",
        load_if_exists=True,
    )
    try:
        study.optimize(objective, n_trials=100)
    except KeyboardInterrupt:
        print("\n--- 強制終了 ---")
    best = study.best_trial
    print("\nBest trial:")
    print(f"number={best.number}")
    print(f"rw_wall_avoid_penalty={best.params['rw_wall_avoid_penalty']}")
    print(f"HideRate={best.value}")

if __name__ == "__main__":
    main()
