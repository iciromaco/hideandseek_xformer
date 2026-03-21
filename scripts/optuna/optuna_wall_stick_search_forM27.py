# optuna_wall_stick_search_forM27.py

import os
import re
import shutil
import subprocess

import optuna

TOML_PATH = "configs/hparams_main27.toml"
BACKUP_PATH = "configs/hparams_main27.toml.bak"
TARGET_SCRIPT = "main27_train_final.py"
N_TRIALS = 20
TIMEOUT_PER_TRIAL = 1800  # 30分
STUDY_NAME = "WallStickPenaltyOpt"
STORAGE_PATH = "sqlite:///optuna_wall_stick.db"
RESULTS_PATH = "optuna_wall_stick_results.txt"


# [runtime.optuna]セクション内だけを書き換える


def set_toml_param_in_section(section, param, value):
    with open(TOML_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    in_section = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"[{section}]"):
            in_section = True
            continue
        # セクション終端検出
        if in_section and line.strip().startswith("[") and not line.strip().startswith(f"[{section}]"):
            in_section = False
        if in_section and line.strip().startswith(f"{param} ="):
            lines[i] = f"{param} = {value}\n"
    with open(TOML_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)


def restore_toml():
    if os.path.exists(BACKUP_PATH):
        shutil.copy(BACKUP_PATH, TOML_PATH)


def objective(trial):
    penalty = trial.suggest_float("rw_hide_wall_stick_penalty", 0.01, 0.5)
    print(f"[Trial {trial.number}] penalty={penalty:.4f} 開始")
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(TOML_PATH, BACKUP_PATH)
    set_toml_param_in_section("runtime.optuna", "rw_hide_wall_stick_penalty", penalty)
    try:
        print(f"[Trial {trial.number}] サブプロセス実行: uv run python {TARGET_SCRIPT} --profile optuna")
        result = subprocess.run(
            ["uv", "run", "python", TARGET_SCRIPT, "--profile", "optuna"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_PER_TRIAL,
        )
        print(f"[Trial {trial.number}] サブプロセス終了 (returncode={result.returncode})")
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
        storage=STORAGE_PATH,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)
    print("Best params:", study.best_params)
    print("Best HideRate:", study.best_value)
    # 結果をファイル保存
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Best params: {study.best_params}\n")
        f.write(f"Best HideRate: {study.best_value}\n")
    # 全履歴も保存
    try:
        df = study.trials_dataframe()
        df.to_csv("optuna_wall_stick_trials.csv", index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
