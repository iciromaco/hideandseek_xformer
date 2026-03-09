import optuna
import pandas as pd

STUDY_NAME = "WallStickPenaltyOpt"
STORAGE_PATH = "sqlite:///optuna_wall_stick.db"


def main():
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
    print(f"Trials: {len(study.trials)}")
    print(f"Best params: {study.best_params}")
    print(f"Best HideRate: {study.best_value}")
    print("\n--- All Trials ---")
    df = study.trials_dataframe()
    print(df[["number", "value", "params_rw_hide_wall_stick_penalty", "state"]])
    # CSV保存も可能
    df.to_csv("optuna_wall_stick_trials.csv", index=False)

if __name__ == "__main__":
    main()
