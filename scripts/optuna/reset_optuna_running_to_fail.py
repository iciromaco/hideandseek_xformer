import optuna

STUDY_NAME = "WallStickPenaltyOpt"
STORAGE_PATH = "sqlite:///optuna_wall_stick.db"

study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_PATH)
count = 0
for t in study.get_trials(deepcopy=False):
    if t.state.name == "RUNNING":
        t.state = optuna.trial.TrialState.FAIL
        count += 1
print(f"RUNNING状態のトライアルを{count}件、FAILに変更しました。")
