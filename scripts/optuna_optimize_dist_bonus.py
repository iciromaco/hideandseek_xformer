import argparse
import numpy as np
import optuna
import os
import shutil
import sys
import subprocess
import torch
import json

# Ensure project root is on sys.path so `from src...` works when run from anywhere
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.envs.hns28_environment import TeamCosEnv


def run_trial(trial, args):
    val = trial.suggest_float("dist_bonus_scale", args.min_val, args.max_val)
    # clean checkpoints to avoid loading models from previous runs
    if args.clean_checkpoints:
        cp_root = os.path.join(os.getcwd(), "checkpoints")
        if os.path.isdir(cp_root):
            for name in os.listdir(cp_root):
                if name == "keepGoodpkl":
                    continue
                path = os.path.join(cp_root, name)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception:
                    pass

        # If requested, run a full training job (profile) that will create checkpoint
        if args.run_training:
            # ensure env var for TeamCosEnv consumption during training
            envp = os.environ.copy()
            envp["DIST_BONUS_SCALE"] = str(val)
            train_cmd = ["python3", "main28_train_final.py", "-p", args.train_profile]
            subprocess.run(train_cmd, cwd=ROOT, env=envp, check=True)

            # try to read local wandb summary for the most recent run and extract learnable_seen_rate
            if args.use_wandb_summary:
                wb_dir = os.path.join(os.getcwd(), "wandb")
                try:
                    runs = []
                    if os.path.isdir(wb_dir):
                        for rn in os.listdir(wb_dir):
                            run_path = os.path.join(wb_dir, rn)
                            files_dir = os.path.join(run_path, "files")
                            summary_path = os.path.join(files_dir, "wandb-summary.json")
                            if os.path.isfile(summary_path):
                                runs.append((os.path.getmtime(summary_path), summary_path))
                    if runs:
                        runs.sort(reverse=True)
                        summary_path = runs[0][1]
                        try:
                            with open(summary_path, "r") as f:
                                data = json.load(f)
                            if "train/learnable_seen_rate" in data:
                                val_metric = float(data["train/learnable_seen_rate"])
                                print(f"[optuna] using wandb summary learnable_seen_rate={val_metric} from {summary_path}")
                                return val_metric
                        except Exception:
                            pass
                except Exception:
                    pass

        # After training (or if not training), evaluate by loading checkpoint into env
        # construct checkpoint filename consistent with main28 naming
        target = args.target
        ck_name = f"HNS_V27_{target}_s{args.n_seekers}_h{args.n_hiders}_b{args.n_boxes}_r{args.n_ramps}.pt"
        ck_path = os.path.join(os.getcwd(), "checkpoints", ck_name)
        # fallback: if checkpoint missing, try to find the newest matching file in checkpoints/
        if not os.path.exists(ck_path):
            ck_dir = os.path.join(os.getcwd(), "checkpoints")
            found = []
            if os.path.isdir(ck_dir):
                for fn in os.listdir(ck_dir):
                    if fn.startswith(f"HNS_V27_{target}_") and fn.endswith('.pt'):
                        found.append(os.path.join(ck_dir, fn))
            if found:
                found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                ck_path = found[0]
                print(f"[optuna] fallback checkpoint chosen: {ck_path}")
            else:
                print(f"[optuna] no checkpoint found for target={target} expected {ck_name}; falling back to scripted evaluation")
                # Fall back to running the environment without a learned checkpoint (use scripted policies)
                eval_env = TeamCosEnv(mode="initial", target=target, n_seekers=args.n_seekers,
                                      n_hiders=args.n_hiders, n_boxes=args.n_boxes, n_ramps=args.n_ramps,
                                      render_mode=None, dist_bonus_scale=val)
                seen_count_fb = 0
                ak_fb = eval_env.learnable_agent_key
                for ep in range(args.episodes):
                    obs = eval_env.reset()
                    done = False
                    ep_seen = False
                    for _ in range(getattr(eval_env, 'prep_steps', 80)):
                        eval_env.step(np.zeros(4, dtype=np.float32))
                    while not done:
                        _, _, term, trunc, info = eval_env.step(np.zeros(4, dtype=np.float32))
                        done = bool(term or trunc)
                        try:
                            if ak_fb.startswith('s'):
                                obs_learn = eval_env._get_obs(eval_env.agent_keys.index(ak_fb))
                                ens = eval_env._ens_orderings.get(ak_fb, [k for k in eval_env.agent_keys if k != ak_fb])
                                for i, e in enumerate(ens[: len(eval_env.idx.OTHERS)]):
                                    if not e.startswith('h'):
                                        continue
                                    en_idx = eval_env.idx.OTHERS[i]
                                    if float(obs_learn[en_idx.VISIBLE]) > 0.5:
                                        ep_seen = True
                                        break
                        except Exception:
                            pass

                    if ak_fb.startswith('s'):
                        if ep_seen:
                            seen_count_fb += 1
                    else:
                        if info.get('dbg_learnable_hider_seen', False):
                            seen_count_fb += 1

                eval_env.close()
                return float(seen_count_fb) / float(max(1, args.episodes))

        # create evaluation env (no viewer)
        eval_env = TeamCosEnv(mode="initial", target=target, n_seekers=args.n_seekers,
                              n_hiders=args.n_hiders, n_boxes=args.n_boxes, n_ramps=args.n_ramps,
                              render_mode=None, dist_bonus_scale=val)

        # load checkpoint and apply to learnable agent
        try:
            state = torch.load(ck_path, map_location="cpu")
        except Exception:
            eval_env.close()
            return 0.0

        ak = eval_env.learnable_agent_key
        try:
            if hasattr(eval_env, 'policy_adapter'):
                eval_env.policy_adapter.set_inference_policy_state([ak], state, seq_len=16, hidden_dim=256)
            else:
                eval_env.set_inference_policy_state([ak], state, seq_len=16, hidden_dim=256)
        except Exception:
            pass

        # prime and warmup
        try:
            norm_obs = eval_env._normalize_obs(eval_env._get_obs(eval_env.learnable_agent_index))
            if hasattr(eval_env, 'policy_adapter'):
                eval_env.policy_adapter._prime_policy_history(ak, 16, norm_obs)
            else:
                eval_env._prime_policy_history(ak, 16, norm_obs)
        except Exception:
            pass

        # enable override so model supplies actions for learnable agent
        try:
            if hasattr(eval_env, 'policy_adapter'):
                eval_env.policy_adapter.set_override_learnable_policy(True)
            else:
                eval_env.set_override_learnable_policy(True)
        except Exception:
            pass

        seen_count = 0
        for ep in range(args.episodes):
            obs = eval_env.reset()
            done = False
            ep_seen = False
            # warmup
            for _ in range(getattr(eval_env, 'prep_steps', 80)):
                _a = np.zeros(4, dtype=np.float32)
                eval_env.step(_a)
            while not done:
                a = np.zeros(4, dtype=np.float32)
                obs, reward, term, trunc, info = eval_env.step(a)
                done = bool(term or trunc)
                # check whether learnable agent (seeker) saw hider via obs
                try:
                    if ak.startswith('s'):
                        obs_learn = eval_env._get_obs(eval_env.agent_keys.index(ak))
                        ens = eval_env._ens_orderings.get(ak, [k for k in eval_env.agent_keys if k != ak])
                        for i, e in enumerate(ens[: len(eval_env.idx.OTHERS)]):
                            if not e.startswith('h'):
                                continue
                            en_idx = eval_env.idx.OTHERS[i]
                            if float(obs_learn[en_idx.VISIBLE]) > 0.5:
                                ep_seen = True
                                break
                except Exception:
                    pass

            if ak.startswith('s'):
                if ep_seen:
                    seen_count += 1
            else:
                if info.get("dbg_learnable_hider_seen", False):
                    seen_count += 1

        try:
            if hasattr(eval_env, 'policy_adapter'):
                eval_env.policy_adapter.set_override_learnable_policy(False)
            else:
                eval_env.set_override_learnable_policy(False)
        except Exception:
            pass
        eval_env.close()

        return float(seen_count) / float(max(1, args.episodes))

    seen_count = 0
    learnable_key = env.learnable_agent_key
    learnable_is_seeker = isinstance(learnable_key, str) and learnable_key.startswith("s")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_seen = False
        while not done:
            # check learnable agent's observation for visible opponents when it's a Seeker
            try:
                if learnable_is_seeker:
                    obs_learn = env._get_obs(env.agent_keys.index(learnable_key))
                    ens = env._ens_orderings.get(learnable_key, [k for k in env.agent_keys if k != learnable_key])
                    for i, e in enumerate(ens[: len(env.idx.OTHERS)]):
                        if not e.startswith("h"):
                            continue
                        en_idx = env.idx.OTHERS[i]
                        if float(obs_learn[en_idx.VISIBLE]) > 0.5:
                            ep_seen = True
                            break
                # otherwise (learnable is Hider), fall back to env-provided info after step
            except Exception:
                pass

            action = np.zeros(4, dtype=np.float32)
            obs, reward, term, trunc, info = env.step(action)
            done = bool(term or trunc)

        if learnable_is_seeker:
            if ep_seen:
                seen_count += 1
        else:
            if info.get("dbg_learnable_hider_seen", False):
                seen_count += 1

    env.close()
    # return fraction of episodes where learnable hider was seen
    return float(seen_count) / float(max(1, args.episodes))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--clean-checkpoints", dest="clean_checkpoints", action="store_true", default=True,
                   help="Remove files/dirs under checkpoints/ (except keepGoodpkl) before each trial")
    p.add_argument("--no-clean-checkpoints", dest="clean_checkpoints", action="store_false",
                   help="Do not remove checkpoints before trials")
    p.add_argument("--min-val", type=float, default=0.0)
    p.add_argument("--max-val", type=float, default=1.0)
    p.add_argument("--sqlite-db", type=str, default="optuna_dist_bonus.db",
                   help="SQLite DB filename for Optuna storage (set --no-sqlite to disable)")
    p.add_argument("--no-sqlite", dest="sqlite_db", action="store_const", const=None,
                   help="Disable SQLite storage and run in-memory")
    p.add_argument("--mode", type=str, default="initial")
    p.add_argument("--target", type=str, default="hider")
    p.add_argument("--n-seekers", type=int, default=1)
    p.add_argument("--n-hiders", type=int, default=2)
    p.add_argument("--n-boxes", type=int, default=2)
    p.add_argument("--n-ramps", type=int, default=1)
    p.add_argument("--run-training", dest="run_training", action="store_true", default=False,
                   help="Run training subprocess (main28_train_final) for each trial before evaluation")
    p.add_argument("--train-profile", dest="train_profile", type=str, default="trainseeker_forRB",
                   help="Profile passed to main28_train_final via -p for training runs")
    p.add_argument("--use-wandb-summary", dest="use_wandb_summary", action="store_true", default=True,
                   help="After training, read local wandb summary JSON and use train/learnable_seen_rate if present")
    p.add_argument("--no-wandb-summary", dest="use_wandb_summary", action="store_false",
                   help="Do not read wandb summary; fall back to checkpoint-based evaluation")
    args = p.parse_args()

    # small wrapper to pass args into optuna objective
    def objective(trial):
        return run_trial(trial, args)

    # create study with optional SQLite persistence
    if args.sqlite_db:
        db_path = os.path.abspath(args.sqlite_db)
        storage = f"sqlite:///{db_path}"
        study = optuna.create_study(direction="maximize", storage=storage, load_if_exists=True)
    else:
        study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=args.trials)

    print("Best trial:")
    print(study.best_trial.params)
    print("Best value:", study.best_trial.value)

    # Export full trial results to CSV and best trial to JSON
    try:
        results_csv = os.path.abspath("optuna_results.csv")
        with open(results_csv, "w") as f:
            f.write("trial_number,value,state,params\n")
            for t in study.trials:
                line = f'{t.number},{getattr(t, "value", "" )},{t.state.name},"{json.dumps(t.params)}"\n'
                f.write(line)
        best_json = os.path.abspath("optuna_best_trial.json")
        with open(best_json, "w") as f:
            json.dump({"params": study.best_trial.params, "value": study.best_trial.value}, f)
        print(f"[optuna] wrote results CSV: {results_csv}")
        print(f"[optuna] wrote best trial JSON: {best_json}")
    except Exception as exc:
        print(f"[optuna] failed to write results: {exc}")


if __name__ == "__main__":
    main()
