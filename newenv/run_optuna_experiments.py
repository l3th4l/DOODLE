import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import optuna

# ---- Customize: define your search space here ----
#TODO: read a json file and fetch search space from there 
# (have the option to take a dictionary or a JSON file)
def suggest_params(trial: optuna.trial.Trial) -> Dict[str, object]:
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"]),
    }


# ---- If your script needs fixed args, set them here ----
FIXED_KWARGS = {
    # "data_root": "/path/to/data",
    # "epochs": 20,
}

def kwargs_to_cli(kwargs: Dict[str, object]) -> List[str]:
    """Turn dict into CLI flags: {"epochs": 20} -> ["--epochs", "20"]"""
    args = []
    for k, v in kwargs.items():
        k = k.replace("_", "-")
        if isinstance(v, bool):
            if v:  # only include true flags
                args.append(f"--{k}")
        else:
            args.extend([f"--{k}", str(v)])
    return args


def run_script_and_get_metric(
    script: str,
    param_kwargs: Dict[str, object],
    fixed_kwargs: Dict[str, object],
    env: Dict[str, str] = None,
) -> float:
    """
    Launch the training script and return the objective metric.
    - parse_mode="file": requires train.py to accept `--metrics_out <path>` and write {"objective": <float>}
    - parse_mode="stdout": looks for a line like `OPTUNA_OBJECTIVE=0.1234`
    """
    if env is None:
        env = os.environ.copy()

    with tempfile.TemporaryDirectory() as td:
        metrics_path = Path(td) / "metrics.json"
        cli = ["python", script]
        cli += kwargs_to_cli(param_kwargs)
        cli += kwargs_to_cli(fixed_kwargs)
        cli += ["--metrics_out", str(metrics_path)]
        print("Running:", shlex.join(cli), "with CUDA_VISIBLE_DEVICES=", env.get("CUDA_VISIBLE_DEVICES"))
        completed = subprocess.run(cli, env=env, capture_output=True, text=True)
        if completed.returncode != 0:
            print("STDOUT:\n", completed.stdout)
            print("STDERR:\n", completed.stderr)
            raise RuntimeError(f"Script failed with code {completed.returncode}")
        if not metrics_path.exists():
            # Fall back to stdout parse in case the script didn’t write the file
            return parse_objective_from_stdout(completed.stdout)
        data = json.loads(metrics_path.read_text())
        return float(data["objective"])


def build_objective(script: str, parse_mode: str, gpu_id: str):
    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial)

        # Optional: set per-trial output dir, seed, etc.
        run_fixed = {
            **FIXED_KWARGS,
            "seed": trial.number,  # example
            "out_dir": f"runs/trial_{trial.number}",  # if your script supports this
        }

        # Pin this worker/trial to the given GPU
        env = os.environ.copy()
        # Each worker process will pass a single GPU id (e.g., "0"). If you want multi-GPU per trial, set "0,1".
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        metric = run_script_and_get_metric(
            script=script,
            param_kwargs=params,
            fixed_kwargs=run_fixed,
            parse_mode=parse_mode,
            env=env,
        )
        # Report intermediate value if you can (for pruning), e.g. trial.report(metric, step)
        return metric
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str, default="train.py", help="Path to your standalone Python file")
    ap.add_argument("--study", type=str, default="my_study")
    ap.add_argument("--storage", type=str, default="sqlite:///optuna.db")
    ap.add_argument("--direction", type=str, choices=["minimize", "maximize"], default="minimize")
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--gpu-id", type=str, default="0", help="Which GPU(s) this worker should use, e.g. '0' or '1' or '0,1'")
    ap.add_argument("--parse-mode", type=str, choices=["file", "stdout"], default="file")
    ap.add_argument("--sampler", type=str, choices=["tpe", "random"], default="tpe")
    ap.add_argument("--pruner", type=str, choices=["none", "median"], default="median")
    args = ap.parse_args()

    sampler = optuna.samplers.TPESampler() if args.sampler == "tpe" else optuna.samplers.RandomSampler()
    pruner = None if args.pruner == "none" else optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
    )

    obj = build_objective(script=args.script, parse_mode=args.parse_mode, gpu_id=args.gpu_id)
    # n_jobs=1 because *this process* is bound to one GPU; we’ll run multiple processes for multiple GPUs.
    study.optimize(obj, n_trials=args.n_trials, n_jobs=1, gc_after_trial=True)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()