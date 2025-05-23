#!/usr/bin/env python3
"""
optuna_search.py
Bayesian/TPE optimisation of the cyclic learning-rate schedule for
`train_policy.py::train_and_eval`.

Search space
------------
* lr               – upper learning-rate for CyclicLR (log-uniform)
* scheduler_mode   – {'triangular', 'triangular2', 'exp_range'}
* scheduler_gamma  – only active if mode=='exp_range'
* step_size_up     – number of iterations for the rising edge of the cycle

Everything else uses the same defaults as in the original script, except that
`scheduler` is *forced* to 'cyclic'.

The objective is the **test MSE** returned by `train_and_eval`.
The run is made robust against crashes/NaNs – such trials are pruned.
"""

from __future__ import annotations
import math, types, argparse, traceback
import optuna

# ---------------------------------------------------------------------------#
# 1.  Import the training routine                                            #
# ---------------------------------------------------------------------------#
from train_with_env import train_and_eval   # your original file

# ---------------------------------------------------------------------------#
# 2.  A tiny helper that reproduces the original CLI defaults as an object   #
# ---------------------------------------------------------------------------#
def default_args() -> argparse.Namespace:
    """Return an argparse.Namespace with the same defaults as train_policy.py."""
    return types.SimpleNamespace(
        batch_size              = 30,
        num_batches             = 5,
        steps                   = 4_000,
        T                       = 4,
        k                       = 4,
        lr                      = 2e-4,          # ← will be overwritten
        device                  = "cuda:2",
        use_lstm                = True,
        disable_scheduler       = False,
        scheduler               = "cyclic",      # ← hard-wired
        scheduler_mode          = "triangular2", # ← will be overwritten
        scheduler_gamma         = 0.99,          # ← may be overwritten
        exp_decay               = 1.8,           # ignored (non-exp scheduler)
        step_size_up            = 300,           # ← will be overwritten
        step_size_down          = 1_000,
        boundary_thresh         = 2e-4,
        anti_spill              = 1.5e4,
        dist_f                  = 1.0e4,
        mse_f                   = 1.0,
        new_errors_every_reset  = False,
        new_sun_pos_every_reset = False,
        warmup_steps            = 80,
        use_mean                = True,          # ← will be overwritten
        scheduler_patience      = 50,           # ← may be overwritten
        scheduler_factor        = 0.5,           # ← may be overwritten
    )

# ---------------------------------------------------------------------------#
# 3.  Optuna objective                                                       #
# ---------------------------------------------------------------------------#
def objective(trial: optuna.Trial) -> float:
    args            = default_args()

    # --- hyper-parameters to optimise --------------------------------------#
    args.lr              = trial.suggest_loguniform("lr", 1e-4, 1.18e-3)
    args.scheduler       = trial.suggest_categorical(
                                "scheduler", 
                                ["cyclic", "plateau"])

    if args.scheduler == "plateau":
        #suggest patience and reduce_lr_factor
        args.patience = trial.suggest_int("scheduler_patience", 250, 750, step=50)         
        args.reduce_lr_factor = trial.suggest_float("scheduler_factor", 0.3, 0.8, step=0.1) 

    elif args.scheduler == "cyclic":
        args.scheduler_mode  = trial.suggest_categorical(
                                    "scheduler_mode",
                                    ["triangular", "triangular2"])

    args.step_size_up    = trial.suggest_int("step_size_up", 100, 1000, step=50)


    args.use_mean        = trial.suggest_categorical(
                                "use_mean", [True, False])

    # ---------------------------------------------------------------------- #
    # Run a single training session and capture its test MSE                #
    # ---------------------------------------------------------------------- #
    try:
        mse = train_and_eval(args, plot_heatmaps_in_tensorboard=True)
    except Exception as e:
        # Any runtime exception or keyboard interrupt → prune the trial
        traceback.print_exc()
        raise optuna.exceptions.TrialPruned() from e

    # Guard against numerical explosions
    if math.isnan(mse) or mse > 1e9:
        raise optuna.exceptions.TrialPruned()

    return mse   # Optuna minimises by default


# ---------------------------------------------------------------------------#
# 4.  Main entry point                                                       #
# ---------------------------------------------------------------------------#
def main() -> None:
    import argparse
    cli = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cli.add_argument("--trials", type=int, default=25, help="Number of Optuna trials.")
    cli.add_argument("--timeout", type=int, default=None,
                     help="Abort optimisation after this many seconds.")
    cli.add_argument("--study-name", type=str, default="heliostat_cyclic_lr")
    cli.add_argument("--storage", type=str, default=None,
                     help="Optuna storage URL for resuming (e.g. sqlite:///bo.db)")
    args = cli.parse_args()

    study = optuna.create_study(
        study_name = args.study_name,
        direction  = "minimize",
        storage    = args.storage,
        load_if_exists = True,
        sampler    = optuna.samplers.TPESampler(),
        pruner     = optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    study.optimize(
        objective,
        n_trials = args.trials,
        timeout  = args.timeout,
        gc_after_trial = True,
        show_progress_bar = True,
    )

    # ---------------------------------------------------------------------- #
    # Print the summary and best parameters                                  #
    # ---------------------------------------------------------------------- #
    print("\n=== Optimisation finished ===")
    print("Best MSE   :", study.best_value)
    print("Best params:", study.best_params)

    # nicety: save the study for later inspection if no external storage
    if args.storage is None:
        study.trials_dataframe().to_csv("optuna_results.csv", index=False)
        print("Full history written to optuna_results.csv")

if __name__ == "__main__":
    main()
