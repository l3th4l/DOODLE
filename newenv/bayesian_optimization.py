#!/usr/bin/env python3
"""
Bayesian optimisation of heliostat hyper‑parameters with Optuna.

What changed
============
* Replaced the old iterative grid search with **Bayesian optimisation (TPE)**
  driven by Optuna.  This yields better sample efficiency than exhaustive
  grids.
* **Error‑robust**: if ``train_batched`` throws (OOM, NaN, etc.) the trial is
  *pruned* and the optimiser continues.
* Enforces the constraint ``cutoff < steps`` by pruning such trials early.
* All trial results (success, pruned, failed) are exported to a JSON file.
* Optional SQLite storage lets you pause / resume long sweeps and analyse
  them later with ``optuna-dashboard``.

Usage
-----
```bash
python bayes_opt_search.py \
    --device cuda:1 \
    --trials 200 \
    --storage sqlite:///study.db \
    --pruner median
```

Make sure `train_batched` accepts `dist_factor` and (ideally) returns
``(final_loss, mse_loss)`` as discussed earlier.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import optuna

# ---------------------------------------------------------------------------
# Point this import to your actual training module --------------------------
# ---------------------------------------------------------------------------
from main_agent_test_random_sun import train_batched  # noqa: E402

# ---------------------------------------------------------------------------
# Search space bounds --------------------------------------------------------
# ---------------------------------------------------------------------------
GLOBAL_BOUNDS = {
    "lr": (0.00789274, 0.07552283),        # log‑uniform
    "steps": (1830, 2894),       # int
    "cutoff": (1161, 2146),      # int (must be < steps)
    "dist_factor": (5523, 8229),  # int
    "lr_reduce_factor": (0.1, 0.95),  # uniform
}


def objective_factory(device: str):
    """Return an Optuna objective function bound to *device*."""

    def objective(trial: optuna.trial.Trial):
        # --- Suggest hyper‑parameters --------------------------------------
        lr = trial.suggest_float("lr", *GLOBAL_BOUNDS["lr"], log=True)
        steps = trial.suggest_int("steps", *GLOBAL_BOUNDS["steps"])
        cutoff = trial.suggest_int("cutoff", *GLOBAL_BOUNDS["cutoff"])
        lr_reduce_factor = trial.suggest_float("lr_reduce_factor", *GLOBAL_BOUNDS["lr_reduce_factor"], log=False)
        if cutoff >= steps:  # infeasible: prune immediately
            raise optuna.exceptions.TrialPruned("cutoff ≥ steps")
        dist_factor = trial.suggest_int("dist_factor", *GLOBAL_BOUNDS["dist_factor"])

        tag = f"bo_1_{trial.number:05d}_{datetime.now().strftime('%m%d_%H%M%S')}"

        # --- Run training --------------------------------------------------
        try:
            ret = train_batched(
                batch_size=25,
                steps=steps,
                device_str=device,
                save_name=tag,
                lr=lr,
                cutoff=cutoff,
                dist_factor=dist_factor,
                lr_reduce_factor=lr_reduce_factor, 
                plot_heatmaps_in_tensorboard=True, 
                save_heatmaps=False
            )
            if isinstance(ret, tuple):
                _, mse_loss = ret
            else:
                mse_loss = float(ret)
        except Exception as exc:
            # Record failure, then prune
            trial.set_user_attr("failed_reason", str(exc))
            raise optuna.exceptions.TrialPruned(str(exc))

        return mse_loss  # lower is better

    return objective


def main():
    parser = argparse.ArgumentParser(description="Bayesian optimisation with Optuna")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--trials", type=int, default=200,
                        help="Number of optimisation trials")
    parser.add_argument("--storage", default=None,
                        help="Optuna storage URL, e.g. sqlite:///study.db")
    parser.add_argument("--study-name", default="heliostat_bayes_opt")
    parser.add_argument("--pruner", choices=["none", "median"], default="median",
                        help="Early‑stopping strategy")
    parser.add_argument("--output", default="bayes_results.json",
                        help="Path to write detailed results JSON")
    args = parser.parse_args()

    pruner = optuna.pruners.MedianPruner() if args.pruner == "median" else optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=args.storage,
        load_if_exists=True,
        pruner=pruner,
    )

    study.optimize(
        objective_factory(args.device),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    # --- Report best trial --------------------------------------------------
    print("\n=== OPTIMISATION COMPLETE ===")
    best = study.best_trial
    print("Best trial value (mse):", best.value)
    print("Best parameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # --- Save all trials ----------------------------------------------------
    trials_data = [
        {
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            "params": t.params,
            **t.user_attrs,
        }
        for t in study.trials
    ]
    Path(args.output).write_text(json.dumps(trials_data, indent=2, default=str))


if __name__ == "__main__":
    main()
