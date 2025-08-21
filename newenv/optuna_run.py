# optuna_run.py
import os, gc, argparse
import optuna
import torch
from types import SimpleNamespace as NS
import train_with_env  # uses train_with_env.train_and_eval(...)

def build_args(trial, device: str):
    return NS(
        # --- tuned hyperparameters ---
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 500]),
        num_batches = trial.suggest_int("num_batches", 1, 2),
        lr = trial.suggest_float("lr", 1e-6, 5e-3, log=True),

        # --- fixed values ---
        k=1, 
        T=4,
        use_lstm = True,
        warmup_steps = 80,
        disable_scheduler = False,
        scheduler = "plateau",
        scheduler_patience = 50,
        scheduler_factor = 0.87,
        scheduler_mode = "triangular",
        step_size_up = 20,
        steps = 10000,
        use_mean = False,
        architecture = "lstm",
        grad_clip = 0.01,
        seed = 666,
        num_heliostats = 1,
        error_scale_mrad = 150.0,
        boundary_thresh = 2e-4,

        # --- unused but required for compatibility ---
        scheduler_gamma = 0.97,
        exp_decay = 1.0,
        step_size_down = 200,
        alignment_f = 50.0,
        anti_spill = 10000.0,
        dist_f = 10000.0,
        mse_f = 1.0,
        alignment_pretrain_steps = 100,
        lstm_hid = 128,
        transformer_layers = 2,
        transformer_heads = 4,
        use_error_mask = False,
        error_mask_ratio = 0.25,
        new_sun_pos_every_reset = False,
        new_errors_every_reset = False,

        # --- device ---
        device = device,
    )


def objective(trial):
    device = os.environ.get("FORCED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    args = build_args(trial, device)

    # Optional: shorten runs during search; do full steps for best params later
    # args.steps = min(args.steps, 2000)

    best_mse = train_with_env.train_and_eval(
        args,
        plot_heatmaps_in_tensorboard=False,
        return_best_mse=True,
    )  # returns best test MSE from the loop
    trial.set_user_attr("device", device)

    # be a good GPU citizen
    del args
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(best_mse)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default="sqlite:///optuna.db")
    ap.add_argument("--study", default="heliostat")
    ap.add_argument("--direction", default="minimize")
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sampler = optuna.samplers.TPESampler(
        seed=args.seed, multivariate=True, group=True
    )
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        load_if_exists=True,
        direction=args.direction,
        sampler=sampler,
        pruner=pruner,
    )
    # IMPORTANT: n_jobs=1 here; we’ll do process-per-GPU for real parallelism
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)
