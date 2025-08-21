#!/usr/bin/env python3
import argparse
import contextlib
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import optuna


# ---------- GPU pool (thread-safe) ----------
class GPUPool:
    def __init__(self, gpu_ids):
        self._q = queue.Queue()
        for gid in gpu_ids:
            self._q.put(str(gid))
    @contextlib.contextmanager
    def lease(self):
        gid = self._q.get()
        try:
            yield gid
        finally:
            self._q.put(gid)

# ---------- Tail a growing file ----------
def stream_last_metric(path, last_seen_step):
    """
    Reads any new lines from CSV at `path` and yields (step:int, metric:float).
    Ignores the header row and already-seen steps.
    """
    if not Path(path).exists():
        return  # nothing yet
    with open(path, "r") as f:
        # Skip header
        header = f.readline()
        if not header:
            return
        for line in f:
            line = line.strip()
            if not line or "," not in line:
                continue
            try:
                step_str, val_str = line.split(",", 1)
                step = int(step_str)
                if step <= last_seen_step:
                    continue
                metric = float(val_str)
                yield step, metric
            except Exception:
                continue

def run_and_monitor(
    py_executable,
    script_path,
    x_value,
    out_file,
    steps,
    sleep_s,
    trial: optuna.trial.Trial,
    prune: bool,
):
    """
    Launches dummy_program.py and live-monitors `out_file`.
    Reports intermediate metrics to Optuna.
    Returns final metric (float). Raises TrialPruned if pruned.
    """
    cmd = [
        py_executable, script_path,
        "--x", str(x_value),
        "--out", str(out_file),
        "--steps", str(steps),
        "--sleep", str(sleep_s),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    last_step = 0
    best_metric = None
    start_time = time.time()

    # We poll both the process output (optional) and the metric file
    # without blocking; simple coarse-grained loop:
    try:
        while True:
            # Drain any subprocess stdout (optional; keeps buffers small)
            if proc.stdout and proc.poll() is None:
                with contextlib.suppress(Exception):
                    while True:
                        # Non-blocking-ish read: break when no line available
                        line = proc.stdout.readline()
                        if not line:
                            break

            # Read any new metrics written so far
            for step, metric in stream_last_metric(out_file, last_step) or []:
                last_step = step
                best_metric = metric if best_metric is None else metric
                # Report to Optuna (step index starts at 1 here)
                trial.report(metric, step=step)
                if prune and trial.should_prune():
                    # Kill the subprocess and prune
                    with contextlib.suppress(Exception):
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    raise optuna.TrialPruned()

            # If process finished and we've consumed the metrics, break
            if proc.poll() is not None:
                # Final drain of metrics in case of race
                for step, metric in stream_last_metric(out_file, last_step) or []:
                    last_step = step
                    best_metric = metric
                    trial.report(metric, step=step)
                break

            time.sleep(0.15)  # be nice to the CPU

        # Ensure return code OK
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Subprocess failed with exit code {rc}")
        if best_metric is None:
            raise RuntimeError("No metric was produced.")
        return best_metric

    finally:
        # Make sure the process is not left running
        if proc and proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.wait(timeout=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str, default="dummy_gou_objective.py",
                    help="Path to the worker script.")
    ap.add_argument("--storage", type=str, default="sqlite:///optuna_demo.db",
                    help="Optuna storage URL (e.g., sqlite:///file.db).")
    ap.add_argument("--study", type=str, default="demo_study",
                    help="Optuna study name.")
    ap.add_argument("--n-trials", type=int, default=12, help="Total trials.")
    ap.add_argument("--n-jobs", type=int, default=2, help="Parallel workers.")
    ap.add_argument("--gpus", type=str, default="0,1",
                    help="Comma-separated GPU ids to round-robin.")
    ap.add_argument("--output-dir", type=str, default="runs",
                    help="Directory to store trial output files.")
    ap.add_argument("--steps", type=int, default=30,
                    help="Steps each subprocess will run.")
    ap.add_argument("--sleep", type=float, default=0.25,
                    help="Delay between steps in the subprocess.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--prune", action="store_true",
                    help="Enable median pruning.")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    gpu_pool = GPUPool(gpu_ids)

    if args.prune:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=5)
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study,
        storage=args.storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    # Objective function
    def objective(trial: optuna.trial.Trial):
        # Example search space
        x = trial.suggest_float("x", -5.0, 10.0)

        # Allocate a GPU id for this trial (even if the script doesn't really use it)
        with gpu_pool.lease() as gpu_id:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id  # the child process sees a single GPU

            # Unique output file per trial
            out_file = Path(args.output_dir) / f"trial_{trial.number:04d}.csv"

            # Launch and monitor
            # We pass the env via Popen by temporarily setting it in our wrapper
            # The helper sets env through Popen; easiest approach is to inject
            # into the environment temporarily via a context, but since our
            # runner uses Popen internally, we’ll set it here and restore.
            # Instead, we’ll modify the environment only for this call:
            py_exec = sys.executable
            script_path = args.script

            # Hack: set env only for the subprocess by wrapping Popen call
            # We modify run_and_monitor to rely on global env;
            # simplest: temporarily set and restore os.environ (thread-local safe enough here).
            old_env = os.environ.copy()
            try:
                os.environ.update(env)
                value = run_and_monitor(
                    py_executable=py_exec,
                    script_path=script_path,
                    x_value=x,
                    out_file=str(out_file),
                    steps=args.steps,
                    sleep_s=args.sleep,
                    trial=trial,
                    prune=args.prune,
                )
            finally:
                # Restore environment for other workers
                # (Optuna uses threads for n_jobs; restoring avoids cross-talk.)
                for k in list(os.environ.keys()):
                    if k not in old_env:
                        del os.environ[k]
                os.environ.update(old_env)

        return value

    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)

    best = study.best_trial
    print("\n=== Best Trial ===")
    print(f"number: {best.number}")
    print(f"value:  {best.value:.6f}")
    print("params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    print(f"\nResults saved under: {args.output_dir}")
    print(f"Storage: {args.storage}")
    print(f"Study:   {args.study}")

if __name__ == "__main__":
    main()
