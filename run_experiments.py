#!/usr/bin/env python3
"""
sweep_runner.py

One-at-a-time sweeps for heliostat RL experiments with robust resume.
- Skips runs that have completed (RUN_DONE).
- Prevents duplicate launches with an atomic RUN_LOCK file.
- Re-runs runs that were started but not completed (RUN_STARTED present, RUN_DONE absent).
- Captures stdout/stderr to files.
- Writes args.json and a small status.json with timing.

Usage:
    python sweep_runner.py --root experiments [--tag v1] [--device cuda] [--dry_run]
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback, socket
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from contextlib import redirect_stdout, redirect_stderr

# --- Import your training entrypoint ---
from train_with_env_com_trunc_advantage_ttt import train_and_eval

# ------------------------------
# Global seeds & defaults
# ------------------------------
SEEDS = [42, 420, 69, 666, 999]

# Your "current setup" defaults (used whenever a property is not being analyzed)
DEFAULTS = {
    "device": "cuda",
    "use_lstm": True,                  # will be overridden by architecture (see _apply_arch_consistency)
    "warmup_steps": 80,
    "batch_size": 500,
    "num_batches": 8,
    "disable_scheduler": False,
    "boundary_thresh": 2e-4,
    "scheduler": "plateau",
    "lr": 0.00013,
    "scheduler_factor": 0.95,
    "step_size_up": 20,
    "scheduler_mode": "triangular",
    "steps": 900,
    "use_mean": False,
    "architecture": "transformer",     # will be overridden when analyzing architecture
    "T": 10,
    "k": 2,
    "grad_clip": 0.01,
    "seed": 42,
    "num_heliostats": 1,
    "error_scale_mrad": 5.0,

    # Not listed in your “current setup” but needed by the trainer; use safe defaults from your parser:
    "fine_steps_per_t": 10,
    "fine_enabled": "always",          # ("none", "test", "always")
    "truncate_every": 5,
    "dropout": 0.3,                    # conservative default; swept separately
    "heliostat_distance": 1500.0,
    "azimuth": 15.0,
    "elevation": 45.0,

    # Keep trainer’s other knobs as fixed constants (matching your parser’s spirit):
    "detach_input": True,
    "extra_steps": 20,
    "lstm_hid": 128,
    "transformer_layers": 2,
    "transformer_heads": 8,
    "scheduler_patience": 50,
    "scheduler_mode_cyclic": "triangular2",  # only used if scheduler == cyclic
    "scheduler_gamma": 0.99,
    "exp_decay": 1.8,
    "step_size_down": 1000,
    "anti_spill": 1.5e4,
    "dist_f": 1.0e4,
    "mse_f": 1.0,
    "alignment_f": 100.0,
    "new_errors_every_reset": False,
    "new_sun_pos_every_reset": False,
    "alignment_pretrain_steps": 100,
    "use_error_mask": False,
    "error_mask_ratio": 0.2,
}

# One-at-a-time analyzable properties (value grids you specified)
ANALYZE = {
    "T": [5, 10, 15],
    "k": [1, 2, 4],  # special batch_size rule when k >= 2
    "error_scale_mrad": [5, 10, 25, 45],
    "fine_steps_per_t": [5, 10],
    "heliostat_distance": [15, 150, 1500],
    "truncate_every": [1, 5, 8, 10],
    "dropout": [0.0, 0.3, 0.6],
    "architecture": ["lstm", "transformer", "mlp"],
}

# ------------------------------
# Helpers
# ------------------------------
def _apply_arch_consistency(args: dict) -> None:
    """Ensure --use_lstm matches chosen --architecture."""
    arch = args.get("architecture", "transformer")
    args["use_lstm"] = (arch == "lstm")

def _apply_k_batchsize_rule(args: dict) -> None:
    """If sweeping k, adjust batch_size = 1000/k if k >= 2 else 500."""
    k = int(args.get("k", DEFAULTS["k"]))
    if k >= 2:
        args["batch_size"] = max(1, 1000 // k)
    else:
        args["batch_size"] = 500

def _args_to_namespace(args: dict) -> SimpleNamespace:
    """
    Convert our dict to a trainer-friendly argparse.Namespace matching your parser names.
    Important: map our internal "scheduler_mode" for cyclic to the expected key.
    """
    ns = dict(args)  # shallow copy
    ns["scheduler_mode"] = args.get("scheduler_mode", DEFAULTS["scheduler_mode"])
    return SimpleNamespace(**ns)

def _run_dir(root: Path, tag: str|None, prop: str, value, seed: int) -> Path:
    ptag = f"{tag}_" if tag else ""
    # Folder structure: <root>/<tag_><prop>/<value>/seed-<seed>/
    return root / f"{ptag}{prop}" / str(value) / f"seed-{seed}"

def _status_paths(rdir: Path):
    return (
        rdir / "RUN_STARTED",
        rdir / "RUN_DONE",
        rdir / "RUN_FAILED",
        rdir / "args.json",
        rdir / "status.json",
        rdir / "stdout.log",
        rdir / "stderr.log",
        rdir / "RUN_LOCK",   # per-run atomic lock file
    )

def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))

def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ------------------------------
# Core execution
# ------------------------------
def run_single(prop: str, value, seed: int, root: Path, tag: str|None, dry_run: bool=False,
              device_override: str|None=None) -> dict:
    """
    Build args, call train_and_eval, write logs & status files.
    Returns a tiny dict for summary.csv row.
    """
    rdir = _run_dir(root, tag, prop, value, seed)
    rdir.mkdir(parents=True, exist_ok=True)
    started_fp, done_fp, failed_fp, args_fp, status_fp, out_fp, err_fp, lock_fp = _status_paths(rdir)

    # Skip if already completed
    if done_fp.exists():
        return {"prop": prop, "value": value, "seed": seed, "status": "skipped_done"}

    # Acquire exclusive lock (atomic). If exists, another process is running it.
    try:
        fd = os.open(lock_fp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w") as lf:
            lf.write(f"{_now()} pid={os.getpid()} host={socket.gethostname()}\n")
    except FileExistsError:
        return {"prop": prop, "value": value, "seed": seed, "status": "skipped_busy"}

    try:
        # Re-check if another process just finished and released/recreated (rare but safe)
        if done_fp.exists():
            return {"prop": prop, "value": value, "seed": seed, "status": "skipped_done"}

        # Assemble args: defaults -> override prop -> seed -> derived rules -> device override
        args = dict(DEFAULTS)
        args[prop] = value
        args["seed"] = seed

        if device_override:
            args["device"] = device_override

        # Derived rules
        if prop == "k":
            _apply_k_batchsize_rule(args)
        _apply_arch_consistency(args)

        # Persist args
        _write_json(args_fp, args)

        if dry_run:
            return {"prop": prop, "value": value, "seed": seed, "status": "planned"}

        # Start marker (informational; lock is the actual guard)
        started_fp.write_text(_now())

        # Capture logs
        t0 = time.time()
        status = "unknown"
        error_msg = ""
        try:
            ns = _args_to_namespace(args)
            with open(out_fp, "a", buffering=1) as out_f, open(err_fp, "a", buffering=1) as err_f:
                with redirect_stdout(out_f), redirect_stderr(err_f):
                    print(f"[{_now()}] START prop={prop} value={value} seed={seed}")
                    train_and_eval(ns)
                    print(f"[{_now()}] END OK prop={prop} value={value} seed={seed}")
            done_fp.write_text(_now())
            status = "done"
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f"{type(e).__name__}: {e}"
            failed_fp.write_text(f"[{_now()}]\n{error_msg}\n\nTraceback:\n{tb}\n")
            status = "failed"
        finally:
            dt = time.time() - t0
            # Write a small status.json
            _write_json(status_fp, {
                "prop": prop, "value": value, "seed": seed,
                "status": status, "seconds": round(dt, 3),
                "started_at": started_fp.read_text() if started_fp.exists() else None,
                "finished_at": _now(),
                "error": error_msg if error_msg else None,
                "run_dir": str(rdir),
            })

        return {"prop": prop, "value": value, "seed": seed, "status": status}
    finally:
        # Always release the lock
        try:
            lock_fp.unlink()
        except FileNotFoundError:
            pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="experiments")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    # fixed default type for start_at (kept for future extensibility)
    ap.add_argument("--start_at", type=int, default=-1)
    # NEW: device override
    ap.add_argument("--device", type=str, default=None,
                    help="Override device for all runs (e.g., cuda, cpu, mps)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    tag = args.tag.strip() or None
    root.mkdir(parents=True, exist_ok=True)

    # Plan runs: one-at-a-time sweeps over ANALYZE
    planned = []
    count = 1
    for prop, values in ANALYZE.items():
        if args.start_at >= 0 and count < args.start_at:
            print(f"Skipping analysis of {prop} (count {count}) due to --start_at {args.start_at}")
            count += 1
            continue
        print(f"Planning analysis of {prop} (count {count}) with values: {values}")
        count += 1
        for val in values:
            for seed in SEEDS:
                planned.append((prop, val, seed))

    # Execute
    rows = []
    for prop, val, seed in planned:
        row = run_single(
            prop, val, seed,
            root=root,
            tag=tag,
            dry_run=args.dry_run,
            device_override=args.device
        )
        rows.append(row)

    # Simple summary CSV (note: concurrent writers can clobber; use tags/pids if needed)
    csv_path = root / (("summary_" + args.tag + ".csv") if tag else "summary.csv")
    with open(csv_path, "w") as f:
        f.write("prop,value,seed,status\n")
        for r in rows:
            f.write(f"{r['prop']},{r['value']},{r['seed']},{r['status']}\n")

    print(f"\nSummary written to: {csv_path}")
    print(f"Root directory: {root}")

if __name__ == "__main__":
    main()
