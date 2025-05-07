#!/usr/bin/env python3
"""
Iterative **grid** search (not random) for hyper‑parameters of the heliostat
policy network, with automatic space shrinking and robust error handling.

Overview
========
* **Full Cartesian grid** over four parameters: ``lr``, ``steps``, ``cutoff``,
  ``dist_factor``.  The number of points per dimension is configurable
  (default 4 → 4⁴ = 256 evaluations per round).
* After each round we keep the *top n* configurations ranked by
  ``mse_loss`` and **shrink** each parameter’s bounds so that the new grid is
  centred on those winners.  You control how aggressively we shrink with
  ``--shrink_ratio``.
* Enforces the constraint ``cutoff < steps`` (invalid pairs are silently
  skipped).
* Any training exception (OOM, NaN, etc.) is **caught** so the search
  continues.
* Writes every successful run to JSON (default ``search_results.json``).

Prerequisites
-------------
Your ``train_batched`` must accept ``dist_factor`` and *ideally* return two
numbers: ``total_loss`` and ``mse_loss``.  If it still returns a single float
we’ll treat that as the metric but you will lose the total‑loss logging.

Minimal patch::

    def train_batched(..., dist_factor=5000.0):
        ...                     # <‑‑ replace the hard‑coded constant
        loss = (... dist_factor ...)
        ...
        return final_loss, mse_loss.item()  # ← add this line

Usage example::

    python iterative_grid_search.py \
        --device cuda:1 \
        --iterations 3 \
        --grid_size 4 \
        --top_n 5

Adjust ``grid_size`` or ``iterations`` to fit your GPU time budget.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Import your training function here ----------------------------------------
# ---------------------------------------------------------------------------
# Rename this according to the location of your actual code.
from main_agent_test_random_sun import train_batched  # noqa: E402

# ---------------------------------------------------------------------------
# Global search space: (min, max, scale)
#   scale == 'log'    → points spaced log‑uniformly
#   scale == 'linear' → points spaced uniformly
# ---------------------------------------------------------------------------
GLOBAL_SPACE: Dict[str, Tuple[float, float, str]] = {
    "lr": (1e-4, 1e2, "log"),
    "steps": (500, 3000, "linear"),
    "cutoff": (100, 3000, "linear"),
    "dist_factor": (1_000, 10_000, "linear"),
}


# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _grid_points(low: float, high: float, n: int, scale: str):
    if n == 1:
        return [low]
    if scale == "log":
        log_low, log_high = math.log10(low), math.log10(high)
        return [10 ** v for v in
                [log_low + i * (log_high - log_low) / (n - 1)
                 for i in range(n)]]
    return [low + i * (high - low) / (n - 1) for i in range(n)]


def build_grid(space: Dict[str, Tuple[float, float, str]], n_points: int):
    """Return a list of configuration dictionaries forming a Cartesian grid."""
    axes = {}
    for k, (low, high, scale) in space.items():
        values = _grid_points(low, high, n_points, scale)
        if k in {"steps", "cutoff"}:
            values = [int(round(v)) for v in values]
        axes[k] = values

    # Cartesian product → list of dicts
    all_cfgs = []
    for combo in itertools.product(*axes.values()):
        cfg = dict(zip(axes.keys(), combo))
        if cfg["cutoff"] < cfg["steps"]:  # enforce constraint
            all_cfgs.append(cfg)
    return all_cfgs


def shrink_space(global_space, winners: List[dict], shrink_ratio: float):
    """Shrink *global_space* around *winners* producing a new space dict."""
    new_space = {}
    for p, (gmin, gmax, scale) in global_space.items():
        vals = [w[p] for w in winners]
        vmin, vmax = min(vals), max(vals)
        if scale == "log":
            lmin, lmax = math.log10(vmin), math.log10(vmax)
            span = (lmax - lmin) or 0.5
            lmin -= span * shrink_ratio
            lmax += span * shrink_ratio
            nlow = max(math.log10(gmin), lmin)
            nhigh = min(math.log10(gmax), lmax)
            new_space[p] = (10 ** nlow, 10 ** nhigh, scale)
        else:
            span = (vmax - vmin) or max(vmin * 0.5, 1.0)
            nlow = max(gmin, vmin - span * shrink_ratio)
            nhigh = min(gmax, vmax + span * shrink_ratio)
            new_space[p] = (nlow, nhigh, scale)
    return new_space


# ---------------------------------------------------------------------------
# Main driver ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Iterative grid search with space shrinking")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--grid_size", type=int, default=5,
                        help="Points per dimension in each round (≥1)")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--shrink_ratio", type=float, default=0.05,
                        help="How much to enlarge the min/max envelope of winners")
    parser.add_argument("--output", default="search_results.json")
    args = parser.parse_args()

    random.seed(42)
    space = GLOBAL_SPACE.copy()
    best_overall = None
    all_results = []

    for it in range(1, args.iterations + 1):
        print(f"\n=== Iteration {it}/{args.iterations} ===")
        configs = build_grid(space, args.grid_size)
        print(f"Grid has {len(configs)} valid combinations (cutoff < steps).")
        iter_results = []

        for idx, cfg in enumerate(configs, 1):
            tag = f"gs_it{it}_idx{idx:03d}_{datetime.now().strftime('%m%d_%H%M%S')}"
            print(f"[{idx}/{len(configs)}] {cfg}")
            try:
                ret = train_batched(
                    batch_size=25,
                    steps=cfg["steps"],
                    device_str=args.device,
                    save_name=tag,
                    lr=cfg["lr"],
                    cutoff=cfg["cutoff"],
                    dist_factor=cfg["dist_factor"],
                )
                # Accept either (total, mse) or mse only
                if isinstance(ret, tuple):
                    total_loss, mse = ret
                else:
                    total_loss, mse = None, float(ret)
                result = {**cfg, "total_loss": total_loss, "mse_loss": mse}
                iter_results.append(result)
                print(f"   ↳ mse={mse:.6f}")
            except Exception as exc:
                print(f"   ! Error: {exc}. Skipped.")
                continue

        if not iter_results:
            print("No successful runs — aborting search.")
            break

        # Sort by mse ascending
        iter_results.sort(key=lambda r: r["mse_loss"])
        best_iter = iter_results[0]
        if best_overall is None or best_iter["mse_loss"] < best_overall["mse_loss"]:
            best_overall = best_iter

        winners = iter_results[: min(args.top_n, len(iter_results))]
        space = shrink_space(GLOBAL_SPACE, winners, args.shrink_ratio)

        print("Best this iteration:")
        print(json.dumps(best_iter, indent=2, default=str))
        all_results.extend(iter_results)

    print("\n=== SEARCH COMPLETE ===")
    if best_overall:
        print("Best overall configuration:")
        print(json.dumps(best_overall, indent=2, default=str))

    Path(args.output).write_text(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
