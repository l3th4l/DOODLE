#!/usr/bin/env python3
import argparse
import os
import random
import sys
import time

def f(x, step, total_steps):
    """A noisy objective: lower is better."""
    base = (x - 3.14) ** 2
    anneal = max(0.05, 1.0 - step / max(1, total_steps))
    noise = random.gauss(0, 0.03) * anneal
    return base + noise

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--x", type=float, required=True, help="Hyperparameter to optimize.")
    p.add_argument("--out", type=str, required=True, help="Path to streaming metrics file.")
    p.add_argument("--steps", type=int, default=30, help="How many reporting steps.")
    p.add_argument("--sleep", type=float, default=0.25, help="Seconds between steps.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Write a simple CSV header and stream values.
    with open(args.out, "w", buffering=1) as f:
        f.write("step,metric\n")
        f.flush()
        for step in range(1, args.steps + 1):
            val = f(args.x, step, args.steps)
            f.write(f"{step},{val:.6f}\n")
            f.flush()
            time.sleep(args.sleep)

    # Exit code 0 indicates success.
    return 0

if __name__ == "__main__":
    sys.exit(main())
