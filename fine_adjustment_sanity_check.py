#!/usr/bin/env python3
import argparse, torch, torch.nn.functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from test_environment import HelioEnv   # must provide .step(normals) and .reset()

def main():
    p = argparse.ArgumentParser()
    # --- Env / geometry ---
    p.add_argument("--num_heliostats", type=int, default=1)
    p.add_argument("--batch_size",     type=int, default=500, help="N")
    p.add_argument("--device",         type=str,   default="cuda")
    p.add_argument("--seed",           type=int,   default=666)
    p.add_argument("--error_scale_mrad", type=float, default=2.0)

    # --- Pretrain (alignment) ---
    p.add_argument("--pretrain_lr",    type=float, default=1e-1)
    p.add_argument("--pretrain_steps", type=int,   default=70)
    p.add_argument("--tol",            type=float, default=1e-7)
    p.add_argument("--patience",       type=int,   default=1,
                   help="Patience (steps) before LR is reduced")
    p.add_argument("--factor",         type=float, default=0.5,
                   help="Factor by which LR is reduced")

    # --- Rollout-like TTC loop (no policy) ---
    p.add_argument("--T",                  type=int,   default=15,
                   help="Number of rollout timesteps")
    p.add_argument("--fine_steps_per_t",   type=int,   default=25,
                   help="Inner steps per time step for TTC")
    p.add_argument("--fine_lr",            type=float, default=1.3e-4,
                   help="LR for the fine_error_vec optimizer")
    p.add_argument("--fine_weight_decay",  type=float, default=0.0)
    p.add_argument("--fine_grad_clip",     type=float, default=0.0,
                   help="0 disables clipping; >0 enables")
    p.add_argument("--fine_init_eps",      type=float, default=1e-3,
                   help="Uniform init in [-eps, +eps] for fine_error_vec")
    p.add_argument("--fine_adjustment_start_t", type=int, default=3,
                   help="Start TTC after this t (0-based).")
    p.add_argument("--fine_from_t0",       action="store_true",
                   help="If set, starts TTC from t=0 regardless of start_t.")
    p.add_argument("--dist_fraction",      type=float, default=0.85,
                   help="Fraction of post-start steps using 'dist' before switching to 'mse'.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ----- Environment -----
    N  = args.num_heliostats
    B  = args.batch_size
    res = 128

    heliostat_pos = torch.rand(N, 3, device=dev) * 10 + 1500
    heliostat_pos[:, 2] = 0
    targ_pos  = torch.tensor([0., -5., 0.], device=dev)
    targ_norm = torch.tensor([0.,  1., 0.], device=dev)
    targ_area = (15., 15.)

    env = HelioEnv(
        heliostat_pos=heliostat_pos,
        targ_pos=targ_pos,
        targ_area=targ_area,
        targ_norm=targ_norm,
        sigma_scale=0.01,
        error_scale_mrad=args.error_scale_mrad,
        initial_action_noise=0.0,
        resolution=res,
        batch_size=B,
        device=str(dev),
        new_sun_pos_every_reset=False,
        new_errors_every_reset=False,
    )
    env.seed(args.seed)
    _ = env.reset()

    # ----- Pretrain: optimize a global normals parameter on alignment_loss -----
    nvecs_raw = nn.Parameter(torch.randn(B, N, 3, device=dev))
    opt   = torch.optim.Adam([nvecs_raw], lr=args.pretrain_lr)
    sched = ReduceLROnPlateau(opt, mode="min",
                              factor=args.factor,
                              patience=args.patience)

    best = float("inf")
    for step in range(args.pretrain_steps):
        opt.zero_grad(set_to_none=True)
        base_normals = F.normalize(nvecs_raw, dim=2)
        _, loss_dict, _ = env.step(base_normals)
        loss = loss_dict["alignment_loss"]
        loss.backward()
        opt.step()
        best = min(best, float(loss.item()))
        sched.step(loss)

        if step % 2 == 0 or loss.item() < args.tol:
            lr = opt.param_groups[0]["lr"]
            print(f"[PRE {step:3d}] align_loss={loss.item():.3e} | best={best:.3e} | lr={lr:.2e}")
        if loss.item() < args.tol:
            break

    print(f"Pretrain done. Final alignment_loss = {loss.item():.3e}, best = {best:.3e}")

    # Freeze base normals for rollout-like TTC phase (no policy, base fixed)
    with torch.no_grad():
        base_normals = F.normalize(nvecs_raw, dim=2).detach()

    # ----- Rollout-like phase with Test-Time Compute (no policy) -----
    # Persistent fine_error_vec across timesteps (like in your rollout)
    fine_error_vec = torch.empty_like(base_normals).uniform_(
        -args.fine_init_eps, args.fine_init_eps
    ).requires_grad_()
    fine_opt = torch.optim.Adam([fine_error_vec],
                                lr=args.fine_lr,
                                weight_decay=args.fine_weight_decay)

    # Time scheduling between dist and mse (post-start)
    start_t = 0 if args.fine_from_t0 else args.fine_adjustment_start_t
    switch_t = start_t + int(max(0, args.T - start_t) * args.dist_fraction)

    print("\n=== Rollout-like TTC (no policy) ===")
    for t in range(args.T):
        # Evaluate BEFORE fine adjustment (for visibility)
        with torch.no_grad():
            normals_before = F.normalize(base_normals + fine_error_vec, dim=2)
            _, losses_before, _ = env.step(normals_before)
            db = float(losses_before["dist"].item())
            mb = float(losses_before["mse"].item())

        # Choose objective for this timestep
        if t < start_t:
            do_fine = False
            objective_key = "dist"  # placeholder, not used when do_fine=False
        else:
            do_fine = True
            objective_key = "dist" #if t < switch_t else "mse"

        # Inner optimization on fine_error_vec ONLY
        if do_fine:
            for _inner in range(args.fine_steps_per_t):
                fine_opt.zero_grad(set_to_none=True)
                candidate = F.normalize(base_normals + fine_error_vec, dim=2)
                _, _losses, _ = env.step(candidate)
                inner_loss = _losses[objective_key]
                inner_loss.backward()
                if args.fine_grad_clip and args.fine_grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_([fine_error_vec], args.fine_grad_clip)
                fine_opt.step()

        # Evaluate AFTER fine adjustment
        with torch.no_grad():
            normals_after = F.normalize(base_normals + fine_error_vec, dim=2)
            _, losses_after, _ = env.step(normals_after)
            da = float(losses_after["dist"].item())
            ma = float(losses_after["mse"].item())

        # Report
        tag = "(fine OFF)" if not do_fine else f"(fine ON → {objective_key})"
        print(f"[t={t:02d}] {tag}  "
              f"dist: {db:.3e} → {da:.3e}   "
              f"mse: {mb:.3e} → {ma:.3e}")

    print("\nDone. Rollout-like TTC phase complete.")

if __name__ == "__main__":
    main()
