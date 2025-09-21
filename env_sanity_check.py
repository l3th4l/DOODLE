#!/usr/bin/env python3
import argparse, torch, torch.nn.functional as F
from torch import nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

from test_environment import HelioEnv   # must provide .step(normals) and .reset()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_heliostats", type=int, default=1)
    p.add_argument("--batch_size",     type=int, default=500, help="N")
    p.add_argument("--lr",             type=float, default=1e1)
    p.add_argument("--steps",          type=int, default=10)
    p.add_argument("--tol",            type=float, default=1e-7)
    p.add_argument("--patience",       type=int,   default=1,
                   help="Patience (steps) before LR is reduced")
    p.add_argument("--factor",         type=float, default=0.5,
                   help="Factor by which LR is reduced")
    p.add_argument("--device",         type=str,   default="cuda")
    p.add_argument("--seed",           type=int,   default=666)
    p.add_argument("--error_scale_mrad", type=float, default=2.0)
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

    # ----- Optimizable normals -----
    nvecs_raw = nn.Parameter(torch.randn(B, N, 3, device=dev))
    opt   = torch.optim.Adam([nvecs_raw], lr=args.lr)
    sched = ReduceLROnPlateau(opt, mode="min",
                              factor=args.factor,
                              patience=args.patience,)

    best = float("inf")
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)

        normals = F.normalize(nvecs_raw, dim=2)
        _, loss_dict, _ = env.step(normals)
        loss = loss_dict["alignment_loss"]

        loss.backward()
        opt.step()

        best = min(best, float(loss.item()))
        sched.step(loss)   # <--- Plateau scheduler update

        if step % 2 == 0 or loss.item() < args.tol:
            lr = opt.param_groups[0]["lr"]
            print(f"[{step:5d}] loss={loss.item():.3e} | best={best:.3e} | lr={lr:.2e}")
        if loss.item() < args.tol:
            break

    print(f"Done. Final alignment_loss = {loss.item():.3e}, best = {best:.3e}")

if __name__ == "__main__":
    main()
