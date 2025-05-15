#!/usr/bin/env python3
"""
Train a policy net in the multi-error heliostat env, then render & plot a
comparison of target vs predicted heat-maps (plus error) for one example.
"""

import math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import distance_transform_edt
from datetime import datetime
import matplotlib.pyplot as plt

from test_environment import HelioEnv  # your multi-error env


# ---------------------------------------------------------------------------
class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),          nn.ReLU(),
            nn.Conv2d(64, 128,5, padding=2),          nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x):
        feat = self.cnn(x).flatten(1)
        return F.relu(self.proj(feat))
# ---------------------------------------------------------------------------
# Use Legacy Policy for now 
'''
class PolicyNet(nn.Module):
    def __init__(self, img_channels, num_heliostats, aux_dim,
                 enc_dim=128, lstm_hid=128, use_lstm=True):
        super().__init__()
        self.encoder = CNNEncoder(img_channels, enc_dim)
        self.lstm    = nn.LSTM(enc_dim, lstm_hid, batch_first=True)
        self.head    = nn.Sequential(
            nn.Linear(lstm_hid+aux_dim, 256), nn.ReLU(),
            nn.Linear(256, num_heliostats*3)
        )
        self.num_h = num_heliostats

    def forward(self, img_seq, aux, hx=None):
        B,T,C,H,W = img_seq.shape
        x = img_seq.view(B*T, C, H, W)
        enc = self.encoder(x).view(B, T, -1)
        out, hx = self.lstm(enc, hx)
        last = out[:, -1]
        x = torch.cat([last, aux], dim=1)
        normals = self.head(x).view(B, self.num_h, 3)
        return F.normalize(normals, dim=2), hx

'''
class PolicyNet(nn.Module):
    def __init__(self,
                 img_channels: int,
                 num_heliostats: int,
                 aux_dim: int,
                 enc_dim: int = 128,
                 lstm_hid: int = 128,
                 use_lstm: bool = True):
        """
        Args:
            img_channels: number of image channels in input
            num_heliostats: how many normals to predict
            aux_dim: dimension of extra (non-image) features
            enc_dim: output dimension of CNNEncoder per frame
            lstm_hid: hidden size of the LSTM (only used if use_lstm=True)
            use_lstm: whether to run an LSTM over the encoded frames;
                      if False, simply use the encoding of the last frame
        """
        super().__init__()
        self.num_h = num_heliostats
        self.use_lstm = use_lstm

        # shared CNN encoder
        self.encoder = CNNEncoder(img_channels, enc_dim)

        if self.use_lstm:
            # recurrent path
            self.rnn   = nn.LSTM(enc_dim, lstm_hid, batch_first=True)
            feat_dim   = lstm_hid
        else:
            # non‐recurrent path just uses the per‐frame encoding
            feat_dim   = enc_dim

        # final head takes [feat_dim + aux_dim] → hidden → num_heliostats*3
        self.head = nn.Sequential(
            nn.Linear(feat_dim + aux_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_heliostats * 3)
        )

    def forward(self,
                img_seq: torch.Tensor,
                aux: torch.Tensor,
                hx: tuple = None):
        """
        Args:
            img_seq: (B, T, C, H, W) input image sequence
            aux:     (B, aux_dim) auxiliary features
            hx:      optional LSTM hidden state (ignored if use_lstm=False)
        Returns:
            normals: (B, num_heliostats, 3) unit‐length surface normals
            hx:      new LSTM state if use_lstm=True, else None
        """
        B, T, C, H, W = img_seq.shape

        # flatten frames into batch
        x   = img_seq.view(B * T, C, H, W)
        enc = self.encoder(x)            # (B*T, enc_dim)
        enc = enc.view(B, T, -1)         # (B, T, enc_dim)

        if self.use_lstm:
            # run recurrent encoder
            out, hx = self.rnn(enc, hx)
            feat = out[:, -1, :]         # last time step
        else:
            # just pick the last‐frame encoding
            feat = enc[:, -1, :]
            hx   = None                  # no hidden state to carry forward

        # concatenate aux features and predict
        x       = torch.cat([feat, aux], dim=1)    # (B, feat_dim+aux_dim)
        normals = self.head(x)                     # (B, num_h*3)
        normals = normals.view(B, self.num_h, 3)
        normals = F.normalize(normals, dim=2)

        return normals, hx

# ---------------------------------------------------------------------------
def rollout(env, policy, k, T, device, use_max=False):
    """Run T steps, return dict of {mse, dist, bound} on final frame."""
    B = env.batch_size
    # init noisy actions & first image
    with torch.no_grad():
        state_dict = env.reset()
        img = state_dict['img']            # (B,1,H,W)
        aux = state_dict['aux']            # (B, aux_dim)

    # history buffer
    res = env.resolution
    hist = torch.zeros(B, k, res, res, device=device)
    hist[:, -1] = img.clone()

    hx = None
    for _ in range(T):
        net_img = hist.unsqueeze(2)           # (B,k,1,H,W)

        if not(hx == None):
            normals, hx = policy(net_img.detach(), aux.detach(), (hx[0].detach(), hx[1]))
        else:
            normals, hx = policy(net_img.detach(), aux.detach(), hx)

        state_dict, loss_dict = env.step(normals)

        img = state_dict['img']            # (B,1,H,W)
        hist = torch.roll(hist, -1, dims=1)
        hist[:, -1] = img

    return loss_dict, img, hist 

# ---------------------------------------------------------------------------
def train_and_eval(args, plot_heatmaps_in_tensorboard = True):
    # device
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_device(dev)

    # geometry
    N = 50
    heliostat_pos = torch.rand(N,3,device=dev)*10; heliostat_pos[:,2]=0
    targ_pos = torch.tensor([0.,-5.,0.],device=dev)
    targ_norm= torch.tensor([0., 1.,0.], device=dev)
    targ_area = (15.,15.)
    res=128

    # envs
    train_envs_list = []
    for i in range(args.num_batches):
        train_env = HelioEnv(
            heliostat_pos = heliostat_pos,
            targ_pos = targ_pos,
            targ_area = targ_area,
            targ_norm = targ_norm,
            sigma_scale=0.1,
            error_scale_mrad=180.0,
            initial_action_noise=0.0,
            resolution=res,
            batch_size=args.batch_size,
            device=args.device,
            new_sun_pos_every_reset=args.new_sun_pos_every_reset,
            new_errors_every_reset=args.new_errors_every_reset,
        )
        train_envs_list.append(train_env)


    # envs
    test_env = HelioEnv(
        heliostat_pos = heliostat_pos,
        targ_pos = targ_pos,
        targ_area = targ_area,
        targ_norm = targ_norm,
        sigma_scale=0.1,
        error_scale_mrad=180.0,
        initial_action_noise=0.0,
        resolution=128,
        batch_size=9,
        device=args.device,
        new_sun_pos_every_reset=False,
        new_errors_every_reset=False,
    )

    # model + opt
    aux_dim = 3+N*3
    policy = PolicyNet(img_channels=1, num_heliostats=N, aux_dim=aux_dim, use_lstm=args.use_lstm).to(dev)
    opt   = torch.optim.Adam(policy.parameters(), lr=args.lr)
    if args.scheduler == "plateau":
        sched = ReduceLROnPlateau(opt, 'min', patience=50, factor=0.27)
    elif args.scheduler == "cyclic":
        sched = CyclicLR(opt, base_lr=1e-5, max_lr=args.lr,
                         step_size_up=1000, mode='triangular')
    elif args.scheduler == "exp":
        sched = ExponentialLR(opt, gamma=1.25)

    # decay-schedule params
    anti_spill = args.anti_spill
    dist_f     = args.dist_f
    mse_f     = args.mse_f
    # warmup-schedule params
    warmup_steps = args.warmup_steps
    active_training_steps = max(1, args.steps - warmup_steps)
    cutoff = int(0.8 * active_training_steps)  # 80 % of post-warm-up steps

    writer = SummaryWriter(f"runs_multi_error_env/{datetime.now():%m%d_%H%M%S}")

    last_boundary_loss = None

    for step in range(args.steps):
        # get batch of envs
        for i in range(args.num_batches):
            train_env = train_envs_list[i]
            opt.zero_grad()
            parts, pred_imgs, _ = rollout(train_env, policy,
                                args.k, args.T, dev)

            # ------------------------------------------------------------
            # Warm-up phase: rely solely on boundary loss to keep the flux
            # inside the target while the policy “finds its feet”.
            # save the boundary loss for later
            last_boundary_loss = parts['bound'].item()

            if (step < warmup_steps) or (last_boundary_loss > args.boundary_thresh):
                # if the boundary loss is too high, use only the boundary loss
                loss = anti_spill * parts['bound']
            else:
                eff_step = step - warmup_steps
                decay = max(1e-5, (cutoff - eff_step) / cutoff)
                loss  = (mse_f * parts['mse']*(1-decay+1e-5)
                        + dist_f*parts['dist']*decay
                        + anti_spill*parts['bound'])

            loss.backward()

            # if loss is NaN, print current lr
            if torch.isnan(loss):
                print(f"NaN loss at step {step} with lr {opt.param_groups[0]['lr']}")

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

            opt.step()
            if not args.disable_scheduler and step > warmup_steps:
                sched.step(parts['mse'].item())

        # ------------------------------------------------------------
        # log train and test loss
        
        if step%25==0 or step==args.steps-1:
            print(f"Step {step} | "
                  f"loss {loss:.4f} | "
                  f"mse_train {parts['mse']:.2e} |"
                  f"current_lr {opt.param_groups[0]['lr']:.2e} | ")

        if step%100==0 or step==args.steps-1:
            #print average gradients wrt. params
            for name, param in policy.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f"gradients/{name}", param.grad.mean(), step)
            # get test loss
            with torch.no_grad():
                test_parts, _, _ = rollout(test_env, policy,
                                           args.k, args.T, dev)

            print(f"[{step:4d}] loss {loss:.4f} | "
                  f"mse_train{parts['mse']:.2e} dist_train {parts['dist']:.2e} "
                  f"bound_train {parts['bound']:.2e} | test_mse {test_parts['mse']:.2e} test_bound {test_parts['bound']:.2e}")

            writer.add_scalar("mse/test", test_parts['mse'], step)
            writer.add_scalar("bound/test", test_parts['bound'], step)

        writer.add_scalar("loss/total", loss.item(), step)
        writer.add_scalar("loss/mse",   parts['mse'], step)
        writer.add_scalar("loss/dist",  parts['dist'], step)
        writer.add_scalar("loss/bound", parts['bound'], step)

        if plot_heatmaps_in_tensorboard and (step % 100 == 0):
            imgs = pred_imgs
            mins  = imgs.view(imgs.size(0), -1).min(1)[0].view(-1,1,1)
            maxs  = imgs.view(imgs.size(0), -1).max(1)[0].view(-1,1,1)
            norm_imgs = (imgs - mins) / (maxs - mins + 1e-6)
            # add_images expects (N, C, H, W); ensure a channel dim exists
            writer.add_images(
                tag="Predicted/normalized_heatmaps",
                img_tensor=norm_imgs.unsqueeze(1),
                global_step=step, 
                dataformats='NCHW'
                )

    writer.close()

# ---------------------------------------------------------------------------
if __name__=="__main__":
    torch.manual_seed(10)
    np.random.seed(10)
    
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--num_batches", type=int, default=1)
    p.add_argument("--steps",      type=int, default=5000)
    p.add_argument("--T",          type=int, default=4)
    p.add_argument("--k",          type=int, default=4)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--use_lstm",     type=bool, default=False)
    p.add_argument("--disable_scheduler", type=bool, default=False)
    p.add_argument("--scheduler", type=str, default="exp",
                   help="Learning rate scheduler: plateau, cyclic, exp")
    p.add_argument("--boundary_thresh", type=float, default=5e-3,
                   help="Upper threshold for boundary loss.")
    p.add_argument("--anti_spill", type=float, default=1.5e4,
                   help="Weight of the anti-spill loss term.")
    p.add_argument("--dist_f",     type=float, default=1.5e4,
                   help="Weight of the distance loss term.")
    p.add_argument("--mse_f",     type=float, default=1.0,
                   help="Weight of the distance loss term.")
    p.add_argument("--new_errors_every_reset", type=bool, default=False,
                   help="Whether to resample errors every reset.")
    p.add_argument("--new_sun_pos_every_reset", type=bool, default=False,
                   help="Whether to sample a new sun position every reset.")
    p.add_argument("--warmup_steps", type=int, default=40,
                   help="Number of initial steps that use only the boundary "
                        "loss before switching to the full loss.")
    args = p.parse_args()
    train_and_eval(args)
