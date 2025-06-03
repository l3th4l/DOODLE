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


torch.autograd.set_detect_anomaly(False)  
# ---------------------------------------------------------------------------
# anomaly loggers
def log_if_nan(tensor, name):
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            print(f"⚠️  NaN/Inf found in {name}: {tensor}")
    elif isinstance(tensor, tuple):
        for i, t in enumerate(tensor):
            log_if_nan(t, f"{name}[{i}]")

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
'''

class PolicyNet(nn.Module):
    def __init__(self,
                 img_channels: int,
                 num_heliostats: int,
                 aux_dim: int,
                 enc_dim: int = 128,
                 lstm_hid: int = 128,
                 transformer_layers: int = 2,
                 transformer_heads: int = 8,
                 architecture: str = "lstm", 
                 use_lstm: bool = True):  # options: 'mlp', 'lstm', 'transformer'
        """
        Args:
            img_channels: number of image channels in input
            num_heliostats: how many normals to predict
            aux_dim: dimension of extra (non-image) features
            enc_dim: output dimension of CNNEncoder per frame
            lstm_hid: hidden size of the LSTM (used if architecture='lstm')
            transformer_layers: number of transformer encoder layers (if 'transformer')
            transformer_heads: number of attention heads (if 'transformer')
            architecture: one of 'mlp', 'lstm', 'transformer'
        """
        super().__init__()
        self.num_h = num_heliostats
        self.arch = architecture.lower()

        # shared CNN encoder
        self.encoder = CNNEncoder(img_channels, enc_dim)

        if self.arch == 'lstm':
            if not use_lstm:
                #warn that "use_lstm" is ignored
                print("Warning: 'use_lstm' is ignored and deprecated. "
                      "Use 'architecture' instead.")
            # recurrent LSTM path
            self.rnn = nn.LSTM(enc_dim, lstm_hid, batch_first=True)
            feat_dim = lstm_hid
        elif self.arch == 'transformer':
            # transformer path
            encoder_layer = nn.TransformerEncoderLayer(d_model=enc_dim,
                                                       nhead=transformer_heads,
                                                       batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer,
                                                     num_layers=transformer_layers)
            feat_dim = enc_dim
        elif self.arch == 'mlp':
            # non-recurrent MLP-only path
            feat_dim = enc_dim
        else:
            raise ValueError(f"Unknown architecture '{architecture}'. Choose 'mlp', 'lstm', or 'transformer'.")

        # final head takes [feat_dim + aux_dim] -> hidden -> num_heliostats*3
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
            hx:      optional LSTM hidden state (only used if arch='lstm')
        Returns:
            normals: (B, num_heliostats, 3) unit-length surface normals
            hx:      new LSTM state if arch='lstm', else None
        """
        B, T, C, H, W = img_seq.shape

        # flatten frames into batch for CNN
        x = img_seq.view(B * T, C, H, W)
        enc = self.encoder(x)            # (B*T, enc_dim)
        enc = enc.view(B, T, -1)         # (B, T, enc_dim)

        if self.arch == 'lstm':
            # LSTM encoder
            out, hx = self.rnn(enc, hx)
            feat = out[:, -1, :]
        elif self.arch == 'transformer':
            # Transformer encoder (batch_first)
            # enc: (B, T, enc_dim)
            trans_out = self.transformer(enc)  # (B, T, enc_dim)
            feat = trans_out[:, -1, :]
            hx = None
        else:
            # MLP path uses last-frame encoding
            feat = enc[:, -1, :]
            hx = None

        # concatenate aux features and predict
        x = torch.cat([feat, aux], dim=1)      # (B, feat_dim+aux_dim)
        normals = self.head(x)                 # (B, num_h*3)
        normals = normals.view(B, self.num_h, 3)
        normals = F.normalize(normals, dim=2)

        return normals, hx

#TODO See if ther is a problem with the dimensions of the input image
#NOTE Apparently not but with k=T, we're not utilizing the power of the LSTM
# ---------------------------------------------------------------------------
def rollout(env, policy, k, T, device, use_mean=False):
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

    #mean loss dict
    mean_loss_dict = {'mse': 0, 'dist': 0, 'bound': 0}
    mse_over_t = []

    hx = None
    for _ in range(T):
        net_img = hist.unsqueeze(2)           # (B,k,1,H,W)

        normals, hx = policy(net_img.detach(), aux.detach(), hx)

        state_dict, loss_dict = env.step(normals)  

        if use_mean:
            # accumulate losses
            mean_loss_dict['mse'] = mean_loss_dict['mse'] + (1/T) * loss_dict['mse']
            mean_loss_dict['dist'] = mean_loss_dict['dist'] + (1/T) * loss_dict['dist']
            mean_loss_dict['bound'] = mean_loss_dict['bound'] + (1/T) * loss_dict['bound']

        mse_over_t.append(loss_dict['mse'].item())


        img = state_dict['img']            # (B,1,H,W)
        hist = torch.roll(hist, -1, dims=1)
        hist[:, -1] = img

    if use_mean:
        return mean_loss_dict, img, hist, mse_over_t
    else:
        return loss_dict, img, hist, mse_over_t

# ---------------------------------------------------------------------------
def train_and_eval(args, plot_heatmaps_in_tensorboard = True, return_best_mse = True):
    # device
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_device(dev)

    # geometry
    N = args.num_heliostats
    # heliostat positions
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
            error_scale_mrad=args.error_scale_mrad,
            initial_action_noise=0.0,
            resolution=res,
            batch_size=args.batch_size,
            device=args.device,
            new_sun_pos_every_reset=args.new_sun_pos_every_reset,
            new_errors_every_reset=args.new_errors_every_reset,
            use_error_mask=args.use_error_mask, 
            error_mask_ratio=args.error_mask_ratio,
        )
        train_env.seed(args.seed + i)
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
    policy = PolicyNet(img_channels=1, num_heliostats=N, aux_dim=aux_dim, architecture= args.architecture,
                        lstm_hid=args.lstm_hid,
                        transformer_layers=args.transformer_layers,
                        transformer_heads=args.transformer_heads,).to(device=dev)  

    # register anomaly hooks
    for n, p in policy.named_parameters():
        p.register_hook(lambda grad, n=n: log_if_nan(grad, f"grad {n}"))

    for n, m in policy.named_modules():
        m.register_forward_hook(
            lambda mod, inp, out, n=n: log_if_nan(out, f"out {n}")
        )

    opt   = torch.optim.Adam(policy.parameters(), lr=args.lr)
    if args.scheduler == "plateau":
        sched = ReduceLROnPlateau(opt, 'min', patience=args.scheduler_patience,
                                 factor=args.scheduler_factor, verbose=True)
    elif args.scheduler == "cyclic":
        sched = CyclicLR(opt, base_lr=1e-5, max_lr=args.lr,
                         step_size_up=args.step_size_up, mode=args.scheduler_mode, gamma=args.scheduler_gamma,)
    elif args.scheduler == "exp":
        sched = ExponentialLR(opt, gamma=args.exp_decay)

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
    last_mse_loss = None
    best_mse_loss = None

    for step in range(args.steps):
        # get batch of envs
        for i in range(args.num_batches):
            train_env = train_envs_list[i]
            opt.zero_grad()
            parts, pred_imgs, _, train_mse_over_t = rollout(train_env, policy,
                                args.k, args.T, dev, use_mean=args.use_mean)

            # ------------------------------------------------------------
            # Warm-up phase: rely solely on boundary loss to keep the flux
            # inside the target while the policy “finds its feet”.
            # save the boundary loss for later
            last_boundary_loss = parts['bound'].item()

            if (args.num_batches * step + i < warmup_steps) or (last_boundary_loss > args.boundary_thresh):
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
                if last_mse_loss is not None:
                    print(f"last mse loss {last_mse_loss}")
                    print(f"best mse loss {best_mse_loss}")
                    return best_mse_loss if return_best_mse else last_mse_loss
                else:
                    print("No last mse loss")
                    return np.nan

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip)

            opt.step()
            if (args.num_batches * step + i > warmup_steps) : #(not args.disable_scheduler) and 
                if args.scheduler == "plateau":
                    sched.step(parts['mse'].item())
                elif args.scheduler == "cyclic":
                    sched.step()
                elif args.scheduler == "exp":
                    sched.step()
        # ------------------------------------------------------------
        # log train and test loss

        
        if step%25==0 or step==args.steps-1:
            print(f"Step {step} | "
                  f"loss {loss:.4f} | "
                  f"mse_train {parts['mse']:.2e} |"
                  f"current_lr {opt.param_groups[0]['lr']:.6f} | ")

        #debug code 
        '''
        bad_suns = [train_env.sun_pos[i].detach().cpu().numpy() for i in range(50) if pred_imgs[i].amax() > 0.005]; good_suns = [train_env.sun_pos[i].detach().cpu().numpy() for i in range(50) if pred_imgs[i].amax() < 0.005]; import pandas as pd; pd.DataFrame({'value': good_suns + bad_suns, 'source': ['good'] * len(good_suns) + ['bad'] * len(bad_suns)}).to_csv('labeled_list.csv', index=False)
        '''

        if step%100==0 or step==args.steps-1:
            #print average gradients wrt. params
            for name, param in policy.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f"gradients/{name}", param.grad.mean(), step)
            # get test loss
            with torch.no_grad():
                test_parts, _, _, test_mse_over_t = rollout(test_env, policy,
                                           args.k, args.T, dev)

            print(f"[{step:4d}] loss {loss:.4f} | "
                  f"mse_train{parts['mse']:.2e} dist_train {parts['dist']:.2e} "
                  f"bound_train {parts['bound']:.2e} | test_mse {test_parts['mse']:.2e} test_bound {test_parts['bound']:.2e}")
            
            last_mse_loss = test_parts['mse'].item()
            best_mse_loss = last_mse_loss if best_mse_loss is None else min(best_mse_loss, last_mse_loss)

            # log test loss
            writer.add_scalar("mse/test", test_parts['mse'], step)
            writer.add_scalar("bound/test", test_parts['bound'], step)

            # log test mse over time for testing
            if step > warmup_steps:
                for t in range(args.T):
                    writer.add_scalar(f"mse/test_over_t/", test_mse_over_t[t], args.T*step + t)

        writer.add_scalar("loss/total", loss.item(), step)
        writer.add_scalar("loss/mse",   parts['mse'], step)
        writer.add_scalar("loss/dist",  parts['dist'], step)
        writer.add_scalar("loss/bound", parts['bound'], step)
        writer.add_scalar("hyperparams/lr", opt.param_groups[0]['lr'], step)

        # log train mse over time for training
        if step > warmup_steps:
            for t in range(args.T):
                writer.add_scalar(f"mse/train_over_t/", train_mse_over_t[t], args.T*step + t)

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

    return best_mse_loss if return_best_mse else last_mse_loss

# ---------------------------------------------------------------------------
if __name__=="__main__":
    
    
    p = argparse.ArgumentParser()
    p.add_argument("--num_heliostats", type=int, default=50)
    p.add_argument("--error_scale_mrad", type=float, default=90.0)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--num_batches", type=int, default=1)
    p.add_argument("--steps",      type=int, default=5000)
    p.add_argument("--T",          type=int, default=4)
    p.add_argument("--k",          type=int, default=4)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--use_lstm",     type=bool, default=False)
    p.add_argument("--grad_clip",  type=float, default=1e-7, 
                   help="Gradient clipping threshold.")
    p.add_argument("--architecture", type=str, default="lstm",
                   help="Network architecture: lstm, transformer, mlp")
    p.add_argument("--lstm_hid",  type=int, default=128)
    p.add_argument("--transformer_layers", type=int, default=2)
    p.add_argument("--transformer_heads", type=int, default=8)
    p.add_argument("--disable_scheduler", type=bool, default=False)
    p.add_argument("--use_mean", type=bool, default=False,
                   help="Whether to use mean loss over the batch.")
    p.add_argument("--scheduler", type=str, default="exp",
                   help="Learning rate scheduler: plateau, cyclic, exp")
    p.add_argument("--scheduler_patience", type=int, default=50,
                   help="Patience for the plateau scheduler.")
    p.add_argument("--scheduler_factor", type=float, default=0.27,
                   help="Factor for the plateau scheduler.")
    p.add_argument("--scheduler_mode", type=str, default="triangular2",
                   help="Cyclic learning rate scheduler mode: triangular, triangular2, exp_range")
    p.add_argument("--scheduler_gamma", type=float, default=0.99,
                   help="Cyclic learning rate scheduler gamma: 0.99 for exp_range")
    p.add_argument("--exp_decay", type=float, default=1.8,
                   help="Exponential decay factor for the learning rate. (only for exp scheduler)")
    p.add_argument("--step_size_up", type=int, default=300,
                   help="Step size up for the cyclic learning rate scheduler.")
    p.add_argument("--step_size_down", type=int, default=1000,
                   help="Step size down for the cyclic learning rate scheduler.")
    p.add_argument("--boundary_thresh", type=float, default=5e-3,
                   help="Upper threshold for boundary loss.")
    p.add_argument("--anti_spill", type=float, default=1.5e4,
                   help="Weight of the anti-spill loss term.")
    p.add_argument("--dist_f",     type=float, default=1.0e4,
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
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.") 
    p.add_argument("--use_error_mask", type=bool, default=False,
                   help="Whether to use only bottom k'th percentile for loss calculation")
    p.add_argument("--error_mask_ratio", type=float, default=0.2,
                   help="Percentile to use for loss calculation (if using error mask)")
    args = p.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_and_eval(args)
