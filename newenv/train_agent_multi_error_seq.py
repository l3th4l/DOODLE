#!/usr/bin/env python3
"""
Train a policy net in the multi-error heliostat env, then render & plot a
comparison of target vs predicted heat-maps (plus error) for one example.
"""

import math, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import distance_transform_edt
from datetime import datetime
import matplotlib.pyplot as plt

from newenv_rl_test_multi_error import HelioField   # your multi-error env

# ---------------------------------------------------------------------------
def make_distance_maps(imgs, thr=0.5):
    maps = []
    for img in imgs.cpu().numpy():
        mask = (img > thr * img.max()).astype(np.uint8)
        maps.append(distance_transform_edt(1 - mask))
    return torch.tensor(np.stack(maps), dtype=torch.float32, device=imgs.device)

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

#Legacy policy (only LSTM) 

class PolicyNet(nn.Module):
    def __init__(self, img_channels, num_heliostats, aux_dim,
                 enc_dim=128, lstm_hid=128):
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
'''
# ---------------------------------------------------------------------------
def rollout(field_ref, field_noisy, policy, k, T,
            heliostat_pos, targ_pos, targ_norm, targ_area,
            distance_maps, sun_pos, device):
    """Run T steps, return dict of {mse, dist, bound} on final frame."""
    B = sun_pos.size(0)
    # init noisy actions & first image
    with torch.no_grad():
        field_noisy.init_actions(sun_pos)
        action = field_noisy.initial_action
        img = field_noisy.render(sun_pos, action)

    # history buffer
    res = field_ref.resolution
    hist = torch.zeros(B, k, res, res, device=device)
    hist[:, -1] = img.clone()

    hx = None
    for _ in range(T):
        net_img = hist.unsqueeze(2)           # (B,k,1,H,W)
        ideal_normals = field_ref.calculate_ideal_normals(sun_pos)
        aux = torch.cat([sun_pos, ideal_normals.flatten(1)], dim=1)

        if not(hx == None):
            normals, hx = policy(net_img.detach(), aux.detach(), (hx[0].detach(), hx[1]))
        else:
            normals, hx = policy(net_img.detach(), aux.detach(), hx)
        img, action = field_noisy.render(sun_pos, normals.flatten(1)), normals.flatten(1)

        hist = torch.roll(hist, -1, dims=1)
        hist[:, -1] = img

    # compute losses
    target_img = field_ref.render(sun_pos, ideal_normals.flatten(1).detach())
    mx = target_img.amax((1,2), keepdim=True).clamp_min(1e-6)
    pred_n = img / mx; targ_n = target_img / mx
    mse      = F.mse_loss(pred_n, targ_n)
    error    = (pred_n - targ_n).abs()
    dist_l   = (error * distance_maps).sum((1,2)).mean()

    # boundary
    def boundary(vects):
        u = torch.tensor([1.,0.,0.], device=device)
        v = torch.tensor([0.,0.,1.], device=device)
        dots = torch.einsum('bij,j->bi', vects, targ_norm)
        eps = 1e-6
        valid = (dots.abs() > eps)
        t = torch.einsum('j,bij->bi', targ_pos, vects)/(dots+(~valid).float()*eps)
        inter = heliostat_pos.unsqueeze(0) + vects*t.unsqueeze(2)
        local = inter - targ_pos
        xl = torch.einsum('bij,j->bi', local, u)
        yl = torch.einsum('bij,j->bi', local, v)
        hw, hh = targ_area[0]/2, targ_area[1]/2
        dx = F.relu(xl.abs()-hw); dy = F.relu(yl.abs()-hh)
        dist = torch.sqrt(dx*dx+dy*dy+1e-8)
        inside = (xl.abs()<=hw)&(yl.abs()<=hh)&valid
        return (dist*(~inside).float()).mean()

    bound = boundary(normals)
    return {'mse': mse, 'dist': dist_l, 'bound': bound}, img, target_img

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
    ref_field   = HelioField(heliostat_pos, targ_pos, targ_area, targ_norm,
                             sigma_scale=0.1, error_scale_mrad=0.0,
                             initial_action_noise=0.0,
                             resolution=res, device=dev,
                             max_batch_size=args.batch_size)
    noisy_field = HelioField(heliostat_pos, targ_pos, targ_area, targ_norm,
                             sigma_scale=0.1, error_scale_mrad=180.0,
                             initial_action_noise=0.0,
                             resolution=res, device=dev,
                             max_batch_size=args.batch_size)

    # precompute distance maps
    sun_dirs = F.normalize(torch.randn(args.batch_size,3,device=dev),dim=1)
    radius   = math.hypot(1000,1000)
    sun_pos  = sun_dirs*radius
    ref_field.init_actions(sun_pos)
    with torch.no_grad():
        timg = ref_field.render(sun_pos, ref_field.initial_action)
    distance_maps = make_distance_maps(timg)

    # model + opt
    aux_dim = 3+N*3
    policy = PolicyNet(img_channels=1, num_heliostats=N, aux_dim=aux_dim, use_lstm=args.use_lstm).to(dev)
    opt   = torch.optim.Adam(policy.parameters(), lr=args.lr)
    sched = ReduceLROnPlateau(opt, 'min', patience=50, factor=0.27)

    # decay-schedule params
    anti_spill = 1.5e4
    dist_f     = 1e4
    warmup_steps = args.warmup_steps
    active_training_steps = max(1, args.steps - warmup_steps)
    cutoff = int(0.8 * active_training_steps)  # 80 % of post-warm-up steps

    writer = SummaryWriter(f"runs_multi_error/{datetime.now():%m%d_%H%M%S}")

    for step in range(args.steps):
        opt.zero_grad()
        parts, pred_imgs, _ = rollout(ref_field, noisy_field, policy,
                              args.k, args.T,
                              heliostat_pos, targ_pos, targ_norm,
                              targ_area, distance_maps, sun_pos, dev)

        # ------------------------------------------------------------
        # Warm-up phase: rely solely on boundary loss to keep the flux
        # inside the target while the policy “finds its feet”.
        if step < warmup_steps:
            loss = anti_spill * parts['bound']
        else:
            eff_step = step - warmup_steps
            decay = max(1e-5, (cutoff - eff_step) / cutoff)
            loss  = (parts['mse']*(1-decay+1e-5)
                    + dist_f*parts['dist']*decay
                    + anti_spill*parts['bound'])

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

        opt.step()
        sched.step(parts['mse'].item())

        if step%100==0 or step==args.steps-1:
            print(f"[{step:4d}] loss {loss:.4f} | "
                  f"mse {parts['mse']:.2e} dist {parts['dist']:.2e} "
                  f"bound {parts['bound']:.2e}")
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

    # ------------------------------------------------------------
    # After training: render one fresh batch and pick sample 0
    parts, pred_img, target_img = rollout(ref_field, noisy_field, policy,
                                          args.k, args.T,
                                          heliostat_pos, targ_pos, targ_norm,
                                          targ_area, distance_maps, sun_pos, dev)

    p = pred_img[0].detach().cpu().numpy()
    t = target_img[0].detach().cpu().numpy()
    err = np.abs(p - t)

    # plot
    fig, ax = plt.subplots(1,3, figsize=(12,4))
    ax[0].imshow(t, cmap='hot'); ax[0].set_title("Target")
    ax[1].imshow(p, cmap='hot'); ax[1].set_title("Predicted")
    ax[2].imshow(err, cmap='viridis'); ax[2].set_title("|Error|")
    for a in ax: a.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig(f"runs_multi_error_{datetime.now():%m%d_%H%M%S}.png")

# ---------------------------------------------------------------------------
if __name__=="__main__":
    torch.manual_seed(10)
    np.random.seed(10)
    
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--steps",      type=int, default=5000)
    p.add_argument("--T",          type=int, default=4)
    p.add_argument("--k",          type=int, default=4)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--use_lstm",     type=bool, default=True)
    p.add_argument("--warmup_steps", type=int, default=500,
                   help="Number of initial steps that use only the boundary "
                        "loss before switching to the full loss.")
    args = p.parse_args()
    train_and_eval(args)
