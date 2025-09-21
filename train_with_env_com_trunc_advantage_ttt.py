#!/usr/bin/env python3
"""
Train a policy net in the multi-error heliostat env, then render & plot a
comparison of target vs predicted heat-maps (plus error) for one example.
"""

import math, argparse
import os 
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
print(torch.__version__)
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, ExponentialLR
from piecewise_constant_lr import PiecewiseConstantLR # NOTE : for testing purpouses
#from torch.utils.tensorboard import SummaryWriter
from mlflow_logger import MLflowWriter
from scipy.ndimage import distance_transform_edt
from datetime import datetime
import matplotlib.pyplot as plt

from adamp import AdamP

from layers.center_of_mass import CenterOfMass2D

from test_environment import HelioEnv  # your multi-error env
from plotting_utils import scatter3d_vectors

#MLflow parameters 


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
class COMEncoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int=128, dropout=0.1):
        super().__init__()
        self.com = CenterOfMass2D()
        self.proj = nn.Sequential(
            nn.Linear(2, out_dim), 
            nn.Dropout(dropout)
            )

    def forward(self, x):
        feat = self.com(x)
        return F.gelu(self.proj(feat))
# ---------------------------------------------------------------------------

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
                 use_lstm: bool = True, 
                 dropout=0.1):  # options: 'mlp', 'lstm', 'transformer'
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
        self.encoder = COMEncoder(img_channels, enc_dim, dropout=dropout)

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
                                                       batch_first=True, 
                                                       dropout=dropout)
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
            #nn.BatchNorm1d(feat_dim + aux_dim),
            nn.LayerNorm(feat_dim + aux_dim),
            nn.Linear(feat_dim + aux_dim, 256),
            nn.Dropout(dropout),
            nn.GELU(),
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
            if not (hx == None):    
                feat = trans_out[:, -1, :] + hx
            else:
                feat = trans_out[:, -1, :]
            hx = feat
        else:
            # MLP path uses last-frame encoding
            feat = enc[:, -1, :]
            hx = None

        # concatenate aux features and predict
        x = torch.cat([feat, aux], dim=1)      # (B, feat_dim+aux_dim)
        normals = self.head(x)                 # (B, num_h*3)
        normals = normals.view(B, self.num_h, 3)

        #normals = F.normalize(normals, dim=2)

        return normals, hx

#TODO See if ther is a problem with the dimensions of the input image
#NOTE Apparently not but with k=T, we're not utilizing the power of the LSTM
# ---------------------------------------------------------------------------
def rollout(
    env,
    policy,
    k,
    T,
    device,
    truncate_every=None,
    use_mean=False,
    detach_input=False,
    # --- TTC / test-time compute options ---
    enable_fine: bool = False,
    fine_adjustment_start_t: int = 6,
    fine_from_t0: bool = False,
    fine_steps_per_t: int = 10,
    fine_lr: float = 1e-4,
    fine_weight_decay: float = 0.0,
    fine_grad_clip: float | None = None,
    dist_fraction: float = 0.85,
    freeze_policy_during_fine: bool = True,
    fine_init_eps: float = 1e-4, 
    test_time: bool = False,
):
    """
    Run T steps and (optionally) perform test-time compute by optimizing a persistent
    fine_error_vec to nudge the normals toward lower distance/MSE.

    If enable_fine is True:
      - Create/keep a learnable fine_error_vec with shape (B, N, 3).
      - At each t >= start (or from t=0 if fine_from_t0), run an inner optimization loop
        over fine_error_vec ONLY (policy frozen by default) for 'fine_steps_per_t' steps.
      - For the first `dist_fraction` of timesteps after the start, minimize 'dist';
        afterward, minimize 'mse'.
      - Apply: normals = normalize(base_normals + fine_error_vec, dim=2)
    """
    B = env.batch_size
    with torch.no_grad():
        state_dict = env.reset()
        img = state_dict['img']          # (B,1,H,W)
        aux = state_dict['aux']          # (B, aux_dim)

    # history buffer
    res = env.resolution
    hist = torch.zeros(B, k, res, res, device=device)
    hist[:, -1] = img.clone()

    truncated_loss_dict = {'mse': 0, 'dist': 0, 'bound': 0, 'alignment_loss': 0}
    previous_reward = 0
    mse_over_t = []
    imgs_over_t = []

    hx = None
    prev_normals = None

    # --- TTC state (constructed lazily when needed) ---
    fine_error_vec = None
    fine_opt = None

    if (truncate_every is not None):
        n_truncs = (T // truncate_every)
        coef_pow = 4.0
        coef_div = np.sum(np.arange(n_truncs) ** coef_pow)

    # Decide the time-based schedule boundary between dist and mse
    # For t in [start, T): optimize 'dist' for first 75% of those steps, then 'mse'.
    start_t = 0 if (enable_fine and fine_from_t0) else fine_adjustment_start_t
    switch_t = start_t + int(max(0, T - start_t) * dist_fraction)

    for t in range(T):
        net_img = hist.unsqueeze(2)  # (B,k,1,H,W)

        # ---- Policy forward (as in your original) ----
        if not test_time:
            if prev_normals is None:
                if detach_input:
                    base_normals, hx = policy(net_img.detach(), aux.detach(), hx)
                else:
                    base_normals, hx = policy(net_img, aux, hx)
                base_normals = F.normalize(base_normals, dim=2)
                prev_normals = base_normals
            else:
                if detach_input or ((truncate_every is not None) and ((t + 1) % truncate_every == 1)):
                    base_normals, hx = policy(net_img.detach(), aux.detach(), hx)
                else:
                    base_normals, hx = policy(net_img, aux, hx)
                base_normals = F.normalize(base_normals + prev_normals, dim=2)
                prev_normals = base_normals
        else:
            with torch.no_grad():
                if prev_normals is None:
                    if detach_input:
                        base_normals, hx = policy(net_img.detach(), aux.detach(), hx)
                    else:
                        base_normals, hx = policy(net_img, aux, hx)
                    base_normals = F.normalize(base_normals, dim=2)
                    prev_normals = base_normals
                else:
                    if detach_input or ((truncate_every is not None) and ((t + 1) % truncate_every == 1)):
                        base_normals, hx = policy(net_img.detach(), aux.detach(), hx)
                    else:
                        base_normals, hx = policy(net_img, aux, hx)
                    base_normals = F.normalize(base_normals + prev_normals, dim=2)
                    prev_normals = base_normals

        normals_to_apply = base_normals

        # ---- Test-Time Compute (fine adjustment) ----
        should_fine = enable_fine and (t >= start_t or (fine_from_t0 and t >= 0))
        if should_fine:
            # Initialize persistent fine_error_vec/optimizer lazily
            if fine_error_vec is None:
                # shape: (B, N, 3) where N = env.num_heliostats inferred from base_normals
                fine_error_vec = torch.empty_like(base_normals).uniform_(
                                -fine_init_eps, fine_init_eps
                            ).requires_grad_()

                fine_opt = torch.optim.Adam([fine_error_vec], lr=fine_lr, weight_decay=fine_weight_decay)

            # Choose current objective (time-based schedule across rollout)
            objective_key = 'dist' #if t < switch_t else 'mse'
            # make sure fine_error_vec is a leaf with grad
            fine_error_vec.requires_grad_(True)
            # Inner optimization loop over fine_error_vec ONLY
            for _ in range(fine_steps_per_t):
                fine_opt.zero_grad(set_to_none=True)

                # Optionally block gradients to policy during TTC
                if freeze_policy_during_fine:
                    _base = base_normals.detach()
                else:
                    _base = base_normals

                candidate = F.normalize(_base + fine_error_vec, dim=2)

                # Allowed to call env.step here (per your note). We do NOT use alignment loss for TTC.
                _, _losses, _ = env.step(candidate)

                loss_inner = _losses[objective_key]
                # Strictly optimize fine_error_vec
                loss_inner.backward()

                if fine_grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_([fine_error_vec], max_norm=fine_grad_clip)

                fine_opt.step()

            # After inner optimization, apply the adjusted normals for the real transition
            normals_to_apply = F.normalize(base_normals + fine_error_vec, dim=2)

        # ---- Environment transition for the actual rollout step ----
        prev_normals = normals_to_apply
        state_dict, loss_dict, monitor = env.step(normals_to_apply)

        # --- Truncated-grad logic (unchanged from your version) ---
        if (truncate_every is not None):
            coeff = 1 / (T // truncate_every)

            truncated_loss_dict['alignment_loss'] = truncated_loss_dict['alignment_loss'] + (
                -loss_dict['alignment_loss'] - previous_reward
            )
            previous_reward = -loss_dict['alignment_loss'].detach()

            if (t == (T - 1)) or ((t + 1) % truncate_every == 0):
                truncated_loss_dict['mse'] = 0 * truncated_loss_dict['mse'] + loss_dict['mse']
                truncated_loss_dict['dist'] = truncated_loss_dict['dist'] + coeff * loss_dict['dist']
                truncated_loss_dict['bound'] = truncated_loss_dict['bound'] + coeff * loss_dict['bound']

                # Detach recurrent state
                if isinstance(hx, tuple):
                    hx = tuple(h.detach() for h in hx)
                elif torch.is_tensor(hx):
                    hx = hx.detach()

                aux = state_dict['aux']
            else:
                aux = state_dict['aux']
        else:
            aux = state_dict['aux']

        mse_over_t.append(loss_dict['mse'].item())

        img = state_dict['img']            # (B,1,H,W)
        imgs_over_t.append(img.detach().cpu())

        hist = torch.roll(hist, -1, dims=1)
        hist[:, -1] = img

    truncated_loss_dict['alignment_loss'] = -truncated_loss_dict['alignment_loss']

    if (truncate_every is not None):
        return truncated_loss_dict, img, hist, mse_over_t, monitor, imgs_over_t
    else:
        return loss_dict, img, hist, mse_over_t, monitor, imgs_over_t

# ---------------------------------------------------------------------------
def train_and_eval(args, plot_heatmaps_in_tensorboard = True, return_best_mse = True):
    # device
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.set_default_device(dev)

    # geometry
    N = args.num_heliostats
    # heliostat positions
    heliostat_pos = torch.rand(N,3,device=dev) + np.sqrt(args.heliostat_distance); heliostat_pos[:,2]=0
    targ_pos = torch.tensor([0.,-5.,0.],device=dev)
    targ_norm= torch.tensor([0., 1.,0.], device=dev)
    targ_area = (15.,15.)
    res=128
    deg_diff = 2.0
    # envs
    train_envs_list = []
    for i in range(args.num_batches):
        train_env = HelioEnv(
            heliostat_pos = heliostat_pos,
            targ_pos = targ_pos,
            targ_area = targ_area,
            targ_norm = targ_norm,
            sigma_scale=0.01,
            error_scale_mrad=args.error_scale_mrad,
            initial_action_noise=0.0,
            resolution=res,
            batch_size=args.batch_size,
            device=args.device,
            new_sun_pos_every_reset=args.new_sun_pos_every_reset,
            new_errors_every_reset=args.new_errors_every_reset,
            use_error_mask=args.use_error_mask, 
            error_mask_ratio=args.error_mask_ratio,
            exponential_risk=False,
            azimuth=args.azimuth + i * deg_diff, 
            elevation=args.elevation,
        )
        train_env.seed(args.seed + i)
        #if not (i == 0):
        #    train_env.set_sun_pos(train_envs_list[0].sun_pos) 
        train_envs_list.append(train_env)

    # envs
    test_size = 60
    test_env = HelioEnv(
        heliostat_pos = heliostat_pos,
        targ_pos = targ_pos,
        targ_area = targ_area,
        targ_norm = targ_norm,
        sigma_scale=0.01,
        error_scale_mrad=args.error_scale_mrad,
        initial_action_noise=0.0,
        resolution=res,
        batch_size=test_size,
        device=args.device,
        new_sun_pos_every_reset=False,
        new_errors_every_reset=False,
        azimuth=args.azimuth, 
        elevation=args.elevation,
    )

    #test_env.set_sun_pos(train_envs_list[0].sun_pos[:test_size])

    # model + opt
    aux_dim = 3+N*3
    policy = PolicyNet(img_channels=1, num_heliostats=N, aux_dim=aux_dim, architecture= args.architecture,
                        lstm_hid=args.lstm_hid,
                        transformer_layers=args.transformer_layers,
                        transformer_heads=args.transformer_heads, dropout=args.dropout).to(device=dev)  

    # register anomaly hooks
    for n, p in policy.named_parameters():
        p.register_hook(lambda grad, n=n: log_if_nan(grad, f"grad {n}"))

    for n, m in policy.named_modules():
        m.register_forward_hook(
            lambda mod, inp, out, n=n: log_if_nan(out, f"out {n}")
        )

    opt   = AdamP(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.scheduler == "plateau":
        sched = ReduceLROnPlateau(opt, 'min', patience=args.scheduler_patience,
                                 factor=args.scheduler_factor)
    elif args.scheduler == "cyclic":
        sched = CyclicLR(opt, base_lr=1e-5, max_lr=args.lr,
                         step_size_up=args.step_size_up, mode=args.scheduler_mode, gamma=args.scheduler_gamma,)
    elif args.scheduler == "exp":
        sched = ExponentialLR(opt, gamma=args.exp_decay)

    # decay-schedule params
    alignment_f = args.alignment_f
    anti_spill  = args.anti_spill
    dist_f      = args.dist_f
    mse_f       = args.mse_f

    # warmup-schedule params
    warmup_steps = args.warmup_steps
    pretrain_steps = args.alignment_pretrain_steps
    active_training_steps = max(1, args.steps - warmup_steps)
    cutoff = int(0.8 * active_training_steps)  # 80 % of post-warm-up steps

    #writer = SummaryWriter(f"runs_multi_error_env/{datetime.now():%m%d_%H%M%S}")

    # Assume you already created the env earlier
    sun_pos = test_env.sun_pos.detach().cpu().numpy().tolist()  # convert tensor → JSON-serializable

    params = vars(args).copy()
    params.update({
        "sun_pos": sun_pos,
    })

    run_name = f"run_{datetime.now():%m%d_%H%M%S}"
    writer = MLflowWriter(
        experiment_id="1490651313414470",
        run_name=run_name,
        params=params
    )

    # run_name = f"run_{datetime.now():%m%d_%H%M%S}"     # ▶ keep the run name for folders
    # writer = MLflowWriter(
    #     experiment_id="1490651313414470",
    #     run_name=run_name,
    #     params=vars(args)
    # )

    last_boundary_loss = None
    last_mse_loss = None
    best_mse_loss = None

    # try with best alignment loss for the iterations when we do use the alignment loss 
    last_alignment_loss = np.inf
    best_alignment_loss = np.inf

    # before: for step in range(args.steps + pretrain_steps):
    prev_total_loss = None  # <-- add this line

    for step in range(args.steps + pretrain_steps):
        # get batch of envs
        if args.fine_enabled == 'always':
            start_fine_adjustment_at = 50
            enable_fine = step > start_fine_adjustment_at 
        else:
            enable_fine = False

        opt.zero_grad(set_to_none=True)
        for i in range(args.num_batches):
            train_env = train_envs_list[i]
            #opt.zero_grad()
            parts, pred_imgs, _, train_mse_over_t, monitor, _= rollout(train_env, policy,
                                args.k, args.T, dev, use_mean=args.use_mean, truncate_every=args.truncate_every, detach_input = args.detach_input, 
                                enable_fine = enable_fine)

            # ------------------------------------------------------------
            # Warm-up phase: rely solely on boundary loss to keep the flux
            # inside the target while the policy “finds its feet”.
            # save the boundary loss for later
            last_boundary_loss = parts['bound'].item()

            #if (args.num_batches * step + i < (pretrain_steps)) or parts['alignment_loss'] > last_alignment_loss:
            if True: #((args.num_batches * step + i < (pretrain_steps)) 
                #or ((args.num_batches * step + i < (warmup_steps + pretrain_steps)) and (parts['alignment_loss'] > best_alignment_loss))
                #or (parts['alignment_loss'] > last_alignment_loss)):

                loss = alignment_f * parts['alignment_loss']
                if args.num_batches * step + i == (pretrain_steps -1):
                    last_alignment_loss = parts['alignment_loss']
                if parts['alignment_loss'] < best_alignment_loss:
                    best_alignment_loss = parts['alignment_loss']

            elif ((args.num_batches * step + i < (warmup_steps + pretrain_steps)) 
                or (last_boundary_loss > args.boundary_thresh)):
                
                # if the boundary loss is too high, use only the boundary loss
                loss = anti_spill * parts['bound']
            else:
                eff_step = step - warmup_steps - pretrain_steps
                decay = max(1e-5, (cutoff - eff_step) / cutoff)
                loss  = (mse_f * parts['mse']*(1-decay+1e-5)
                        + dist_f*parts['dist']*decay)

                        #+ anti_spill*parts['bound'])

            # Objective: maximize decrease in loss
            if prev_total_loss is None:
                objective = loss  # first update: fall back to standard minimization
            else:
                objective = loss - prev_total_loss.detach()  # minimize (current - previous)

            (objective / args.num_batches).backward()

            # track for next iteration (no grad through the baseline)
            prev_total_loss = loss.detach()

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

            if i % (args.num_batches - 1) == 0:
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=args.grad_clip)

                opt.step()
                if (args.num_batches * step + i > (warmup_steps + pretrain_steps)) : #(not args.disable_scheduler) and 
                    if args.scheduler == "plateau":
                        sched.step(parts['mse'].item())
                    elif args.scheduler == "cyclic":
                        sched.step()
                    elif args.scheduler == "exp":
                        sched.step()

            
            #save plots of normals and losses every k'th step 
            if (step % 50 == 0) or step == args.steps-1:
                scatter3d_vectors(
                    monitor['normals'].view([-1, 3]).detach().cpu().numpy(), 
                    monitor['all_bounds'].view([-1,]).detach().cpu().numpy(),
                    html_file=f'./monitors_debug/step_{step}/batch_{i}_bounds.html')

                scatter3d_vectors(
                    monitor['normals'].view([-1, 3]).detach().cpu().numpy(), 
                    monitor['mae_image'].view([-1,]).detach().cpu().numpy(),
                    html_file=f'./monitors_debug/step_{step}/batch_{i}_mae_image.html')
                
                scatter3d_vectors(
                    monitor['ideal_normals'].view([-1, 3]).detach().cpu().numpy(), 
                    monitor['mae_image'].view([-1,]).detach().cpu().numpy(),
                    html_file=f'./monitors_debug/step_{step}/batch_{i}_mae_image_ideal.html')
                    
                scatter3d_vectors(
                    monitor['reflected_rays'].view([-1, 3]).detach().cpu().numpy(), 
                    monitor['all_bounds'].view([-1,]).detach().cpu().numpy(),
                    html_file=f'./monitors_debug/step_{step}/batch_{i}_r_bounds.html')

                scatter3d_vectors(
                    monitor['reflected_rays'].view([-1, 3]).detach().cpu().numpy(), 
                    monitor['mae_image'].view([-1,]).detach().cpu().numpy(),
                    html_file=f'./monitors_debug/step_{step}/batch_{i}_r_mae_image.html')
        # ------------------------------------------------------------
        # log train and test loss

        if step%25==0 or step==args.steps-1:
            print(f"Step {step} | "
                  f"loss {loss.detach():.4f} | "
                  f"mse_train {parts['mse'].detach():.2e} |"
                  f"alignment_train {parts['alignment_loss'].detach():.2e} |"
                  f"current_lr {opt.param_groups[0]['lr']:.6f} | ")

        #debug code 
        '''
        bad_suns = [train_env.sun_pos[i].detach().cpu().numpy() for i in range(50) if pred_imgs[i].amax() > 0.005]; good_suns = [train_env.sun_pos[i].detach().cpu().numpy() for i in range(50) if pred_imgs[i].amax() < 0.005]; import pandas as pd; pd.DataFrame({'value': good_suns + bad_suns, 'source': ['good'] * len(good_suns) + ['bad'] * len(bad_suns)}).to_csv('labeled_list.csv', index=False)
        '''
        enable_fine_test = (args.fine_enabled == 'always') or (args.fine_enabled == 'test') 
        if step%100==0 or step==args.steps-1:
            #print average gradients wrt. params
            for name, param in policy.named_parameters():
                if param.grad is not None:
                    writer.add_scalar(f"gradients/{name}", param.grad.mean(), step)
            # get test loss

            policy.eval() # disable training time thingies 
            #with torch.no_grad():
            test_parts, _, _, test_mse_over_t, test_monitor, test_imgs_over_t = rollout(test_env, policy,
                                        args.k, args.T+args.extra_steps, dev, enable_fine = enable_fine_test, test_time = True)
            policy.train() # enable training time thingies 

            # ▶ Save test heat-maps: run_name/step_<step>/idx_<i>/t_<t>.png
            base_dir = os.path.join(run_name, f"step_{step}")
            os.makedirs(base_dir, exist_ok=True)

            # test_imgs_over_t is a list of length (T+extra_steps), each (B,1,H,W) on CPU
            for i in range(test_env.batch_size):
                idx_dir = os.path.join(base_dir, f"idx_{i:03d}")
                os.makedirs(idx_dir, exist_ok=True)

                for t, frame in enumerate(test_imgs_over_t):
                    # frame: (B,1,H,W) → select sample i, squeeze to (H,W)
                    arr = frame[i].squeeze().numpy()
                    out_path = os.path.join(idx_dir, f"t_{t:03d}.png")
                    plt.imsave(out_path, arr, cmap='inferno')  # inferno is great for heatmaps

            print(f"[{step:4d}] loss {loss.detach():.4f} | "
                  f"mse_train{parts['mse'].detach():.2e} dist_train {parts['dist'].detach():.2e} "
                  f"bound_train {parts['bound'].detach():.2e} | test_mse {test_parts['mse'].detach():.2e} test_bound {test_parts['bound'].detach():.2e} test_alignment {test_parts['alignment_loss'].detach():.2e}")
        
            scatter3d_vectors(
                test_monitor['reflected_rays'].view([-1, 3]).detach().cpu().numpy(), 
                test_monitor['mae_image'].view([-1,]).detach().cpu().numpy(),
                html_file=f'./monitors_debug/step_{step}/batch_{i}_test_r_mae_image.html')

            last_mse_loss = test_parts['mse'].item()
            best_mse_loss = last_mse_loss if best_mse_loss is None else min(best_mse_loss, last_mse_loss)

            # log test loss
            writer.add_scalar("mse/test", test_parts['mse'].detach(), step)
            writer.add_scalar("bound/test", test_parts['bound'].detach(), step)

            # log test mse over time for testing
            if step > (warmup_steps + pretrain_steps):
                for t in range(args.T+args.extra_steps):
                    writer.add_scalar(f"mse/test_over_t/", test_mse_over_t[t], args.T*step + t)

        writer.add_scalar("loss/total", loss.item(), step)
        writer.add_scalar("loss/mse",   parts['mse'].detach(), step)
        writer.add_scalar("loss/dist",  parts['dist'].detach(), step)
        writer.add_scalar("loss/bound", parts['bound'].detach(), step)
        writer.add_scalar("hyperparams/lr", opt.param_groups[0]['lr'], step)

        # log train mse over time for training
        if step > (warmup_steps + pretrain_steps):
            for t in range(args.T):
                writer.add_scalar(f"mse/train_over_t/", train_mse_over_t[t], args.T*step + t)

        if plot_heatmaps_in_tensorboard and (step % 100 == 0):
            imgs = pred_imgs
            mins  = train_env.ref_min #imgs.view(imgs.size(0), -1).min(1)[0].view(-1,1,1)
            maxs  = train_env.ref_max #imgs.view(imgs.size(0), -1).max(1)[0].view(-1,1,1)
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
    p.add_argument("--heliostat_distance", type=float, default=1500.0)
    p.add_argument("--azimuth", type=float, default=15.0)
    p.add_argument("--elevation", type=float, default=45.0)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--num_batches", type=int, default=1)
    p.add_argument("--steps",      type=int, default=5000)
    p.add_argument("--T",          type=int, default=6)
    p.add_argument("--k",          type=int, default=4)
    p.add_argument("--truncate_every",          type=int, default=5)
    p.add_argument("--fine_enabled", type=str, default="always",
                   help="enable fine adjustment block: none, test, always")
    p.add_argument("--detach_input", type=bool, default=True,
                   help="Whether to stop gradients from flowing theough the input of the network")
    p.add_argument("--extra_steps",          type=int, default=20)
    p.add_argument("--lr",         type=float, default=2e-4)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--use_lstm",     type=bool, default=False)
    p.add_argument("--dropout", type=float, default=0.3,
                   help="dropout rate")
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
    p.add_argument("--alignment_f",     type=float, default=100,
                   help="Weight of the alignment loss term. (only during pretraining)")
    p.add_argument("--new_errors_every_reset", type=bool, default=False,
                   help="Whether to resample errors every reset.")
    p.add_argument("--new_sun_pos_every_reset", type=bool, default=False,
                   help="Whether to sample a new sun position every reset.")
    p.add_argument("--warmup_steps", type=int, default=40,
                   help="Number of initial steps that use only the boundary "
                        "loss before switching to the full loss.")
    p.add_argument("--alignment_pretrain_steps", type=int, default=100, 
                   help="Number of pretraining steps with the alignment loss to"
                        " ensure that the normals are focusing in the right direction.")
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
