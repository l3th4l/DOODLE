#!/usr/bin/env python3
import argparse
import csv
import math

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

#import torchvision        # <-- for image grids in TensorBoard
from pathlib import Path  

from datetime import datetime

from newenv_rl_test import HelioField

torch.autograd.set_detect_anomaly(True)

class CNNPolicyNetwork(nn.Module):
    def __init__(self, image_size, num_heliostats, sun_hidden_dim=32, combined_hidden_dim=256):
        super().__init__()
        # CNN for image feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # MLP for sun-position input
        self.sun_mlp = nn.Sequential(
            nn.Linear(3, sun_hidden_dim),
            nn.ReLU(),
            nn.Linear(sun_hidden_dim, sun_hidden_dim),
            nn.ReLU()
        )
        # MLP to combine image and sun features
        self.combined_mlp = nn.Sequential(
            nn.Linear(128 + sun_hidden_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Linear(combined_hidden_dim, num_heliostats * 3)
        )

    def forward(self, x, sun_pos):
        # x: (B, H, W), sun_pos: (B, 3)
        B, H, W = x.shape
        # Image path
        x = x.view(B, 1, H, W)
        x = F.normalize(x, p=2, dim=1)
        img_feat = self.conv(x).view(B, -1)
        img_feat = F.normalize(img_feat, p=2, dim=1)
        # Sun-position path
        sun_norm = sun_pos / (sun_pos.norm(dim=1, keepdim=True) + 1e-6)
        sun_feat = self.sun_mlp(sun_norm)
        # Combine
        combined = torch.cat([img_feat, sun_feat], dim=1)
        out = self.combined_mlp(combined)
        # Reshape and normalize each vector
        out = out.view(B, -1, 3)
        out = out / (out.norm(dim=2, keepdim=True) + 1e-6)
        return out


def boundary_loss(vectors, heliostat_positions, plane_center, plane_normal,
                  plane_width, plane_height):
    dot_products = torch.einsum('ij,j->i', vectors, plane_normal)
    epsilon = 1e-6
    valid = torch.abs(dot_products) > epsilon
    t = torch.einsum('ij,j->i', plane_center - heliostat_positions, plane_normal) / (
        dot_products + (~valid).float() * epsilon
    )
    intersections = heliostat_positions + vectors * t.unsqueeze(1)
    plane_x_axis = torch.tensor([1.0, 0.0, 0.0], device=vectors.device)
    plane_y_axis = torch.tensor([0.0, 0.0, 1.0], device=vectors.device)
    local = intersections - plane_center
    x_local = torch.einsum('ij,j->i', local, plane_x_axis)
    y_local = torch.einsum('ij,j->i', local, plane_y_axis)
    half_w = plane_width * 0.75 #/ 2
    half_h = plane_height * 0.75 #/ 2
    dx = F.relu(torch.abs(x_local) - half_w)
    dy = F.relu(torch.abs(y_local) - half_h)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    inside = (torch.abs(x_local) <= half_w) & (torch.abs(y_local) <= half_h) & valid
    dist = dist * (~inside).float()
    return dist.mean()


def train_batched(batch_size=25, 
                  steps=500, 
                  device_str='cpu', 
                  save_name="run", 
                  lr=1e-2, 
                  cutoff=800, 
                  dist_factor=5000.0, 
                  lr_reduce_factor=0.9,
                  save_heatmaps: bool = False,
                  all_losses: bool = False,
                  plot_heatmaps_in_tensorboard: bool = False):

    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
    print(f"\n=== Training on {device} ===")

    writer = SummaryWriter(log_dir=f"runs_newenv/{save_name}")

    # Sample random sun positions on a sphere of radius sqrt(1000^2 + 1000^2)
    radius = math.sqrt(1000**2 + 1000**2)
    sun_directions = torch.randn(batch_size, 3, device=device)
    sun_directions = sun_directions / sun_directions.norm(dim=1, keepdim=True)
    sun_positions = sun_directions * radius

    N = 50
    heliostat_positions = torch.rand(N, 3, device=device) * 10
    heliostat_positions[:, 2] = 0.0
    target_position = torch.tensor([0.0, -5.0, 0.0], device=device)
    target_normal = torch.tensor([0.0, 1.0, 0.0], device=device)
    target_area = (15.0, 15.0)
    resolution = 128

    reference_field = HelioField(
        heliostat_positions, target_position, target_area, target_normal,
        sigma_scale=0.1, error_scale_mrad=0.0, initial_action_noise=0.0,
        resolution=resolution, device=device
    )
    noisy_field = HelioField(
        heliostat_positions, target_position, target_area, target_normal,
        sigma_scale=0.1, error_scale_mrad=180.0, initial_action_noise=0.0,
        resolution=resolution, device=device
    )

    reference_field.init_actions(sun_positions)
    with torch.no_grad():
        target_images = reference_field.render(sun_positions, reference_field.initial_action)
        target_images_max = target_images.amax(dim=(1, 2), keepdim=True).detach()
    distance_maps = []
    for i in range(batch_size):
        img = target_images[i].cpu().numpy()
        mask = (img > 0.5 * img.max()).astype(np.uint8)
        dist = distance_transform_edt(1 - mask)
        distance_maps.append(torch.tensor(dist, dtype=torch.float32, device=device))
    distance_maps = torch.stack(distance_maps, dim=0)

    noisy_field.init_actions(sun_positions)
    old_images = noisy_field.render(sun_positions, noisy_field.initial_action).to(device)

    policy_net = CNNPolicyNetwork(resolution, N).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)

    # Add ReduceLROnPlateau scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        'min', 
        patience=40, 
        factor=lr_reduce_factor, 
        verbose=True, 
        threshold=1e-6, 
        threshold_mode='abs'
    )

    anti_spillage_factor = 10000.0
    dist_factor = dist_factor
    mse_factor = 0.1
    final_loss = None

    for step in range(steps):
        optimizer.zero_grad()
        action = policy_net(old_images, sun_positions)
        images = noisy_field.render(sun_positions, action.view(batch_size, -1)).to(device)
        loss_dist = (images * distance_maps).sum(dim=(1, 2)).mean()
        loss_bound = sum(
            boundary_loss(
                vectors=action[i],
                heliostat_positions=heliostat_positions,
                plane_center=target_position,
                plane_normal=target_normal,
                plane_width=target_area[0],
                plane_height=target_area[1]
            ) for i in range(batch_size)
        ) / batch_size

        predicted_images_normed = torch.div(images, target_images_max)
        target_images_normed = torch.div(target_images, target_images_max).detach()
        mse_loss = nn.functional.mse_loss(predicted_images_normed, target_images_normed)

        error = torch.abs(predicted_images_normed - target_images_normed)
        weighted_error = error * distance_maps 

        decay_factor = max(1e-5, (cutoff - step) / cutoff)

        loss = (anti_spillage_factor * loss_bound
                #+ anti_spillage_factor * loss_bound * decay_factor
                + mse_loss * (1-decay_factor + 1e-5)
                + weighted_error.mean() * decay_factor *dist_factor)

        loss.backward()
        optimizer.step()

        final_loss = loss.item()
        lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('Loss/total', final_loss, step)
        writer.add_scalar('Loss/dist', loss_dist.item(), step)
        writer.add_scalar('Loss/bound', loss_bound.item(), step)
        writer.add_scalar('LearningRate', lr, step)

        if plot_heatmaps_in_tensorboard and (step % 100 == 0):
            # add_images expects (N, C, H, W); ensure a channel dim exists
            writer.add_images(tag="Predicted/normalized_heatmaps",
                              img_tensor=predicted_images_normed.unsqueeze(1),
                              global_step=step, dataformats='NCHW')

        # Update the learning rate scheduler
        scheduler.step(final_loss)

        if step % 50 == 0 or step == steps - 1:
            print(f"Step {step:3d} | Loss: {final_loss:.6f}  "
                  f"[dist {loss_dist.item():.6f}, bound {loss_bound:.6f}, mse {mse_loss.item():.6f}] | LR: {lr:.6e}")

    writer.add_scalar('Loss/final', final_loss, steps)
    writer.close()

    with torch.no_grad():
        final_actions = policy_net(old_images, sun_positions)
        final_images = noisy_field.render(sun_positions, final_actions.view(batch_size, -1)).to(device)

    if save_heatmaps:
        Path("heatmaps").mkdir(exist_ok=True)            # keep folder tidy
        np.savez(f"heatmaps/{save_name}_heatmaps.npz",
                 target=target_images_normed.cpu().numpy(),
                 predicted=predicted_images_normed.detach().cpu().numpy())

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    extent = [-7.5, 7.5, -7.5, 7.5]
    im_max = torch.max(target_images[1])

    axs[0].imshow(old_images[1].cpu().numpy().T, cmap='hot', extent=extent, origin='lower', vmin=0, vmax=im_max)
    axs[0].set_title("Initial Heatmap (sample 0)")
    axs[1].imshow(target_images[1].cpu().numpy().T, cmap='hot', extent=extent, origin='lower', vmin=0, vmax=im_max)
    axs[1].set_title("Reference Heatmap")
    axs[2].imshow(final_images[1].cpu().numpy().T, cmap='hot', extent=extent, origin='lower', vmin=0, vmax=im_max)
    axs[2].set_title("Optimized Heatmap")
    diff = (final_images - target_images).abs()[1]
    axs[3].imshow(diff.cpu().numpy().T, cmap='hot', extent=extent, origin='lower', vmin=0, vmax=im_max)
    axs[3].set_title("Absolute Error")
    for ax in axs:
        ax.set_xlabel("East (m)")
        ax.set_ylabel("Up (m)")
    plt.tight_layout()

    filename = f"results_{save_name}.png" if save_name else 'results.png'
    plt.savefig(filename)
    plt.show()

    if all_losses:
        return loss_dist.item(), mse_loss.item()
    return mse_loss.item()


def main():
    parser = argparse.ArgumentParser(description="Train batched heliostat policy")
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size')
    parser.add_argument('--steps', type=int, default=4000, help='Training steps')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device (cpu or cuda)')
    parser.add_argument('--lr', type=float, default=0.000717361, help='Learning rate')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    torch.set_default_device(device)

    torch.manual_seed(42)
    np.random.seed(42)

    current_time = torch.tensor([], device=device)  # unused placeholder

    train_batched(
        batch_size=args.batch_size,
        steps=args.steps,
        device_str=args.device,
        save_name=f"fixed_lr_run_{args.lr:.3f}_{datetime.now().strftime('%m_%d_%y_%H_%M')}",
        lr=args.lr, 
        lr_reduce_factor=0.274977448,
        cutoff=3789,
        dist_factor=10004.0,
        save_heatmaps=True,
        all_losses=True,
        plot_heatmaps_in_tensorboard=True
    )


if __name__ == "__main__":
    main()
