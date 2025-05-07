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
import optuna
from torch.optim.lr_scheduler import ReduceLROnPlateau
from newenv_rl_test import HelioField

torch.autograd.set_detect_anomaly(True)

# NEXT GOAL: get to converge < 1000 eps
# Perform Bayesian Optimization on: lr only

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
    half_w = plane_width / 2
    half_h = plane_height / 2
    dx = F.relu(torch.abs(x_local) - half_w)
    dy = F.relu(torch.abs(y_local) - half_h)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
    inside = (torch.abs(x_local) <= half_w) & (torch.abs(y_local) <= half_h) & valid
    dist = dist * (~inside).float()
    return dist.mean()


def train_batched(batch_size=5, steps=500, device_str='cpu', save_name="run", lr=1e-2, cutoff= 500, distance_factor=100.0):
    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
    print(f"\n=== Training on {device} | lr={lr:.3e} | distance_factor={distance_factor:.3e}===")

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
        mask = (img > 0.01 * img.max()).astype(np.uint8)
        dist = distance_transform_edt(1 - mask)
        distance_maps.append(torch.tensor(dist, dtype=torch.float32, device=device))
    distance_maps = torch.stack(distance_maps, dim=0)

    noisy_field.init_actions(sun_positions)
    old_images = noisy_field.render(sun_positions, noisy_field.initial_action).to(device)

    policy_net = CNNPolicyNetwork(resolution, N).to(device)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50,
                                  factor=0.95, verbose=True,
                                  threshold=1e-6, threshold_mode='abs')

    anti_spillage_factor = 1000.0
    distance_factor = distance_factor
    final_loss = None

    for step in range(steps):
        optimizer.zero_grad()
        # pass sun_positions into the policy
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

        predicted = images / target_images_max
        target_normed = target_images / target_images_max
        mse_loss = F.mse_loss(predicted, target_normed)
        weighted_error = (predicted - target_normed).abs() * distance_maps

        decay = max(0, (cutoff - step) / cutoff)
        loss = anti_spillage_factor * loss_bound + mse_loss * (1 - decay) + weighted_error.mean() * decay * distance_factor

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        final_loss = loss.item()
        lr_cur = optimizer.param_groups[0]['lr']
        writer.add_scalar('Loss/total', final_loss, step)
        writer.add_scalar('Loss/dist', loss_dist.item(), step)
        writer.add_scalar('Loss/bound', loss_bound.item(), step)
        writer.add_scalar('LearningRate', lr_cur, step)

        if step % 50 == 0 or step == steps - 1:
            print(f"Step {step:3d} | Loss {final_loss:.4f} | dist {loss_dist:.4f} | bound {loss_bound:.4f} | mse {mse_loss:.4f} | LR {lr_cur:.1e}")

    writer.add_scalar('Loss/final', final_loss, steps)
    writer.close()

    with torch.no_grad():
        # inference with sun_positions
        final_actions = policy_net(old_images, sun_positions)
        final_images = noisy_field.render(sun_positions, final_actions.view(batch_size, -1)).to(device)

    # Visualization omitted for brevity
    return final_loss


def main():
    parser = argparse.ArgumentParser(description="Train batched heliostat policy with Optuna hyperparameter search")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--trials', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    torch.set_default_device(device)
    torch.manual_seed(42)
    np.random.seed(42)

    def objective(trial):
        lr = trial.suggest_float('lr', 1e1, 5e2, log=True)
        distance_factor = trial.suggest_float('distance_factor', 1e0, 1e3, log=True)
        save_name = f"optuna_lr{lr:.3f}_{distance_factor:.3f}_{trial.number}"
        print(f"\n--- Trial {trial.number} | lr={lr:.3f} ---")
        return train_batched(
            batch_size=args.batch_size,
            steps=args.steps,
            device_str=args.device,
            save_name=save_name,
            lr=lr, 
            distance_factor=distance_factor
        )

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.trials)

    best = study.best_params
    print(f"Best learning rate: lr={best['lr']:.3f}, Best distance factor: lr={best['distance_factor']:.3f}")

    # Final run with best lr
    train_batched(
        batch_size=args.batch_size,
        steps=args.steps,
        device_str=args.device,
        save_name=f"final_lr{best['lr']:.3f}_{best['distance_factor']:.3f}_final",
        lr=best['lr'], 
        distance_factor=best['distance_factor']
    )

if __name__ == "__main__":
    main()
