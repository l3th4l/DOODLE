#!/usr/bin/env python3
import argparse
import itertools
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CyclicLR  # switched to CyclicLR

from newenv_rl_test import HelioField

torch.autograd.set_detect_anomaly(True)

class CNNPolicyNetwork(nn.Module):
    def __init__(self, image_size, num_heliostats):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, num_heliostats * 3)

    def forward(self, x):
        B, H, W = x.shape
        x = x.view(B, 1, H, W)
        x = self.conv(x)
        x = x.view(B, -1)
        x = self.fc(x)
        return x.view(B, -1, 3)


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


def train_batched(batch_size=5, steps=500, device_str='cpu', scheduler_params=None, save_name=""):
    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
    print(f"\n=== Training on {device} with scheduler params: {scheduler_params} ===")
    # Configuration
    sun_positions = torch.tensor([[0.0, 1000.0, 1000.0]] * batch_size, device=device)
    N = 50
    heliostat_positions = torch.rand(N, 3, device=device) * 10
    heliostat_positions[:, 2] = 0.0
    #heliostat_positions[:, 0] -= 5
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
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-6)

    scheduler = None
    if scheduler_params is not None:
        # Map existing params: expect {'base_lr', 'max_lr', 'step_size_up', 'step_size_down', 'mode'}
        scheduler = CyclicLR(
            optimizer,
            base_lr=scheduler_params.get('base_lr', 1e-4),
            max_lr=scheduler_params.get('max_lr', 1e-2),
            step_size_up=scheduler_params.get('step_size_up', steps // 2),
            step_size_down=scheduler_params.get('step_size_down', steps // 2),
            mode=scheduler_params.get('mode', 'triangular2'),
            cycle_momentum=False
        )

    anti_spillage = 1000.0
    final_loss = None

    for step in range(steps):
        optimizer.zero_grad()
        action = policy_net(old_images)
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
            )
            for i in range(batch_size)
        ) / batch_size

        loss = loss_dist + anti_spillage * loss_bound
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        final_loss = loss.item()
        if step % 50 == 0 or step == steps - 1:
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            print(f"Step {step:3d} | Loss: {final_loss:.6f}  "
                  f"[dist {loss_dist.item():.6f}, bound {loss_bound.item():.6f}] | LR: {lr:.6e}")

    with torch.no_grad():
        final_actions = policy_net(old_images)
        final_images = noisy_field.render(sun_positions, final_actions.view(batch_size, -1)).to(device)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    extent = [-7.5, 7.5, -7.5, 7.5]
    axs[0].imshow(old_images[0].cpu().numpy().T, cmap='hot', extent=extent, origin='lower')
    axs[0].set_title("Initial Heatmap (sample 0)")
    axs[1].imshow(target_images[0].cpu().numpy().T, cmap='hot', extent=extent, origin='lower')
    axs[1].set_title("Reference Heatmap")
    axs[2].imshow(final_images[0].cpu().numpy().T, cmap='hot', extent=extent, origin='lower')
    axs[2].set_title("Optimized Heatmap")
    diff = (final_images - target_images).abs()[0]
    axs[3].imshow(diff.cpu().numpy().T, cmap='hot', extent=extent, origin='lower')
    axs[3].set_title("Absolute Error")
    for ax in axs:
        ax.set_xlabel("East (m)")
        ax.set_ylabel("Up (m)")
    plt.tight_layout()

    filename = f"results_{save_name}.png" if save_name else 'results.png'
    plt.savefig(filename)
    plt.show()

    return final_loss


def main():
    parser = argparse.ArgumentParser(description="Train batched heliostat policy with CyclicLR grid search")
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('--steps', type=int, default=15000, help='Training steps')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device (cpu or cuda)')
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    # Define grid for CyclicLR parameters
    grid = {
        'base_lr': [1e-3],
        'max_lr': [1e-2],
        'step_size_up': [args.steps // 2],
        'step_size_down': [args.steps // 2],
        'mode': ['triangular2'],
    }

    csv_file = 'cyclic_lr_report.csv'
    fieldnames = list(grid.keys()) + ['avg_loss']
    with open(csv_file, 'w', newline='') as report:
        writer = csv.DictWriter(report, fieldnames=fieldnames)
        writer.writeheader()

        for vals in itertools.product(*grid.values()):
            sched_params = None #dict(zip(grid.keys(), vals))
            config_name = '_'#.join(f"{k}{v}" for k, v in sched_params.items())

            losses = []
            for run_idx in range(2):
                save_name = f"{config_name}_run{run_idx+1}"
                try:
                    loss_val = train_batched(
                        batch_size=args.batch_size,
                        steps=args.steps,
                        device_str=args.device,
                        scheduler_params=sched_params,
                        save_name=save_name
                    )
                    losses.append(loss_val)
                except Exception as e:
                    print(f"Run {run_idx+1} for {config_name} failed with error: {e}, skipping")

            avg_loss = sum(losses) / len(losses) if losses else float('nan')
            row = {**sched_params, 'avg_loss': avg_loss}
            writer.writerow(row)

if __name__ == "__main__":
    main()
