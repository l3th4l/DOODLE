#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import scipy

from newenv_rl_test import HelioField
import torch.nn.functional as F
from torch import nn

torch.autograd.set_detect_anomaly(True)

class CNNPolicyNetwork(nn.Module):
    def __init__(self, image_size, num_heliostats):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(16, num_heliostats * 3)

    def forward(self, x):
        # x: [B, H, W]  -> reshape to [B,1,H,W]
        B, H, W = x.shape
        x = x.view(B, 1, H, W)
        x = self.conv(x)
        x = x.view(B, -1)           # [B,16]
        x = self.fc(x)              # [B, N*3]
        return x.view(B, -1, 3)     # [B, N, 3]

def boundary_loss(vectors, heliostat_positions, plane_center, plane_normal, plane_width, plane_height):
    """
    vectors: [N,3]
    other args are as before.
    """
    # compute exactly as before per-sample
    dot_products = torch.einsum('ij,j->i', vectors, plane_normal)
    epsilon = 1e-6
    valid = torch.abs(dot_products) > epsilon
    t = torch.einsum('ij,j->i', plane_center - heliostat_positions, plane_normal) / (
        dot_products + (~valid).float() * epsilon
    )
    intersections = heliostat_positions + vectors * t.unsqueeze(1)
    # local coords
    plane_x_axis = torch.tensor([1.0,0.0,0.0], device=vectors.device)
    plane_y_axis = torch.tensor([0.0,0.0,1.0], device=vectors.device)
    local = intersections - plane_center
    x_local = torch.einsum('ij,j->i', local, plane_x_axis)
    y_local = torch.einsum('ij,j->i', local, plane_y_axis)
    half_w = plane_width/2; half_h = plane_height/2
    dx = F.relu(torch.abs(x_local) - half_w)
    dy = F.relu(torch.abs(y_local) - half_h)
    dist = torch.sqrt(dx*dx + dy*dy + 1e-8)
    inside = (torch.abs(x_local)<=half_w) & (torch.abs(y_local)<=half_h) & valid
    dist = dist * (~inside).float()
    return dist.mean()

def train_batched(batch_size=5, steps=500):
    # Configuration
    '''    
    sun_positions = torch.tensor([
        [5.0,  20.0, 15.0],
        [10.0, 25.0, 10.0],
        [15.0, 30.0, 20.0],
    ])[:batch_size]
    '''
    sun_positions = torch.tensor([
        [0.0,  1000.0, 1000.0],
        [0.0,  1000.0, 1000.0],
        [0.0,  1000.0, 1000.0],
        [0.0,  1000.0, 1000.0],
        [0.0,  1000.0, 1000.0],
        [0.0,  1000.0, 1000.0],
    ])[:batch_size]

    # Random heliostat layout
    N = 50
    heliostat_positions = torch.rand(N,3)*10
    heliostat_positions[:,2]=0.0

    target_position = torch.tensor([0.0,-5.0,0.0])
    target_normal   = torch.tensor([0.0, 1.0, 0.0])
    target_area     = (15.0, 15.0)
    resolution      = 128

    # Reference (noise-free) field
    reference_field = HelioField(
        heliostat_positions, target_position, target_area, target_normal,
        sigma_scale=0.1, error_scale_mrad=0.0, initial_action_noise=0.0,
        resolution=resolution, device=torch.device('cpu')
    )
    # Noisy field to train
    noisy_field = HelioField(
        heliostat_positions, target_position, target_area, target_normal,
        sigma_scale=0.1, error_scale_mrad=180.0, initial_action_noise=0.0,
        resolution=resolution, device=torch.device('cpu')
    )

    # --- Build reference images and distance maps for the batch ---
    reference_field.init_actions(sun_positions)
    with torch.no_grad():
        target_images = reference_field.render(sun_positions, reference_field.initial_action)
        # shape [B, res, res]
    # build distance maps per sample
    distance_maps = []
    for i in range(batch_size):
        img = target_images[i].cpu().numpy()
        mask = (img > 0.01 * img.max()).astype(np.uint8)
        dist = distance_transform_edt(1 - mask)
        distance_maps.append(torch.tensor(dist, dtype=torch.float32))
    distance_maps = torch.stack(distance_maps,dim=0)  # [B, res, res]

    # Initial images for plotting and policy
    noisy_field.init_actions(sun_positions)
    old_images = noisy_field.render(sun_positions, noisy_field.initial_action)  # [B, res, res]

    # Setup policy & optimizer
    policy_net = CNNPolicyNetwork(resolution, N)
    optimizer  = torch.optim.Adam(policy_net.parameters(), lr=1e-2)

    anti_spillage = 1000.0

    for step in range(steps):
        optimizer.zero_grad()
        # policy outputs [B,N,3]
        action = policy_net(old_images)
        action_flat = action.view(batch_size, -1)

        # render batch
        images = noisy_field.render(sun_positions, action_flat)  # [B,res,res]

        # distance‐map loss
        loss_dist = (images * distance_maps).sum(dim=(1,2)).mean()

        # boundary loss (loop per sample)
        loss_bound = 0.0
        for i in range(batch_size):
            loss_bound += boundary_loss(
                vectors=action[i],
                heliostat_positions=heliostat_positions,
                plane_center=target_position,
                plane_normal=target_normal,
                plane_width=target_area[0],
                plane_height=target_area[1]
            )
        loss_bound = loss_bound / batch_size

        loss = loss_dist + anti_spillage * loss_bound
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == steps-1:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}  "
                  f"[dist {loss_dist.item():.6f}, bound {loss_bound.item():.6f}]")

    # Final results
    with torch.no_grad():
        final_actions = policy_net(old_images)
        final_images  = noisy_field.render(sun_positions, final_actions.view(batch_size,-1))

    # Visualization of first sample in batch
    fig, axs = plt.subplots(1,4,figsize=(16,4))
    extent = [-7.5,7.5,-7.5,7.5]
    axs[0].imshow(old_images[0].numpy().T, cmap='hot', extent=extent, origin='lower')
    axs[0].set_title("Initial Heatmap (sample 0)")
    axs[1].imshow(target_images[0].numpy().T, cmap='hot', extent=extent, origin='lower')
    axs[1].set_title("Reference Heatmap")
    axs[2].imshow(final_images[0].numpy().T, cmap='hot', extent=extent, origin='lower')
    axs[2].set_title("Optimized Heatmap")
    diff = (final_images - target_images).abs()[0]
    axs[3].imshow(diff.numpy().T, cmap='hot', extent=extent, origin='lower')
    axs[3].set_title("Absolute Error")
    for ax in axs:
        ax.set_xlabel("East (m)"); ax.set_ylabel("Up (m)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    train_batched(batch_size=3, steps=300)
