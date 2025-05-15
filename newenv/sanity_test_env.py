#!/usr/bin/env python3
"""
Sanity check for HelioEnv:
 1) Step with perfect (ideal) normals
 2) Reset and display initial images
 3) Random action and display images
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

from test_environment import HelioEnv

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Geometry setup (same as in your env)
N = 50
heliostat_pos = np.random.rand(N, 3) * 10
heliostat_pos[:, 2] = 0.0

targ_pos = np.array([0.0, -5.0, 0.0])
targ_norm = np.array([0.0, 1.0, 0.0])
targ_area = (15.0, 15.0)

# Instantiate environment
env = HelioEnv(
    heliostat_pos=heliostat_pos,
    targ_pos=targ_pos,
    targ_area=targ_area,
    targ_norm=targ_norm,
    sigma_scale=0.1,
    error_scale_mrad=180.0,
    initial_action_noise=0.0,
    resolution=128,
    batch_size=25,
    device='cpu',  # or 'cpu'
    new_sun_pos_every_reset=False,
    new_errors_every_reset=True,
)

# 1) Step with ideal normals
obs0 = env.reset()
ideal_normals = env.ref_field.calculate_ideal_normals(env.sun_pos).flatten(1)
obs1, metrics1 = env.step(ideal_normals)
print("Metrics with ideal normals:", {k: v.item() for k, v in metrics1.items()})

imgs_reset = obs0['img'].cpu().numpy()
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(imgs_reset[idx], cmap='hot')
    ax.axis('off')
fig.suptitle('Images Initial')
plt.tight_layout()
plt.show()


# 3) ideal action and compare images
obs_ideal, metrics_ideal = env.step(ideal_normals)
print("Metrics with ideal action:", {k: v.item() for k, v in metrics_ideal.items()})
imgs_rand = obs_ideal['img'].cpu().numpy()
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(imgs_rand[idx], cmap='hot')
    ax.axis('off')
fig.suptitle('Images After Ideal Action')
plt.tight_layout()
plt.show()

#difference between initial and ideal action images
diff_imgs = np.abs(obs1['img'].cpu().numpy() - obs0['img'].cpu().numpy())
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(diff_imgs[idx], cmap='hot')
    ax.axis('off')
fig.suptitle('Absolute Difference Between Initial and Ideal Action Images')
plt.tight_layout()
plt.show()

# 2) Reset and show images
obs_reset = env.reset()
imgs_reset = obs_reset['img'].cpu().numpy()
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(imgs_reset[idx], cmap='hot')
    ax.axis('off')
fig.suptitle('Images After Reset')
plt.tight_layout()
plt.show()

#absolute difference between the two images
diff_imgs = np.abs(imgs_reset - obs0['img'].cpu().numpy())
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(diff_imgs[idx], cmap='hot')
    ax.axis('off')
fig.suptitle('Absolute Difference Between Initial and Reset Images')
plt.tight_layout()
plt.show()

# 3) Random action and show images
rnd = env.action_space.sample()
rnd_action = ideal_normals + 8#torch.tensor(rnd, dtype=torch.float32, device=env.device)
obs_rand, metrics_rand = env.step(rnd_action)
print("Metrics with random action:", {k: v.item() for k, v in metrics_rand.items()})
imgs_rand = obs_rand['img'].cpu().numpy()
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(imgs_rand[idx], cmap='hot')
    ax.axis('off')
fig.suptitle('Images After Random Action')
plt.tight_layout()
plt.show()