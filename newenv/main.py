from newenv import HelioField
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import scipy

torch.autograd.set_detect_anomaly(True)


# Configuration
sun_position = torch.tensor([5.0, 20.0, 15.0])

heliostat_positions = torch.tensor([
                                    [-10.0, 10.0, 0.0], 
                                    [10.0, 10.0, 0.0], 
                                    [-10.0, 20.0, 0.0], 
                                    [10.0, 20.0, 0.0], 
                                    [0.0, 5.0, 0.0]
                                    ])

heliostat_positions = torch.rand(size = [100, 3]) * 10
heliostat_positions [:, -1] = heliostat_positions [:, -1] * 0

target_position = torch.tensor([0.0, -5.0, 0.0])
target_normal = torch.tensor([0.0, 1.0, 0.0])
target_area = (15.0, 15.0)


# Reference (noise-free) field
reference_field = HelioField(
    heliostat_positions=heliostat_positions,
    target_position=target_position,
    target_area=target_area,
    target_normal=target_normal,
    error_scale=0.0
)

# Noisy field with trainable error vectors
noisy_field = HelioField(
    heliostat_positions=heliostat_positions,
    target_position=target_position,
    target_area=target_area,
    target_normal=target_normal,
    error_scale=0.05
)

# Freeze other parameters and optimize error vectors
noisy_field.error_vectors.requires_grad_(True)
optimizer = torch.optim.Adam([noisy_field.error_vectors], lr=0.1)

# Generate reference image
target_image = reference_field.render(sun_position).detach() 
with torch.no_grad():
    reference_binary = (target_image > 0.01 * torch.max(target_image)).cpu().numpy()
    distance_map_np = scipy.ndimage.distance_transform_edt(1 - reference_binary)
    distance_map = torch.tensor(distance_map_np, device=target_image.device, dtype=torch.float32)

#Initial image for plotting 
init_image = noisy_field.render(sun_position)

# Optimization loop
for step in range(100):
    optimizer.zero_grad()
    image = noisy_field.render(sun_position)
    #loss = F.mse_loss(image, target_image)
    #loss = F.cross_entropy(image, target_image) + F.mse_loss(image, target_image)
    loss_dist = (image * distance_map).sum()
    #loss_ce = F.cross_entropy(image, target_image) + F.mse_loss(image, target_image)
    loss = loss_dist #+ loss_ce
    loss.backward()
    optimizer.step()

    if step % 20 == 0 or step == 199:
        print(f"Step {step} | Loss: {loss.item():.6f}")

# Compute error vector norms and (trivially) dot products with zero reference
learned_errors = noisy_field.error_vectors.detach()
reference_errors = reference_field.error_vectors.detach()  # Should be all zeros

# L2 norm of predicted error vectors
error_magnitudes = torch.norm(learned_errors, dim=1).cpu().numpy()

# Dot product with reference (will be 0 if reference is zero)
# For a more general case where reference_errors is non-zero:
# dot_products = (F.normalize(learned_errors, dim=1) * F.normalize(reference_errors, dim=1)).sum(dim=1).cpu().numpy()
# But since reference_errors = 0, we just skip it

# Plot error magnitudes
plt.figure(figsize=(8, 4))
plt.bar(range(len(error_magnitudes)), error_magnitudes)
plt.xlabel("Heliostat Index")
plt.ylabel("Error Vector Magnitude")
plt.title("Learned Error Vector Magnitudes per Heliostat")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print summary stats
print(f"Min error magnitude: {error_magnitudes.min():.4f}")
print(f"Max error magnitude: {error_magnitudes.max():.4f}")
print(f"Mean error magnitude: {error_magnitudes.mean():.4f}")


# Final results
final_image = noisy_field.render(sun_position).detach()

# Visualization
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
extent = [-7.5, 7.5, -7.5, 7.5]


axs[0].imshow(init_image.detach().cpu().numpy().T, cmap='hot', extent=extent, origin='lower')
axs[0].set_title("Initial Heatmap")

axs[1].imshow(target_image.cpu().numpy().T, cmap='hot', extent=extent, origin='lower')
axs[1].set_title("Reference Heatmap")

axs[2].imshow(final_image.cpu().numpy().T, cmap='hot', extent=extent, origin='lower')
axs[2].set_title("Optimized Heatmap")

diff_image = (final_image - target_image).abs()
axs[3].imshow(diff_image.cpu().numpy().T, cmap='hot', extent=extent, origin='lower')
axs[3].set_title("Absolute Error")

axs[4].imshow(distance_map_np.T, cmap='hot', extent=extent, origin='lower')
axs[4].set_title("Distance Map")

for ax in axs:
    ax.set_xlabel("East (m)")
    ax.set_ylabel("Up (m)")

plt.tight_layout()
plt.show()
