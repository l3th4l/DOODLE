from newenv import HelioField
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

# Define a function to apply the Sobel filter.
# This function expects a 2D tensor (height, width) as input and returns a 2-channel tensor
# containing the gradients in the x and y directions.
def apply_sobel(x):
    # Expand dimensions to (1, 1, H, W) so that convolution can be applied.
    x = x.unsqueeze(0).unsqueeze(0)
    # Define Sobel kernels.
    sobel_x = torch.tensor([[1., 0., -1.],
                            [2., 0., -2.],
                            [1., 0., -1.]], device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1., 2., 1.],
                            [0., 0., 0.],
                            [-1., -2., -1.]], device=x.device).view(1, 1, 3, 3)
    # Pad the image to maintain the same output dimensions.
    x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
    # Convolve with the Sobel kernels.
    grad_x = F.conv2d(x_pad, sobel_x)
    grad_y = F.conv2d(x_pad, sobel_y)
    # Concatenate the gradients along the channel dimension.
    grad = torch.cat((grad_x, grad_y), dim=1)  # Shape: (1, 2, H, W)
    grad = grad.squeeze(0)  # Remove the batch dimension → (2, H, W)
    return grad

# Configuration
sun_position = torch.tensor([5.0, 20.0, 15.0])
heliostat_positions = torch.tensor([[0.0, 0.0, 0.0]])
heliostat_normals = torch.tensor([[0.0, -1.0, 1.0]])
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
    error_scale=0.4
)

# Freeze other parameters and optimize error vectors.
noisy_field.error_vectors.requires_grad_(True)
optimizer = torch.optim.Adam([noisy_field.error_vectors], lr=0.05)

# Generate the reference image.
target_image = reference_field.render(sun_position).detach()

# Initial image for plotting.
init_image = noisy_field.render(sun_position)

# Optimization loop.
for step in range(200):
    optimizer.zero_grad()
    image = noisy_field.render(sun_position)
    
    # --- New steps before calculating the loss ---
    # 1. Apply Sobel filter to both the generated and the target image.
    image_sobel = apply_sobel(image)      # Shape: (2, H, W)
    target_sobel = apply_sobel(target_image)
    
    # 2. Apply softmax normalization.
    # Unsqueeze to add a batch dimension so that softmax is applied along the channel axis.
    image_softmax = F.softmax(image_sobel.unsqueeze(0), dim=1)   # → shape: (1, 2, H, W)
    target_softmax = F.softmax(target_sobel.unsqueeze(0), dim=1)   # → shape: (1, 2, H, W)
    
    # 3. Prepare targets for cross entropy loss.
    # For cross entropy, the target should consist of class indices.
    # Here we take the argmax along the channel dimension.
    target_labels = target_softmax.argmax(dim=1)  # → shape: (1, H, W)
    
    # 4. Calculate the loss on the updated images.
    loss_ce = F.cross_entropy(image_softmax, target_labels)
    loss_mse = F.mse_loss(image_softmax, target_softmax)
    loss_kl = F.kl_div(image_softmax / torch.sum(image_softmax), target_softmax / torch.sum(target_softmax))
    loss = loss_kl + loss_mse
    # --- End of new steps ---
    
    loss.backward()
    optimizer.step()

    if step % 20 == 0 or step == 199:
        print(f"Step {step} | Loss: {loss.item():.6f}")

# Final result.
final_image = noisy_field.render(sun_position).detach()

# Visualization.
fig, axs = plt.subplots(1, 4, figsize=(12, 4))
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

for ax in axs:
    ax.set_xlabel("East (m)")
    ax.set_ylabel("Up (m)")

plt.tight_layout()
plt.show()
