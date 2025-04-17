from newenv_rl import HelioField
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from scipy.ndimage import distance_transform_edt
import scipy

torch.autograd.set_detect_anomaly(True)



class CNNPolicyNetwork(torch.nn.Module):
    def __init__(self, image_size, num_heliostats):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(16, num_heliostats * 3)  # Assuming 3D action per heliostat

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions: [B, C, H, W]
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x.view(-1, 3)  # [num_heliostats, 3]


# Configuration
sun_position = torch.tensor([5.0, 20.0, 15.0])

heliostat_positions = torch.tensor([
                                    [-10.0, 10.0, 0.0], 
                                    [10.0, 10.0, 0.0], 
                                    [-10.0, 20.0, 0.0], 
                                    [10.0, 20.0, 0.0], 
                                    [0.0, 5.0, 0.0]
                                    ])

heliostat_positions = torch.rand(size = [50, 3]) * 10
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
    error_scale=0.0, 
    initial_error=0.0,
)

# Noisy field with trainable error vectors
noisy_field = HelioField(
    heliostat_positions=heliostat_positions,
    target_position=target_position,
    target_area=target_area,
    target_normal=target_normal,
    error_scale=0.2
)


# Generate reference image
reference_field.init_actions(sun_position)
target_image = reference_field.render(sun_position, reference_field.initial_action).detach() 
with torch.no_grad():
    reference_binary = (target_image > 0.01 * torch.max(target_image)).cpu().numpy()
    distance_map_np = scipy.ndimage.distance_transform_edt(1 - reference_binary)
    distance_map = torch.tensor(distance_map_np, device=target_image.device, dtype=torch.float32)

#Initial image for plotting 
noisy_field.init_actions(sun_position)
init_image = noisy_field.render(sun_position, noisy_field.initial_action)
old_image = init_image.detach()

image_size = target_image.shape[-1]  # Assuming square heatmap
num_heliostats = heliostat_positions.shape[0]
policy_net = CNNPolicyNetwork(image_size, num_heliostats).to(target_image.device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-2)

# Optimization loop
for step in range(500):
    optimizer.zero_grad()

    action = policy_net(old_image)  # Get action from current image
    image = noisy_field.render(sun_position, action)

    #loss = F.mse_loss(image, target_image)
    #loss = F.cross_entropy(image, target_image) + F.mse_loss(image, target_image)
    loss_dist = (image * distance_map).sum()
    #loss_ce = F.cross_entropy(image, target_image) + F.mse_loss(image, target_image)
    loss = loss_dist #+ loss_ce
    loss.backward()
    optimizer.step()

    if step % 20 == 0 or step == 199:
        print(f"Step {step} | Loss: {loss.item():.6f}")

# Final results
action = policy_net(old_image)
final_image = noisy_field.render(sun_position, action).detach()

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
