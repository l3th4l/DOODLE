import gymnasium as gym
import torch
import numpy as np
import torch.nn.functional as F
from gymnasium import spaces
from utils import reflect_ray, calculate_normals, display
from matplotlib import pyplot as plt
from matplotlib import animation
# Assume that TargetArea and display are also imported from utils
from target_area import TargetArea, calculate_target_coordinates

class DifferentiableHeliostatEnv(gym.Env):
    """
    A completely differentiable gym environment for heliostat control.
    
    The environment supports two control methods:
      - "aim_point": Actions shift the target coordinates (aim points).
      - "m_pos": Actions represent angular changes (rotations) applied to the reflector normals.
    
    The observation is a concatenation of the flattened rendered heatmap, the flattened reflector normals,
    the flattened heliostat positions, and the flattened sun position.
    """
    def __init__(self, control_method="aim_point", num_heliostats=3, max_steps=20, device=torch.device("cpu")):
        super().__init__()
        self.control_method = control_method
        self.device = device
        self.num_heliostats = num_heliostats

        # Placeholder for heliostat positions (will be set in reset)
        self.heliostat_positions = torch.zeros((self.num_heliostats, 3), device=self.device)
        # Light source (sun) position (will be set in reset)
        self.sun_position = torch.zeros(3, device=self.device)
        # Create a TargetArea instance (center is fixed at (0, height, 0) in meters)
        self.target_area = TargetArea(height=15.0, width=10.0)
        # Maximum number of steps per episode.
        self.max_steps = max_steps
        self.current_step = 0

        self.frames = []

        # Define action space: for either control method, we use 2 dimensions per heliostat.
        act_dim = 2 * self.num_heliostats
        self.act_dim = act_dim
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        # Define observation space:
        # For example, we render a heatmap of size (40, 60) -> 2400 elements,
        # plus normals (3*num_heliostats), heliostat positions (3*num_heliostats), and sun (3).
        obs_dim = 2400 + 6 * self.num_heliostats + 3
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Initialize placeholders for current computed values.
        self.current_normals = torch.zeros((self.num_heliostats, 3), device=self.device)
        self.current_reflections = torch.zeros((self.num_heliostats, 3), device=self.device)
        self.current_targets = torch.zeros((self.num_heliostats, 3), device=self.device)
        self.current_heatmap = torch.zeros((40, 60), device=self.device)

    def reset(self):
        #frames to store animation
        self.frames = []

        self.current_step = 0
        # Randomly sample a sun (light source) position.
        # Ensure the sun's height (y-coordinate) is positive.
        self.sun_position = torch.empty(3, device=self.device).uniform_(10, 80).requires_grad_(True)
        # Randomly sample reflector positions.
        # We ensure that the x-coordinate (index 0) is always positive.
        self.heliostat_positions = torch.empty((self.num_heliostats, 3), device=self.device).uniform_(10, 50)
        self.heliostat_positions[:, 1] *= 0

        # Randomly sample target points around the target area center.
        target_center = self.target_area.center  # [0, height, 0]
        spread = 4.0  # meters
        M2_y = target_center[1] + torch.empty(self.num_heliostats, device=self.device).uniform_(-spread, spread)
        M2_z = torch.empty(self.num_heliostats, device=self.device).uniform_(-spread, spread)
        # Global target points are in the form [0, m1, m2] (x=0).
        M2 = torch.stack([torch.zeros_like(M2_y), M2_y, M2_z], dim=1)
        
        # Compute normals using target points as M.
        self.current_normals = calculate_normals(self.sun_position, M2, self.heliostat_positions, device=self.device)
        # Compute reflection directions.
        self.current_reflections = reflect_ray(self.sun_position, self.heliostat_positions, self.current_normals,
                                                device=self.device, return_numpy=False)
        # Calculate target coordinates from these reflections.
        self.current_targets = calculate_target_coordinates(self.heliostat_positions, self.current_reflections)
        # Render heatmap using the target area's global_to_gaussian_blobs method.
        self.current_heatmap = self.target_area.global_to_gaussian_blobs(self.current_targets, image_size=(40, 60))

        self.frames.append(self.current_heatmap)

        # Build observation: flatten the heatmap, normals, heliostat positions, and sun position, then concatenate.
        obs = torch.cat([self.current_heatmap.flatten(), 
                         self.current_normals.flatten(),
                         self.heliostat_positions.flatten(),
                         self.sun_position.flatten()])
        return obs

    def step(self, action):
        self.current_step += 1
        # Ensure action is a tensor.
        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        action = action.view(self.num_heliostats, 2)

        if self.control_method == "m_pos":
            # Actions represent angular changes (in radians) for each heliostat.
            # For each heliostat, apply two rotations: about the Y-axis and Z-axis.
            theta_y = action[:, 0]
            theta_z = action[:, 1]
            cos_y = torch.cos(theta_y)
            sin_y = torch.sin(theta_y)
            cos_z = torch.cos(theta_z)
            sin_z = torch.sin(theta_z)
            # Rotation matrices: shape (num_heliostats, 3, 3)
            R_y = torch.zeros((self.num_heliostats, 3, 3), device=self.device)
            R_y[:, 0, 0] = cos_y
            R_y[:, 0, 2] = sin_y
            R_y[:, 1, 1] = 1
            R_y[:, 2, 0] = -sin_y
            R_y[:, 2, 2] = cos_y

            R_z = torch.zeros((self.num_heliostats, 3, 3), device=self.device)
            R_z[:, 0, 0] = cos_z
            R_z[:, 0, 1] = -sin_z
            R_z[:, 1, 0] = sin_z
            R_z[:, 1, 1] = cos_z
            R_z[:, 2, 2] = 1

            # Update normals: new_normals = R_z * R_y * current_normals.
            old_normals = self.current_normals.unsqueeze(-1)  # (N, 3, 1)
            new_normals = torch.bmm(R_z, torch.bmm(R_y, old_normals)).squeeze(-1)
            self.current_normals = new_normals
            # Update reflection directions and target coordinates.
            self.current_reflections = reflect_ray(self.sun_position, self.heliostat_positions, self.current_normals,
                                                    device=self.device, return_numpy=False)
            self.current_targets = calculate_target_coordinates(self.heliostat_positions, self.current_reflections)
        elif self.control_method == "aim_point":
            # Actions represent shifts in the target coordinates (y and z).
            self.current_targets[:, 1:] = self.current_targets[:, 1:] + action
            # Recompute normals using the helper function.
            self.current_normals = calculate_normals(self.sun_position, self.current_targets, self.heliostat_positions, device=self.device)
            self.current_reflections = reflect_ray(self.sun_position, self.heliostat_positions, self.current_normals,
                                                    device=self.device, return_numpy=False)
        else:
            raise ValueError("Unknown control method")

        # Render updated heatmap.
        self.current_heatmap = self.target_area.global_to_gaussian_blobs(self.current_targets, image_size=(40, 60))

        self.frames.append(self.current_heatmap)

        # Compute reward.
        max_dist = min(self.target_area.height / 2.0, self.target_area.width / 2.0)
        target_center_yz = self.target_area.center[1:].unsqueeze(0)  # (1,2)
        target_positions_yz = self.current_targets[:, 1:]  # (N,2)
        distances = torch.norm(target_positions_yz - target_center_yz, dim=1)
        reward = torch.sum(max_dist - distances)
        reward = reward - 0.01 * F.l1_loss(action, torch.zeros_like(action))
        # Build observation.
        obs = torch.cat([self.current_heatmap.flatten(), 
                         self.current_normals.flatten(),
                         self.heliostat_positions.flatten(),
                         self.sun_position.flatten()])
        done = self.current_step >= self.max_steps
        info = {}
        return obs, reward, done, False, info

    def render(self, mode = "human", name_suffix = '_apg', interval = 200):
        # Use the display function to show the 3D scene.
        if mode == "human":
            if self.control_method == "aim_point":
                display(self.heliostat_positions, self.sun_position, M=self.current_targets, device=self.device,
                        target_center=self.target_area.center,
                        target_width=self.target_area.width,
                        target_height=self.target_area.height)
            else:
                display(self.heliostat_positions, self.sun_position, n=self.current_normals, device=self.device,
                        target_center=self.target_area.center,
                        target_width=self.target_area.width,
                        target_height=self.target_area.height)
        elif mode == "rgb_array":
            fig, ax = plt.subplots()
            ims = []

            for frame in self.frames:
                frame_np = frame.detach().cpu().numpy()
                im = ax.imshow(frame_np, cmap="viridis", animated=True)
                ims.append([im])
            ax.axis("off")
            ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=1000)
           
            # Save as GIF and return path if needed.
            gif_filename = f"anim_output_episode_{name_suffix}.gif"
            ani.save(gif_filename, writer="pillow")
            plt.close(fig)

            return gif_filename
        else:
            raise NotImplementedError(f"Render mode {mode} not supported.")

        


def main():
    from matplotlib import pyplot as plt
    # Assume DifferentiableHeliostatEnv is defined and imported

    device = torch.device("cpu")  # or use "cuda" if available

    # ---------------- Test with control method "aim_point" ----------------
    print("Testing control method: aim_point")
    env_aim = DifferentiableHeliostatEnv(control_method="aim_point", device=device)
    obs = env_aim.reset()
    print("Initial observation shape (aim_point):", obs.shape)
    action = env_aim.action_space.sample()  # sample a random action
    obs, reward, done, truncated, info = env_aim.step(action)
    print("After one step (aim_point):")
    print("Observation shape:", obs.shape)
    print("Reward:", reward.item())
    print("Done:", done)
    # Render 3D scene with target area overlay
    env_aim.render()

    # Display the observation heatmap (if part of obs, e.g., first 2400 elements)
    heatmap_size = 40 * 60  # as defined in the environment
    heatmap_obs = obs[:heatmap_size].reshape(60, 40).detach().cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(heatmap_obs, cmap="hot")
    plt.title("Heatmap (aim_point control)")
    plt.colorbar()
    plt.show()

    # ---------------- Test with control method "m_pos" ----------------
    print("Testing control method: m_pos")
    env_mpos = DifferentiableHeliostatEnv(control_method="m_pos", device=device)
    obs = env_mpos.reset()
    print("Initial observation shape (m_pos):", obs.shape)
    action = env_mpos.action_space.sample()/10  # sample a random action
    obs, reward, done, truncated, info = env_mpos.step(action)
    print("After one step (m_pos):")
    print("Observation shape:", obs.shape)
    print("Reward:", reward.item())
    print("Done:", done)
    # Render 3D scene with target area overlay
    env_mpos.render()

    # Display the observation heatmap for m_pos control.
    heatmap_obs = obs[:heatmap_size].reshape(60, 40).detach().cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(heatmap_obs, cmap="hot")
    plt.title("Heatmap (m_pos control)")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
