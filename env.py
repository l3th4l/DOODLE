import gymnasium as gym
import torch
import numpy as np
import torch.nn.functional as F
from gymnasium import spaces
from utils import reflect_ray, calculate_normals, display
from matplotlib import pyplot as plt
from matplotlib import animation
from target_area import TargetArea, calculate_target_coordinates
import math

class DifferentiableHeliostatEnv(gym.Env):
    """
    A completely differentiable gym environment for heliostat control.
    
    The environment supports two control methods:
      - "aim_point": Actions shift the target coordinates (aim points).
      - "m_pos": Actions represent angular changes (rotations) applied to the reflector normals.
    
    The observation is a concatenation of the flattened rendered heatmap, the flattened reflector normals,
    the flattened heliostat positions, and the flattened sun position.
    """
    def __init__(self, control_method="aim_point", num_heliostats=3, max_steps=20, device=torch.device("cpu"), error_magnitude=0, state_type = 'flat'):
        super().__init__()
        self.control_method = control_method
        self.device = device
        self.num_heliostats = num_heliostats

        self.error_magnitude = error_magnitude
        self.error_angles = None

        assert((state_type == 'flat') or (state_type == 'sep'))
        self.state_type = state_type

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

        self.movement_coef = 10.0

    def _rotate_vector(self, v, axis, angle):
        """
        Rotate vector v around the given axis by angle (in radians) using Rodrigues' rotation formula.
        v and axis are tensors of shape (3,).
        """
        cos_angle = torch.cos(torch.tensor(angle, device=self.device))
        sin_angle = torch.sin(torch.tensor(angle, device=self.device))
        return v * cos_angle + torch.cross(axis, v) * sin_angle + axis * torch.dot(axis, v) * (1 - cos_angle)

    def reset(self):
        error_magnitude = self.error_magnitude

        self.movement_coef *= 0.8
        
        # Sample error angles if errors are enabled.
        if error_magnitude > 0:
            self.error_angles = torch.empty((self.num_heliostats, 3), device=self.device).uniform_(-error_magnitude, error_magnitude)
        else:
            self.error_angles = None

        # Reset frames for animation.
        self.frames = []
        self.current_step = 0

        # Randomly sample a sun (light source) position. Ensure the sun's height is positive.
        self.sun_position = torch.empty(3, device=self.device).uniform_(10, 80).requires_grad_(True)
        # Sample a random rotation axis for the sun (fixed for the episode).
        rand_axis = torch.randn(3, device=self.device)
        self.sun_rotation_axis = rand_axis / torch.norm(rand_axis)

        # Randomly sample reflector positions. Ensure x-coordinate is positive.
        self.heliostat_positions = torch.empty((self.num_heliostats, 3), device=self.device).uniform_(10, 50)
        self.heliostat_positions[:, 1] *= 0

        # Randomly sample target points around the target area center.
        target_center = self.target_area.center  # [0, height, 0]
        spread = min(self.target_area.height / 2.0, self.target_area.width / 2.0)
        M2_y = target_center[1] + torch.empty(self.num_heliostats, device=self.device).uniform_(-spread, spread)
        M2_z = torch.empty(self.num_heliostats, device=self.device).uniform_(-spread, spread)
        M2 = torch.stack([torch.zeros_like(M2_y), M2_y, M2_z], dim=1)
        
        # Compute normals using target points as M.
        self.current_normals = calculate_normals(self.sun_position, M2, self.heliostat_positions, device=self.device)
        # Compute reflection directions.
        self.current_reflections = reflect_ray(self.sun_position, self.heliostat_positions, self.current_normals,
                                                device=self.device, return_numpy=False)
        # Calculate target coordinates.
        self.current_targets = calculate_target_coordinates(self.heliostat_positions, self.current_reflections)
        # Render heatmap.
        self.current_heatmap = self.target_area.global_to_gaussian_blobs(self.current_targets, image_size=(40, 60))
        self.frames.append(self.current_heatmap)

        # Save the initial features as "previous" for later updates.
        if self.control_method == "aim_point":
            self.prev_features = self.current_targets.clone()
        else:
            self.prev_features = self.current_normals.clone()

        # Build observation.
        if error_magnitude > 0:
            obs_normals = self._apply_error(self.current_normals, self.error_angles).flatten()
        else:
            obs_normals = self.current_normals.flatten()
        
        if self.state_type == 'flat':
            obs = torch.cat([self.current_heatmap.flatten(), 
                            obs_normals,
                            self.heliostat_positions.flatten(),
                            self.sun_position.flatten()])
        elif self.state_type == 'sep':
            obs = dict()
            obs['heatmap'] = self.current_heatmap
            obs['info_vec'] = torch.cat([obs_normals,
                            self.heliostat_positions.flatten(),
                            self.sun_position.flatten()])
        return obs

    def step(self, action):
        error_magnitude = self.error_magnitude

        # Rotate the sun every step by 15/60 degrees (pi/720 radians) around the fixed rotation axis.
        angle = math.pi / 720.0
        self.sun_position = self._rotate_vector(self.sun_position, self.sun_rotation_axis, angle)

        # Save current features before updating.
        if self.control_method == "aim_point":
            prev_features = self.current_targets.clone()
        else:
            prev_features = self.current_normals.clone()

        self.current_step += 1
        if not torch.is_tensor(action):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        action = action.view(self.num_heliostats, 2)

        if self.control_method == "m_pos":
            theta_y = action[:, 0]
            theta_z = action[:, 1]
            cos_y = torch.cos(theta_y)
            sin_y = torch.sin(theta_y)
            cos_z = torch.cos(theta_z)
            sin_z = torch.sin(theta_z)
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

            old_normals = self.current_normals.unsqueeze(-1)
            new_normals = torch.bmm(R_z, torch.bmm(R_y, old_normals)).squeeze(-1)
            self.current_normals = new_normals
            self.current_reflections = reflect_ray(self.sun_position, self.heliostat_positions, self.current_normals,
                                                    device=self.device, return_numpy=False)
            self.current_targets = calculate_target_coordinates(self.heliostat_positions, self.current_reflections)
        elif self.control_method == "aim_point":
            self.current_targets[:, 1:] = self.current_targets[:, 1:] + action
            self.current_normals = calculate_normals(self.sun_position, self.current_targets, self.heliostat_positions, device=self.device)
            self.current_reflections = reflect_ray(self.sun_position, self.heliostat_positions, self.current_normals,
                                                    device=self.device, return_numpy=False)
        else:
            raise ValueError("Unknown control method")

        self.current_heatmap = self.target_area.global_to_gaussian_blobs(self.current_targets, image_size=(40, 60))
        self.frames.append(self.current_heatmap)

        max_dist = min(self.target_area.height / 2.0, self.target_area.width / 2.0)**2
        target_center_yz = self.target_area.center[1:].unsqueeze(0)
        target_positions_yz = self.current_targets[:, 1:]
        distances = torch.norm(target_positions_yz - target_center_yz, dim=1)

        reward = torch.sum(max_dist - torch.square(distances))
        # Instead of marking done immediately, update features where (max_dist - distances²) < 0.
        mask = (max_dist - torch.square(distances)) < 0
        if mask.any():
            if self.control_method == "aim_point":
                self.current_targets[mask] += (prev_features[mask] - self.current_targets[mask]).detach()
            else:
                self.current_normals[mask] += (prev_features[mask] - self.current_normals[mask]).detach()
        reward = reward - 0.000005 * F.l1_loss(action, torch.zeros_like(action)) * (self.movement_coef >= 0.1)
        
        if error_magnitude > 0:
            obs_normals = self._apply_error(self.current_normals, self.error_angles).flatten()
        else:
            obs_normals = self.current_normals.flatten()
        
        if self.state_type == 'flat':
            obs = torch.cat([self.current_heatmap.flatten(), 
                            obs_normals,
                            self.heliostat_positions.flatten(),
                            self.sun_position.flatten()])
        elif self.state_type == 'sep':
            obs = dict()
            obs['heatmap'] = self.current_heatmap
            obs['info_vec'] = torch.cat([obs_normals,
                            self.heliostat_positions.flatten(),
                            self.sun_position.flatten()])

        done = (self.current_step >= self.max_steps)
        done = done or (reward <= 0)
        info = {}
        return obs, reward, done, False, info

    def render(self, mode="human", name_suffix='_apg', interval=200):
        if mode == "human":
            if self.control_method == "aim_point":
                display(self.heliostat_positions, self.sun_position, 
                        M=self.current_targets, device=self.device,
                        target_center=self.target_area.center,
                        target_width=self.target_area.width,
                        target_height=self.target_area.height)
            else:
                display(self.heliostat_positions, self.sun_position, 
                        n=self.current_normals, device=self.device,
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
            gif_filename = f"anim_output_episode_{name_suffix}.gif"
            ani.save(gif_filename, writer="pillow")
            plt.close(fig)
            return gif_filename
        else:
            raise NotImplementedError(f"Render mode {mode} not supported.")

    def _apply_error(self, normals, error_angles):
        N = normals.shape[0]
        cos_x = torch.cos(error_angles[:, 0])
        sin_x = torch.sin(error_angles[:, 0])
        R_x = torch.zeros((N, 3, 3), device=self.device)
        R_x[:, 0, 0] = 1
        R_x[:, 1, 1] = cos_x
        R_x[:, 1, 2] = -sin_x
        R_x[:, 2, 1] = sin_x
        R_x[:, 2, 2] = cos_x

        cos_y = torch.cos(error_angles[:, 1])
        sin_y = torch.sin(error_angles[:, 1])
        R_y = torch.zeros((N, 3, 3), device=self.device)
        R_y[:, 0, 0] = cos_y
        R_y[:, 0, 2] = sin_y
        R_y[:, 1, 1] = 1
        R_y[:, 2, 0] = -sin_y
        R_y[:, 2, 2] = cos_y

        cos_z = torch.cos(error_angles[:, 2])
        sin_z = torch.sin(error_angles[:, 2])
        R_z = torch.zeros((N, 3, 3), device=self.device)
        R_z[:, 0, 0] = cos_z
        R_z[:, 0, 1] = -sin_z
        R_z[:, 1, 0] = sin_z
        R_z[:, 1, 1] = cos_z
        R_z[:, 2, 2] = 1

        R = torch.bmm(R_z, torch.bmm(R_y, R_x))
        normals_rot = torch.bmm(R, normals.unsqueeze(-1)).squeeze(-1)
        return normals_rot


def main():
    from matplotlib import pyplot as plt
    device = torch.device("cpu")
    print("Testing control method: aim_point")
    env_aim = DifferentiableHeliostatEnv(control_method="aim_point", device=device)
    obs = env_aim.reset()
    print("Initial observation shape (aim_point):", obs.shape)
    action = env_aim.action_space.sample()
    obs, reward, done, truncated, info = env_aim.step(action)
    print("After one step (aim_point):")
    print("Observation shape:", obs.shape)
    print("Reward:", reward.item())
    print("Done:", done)
    env_aim.render()

    heatmap_size = 40 * 60
    heatmap_obs = obs[:heatmap_size].reshape(60, 40).detach().cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(heatmap_obs, cmap="hot")
    plt.title("Heatmap (aim_point control)")
    plt.colorbar()
    plt.show()

    print("Testing control method: m_pos")
    env_mpos = DifferentiableHeliostatEnv(control_method="m_pos", device=device)
    obs = env_mpos.reset()
    print("Initial observation shape (m_pos):", obs.shape)
    action = env_mpos.action_space.sample()/10
    obs, reward, done, truncated, info = env_mpos.step(action)
    print("After one step (m_pos):")
    print("Observation shape:", obs.shape)
    print("Reward:", reward.item())
    print("Done:", done)
    env_mpos.render()

    heatmap_obs = obs[:heatmap_size].reshape(60, 40).detach().cpu().numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(heatmap_obs, cmap="hot")
    plt.title("Heatmap (m_pos control)")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
