import torch
import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from torch.nn.functional import normalize
from gymnasium import Env
from gymnasium.spaces import Box, Dict

from newenv_rl_test_multi_error import HelioField
from main_agent_test_random_sun import boundary_loss

class HelioEnv(Env):
    """
    A vectorized Gymnasium-like environment for heliostat control tasks.
    Runs `batch_size` parallel environments using only torch operations.

    Observation:
        A dict with keys:
        - 'image': Tensor of shape (batch_size, history_len, res, res)
        - 'ideal_normals': Tensor of shape (batch_size, history_len * num_heliostats * 3)
        - 'sun_positions': Tensor of shape (batch_size, history_len * 3)

    Step returns:
        obs, loss_dict
        where loss_dict has keys 'mse', 'dist', 'bound', each a Tensor of shape (batch_size,).
    """
    def __init__(
        self,
        heliostat_positions: torch.Tensor,
        target_position: torch.Tensor,
        target_area: tuple,
        target_normal: torch.Tensor,
        batch_size: int,
        max_past_steps: int,
        error_scale_mrad: float = 1.0,
        sigma_scale: float = 0.01,
        initial_action_noise: float = 0.01,
        resolution: int = 128,
        radius: float | None = None,
        device: str | torch.device = 'cpu',
        resample_errors_per_reset: bool = False,
        sample_random_actions: bool = True,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_past_steps = max_past_steps
        self.history_len = max_past_steps + 1
        self.resample_errors_per_reset = resample_errors_per_reset
        self.sample_random_actions = sample_random_actions
        self.initial_action_noise = initial_action_noise

        # Geometry and field
        self.heliostat_positions = torch.as_tensor(
            heliostat_positions, dtype=torch.float32, device=self.device
        )
        self.num_heliostats = self.heliostat_positions.shape[0]

        self.target_position = torch.as_tensor(
            target_position, dtype=torch.float32, device=self.device
        )
        self.target_width, self.target_height = target_area
        self.target_normal = torch.as_tensor(
            target_normal, dtype=torch.float32, device=self.device
        )
        self.target_normal = self.target_normal / self.target_normal.norm().clamp_min(1e-9)

        # resolution
        self.resolution = resolution

        # Instantiate HelioField for reference and noisy
        self.reference_field = HelioField(
            heliostat_positions, target_position, target_area, target_normal,
            error_scale_mrad=0.0, sigma_scale=sigma_scale,
            initial_action_noise=0.0, resolution=resolution,
            device=self.device, max_batch_size=batch_size
        )
        self.noisy_field = HelioField(
            heliostat_positions, target_position, target_area, target_normal,
            error_scale_mrad=error_scale_mrad, sigma_scale=sigma_scale,
            initial_action_noise=initial_action_noise, resolution=resolution,
            device=self.device, max_batch_size=batch_size
        )

        # History buffers (no gradients)
        zeros = lambda *shape: torch.zeros(*shape, device=self.device, requires_grad=False)
        self.heatmap_history = zeros(batch_size, self.history_len, resolution, resolution)
        self.ideals_history = zeros(batch_size, self.history_len, self.num_heliostats * 3)
        self.sunpos_history = zeros(batch_size, self.history_len, 3)

        # Sampling radius
        self.radius = radius or math.sqrt(1000**2 + 1000**2)
        dirs = torch.randn(self.batch_size, 3, device=self.device)
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-9)
        self.base_sun_pos = dirs * self.radius

        # Spaces
        self.action_space = Box(low=-np.inf, high=np.inf,
                                 shape=(self.num_heliostats * 3,), dtype=np.float32)
        self.observation_space = Dict({
            'image': Box(low=0.0, high=1.0,
                          shape=(self.history_len, resolution, resolution), dtype=np.float32),
            'ideal_normals': Box(low=-1.0, high=1.0,
                                  shape=(self.history_len * self.num_heliostats * 3,), dtype=np.float32),
            'sun_positions': Box(low=-np.inf, high=np.inf,
                                  shape=(self.history_len * 3,), dtype=np.float32),
        })

    def reset(self):
        """Reset environment, optionally resample errors, fill history."""
        if self.resample_errors_per_reset:
            self.reference_field.reset_errors()
            self.noisy_field.reset_errors()

        # Clear histories
        self.heatmap_history.zero_()
        self.ideals_history.zero_()
        self.sunpos_history.zero_()

        sun_pos = self.base_sun_pos
        ideal_normals = self.reference_field.calculate_ideal_normals(sun_pos)
        ideal_normals = ideal_normals.view(self.batch_size, -1).detach()

        if self.sample_random_actions:
            for t in range(self.history_len):
                # sample around ideal and normalize
                sampled = ideal_normals + torch.randn_like(ideal_normals) * self.initial_action_noise
                sampled = normalize(sampled, p=2, dim=1).detach()
                self.ideals_history[:, t, :] = sampled
                heat = self.noisy_field.render(sun_pos, sampled).detach()
                self.heatmap_history[:, t, :, :] = heat
                self.sunpos_history[:, t, :] = sun_pos.detach()
        else:
            self.reference_field.init_actions(sun_pos)
            target_images = self.reference_field.render(sun_pos, self.reference_field.initial_action)
            self.ideals_history[:, 0, :] = normalize(ideal_normals, p=2, dim=1)
            self.heatmap_history[:, 0, :, :] = target_images.detach()
            self.sunpos_history[:, 0, :] = sun_pos.detach()

        return self._get_obs()

    def step(self, action: torch.Tensor):
        """Take a step, update histories, compute losses."""
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device).view(self.batch_size, -1)
        sun_pos = self.base_sun_pos

        # Reference
        self.reference_field.init_actions(sun_pos)
        target_images = self.reference_field.render(sun_pos, self.reference_field.initial_action)
        ideal_normals = self.reference_field.calculate_ideal_normals(sun_pos)
        ideal_normals = normalize(ideal_normals.view(self.batch_size, -1), p=2, dim=1).detach()

        pred_images = self.noisy_field.render(sun_pos, action)

        # Shift
        self.heatmap_history = torch.roll(self.heatmap_history, -1, dims=1)
        self.ideals_history = torch.roll(self.ideals_history, -1, dims=1)
        self.sunpos_history = torch.roll(self.sunpos_history, -1, dims=1)

        # Append
        self.ideals_history[:, -1, :] = ideal_normals
        self.heatmap_history[:, -1, :, :] = pred_images.detach()
        self.sunpos_history[:, -1, :] = sun_pos.detach()

        # Losses
        target_max = target_images.amax((1,2), keepdim=True).clamp_min(1e-9)
        mse = (((pred_images/target_max) - (target_images/target_max))**2).mean(dim=(1,2))

        dist_maps = []
        for i in range(self.batch_size):
            mask = (target_images[i].detach().cpu().numpy() > 0.5 * target_images[i].detach().cpu().numpy().max()).astype(np.uint8)
            dist_maps.append(torch.as_tensor(distance_transform_edt(1 - mask), device=self.device))
        dist = (pred_images * torch.stack(dist_maps, dim=0)).sum(dim=(1,2))

        bounds = [
            boundary_loss(
                vectors=action[i],
                heliostat_positions=self.heliostat_positions,
                plane_center=self.target_position,
                plane_normal=self.target_normal,
                plane_width=self.target_width,
                plane_height=self.target_height
            ) for i in range(self.batch_size)
        ]
        bound = torch.stack(bounds, dim=0)

        return self._get_obs(), {'mse': mse, 'dist': dist, 'bound': bound}

    def _get_obs(self):
        return {
            'image': self.heatmap_history.clone().detach(),
            'ideal_normals': self.ideals_history.clone().detach(),
            'sun_positions': self.sunpos_history.clone().detach(),
        }
