import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

import gymnasium as gym
from gymnasium import spaces

from newenv_rl_test_multi_error import HelioField


def make_distance_maps(imgs, thr=0.5, device='cpu'):
    """
    Compute distance transform maps for a batch of images.
    """
    maps = []
    for img in imgs.cpu().numpy():
        mask = (img > thr * img.max()).astype(np.uint8)
        maps.append(distance_transform_edt(1 - mask))
    return torch.tensor(np.stack(maps), dtype=torch.float32, device=device)


class HelioEnv(gym.Env):
    """
    Gym-like environment wrapping HelioField for multi-error heliostat control.
    Observation: dict with keys 'img' (current rendered heatmap) and 'aux' (sun + ideal normals).
    Action: flattened normals of shape (batch_size, num_heliostats * 3), values in [-1,1].
    Step returns (obs, metrics)
    """
    metadata = {'render.modes': []}

    def __init__(
        self,
        heliostat_pos,
        targ_pos,
        targ_area,
        targ_norm,
        sigma_scale=0.1,
        error_scale_mrad=180.0,
        initial_action_noise=0.0,
        resolution=128,
        batch_size=25,
        device='cuda',
        new_sun_pos_every_reset=False,
        new_errors_every_reset=False,
    ):
        super().__init__()
        # Device and parameters
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.new_sun_pos_every_reset = new_sun_pos_every_reset
        self.new_errors_every_reset = new_errors_every_reset

        # Geometry (ensure float32)
        self.heliostat_pos = torch.tensor(heliostat_pos, device=self.device, dtype=torch.float32)
        self.targ_pos = torch.tensor(targ_pos, device=self.device, dtype=torch.float32)
        self.targ_norm = torch.tensor(targ_norm, device=self.device, dtype=torch.float32)
        self.targ_area = targ_area
        self.resolution = resolution

        # Create reference and noisy fields
        self.ref_field = HelioField(
            self.heliostat_pos, self.targ_pos, self.targ_area, self.targ_norm,
            sigma_scale=sigma_scale, error_scale_mrad=0.0,
            initial_action_noise=initial_action_noise,
            resolution=resolution, device=self.device,
            max_batch_size=self.batch_size,
        )
        self.noisy_field = HelioField(
            self.heliostat_pos, self.targ_pos, self.targ_area, self.targ_norm,
            sigma_scale=sigma_scale, error_scale_mrad=error_scale_mrad,
            initial_action_noise=initial_action_noise,
            resolution=resolution, device=self.device,
            max_batch_size=self.batch_size,
        )

        # Sample initial sun positions and compute distance maps
        self._sample_sun_pos()
        self.ref_field.init_actions(self.sun_pos)
        with torch.no_grad():
            timg = self.ref_field.render(self.sun_pos, self.ref_field.initial_action)
        self.distance_maps = make_distance_maps(timg, device=self.device)

        # Define action and observation spaces
        num_h = self.heliostat_pos.shape[0]
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.batch_size, num_h * 3), dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            'img': spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.batch_size, resolution, resolution), dtype=np.float32
            ),
            'aux': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.batch_size, 3 + num_h * 3), dtype=np.float32
            )
        })

    def _sample_sun_pos(self):
        sun_dirs = F.normalize(torch.randn(self.batch_size, 3, device=self.device), dim=1)
        radius = math.hypot(1000, 1000)
        self.sun_pos = sun_dirs * radius

    def reset(self):
        """
        Reset environment state. Samples new sun positions or errors if flags set.
        Returns:
            obs (dict): {'img': Tensor, 'aux': Tensor}
        """
        if self.new_sun_pos_every_reset:
            self._sample_sun_pos()
            self.ref_field.init_actions(self.sun_pos)
            with torch.no_grad():
                timg = self.ref_field.render(self.sun_pos, self.ref_field.initial_action)
            self.distance_maps = make_distance_maps(timg, device=self.device)
        if self.new_errors_every_reset:
            self.noisy_field.reset_errors()

        with torch.no_grad():
            self.noisy_field.init_actions(self.sun_pos)
            init_action = self.noisy_field.initial_action
            img = self.noisy_field.render(self.sun_pos, init_action)

        ideal_normals = self.ref_field.calculate_ideal_normals(self.sun_pos)
        aux = torch.cat([self.sun_pos, ideal_normals.flatten(1)], dim=1)

        return {'img': img, 'aux': aux}

    def step(self, action):
        """
        Apply action to noisy field, render new image, and compute metrics.
        Args:
            action (Tensor): shape (batch_size, num_h * 3)
        Returns:
            obs (dict): {'img': Tensor, 'aux': Tensor}
            metrics (dict): {'mse', 'dist', 'bound'}
        """
        with torch.no_grad():
            img = self.noisy_field.render(self.sun_pos, action)

        # Compute auxiliary input
        ideal_normals = self.ref_field.calculate_ideal_normals(self.sun_pos)
        aux = torch.cat([self.sun_pos, ideal_normals.flatten(1)], dim=1)

        # Compute losses
        mx = img.amax((1,2), keepdim=True).clamp_min(1e-6)
        target = self.ref_field.render(self.sun_pos, ideal_normals.flatten(1).detach())
        pred_n = img / mx
        targ_n = target / mx

        mse = F.mse_loss(pred_n, targ_n)
        err = (pred_n - targ_n).abs()
        dist_l = (err * self.distance_maps).sum((1,2)).mean()

        # Boundary error
        normals = action.view(self.batch_size, -1, 3)
        u = torch.tensor([1., 0., 0.], device=self.device, dtype=torch.float32)
        v = torch.tensor([0., 0., 1.], device=self.device, dtype=torch.float32)
        dots = torch.einsum('bij,j->bi', normals, self.targ_norm)
        eps = 1e-6
        valid = (dots.abs() > eps)
        tval = torch.einsum('j,bij->bi', self.targ_pos, normals) / (dots + (~valid).float()*eps)
        inter = self.heliostat_pos.unsqueeze(0) + normals * tval.unsqueeze(2)
        local = inter - self.targ_pos
        xl = torch.einsum('bij,j->bi', local, u)
        yl = torch.einsum('bij,j->bi', local, v)
        hw, hh = self.targ_area[0] / 2, self.targ_area[1] / 2
        dx = F.relu(xl.abs() - hw)
        dy = F.relu(yl.abs() - hh)
        dist = torch.sqrt(dx*dx + dy*dy + 1e-8)
        inside = (xl.abs() <= hw) & (yl.abs() <= hh) & valid
        bound = (dist * (~inside).float()).mean()

        metrics = {'mse': mse, 'dist': dist_l, 'bound': bound}
        obs = {'img': img, 'aux': aux}
        return obs, metrics
