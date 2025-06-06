#A gymnasium-like environment for the heliostat field

from newenv_rl_test_multi_error import HelioField, rotate_normals_batch # the multi-error env
from scipy.ndimage import distance_transform_edt

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import math

import gymnasium as gym
from gymnasium import spaces

#----------------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------------

# distance loss function
def make_distance_maps(imgs, thr=0.5):
    maps = []
    for img in imgs.cpu().numpy():
        mask = (img > thr * img.max()).astype(np.uint8)
        maps.append(distance_transform_edt(1 - mask))
    return torch.tensor(np.stack(maps), dtype=torch.float32, device=imgs.device)

# boundary loss function
#NOTE: the boundary is now 75% of what it was before
def boundary(vects, 
            heliostat_pos, 
            targ_pos, 
            targ_norm, 
            targ_area, 
            target_east_axis,
            target_up_axis,
            ):
    u = target_east_axis #torch.tensor([1.,0.,0.], device=device)
    v = target_up_axis #torch.tensor([0.,0.,1.], device=device)

    border_tolerance = 0.75

    dots = torch.einsum('bij,j->bi', vects, targ_norm)
    eps = 1e-6
    valid = (dots.abs() > eps)
    t = torch.einsum('j,bij->bi', targ_pos, vects)/(dots+(~valid).float()*eps)
    inter = heliostat_pos.unsqueeze(0) + vects*t.unsqueeze(2)
    local = inter - targ_pos
    xl = torch.einsum('bij,j->bi', local, u)
    yl = torch.einsum('bij,j->bi', local, v)
    hw, hh = targ_area[0]/2, targ_area[1]/2
    dx = F.relu(xl.abs()-hw*border_tolerance); dy = F.relu(yl.abs()-hh*border_tolerance)
    dist = torch.sqrt(dx*dx+dy*dy+1e-8)
    inside = (xl.abs()<=hw)&(yl.abs()<=hh)&valid
    return (dist*(~inside).float()).mean()

class HelioEnv(gym.Env):

    def __init__(self,
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
                 new_errors_every_reset=True,
                 use_error_mask=False, 
                 error_mask_ratio=0.2
                ):
        super(HelioEnv, self).__init__()

        #check if heliostat_pos, targ_pos, and targ_norm are torch tensors
        if not isinstance(heliostat_pos, torch.Tensor):
            heliostat_pos = torch.tensor(heliostat_pos, dtype=torch.float32, device=device)
        if not isinstance(targ_pos, torch.Tensor):      
            targ_pos = torch.tensor(targ_pos, dtype=torch.float32, device=device)
        if not isinstance(targ_norm, torch.Tensor):
            targ_norm = torch.tensor(targ_norm, dtype=torch.float32, device=device)

        # Environment parameters
        self.resolution = resolution
        self.batch_size = batch_size
        self.device = device

        # Geometry setup
        self.heliostat_pos = heliostat_pos
        self.targ_pos = targ_pos
        self.targ_area = targ_area
        self.targ_norm = targ_norm

        # Error parameters
        self.sigma_scale = sigma_scale
        self.error_scale_mrad = error_scale_mrad

        # Action noise parameters
        self.initial_action_noise = initial_action_noise

        # Environment state variables
        self.sun_pos = None
        self.sun_errors = None

        # Flags for resetting conditions
        self.new_sun_pos_every_reset = new_sun_pos_every_reset
        self.new_errors_every_reset = new_errors_every_reset

        # Action space and observation space setup
        action_dim = heliostat_pos.shape[0] * 2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'img': spaces.Box(
                low=0.0, high=np.inf,
                shape=(self.batch_size, resolution, resolution), dtype=np.float32
            ),
            'aux': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.batch_size, 3 + self.heliostat_pos.shape[0] * 3), dtype=np.float32
            )
        })

        # Initialize the heliostat field
        self.ref_field = HelioField(
            heliostat_positions=self.heliostat_pos,
            target_position=self.targ_pos,
            target_area=self.targ_area,
            target_normal=self.targ_norm,
            sigma_scale=self.sigma_scale,
            error_scale_mrad=0.0,
            resolution=self.resolution,
            max_batch_size=self.batch_size,
            device=self.device,
        )

        self.noisy_field = HelioField(
            heliostat_positions=self.heliostat_pos,
            target_position=self.targ_pos,
            target_area=self.targ_area,
            target_normal=self.targ_norm,
            sigma_scale=self.sigma_scale,
            error_scale_mrad=self.error_scale_mrad,
            resolution=self.resolution,
            max_batch_size=self.batch_size,
            device=self.device,
        )

        # error computation parameters 
        self.use_error_mask = use_error_mask
        self.error_mask_ratio = error_mask_ratio


        # precompute distance maps
        sun_dirs = F.normalize(torch.randn(self.batch_size,3,device=self.device),dim=1)
        #make sure sun is always in the upper hemisphere (U coordinate is always positive)
        sun_dirs[:, 2] = torch.abs(sun_dirs[:, 2])
        radius   = math.hypot(1000,1000)
        self.sun_pos  = sun_dirs*radius

        self.ref_field.init_actions(self.sun_pos)
        with torch.no_grad():
            timg = self.ref_field.render(self.sun_pos, self.ref_field.initial_action)
        self.distance_maps = make_distance_maps(timg)

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
        if isinstance(action, np.ndarray):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        
        #NOTE (experimental): create dummy normals of shape (batch_size, num_h, 3) from action, all pointing north 
        dummy_normals_unbatched = torch.zeros((self.batch_size * self.heliostat_pos.shape[0], 3), device=self.device, dtype=torch.float32)
        dummy_normals_unbatched[:, 1] = 1.0  # Pointing north

        #NOTE (QUESTION) Find rotational constraints for the heliostat normals
        action_unbatched = action.view(-1, 2) # actions are angles in radians, so we can use them directly

        #TODO: rotate the normals on east axis and up axis by action degrees (allow for room for error by error_scale_mrad)
        action_normals = rotate_normals_batch(dummy_normals_unbatched, action_unbatched)
        action_normals_batched = action_normals.view(self.batch_size, -1, 3)

        img = self.noisy_field.render(self.sun_pos, action_normals_batched)

        # Compute auxiliary input
        ideal_normals = self.ref_field.calculate_ideal_normals(self.sun_pos)
        aux = torch.cat([self.sun_pos, ideal_normals.flatten(1)], dim=1)

        # Compute losses
        mx = img.amax((1,2), keepdim=True).clamp_min(1e-6)
        target = self.ref_field.render(self.sun_pos, ideal_normals.flatten(1)).detach()
        tx = target.amax((1,2), keepdim=True).clamp_min(1e-6)

        pred_n = img / mx
        targ_n = target / tx

        err = (pred_n - targ_n).abs()
        avg_error_per_heatmap = err.mean(dim=[-2, -1])

        # create a mask for worst 20% of the heatmaps 
        cutoff = torch.quantile(avg_error_per_heatmap, 1 - self.error_mask_ratio)
        error_mask = (avg_error_per_heatmap > cutoff).float()
        error_mask = error_mask.unsqueeze(-1).unsqueeze(-1)

        if self.use_error_mask:
            mse = (F.mse_loss(pred_n * error_mask, targ_n*error_mask))#.clamp_min(1e-6)
            dist_l = (error_mask*(err * self.distance_maps)).sum((1,2)).mean()

        else:

            mse = (F.mse_loss(pred_n, targ_n)).clamp_min(1e-6)
            dist_l = (err * self.distance_maps).sum((1,2)).mean()

        # Boundary error
        #normals = action.view(self.batch_size, -1, 3)
        u = torch.tensor([1., 0., 0.], device=self.device, dtype=torch.float32)
        v = torch.tensor([0., 0., 1.], device=self.device, dtype=torch.float32)
        
        bound = boundary(action_normals_batched, 
                        self.heliostat_pos, 
                        self.targ_pos, 
                        self.targ_norm, 
                        self.targ_area, 
                        u, v)

        #assert no nan in metrics
        assert not torch.isnan(mse).any(), "MSE is NaN"
        assert not torch.isnan(dist_l).any(), "Distance loss is NaN"
        assert not torch.isnan(bound).any(), "Boundary loss is NaN"
        #assert no inf in metrics   
        assert not torch.isinf(mse).any(), "MSE is Inf"
        assert not torch.isinf(dist_l).any(), "Distance loss is Inf"
        assert not torch.isinf(bound).any(), "Boundary loss is Inf"


        metrics = {'mse': mse, 'dist': dist_l, 'bound': bound}
        obs = {'img': img, 'aux': aux}
        return obs, metrics

    def seed(self, seed=None):
        """
        Set random seed for reproducibility.
        Args:
            seed (int): Random seed value.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)