#A gymnasium-like environment for the heliostat field

from newenv_rl_test_multi_error_debug import HelioField, ray_plane_intersection_batch   # the multi-error env
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
'''
def boundary(vects, 
            heliostat_pos, 
            targ_pos, 
            targ_norm, 
            targ_area, 
            target_east_axis,
            target_up_axis,
            return_all = False, 
            ):
    u = target_east_axis #torch.tensor([1.,0.,0.], device=device)
    v = target_up_axis #torch.tensor([0.,0.,1.], device=device)

    border_tolerance = 0.75

    inv_vects = -vects

    dots = torch.einsum('bij,j->bi', -vects, targ_norm)
    eps = 1e-6
    valid = (dots.abs() > eps)
    t = torch.einsum('j,bij->bi', targ_pos, inv_vects)/(dots+(~valid).float()*eps)
    inter = heliostat_pos.unsqueeze(0) + inv_vects*t.unsqueeze(2)
    local = inter - targ_pos
    xl = torch.einsum('bij,j->bi', local, u)
    yl = torch.einsum('bij,j->bi', local, v)

    hw, hh = (targ_area[0] * border_tolerance)/2, (targ_area[1]*border_tolerance)/2
    dx = F.relu(xl.abs()-hw*border_tolerance); dy = F.relu(yl.abs()-hh*border_tolerance)
    dist = torch.sqrt(dx*dx+dy*dy+1e-8)
    inside = (xl.abs()<=hw)&(yl.abs()<=hh)&valid
    if not return_all:
        return (dist*(~inside).float()).mean()
    else:
        return (dist*(~inside).float())
'''

def boundary(vects, 
            heliostat_pos, 
            targ_pos, 
            targ_norm, 
            targ_area, 
            target_east_axis,
            target_up_axis,
            return_all = False, 
            ):

    intersections, valid_float_mask = ray_plane_intersection_batch(
                                                heliostat_pos.view([-1, 3]), 
                                                vects.view([-1, 3]), 
                                                targ_pos, 
                                                targ_norm 
                                                ) 

    local_points = intersections - targ_pos

    xl = local_points @ target_east_axis
    yl = local_points @ target_up_axis 

    
    hw, hh = (targ_area[0] )/2, (targ_area[1])/2
    dx = F.relu(xl.abs()-hw); dy = F.relu(yl.abs()-hh)
    
    dist = torch.sqrt(dx*dx+dy*dy+1e-8)
    inside = (xl.abs()<=hw)&(yl.abs()<=hh)#&(valid_float_mask > 0)

    if not return_all:
        return (dist*(~inside).float()).mean()
    else:
        return (dist*(~inside).float()).view([-1,]).clamp_max(20)

def calculate_angles_mrad(
        v1: torch.Tensor,
        v2: torch.Tensor,
        epsilon: float = 1e-8
) -> torch.Tensor:
    # in case v1 or v2 have shape (4,), bring to (1, 4)
    v1 = v1.unsqueeze(0) if v1.dim() == 1 else v1
    v2 = v2.unsqueeze(0) if v2.dim() == 1 else v2

    m1 = torch.norm(v1, dim=-1)
    m2 = torch.norm(v2, dim=-1)
    dot_products = torch.sum(v1 * v2, dim=-1)
    cos_angles = dot_products / (m1 * m2 + epsilon)
    angles_rad = torch.acos(
        torch.clamp(cos_angles, min= -1.0 + 1e-7, max= 1.0 - 1e-7)
    )
    return angles_rad * 1000


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
                 error_mask_ratio=0.2,
                 exponential_risk=False, 
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
        action_dim = heliostat_pos.shape[0] * 3
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
        sun_dirs = sun_dirs.repeat(self.batch_size, 1)

        #make sure sun is always in the upper hemisphere (U coordinate is always positive)
        sun_dirs[:, 2] = torch.abs(sun_dirs[:, 2])
        radius   = math.hypot(1000,1000)
        self.sun_pos  = sun_dirs*radius

        self.ref_field.init_actions(self.sun_pos)
        with torch.no_grad():
            timg, _ = self.ref_field.render(self.sun_pos, self.ref_field.initial_action)
        self.distance_maps = make_distance_maps(timg)

        self.ref_min = torch.min(timg)
        self.ref_max = torch.max(timg)

        #exponential risk 
        self.exponential_risk = exponential_risk

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
                timg, _ = self.ref_field.render(self.sun_pos, self.ref_field.initial_action)
            self.distance_maps = make_distance_maps(timg, device=self.device)

        if self.new_errors_every_reset:
            self.noisy_field.reset_errors()

        with torch.no_grad():
            self.noisy_field.init_actions(self.sun_pos)
            init_action = self.noisy_field.initial_action
            img, _ = self.noisy_field.render(self.sun_pos, init_action)

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

        img, actual_normals, reflected_rays = self.noisy_field.render(
                                    self.sun_pos, 
                                    action, 
                                    monitor=True,
                                    )

        # Compute auxiliary input
        ideal_normals = self.ref_field.calculate_ideal_normals(self.sun_pos)
        aux = torch.cat([self.sun_pos, ideal_normals.flatten(1)], dim=1)

        # Compute losses
        mx = img.amax((1,2), keepdim=True).clamp_min(1e-6)
        target, _ = self.ref_field.render(
                                        self.sun_pos, 
                                        ideal_normals.flatten(1), 
                                        monitor=False
                                        )
        target = target.detach()                                            
        tx = target.amax((1,2), keepdim=True).clamp_min(1e-6)

        pred_n = img / tx
        targ_n = target / tx

        err = (pred_n - targ_n).abs()
        avg_error_per_heatmap = err.mean(dim=[-2, -1])

        # create a mask for worst 20% of the heatmaps 
        cutoff = torch.quantile(avg_error_per_heatmap, 1 - self.error_mask_ratio)
        error_mask = (avg_error_per_heatmap > cutoff).float()
        error_mask = error_mask.unsqueeze(-1).unsqueeze(-1)

        # create a mask for the worst 10% of the alignment loss 
        alignment_errors = calculate_angles_mrad(ideal_normals, actual_normals)
        alignment_cutoff = torch.quantile(alignment_errors, 1 - self.error_mask_ratio)

        alignment_mask = (alignment_errors > alignment_cutoff).float()
    
        alignment_loss = torch.mean(calculate_angles_mrad(ideal_normals, actual_normals))
        masked_alignment_loss = torch.mean(alignment_errors * alignment_mask)

        all_alignment_loss = calculate_angles_mrad(ideal_normals, actual_normals)

        if self.use_error_mask:
            mse = (F.mse_loss(pred_n * error_mask, targ_n*error_mask))#.clamp_min(1e-6)
            dist_l = (error_mask*(err * self.distance_maps)).sum((1,2)).mean()

        else:
            mse = (F.mse_loss(pred_n, targ_n))#.clamp_min(1e-6)
            dist_l = (err * self.distance_maps).sum((1,2)).mean()

        # Boundary error
        normals = action.view(self.batch_size, -1, 3)
        u = torch.tensor([1., 0., 0.], device=self.device, dtype=torch.float32)
        v = torch.tensor([0., 0., 1.], device=self.device, dtype=torch.float32)
        
        if not self.exponential_risk:
            bound = boundary(reflected_rays.view([self.batch_size, -1, 3]), 
                            self.heliostat_pos, 
                            self.targ_pos, 
                            self.targ_norm, 
                            self.targ_area, 
                            u, v)
        else:
            bound = boundary(reflected_rays.view([self.batch_size, -1, 3]), 
                            self.heliostat_pos, 
                            self.targ_pos, 
                            self.targ_norm, 
                            self.targ_area, 
                            u, v, return_all=True)

            bound = torch.exp(bound + 1e-6)
            bound = torch.mean(bound)

        #monitors 
        all_bounds = boundary(reflected_rays.view([self.batch_size, -1, 3]), 
                        self.heliostat_pos, 
                        self.targ_pos, 
                        self.targ_norm, 
                        self.targ_area, 
                        u, v, return_all=True)

        mean_absolute_error = err.mean(dim = [-1, -2]).view([-1, 1])

        #assert no nan in metrics
        assert not torch.isnan(mse).any(), "MSE is NaN"
        assert not torch.isnan(dist_l).any(), "Distance loss is NaN"
        assert not torch.isnan(bound).any(), "Boundary loss is NaN"
        #assert no inf in metrics   
        assert not torch.isinf(mse).any(), "MSE is Inf"
        assert not torch.isinf(dist_l).any(), "Distance loss is Inf"
        assert not torch.isinf(bound).any(), "Boundary loss is Inf"


        metrics =   {
                    'mse': mse, 
                    'dist': dist_l, 
                    'bound': bound, 
                    'alignment_loss' : alignment_loss, 
                    'masked_alignment_loss': masked_alignment_loss, 
                    }

        obs = {'img': img, 'aux': aux}

        monitor =   {
                    'normals': normals, 
                    'reflected_rays' : reflected_rays.view([-1, 3]),
                    'ideal_normals': ideal_normals.view([-1, 3]), 
                    'all_bounds': all_bounds, 
                    'mae_image': mean_absolute_error,
                    'all_alignment_loss' : all_alignment_loss.view([-1,])
                    }

        return obs, metrics, monitor

    def seed(self, seed=None):
        """
        Set random seed for reproducibility.
        Args:
            seed (int): Random seed value.
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)