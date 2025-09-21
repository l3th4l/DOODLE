#A gymnasium-like environment for the heliostat field

from newenv_rl_test_multi_error import HelioField   # the multi-error env
from scipy.ndimage import distance_transform_edt

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import math

import gymnasium as gym
from gymnasium import spaces

#----------------------------------------------------------------------
# Helper functions
#----------------------------------------------------------------------

def azimuth_elevation_to_primary_direction(azimuth_deg: float,
                                           elevation_deg: float,
                                           device=None) -> torch.Tensor:
    """
    Convert azimuth and elevation angles (in degrees) into a 3D unit direction vector.

    Args:
        azimuth_deg:   Azimuth angle in degrees (0° = +X axis, increases CCW towards +Y).
        elevation_deg: Elevation angle in degrees (0° = horizon, 90° = zenith).
        device:        Torch device for the output tensor (optional).

    Returns:
        Tensor of shape (3,) representing the direction vector.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)

    x = math.cos(el) * math.cos(az)
    y = math.cos(el) * math.sin(az)
    z = math.sin(el)

    vec = torch.tensor([x, y, z], dtype=torch.float32, device=device)
    return vec / torch.norm(vec)

def sample_cone_directions(
    n: int,
    axis: torch.Tensor,          # shape: (3,)
    half_angle_deg: float,
    device=None,
    force_upper_hemisphere: bool = False
) -> torch.Tensor:
    """
    Uniformly sample 'n' unit vectors on the spherical cap (cone) defined by:
      - central axis 'axis' (any non-zero 3D vector)
      - half-angle 'half_angle_deg'

    Returns: (n, 3) tensor of unit vectors.
    """
    device = device or axis.device
    a = F.normalize(axis.to(device), dim=0)
    alpha = math.radians(half_angle_deg)

    # Build an orthonormal basis {u, v, a}
    # Pick a helper not (nearly) parallel to 'a'
    helper = torch.tensor([0.0, 0.0, 1.0], device=device)
    if torch.abs(a[2]) > 0.999:  # if almost parallel to z, switch helper
        helper = torch.tensor([0.0, 1.0, 0.0], device=device)

    u = F.normalize(torch.cross(helper, a), dim=0)
    v = torch.cross(a, u)

    # Sample uniformly on spherical cap:
    # cos(theta) ~ Uniform[cos(alpha), 1], phi ~ Uniform[0, 2π)
    u01 = torch.rand(n, device=device)
    cos_theta = 1.0 - u01 * (1.0 - math.cos(alpha))
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=0.0))
    phi = 2.0 * math.pi * torch.rand(n, device=device)

    # Construct directions in {u, v, a} basis
    dirs = (
        u[None, :] * (sin_theta * torch.cos(phi))[:, None] +
        v[None, :] * (sin_theta * torch.sin(phi))[:, None] +
        a[None, :] *  cos_theta[:, None]
    )
    dirs = F.normalize(dirs, dim=1)

    if force_upper_hemisphere:
        # Note: this slightly distorts the exact cone if the cone dips below z=0.
        dirs[:, 2] = torch.abs(dirs[:, 2])

    return dirs


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
            return_all = False, 
            ):
    u = target_east_axis #torch.tensor([1.,0.,0.], device=device)
    v = target_up_axis #torch.tensor([0.,0.,1.], device=device)

    border_tolerance = 0.75

    dots = torch.einsum('bij,j->bi', -vects, targ_norm)
    eps = 1e-6
    valid = (dots.abs() > eps)
    t = torch.einsum('j,bij->bi', targ_pos, vects)/(dots+(~valid).float()*eps)
    inter = heliostat_pos.unsqueeze(0) + vects*t.unsqueeze(2)
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

def calculate_angles_mrad(
        v1: torch.Tensor,
        v2: torch.Tensor,
        epsilon: float = 1e-10
) -> torch.Tensor:
    # in case v1 or v2 have shape (4,), bring to (1, 4)
    v1 = v1.unsqueeze(0) if v1.dim() == 1 else v1
    v2 = v2.unsqueeze(0) if v2.dim() == 1 else v2

    m1 = v1.double() #torch.norm(v1, dim=-1)
    m2 = v2.double() #torch.norm(v2, dim=-1)
    
    dot_products = torch.sum(v1 * v2, dim=-1)
    cos_angles = dot_products
    
    # safest upper bound just below 1 using nextafter
    one = torch.tensor(1.0, dtype=cos_angles.dtype, device=cos_angles.device)
    upper = torch.nextafter(one, torch.tensor(0.0, dtype=cos_angles.dtype, device=cos_angles.device))
    lower = -upper  # symmetric

    angles_rad = torch.acos(
        torch.clamp(cos_angles, min=lower.item()+epsilon, max=upper.item()-epsilon)
    )
    return angles_rad * 1000

""" #Alignment loss 
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
    angles_rad = 1-dot_products
    return angles_rad * 1000 """



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
                 single_sun=False,
                 azimuth = 45.0, 
                 elevation = 45.0, 
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
        self.num_heliostats =  heliostat_pos.shape[0]
        self.targ_pos = targ_pos
        self.targ_area = targ_area
        self.targ_norm = targ_norm

        self.azimuth = azimuth 
        self.elevation = elevation

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

        self.single_sun = single_sun

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
        # --- config knobs ---
        if (self.azimuth == None) or (self.elevation == None):
            use_cone = False
        else:
            use_cone = True
            primary_dir = azimuth_elevation_to_primary_direction(azimuth_deg = self.azimuth,
                                            elevation_deg = self.elevation,
                                            device=self.device).to(self.device) # example: pointing "up"
        half_angle_deg = 2.0                                             # small cone (tune as needed)
        force_upper = True                                               # mimic your abs(z) behavior

        if use_cone:
            if single_sun:
                # one direction, copy across the batch
                sun_dirs = sample_cone_directions(
                    n=1,
                    axis=primary_dir,
                    half_angle_deg=half_angle_deg,
                    device=self.device,
                    force_upper_hemisphere=force_upper
                ).repeat(self.batch_size, 1)
            else:
                # one independent direction per batch element
                sun_dirs = sample_cone_directions(
                    n=self.batch_size,
                    axis=primary_dir,
                    half_angle_deg=half_angle_deg,
                    device=self.device,
                    force_upper_hemisphere=force_upper
                )
        else:
            # your old fallback
            if single_sun:
                sun_dirs = F.normalize(torch.randn(1,3,device=self.device), dim=1).repeat(self.batch_size, 1)
            else:
                sun_dirs = F.normalize(torch.randn(self.batch_size,3,device=self.device), dim=1)
            sun_dirs[:, 2] = torch.abs(sun_dirs[:, 2])

        # Position the suns at a fixed range from origin
        radius  = math.hypot(10000, 10000)
        sun_pos = sun_dirs * radius
        self.set_sun_pos(sun_pos)


        #exponential risk 
        self.exponential_risk = exponential_risk

    def set_sun_pos_from_azimuth_elevation(self, azimuth_deg: float,
                                           elevation_deg: float,
                                           device=None):
                                           
        primary_dir = azimuth_elevation_to_primary_direction(azimuth_deg = self.azimuth,
                                                    dlevation_deg = self.elevation,
                                                    device=self.device).to(self.device) # example: pointing "up"

        if self.single_sun:
            # one direction, copy across the batch
            sun_dirs = sample_cone_directions(
                n=1,
                axis=primary_dir,
                half_angle_deg=half_angle_deg,
                device=self.device,
                force_upper_hemisphere=force_upper
            ).repeat(self.batch_size, 1)
        else:
            # one independent direction per batch element
            sun_dirs = sample_cone_directions(
                n=self.batch_size,
                axis=primary_dir,
                half_angle_deg=half_angle_deg,
                device=self.device,
                force_upper_hemisphere=force_upper
            )

    def set_sun_pos(self, sun_positions:torch.Tensor):
        #sets current sun positions to the one in the arguments (useful for evaluating test performance)
        self.sun_pos = sun_positions.clone().detach()

        self.ref_field.init_actions(self.sun_pos)
        with torch.no_grad():
            ideal_normals = self.ref_field.calculate_ideal_normals(self.sun_pos)
            timg, _ = self.ref_field.render(self.sun_pos, self.ref_field.initial_action, ideal_normals)
        self.distance_maps = make_distance_maps(timg)

        self.ref_min = torch.min(timg)
        self.ref_max = torch.max(timg)

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
                    
                ideal_normals = self.ref_field.calculate_ideal_normals(self.sun_pos)
                timg, _ = self.ref_field.render(self.sun_pos, self.ref_field.initial_action, ideal_normals)
            self.distance_maps = make_distance_maps(timg, device=self.device)

        if self.new_errors_every_reset:
            self.noisy_field.reset_errors()

        with torch.no_grad():
            self.noisy_field.init_actions(self.sun_pos)
            init_action = self.noisy_field.initial_action
            ideal_normals = self.ref_field.calculate_ideal_normals(self.sun_pos)
            img, _ = self.noisy_field.render(self.sun_pos, init_action, ideal_normals)

        ideal_normals = self.ref_field.calculate_ideal_normals(self.sun_pos)
        self.ideal_normals = ideal_normals
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

        ideal_normals = self.ref_field.calculate_ideal_normals(self.sun_pos)

        img, actual_normals, reflected_rays = self.noisy_field.render(
                                    self.sun_pos, 
                                    action, 
                                    ideal_normals,
                                    monitor=True,
                                    )

        # Compute auxiliary input
        aux = torch.cat([self.sun_pos.detach(), action.flatten(1)], dim=1)

        # Compute losses
        mx = img.amax((1,2), keepdim=True).clamp_min(1e-6)
        
        target, _ = self.ref_field.render(
                                        self.sun_pos, 
                                        ideal_normals.flatten(1), 
                                        ideal_normals,
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

        if self.use_error_mask:
            alignment_loss = torch.mean(calculate_angles_mrad(ideal_normals, actual_normals))
            mse = (F.mse_loss(pred_n * error_mask, targ_n*error_mask))#.clamp_min(1e-6)
            dist_l = (error_mask*(err * self.distance_maps)).sum((1,2)).mean()

        else:
            alignment_loss = torch.mean(calculate_angles_mrad(ideal_normals, actual_normals))
            mse = (F.mse_loss(pred_n, targ_n))#.clamp_min(1e-6)
            dist_l = (err * self.distance_maps).sum((1,2)).mean()

        # Boundary error
        normals = action.view(self.batch_size, -1, 3)
        u = torch.tensor([1., 0., 0.], device=self.device, dtype=torch.float32)
        v = torch.tensor([0., 0., 1.], device=self.device, dtype=torch.float32)
        
        if not self.exponential_risk:
            bound = boundary(normals, 
                            self.heliostat_pos, 
                            self.targ_pos, 
                            self.targ_norm, 
                            self.targ_area, 
                            u, v)
        else:
            bound = boundary(normals, 
                            self.heliostat_pos, 
                            self.targ_pos, 
                            self.targ_norm, 
                            self.targ_area, 
                            u, v, return_all=True)

            bound = torch.exp(bound + 1e-6)
            bound = torch.mean(bound)

        #monitors 
        all_bounds = boundary(normals, 
                        self.heliostat_pos, 
                        self.targ_pos, 
                        self.targ_norm, 
                        self.targ_area, 
                        u, v, return_all=True)

        all_alignment_errors = calculate_angles_mrad(ideal_normals, actual_normals).detach().view([-1])

        mean_absolute_error = err.mean(dim = [-1, -2]).view([-1, 1])

        #assert no nan in metrics
        assert not torch.isnan(mse).any(), "MSE is NaN"
        assert not torch.isnan(dist_l).any(), "Distance loss is NaN"
        assert not torch.isnan(bound).any(), "Boundary loss is NaN"
        #assert no inf in metrics   
        assert not torch.isinf(mse).any(), "MSE is Inf"
        assert not torch.isinf(dist_l).any(), "Distance loss is Inf"
        assert not torch.isinf(bound).any(), "Boundary loss is Inf"


        metrics = {'mse': mse, 'dist': dist_l, 'bound': bound, 'alignment_loss' : alignment_loss}
        obs = {'img': img, 'aux': aux}

        monitor =   {
                    'normals': normals, 
                    'reflected_rays' : reflected_rays.view([-1, 3]),
                    'ideal_normals': ideal_normals.view([-1, 3]), 
                    'all_bounds': all_bounds, 
                    'mae_image': mean_absolute_error,
                    'alignment_errors': all_alignment_errors,
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