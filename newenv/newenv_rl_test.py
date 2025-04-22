import torch
import math

# --- Vectorized Helper Functions ---

def reflect_vectors(incidents: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Reflects a batch of incident vectors about corresponding normals."""
    normals_unit = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-9)
    dots = (incidents * normals_unit).sum(dim=1, keepdim=True)
    return incidents - 2 * dots * normals_unit


def ray_plane_intersection_batch(ray_origins: torch.Tensor,
                                 ray_dirs: torch.Tensor,
                                 plane_point: torch.Tensor,
                                 plane_normal: torch.Tensor) -> torch.Tensor:
    """Calculates intersection points of multiple rays with a single plane."""
    n_unit = plane_normal / plane_normal.norm().clamp_min(1e-9)
    denom = (ray_dirs * n_unit).sum(dim=1, keepdim=True)
    t = ((plane_point - ray_origins) * n_unit).sum(dim=1, keepdim=True) / denom
    return ray_origins + t * ray_dirs


def rotate_normals_batch(normals: torch.Tensor,
                          error_angles_mrad: torch.Tensor) -> torch.Tensor:
    """
    Applies rotational errors (in mrad) around East (X) and Up (Z) axes to normals.
    Input shapes: normals [M,3], error_angles_mrad [M,2]
    """
    angle_e = error_angles_mrad[:, 0] * 1e-3
    angle_u = error_angles_mrad[:, 1] * 1e-3
    cos_e, sin_e = angle_e.cos(), angle_e.sin()
    cos_u, sin_u = angle_u.cos(), angle_u.sin()
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    # Rotate around Z-axis (Up)
    x_u = cos_u * x - sin_u * y
    y_u = sin_u * x + cos_u * y
    z_u = z
    # Rotate around X-axis (East)
    y_e = cos_e * y_u - sin_e * z_u
    z_e = sin_e * y_u + cos_e * z_u
    return torch.stack([x_u, y_e, z_e], dim=1)


def gaussian_blur_batch(intersections: torch.Tensor,
                        heliostat_positions: torch.Tensor,
                        plane_origin: torch.Tensor,
                        plane_u: torch.Tensor,
                        plane_v: torch.Tensor,
                        width: float,
                        height: float,
                        resolution: int,
                        sigma_scale: float) -> torch.Tensor:
    """
    Computes Gaussian kernels on the target plane for each intersection.
    Returns tensor of shape [M, resolution, resolution].
    """
    M = intersections.shape[0]
    device = intersections.device
    distances = (intersections - heliostat_positions).norm(dim=1)
    sigma = (sigma_scale * distances).clamp_min(1e-9).view(M, 1, 1)
    xs = torch.linspace(-width/2, width/2, resolution, device=device)
    ys = torch.linspace(-height/2, height/2, resolution, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    base = plane_origin.view(1,1,1,3)
    pts = (base
           + grid_x.view(1,resolution,resolution,1)*plane_u.view(1,1,1,3)
           + grid_y.view(1,resolution,resolution,1)*plane_v.view(1,1,1,3))
    diffs = pts - intersections.view(M,1,1,3)
    dist_sq = diffs.pow(2).sum(dim=3)
    two_sigma_sq = 2 * sigma.pow(2)
    gauss = torch.exp(-dist_sq / two_sigma_sq)
    return gauss


class HelioField:
    def __init__(self,
                 heliostat_positions: torch.Tensor,
                 target_position: torch.Tensor,
                 target_area: tuple,
                 target_normal: torch.Tensor,
                 error_scale_mrad: float = 1.0,
                 sigma_scale: float = 0.01,
                 initial_action_noise: float = 0.01,
                 resolution: int = 100,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.heliostat_positions = torch.as_tensor(
            heliostat_positions, dtype=torch.float32, device=device)
        self.num_heliostats = self.heliostat_positions.shape[0]
        self.target_position = torch.as_tensor(
            target_position, dtype=torch.float32, device=device)
        self.target_width, self.target_height = target_area
        self.target_normal = torch.as_tensor(
            target_normal, dtype=torch.float32, device=device)
        self.target_normal = self.target_normal / self.target_normal.norm().clamp_min(1e-9)
        self.error_scale_mrad = error_scale_mrad
        self.initial_action_noise = initial_action_noise
        self.sigma_scale = sigma_scale
        self.resolution = resolution
        self.error_angles_mrad = (
            torch.randn(self.num_heliostats, 2, device=device)
            * self.error_scale_mrad)
        self.plane_u = torch.tensor([1.0,0.0,0.0], device=device)
        if torch.allclose(self.target_normal, torch.tensor([0.,1.,0.],device=device)):
            self.plane_v = torch.tensor([0.0,0.0,1.0], device=device)
        else:
            v = torch.cross(self.target_normal, self.plane_u)
            self.plane_v = v / v.norm().clamp_min(1e-9)
        self.initial_action = None

    def reset_errors(self):
        self.error_angles_mrad = (
            torch.randn(self.num_heliostats, 2, device=self.device)
            * self.error_scale_mrad)

    def calculate_ideal_normals(self, sun_position: torch.Tensor) -> torch.Tensor:
        sun = torch.as_tensor(sun_position, dtype=torch.float32, device=self.device)
        if sun.dim() == 1:
            incidents = sun.view(1,3) - self.heliostat_positions
            reflected = self.target_position.view(1,3) - self.heliostat_positions
            inc_dir = incidents / incidents.norm(dim=1,keepdim=True).clamp_min(1e-9)
            ref_dir = reflected / reflected.norm(dim=1,keepdim=True).clamp_min(1e-9)
            normals = -(inc_dir + ref_dir)
            return normals / normals.norm(dim=1,keepdim=True).clamp_min(1e-9)
        B = sun.shape[0]
        helios = self.heliostat_positions.view(1,self.num_heliostats,3)
        incidents = sun.view(B,1,3) - helios
        reflected = self.target_position.view(1,1,3) - helios
        inc_dir = incidents / incidents.norm(dim=2,keepdim=True).clamp_min(1e-9)
        ref_dir = reflected / reflected.norm(dim=2,keepdim=True).clamp_min(1e-9)
        normals = -(inc_dir + ref_dir)
        return normals / normals.norm(dim=2,keepdim=True).clamp_min(1e-9)

    def init_actions(self, sun_position: torch.Tensor):
        ideal = self.calculate_ideal_normals(sun_position)
        if ideal.dim() == 2:
            noise = torch.randn_like(ideal) * self.initial_action_noise
            noisy = ideal + noise
            noisy = noisy / noisy.norm(dim=1,keepdim=True).clamp_min(1e-9)
            self.initial_action = noisy.flatten()
        else:
            B = ideal.shape[0]
            noise = torch.randn_like(ideal) * self.initial_action_noise
            noisy = ideal + noise
            normed = noisy / noisy.norm(dim=2,keepdim=True).clamp_min(1e-9)
            self.initial_action = normed.view(B, -1)

    def render(self,
               sun_position: torch.Tensor,
               action: torch.Tensor,
               show_spillage: bool = False) -> torch.Tensor:
        sun = torch.as_tensor(sun_position, dtype=torch.float32, device=self.device)
        single = (sun.dim() == 1)
        if single:
            sun = sun.unsqueeze(0)
        B = sun.shape[0]
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        normals = act.reshape(B, self.num_heliostats, 3)
        errs = self.error_angles_mrad.unsqueeze(0).expand(B,-1,-1)
        flats = normals.reshape(-1,3)
        errs_flat = errs.reshape(-1,2)
        actual = rotate_normals_batch(flats, errs_flat)
        actual = actual / actual.norm(dim=1,keepdim=True).clamp_min(1e-9)
        actual = actual.reshape(B, self.num_heliostats, 3)
        helios = self.heliostat_positions.view(1,self.num_heliostats,3).expand(B,-1,-1)
        incidents = sun.view(B,1,3) - helios
        inc_flat = incidents.reshape(-1,3)
        refl_flat = reflect_vectors(inc_flat, actual.reshape(-1,3))
        orig_flat = helios.reshape(-1,3)
        inter_flat = ray_plane_intersection_batch(
            orig_flat, refl_flat, self.target_position, self.target_normal)
        gauss_flat = gaussian_blur_batch(
            inter_flat, orig_flat,
            self.target_position, self.plane_u, self.plane_v,
            self.target_width, self.target_height,
            self.resolution, self.sigma_scale)
        res = self.resolution
        gauss = gauss_flat.view(B, self.num_heliostats, res, res)
        images = gauss.sum(dim=1)
        sums = images.reshape(B, -1).sum(dim=1).view(B,1,1).clamp_min(1e-9)
        images = images / sums
        return images[0] if single else images