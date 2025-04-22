import torch
import math

# --- Vectorized Helper Functions ---

def reflect_vectors(incidents: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Reflects a batch of incident vectors about corresponding normals."""
    # Normalize normals
    normals_unit = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-9)
    # Compute dot products
    dots = (incidents * normals_unit).sum(dim=1, keepdim=True)
    # Reflect
    return incidents - 2 * dots * normals_unit


def ray_plane_intersection_batch(ray_origins: torch.Tensor,
                                 ray_dirs: torch.Tensor,
                                 plane_point: torch.Tensor,
                                 plane_normal: torch.Tensor) -> torch.Tensor:
    """Calculates intersection points of multiple rays with a single plane."""
    # Normalize plane normal
    n_unit = plane_normal / plane_normal.norm().clamp_min(1e-9)
    # Compute denominators
    denom = (ray_dirs * n_unit).sum(dim=1, keepdim=True)
    # Compute t parameters
    t = ((plane_point - ray_origins) * n_unit).sum(dim=1, keepdim=True) / denom
    # Intersection points
    return ray_origins + t * ray_dirs


def rotate_normals_batch(normals: torch.Tensor,
                          error_angles_mrad: torch.Tensor) -> torch.Tensor:
    """
    Applies rotational errors (in mrad) around East (X) and Up (Z) axes to normals.
    Input shapes: normals [N,3], error_angles_mrad [N,2]
    """
    # Convert mrad to rad
    angle_e = error_angles_mrad[:, 0] * 1e-3  # rotation about X
    angle_u = error_angles_mrad[:, 1] * 1e-3  # rotation about Z

    cos_e = angle_e.cos()
    sin_e = angle_e.sin()
    cos_u = angle_u.cos()
    sin_u = angle_u.sin()

    # Rotate around Up (Z-axis)
    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]
    x_u = cos_u * x - sin_u * y
    y_u = sin_u * x + cos_u * y
    z_u = z

    # Rotate result around East (X-axis)
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
    Computes a batch of Gaussian blur kernels on the target plane for each intersection.
    Returns tensor of shape [N, resolution, resolution].
    """
    N = intersections.shape[0]
    device = intersections.device

    # Distance-based sigma for each heliostat
    distances = (intersections - heliostat_positions).norm(dim=1)
    sigma = (sigma_scale * distances).clamp_min(1e-9).view(N, 1, 1)

    # Create grid on plane
    xs = torch.linspace(-width / 2, width / 2, resolution, device=device)
    ys = torch.linspace(-height / 2, height / 2, resolution, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')  # [res, res]

    # Plane points [1, res, res, 3]
    plane_base = plane_origin.view(1, 1, 1, 3)
    plane_points = (
        plane_base
        + grid_x.view(1, resolution, resolution, 1) * plane_u.view(1, 1, 1, 3)
        + grid_y.view(1, resolution, resolution, 1) * plane_v.view(1, 1, 1, 3)
    )  # [1, res, res, 3]

    # Differences [N, res, res, 3]
    diffs = plane_points - intersections.view(N, 1, 1, 3)
    dist_sq = (diffs ** 2).sum(dim=3)  # [N, res, res]

    # Gaussian
    two_sigma_sq = 2 * (sigma ** 2)
    gauss = torch.exp(-dist_sq / two_sigma_sq)

    # Normalize each kernel
    gauss_sum = gauss.sum(dim=(1, 2), keepdim=True).clamp_min(1e-9)
    return gauss / gauss_sum


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
            heliostat_positions, dtype=torch.float32, device=device
        )
        self.num_heliostats = self.heliostat_positions.shape[0]

        self.target_position = torch.as_tensor(
            target_position, dtype=torch.float32, device=device
        )
        self.target_width, self.target_height = target_area
        self.target_normal = torch.as_tensor(
            target_normal, dtype=torch.float32, device=device
        )
        self.target_normal = self.target_normal / self.target_normal.norm().clamp_min(1e-9)

        self.error_scale_mrad = error_scale_mrad
        self.initial_action_noise = initial_action_noise
        self.sigma_scale = sigma_scale
        self.resolution = resolution

        # Sample persistent error angles [N, 2]
        self.error_angles_mrad = (
            torch.randn(self.num_heliostats, 2, device=device)
            * self.error_scale_mrad
        )

        # Plane basis
        self.plane_u = torch.tensor([1.0, 0.0, 0.0], device=device)
        if torch.allclose(self.target_normal, torch.tensor([0., 1., 0.], device=device)):
            self.plane_v = torch.tensor([0.0, 0.0, 1.0], device=device)
        else:
            v = torch.cross(self.target_normal, self.plane_u)
            self.plane_v = v / v.norm().clamp_min(1e-9)

        self.initial_action = None

    def reset_errors(self):
        """Resample rotational errors for all heliostats."""
        self.error_angles_mrad = (
            torch.randn(self.num_heliostats, 2, device=self.device)
            * self.error_scale_mrad
        )

    def calculate_ideal_normals(self, sun_position: torch.Tensor) -> torch.Tensor:
        sun = torch.as_tensor(sun_position, dtype=torch.float32, device=self.device)
        # Vector from heliostats to sun
        incidents = sun.view(1, 3) - self.heliostat_positions
        # Vector from heliostats to target
        reflected = self.target_position.view(1, 3) - self.heliostat_positions
        inc_dir = incidents / incidents.norm(dim=1, keepdim=True).clamp_min(1e-9)
        ref_dir = reflected / reflected.norm(dim=1, keepdim=True).clamp_min(1e-9)
        normals = -(inc_dir + ref_dir)
        return normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-9)

    def init_actions(self, sun_position: torch.Tensor):
        """Initialize action normals with noise."""
        ideal = self.calculate_ideal_normals(sun_position)
        noise = torch.randn_like(ideal) * self.initial_action_noise
        noisy = ideal + noise
        noisy = noisy / noisy.norm(dim=1, keepdim=True).clamp_min(1e-9)
        self.initial_action = noisy.flatten()

    def render(self,
               sun_position: torch.Tensor,
               action: torch.Tensor,
               show_spillage: bool = False) -> torch.Tensor:
        sun = torch.as_tensor(sun_position, dtype=torch.float32, device=self.device)
        # Reshape action to normals
        intended_normals = action.to(self.device).view(self.num_heliostats, 3)
        # Apply error rotations
        actual_normals = rotate_normals_batch(intended_normals, self.error_angles_mrad)
        actual_normals = actual_normals / actual_normals.norm(dim=1, keepdim=True).clamp_min(1e-9)

        # Compute incident and reflected rays
        incidents = sun.view(1, 3) - self.heliostat_positions
        reflected = reflect_vectors(incidents, actual_normals)

        # Intersection points
        intersections = ray_plane_intersection_batch(
            self.heliostat_positions, reflected,
            self.target_position, self.target_normal
        )

        # Gaussian blur contributions
        gaussians = gaussian_blur_batch(
            intersections,
            self.heliostat_positions,
            self.target_position,
            self.plane_u,
            self.plane_v,
            self.target_width,
            self.target_height,
            self.resolution,
            self.sigma_scale
        )  # [N, res, res]

        # Sum contributions
        image = gaussians.sum(dim=0)

        # Spillage count
        proj_u = ((intersections - self.target_position) @ self.plane_u) / (self.plane_u.norm()**2)
        proj_v = ((intersections - self.target_position) @ self.plane_v) / (self.plane_v.norm()**2)
        spillage_mask = (proj_u.abs() > self.target_width/2) | (proj_v.abs() > self.target_height/2)
        spillage_count = int(spillage_mask.sum().item())
        if show_spillage and spillage_count > 0:
            print("Spillage count:", spillage_count)

        # Normalize image
        total_intensity = image.sum()
        image = image / total_intensity.clamp_min(1e-9)
        return image

# --- End of module ---
