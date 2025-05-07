import torch
import math

# --- Vectorized Helper Functions ------------------------------------------------

def reflect_vectors(incidents: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Reflects a batch of incident vectors about corresponding normals."""
    normals_unit = normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-9)
    dots = (incidents * normals_unit).sum(dim=1, keepdim=True)
    return incidents - 2 * dots * normals_unit


def ray_plane_intersection_batch(
    ray_origins: torch.Tensor,
    ray_dirs: torch.Tensor,
    plane_point: torch.Tensor,
    plane_normal: torch.Tensor,
) -> torch.Tensor:
    """Calculates intersection points of multiple rays with a single plane."""
    n_unit = plane_normal / plane_normal.norm().clamp_min(1e-9)
    denom = (ray_dirs * n_unit).sum(dim=1, keepdim=True)
    t = ((plane_point - ray_origins) * n_unit).sum(dim=1, keepdim=True) / denom
    return ray_origins + t * ray_dirs


def rotate_normals_batch(normals: torch.Tensor, error_angles_mrad: torch.Tensor) -> torch.Tensor:
    """Applies rotational errors (in mrad) around East (X) and Up (Z) axes to normals.

    Args:
        normals:            [N, 3] tensor of surface normals.
        error_angles_mrad:  [N, 2] tensor with (East‑axis, Up‑axis) rotation in milliradians.
    Returns:
        Rotated normals with the same shape as *normals*.
    """
    angle_e = error_angles_mrad[:, 0] * 1e-3  # convert mrad → rad
    angle_u = error_angles_mrad[:, 1] * 1e-3

    cos_e, sin_e = angle_e.cos(), angle_e.sin()
    cos_u, sin_u = angle_u.cos(), angle_u.sin()

    x, y, z = normals[:, 0], normals[:, 1], normals[:, 2]

    # Rotate around Z‑axis (Up)
    x_u = cos_u * x - sin_u * y
    y_u = sin_u * x + cos_u * y
    z_u = z

    # Rotate around X‑axis (East)
    y_e = cos_e * y_u - sin_e * z_u
    z_e = sin_e * y_u + cos_e * z_u

    return torch.stack([x_u, y_e, z_e], dim=1)


def gaussian_blur_batch(
    intersections: torch.Tensor,
    heliostat_positions: torch.Tensor,
    plane_origin: torch.Tensor,
    plane_u: torch.Tensor,
    plane_v: torch.Tensor,
    width: float,
    height: float,
    resolution: int,
    sigma_scale: float,
) -> torch.Tensor:
    """Computes Gaussian kernels on the target plane for each intersection.

    Returns a tensor of shape ``[M, resolution, resolution]``.
    """
    M = intersections.shape[0]
    device = intersections.device

    distances = (intersections - heliostat_positions).norm(dim=1)
    sigma = (sigma_scale * distances).clamp_min(1e-9).view(M, 1, 1)

    xs = torch.linspace(-width / 2, width / 2, resolution, device=device)
    ys = torch.linspace(-height / 2, height / 2, resolution, device=device)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

    base = plane_origin.view(1, 1, 1, 3)
    pts = (
        base
        + grid_x.view(1, resolution, resolution, 1) * plane_u.view(1, 1, 1, 3)
        + grid_y.view(1, resolution, resolution, 1) * plane_v.view(1, 1, 1, 3)
    )

    diffs = pts - intersections.view(M, 1, 1, 3)
    dist_sq = diffs.pow(2).sum(dim=3)
    two_sigma_sq = 2 * sigma.pow(2)

    gauss = torch.exp(-dist_sq / two_sigma_sq)
    return gauss


# --- HelioField -----------------------------------------------------------------

class HelioField:
    """Simple heliostat field model with per‑sun‑position error sampling.

    *render()* re‑uses deterministic mirror‑orientation errors for each sun‑position
    up to *max_batch_size* until :py:meth:`reset_errors` is called again, while the
    single‑sun API remains fully backwards compatible.
    """

    def __init__(
        self,
        heliostat_positions: torch.Tensor,
        target_position: torch.Tensor,
        target_area: tuple,
        target_normal: torch.Tensor,
        error_scale_mrad: float = 1.0,
        sigma_scale: float = 0.01,
        initial_action_noise: float = 0.01,
        resolution: int = 100,
        device: torch.device | str = "cpu",
        max_batch_size: int = 25,
    ) -> None:
        self.device = torch.device(device)
        self.max_batch_size = int(max_batch_size)

        # Scene geometry -------------------------------------------------------
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

        # Error & rendering parameters ----------------------------------------
        self.error_scale_mrad = float(error_scale_mrad)
        self.initial_action_noise = float(initial_action_noise)
        self.sigma_scale = float(sigma_scale)
        self.resolution = int(resolution)

        # Initialise error tensors --------------------------------------------
        # NOTE: both single‑sun and batch error tensors are created here via
        #       the shared helper to keep behaviour consistent across calls.
        self.reset_errors()

        # Basis vectors on the target plane ------------------------------------
        self.plane_u = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        if torch.allclose(
            self.target_normal, torch.tensor([0.0, 1.0, 0.0], device=self.device)
        ):
            self.plane_v = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        else:
            v = torch.cross(self.target_normal, self.plane_u)
            self.plane_v = v / v.norm().clamp_min(1e-9)

        # Optional: store initial action ---------------------------------------
        self.initial_action = None

    # --------------------------------------------------------------------- API

    def reset_errors(self) -> None:
        """Regenerates mirror‑orientation error tensors.

        * A dedicated error tensor of shape ``[num_heliostats, 2]`` is used for
          single‑sun rendering calls to ensure **backwards compatibility**.
        * A second tensor of shape
          ``[max_batch_size, num_heliostats, 2]`` is pre‑allocated for batched
          rendering. Errors remain **deterministic** across successive calls to
          :py:meth:`render` until :py:meth:`reset_errors` is invoked again.
        """
        # Single‑sun error tensor (legacy behaviour)
        self.error_angles_mrad = (
            torch.randn(self.num_heliostats, 2, device=self.device) * self.error_scale_mrad
        )

        # Batched error tensor (re‑used until next reset)
        if self.max_batch_size >= 1:
            self.batch_error_angles_mrad = self._sample_error_angles(self.max_batch_size)
        else:
            self.batch_error_angles_mrad = None

    # ------------------------------------------------------------------ Helpers

    def _sample_error_angles(self, batch_size: int) -> torch.Tensor:
        """Draws *batch_size* independent error tensors.

        Each sample has shape ``[num_heliostats, 2]``. The returned tensor shape
        is therefore ``[batch_size, num_heliostats, 2]``.
        """
        return (
            torch.randn(batch_size, self.num_heliostats, 2, device=self.device)
            * self.error_scale_mrad
        )

    # ----------------------------------------------------------- Optics helpers

    def calculate_ideal_normals(self, sun_position: torch.Tensor) -> torch.Tensor:
        """Per‑heliostat surface normals that perfectly hit the target."""
        sun = torch.as_tensor(sun_position, dtype=torch.float32, device=self.device)

        if sun.dim() == 1:
            incidents = sun.view(1, 3) - self.heliostat_positions  # [H, 3]
            reflected = self.target_position.view(1, 3) - self.heliostat_positions

            inc_dir = incidents / incidents.norm(dim=1, keepdim=True).clamp_min(1e-9)
            ref_dir = reflected / reflected.norm(dim=1, keepdim=True).clamp_min(1e-9)
            normals = -(inc_dir + ref_dir)
            return normals / normals.norm(dim=1, keepdim=True).clamp_min(1e-9)

        # batched input ----------------------------------------------------
        B = sun.shape[0]
        helios = self.heliostat_positions.view(1, self.num_heliostats, 3)
        incidents = sun.view(B, 1, 3) - helios
        reflected = self.target_position.view(1, 1, 3) - helios

        inc_dir = incidents / incidents.norm(dim=2, keepdim=True).clamp_min(1e-9)
        ref_dir = reflected / reflected.norm(dim=2, keepdim=True).clamp_min(1e-9)
        normals = -(inc_dir + ref_dir)
        return normals / normals.norm(dim=2, keepdim=True).clamp_min(1e-9)

    def init_actions(self, sun_position: torch.Tensor) -> None:
        """Generate noisy initial mirror orientations for optimisation / RL."""
        ideal = self.calculate_ideal_normals(sun_position)

        if ideal.dim() == 2:  # single sun
            noise = torch.randn_like(ideal) * self.initial_action_noise
            noisy = ideal + noise
            noisy = noisy / noisy.norm(dim=1, keepdim=True).clamp_min(1e-9)
            self.initial_action = noisy.flatten()
        else:
            noise = torch.randn_like(ideal) * self.initial_action_noise
            noisy = ideal + noise
            normed = noisy / noisy.norm(dim=2, keepdim=True).clamp_min(1e-9)
            self.initial_action = normed.view(ideal.shape[0], -1)

    # ------------------------------------------------------------------ Render

    def render(
        self,
        sun_position: torch.Tensor,
        action: torch.Tensor,
        show_spillage: bool = False,  # kept for API completeness (unused)
    ) -> torch.Tensor:
        """Return irradiance image(s) on the target plane.

        Args:
            sun_position: [3] *or* [B, 3] world‑space sun positions.
            action:        Flattened normal vectors (same layout as before).
            show_spillage: Placeholder flag (reserved for future use).
        Returns:
            If *sun_position* is 1‑D → *[res, res]* tensor.
            Else                       → *[B, res, res]* tensor.
        """
        sun = torch.as_tensor(sun_position, dtype=torch.float32, device=self.device)
        batched = sun.dim() > 1  # B > 1?
        if not batched:
            sun = sun.unsqueeze(0)  # B = 1

        B = sun.shape[0]

        # ---- Mirror normals --------------------------------------------------
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if act.dim() == 1:
            act = act.unsqueeze(0)
        normals = act.reshape(B, self.num_heliostats, 3)

        # ---- Retrieve (or sample) error tensors ------------------------------
        if B == 1:
            # legacy single‑sun behaviour
            errs = self.error_angles_mrad.unsqueeze(0)
        else:
            # Try to reuse a pre‑sampled batch. If the requested batch size is
            # larger than the pre‑allocated tensor, fall back to on‑the‑fly
            # sampling (deterministic only for the current call).
            if (
                self.batch_error_angles_mrad is not None
                and B <= self.batch_error_angles_mrad.shape[0]
            ):
                errs = self.batch_error_angles_mrad[:B]
            else:
                errs = self._sample_error_angles(B)

        # ----------------------------------------------------------------------
        flats = normals.reshape(-1, 3)
        errs_flat = errs.reshape(-1, 2)

        actual = rotate_normals_batch(flats, errs_flat)
        actual = actual / actual.norm(dim=1, keepdim=True).clamp_min(1e-9)
        actual = actual.view(B, self.num_heliostats, 3)

        # ---- Ray tracing ------------------------------------------------------
        helios = self.heliostat_positions.view(1, self.num_heliostats, 3).expand(B, -1, -1)
        incidents = sun.view(B, 1, 3) - helios

        inc_flat = incidents.reshape(-1, 3)
        refl_flat = reflect_vectors(inc_flat, actual.reshape(-1, 3))
        orig_flat = helios.reshape(-1, 3)

        inter_flat = ray_plane_intersection_batch(
            orig_flat, refl_flat, self.target_position, self.target_normal
        )

        gauss_flat = gaussian_blur_batch(
            inter_flat,
            orig_flat,
            self.target_position,
            self.plane_u,
            self.plane_v,
            self.target_width,
            self.target_height,
            self.resolution,
            self.sigma_scale,
        )

        res = self.resolution
        gauss = gauss_flat.view(B, self.num_heliostats, res, res)
        images = gauss.sum(dim=1)

        # ---- Normalise total energy ------------------------------------------
        sums = images.view(B, -1).sum(dim=1).view(B, 1, 1).clamp_min(1e-9)
        images = images / sums

        return images[0] if not batched else images
