import torch
import matplotlib.pyplot as plt
import math # Added for pi

# --- Helper Functions (Mostly Unchanged) ---

def reflect_vector(incident: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Reflects an incident vector about a normal vector."""
    # Ensure normal is unit vector before use
    norm_mag = torch.norm(normal)
    # Avoid division by zero if normal is zero vector (though unlikely)
    if norm_mag < 1e-9:
        # Return incident vector or raise error, depending on desired behavior
        # For reflection, a zero normal doesn't make sense, but let's avoid crashing
        return incident
    normal = normal / norm_mag
    return incident - 2 * torch.dot(incident, normal) * normal

def ray_plane_intersection(ray_origin: torch.Tensor, ray_direction: torch.Tensor,
                           plane_point: torch.Tensor, plane_normal: torch.Tensor) -> torch.Tensor:
    """Calculates intersection of a ray with a plane."""
    # Ensure normal is unit vector before use
    norm_mag = torch.norm(plane_normal)
    if norm_mag < 1e-9:
         raise ValueError("Plane normal cannot be a zero vector.")
    plane_normal = plane_normal / norm_mag

    denom = torch.dot(ray_direction, plane_normal)
    if torch.abs(denom) < 1e-6:
        # Ray is parallel or near parallel to the plane
        # Depending on context, might return None, inf, or raise error
        raise ValueError("Ray is parallel to the plane, no unique intersection.")
    t = torch.dot(plane_point - ray_origin, plane_normal) / denom

    # Check if intersection is behind the ray origin (t < 0)
    # This might be relevant depending on the application
    # if t < 0:
    #    raise ValueError("Intersection point is behind the ray origin.")

    intersection = ray_origin + t * ray_direction
    return intersection

def reflected_ray_intersects_plane(sun_position: torch.Tensor,
                                   heliostat_normal: torch.Tensor, # This normal is AFTER error rotation
                                   heliostat_position: torch.Tensor,
                                   target_point: torch.Tensor,
                                   target_normal: torch.Tensor) -> torch.Tensor:
    """
    Reflects the sun_position vector at the heliostat_normal and
    computes the intersection of the reflected ray with the target plane.
    All vectors are in East-North-Up (ENU) coordinate system.
    """
    incident = sun_position - heliostat_position
    # Ensure heliostat_normal is normalized before reflection
    heliostat_normal = heliostat_normal / torch.norm(heliostat_normal)
    reflected = reflect_vector(incident, heliostat_normal)
    intersection = ray_plane_intersection(heliostat_position, reflected, target_point, target_normal)
    return intersection

def calculate_heliostat_normals_from_sun_position(sun_position: torch.Tensor,
                                                  heliostat_positions: torch.Tensor,
                                                  target_position: torch.Tensor) -> torch.Tensor:
    """
    Calculates the ideal normal vectors for each heliostat such that sunlight
    reflects perfectly to the target position.
    """
    # Ensure inputs are tensors
    sun_position = torch.as_tensor(sun_position, dtype=torch.float32)
    heliostat_positions = torch.as_tensor(heliostat_positions, dtype=torch.float32)
    target_position = torch.as_tensor(target_position, dtype=torch.float32)

    # Ensure heliostat_positions is 2D [N, 3]
    if heliostat_positions.dim() == 1:
        heliostat_positions = heliostat_positions.unsqueeze(0)

    incident_vectors = sun_position - heliostat_positions # Vector from heliostat to sun
    reflected_vectors = target_position - heliostat_positions # Vector from heliostat to target

    # Normalize direction vectors
    incident_directions = incident_vectors / torch.norm(incident_vectors, dim=1, keepdim=True)
    reflected_directions = reflected_vectors / torch.norm(reflected_vectors, dim=1, keepdim=True)

    # The ideal normal is the angle bisector of the incident and reflected directions
    # It points "outwards" from the mirror surface
    mirror_normals = -(incident_directions + reflected_directions) # Note the negative sign convention might differ
    # Renormalize the resulting normal vectors
    mirror_normals = mirror_normals / torch.norm(mirror_normals, dim=1, keepdim=True)

    return mirror_normals

def intersection_outside_target_area(intersection_point: torch.Tensor, 
                                            plane_origin: torch.Tensor, 
                                            plane_u: torch.Tensor, 
                                            plane_v: torch.Tensor, 
                                            width: float = 15.0, 
                                            height: float = 15.0):
    # project interseection on plane_u and plane_v 
    project_u = ((intersection_point - plane_origin) @ plane_u / torch.square(torch.linalg.norm(plane_u)))
    project_v = ((intersection_point - plane_origin) @ plane_v / torch.square(torch.linalg.norm(plane_u))) 

    return (torch.abs(project_u) > width/2) or (torch.abs(project_v) > height/2)

def gaussian_blur_on_plane_dynamic_sigma(intersection_point: torch.Tensor,
                                         heliostat_position: torch.Tensor,
                                         plane_origin: torch.Tensor,
                                         plane_u: torch.Tensor, # East vector [1, 0, 0]
                                         plane_v: torch.Tensor, # Up vector [0, 0, 1]
                                         width: float = 15.0,
                                         height: float = 15.0,
                                         resolution: int = 100,
                                         sigma_scale: float = 0.01) -> torch.Tensor:
    """
    Creates a 2D Gaussian blur centered at the intersection_point projected onto
    the plane defined by plane_origin, plane_u, and plane_v.
    Sigma is scaled by the distance from the heliostat to the intersection point.
    """
    device = intersection_point.device
    distance = torch.norm(intersection_point - heliostat_position)
    sigma = sigma_scale * distance

    # Create grid points on the target plane in 3D space
    x = torch.linspace(-width / 2, width / 2, resolution, device=device)
    y = torch.linspace(-height / 2, height / 2, resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij') # Use 'ij' for matrix indexing consistency

    # plane_points[i, j, :] = plane_origin + grid_x[i, j] * plane_u + grid_y[i, j] * plane_v
    plane_points = plane_origin.view(1, 1, 3) + grid_x.unsqueeze(-1) * plane_u.view(1, 1, 3) + grid_y.unsqueeze(-1) * plane_v.view(1, 1, 3)

    # Calculate squared Euclidean distance from intersection point to each grid point
    diff = plane_points - intersection_point.view(1, 1, 3)
    dist_squared = torch.sum(diff ** 2, dim=-1)

    # Calculate Gaussian values
    # Avoid division by zero if sigma is extremely small
    if sigma < 1e-9:
        gaussian = torch.zeros_like(dist_squared)
        # Find the closest grid point and set it to 1 (or handle differently)
        min_dist_idx = torch.argmin(dist_squared)
        gaussian.view(-1)[min_dist_idx] = 1.0 # Approximate a Dirac delta
    else:
         # Gaussian formula: exp(-distance^2 / (2 * sigma^2))
         gaussian = torch.exp(-dist_squared / (2 * sigma ** 2))

    # Normalize the Gaussian intensity (optional, depends on desired interpretation)
    # sum_gauss = torch.sum(gaussian)
    # if sum_gauss > 1e-9:
    #     gaussian = gaussian / sum_gauss
    # else:
    #      # Handle case where gaussian is zero everywhere (e.g., intersection far off-plane)
    #      gaussian = torch.zeros_like(gaussian)

    return gaussian


# --- NEW: Rotation Function ---
def rotate_normal_by_enu_angles(normal: torch.Tensor, error_angles_mrad: torch.Tensor) -> torch.Tensor:
    """
    Rotates a normal vector around the East (X) axis and then the Up (Z) axis.
    Args:
        normal (torch.Tensor): The [3,] normal vector to rotate.
        error_angles_mrad (torch.Tensor): The [2,] tensor containing rotation angles
                                           [angle_E_mrad, angle_U_mrad] in milliradians.
    Returns:
        torch.Tensor: The rotated [3,] normal vector.
    """
    angle_e_rad = error_angles_mrad[0] * 0.001
    angle_u_rad = error_angles_mrad[1] * 0.001
    device = normal.device

    # Rotation matrix around East (X-axis)
    cos_e = torch.cos(angle_e_rad)
    sin_e = torch.sin(angle_e_rad)
    rot_e = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, cos_e, -sin_e],
        [0.0, sin_e, cos_e]
    ], dtype=normal.dtype, device=device)

    # Rotation matrix around Up (Z-axis)
    cos_u = torch.cos(angle_u_rad)
    sin_u = torch.sin(angle_u_rad)
    rot_u = torch.tensor([
        [cos_u, -sin_u, 0.0],
        [sin_u, cos_u, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=normal.dtype, device=device)

    # Apply rotations: First around E, then around U. Matrix multiplication order is R_u @ R_e @ normal
    # Or: rotate around U first, then E: R_e @ R_u @ normal. Let's choose U then E.
    # rotated_normal = rot_e @ rot_u @ normal.unsqueeze(-1) # Reshape normal to [3, 1] for matmul
    # return rotated_normal.squeeze(-1) # Reshape back to [3,]

    # Alternative: Apply U rotation, then E rotation to the result
    rotated_u = rot_u @ normal.unsqueeze(-1)
    rotated_final = rot_e @ rotated_u
    return rotated_final.squeeze(-1)


class HelioField:
    def __init__(self, heliostat_positions: torch.Tensor,
                 target_position: torch.Tensor,
                 target_area: tuple,
                 target_normal: torch.Tensor,
                 error_scale_mrad: float = 1.0, # Renamed for clarity: Std Dev of error angles in mrad
                 sigma_scale: float = 0.01,     # Renamed gaussian blur sigma_scale for clarity
                 initial_action_noise: float = 0.01, # Renamed for clarity
                 resolution: int = 100,
                 device: torch.device = torch.device("cpu")): # Added device parameter

        self.device = device
        self.heliostat_positions = torch.as_tensor(heliostat_positions, dtype=torch.float32, device=self.device)
        self.num_heliostats = self.heliostat_positions.shape[0]
        self.target_position = torch.as_tensor(target_position, dtype=torch.float32, device=self.device)
        self.target_width, self.target_height = target_area
        self.target_normal = torch.as_tensor(target_normal, dtype=torch.float32, device=self.device)
        self.target_normal = self.target_normal / torch.norm(self.target_normal)

        self.error_scale_mrad = error_scale_mrad # Std Dev of error angles in mrad
        self.initial_action_noise = initial_action_noise
        self.sigma_scale = sigma_scale # Scale factor for Gaussian blur width
        self.resolution = resolution

        # --- Modified: Sample error angles ---
        # Sample initial error angles (in mrad) for each heliostat [N, 2] -> [angle_E, angle_U]
        self.error_angles_mrad = (torch.randn(self.num_heliostats, 2, device=self.device)
                                  * self.error_scale_mrad)

        # Define target plane basis vectors (assuming ENU frame)
        # Ensure they are on the correct device
        self.plane_u = torch.tensor([1.0, 0.0, 0.0], device=self.device) # East
        # Make plane_v orthogonal to target_normal and plane_u
        # If target_normal is [0, 1, 0] (North), plane_v should be [0, 0, 1] (Up)
        # If target_normal is arbitrary, calculate v = normal x u
        if torch.allclose(self.target_normal, torch.tensor([0., 1., 0.], device=self.device)):
             self.plane_v = torch.tensor([0.0, 0.0, 1.0], device=self.device) # Up
        else:
             # General case: Find v orthogonal to normal and u
             self.plane_v = torch.cross(self.target_normal, self.plane_u)
             self.plane_v = self.plane_v / torch.norm(self.plane_v) # Normalize

        self.initial_action = None # Initialize to None


    # --- NEW: Reset Errors Function ---
    def reset_errors(self):
        """Resamples the error angles for all heliostats."""
        self.error_angles_mrad = (torch.randn(self.num_heliostats, 2, device=self.device)
                                  * self.error_scale_mrad)
        print(f"Heliostat errors reset. New sample drawn with std dev {self.error_scale_mrad} mrad.")


    def calculate_ideal_normals(self, sun_position: torch.Tensor) -> torch.Tensor:
         """Calculates the ideal normals based on current sun position."""
         sun_position_dev = torch.as_tensor(sun_position, dtype=torch.float32, device=self.device)
         return calculate_heliostat_normals_from_sun_position(
             sun_position_dev, self.heliostat_positions, self.target_position
         )

    def init_actions(self, sun_position: torch.Tensor):
        """Initializes the action based on ideal normals plus some noise."""
        ideal_normals = self.calculate_ideal_normals(sun_position)
        # Add initial noise to the ideal normals
        noise = (torch.randn(self.num_heliostats, 3, device=self.device) * self.initial_action_noise)
        # Normalize after adding noise
        noisy_normals = ideal_normals + noise
        noisy_normals = noisy_normals / torch.norm(noisy_normals, dim=1, keepdim=True)
        self.initial_action = noisy_normals.flatten()
        print("Initial action set based on ideal normals plus noise.")


    # --- Modified: Render function ---
    def render(self, sun_position: torch.Tensor, action: torch.Tensor, show_spillage : bool = False) -> torch.Tensor:
        """
        Renders the combined Gaussian blur image from all heliostats.
        Applies sampled rotational errors to the normals provided by 'action'.

        Args:
            sun_position (torch.Tensor): The [3,] position of the sun.
            action (torch.Tensor): The [N*3,] flat tensor representing the intended
                                   normal vector for each heliostat.

        Returns:
            torch.Tensor: A 2D torch tensor of shape (resolution, resolution)
                          representing the heatmap on the target plane.
        """
        sun_position_dev = torch.as_tensor(sun_position, dtype=torch.float32, device=self.device)
        action_dev = torch.as_tensor(action, dtype=torch.float32, device=self.device)

        image = torch.zeros((self.resolution, self.resolution), device=self.device)

        # Reshape action provided by the agent/optimizer into normals [N, 3]
        # These are the *intended* normals before physical errors are applied.
        intended_normals = action_dev.view(self.num_heliostats, 3)

        spillage_count = 0

        for i in range(self.num_heliostats):
            heliostat_pos = self.heliostat_positions[i]
            base_normal = intended_normals[i] # Normal from the action
            error_angle = self.error_angles_mrad[i] # Get the persistent error for this heliostat

            # --- Apply rotational error ---
            # Rotate the *intended* normal by the sampled error angles
            actual_normal = rotate_normal_by_enu_angles(base_normal, error_angle)
            # Ensure the final normal used for reflection is a unit vector
            actual_normal = actual_normal / torch.norm(actual_normal)

            try:
                # Use the *actual_normal* (with error) for reflection calculation
                intersection = reflected_ray_intersects_plane(
                    sun_position_dev, actual_normal, heliostat_pos,
                    self.target_position, self.target_normal
                )

                # Project the intersection point onto the target plane
                gaussian = gaussian_blur_on_plane_dynamic_sigma(
                    intersection_point=intersection,
                    heliostat_position=heliostat_pos,
                    plane_origin=self.target_position,
                    plane_u=self.plane_u,
                    plane_v=self.plane_v,
                    width=self.target_width,
                    height=self.target_height,
                    resolution=self.resolution,
                    sigma_scale=self.sigma_scale # Use the blur sigma scale
                )

                #Calculate the number of reflection centers outside the reciever area 
                if intersection_outside_target_area(
                    intersection_point=intersection, 
                    plane_origin=self.target_position, 
                    plane_u=self.plane_u, 
                    plane_v=self.plane_v,
                    width=self.target_width, 
                    height=self.target_height
                ):
                    spillage_count = spillage_count + 1

                image = image + gaussian

            except ValueError as e:
                # print(f"Skipping heliostat {i} due to calculation error: {e}") # Optional warning
                continue

        # Normalize total intensity across the image (optional, depends on use case)
        total_intensity = torch.sum(image)
        if total_intensity > 1e-9:
            image = image / total_intensity
        else:
             # Handle case where image is all zeros (e.g., all rays missed)
             print("Warning: Rendered image has zero total intensity.")

        if (spillage_count > 0) and show_spillage:
            print("Spillage count: ", spillage_count)

        return image

# --- Example Usage ---
if __name__ == '__main__':
    # Use GPU if available, otherwise CPU
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("Using GPU")
    else:
        dev = torch.device("cpu")
        print("Using CPU")

    torch.manual_seed(42)


    # --- Simulation Parameters ---
    sun_pos = torch.tensor([0.0, 1000.0, 1000.0], device=dev) # Sun far away in North-Up direction
    sun_pos_1 = torch.tensor([0.0, 1020.0, 1000.0], device=dev) # Sun far away in North-Up direction
    target_pos = torch.tensor([0.0, -5.0, 0.0])   # Target center 10m Up on tower base
    target_norm = torch.tensor([0.0, 1.0, 0.0])   # Target facing down (horizontal plane)

    target_dims = (15.0, 15.0) # Target width (E-W) and height (N-S) in meters
    resolution_pix = 128       # Resolution of the target image

    # Simple heliostat field layout (e.g., 5x5 grid south of the tower)
    '''
    num_x = 2
    num_y = 2
    spacing = 10.0 # meters
    helio_x = torch.linspace(-spacing * (num_x - 1) / 2, spacing * (num_x - 1) / 2, num_x, device=dev)
    helio_y = torch.linspace(-50.0 - spacing * (num_y - 1), -50.0, num_y, device=dev) # Place field 50m South
    grid_x, grid_y = torch.meshgrid(helio_x, helio_y, indexing='ij')
    helio_z = torch.zeros_like(grid_x) # Assume flat ground at z=0
    helio_positions = torch.stack((grid_x.flatten(), grid_y.flatten(), helio_z.flatten()), dim=1)
    '''
    helio_positions = torch.rand(size = [5, 3]) * 20
    helio_positions [:, -1] = helio_positions [:, -1] * 0
    helio_positions [:, 0] = helio_positions [:, 0] - 10
    helio_positions [:, 1] = helio_positions [:, 0] + 10

    # --- Instantiate HelioField ---
    field = HelioField(
        heliostat_positions=helio_positions,
        target_position=target_pos,
        target_area=target_dims,
        target_normal=target_norm,
        error_scale_mrad=80.0,        # Std Dev of angular error = 1.5 mrad
        sigma_scale=0.1,           # Controls Gaussian blur size (adjust based on sun size/mirror quality)
        initial_action_noise=0.00,   # Initial random noise added to ideal normals for action
        resolution=resolution_pix,
        device=dev
    )

    # --- Initialize Action (e.g., before starting optimization) ---
    field.init_actions(sun_pos)
    current_action = field.initial_action.clone() # Start with the initial action

    # --- Render with Initial Errors ---
    print("\nRendering with initial errors...")
    image_initial = field.render(sun_pos, current_action)

    # --- Simulate Resetting Errors (e.g., during training) ---
    #print("\nResetting errors...")
    #field.reset_errors()

    # --- Render Again with New Errors ---
    # In a real scenario, the 'action' might have been updated by an RL agent here
    print("\nRendering with new sun position (using the same action)...")
    image_reset = field.render(sun_pos_1, current_action)


    # --- Visualization ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Initial Render
    im1 = axs[0].imshow(image_initial.cpu().numpy(), cmap='hot',
                       extent=[-target_dims[0]/2, target_dims[0]/2, -target_dims[1]/2, target_dims[1]/2],
                       origin='lower', interpolation='nearest')
    axs[0].set_title("Rendered Heatmap (Initial Errors)")
    axs[0].set_xlabel("East (m)")
    axs[0].set_ylabel("North (m)") # Assuming target plane U=[1,0,0] E, V=[0,1,0] N if normal is Z
    fig.colorbar(im1, ax=axs[0], label='Normalized Intensity')

    # Plot Render After Reset
    im2 = axs[1].imshow(image_reset.cpu().numpy(), cmap='hot',
                       extent=[-target_dims[0]/2, target_dims[0]/2, -target_dims[1]/2, target_dims[1]/2],
                       origin='lower', interpolation='nearest')
    axs[1].set_title("Rendered Heatmap (After Error Reset)")
    axs[1].set_xlabel("East (m)")
    axs[1].set_ylabel("North (m)") # Adjust label based on plane_v
    fig.colorbar(im2, ax=axs[1], label='Normalized Intensity')

    plt.tight_layout()
    plt.show()

    # --- Example: Accessing errors (optional) ---
    # print("\nSampled error angles (mrad) for first 5 heliostats (initial):")
    # print(field.error_angles_mrad[:5])