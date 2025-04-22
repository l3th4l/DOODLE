import torch
import matplotlib.pyplot as plt

#TODO: 
# 1. Currently, our errors are sampled from a box distribution, which is not very interpretable 
# Ideally, we want to sample error angles (in mrads) along the axis of rotations of the heliostats at the beginning
# and then rotate our normals by those error angles while calling the render function. Create a seperate function which 
# takes in a [3,] dim normal and a [2,] dim error angles, and then rotates the normal along the E and U axis 

# 2. Currently we only train a single error sample, while we want to re-sample our errors at certain frequencies.
# So we need to create a reset function which then re-samples our errors  

def reflect_vector(incident: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    """Reflects an incident vector about a normal vector."""
    normal = normal / torch.norm(normal)
    return incident - 2 * torch.dot(incident, normal) * normal

def ray_plane_intersection(ray_origin: torch.Tensor, ray_direction: torch.Tensor,
                           plane_point: torch.Tensor, plane_normal: torch.Tensor) -> torch.Tensor:
    """Calculates intersection of a ray with a plane."""
    plane_normal = plane_normal / torch.norm(plane_normal)
    denom = torch.dot(ray_direction, plane_normal)
    if torch.abs(denom) < 1e-6:
        raise ValueError("Ray is parallel to the plane, no intersection.")
    t = torch.dot(plane_point - ray_origin, plane_normal) / denom
    intersection = ray_origin + t * ray_direction
    return intersection

def reflected_ray_intersects_plane(sun_position: torch.Tensor,
                                   heliostat_normal: torch.Tensor,
                                   heliostat_position: torch.Tensor,
                                   target_point: torch.Tensor,
                                   target_normal: torch.Tensor) -> torch.Tensor:
    """
    Reflects the sun_position vector at the heliostat_normal and
    computes the intersection of the reflected ray with the target plane.
    All vectors are in East-North-Up (ENU) coordinate system.
    """
    incident = sun_position - heliostat_position
    reflected = reflect_vector(incident, heliostat_normal)
    intersection = ray_plane_intersection(heliostat_position, reflected, target_point, target_normal)
    return intersection

def calculate_heliostat_normals_from_sun_position(sun_position: torch.Tensor,
                                                   heliostat_positions: torch.Tensor,
                                                   target_position: torch.Tensor) -> torch.Tensor:
    """
    Calculates the normal vectors for each heliostat such that sunlight reflects to the target position.
    """
    incident_vectors = sun_position - heliostat_positions
    reflected_vectors = target_position - heliostat_positions
    incident_directions = incident_vectors / torch.norm(incident_vectors, dim=1, keepdim=True)
    reflected_directions = reflected_vectors / torch.norm(reflected_vectors, dim=1, keepdim=True)

    mirror_normals = incident_directions + reflected_directions
    mirror_normals = mirror_normals / torch.norm(mirror_normals, dim=1, keepdim=True)
    return mirror_normals

def gaussian_blur_on_plane_dynamic_sigma(intersection_point: torch.Tensor,
                                         heliostat_position: torch.Tensor,
                                         plane_origin: torch.Tensor,
                                         plane_u: torch.Tensor,
                                         plane_v: torch.Tensor,
                                         width: float = 15.0,
                                         height: float = 15.0,
                                         resolution: int = 100,
                                         sigma_scale: float = 0.01) -> torch.Tensor:
    """
    Creates a 2D Gaussian blur on a plane in ENU coordinates.
    """
    distance = torch.norm(intersection_point - heliostat_position)
    sigma = sigma_scale * distance

    x = torch.linspace(-width / 2, width / 2, resolution, device=intersection_point.device)
    y = torch.linspace(-height / 2, height / 2, resolution, device=intersection_point.device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    plane_points = plane_origin.view(1, 1, 3) + grid_x.unsqueeze(-1) * plane_u + grid_y.unsqueeze(-1) * plane_v
    diff = plane_points - intersection_point.view(1, 1, 3)
    dist_squared = torch.sum(diff ** 2, dim=-1)

    gaussian = torch.exp(-dist_squared / (2 * sigma ** 2))
    gaussian = gaussian / torch.sum(gaussian)  
    return gaussian


class HelioField:
    def __init__(self, heliostat_positions: torch.Tensor,
                 target_position: torch.Tensor,
                 target_area: tuple,
                 target_normal: torch.Tensor,
                 error_scale: float = 0.0, 
                 sigma_scale: float = 0.1,
                 initial_error: float = 0.1, 
                 resolution: int = 100):
        self.heliostat_positions = heliostat_positions
        self.num_heliostats = self.heliostat_positions.shape[0]
        self.target_position = target_position
        self.target_width, self.target_height = target_area
        self.target_normal = target_normal / torch.norm(target_normal)
        self.error_scale = error_scale
    
        self.error_vectors = torch.randn(size = [self.num_heliostats, 3], dtype=torch.float32) * self.error_scale
        
        self.initial_action = None
        self.initial_error = initial_error

        self.sigma_scale = sigma_scale
        self.resolution = resolution

    def init_actions(self, sun_position: torch.Tensor):

        self.initial_action = (calculate_heliostat_normals_from_sun_position(
            sun_position, self.heliostat_positions, self.target_position
        ) + torch.randn(size = [self.num_heliostats, 3], dtype=torch.float32) * self.initial_error).flatten() 

#TODO: remove heliostat_normals from the inputs in the render function, instead calculate it from the sun position, 
# the center of the target plane, and the heliostat positions. make sure that this is in a seperate function called 
# calculate_heliostat_normals_from_sun_positions
    def render(self, sun_position: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Renders the combined Gaussian blur image from all heliostats.
        Returns a 2D torch tensor of shape (resolution, resolution).
        """
        device = sun_position.device
        image = torch.zeros((self.resolution, self.resolution), device=device)

        plane_u = torch.tensor([1.0, 0.0, 0.0], device=device)  # East
        plane_v = torch.tensor([0.0, 0.0, 1.0], device=device)  # Up

        '''
        heliostat_normals = calculate_heliostat_normals_from_sun_position(
            sun_position, self.heliostat_positions, self.target_position
        )
        '''
        #Reshape action
        heliostat_normals = torch.reshape(action, [self.num_heliostats, 3])

        for i in range(self.num_heliostats):
            heliostat_pos = self.heliostat_positions[i]
            normal = heliostat_normals[i]

            #TODO : need to change this to rotation by error angles (mrads) on the axes 
            normal = normal + self.error_vectors[i]
            normal = normal / torch.norm(normal)

            try:
                intersection = reflected_ray_intersects_plane(
                    sun_position, normal, heliostat_pos,
                    self.target_position, self.target_normal
                )

                gaussian = gaussian_blur_on_plane_dynamic_sigma(
                    intersection_point=intersection,
                    heliostat_position=heliostat_pos,
                    plane_origin=self.target_position,
                    plane_u=plane_u,
                    plane_v=plane_v,
                    width=self.target_width,
                    height=self.target_height,
                    resolution=self.resolution,
                    sigma_scale=self.sigma_scale
                )

                image = image + gaussian 

            except ValueError:
                continue

        image = image / torch.sum(image)  # Normalize total intensity
        return image

'''
if __name__ == '__main__':
    sun_position = torch.tensor([0.0, 0.0, 10.0])
    heliostat_positions = torch.tensor([
        [0.0, 0.0, 0.0],
    ])
    heliostat_normals = torch.tensor([
        [0.0, -1.0, 1.0],
    ])
    target_position = torch.tensor([0.0, -5.0, 0.0])
    target_normal = torch.tensor([0.0, 1.0, 0.0])

    field = HelioField(
        heliostat_positions=heliostat_positions,
        target_position=target_position,
        target_area=(15.0, 15.0),
        target_normal=target_normal,
        error_scale=0.05
    )

    image = field.render(sun_position, heliostat_normals)

    plt.imshow(image.cpu().numpy().T, cmap='hot', extent=[-7.5, 7.5, -7.5, 7.5], origin='lower')
    plt.title("Rendered Heatmap on Target Plane")
    plt.xlabel("East (m)")
    plt.ylabel("Up (m)")
    plt.colorbar(label='Intensity')
    plt.show()
'''