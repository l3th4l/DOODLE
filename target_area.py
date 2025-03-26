import torch
import numpy as np
from utils import reflect_ray, display, calculate_normals
from matplotlib import pyplot as plt

# Assume the following have already been imported from utils:
# from utils import reflect_ray, render_gaussian_blobs

class TargetArea:
    def __init__(self, height, width):
        """
        Create a target area whose center is fixed at (0, height, 0) (all units in meters).
        Parameters:
            height (float): Determines both the vertical extent of the target area and its center y-coordinate.
            width (float): The horizontal width (in meters) of the target area.
        """
        # The center is (0, height, 0)
        self.center = torch.tensor([0.0, float(height), 0.0], dtype=torch.float32)
        self.width = float(width)
        self.height = float(height)
    
    def global_to_relative(self, coords):
        """
        Given a set of 3D coordinates (assumed to lie on the target plane in the form [0, y, x]),
        compute the position in the target area relative to its bottom‐left corner.
        
        We define the target area’s bottom‐left corner (in meters) as:
            ( -width/2 , center_y - height/2 )
        where center_y is the y component of the target area center.
        
        Parameters:
            coords (torch.Tensor or np.ndarray): Coordinates with shape (N, 3) or (3,). 
                The expected ordering is (x, y, z) but with x always 0.
                
        Returns:
            torch.Tensor: A tensor of shape (N, 2) containing the relative (x, y) positions.
        """
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, dtype=torch.float32)
        # Assume coords are in the format (0, Y, X); then:
        # Relative x = (global X - ( -width/2)) = X + width/2.
        # Relative y = (global Y - (center_y - height/2))
        # Since center_y is height (from our __init__), this becomes: Y - (height - height/2) = Y - height/2.
        rel_x = coords[..., 2] + self.width / 2.0
        rel_y = coords[..., 1] - self.height / 2.0
        return torch.stack([rel_x, rel_y], dim=-1)

    def global_to_gaussian_blobs(self, coords, image_size=(64, 64), sigma=0.35, amplitude=1.0):
        relative_coords = self.global_to_relative(coords)

        x_step = self.width / image_size[0]
        y_step = self.height / image_size[1]
            
        ys = torch.arange(0, self.height, step = y_step, device=coords.device, dtype=torch.float32)
        xs = torch.arange(0, self.width, step = x_step, device=coords.device, dtype=torch.float32)
        
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")
        xs = xs.unsqueeze(-1)
        ys = ys.unsqueeze(-1)
        blob_x = relative_coords[:, 0].view(1, 1, -1)
        blob_y = relative_coords[:, 1].view(1, 1, -1)
        dist_sq = (xs - blob_x)**2 + (ys - blob_y)**2
        blobs = amplitude * torch.exp(-dist_sq / (2 * sigma**2))
        image = blobs.sum(dim=-1)
        return image


def calculate_target_coordinates(H, D_r):
    """
    Given reflector positions H (shape (N,3)) and reflection directions D_r (shape (N,3)),
    solve for the target coordinates M such that for each reflector:
    
         c * D_r + H = [0, m1, m2]^T,
    
    where the constant is computed as:
         c = -H[0] / D_r[0]
    
    Then,
         [m1, m2] = c * [D_r[1], D_r[2]] + [H[1], H[2]].
    
    Parameters:
        H (torch.Tensor): Reflector positions of shape (N,3).
        D_r (torch.Tensor): Reflection directions of shape (N,3).
    
    Returns:
        torch.Tensor: A tensor of shape (N,2) containing the computed target positions [m1, m2] 
                      in the target plane.
    """
    # Compute constant c for each reflector.
    c = - H[:, 0] / D_r[:, 0]  # shape (N,)
    m1 = c * D_r[:, 1] + H[:, 1]
    m2 = c * D_r[:, 2] + H[:, 2]
    M = torch.stack([torch.zeros_like(m1), m1, m2], dim=1)
    return M


def main():
    device = torch.device("cpu")  # or torch.device("cuda") if available

    # 1. Randomly sample a light source (sun) position.
    P = torch.empty(3).uniform_(10, 80).to(device).requires_grad_(True)

    # 2. Randomly sample reflector (heliostat) positions.
    H = torch.empty((3, 3)).uniform_(10, 50).to(device).requires_grad_(False)
    H[:, 1] *= 0  # set the second dimension (Y coordinate) to 0 for all reflectors

    # 3. Create a TargetArea with a sensible center, height, and width.
    #    (Center is fixed at (0, height, 0) in meters.)
    target_area = TargetArea(height=15.0, width=10.0)

    # ----- Task 2: Using Randomly Sampled Target Points -----
    # Randomly sample target points around the target area center.
    target_center = target_area.center  # [0, height, 0]
    spread = 4.0  # meters
    M2_y = target_center[1] + torch.empty(3).uniform_(-spread, spread).to(device)
    M2_x = torch.empty(3).uniform_(-spread, spread).to(device)
    # Global target points are in the form [0, m1, m2] (x=0).
    M2 = torch.stack([torch.zeros_like(M2_x), M2_y, M2_x], dim=1)

    # Compute normals using the target points as M.
    computed_normals = calculate_normals(P, M2, H, device=device)
    # Compute reflection directions using these computed normals.
    D_r2 = reflect_ray(P, H, computed_normals, device=device, return_numpy=False)
    # Calculate new target coordinates from these reflection directions.
    M2_computed = calculate_target_coordinates(H, D_r2)
    # Render heatmap using target_area.global_to_gaussian_blobs.
    heatmap2 = target_area.global_to_gaussian_blobs(M2_computed, image_size=(40, 60), sigma=1.5, amplitude=1.0)

    # Use the display function to show the 3D plot with the heliostat positions and target area.
    # Pass the target points (M2) so that normals are computed from them.
    display(H, P, M=M2, device=device,
            target_center=target_area.center,
            target_width=target_area.width,
            target_height=target_area.height)

    # Display the rendered heatmap.
    plt.figure(figsize=(6,6))
    plt.imshow(heatmap2.detach().cpu().numpy(), cmap="hot")
    plt.title("Heatmap (Task 2: Computed Normals from Target Points)")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()