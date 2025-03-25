import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def set_axes_equal(ax):
    """Make the 3D plot axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def render_gaussian_blobs(coordinates, image_size=(64, 64), sigma=2.0, amplitude=1.0):
    height, width = image_size
    ys = torch.arange(0, height, device=coordinates.device, dtype=torch.float32)
    xs = torch.arange(0, width, device=coordinates.device, dtype=torch.float32)
    ys, xs = torch.meshgrid(ys, xs, indexing="ij")
    xs = xs.unsqueeze(-1)
    ys = ys.unsqueeze(-1)
    blob_x = coordinates[:, 0].view(1, 1, -1)
    blob_y = coordinates[:, 1].view(1, 1, -1)
    dist_sq = (xs - blob_x)**2 + (ys - blob_y)**2
    blobs = amplitude * torch.exp(-dist_sq / (2 * sigma**2))
    image = blobs.sum(dim=-1)
    return image

def calculate_normals(P, M, H, device=torch.device("cpu")):
    """
    Calculate normals based on:
        P_H = (P-H)/||P-H||,
        M_H = (M-H)/||M-H||,
        N_unnormed = 0.5*(P_H + M_H),
        N = N_unnormed / ||N_unnormed||
    
    Parameters:
        P (torch.Tensor or np.ndarray): Light source position(s). Shape (3,) or (N,3).
        M (torch.Tensor or np.ndarray): Target vector(s). Shape (3,) or (N,3).
        H (torch.Tensor or np.ndarray): Hit positions. Shape (N,3) or (3,) for a single hit.
        device (torch.device): Device to perform computations on.
    
    Returns:
        torch.Tensor: Normalized normals (shape (N,3)).
    """
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float, device=device, requires_grad=True)
        elif isinstance(x, torch.Tensor):
            return x.to(device).detach().clone().requires_grad_(True)
        else:
            raise TypeError("Input must be a numpy array or a torch.Tensor.")
    
    P = to_tensor(P)
    M = to_tensor(M)
    H = to_tensor(H)
    
    if H.dim() == 1:
        H = H.unsqueeze(0)
    
    if P.dim() == 1:
        P = P.unsqueeze(0).expand(H.shape[0], -1)
    elif P.shape[0] != H.shape[0]:
        try:
            P = P.expand_as(H)
        except RuntimeError as e:
            raise ValueError("P must be either a single vector or have the same number of rows as H.") from e

    if M.dim() == 1:
        M = M.unsqueeze(0).expand(H.shape[0], -1)
    elif M.shape[0] != H.shape[0]:
        try:
            M = M.expand_as(H)
        except RuntimeError as e:
            raise ValueError("M must be either a single vector or have the same number of rows as H.") from e

    # Compute normalized differences.
    P_H = P - H
    P_H = P_H / torch.norm(P_H, dim=1, keepdim=True)
    M_H = M - H
    M_H = M_H / torch.norm(M_H, dim=1, keepdim=True)

    # Compute unnormalized normals.
    N_unnormed = 0.5 * (P_H + M_H)
    N = N_unnormed / torch.norm(N_unnormed, dim=1, keepdim=True)
    return N


def reflect_ray(P, H, n, device=torch.device("cpu"), return_numpy=False):
    """
    Compute the reflected ray direction D_r for each hit point H and corresponding normal n,
    given a light source position P. If P is a 1D tensor it is broadcast to all hit points.
    
    Parameters:
        P (torch.Tensor or np.ndarray): The position of the light source.
            Shape can be (3,) or (N,3).
        H (torch.Tensor or np.ndarray): Hit points. Shape (N,3) or (3,) for one hit point.
        n (torch.Tensor or np.ndarray): Surface normals corresponding to H. Shape (N,3) or (3,).
        device (torch.device): Device for calculations.
        return_numpy (bool): If True, returns a NumPy array.
        
    Returns:
        D_r (torch.Tensor or np.ndarray): The reflected ray directions (shape (N,3)).
    """
    # Convert inputs to torch tensors with gradients enabled.
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float, device=device, requires_grad=True)
        else:
            return x.to(device).detach().clone().requires_grad_(True)
    
    P = to_tensor(P)
    H = to_tensor(H)
    n = to_tensor(n)
    
    # Ensure H and n are batched (shape: (N, 3))
    if P.dim() == 1:
        P = P.unsqueeze(0)  # shape (1, 3)
    if H.dim() == 1:
        H = H.unsqueeze(0)  # shape (1, 3)
    if n.dim() == 1:
        n = n.unsqueeze(0)  # shape (1, 3)
    
    # If P is (1,3) and H has shape (N,3), broadcast P.
    if P.shape[0] == 1 and H.shape[0] != 1:
        P = P.expand(H.shape[0], -1)
    elif P.shape[0] != H.shape[0]:
        # Otherwise, they should match or be broadcastable.
        try:
            P = P.expand_as(H)
        except RuntimeError as e:
            raise ValueError("P must be either a single point or have the same number of rows as H.") from e

    # Normalize each normal vector.
    n_norm = torch.norm(n, dim=1, keepdim=True)
    if (n_norm == 0).any():
        raise ValueError("One of the normal vectors has zero norm!")
    n = n / n_norm

    # Compute the incident direction: D_p = (P - H) / ||P - H|| (per hit point)
    D_p = P - H
    D_p_norm = torch.norm(D_p, dim=1, keepdim=True)
    if (D_p_norm == 0).any():
        raise ValueError("One of the hit points is identical to the light source position P.")
    D_p = D_p / D_p_norm

    # Compute dot product for each hit point.
    dot_val = torch.sum(D_p * n, dim=1, keepdim=True)
    # Reflected ray direction: D_r = 2*(D_p dot n)*n - D_p
    D_r = 2 * dot_val * n - D_p

    if return_numpy:
        return D_r.detach().cpu().numpy()
    else:
        return D_r
def display(H, P, n=None, M=None, device=torch.device("cpu"), 
            target_center=None, target_width=None, target_height=None):
    """
    Display a 3D plot of rays given reflector positions H, light source P, and optionally either provided
    surface normals n or target vectors M. The plot includes:
      • The light source position.
      • The hit points (H).
      • Incoming rays from P to each H.
      • Reflected rays computed using either provided normals n or normals computed from target vectors M.
      • The normals (or computed normals) as arrows.
      • If target vectors M are provided (and n is None), also plot the target direction (i.e. normalized M-H).
      • In-plane vectors (u and w) and a surface patch for each hit point.
      • Optionally, if target_center, target_width, and target_height are provided, display the target area as a surface.
    
    Parameters:
      H (torch.Tensor or np.ndarray): Reflector positions, shape (N,3) or (3,) for a single hit.
      P (torch.Tensor or np.ndarray): Light source position, shape (3,) or (N,3); if a single point is provided,
                                        it is broadcast to all H.
      n (Optional[torch.Tensor or np.ndarray]): Provided surface normals, shape (N,3) or (3,).
      M (Optional[torch.Tensor or np.ndarray]): Target vectors, shape (N,3) or (3,). Used to compute normals
                                                  if n is not provided.
      device (torch.device): Device on which the calculations are performed.
      target_center (Optional[torch.Tensor or np.ndarray]): Global coordinates (3,) of the target area's center.
      target_width (Optional[float]): Width (in meters) of the target area.
      target_height (Optional[float]): Height (in meters) of the target area.
    
    Returns:
      None. Displays the 3D plot.
    
    Note: This function assumes that helper functions 'reflect_ray', 'calculate_normals', and 'set_axes_equal'
          have already been imported.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float, device=device, requires_grad=True)
        elif isinstance(x, torch.Tensor):
            return x.to(device).detach().clone().requires_grad_(True)
        else:
            raise TypeError("Input must be a numpy array or a torch.Tensor.")

    H = to_tensor(H)
    P = to_tensor(P)
    
    if H.dim() == 1:
        H = H.unsqueeze(0)
    if P.dim() == 1:
        P = P.unsqueeze(0).expand(H.shape[0], -1)
    elif P.shape[0] != H.shape[0]:
        try:
            P = P.expand_as(H)
        except RuntimeError as e:
            raise ValueError("P must be either a single point or have the same number of rows as H.") from e

    if n is not None:
        n = to_tensor(n)
        if n.dim() == 1:
            n = n.unsqueeze(0)
        if n.shape[0] != H.shape[0]:
            try:
                n = n.expand_as(H)
            except RuntimeError as e:
                raise ValueError("n must be either a single vector or have the same number of rows as H.") from e
        normals = n
        target_direction = None
    elif M is not None:
        M = to_tensor(M)
        if M.dim() == 1:
            M = M.unsqueeze(0)
        if M.shape[0] != H.shape[0]:
            try:
                M = M.expand_as(H)
            except RuntimeError as e:
                raise ValueError("M must be either a single vector or have the same number of rows as H.") from e
        normals = calculate_normals(P, M, H, device=device)
        target_direction = (M - H) / torch.norm(M - H, dim=1, keepdim=True)
    else:
        raise ValueError("Either n or M must be provided.")

    D_r = reflect_ray(P, H, normals, device=device, return_numpy=False)
    D_p = (P - H) / torch.norm(P - H, dim=1, keepdim=True)
    
    P_np = P.detach().cpu().numpy()[0]  # light source (assumed common)
    H_np = H.detach().cpu().numpy()
    normals_np = normals.detach().cpu().numpy()
    D_r_np = D_r.detach().cpu().numpy()
    D_p_np = D_p.detach().cpu().numpy()
    if target_direction is not None:
        target_dir_np = target_direction.detach().cpu().numpy()
    
    colors = ['blue', 'orange', 'green', 'purple', 'brown']
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    ax.scatter(P_np[0], P_np[1], P_np[2], color='red', s=100, label='P (Light Source)')
    
    for i in range(H_np.shape[0]):
        Hi = H_np[i]
        Di_p = D_p_np[i]
        Di_r = D_r_np[i]
        ni = normals_np[i]
        ax.scatter(Hi[0], Hi[1], Hi[2], color='black', s=80, label=f'H{i+1}' if i==0 else None)
        t_inc = np.linspace(0, 1, 100)
        incoming_line = np.outer(1 - t_inc, P_np) + np.outer(t_inc, Hi)
        ax.plot(incoming_line[:, 0], incoming_line[:, 1], incoming_line[:, 2],
                color='gray', linestyle='--', linewidth=1, label='Incoming Ray' if i==0 else None)
        L = 10
        t_ref = np.linspace(0, L, 100)
        reflected_line = Hi + np.outer(t_ref, Di_r)
        ax.plot(reflected_line[:, 0], reflected_line[:, 1], reflected_line[:, 2],
                color=colors[i % len(colors)], linewidth=2, label=f'Reflected Ray {i+1}')
        normal_scale = 2
        ax.quiver(Hi[0], Hi[1], Hi[2],
                  ni[0], ni[1], ni[2],
                  length=normal_scale, color='magenta', arrow_length_ratio=0.1,
                  label='Normal' if i==0 else None)
        if target_direction is not None:
            ax.quiver(Hi[0], Hi[1], Hi[2],
                      target_dir_np[i, 0], target_dir_np[i, 1], target_dir_np[i, 2],
                      length=normal_scale, color='purple', arrow_length_ratio=0.1,
                      label='Target Direction' if i==0 else None)
        if abs(ni[0]) < 0.9:
            v = np.array([1, 0, 0])
        else:
            v = np.array([0, 1, 0])
        u = v - np.dot(v, ni) * ni
        u = u / np.linalg.norm(u)
        w = np.cross(ni, u)
        vector_scale = 2
        ax.quiver(Hi[0], Hi[1], Hi[2],
                  u[0], u[1], u[2],
                  length=vector_scale, color='black', arrow_length_ratio=0.1,
                  label='u (in-plane)' if i==0 else None)
        ax.quiver(Hi[0], Hi[1], Hi[2],
                  w[0], w[1], w[2],
                  length=vector_scale, color='brown', arrow_length_ratio=0.1,
                  label='w (in-plane)' if i==0 else None)
        s_vals = np.linspace(-5, 5, 10)
        t_vals = np.linspace(-5, 5, 10)
        S, T = np.meshgrid(s_vals, t_vals)
        plane_points = Hi.reshape(3, 1, 1) + u.reshape(3, 1, 1) * S + w.reshape(3, 1, 1) * T
        X = plane_points[0]
        Y = plane_points[1]
        Z = plane_points[2]
        ax.plot_surface(X, Y, Z, alpha=0.3, color='cyan', rstride=1, cstride=1, edgecolor='none')
    
    # If target area parameters are provided, plot the target area surface.
    if (target_center is not None) and (target_width is not None) and (target_height is not None):
        tc = to_tensor(target_center).detach().cpu().numpy().flatten()  # expected shape (3,)
        x0, y0, z0 = tc  # target area center
        y_min = y0 - target_height/2.0
        y_max = y0 + target_height/2.0
        z_min = z0 - target_width/2.0
        z_max = z0 + target_width/2.0
        ys_target = np.linspace(y_min, y_max, 10)
        zs_target = np.linspace(z_min, z_max, 10)
        YT, ZT = np.meshgrid(ys_target, zs_target, indexing="ij")
        XT = np.full_like(YT, x0)
        ax.plot_surface(XT, YT, ZT, alpha=0.3, color='green', rstride=1, cstride=1, edgecolor='none', label='Target Area')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    title_str = 'Display: Reflected Rays Using Provided Normals' if n is not None else 'Display: Reflected Rays Using Normals Computed from Target Vectors'
    ax.set_title(title_str)
    ax.legend(loc='upper right')
    set_axes_equal(ax)
    plt.show()

def main():
    device = torch.device("cpu")  # Change to torch.device("cuda") if GPU is available
    num_surfaces = 3

    # ------------------------ Part 1: Randomly Sampled Normals ------------------------
    # Generate common light source position P and hit points H.
    P = torch.empty(3).uniform_(-10, 10).to(device).requires_grad_(True)
    H = torch.empty((num_surfaces, 3)).uniform_(-10, 10).to(device).requires_grad_(True)
    # Randomly sample surface normals.
    n_random = torch.empty((num_surfaces, 3)).uniform_(-1, 1).to(device).requires_grad_(True)

    # Compute reflected ray directions using the random normals.
    D_r1 = reflect_ray(P, H, n_random, device=device, return_numpy=False)
    D_p1 = (P - H) / torch.norm(P - H, dim=1, keepdim=True)

    # Convert to NumPy arrays for plotting.
    P_np = P.detach().cpu().numpy().flatten()
    H_np = H.detach().cpu().numpy()
    n_rand_np = n_random.detach().cpu().numpy()
    D_r1_np = D_r1.detach().cpu().numpy()
    D_p1_np = D_p1.detach().cpu().numpy()

    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(P_np[0], P_np[1], P_np[2], color='red', s=100, label='P (Light Source)')

    colors = ['blue', 'orange', 'green']
    for i in range(num_surfaces):
        Hi = H_np[i]
        Di_p = D_p1_np[i]
        Di_r = D_r1_np[i]
        ni = n_rand_np[i]
        # Plot hit points.
        ax1.scatter(Hi[0], Hi[1], Hi[2], color='black', s=80, label=f'H{i+1}' if i==0 else None)
        # Incoming ray from P to Hi.
        t_inc = np.linspace(0, 1, 100)
        incoming_line = np.outer(1 - t_inc, P_np) + np.outer(t_inc, Hi)
        ax1.plot(incoming_line[:, 0], incoming_line[:, 1], incoming_line[:, 2],
                 color='gray', linestyle='--', linewidth=1, label='Incoming Ray' if i==0 else None)
        # Reflected ray from Hi along Di_r.
        L = 10
        t_ref = np.linspace(0, L, 100)
        reflected_line = Hi + np.outer(t_ref, Di_r)
        ax1.plot(reflected_line[:, 0], reflected_line[:, 1], reflected_line[:, 2],
                 color=colors[i % len(colors)], linewidth=2, label=f'Reflected Ray {i+1}')
        # Plot random normal at Hi.
        normal_scale = 2
        ax1.quiver(Hi[0], Hi[1], Hi[2],
                   ni[0], ni[1], ni[2],
                   length=normal_scale, color='magenta', arrow_length_ratio=0.1, label='Random Normal' if i==0 else None)
        # Compute and plot in-plane vectors for the surface.
        if abs(ni[0]) < 0.9:
            v = np.array([1, 0, 0])
        else:
            v = np.array([0, 1, 0])
        u = v - np.dot(v, ni) * ni
        u = u / np.linalg.norm(u)
        w = np.cross(ni, u)
        vector_scale = 2
        ax1.quiver(Hi[0], Hi[1], Hi[2],
                   u[0], u[1], u[2],
                   length=vector_scale, color='black', arrow_length_ratio=0.1, label='u (in-plane)' if i==0 else None)
        ax1.quiver(Hi[0], Hi[1], Hi[2],
                   w[0], w[1], w[2],
                   length=vector_scale, color='brown', arrow_length_ratio=0.1, label='w (in-plane)' if i==0 else None)
        # Draw surface patch.
        plane_extent = 5
        s_vals = np.linspace(-plane_extent, plane_extent, 10)
        t_vals = np.linspace(-plane_extent, plane_extent, 10)
        S, T = np.meshgrid(s_vals, t_vals)
        plane_points = Hi.reshape(3, 1, 1) + u.reshape(3, 1, 1) * S + w.reshape(3, 1, 1) * T
        X = plane_points[0]
        Y = plane_points[1]
        Z = plane_points[2]
        ax1.plot_surface(X, Y, Z, alpha=0.3, color='cyan', rstride=1, cstride=1, edgecolor='none')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Part 1: Reflected Rays Using Randomly Sampled Normals')
    custom_lines1 = [
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Incoming Ray'),
        Line2D([0], [0], color='magenta', lw=2, label='Random Normal'),
        Line2D([0], [0], color='black', lw=2, label='u (in-plane)'),
        Line2D([0], [0], color='brown', lw=2, label='w (in-plane)'),
        Patch(facecolor='cyan', edgecolor='cyan', alpha=0.3, label='Surface')
    ]
    ax1.legend(handles=custom_lines1, loc='upper right')
    set_axes_equal(ax1)

    # ------------------------ Part 2: Using Randomly Sampled Targets ------------------------
    # Here we generate target vectors M, compute normals from them, and then reflect.
    M = torch.empty((num_surfaces, 3)).uniform_(-10, 10).to(device).requires_grad_(True)
    computed_normals = calculate_normals(P, M, H, device=device)
    D_r2 = reflect_ray(P, H, computed_normals, device=device, return_numpy=False)
    D_p2 = (P - H) / torch.norm(P - H, dim=1, keepdim=True)
    # Also compute the "actual" target direction (M-H normalized).
    target_dir = (M - H) / torch.norm(M - H, dim=1, keepdim=True)

    # Convert to NumPy for plotting.
    M_np = M.detach().cpu().numpy()
    computed_normals_np = computed_normals.detach().cpu().numpy()
    D_r2_np = D_r2.detach().cpu().numpy()
    D_p2_np = D_p2.detach().cpu().numpy()
    target_dir_np = target_dir.detach().cpu().numpy()

    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(P_np[0], P_np[1], P_np[2], color='red', s=100, label='P (Light Source)')

    for i in range(num_surfaces):
        Hi = H_np[i]
        Di_p = D_p2_np[i]
        Di_r = D_r2_np[i]
        ni = computed_normals_np[i]
        Mi = M_np[i]
        targ_dir = target_dir_np[i]
        # Plot hit point.
        ax2.scatter(Hi[0], Hi[1], Hi[2], color='black', s=80, label=f'H{i+1}' if i==0 else None)
        # Incoming ray.
        t_inc = np.linspace(0, 1, 100)
        incoming_line = np.outer(1 - t_inc, P_np) + np.outer(t_inc, Hi)
        ax2.plot(incoming_line[:, 0], incoming_line[:, 1], incoming_line[:, 2],
                 color='gray', linestyle='--', linewidth=1, label='Incoming Ray' if i==0 else None)
        # Reflected ray.
        L = 10
        t_ref = np.linspace(0, L, 100)
        reflected_line = Hi + np.outer(t_ref, Di_r)
        ax2.plot(reflected_line[:, 0], reflected_line[:, 1], reflected_line[:, 2],
                 color=colors[i % len(colors)], linewidth=2, label=f'Reflected Ray {i+1}')
        # Plot computed normal.
        normal_scale = 2
        ax2.quiver(Hi[0], Hi[1], Hi[2],
                   ni[0], ni[1], ni[2],
                   length=normal_scale, color='magenta', arrow_length_ratio=0.1,
                   label='Computed Normal' if i==0 else None)
        # Plot target vector (from H to M).
        ax2.quiver(Hi[0], Hi[1], Hi[2],
                   (Mi-Hi)[0], (Mi-Hi)[1], (Mi-Hi)[2],
                   length=normal_scale, color='purple', arrow_length_ratio=0.1,
                   label='Target Direction' if i==0 else None)
        # Also plot in-plane vectors (using computed normal).
        if abs(ni[0]) < 0.9:
            v = np.array([1, 0, 0])
        else:
            v = np.array([0, 1, 0])
        u = v - np.dot(v, ni) * ni
        u = u / np.linalg.norm(u)
        w = np.cross(ni, u)
        vector_scale = 2
        ax2.quiver(Hi[0], Hi[1], Hi[2],
                   u[0], u[1], u[2],
                   length=vector_scale, color='black', arrow_length_ratio=0.1,
                   label='u (in-plane)' if i==0 else None)
        ax2.quiver(Hi[0], Hi[1], Hi[2],
                   w[0], w[1], w[2],
                   length=vector_scale, color='brown', arrow_length_ratio=0.1,
                   label='w (in-plane)' if i==0 else None)
        # Draw surface patch.
        plane_extent = 5
        s_vals = np.linspace(-plane_extent, plane_extent, 10)
        t_vals = np.linspace(-plane_extent, plane_extent, 10)
        S, T = np.meshgrid(s_vals, t_vals)
        plane_points = Hi.reshape(3, 1, 1) + u.reshape(3, 1, 1) * S + w.reshape(3, 1, 1) * T
        X = plane_points[0]
        Y = plane_points[1]
        Z = plane_points[2]
        ax2.plot_surface(X, Y, Z, alpha=0.3, color='cyan', rstride=1, cstride=1, edgecolor='none')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Part 2: Reflected Rays Using Computed Normals from Target Vectors')
    custom_lines2 = [
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Incoming Ray'),
        Line2D([0], [0], color='magenta', lw=2, label='Computed Normal'),
        Line2D([0], [0], color='purple', lw=2, label='Target Direction'),
        Line2D([0], [0], color='black', lw=2, label='u (in-plane)'),
        Line2D([0], [0], color='brown', lw=2, label='w (in-plane)'),
        Patch(facecolor='cyan', edgecolor='cyan', alpha=0.3, label='Surface')
    ]
    ax2.legend(handles=custom_lines2, loc='upper right')
    set_axes_equal(ax2)

    plt.show()

if __name__ == "__main__":
    main()