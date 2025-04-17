import torch
import torch.nn.functional as F

def chamfer_loss(pred_image, target_image, threshold=0.1):
    # Binarize images using threshold
    pred_bin = (pred_image > threshold).float()
    target_bin = (target_image > threshold).float()

    # Compute distance transforms
    def distance_transform(img_bin):
        # Create an inverse mask
        inv_bin = 1 - img_bin
        # Compute distance transform approximation using conv2d
        kernel_size = 15  # Adjust this depending on image size/resolution
        padding = kernel_size // 2

        # Create a distance kernel
        coords = torch.stack(torch.meshgrid(
            torch.arange(kernel_size),
            torch.arange(kernel_size),
            indexing='ij'), dim=-1).float() - padding

        dist_kernel = coords.norm(dim=-1)
        dist_kernel = dist_kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Move kernel to device of the image
        dist_kernel = dist_kernel.to(img_bin.device)

        # Convolve inverse binary mask with distance kernel
        distance_map = F.conv2d(inv_bin.unsqueeze(0).unsqueeze(0), dist_kernel, padding=padding)
        return distance_map.squeeze()

    dt_pred = distance_transform(pred_bin)
    dt_target = distance_transform(target_bin)

    # Symmetric Chamfer Distance
    chamfer_pred_to_target = (pred_bin * dt_target).sum()
    chamfer_target_to_pred = (target_bin * dt_pred).sum()

    chamfer_dist = chamfer_pred_to_target + chamfer_target_to_pred
    return chamfer_dist
