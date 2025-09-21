import torch
import torch.nn as nn

class CenterOfMass2D(nn.Module):
    """
    Differentiable Center of Mass for batches of single-channel images.

    Input:
        x: Tensor of shape (B, H, W) or (B, 1, H, W), values in [0, 1] (or any nonnegative).
           Pixel intensities are treated as mass. For strict black/white, pass 0/1.

    Output:
        coords: Tensor of shape (B, 2) where coords[b] = (x_com, y_com).
                Origin is the top-left corner. x increases to the right (columns),
                y increases downward (rows). If an image has zero mass, returns (-1, -1).
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps  # avoid division-by-zero while keeping gradients stable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, 1, H, W) or (B, H, W)
        if x.dim() == 4:
            if x.size(1) != 1:
                raise ValueError("Expected single-channel images with shape (B, 1, H, W).")
            x = x[:, 0, ...]  # (B, H, W)
        elif x.dim() != 3:
            raise ValueError("Expected input shape (B, H, W) or (B, 1, H, W).")

        B, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # Ensure non-negative "mass" (keeps gradients intact for positive values)
        w = x.clamp_min(0)

        # Create coordinate grids (origin at top-left: y=row index, x=col index)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )  # both (H, W)

        # Weighted sums over H and W
        w_sum   = w.sum(dim=(1, 2))                         # (B,)
        x_wsum  = (w * xx).sum(dim=(1, 2))                  # (B,)
        y_wsum  = (w * yy).sum(dim=(1, 2))                  # (B,)

        # Compute CoM; add eps to keep it differentiable where mass is tiny
        x_com = x_wsum / (w_sum + self.eps)
        y_com = y_wsum / (w_sum + self.eps)

        coords = torch.stack([x_com, y_com], dim=-1)        # (B, 2)

        # Replace no-mass cases with (-1, -1) without breaking grads for others
        no_mass = (w_sum <= 0)
        if no_mass.any():
            coords[no_mass] = torch.tensor([-1.0, -1.0], device=device, dtype=dtype)

        return coords

if __name__ == '__main__':
    B, H, W = 4, 64, 64
    imgs = torch.zeros(B, 1, H, W)      # no grad yet

    # populate pixels freely
    imgs[0, 0, 20, 30] = 1.0
    imgs[1, 0, 10:20, 40:50] = 0.5
    # imgs[2] stays all zeros
    imgs[3, 0].uniform_(0, 1)

    # now turn on autograd tracking
    imgs.requires_grad_()

    com = CenterOfMass2D()
    coords = com(imgs)  # (B, 2), each (x, y) with origin at top-left
    print(coords)
