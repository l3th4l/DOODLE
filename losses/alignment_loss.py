import torch
from math import pi

#Alignment loss 
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
    return angles_rad * 1000


# ---------------- sanity check ----------------
def sanity_check():
    # 1. identical vectors -> expect 0 mrad (minimum attainable value)
    v = torch.tensor([1.0, 0.0, 0.0])
    res_same = calculate_angles_mrad(v, v)
    print(f"Identical vectors (v vs v): {res_same.item():.6f} mrad")

    # 2. perpendicular vectors -> expect π/2 rad = 1.5708 rad ≈ 1570.8 mrad
    v_perp1 = torch.tensor([1.0, 0.0, 0.0])
    v_perp2 = torch.tensor([0.0, 1.0, 0.0])
    res_perp = calculate_angles_mrad(v_perp1, v_perp2)
    print(f"Perpendicular vectors:      {res_perp.item():.6f} mrad (expected ≈ {pi/2*1000:.4f})")

    # 3. opposite vectors -> expect π rad = 3.1416 rad ≈ 3141.6 mrad
    v_neg = -v
    res_opp = calculate_angles_mrad(v, v_neg)
    print(f"Opposite vectors:           {res_opp.item():.6f} mrad (expected ≈ {pi*1000:.4f})")

    # Additional sanity: random vectors should always give >= 0
    rand1 = torch.randn(5, 3)
    rand2 = torch.randn(5, 3)
    vals = calculate_angles_mrad(rand1, rand2)
    assert torch.all(vals >= 0), "Angle should never be negative!"
    print("Random vector check passed: all angles ≥ 0 mrad")

sanity_check()