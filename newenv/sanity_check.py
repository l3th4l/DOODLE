#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt

# import original (loop-based) environment
from newenv_rl_test_loops import HelioField as HelioFieldLoop
# import vectorized environment
from newenv_rl_test import HelioField as HelioFieldVec

def compare_fields_and_plot():
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # create a small random field of 5 heliostats on CPU
    N = 5
    heliostat_positions = torch.rand(N, 3) * 20.0
    heliostat_positions[:, 2] = 0.0
    heliostat_positions[:, 1] = heliostat_positions[:, 0] + 10.0
    heliostat_positions[:, 0] -= 10.0

    # shared parameters
    target_pos   = torch.tensor([0.0, -5.0, 0.0])
    target_norm  = torch.tensor([0.0,  1.0, 0.0])
    target_dims  = (15.0, 15.0)
    resolution   = 64
    error_scale  = 80.0
    sigma_scale  = 0.01
    action_noise = 0.0  # deterministic

    # two single sun positions
    sun_pos  = torch.tensor([  0.0, 1000.0, 1000.0])
    sun_pos2 = torch.tensor([  0.0, 1200.0, 1000.0])

    # instantiate both fields on CPU
    loop_field = HelioFieldLoop(
        heliostat_positions, target_pos, target_dims, target_norm,
        error_scale_mrad=error_scale,
        sigma_scale=sigma_scale,
        initial_action_noise=action_noise,
        resolution=resolution,
        device=torch.device('cpu')
    )
    vec_field = HelioFieldVec(
        heliostat_positions, target_pos, target_dims, target_norm,
        error_scale_mrad=error_scale,
        sigma_scale=sigma_scale,
        initial_action_noise=action_noise,
        resolution=resolution,
        device=torch.device('cpu')
    )

    # force both to use the same error angles
    torch.manual_seed(999)
    errs = torch.randn(N, 2) * error_scale
    loop_field.error_angles_mrad = errs.clone()
    vec_field.error_angles_mrad  = errs.clone()

    # init identical actions for the first sun
    torch.manual_seed(555)
    loop_field.init_actions(sun_pos)
    torch.manual_seed(555)
    vec_field.init_actions(sun_pos)

    action_loop = loop_field.initial_action.clone()
    action_vec  = vec_field.initial_action.clone()

    # render both for sun_pos
    img_loop  = loop_field.render(sun_pos,  action_loop)
    img_vec   = vec_field.render(sun_pos,   action_vec)
    # and for sun_pos2
    img_loop2 = loop_field.render(sun_pos2, action_loop)
    img_vec2  = vec_field.render(sun_pos2,  action_vec)

    # plot them side by side
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    titles = [
        "Loop-based, sun_pos",
        "Vectorized,  sun_pos",
        "Loop-based, sun_pos2",
        "Vectorized,  sun_pos2",
    ]
    images = [img_loop, img_vec, img_loop2, img_vec2]
    for ax, im, title in zip(axes.flatten(), images, titles):
        ax.imshow(im.cpu().numpy(), origin='lower',
                  extent=[-target_dims[0]/2, target_dims[0]/2,
                          -target_dims[1]/2, target_dims[1]/2],
                  cmap='hot')
        ax.set_title(title)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
    plt.tight_layout()
    plt.show()

    # exact comparisons
    assert torch.allclose(action_loop, action_vec, atol=1e-6), "Initial actions differ!"
    assert torch.allclose(img_loop, img_vec, atol=1e-6),    f"First render mismatch (max error {(img_loop-img_vec).abs().max()})"
    assert torch.allclose(img_loop2, img_vec2, atol=1e-6), f"Second render mismatch (max error {(img_loop2-img_vec2).abs().max()})"
    print("Single‑sun tests passed — loop and vector match exactly.\n")

    # --- BATCH TEST (batch_size = 3) ---
    # define three sun positions
    sun_pos3 = torch.tensor([0.0, 1400.0, 100.0])
    sun_batch = torch.stack([sun_pos, sun_pos2, sun_pos3], dim=0)  # [3,3]

    # batched init_actions + render
    vec_field.init_actions(sun_batch)                   # produces [3, N*3]
    action_batch_vec = vec_field.initial_action.clone() # [3, N*3]
    images_batch_vec = vec_field.render(sun_batch, action_batch_vec)  # [3, res, res]

    # loop-based: do init_actions + render separately
    action_batch_loop = []
    images_batch_loop = []
    for i in range(sun_batch.shape[0]):
        loop_field.init_actions(sun_batch[i])
        a = loop_field.initial_action.clone()
        action_batch_loop.append(a)
        img = loop_field.render(sun_batch[i], a)
        images_batch_loop.append(img)
    action_batch_loop = torch.stack(action_batch_loop, dim=0)   # [3, N*3]
    images_batch_loop = torch.stack(images_batch_loop, dim=0)   # [3, res, res]

    # plot batch comparisons
    fig2, axes2 = plt.subplots(3, 2, figsize=(8, 12))
    batch_titles = ["Sun 1", "Sun 2", "Sun 3"]
    for idx in range(3):
        # loop-based
        axes2[idx,0].imshow(images_batch_loop[idx].cpu().numpy(), origin='lower',
                            extent=[-target_dims[0]/2, target_dims[0]/2,
                                    -target_dims[1]/2, target_dims[1]/2],
                            cmap='hot')
        axes2[idx,0].set_title(f"Loop-based, {batch_titles[idx]}")
        axes2[idx,0].set_xlabel("East (m)")
        axes2[idx,0].set_ylabel("North (m)")
        # vectorized
        axes2[idx,1].imshow(images_batch_vec[idx].cpu().numpy(), origin='lower',
                            extent=[-target_dims[0]/2, target_dims[0]/2,
                                    -target_dims[1]/2, target_dims[1]/2],
                            cmap='hot')
        axes2[idx,1].set_title(f"Vectorized,  {batch_titles[idx]}")
        axes2[idx,1].set_xlabel("East (m)")
        axes2[idx,1].set_ylabel("North (m)")
    plt.tight_layout()
    plt.show()

    # compare batches
    assert torch.allclose(action_batch_loop, action_batch_vec, atol=1e-6), \
        "Batch actions differ between loop and vector!"
    assert torch.allclose(images_batch_loop, images_batch_vec, atol=1e-6), \
        "Batch renders differ between loop and vector!"
    print("Batch‑sun tests passed — loop and vector batch match exactly.")

if __name__ == "__main__":
    compare_fields_and_plot()
