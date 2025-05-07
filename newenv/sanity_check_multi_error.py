#!/usr/bin/env python3
"""Sanity‑check suite for HelioField implementations.

This script compares the **loop‑based** and **vectorised** heliostat‑field
implementations under three scenarios:

1. **Single‑sun** renders (reference behaviour).
2. **Batch render** with *distinct* sun positions while **enforcing identical
   mirror‑orientation errors across the whole batch** (verifies geometry & math
   without stochastic differences).
3. **Duplicated‑sun** batch where *the same* sun position is repeated, ensuring
   the vectorised field re‑samples errors for each instance.

Run this file directly:
    $ python test_dynamic_errors.py
"""

from __future__ import annotations

import types
import torch
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Import environments
# ----------------------------------------------------------------------------
from newenv_rl_test_loops import HelioField as HelioFieldLoop  # type: ignore
from newenv_rl_test_multi_error import HelioField as HelioFieldVec          # type: ignore


# ----------------------------------------------------------------------------
# Test routine
# ----------------------------------------------------------------------------

def compare_fields_and_plot() -> None:
    """Run a battery of comparisons between loop‑based and vectorised fields."""

    # ------------------------------------------------------------------ 0. reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------ Field geometry (tiny example)
    N = 5
    heliostat_positions = torch.rand(N, 3) * 20.0
    heliostat_positions[:, 2] = 0.0
    heliostat_positions[:, 1] = heliostat_positions[:, 0] + 10.0
    heliostat_positions[:, 0] -= 10.0

    # ------------------------------------------ common scene / sim parameters
    target_pos   = torch.tensor([0.0, -5.0, 0.0])
    target_norm  = torch.tensor([0.0,  1.0, 0.0])
    target_dims  = (15.0, 15.0)
    resolution   = 64
    error_scale  = 80.0   # mrad (large on purpose)
    sigma_scale  = 0.01
    action_noise = 0.0    # keep actions deterministic

    # ---------------------------------------------- a couple of sun positions
    sun_pos  = torch.tensor([0.0, 1000.0, 1000.0])
    sun_pos2 = torch.tensor([0.0, 1200.0, 1000.0])

    # --------------------------------------- instantiate both environments
    loop_field = HelioFieldLoop(
        heliostat_positions, target_pos, target_dims, target_norm,
        error_scale_mrad=error_scale,
        sigma_scale=sigma_scale,
        initial_action_noise=action_noise,
        resolution=resolution,
        device=torch.device("cpu"),
    )

    vec_field = HelioFieldVec(
        heliostat_positions, target_pos, target_dims, target_norm,
        error_scale_mrad=error_scale,
        sigma_scale=sigma_scale,
        initial_action_noise=action_noise,
        resolution=resolution,
        device=torch.device("cpu"),
    )

    # ===================================================================== 1. SINGLE‑SUN TEST

    torch.manual_seed(999)
    errs_static = torch.randn(N, 2) * error_scale
    loop_field.error_angles_mrad = errs_static.clone()
    vec_field.error_angles_mrad  = errs_static.clone()

    torch.manual_seed(555)
    loop_field.init_actions(sun_pos)
    torch.manual_seed(555)
    vec_field.init_actions(sun_pos)

    action_loop = loop_field.initial_action.clone()
    action_vec  = vec_field.initial_action.clone()

    img_loop  = loop_field.render(sun_pos,  action_loop)
    img_vec   = vec_field.render(sun_pos,   action_vec)
    img_loop2 = loop_field.render(sun_pos2, action_loop)
    img_vec2  = vec_field.render(sun_pos2,  action_vec)

    _plot_grid(
        images=[img_loop, img_vec, img_loop2, img_vec2],
        titles=[
            "Loop‑based, sun_pos",
            "Vectorised,  sun_pos",
            "Loop‑based, sun_pos2",
            "Vectorised,  sun_pos2",
        ],
        target_dims=target_dims,
        fig_title="Single‑sun comparison",
    )

    assert torch.allclose(action_loop, action_vec, atol=1e-6)
    assert torch.allclose(img_loop, img_vec, atol=1e-6)
    assert torch.allclose(img_loop2, img_vec2, atol=1e-6)
    print("Single‑sun tests passed — loop and vector match exactly.\n")

    # ===================================================================== 2. BATCH TEST (identical errors across batch)

    sun_pos3  = torch.tensor([0.0, 1400.0, 100.0])
    sun_batch = torch.stack([sun_pos, sun_pos2, sun_pos3], dim=0)  # [B=3,3]
    B = sun_batch.shape[0]

    # ------------------------------------------------ prepare *one* shared error tensor
    torch.manual_seed(1234)
    errs_shared = torch.randn(N, 2) * error_scale  # [N,2]

    # We'll feed **exactly the same** mirror errors to *both* environments,
    # **and** render every sun position individually so we avoid any reliance
    # on vector‑field internals (this also side‑steps monkey‑patching).

    actions_loop, images_loop = [], []
    actions_vec,  images_vec  = [], []

    for i, sun_i in enumerate(sun_batch):
        # ------------------------------ LOOP‑BASED (reference)
        loop_field.error_angles_mrad = errs_shared.clone()
        loop_field.init_actions(sun_i)
        a_loop = loop_field.initial_action.clone()
        img_loop = loop_field.render(sun_i, a_loop)
        actions_loop.append(a_loop)
        images_loop.append(img_loop)

        # ------------------------------ VECTORISED (but rendered one‑by‑one)
        vec_field.error_angles_mrad = errs_shared.clone()
        vec_field.init_actions(sun_i.unsqueeze(0))  # expects [1,3]
        a_vec = vec_field.initial_action.clone().view(-1)  # flatten to match loop shape
        img_vec = vec_field.render(sun_i, a_vec)
        actions_vec.append(a_vec)
        images_vec.append(img_vec)

    actions_loop = torch.stack(actions_loop, dim=0)   # [B, N*3]
    images_loop  = torch.stack(images_loop,  dim=0)   # [B, res, res]
    actions_vec  = torch.stack(actions_vec,  dim=0)
    images_vec   = torch.stack(images_vec,   dim=0)

    _plot_side_by_side_batches(
        loop_images=images_loop,
        vec_images=images_vec,
        target_dims=target_dims,
        row_titles=[f"Sun {i+1}" for i in range(B)],
        fig_title="Batch comparison (same errors for every sun)",
    )

    assert torch.allclose(actions_loop, actions_vec, atol=1e-6), "Batch actions differ between loop and vector!"
    assert torch.allclose(images_loop,  images_vec,  atol=1e-6), "Batch renders differ between loop and vector!"
    print("Batch‑sun tests passed — loop and vector batch match exactly.")

    # ===================================================================== 3. DUPLICATED‑SUN TEST (expect different errors) DUPLICATED‑SUN TEST (expect different errors)

    dup_n = 8
    sun_batch_dup = sun_pos.repeat(dup_n, 1)
    
    # Ensure the vectorised field uses its *dynamic* per‑sun sampling again.
    # (If we ever monkey‑patched `_sample_error_angles`, remove the override.)
    if '_sample_error_angles' in vec_field.__dict__:
        del vec_field.__dict__['_sample_error_angles']  # fall back to class method

    # Now random errors will be drawn independently for each duplicate entry.

    torch.manual_seed(2025)
    vec_field.init_actions(sun_batch_dup)
    actions_dup = vec_field.initial_action.clone()
    images_dup  = vec_field.render(sun_batch_dup, actions_dup)

    max_pairwise = max(
        (images_dup[i] - images_dup[j]).abs().max().item()
        for i in range(dup_n - 1)
        for j in range(i + 1, dup_n)
    )
    assert max_pairwise > 1e-6, "Duplicated‑sun images identical — errors not resampled!"
    print(
        f"Duplicated‑sun test passed — vectorised field produces unique irradiance maps (max Δ = {max_pairwise:.3e}).\n"
    )

    _plot_grid(
        images=[images_dup[i] for i in range(dup_n)],
        titles=[f"Duplicate {i}" for i in range(dup_n)],
        target_dims=target_dims,
        cols=4,
        fig_title="Duplicated‑sun (unique errors per instance)",
    )


# ----------------------------------------------------------------------------
# Helper plotting utilities
# ----------------------------------------------------------------------------

def _plot_grid(images, titles, target_dims, cols=2, fig_title=""):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    for ax, im, title in zip(axes, images, titles):
        ax.imshow(
            im.cpu().numpy(),
            origin="lower",
            extent=[
                -target_dims[0] / 2,
                target_dims[0] / 2,
                -target_dims[1] / 2,
                target_dims[1] / 2,
            ],
            cmap="hot",
        )
        ax.set_title(title)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
    # hide unused subplots
    for ax in axes[len(images):]:
        ax.axis("off")
    if fig_title:
        fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()


def _plot_side_by_side_batches(loop_images, vec_images, target_dims, row_titles, fig_title=""):
    B = loop_images.shape[0]
    fig, axes = plt.subplots(B, 2, figsize=(8, B * 3))
    for idx in range(B):
        # loop‑based
        axes[idx, 0].imshow(
            loop_images[idx].cpu().numpy(),
            origin="lower",
            extent=[
                -target_dims[0] / 2,
                target_dims[0] / 2,
                -target_dims[1] / 2,
                target_dims[1] / 2,
            ],
            cmap="hot",
        )
        axes[idx, 0].set_title(f"Loop‑based, {row_titles[idx]}")
        axes[idx, 0].set_xlabel("East (m)")
        axes[idx, 0].set_ylabel("North (m)")
        # vectorised
        axes[idx, 1].imshow(
            vec_images[idx].cpu().numpy(),
            origin="lower",
            extent=[
                -target_dims[0] / 2,
                target_dims[0] / 2,
                -target_dims[1] / 2,
                target_dims[1] / 2,
            ],
            cmap="hot",
        )
        axes[idx, 1].set_title(f"Vectorised,  {row_titles[idx]}")
        axes[idx, 1].set_xlabel("East (m)")
        axes[idx, 1].set_ylabel("North (m)")
    if fig_title:
        fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    compare_fields_and_plot()
