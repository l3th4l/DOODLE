"""3-D scatter plotting utility with optional direction arrow.

This module defines a single public function, ``scatter3d_vectors``, which takes
XYZ points plus one scalar per point and renders them with Plotly.  Values can
be mapped either to marker colour or marker size, and the plot can be written
as a self-contained HTML file for easy sharing.

Changes in this revision
========================
* **Arrow now always originates at (0, 0, 0).**  When ``origin_position`` and
  ``target_position`` are supplied the function computes the direction vector
  ``target − origin``, normalises it, and draws an arrow from the coordinate-
  frame origin to that unit vector.  In other words, the shaft runs from the
  global origin to the *direction*, not from *origin_position*.
* Internal refactor: simplified axis-range handling and added extra comments.

Quick example
-------------
>>> import numpy as np
>>> from scatter3d_vectors import scatter3d_vectors
>>> np.random.seed(0)
>>> vecs = np.random.randn(100, 3)
>>> vals = np.random.rand(100)
>>> scatter3d_vectors(
...     vecs,
...     vals,
...     origin_position=[0, 0, 0],
...     target_position=[1.5, 2.0, -1.0],
... )
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _as_1d_array(x: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Cast *x* to a NumPy 1-D float array of length 3, checking shape."""
    if isinstance(x, torch.Tensor):
        x = x.cpu()
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name!r} must be an array-like of length 3")
    return arr


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def scatter3d_vectors(
    vectors: Sequence[Sequence[float]] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    mode: str = "color",              # "color" → values ↦ colours,
                                       # "size"  → values ↦ marker size
    colorscale: str = "Jet",
    size_range: tuple[int | float, int | float] = (3, 12),
    equal_axes: bool = True,
    origin_position: Sequence[float] | None = None,
    target_position: Sequence[float] | None = None,
    html_file: str | Path | None = "scatter.html",
    auto_open: bool = False,
):
    """Create a 3-D scatter plot of *vectors*.

    Parameters
    ----------
    vectors : array-like, shape (n, 3)
        XYZ coordinates of the points to plot.
    values  : array-like, length *n*
        Scalar associated with each point.
    mode    : {"color", "size"}, default "color"
        If "color", map *values* to colours; if "size", map them to marker sizes.
    colorscale : str, default "Jet"
        Plotly colourscale name used when *mode*="color".
    size_range : (min_px, max_px), default (3, 12)
        Pixel diameter bounds when *mode*="size".
    equal_axes : bool, default True
        If True, pad ranges so all axes have identical numeric scale and the
        scene renders as a cube.
    origin_position, target_position : array-like length 3, optional together
        If **both** are provided the function will compute the direction vector
        ``target − origin``, normalise it, and draw an arrow from *(0, 0, 0)* to
        that unit vector.
    html_file : str | Path | None, default "scatter.html"
        Destination for the standalone HTML file.  Pass ``None`` to skip
        writing.
    auto_open : bool, default False
        Attempt to open the file in the default browser (ignored on headless
        setups).

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.
    """

    # ---------------------------------------------------------------------
    # Basic validation & preprocessing
    # ---------------------------------------------------------------------
    vec = np.asarray(vectors, dtype=float)
    val = np.asarray(values, dtype=float)

    if vec.ndim != 2 or vec.shape[1] != 3 or vec.shape[0] != val.shape[0]:
        raise ValueError("`vectors` must have shape (n, 3) and `values` length n")

    if (origin_position is None) ^ (target_position is None):
        raise ValueError("Provide *both* `origin_position` and `target_position`, or neither.")

    # ---------------------------------------------------------------------
    # Marker specification
    # ---------------------------------------------------------------------
    if mode == "color":
        marker = dict(
            size=6,
            color=val,
            colorscale=colorscale,
            colorbar=dict(title="Value"),
            opacity=0.85,
        )
    elif mode == "size":
        vmin, vmax = float(val.min()), float(val.max())
        sizes = np.interp(val, (vmin, vmax), size_range)
        marker = dict(size=sizes, color="royalblue", opacity=0.85)
    else:
        raise ValueError("`mode` must be either 'color' or 'size'")

    # ---------------------------------------------------------------------
    # Initialise figure with scatter of points
    # ---------------------------------------------------------------------
    fig = go.Figure(
        go.Scatter3d(
            x=vec[:, 0],
            y=vec[:, 1],
            z=vec[:, 2],
            mode="markers",
            marker=marker,
            name="Points",
        )
    )

    # ---------------------------------------------------------------------
    # Optional direction arrow (always from global origin)
    # ---------------------------------------------------------------------
    arrow_end: np.ndarray | None = None
    if origin_position is not None and target_position is not None:
        origin = _as_1d_array(origin_position, "origin_position")
        target = _as_1d_array(target_position, "target_position")

        direction_vec = target - origin
        norm = float(np.linalg.norm(direction_vec))
        if norm == 0:
            raise ValueError("`origin_position` and `target_position` are identical –\n"
                             "direction vector has zero length.")
        unit_vec = direction_vec / norm
        arrow_start = np.zeros(3)
        arrow_end = unit_vec  # end-point of the arrow after normalisation

        # Arrow shaft (black line)
        fig.add_trace(
            go.Scatter3d(
                x=[arrow_start[0], arrow_end[0]],
                y=[arrow_start[1], arrow_end[1]],
                z=[arrow_start[2], arrow_end[2]],
                mode="lines",
                line=dict(color="black", width=5),
                name="Direction",
            )
        )

        # Arrow head (cone) – anchor at tip so it sits at arrow_end
        fig.add_trace(
            go.Cone(
                x=[arrow_end[0]],
                y=[arrow_end[1]],
                z=[arrow_end[2]],
                u=[unit_vec[0]],
                v=[unit_vec[1]],
                w=[unit_vec[2]],
                anchor="tip",
                sizemode="absolute",
                sizeref=0.2,
                showscale=False,
                colorscale=[[0.0, "black"], [1.0, "black"]],
                name="Arrow head",
            )
        )

    # ---------------------------------------------------------------------
    # Axis range and aspect-ratio handling
    # ---------------------------------------------------------------------
    scene_kwargs: dict[str, object] = dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")

    if equal_axes:
        # Combine scatter points and arrow_end (if present) for range calculation
        bounds_points = vec.copy()
        if arrow_end is not None:
            bounds_points = np.vstack([bounds_points, arrow_end])

        mins, maxs = bounds_points.min(axis=0), bounds_points.max(axis=0)

        # Ensure the global origin is always visible
        mins = np.minimum(mins, 0)
        maxs = np.maximum(maxs, 0)

        max_range = (maxs - mins).max()
        mid = (maxs + mins) / 2.0

        scene_kwargs.update(
            xaxis=dict(range=[mid[0] - max_range / 2, mid[0] + max_range / 2]),
            yaxis=dict(range=[mid[1] - max_range / 2, mid[1] + max_range / 2]),
            zaxis=dict(range=[mid[2] - max_range / 2, mid[2] + max_range / 2]),
            aspectmode="cube",
        )

    # ---------------------------------------------------------------------
    # Final layout tweaks
    # ---------------------------------------------------------------------
    fig.update_layout(
        title="3-D scatter of vectors" + (" with direction arrow" if arrow_end is not None else ""),
        scene=scene_kwargs,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # ---------------------------------------------------------------------
    # Write HTML if requested
    # ---------------------------------------------------------------------
    if html_file is not None:
        html_path = Path(html_file)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
        if auto_open:
            try:
                import os
                import webbrowser

                webbrowser.open("file://" + os.path.abspath(html_path))
            except Exception:
                pass  # ignore if browser cannot be launched (headless etc.)

    return fig


# ---------------------------------------------------------------------------
# Demonstration when run as a script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    demo_vecs = np.random.randn(100, 3)
    demo_vals = np.random.rand(100)
    scatter3d_vectors(
        demo_vecs,
        demo_vals,
        mode="color",
        origin_position=[0, 0, 0],
        target_position=[1.5, 2.0, -1.0],
        html_file="scatter.html",
    )
    print("Saved to scatter.html – open in any browser.")