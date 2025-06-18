import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def scatter3d_vectors(
    vectors,
    values,
    mode: str = "color",          # "color"  ➜ values ↦ colour
                                   # "size"   ➜ values ↦ marker size
    colorscale: str = "Jet",   # any Plotly colourscale name
    size_range: tuple = (3, 12),   # min / max marker diameters (pixels) when mode="size"
    equal_axes: bool = True,       # force identical axis scales if True
    html_file: str | Path | None = "scatter.html",  # where to write the standalone HTML
    auto_open: bool = False,       # set True to attempt opening the file automatically
):
    """Create a 3-D scatter and (optionally) save it as a self-contained HTML file.

    Parameters
    ----------
    vectors : array-like, shape (n, 3)
        XYZ coordinates of the points.
    values  : array-like, length n
        Scalar associated with each point.
    mode    : {"color", "size"}
        If "color", map *values* to colours; if "size", map them to marker sizes.
    colorscale : str
        Any Plotly colourscale name when *mode*="color".
    size_range : (min_px, max_px)
        Pixel diameter bounds when *mode*="size".
    equal_axes : bool
        If True, pad ranges so all axes share identical numeric scale and render a cube.
    html_file : str | Path | None
        Destination for the standalone HTML. Set to None to skip writing.
    auto_open : bool
        Attempt to open the file with the default browser (may not work on headless SSH).
    """

    # --- basic validation ----------------------------------------------------
    vec = np.asarray(vectors, dtype=float)
    val = np.asarray(values, dtype=float)

    if vec.ndim != 2 or vec.shape[1] != 3 or vec.shape[0] != val.shape[0]:
        raise ValueError("`vectors` must be (n,3) and `values` length-n")

    # --- build marker spec ---------------------------------------------------
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
        raise ValueError("`mode` must be 'color' or 'size'")

    # --- build figure --------------------------------------------------------
    fig = go.Figure(
        go.Scatter3d(
            x=vec[:, 0], y=vec[:, 1], z=vec[:, 2],
            mode="markers",
            marker=marker,
        )
    )

    # --- axis & aspect handling ---------------------------------------------
    scene_kwargs = dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z")

    if equal_axes:
        mins, maxs = vec.min(axis=0), vec.max(axis=0)

        # Ensure (0,0,0) is included in the axis bounds
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


    fig.update_layout(title="3-D scatter of vectors", scene=scene_kwargs,
                      margin=dict(l=0, r=0, b=0, t=40))

    # --- write self-contained HTML ------------------------------------------
    if html_file is not None:
        html_file = Path(html_file)
        html_file.parent.mkdir(parents=True, exist_ok=True) 
        fig.write_html(html_file, include_plotlyjs="cdn", full_html=True)
        if auto_open:
            try:
                import webbrowser, os
                webbrowser.open("file://" + os.path.abspath(html_file))
            except Exception:
                # silently ignore if browser cannot be launched (e.g., headless SSH)
                pass

    return fig


# ---------------------------------------------------------------------------
# Demo / quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    vecs = np.random.randn(100, 3)
    vals = np.random.rand(100)
    scatter3d_vectors(vecs, vals, mode="color", html_file="scatter.html")
    print("Saved to scatter.html → open in VS Code or any browser.")