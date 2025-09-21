

<p align="center">
<img src="./images/Differentiable Optics.png" alt="logo" width="1000"/>
</p>

# DOODLE (Not an acronym): Differentiable Heliostat Optics Simulator

A tiny, fast, **differentiable ray-tracing simulator** inspired by [ARTIST](https://github.com/ARTIST-Association/ARTIST) for heliostat fields with **batched heliostat-orientation errors**—plus a Gymnasium-style environment and a reference training loop.

This repo is designed for research on **closed-loop control** of heliostats in **concentrated solar power (CSP) tower** plants. It renders irradiance (“flux”) images on a planar receiver, supports per-episode error sampling, and exposes clean hooks for RL / supervised training.

---

## What’s inside

```
.
├─ newenv_rl_test_multi_error.py   # Core differentiable optics + HelioField
├─ test_environment.py             # Gymnasium-like env (HelioEnv) + losses
├─ train_with_env.py               # Reference training loop & policy nets
└─ plotting_utils.py               # Pretty 3D plots of normals/rays (Plotly)
```

### Key ideas

* **Differentiable optics**: vectorized reflection, ray–plane intersections, Gaussian footprint on the receiver.&#x20;
* **Deterministic error modeling**: per-sun-position, per-heliostat **orientation errors** (East/Up in mrad), pre-sampled and **reused** across calls until you `reset_errors()`.&#x20;
* **Batched rendering**: render B sun positions at once; pre-allocated error tensors keep results reproducible.&#x20;
* **Gym-friendly**: `HelioEnv` returns images + auxiliary features and computes **MSE / distance-weighted / boundary / alignment** losses.&#x20;
* **Plug-in policies**: CNN encoder with MLP / LSTM / Transformer heads; schedulers, gradient clipping, TensorBoard logging, and 3-D diagnostics included.&#x20;

---

## Installation

> Python ≥ 3.10 recommended.

```bash
# create and activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip wheel

# Core deps (pick a CUDA/CPU build that fits your system)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Project deps
pip install gymnasium numpy scipy plotly tensorboard matplotlib adamp
```

*We use the AdamP optimizer from `adamp` in the training script.*&#x20;

---

## Quick start

### 1) Render once with the differentiable optics core

```python
import torch
from newenv_rl_test_multi_error import HelioField

device = "cuda" if torch.cuda.is_available() else "cpu"

# Minimal toy geometry
N = 50
helios = torch.rand(N, 3, device=device) * 10
helios[:, 2] = 0.0  # on ground
target_pos  = torch.tensor([0., -5., 0.], device=device)
target_norm = torch.tensor([0.,  1., 0.], device=device)  # faces +Y
target_area = (15., 15.)  # (width, height)

field = HelioField(
    heliostat_positions=helios,
    target_position=target_pos,
    target_area=target_area,
    target_normal=target_norm,
    error_scale_mrad=90.0,
    sigma_scale=0.1,
    resolution=128,
    device=device,
    max_batch_size=25,
)

# Prepare one sun position and an initial action (normals)
sun = torch.tensor([700., 700., 700.], device=device)
ideal = field.calculate_ideal_normals(sun)
field.init_actions(sun)

# NOTE: render() now expects (sun, action, ideal_normals, ...)
img, actual_normals = field.render(sun, field.initial_action, ideal)
print(img.shape, actual_normals.shape)
```

*`render(sun, action, ideal_normals, monitor=False)` returns `(H, W), (N, 3)` for single-sun or `(B, H, W), (B, N, 3)` for batched calls; set `monitor=True` to also receive reflected ray directions.*&#x20;

### 2) Use the Gym-style environment

```python
import torch
from test_environment import HelioEnv

device = "cuda" if torch.cuda.is_available() else "cpu"
N = 50
helios = torch.rand(N, 3, device=device); helios[:, 2] = 0.0

env = HelioEnv(
    heliostat_pos=helios,
    targ_pos=torch.tensor([0., -5., 0.], device=device),
    targ_area=(15., 15.),
    targ_norm=torch.tensor([0., 1., 0.], device=device),
    sigma_scale=0.1,
    error_scale_mrad=180.0,
    resolution=128,
    batch_size=25,
    device=device,
    new_sun_pos_every_reset=False,
    new_errors_every_reset=True,
    use_error_mask=False,    # set True to focus on worst-k% images
    error_mask_ratio=0.2,    # k (e.g., 0.2 = worst 20%)
)

obs = env.reset()                          # {'img': (B, H, W), 'aux': (B, 3 + N*3)}
action = torch.randn(env.batch_size, N*3, device=device)  # predicted normals (flattened)
obs, metrics, monitor = env.step(action)   # metrics: mse, dist, bound, alignment_loss
print({k: float(v) for k, v in metrics.items()})
```

*`obs['img']` is `(B, H, W)` (no channel); `obs['aux']` concatenates sun position (3) and ideal normals (N×3). Losses include MSE, distance-weighted error, boundary penalty, and an alignment loss in milliradians.*&#x20;

---

## Train a policy

A reference training script with a CNN encoder + (MLP/LSTM/Transformer) head, TensorBoard logging, LR schedulers (plateau / cyclic / exponential), gradient clipping, error-focused masking, and periodic 3-D diagnostics.

```bash
# default LSTM policy
python train_with_env.py --device cuda --num_heliostats 50 --steps 5000

# try a Transformer head
python train_with_env.py --device cuda --architecture transformer --transformer_layers 2 --transformer_heads 8

# alignment pretraining before main losses
python train_with_env.py --alignment_pretrain_steps 200 --alignment_f 150
```

**Notable CLI knobs**

* **Geometry & data**: `--num_heliostats` (50), `--resolution` (128), `--batch_size` (25), `--num_batches` (1).&#x20;
* **Loss scheduling**: `--alignment_pretrain_steps` (100) with `--alignment_f` (100); `--warmup_steps` (40) using boundary-only; then blend MSE (`--mse_f`) and distance loss (`--dist_f`).&#x20;
* **Schedulers**: `--scheduler [plateau|cyclic|exp]` with `--scheduler_*` or `--exp_decay`.&#x20;
* **Regularization & stability**: `--grad_clip` (default `1e-7`), NaN/Inf forward/grad hooks, plus optional **error masking** via `--use_error_mask` & `--error_mask_ratio`.
* **Logs & outputs**: TensorBoard logs under `runs_multi_error_env/...`; periodic 3-D plots (normals, reflected rays) saved to `./monitors_debug/step_XXX/…`.

```bash
tensorboard --logdir runs_multi_error_env
```

---

## Visual diagnostics (3D)

`plotting_utils.scatter3d_vectors(...)` writes standalone Plotly HTML:

* Predicted **normals** colored by boundary violations,
* Predicted **normals** (or **reflected rays**) colored by per-image MAE.

---

## API overview

### `HelioField` (core optics)

```python
HelioField(
    heliostat_positions: Tensor[N,3],
    target_position:     Tensor[3],
    target_area:         tuple[float, float],   # (width, height)
    target_normal:       Tensor[3],
    error_scale_mrad:    float = 1.0,           # orientation error σ (mrad)
    sigma_scale:         float = 0.01,          # Gaussian σ ∝ distance
    initial_action_noise: float = 0.01,         # noise for init actions
    resolution:          int = 100,
    device:              str|torch.device = "cpu",
    max_batch_size:      int = 25,              # pre-alloc error cache
)
```

Most-used methods:

* `calculate_ideal_normals(sun_position) -> (N,3) or (B,N,3)`
* `init_actions(sun_position) -> None` (stores `initial_action`)
* `reset_errors() -> None` (resamples single-sun + batched error tensors)
* `render(sun_position, action, ideal_normals, monitor=False) ->`

  * single sun: `(H,W), (N,3)` **or** `(H,W), (N,3), (H*N,3)` with `monitor=True`
  * batched:    `(B,H,W), (B,N,3)` **or** `(B,H,W), (B,N,3), (B*N,3)`&#x20;

Internals you may reuse:

* `reflect_vectors(incidents, normals)`
* `ray_plane_intersection_batch(origins, dirs, plane_point, plane_normal)`
* `gaussian_blur_batch(...)` → per-mirror Gaussian footprints on the receiver.&#x20;

### `HelioEnv` (Gymnasium-like)

Observations:

* `obs['img']` : `(B, H, W)` flux image(s)
* `obs['aux']` : `(B, 3 + N*3)` → **sun position (3)** + **ideal normals (N×3)**&#x20;

`step(action)`:

* `action`: `(B, N*3)` flattened predicted normals (normalized inside).
* Returns `(obs, metrics, monitor)` where **metrics** include:

  * `mse` — image MSE (normalized by target peak),
  * `dist` — distance-weighted absolute error (via EDT),
  * `bound` — anti-spillage penalty (inside a 75% receiver box),
  * `alignment_loss` — mean angle (mrad) between ideal & actual normals.&#x20;

---

## Design notes & tips

* **Deterministic errors across calls**: batched error tensors are pre-sampled up to `max_batch_size` and **reused** until `reset_errors()`—great for debugging and curriculum schedules.&#x20;
* **Up-axis safety**: a leaky-ReLU constraint keeps the **Up (Z) component** of rotated normals non-negative to avoid shooting into the ground.&#x20;
* **Distance-aware loss + masking**: errors far from the receiver center are penalized more (EDT maps), and you can focus on the worst-k% images via `use_error_mask`.&#x20;

---

## Troubleshooting

* **NaNs/Infs during training**
  The training loop registers forward/grad hooks and prints offenders; also consider LR (`--lr`), `--grad_clip`, and scheduler aggressiveness.&#x20;
* **All-black images**
  Ensure sun Z component is positive and predicted normals aren’t flipped; `HelioEnv` samples sun directions in the upper hemisphere by construction.&#x20;
* **Shape mismatches**
  `action` must flatten to `(B, N*3)`; the env reshapes and normalizes internally. Images are `(B, H, W)` (add a channel dim yourself if your model expects one).&#x20;

---

## Citation

If you use this simulator in academic work, please cite the repository (and, if applicable, your thesis on heliostat control with differentiable ray tracing).

```
@software{heliostat_differentiable_sim,
  title  = {Differentiable Heliostat Optics Simulator (DOODLE)},
  year   = {2025},
  note   = {GitHub repository},
}
```

---

## Acknowledgements

Built as part of a research project on **RL for heliostat control in differentiable ray-tracing simulators** at the German Aerospace Center (DLR).

<div align="center">
  <a href="https://www.dlr.de/EN/Home/home_node.html"><img src="https://www.dlr.de/static/media/Logo-en.bc10c5b6.svg" height="80px" hspace="3%" vspace="25px"></a>
</div>


