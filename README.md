<p align="center">
<img src="./images/Differentiable Optics.png" alt="logo" width="1000"/>
</p>

# DOODLE (Not an acronym): Differentiable Heliostat Optics Simulator

A tiny, fast, **differentiable ray‑tracing simulator** inspired by [ARTIST](https://github.com/ARTIST-Association/ARTIST) for heliostat fields with **batched mirror‑orientation errors**—plus a Gymnasium‑style environment and a reference training loop.

This repo is designed for research on **closed‑loop control** of heliostats in **concentrated solar power (CSP) tower** plants. It renders irradiance (“flux”) images on a planar receiver, supports per‑episode error sampling, and exposes clean hooks for RL / supervised training.

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

* **Differentiable optics**: vectorized reflection, ray–plane intersections, Gaussian footprint on the receiver.
* **Error modeling**: per‑sun‑position, per‑heliostat **orientation errors** (East/Up in mrad), reusable across batches for determinism.
* **Batched rendering**: render B sun positions at once; errors are pre‑sampled and cached for speed/repeatability.
* **Gym‑friendly**: `HelioEnv` returns images + auxiliary features and computes **MSE / distance‑weighted / boundary / alignment** losses.
* **Plug‑in policies**: MLP / LSTM / Transformer heads with a shared CNN encoder.

---

## Installation

> Python ≥ 3.10 recommended.

```bash
# create and activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip wheel

# core deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA/CPU build
pip install gymnasium numpy scipy plotly tensorboard matplotlib

# (optional) jupyter for quick experiments
pip install jupyter
```

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
target_pos = torch.tensor([0., -5., 0.], device=device)
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
    max_batch_size=25,   # pre-allocates reusable error batches
)

# Prepare one sun position and an initial action (normals)
sun = torch.tensor([700., 700., 700.], device=device)  # arbitrary
field.init_actions(sun)
img, actual_normals = field.render(sun, field.initial_action)  # (128,128), (N,3)
print(img.shape, actual_normals.shape)
```

### 2) Use the Gym‑style environment

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
)

obs = env.reset()                          # {'img': (B,1,H,W), 'aux': (B, 3+N*3)}
action = torch.randn(env.batch_size, N*3, device=device)  # dummy normals
obs, metrics, monitor = env.step(action)   # metrics: mse, dist, bound, alignment_loss
print({k: float(v) for k, v in metrics.items()})
```

---

## Train a policy

A reference training script with CNN encoder + (MLP/LSTM/Transformer) head, TensorBoard logging, LR schedulers, gradient clipping, and periodic 3D diagnostics.

```bash
# default LSTM policy, single training batch, batch_size=25
python train_with_env.py --device cuda --num_heliostats 50 --steps 5000

# try a Transformer head
python train_with_env.py --architecture transformer --transformer_layers 2 --transformer_heads 8

# ramp up misalignment pretraining
python train_with_env.py --alignment_pretrain_steps 200 --alignment_f 150
```

### Notable CLI knobs

* **Geometry & data**

  * `--num_heliostats` (default 50), `--resolution` (128)
  * `--batch_size` (25), `--num_batches` (1): number of *environment replicas* per step
* **Loss scheduling**

  * `--alignment_pretrain_steps` (100) & `--alignment_f` (100)
  * `--warmup_steps` (40) using only boundary loss
  * Post warm‑up blend of MSE and distance‑weighted errors (`--mse_f`, `--dist_f`)
* **Schedulers**

  * `--scheduler [exp|plateau|cyclic]`, with `--exp_decay`, or `--scheduler_patience/factor`, or `--step_size_up/down`
* **Regularization**

  * `--grad_clip` (default `1e-7`)
  * `--use_error_mask` + `--error_mask_ratio`: focus losses on the worst‑k% images

TensorBoard runs are written under `runs_multi_error_env/…`. Launch:

```bash
tensorboard --logdir runs_multi_error_env
```

---

## Visual diagnostics (3D)

`plotting_utils.scatter3d_vectors(...)` helps visualize:

* predicted normals vs. **boundary violations**,
* predicted normals vs. **per‑image MAE**,
* reflected ray directions.

The training script periodically writes standalone HTML plots under `./monitors_debug/step_XXX/…`, which you can open in any browser (even from remote machines via VS Code).

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
    max_batch_size:      int = 25,              # pre‑alloc error cache
)
```

Methods (most used):

* `calculate_ideal_normals(sun_position) -> (N,3) or (B,N,3)`
* `init_actions(sun_position) -> None` (stores `initial_action`)
* `reset_errors() -> None` (resamples both single‑sun & batched error tensors)
* `render(sun_position, action, monitor=False) ->`

  * if single sun: `(H,W), (N,3)` **or** `(H,W), (N,3), (H*N,3)` with `monitor=True`
  * if batched: `(B,H,W), (B,N,3)` **or** `(B,H,W), (B,N,3), (B*N,3)`

Internals you might reuse:

* `reflect_vectors(incidents, normals)`
* `ray_plane_intersection_batch(origins, dirs, plane_point, plane_normal)`
* `gaussian_blur_batch(...)` → per‑mirror Gaussian footprints on the receiver

### `HelioEnv` (Gymnasium‑like)

Observations:

* `obs['img']` : `(B, 1, H, W)` flux image(s)
* `obs['aux']` : `(B, 3 + N*3)` → **sun position (3)** + **ideal normals (N\*3)**

`step(action)`:

* `action`: `(B, N*3)` flattened predicted normals (will be normalized)
* returns `(obs, metrics, monitor)` where:

  * `metrics`:

    * `mse` — image MSE (normalized by target peak),
    * `dist` — distance‑weighted absolute error (uses EDT maps),
    * `bound` — anti‑spillage penalty (outside receiver box),
    * `alignment_loss` — mean angle (mrad) between ideal & actual normals.
  * `monitor`:

    * `normals`, `ideal_normals`, `reflected_rays`, `all_bounds`, `mae_image`

---

## Design notes & tips

* **Deterministic errors across calls**: for batched calls, orientation error tensors are pre‑sampled up to `max_batch_size` and **reused** until `reset_errors()`. This makes debugging and curriculum schedules stable.
* **Up‑axis safety**: a sigmoid constraint keeps the **Up component** of rotated normals non‑negative to avoid shooting into the ground.
* **Distance‑aware loss**: errors far from the receiver center are penalized more (via Euclidean Distance Transform on a high‑flux mask).
* **Boundary penalty**: encourages all intersections to lie within a **shrunk (75%) receiver box** (tunable), robust to rays nearly parallel to the plane.

---

## Troubleshooting

* **NaNs/Infs during training**
  The training loop registers forward/grad hooks and prints offenders. Check LR (`--lr`), `--grad_clip`, and whether you enabled an aggressive scheduler.
* **All‑black images**
  Ensure sun Z component is positive (upper hemisphere) and your normals aren’t flipped. In `HelioEnv`, sun directions are normalized and Z is forced ≥ 0.
* **Mismatch shapes**
  `action` must flatten to `(B, N*3)` and will be reshaped internally. Your policy should output unit normals or leave normalization to the env.

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

Built as part of a research project on **RL for heliostat control in differentiable ray‑tracing simulators** at the German Aerospace Center (DLR).


-----------
<div align="center">
  <a href="https://www.dlr.de/EN/Home/home_node.html"><img src="https://www.dlr.de/static/media/Logo-en.bc10c5b6.svg" height="80px" hspace="3%" vspace="25px"></a>

</div>
