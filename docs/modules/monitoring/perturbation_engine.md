# `perturbation_engine.py`

**Stage:** 2 / 6
**Package:** `src.monitoring.perturbation_engine`

## Purpose

This module is a GPU-accelerated (falls back to CPU) weather perturbation engine. It applies synthetic brightness/contrast shifts, Gaussian sensor noise, and radial fog/haze to torch image tensors, using three "regional Indian weather profiles" (Haryana, Punjab, Himachal) that sample randomized perturbation strengths from hand-picked `(min, max)` ranges. It also provides a small segmentation-metrics helper (pixel accuracy / mIoU) and an evaluation routine that measures how much a trained SegNet's accuracy degrades when its input images are perturbed under each regional profile versus a clean baseline.

It exists for two purposes in the pipeline: (1) as a standalone robustness benchmark for Stage 2's SegNet model (how much does mIoU drop under simulated Haryana/Punjab/Himachal weather?), and (2) as the shared low-level perturbation primitive that Stage 6's `perception_monitor.py` reuses to synthesize a pseudo-sequence of "consecutive frames" out of a single static IDD-Lite image (see that module's docs for why).

## Paper / Formula Provenance

Per the README's "Formula provenance" table: the brightness/contrast and fog perturbation models, and the general robust-learning framing, follow **Sun, Cui, Ning, Lu, Cao, Khajepour, "Extending Operational Design Domain for Perception Systems Through Robust Learning," IEEE TIV, Oct 2024**.

Concretely, the module docstring and function docstrings state:
- `adjust_brightness_contrast_gpu`: `x' = delta_contrast * x + delta_bright`.
- `add_fog_gpu`: Koschmieder's transmittance law, `x' = T(x) * x + A * [1 - T(x)]`, where `T(x) = e^(-fog_strength * d(x))` and `d(x)` is each pixel's normalized radial distance from the image center.
- `add_gaussian_noise_gpu` models sensor noise under poor lighting as additive Gaussian noise (a standard robust-learning perturbation, not attributed to a specific numbered equation in the module docstring).

The regional weather profiles (`HARYANA_DELTA`, `PUNJAB_DELTA`, `HIMACHAL_DELTA`) and their `(min, max)` sampling ranges are this project's own India-specific calibration choices layered on top of the paper's perturbation model — they are not themselves taken from the paper.

## Public API

```python
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HARYANA_DELTA: Dict[str, Tuple[float, float]]
PUNJAB_DELTA: Dict[str, Tuple[float, float]]
HIMACHAL_DELTA: Dict[str, Tuple[float, float]]
REGIONAL_PROFILES: Dict[str, Dict[str, Tuple[float, float]]]  # {"haryana": ..., "punjab": ..., "himachal": ...}
```

Each regional dict maps `"brightness"`, `"contrast"`, `"noise"`, `"fog"` to a `(min, max)` sampling range, e.g. `HARYANA_DELTA = {"brightness": (-30, 45), "contrast": (0.85, 1.3), "noise": (5, 18), "fog": (0.05, 0.4)}`.

```python
def adjust_brightness_contrast_gpu(
    img_tensor: torch.Tensor, delta_bright: float = 0.0, delta_contrast: float = 1.0
) -> torch.Tensor
```
Applies `x' = delta_contrast * x + delta_bright` to a `[0, 1]`-range image tensor (`delta_bright` given in raw 0-255 pixel units and divided by 255 internally). Returns the result clamped to `[0.0, 1.0]`.

```python
def add_gaussian_noise_gpu(img_tensor: torch.Tensor, std: float = 10.0) -> torch.Tensor
```
Adds `N(0, std/255)` Gaussian noise to a `[0, 1]`-range tensor (`std` given in raw 0-255 units). Returns the result clamped to `[0.0, 1.0]`.

```python
def add_fog_gpu(img_tensor: torch.Tensor, fog_strength: float = 0.5) -> torch.Tensor
```
Applies a radial fog effect per Koschmieder's law using a per-pixel transmittance mask computed from normalized distance-from-center. Works on tensors of shape `(..., H, W)`. Returns the result clamped to `[0.0, 1.0]`.

```python
def perturb_image_gpu(img_tensor: torch.Tensor, delta: Dict[str, float]) -> torch.Tensor
```
Applies brightness/contrast, then noise, then fog, in that fixed order, using whichever of `"brightness"`/`"contrast"`/`"noise"`/`"fog"` keys are present in `delta` (each is optional; the corresponding step is only run when its key exists). Typically fed the output of `sample_delta()`.

```python
def sample_delta(state: str) -> Dict[str, float]
```
Randomly samples one perturbation delta for the named region (`"haryana"`, `"punjab"`, or `"himachal"`, case-insensitive) — `brightness` via `random.randint`, and `contrast`/`noise`/`fog` via `random.uniform`, each within that region's configured range. Raises `ValueError` for any unrecognized region name.

```python
def calculate_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int = 8
) -> Tuple[float, float]
```
Computes `(miou, pixel_accuracy)` between a predicted and ground-truth label map of matching shape. Pixel accuracy is `(pred == target).mean()`. mIoU is the mean of per-class IoU over the `num_classes` classes, **skipping any class whose union (pred ∪ target) is empty** rather than counting it as 0 — so a frame with no instances of a class doesn't drag the mean down. Returns `(0.0, ...)` for mIoU if no class has nonzero union at all.

```python
def evaluate_perturbation_robustness(
    model: nn.Module, images: np.ndarray, labels: np.ndarray, num_classes: int = 8
) -> Dict[str, Tuple[float, float]]
```
Evaluates a trained SegNet's mIoU/accuracy on a random sample of up to 100 images (seeded via `RANDOM_STATE`), first on the clean baseline, then again once per region in `("haryana", "punjab", "himachal")` with a freshly sampled perturbation delta applied per image. Prints a human-readable summary table as it runs. Returns a dict: `results["baseline"] = (miou, accuracy)` and `results[region] = (miou, accuracy, drop_from_baseline)` for each region.

## Key Design Decisions & Edge Cases

- **Perturbation order is fixed** (brightness/contrast → noise → fog) inside `perturb_image_gpu` — this is a documented simplification, not derived from the paper, since the paper does not specify a compositing order for multiple simultaneous perturbations.
- **Regional profiles are this project's own calibration**, not sourced from the cited paper — the paper supplies the perturbation *formulas* (brightness/contrast linear model, Koschmieder fog law), while the `(min, max)` numeric ranges per region are hand-chosen to represent plausible Haryana/Punjab/Himachal weather variation.
- **mIoU skips zero-union classes** in `calculate_metrics` rather than counting them as 0 IoU, which avoids penalizing a frame simply for not containing a particular class (e.g. no `living-things` pixels in a given crop).
- **CPU fallback with a warning.** `DEVICE` is chosen automatically (`cuda` if available, else `cpu`), and a `print()` warning fires at import time if no GPU is detected — there's no hard failure, since the machine this project runs on has no discrete GPU (see the README's "Known scope limitations").
- **Reproducibility.** Both Python's `random` module and NumPy's RNG are seeded from the same `RANDOM_STATE = 42` (module-level `random.seed()` call plus a local `np.random.default_rng(RANDOM_STATE)` inside `evaluate_perturbation_robustness`), so the 100-image evaluation sample and the sampled deltas are deterministic across runs.
- **Fog transmittance at `fog_strength=0.0`** degenerates to `T(x) = e^0 = 1` everywhere, making `add_fog_gpu` a near-identity transform (this is exercised directly by the test suite).

## Dependencies

**Internal:** None at module scope. The `if __name__ == "__main__":` block imports `src.common.paths` (`DATA_DIR`, `SEGNET_CHECKPOINT`), `src.perception.data_pipeline.load_and_clean_dataset`, and `src.perception.segnet_model.load_segnet`. This module is itself a low-level dependency of `src.monitoring.perception_monitor` and `src.pipeline.full_dataset_pipeline` (both import `calculate_metrics`; `perception_monitor` also imports `perturb_image_gpu`, `sample_delta`, and `DEVICE`).

**External:** `torch`, `torch.nn`, `torch.nn.functional`, `numpy`, plus Python's standard `random`.

## Usage Example

```python
import torch
from src.monitoring.perturbation_engine import sample_delta, perturb_image_gpu, evaluate_perturbation_robustness

# Perturb a single clean image tensor with sampled Punjab-style weather.
img = torch.rand(3, 256, 256)  # (C, H, W), values in [0, 1]
delta = sample_delta("punjab")
foggy_noisy_img = perturb_image_gpu(img, delta)

# Benchmark a trained SegNet's robustness across all three regions.
# results = evaluate_perturbation_robustness(segnet_model, images, labels)
```

## Running Standalone

```bash
python -m src.monitoring.perturbation_engine
```

Loads the IDD-Lite dataset and a trained SegNet checkpoint via `src.common.paths`, then runs `evaluate_perturbation_robustness()`, printing baseline mIoU/accuracy followed by each region's perturbed mIoU/accuracy and its drop from baseline.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/monitoring/test_perturbation_engine.py` covers: identity behavior of `adjust_brightness_contrast_gpu` at default params, output clamping to `[0, 1]` under extreme brightness/contrast, zero-std Gaussian noise being a no-op, zero-strength fog being near-identity, fog output staying bounded under a strong fog strength, `perturb_image_gpu` actually changing the image and preserving shape, `sample_delta` producing values within each region's configured ranges (20 draws per region), `sample_delta` raising `ValueError` for an unknown region, and `calculate_metrics` returning `(1.0, 1.0)` for a perfect match and `(0.0, 0.0)` for zero overlap.
