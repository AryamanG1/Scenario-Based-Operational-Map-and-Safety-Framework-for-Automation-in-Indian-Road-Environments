"""Module 3: GPU-accelerated weather perturbation engine.

Simulates regional Indian weather conditions (brightness/contrast shifts,
sensor noise, fog) as torch tensor operations and measures SegNet accuracy
degradation under each condition. Physics-based perturbation models follow
Sun et al. (IEEE TIV 2024).
"""

import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("Warning: No GPU detected. perturbation_engine will run on CPU.")

# Regional weather profiles (India-specific), each a (min, max) sampling range.
HARYANA_DELTA: Dict[str, Tuple[float, float]] = {
    "brightness": (-30, 45),
    "contrast": (0.85, 1.3),
    "noise": (5, 18),
    "fog": (0.05, 0.4),
}
PUNJAB_DELTA: Dict[str, Tuple[float, float]] = {
    "brightness": (-40, 20),
    "contrast": (0.5, 0.9),
    "noise": (10, 25),
    "fog": (0.3, 0.85),
}
HIMACHAL_DELTA: Dict[str, Tuple[float, float]] = {
    "brightness": (-15, 25),
    "contrast": (0.9, 1.5),
    "noise": (2, 10),
    "fog": (0.0, 0.35),
}
REGIONAL_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "haryana": HARYANA_DELTA,
    "punjab": PUNJAB_DELTA,
    "himachal": HIMACHAL_DELTA,
}


def adjust_brightness_contrast_gpu(
    img_tensor: torch.Tensor, delta_bright: float = 0.0, delta_contrast: float = 1.0
) -> torch.Tensor:
    """Applies a brightness/contrast shift to a [0, 1]-range image tensor.

    Based on: x' = delta_contrast * x + delta_bright.

    Args:
        img_tensor: Image tensor with values in [0.0, 1.0].
        delta_bright: Brightness offset in raw 0-255 pixel units.
        delta_contrast: Multiplicative contrast factor.

    Returns:
        The perturbed tensor, clamped to [0.0, 1.0].
    """
    db = delta_bright / 255.0
    out = img_tensor * delta_contrast + db
    return torch.clamp(out, 0.0, 1.0)


def add_gaussian_noise_gpu(img_tensor: torch.Tensor, std: float = 10.0) -> torch.Tensor:
    """Adds Gaussian sensor noise, simulating poor-lighting camera conditions.

    Args:
        img_tensor: Image tensor with values in [0.0, 1.0].
        std: Noise standard deviation in raw 0-255 pixel units.

    Returns:
        The noisy tensor, clamped to [0.0, 1.0].
    """
    std_norm = std / 255.0
    noise = torch.randn_like(img_tensor) * std_norm
    return torch.clamp(img_tensor + noise, 0.0, 1.0)


def add_fog_gpu(img_tensor: torch.Tensor, fog_strength: float = 0.5) -> torch.Tensor:
    """Applies a radial fog/haze effect based on Koschmieder's transmittance law.

    x' = T(x) * x + A * [1 - T(x)], where T(x) = e^(-fog_strength * d(x)) and
    d(x) is each pixel's normalized distance from the image center.

    Args:
        img_tensor: Image tensor of shape (..., H, W) with values in [0.0, 1.0].
        fog_strength: Controls how quickly transmittance decays with distance.

    Returns:
        The foggy tensor, clamped to [0.0, 1.0].
    """
    h, w = img_tensor.shape[-2], img_tensor.shape[-1]
    x = torch.linspace(-1, 1, w, device=img_tensor.device)
    y = torch.linspace(-1, 1, h, device=img_tensor.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    distance = torch.sqrt(grid_x**2 + grid_y**2)
    fog_mask = torch.exp(-distance * fog_strength)
    foggy = img_tensor * fog_mask + 1.0 * (1.0 - fog_mask)
    return torch.clamp(foggy, 0.0, 1.0)


def perturb_image_gpu(img_tensor: torch.Tensor, delta: Dict[str, float]) -> torch.Tensor:
    """Applies brightness/contrast, then noise, then fog, in sequence.

    Args:
        img_tensor: Image tensor with values in [0.0, 1.0].
        delta: Dict optionally containing 'brightness', 'contrast', 'noise',
            'fog' keys, as produced by sample_delta().

    Returns:
        The fully perturbed tensor, clamped to [0.0, 1.0].
    """
    img = img_tensor
    if "brightness" in delta or "contrast" in delta:
        img = adjust_brightness_contrast_gpu(
            img,
            delta_bright=delta.get("brightness", 0.0),
            delta_contrast=delta.get("contrast", 1.0),
        )
    if "noise" in delta:
        img = add_gaussian_noise_gpu(img, std=delta["noise"])
    if "fog" in delta:
        img = add_fog_gpu(img, fog_strength=delta["fog"])
    return img


def sample_delta(state: str) -> Dict[str, float]:
    """Randomly samples a weather perturbation delta for a given region.

    Args:
        state: One of 'haryana', 'punjab', 'himachal' (case-insensitive).

    Returns:
        A dict with 'brightness' (int), 'contrast', 'noise', 'fog' (floats).

    Raises:
        ValueError: If state is not a recognized region.
    """
    state_key = state.lower()
    if state_key not in REGIONAL_PROFILES:
        raise ValueError(f"Unknown region '{state}'. Expected one of {list(REGIONAL_PROFILES)}.")
    ranges = REGIONAL_PROFILES[state_key]
    return {
        "brightness": random.randint(*ranges["brightness"]),
        "contrast": random.uniform(*ranges["contrast"]),
        "noise": random.uniform(*ranges["noise"]),
        "fog": random.uniform(*ranges["fog"]),
    }


def calculate_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int = 8
) -> Tuple[float, float]:
    """Computes pixel accuracy and mean IoU between a prediction and target mask.

    Args:
        pred: Predicted label map, any integer tensor shape.
        target: Ground-truth label map, same shape as pred.
        num_classes: Number of semantic classes.

    Returns:
        A tuple (miou, pixel_accuracy). Classes with zero union are skipped
        when averaging IoU.
    """
    pixel_accuracy = (pred == target).sum().item() / target.numel()

    ious = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    miou = sum(ious) / len(ious) if ious else 0.0

    return miou, pixel_accuracy


def evaluate_perturbation_robustness(
    model: nn.Module, images: np.ndarray, labels: np.ndarray, num_classes: int = 8
) -> Dict[str, Tuple[float, float]]:
    """Evaluates SegNet mIoU/accuracy degradation under regional weather perturbations.

    Args:
        model: A trained SegNet (or compatible) segmentation model.
        images: uint8 array of shape (M, H, W, 3).
        labels: uint8 array of shape (M, H, W).
        num_classes: Number of semantic classes.

    Returns:
        A dict mapping 'baseline' -> (miou, accuracy) and each region name ->
        (miou, accuracy, drop_from_baseline).
    """
    n = min(100, len(images))
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(images), size=n, replace=False)

    model.eval()

    def _run_eval(region: Optional[str]) -> Tuple[float, float]:
        total_miou = 0.0
        total_acc = 0.0
        with torch.no_grad():
            for idx in indices:
                img_t = torch.tensor(images[idx]).permute(2, 0, 1).float().to(DEVICE) / 255.0
                if region is not None:
                    img_t = perturb_image_gpu(img_t, sample_delta(region))
                lbl_t = torch.tensor(labels[idx]).long().to(DEVICE)

                with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    output = model(img_t.unsqueeze(0))
                output = F.interpolate(
                    output, size=lbl_t.shape, mode="bilinear", align_corners=False
                )
                pred = output.argmax(dim=1).squeeze(0)

                miou, acc = calculate_metrics(pred, lbl_t, num_classes)
                total_miou += miou
                total_acc += acc
        return total_miou / n, total_acc / n

    results: Dict[str, Tuple[float, float]] = {}
    baseline_miou, baseline_acc = _run_eval(None)
    results["baseline"] = (baseline_miou, baseline_acc)

    print(f"=== Evaluation Summary (Tested on {n} images) ===")
    print(f"Baseline (Clean):      mIoU = {baseline_miou:.4f} | Accuracy = {baseline_acc:.4f}")

    for region in ("haryana", "punjab", "himachal"):
        perturbed_miou, perturbed_acc = _run_eval(region)
        drop = baseline_miou - perturbed_miou
        results[region] = (perturbed_miou, perturbed_acc, drop)
        print(
            f"Perturbed ({region.upper()}):   mIoU = {perturbed_miou:.4f} | "
            f"Accuracy = {perturbed_acc:.4f} (Drop = {drop:.4f})"
        )

    return results


if __name__ == "__main__":
    from src.common.paths import DATA_DIR, SEGNET_CHECKPOINT
    from src.perception.data_pipeline import load_and_clean_dataset
    from src.perception.segnet_model import load_segnet

    images, labels = load_and_clean_dataset(DATA_DIR)
    segnet = load_segnet(SEGNET_CHECKPOINT, device=DEVICE)
    evaluate_perturbation_robustness(segnet, images, labels)
