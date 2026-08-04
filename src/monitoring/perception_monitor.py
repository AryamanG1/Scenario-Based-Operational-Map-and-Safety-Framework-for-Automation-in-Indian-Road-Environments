"""Stage 6: Real-Time Performance Monitoring.

Adapts Jiang et al. 2024's ("Enhancing Autonomous Vehicle Safety Based on
Operational Design Domain Definition, Monitoring, and Functional
Degradation", IEEE TIV) counterfactual-based online monitoring pattern
(their Algorithm 1: raise a functional-degradation fallback flag once more
than 5 consecutive checks show inferred accuracy below a threshold) to
this project's perturbation-sequence proxy for "consecutive frames": IDD-
Lite is a static-image dataset with no true video sequence, so temporal
instability is simulated by drawing repeated perturbed versions of the
SAME frame via perturbation_engine.py's regional weather deltas, and
tracking how consistent the perception pipeline's outputs remain across
that sequence (this is the scope decision documented in this project's
plan -- no live simulator/CARLA is used).

Two consistency signals are tracked per pseudo-frame, relative to that
frame's own clean (unperturbed) baseline:
    - SegNet predicted-mask IoU against the baseline mask.
    - YOLO mean detection-confidence drop against the baseline confidence.

A pseudo-frame is flagged "bad" if either signal crosses its threshold.
The longest run of consecutive bad flags determines the frame's
Nominal/Warning/Critical monitoring state, mirroring the paper's
consecutive-check pattern (scaled down from their "5 consecutive checks"
to this project's much shorter pseudo-sequence length).
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from src.perception.feature_extraction import CONF_THRESHOLD, run_detection
from src.monitoring.perturbation_engine import DEVICE, calculate_metrics, perturb_image_gpu, sample_delta

SEQUENCE_LENGTH = 6
IOU_CONSISTENCY_THRESHOLD = 0.5
CONFIDENCE_DROP_THRESHOLD = 0.3
CONSECUTIVE_WARNING = 2
CONSECUTIVE_CRITICAL = 4
MONITORED_REGIONS = ("haryana", "punjab", "himachal")

NOMINAL = "Nominal"
WARNING = "Warning"
CRITICAL = "Critical"


@dataclass
class FrameCheckResult:
    """One pseudo-frame's consistency check outcome.

    Attributes:
        mask_iou: IoU between this pseudo-frame's predicted mask and the
            clean baseline's predicted mask.
        confidence_drop: max(0, baseline_confidence - this_confidence).
        is_bad: Whether either signal crossed its instability threshold.
    """

    mask_iou: float
    confidence_drop: float
    is_bad: bool


@dataclass
class MonitoringResult:
    """The Stage 6 monitoring outcome for one frame.

    Attributes:
        state: One of "Nominal", "Warning", "Critical".
        max_consecutive_bad: The longest run of consecutive bad checks
            observed across the pseudo-sequence.
        checks: Every individual FrameCheckResult, in sequence order.
    """

    state: str
    max_consecutive_bad: int
    checks: List[FrameCheckResult]


def _predicted_mask(segnet_model: torch.nn.Module, img_tensor_01: torch.Tensor) -> np.ndarray:
    """Runs a SegNet forward pass and returns the argmax predicted mask.

    Args:
        segnet_model: Trained SegNet.
        img_tensor_01: Image tensor of shape (3, H, W), values in [0, 1].

    Returns:
        A uint8 predicted mask of shape (H, W).
    """
    segnet_model.eval()
    with torch.no_grad():
        output = segnet_model(img_tensor_01.unsqueeze(0).to(DEVICE))
    return output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def _mean_confident_confidence(detections: List[dict]) -> float:
    """Mean confidence of detections at/above CONF_THRESHOLD, or 0.0 if none."""
    confident = [d["confidence"] for d in detections if d["confidence"] >= CONF_THRESHOLD]
    return float(np.mean(confident)) if confident else 0.0


def run_perception_monitor(
    image: np.ndarray,
    segnet_model: torch.nn.Module,
    yolo_model,
    num_checks: int = SEQUENCE_LENGTH,
) -> MonitoringResult:
    """Runs the Stage 6 monitoring pseudo-sequence for one frame.

    Args:
        image: BGR image array of shape (H, W, 3).
        segnet_model: Trained SegNet.
        yolo_model: Loaded Ultralytics YOLO model.
        num_checks: Number of perturbed pseudo-frames to draw and check.

    Returns:
        A MonitoringResult.
    """
    baseline_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    baseline_mask = _predicted_mask(segnet_model, baseline_tensor)
    baseline_confidence = _mean_confident_confidence(run_detection(image, yolo_model))

    checks: List[FrameCheckResult] = []
    for i in range(num_checks):
        region = MONITORED_REGIONS[i % len(MONITORED_REGIONS)]
        delta = sample_delta(region)
        perturbed_tensor = perturb_image_gpu(baseline_tensor.to(DEVICE), delta)

        perturbed_mask = _predicted_mask(segnet_model, perturbed_tensor)
        mask_iou, _ = calculate_metrics(
            torch.tensor(perturbed_mask), torch.tensor(baseline_mask), num_classes=8
        )

        perturbed_img = (perturbed_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        perturbed_confidence = _mean_confident_confidence(run_detection(perturbed_img, yolo_model))
        confidence_drop = max(baseline_confidence - perturbed_confidence, 0.0)

        is_bad = (mask_iou < IOU_CONSISTENCY_THRESHOLD) or (confidence_drop > CONFIDENCE_DROP_THRESHOLD)
        checks.append(FrameCheckResult(mask_iou=mask_iou, confidence_drop=confidence_drop, is_bad=is_bad))

    max_consecutive = 0
    current_streak = 0
    for check in checks:
        if check.is_bad:
            current_streak += 1
            max_consecutive = max(max_consecutive, current_streak)
        else:
            current_streak = 0

    if max_consecutive >= CONSECUTIVE_CRITICAL:
        state = CRITICAL
    elif max_consecutive >= CONSECUTIVE_WARNING:
        state = WARNING
    else:
        state = NOMINAL

    return MonitoringResult(state=state, max_consecutive_bad=max_consecutive, checks=checks)


if __name__ == "__main__":
    from ultralytics import YOLO

    from src.common.paths import DATA_DIR, SEGNET_CHECKPOINT, YOLO_WEIGHTS
    from src.perception.data_pipeline import load_and_clean_dataset
    from src.perception.segnet_model import load_segnet

    dataset_dir = DATA_DIR
    checkpoint_path = SEGNET_CHECKPOINT
    yolo_weights_path = YOLO_WEIGHTS

    images, _ = load_and_clean_dataset(dataset_dir)
    segnet = load_segnet(checkpoint_path)
    yolo = YOLO(yolo_weights_path)

    for i in range(3):
        result = run_perception_monitor(images[i], segnet, yolo)
        print(f"Frame {i}: state={result.state}, max_consecutive_bad={result.max_consecutive_bad}")
        for j, check in enumerate(result.checks):
            print(
                f"  check {j}: mask_iou={check.mask_iou:.3f}, "
                f"confidence_drop={check.confidence_drop:.3f}, bad={check.is_bad}"
            )
