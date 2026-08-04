# `perception_monitor.py`

**Stage:** 6
**Package:** `src.monitoring.perception_monitor`

## Purpose

This module implements Stage 6 of the pipeline: Real-Time Performance Monitoring. Given a single image and the already-trained SegNet and YOLO models, it draws a short pseudo-sequence of perturbed variants of that same image, checks how consistent the perception pipeline's outputs stay across the sequence relative to the image's own clean baseline, and classifies the frame's monitoring state as **Nominal**, **Warning**, or **Critical**.

It exists to give the pipeline a runtime self-check: if SegNet's predicted mask or YOLO's detection confidence becomes unstable under small, realistic perturbations (simulated regional weather), that instability is a proxy signal that the perception outputs for this scene shouldn't be trusted as-is, which downstream feeds into the Stage 7 decision system's Normal/Degraded/Takeover call.

## Paper / Formula Provenance

Per the module docstring and the README's citation table, this module adapts **Jiang, Pan, Liu, Han, Pan, Li, Pan, "Enhancing Autonomous Vehicle Safety Based on Operational Design Domain Definition, Monitoring, and Functional Degradation," IEEE TIV, Oct 2024** — specifically their **Algorithm 1**: a counterfactual-based online monitor that raises a functional-degradation fallback flag once more than 5 consecutive checks show inferred accuracy below a threshold.

**This is an adaptation, not a literal transcription, and that adaptation is the single most important thing to understand about this module.** The paper's monitor is designed for true video: consecutive real frames from a live camera feed. IDD-Lite is a static-image dataset with no video sequences at all. Rather than skip Stage 6 monitoring entirely, this module substitutes a **perturbation-sequence proxy**: it draws several repeated, differently-perturbed versions of the *same single frame* (via `perturbation_engine.py`'s regional weather deltas) and treats consistency across that synthetic sequence as a stand-in for temporal stability across real consecutive frames. The paper's "5 consecutive checks" threshold is also scaled down (`CONSECUTIVE_WARNING = 2`, `CONSECUTIVE_CRITICAL = 4`) to fit this project's much shorter pseudo-sequence (`SEQUENCE_LENGTH = 6`). This is explicitly called out in the README's "Known scope limitations" section: "Stage 6 monitoring uses a perturbation-sequence proxy, not true video."

## Public API

```python
SEQUENCE_LENGTH = 6
IOU_CONSISTENCY_THRESHOLD = 0.5
CONFIDENCE_DROP_THRESHOLD = 0.3
CONSECUTIVE_WARNING = 2
CONSECUTIVE_CRITICAL = 4
MONITORED_REGIONS = ("haryana", "punjab", "himachal")

NOMINAL = "Nominal"
WARNING = "Warning"
CRITICAL = "Critical"
```

```python
@dataclass
class FrameCheckResult:
    mask_iou: float
    confidence_drop: float
    is_bad: bool
```
One pseudo-frame's consistency check outcome. `mask_iou` is the IoU between this pseudo-frame's predicted SegNet mask and the clean baseline's predicted mask. `confidence_drop` is `max(0, baseline_confidence - this_confidence)` for YOLO's mean detection confidence. `is_bad` is `True` if either signal crosses its instability threshold.

```python
@dataclass
class MonitoringResult:
    state: str
    max_consecutive_bad: int
    checks: List[FrameCheckResult]
```
The Stage 6 monitoring outcome for one frame. `state` is one of `"Nominal"`, `"Warning"`, `"Critical"`. `max_consecutive_bad` is the longest run of consecutive bad checks observed in the pseudo-sequence. `checks` holds every individual `FrameCheckResult` in sequence order.

```python
def run_perception_monitor(
    image: np.ndarray,
    segnet_model: torch.nn.Module,
    yolo_model,
    num_checks: int = SEQUENCE_LENGTH,
) -> MonitoringResult
```
Runs the full Stage 6 monitoring pseudo-sequence for one frame:
1. Computes the clean baseline SegNet mask and YOLO mean confident-detection confidence for `image`.
2. For `num_checks` iterations, cycles through `MONITORED_REGIONS` (`i % len(MONITORED_REGIONS)`), samples a perturbation delta for that region via `sample_delta()`, perturbs the baseline tensor via `perturb_image_gpu()`, re-runs SegNet and YOLO on the perturbed frame, and computes `mask_iou` (against the baseline mask) and `confidence_drop` (against the baseline confidence).
3. Flags a check `is_bad` if `mask_iou < IOU_CONSISTENCY_THRESHOLD` **or** `confidence_drop > CONFIDENCE_DROP_THRESHOLD`.
4. Finds the longest consecutive run of bad checks (`max_consecutive_bad`) and maps it to a state: `>= CONSECUTIVE_CRITICAL` → `Critical`, `>= CONSECUTIVE_WARNING` → `Warning`, else `Nominal`.

Args: `image` — a BGR image array of shape `(H, W, 3)`; `segnet_model` — trained SegNet; `yolo_model` — loaded Ultralytics YOLO model; `num_checks` — number of perturbed pseudo-frames to draw (default `SEQUENCE_LENGTH = 6`). Returns a `MonitoringResult`.

**Private helpers** (not part of the public API but relevant to behavior):
- `_predicted_mask(segnet_model, img_tensor_01) -> np.ndarray` — runs a SegNet forward pass and returns the `argmax` predicted mask as `uint8`.
- `_mean_confident_confidence(detections) -> float` — mean confidence of detections at/above `CONF_THRESHOLD` (imported from `feature_extraction.py`), or `0.0` if there are none.

## Key Design Decisions & Edge Cases

- **No real video, by necessity.** IDD-Lite is a static-image dataset. The module docstring is explicit that "consecutive frames" are simulated by drawing repeated perturbed versions of the SAME frame, and that this is "the scope decision documented in this project's plan — no live simulator/CARLA is used." Anyone reading this module's output as if it reflected true temporal consistency across a real driving sequence would be misinterpreting it.
- **Two independent instability signals, OR'd together.** A pseudo-frame is bad if *either* the SegNet mask IoU drops below threshold *or* YOLO's mean confidence drops too much — either signal alone is sufficient to flag instability, since either a segmentation or a detection failure independently threatens downstream safety.
- **Confidence drop is one-sided (`max(0, ...)`).** An *increase* in confidence under perturbation is not penalized — only degradation counts toward instability.
- **Thresholds are scaled-down from the source paper.** `CONSECUTIVE_WARNING = 2` and `CONSECUTIVE_CRITICAL = 4` are deliberately smaller than the paper's "more than 5 consecutive checks," to match this project's much shorter `SEQUENCE_LENGTH = 6` pseudo-sequence (using the paper's literal threshold of 5 would make `Critical` essentially unreachable within a 6-check sequence).
- **Regions cycle deterministically, not randomly**, via `i % len(MONITORED_REGIONS)` — every run of `num_checks >= 3` samples all three regions at least once, keeping coverage even rather than leaving it to chance.
- **Only used per-scene, not per-batch.** Because it re-runs SegNet and YOLO up to `num_checks` extra times per image, it's too expensive to run over an entire dataset — `feasibility_map.py` explicitly assumes a "Nominal" Stage 6 status for batch feasibility maps rather than invoking this monitor per row (see `feasibility_map.py`'s docs).

## Dependencies

**Internal:** `src.perception.feature_extraction` (`CONF_THRESHOLD`, `run_detection`), `src.monitoring.perturbation_engine` (`DEVICE`, `calculate_metrics`, `perturb_image_gpu`, `sample_delta`). The `if __name__ == "__main__":` block additionally imports `src.common.paths` (`DATA_DIR`, `SEGNET_CHECKPOINT`, `YOLO_WEIGHTS`), `src.perception.data_pipeline.load_and_clean_dataset`, and `src.perception.segnet_model.load_segnet`.

**External:** `numpy`, `torch`, and (in the standalone block) `ultralytics.YOLO`.

## Usage Example

```python
from src.monitoring.perception_monitor import run_perception_monitor

# image: BGR np.ndarray (H, W, 3); segnet, yolo: already-loaded models
result = run_perception_monitor(image, segnet, yolo, num_checks=6)
print(result.state)              # "Nominal" / "Warning" / "Critical"
print(result.max_consecutive_bad)
for check in result.checks:
    print(check.mask_iou, check.confidence_drop, check.is_bad)
```

## Running Standalone

```bash
python -m src.monitoring.perception_monitor
```

Loads the IDD-Lite dataset, a trained SegNet checkpoint, and YOLO weights via `src.common.paths`, then runs `run_perception_monitor()` on the first 3 images, printing each frame's resulting state, `max_consecutive_bad`, and per-check `mask_iou`/`confidence_drop`/`is_bad` values.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/monitoring/test_perception_monitor.py` covers: the consecutive-bad-streak → state classification thresholds (`_classify` helper mirroring the module's inline logic, checked at and around both `CONSECUTIVE_WARNING` and `CONSECUTIVE_CRITICAL` boundaries), `FrameCheckResult` field access, and an end-to-end `test_run_perception_monitor_on_real_frame` (using `small_images_labels`/`segnet`/`yolo` pytest fixtures) that runs the monitor on a real frame with `num_checks=2` and asserts the resulting state is one of the three valid values, exactly 2 checks were run, and each check's `mask_iou` is in `[0, 1]` with `confidence_drop >= 0`.
