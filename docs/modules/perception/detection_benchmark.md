# `detection_benchmark.py`

**Stage:** 2 (Perception Layer)
**Package:** `src.perception.detection_benchmark`

## Purpose

This module implements a standard object-detection evaluation toolkit — mAP, precision, recall, N-point interpolated AP, AUC-based (Waymo-style) AP, and orientation-similarity metrics (AOS/APH) — and applies the geometric subset of it (mAP/AP/precision/recall) to benchmark YOLOv8 vehicle detections against IDD-Lite. It exists to give the project an objective, paper-grounded measure of how good the perception layer's own vehicle detector is, independent of how that detector's output is later used by feature extraction or fusion.

Since IDD-Lite has no real per-instance detection ground truth, this module also derives *pseudo*-ground-truth vehicle boxes directly from the semantic segmentation mask's Vehicle class via connected-component analysis, and documents the resulting limitation explicitly (see below).

## Paper / Formula Provenance

Per the README's "Formula provenance" table, this module implements the mAP/precision/recall/AP/AOS/APH evaluation toolkit from **Nawaz, Tang, Bibi, Xiao, Ho, Yuan, "Robust Cognitive Capability in Autonomous Driving Using Sensor Fusion Techniques: A Survey," IEEE T-ITS 2024**, Section V. Specific equation mappings, per the module's own docstrings:

| Function | Formula | Paper reference |
|---|---|---|
| `precision_recall` | Precision/Recall | Nawaz et al. Eqs. 2–3 |
| `interpolated_ap` | N-point interpolated AP: `AP = (1/N) * sum_{r in S} P_interp(r)`, `P_interp(r) = max{p(r_hat) : r_hat >= r}` | Nawaz et al. Eqs. 4–5 |
| `average_orientation_score` | AOS: `AOS = (1/N) * sum_{r in S} max_{r_hat >= r} s(r_hat)` | Nawaz et al. Eq. 6 |
| `orientation_similarity` | `s(r) = (1/|D(r)|) * sum_{i in D(r)} [(1 + cos(angle_delta_i)) / 2] * delta_i` | Nawaz et al. Eq. 7 |
| `auc_average_precision` | AUC-based (Waymo) AP: `AP = 100 * integral_0^1 max{p(r) : r' >= r} dr` | Nawaz et al. Eq. 8 |
| `compute_mean_ap` | mAP = mean AP across classes | Nawaz et al. Eq. 1 |

## Public API

### Module Constants

```python
VEHICLE_CLASS_ID = 3  # IDD-Lite level3Id 'vehicles' class -- see feature_extraction.py
MIN_BLOB_AREA = 20
DEFAULT_IOU_THRESHOLD = 0.5

Box = Tuple[int, int, int, int]  # (x, y, w, h)
```

### `extract_pseudo_gt_boxes`

```python
def extract_pseudo_gt_boxes(
    mask: np.ndarray, class_id: int = VEHICLE_CLASS_ID, min_area: int = MIN_BLOB_AREA
) -> List[Box]
```

Derives pseudo-ground-truth object boxes from a semantic mask via `cv2.connectedComponentsWithStats` on `(mask == class_id)` (8-connectivity). Skips label `0` (background) and any component with `area < min_area`. Returns a list of `(x, y, w, h)` boxes, one per surviving connected component.

### `compute_iou`

```python
def compute_iou(box_a: Box, box_b: Box) -> float
```

Standard axis-aligned intersection-over-union between two `(x, y, w, h)` boxes. Returns `0.0` if the union area is `0`.

### `match_detections_to_gt`

```python
def match_detections_to_gt(
    scored_boxes: List[Tuple[Box, float]], gt_boxes: List[Box], iou_threshold: float
) -> List[bool]
```

Greedily matches confidence-sorted detections to ground-truth boxes — the standard greedy matching rule used by KITTI/Pascal VOC-style AP evaluation: each GT box can be claimed by at most one detection (its highest-IoU match among detections processed so far, in descending confidence order).

- **Args:** `scored_boxes` — `(box, confidence)` pairs for one image, any order; `gt_boxes` — GT boxes for the same image; `iou_threshold` — minimum IoU for a match to count as true positive.
- **Returns:** a list of `True`/`False` flags, ordered by descending confidence.

### `precision_recall`

```python
def precision_recall(n_tp: int, n_all: int, n_all_gt: int) -> Tuple[float, float]
```

`precision = n_tp / n_all if n_all > 0 else 0.0`; `recall = n_tp / n_all_gt if n_all_gt > 0 else 0.0`. Both `0.0` on zero denominator.

### `interpolated_ap`

```python
def interpolated_ap(sorted_tp_flags: List[bool], num_gt: int, num_recall_points: int = 11) -> float
```

Computes N-point interpolated Average Precision. `sorted_tp_flags` must be in descending-confidence order across the whole evaluation set (not just one image). Returns `0.0` if `num_gt == 0`. Builds cumulative precision/recall arrays, samples `num_recall_points` equally-spaced recall levels via `np.linspace(0.0, 1.0, num_recall_points)`, and for each level takes the max precision among all points with `recall >= level` (`0.0` if none). KITTI historically used `S11` (`num_recall_points=11`, the default here); the 2019+ KITTI convention uses `S40`.

### `auc_average_precision`

```python
def auc_average_precision(sorted_tp_flags: List[bool], num_gt: int) -> float
```

Computes AUC-based AP (Waymo convention) via trapezoidal integration (`np.trapezoid`) over the monotonic precision envelope (`np.maximum.accumulate` on the reversed precision array, then reversed back — standard AP smoothing to remove the sawtooth PR curve), rather than fixed interpolation points. Explicitly prepends the `(recall=0, envelope[0])` origin point before integrating — the module's comment explains why: *"The PR curve implicitly starts at recall=0 (before any detection is made); without prepending that origin point, trapezoidal integration would silently drop the area between recall=0 and the first detection's recall, undercounting AP for every detector."* Returns AP scaled to `[0.0, 100.0]`; `0.0` if `num_gt == 0`.

### `compute_mean_ap`

```python
def compute_mean_ap(ap_per_class: Dict[str, float]) -> float
```

Unweighted mean of the per-class AP values (`0.0` if `ap_per_class` is empty).

### `orientation_similarity`

```python
def orientation_similarity(deltas: Sequence[float], is_true_positive: Sequence[bool]) -> float
```

Computes Average Orientation Similarity at one recall level. `deltas` are angular deviations (radians) between predicted and ground-truth heading, one per detection; `is_true_positive` are parallel True/False flags — per the standard KITTI AOS convention, orientation similarity is only accumulated over true positives (a false positive's `angle` contributes `0` via multiplication by `float(tp)`). Returns `0.0` if `deltas` is empty.

### `average_orientation_score`

```python
def average_orientation_score(
    recall_levels: Sequence[float],
    similarity_by_recall: Sequence[float],
) -> float
```

Computes AOS across recall levels: for each level `r` in `recall_levels`, takes the max of `similarity_by_recall` among entries whose paired `recall_levels` value is `>= r` (`0.0` default if none), then averages. Note the docstring's caveat: this "simplified interface" assumes `recall_levels` and `similarity_by_recall` are the same length/order (i.e. `similarity_by_recall[i]` corresponds to `recall_levels[i]`), rather than sampling `s(r)` at a separately, more finely, sampled set of recall points. Returns `0.0` if `recall_levels` is empty.

### `evaluate_vehicle_detections`

```python
def evaluate_vehicle_detections(
    images: np.ndarray,
    labels: np.ndarray,
    yolo_model,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict[str, float]
```

Benchmarks YOLO vehicle detections against IDD-Lite ground truth end to end.

- **Args:** `images` — `uint8` array `(M, H, W, 3)`; `labels` — `uint8` **ground-truth** semantic mask array `(M, H, W)` (not a SegNet prediction — this module measures YOLO's own accuracy independent of SegNet); `yolo_model` — a loaded Ultralytics YOLO model; `iou_threshold` — IoU required for a match.
- **Returns:** a dict with keys `'ap'` (11-point interpolated), `'auc_ap'`, `'mean_precision'`, `'mean_recall'`, `'num_ground_truth'`, and `'num_detections'`.
- For every image: extracts pseudo-GT vehicle boxes from `labels[i]` via `extract_pseudo_gt_boxes`; runs `run_detection` (imported from `feature_extraction.py`) and keeps only detections whose class is in `VEHICLES` and whose confidence is `>= CONF_THRESHOLD` (both also imported from `feature_extraction.py`). Pools all kept detections across the whole image set, sorts by descending confidence, and greedily matches each to the best unclaimed same-image GT box (inlined logic equivalent to `match_detections_to_gt`, per-image claimed-state tracked in `claimed_per_image`). Computes `interpolated_ap` (11 points), `auc_average_precision`, and final `precision_recall` from the accumulated true-positive flags.

## Key Design Decisions & Edge Cases

- **Pseudo-ground-truth derivation and its documented limitation.** The module's own top-of-file docstring states: *"IDD-Lite's `_inst_label.png` files were verified to be byte-identical to `_label.png` (no real per-instance IDs exist in this 'Lite' dataset variant). Ground-truth vehicle boxes are therefore derived from the semantic mask's Vehicle class ... via connected-component analysis — each connected blob approximates one vehicle instance. This under-counts touching/overlapping vehicles and is documented here as a real data limitation, not a hidden approximation."*
- **AOS/APH are implemented and unit-tested but not used in the practical evaluation function.** Per the module docstring and the top-level README's "Known scope limitations": *"AOS/APH require a per-detection orientation angle, which a 2D bounding-box-only pipeline (ours) does not produce."* `evaluate_vehicle_detections()` therefore only returns the geometric metrics (`ap`, `auc_ap`, `mean_precision`, `mean_recall`); `orientation_similarity`/`average_orientation_score` exist as correctly-implemented, independently-unit-tested formulas without a live data source to feed them in this project.
- **An inconsistency was found between this module's own top-of-file docstring and its code**, worth flagging explicitly: the docstring (lines ~14–15) says ground-truth vehicle boxes come from *"the semantic mask's Vehicle class (id 6)"*, but the actual code constant a few lines below is `VEHICLE_CLASS_ID = 3` (annotated `# IDD-Lite level3Id 'vehicles' class`). Per the top-level README's verified class scheme (`0=drivable-area, 1=non-drivable-area, 2=living-things, 3=vehicles, 4=road-side-objects, 5=far-objects, 6=sky, 7=void`), **id 3 is correct** (vehicles) and **id 6 is sky**, not vehicles. The actual runtime behavior (`VEHICLE_CLASS_ID = 3`) matches the corrected class scheme and is consistent with `feature_extraction.py`'s `VEHICLE_CLASS_ID = 3`; only the docstring's parenthetical "(id 6)" appears to be a stale/incorrect leftover phrase that was not updated. This does not affect behavior — evaluation genuinely runs against class 3 — but is a documentation accuracy issue in the source file itself.
- **Greedy per-image matching is duplicated, not shared, between `match_detections_to_gt` and `evaluate_vehicle_detections`.** The latter re-implements equivalent greedy-matching logic inline (operating across the whole pooled, multi-image detection list with per-image claimed-state) rather than calling `match_detections_to_gt` directly, since that helper assumes a single image's `gt_boxes` list.
- **Missing-data handling:** `interpolated_ap`, `auc_average_precision`, and `orientation_similarity`/`average_orientation_score` all explicitly return `0.0` on empty/zero-ground-truth input rather than raising (divide-by-zero guards).

## Dependencies

- **Standard library:** `math`, `os`, `typing.Dict`, `typing.List`, `typing.Sequence`, `typing.Tuple`
- **External:** `cv2`, `numpy`
- **Internal:** `src.perception.feature_extraction` (`CONF_THRESHOLD`, `VEHICLES`, `run_detection`). The `__main__` block additionally imports `ultralytics.YOLO`, `src.common.paths` (`DATA_DIR`, `YOLO_WEIGHTS`), and `src.perception.data_pipeline.load_and_clean_dataset`.

## Usage Example

```python
from ultralytics import YOLO
from src.perception.data_pipeline import load_and_clean_dataset
from src.perception.detection_benchmark import evaluate_vehicle_detections

images, labels = load_and_clean_dataset("data/idd20k_lite")
yolo = YOLO("models/yolov8n.pt")

results = evaluate_vehicle_detections(images[:200], labels[:200], yolo)
print(results["ap"], results["mean_precision"], results["mean_recall"])
```

## Running Standalone

```bash
python -m src.perception.detection_benchmark
```

Loads and cleans the dataset from `DATA_DIR`, loads YOLOv8 weights from `YOLO_WEIGHTS`, runs `evaluate_vehicle_detections` on the first 200 images/labels, and prints each result key/value under the header `"=== YOLOv8 Vehicle Detection Benchmark (Nawaz et al. 2023 metrics) ==="`.

## Tests

`tests/perception/test_detection_benchmark.py` covers: `compute_iou` for identical, disjoint, and hand-computed partial-overlap boxes; `extract_pseudo_gt_boxes` blob-count correctness and small-blob filtering; `precision_recall` (normal and zero-denominator cases); `interpolated_ap` for a perfect detector (≈1.0), worst detector (0.0), and zero-ground-truth (0.0); `auc_average_precision` for a perfect detector (≈100.0); `compute_mean_ap` (averaging and empty-input cases); `orientation_similarity` (perfect alignment, opposite alignment ≈0, and a false positive contributing 0); and `match_detections_to_gt`'s greedy assignment behavior (correct TP/FP split, and that each GT box is claimed at most once). This file does not exercise `evaluate_vehicle_detections` or `average_orientation_score` directly, nor does it include a real-data integration test.
