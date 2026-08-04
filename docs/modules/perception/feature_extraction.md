# `feature_extraction.py`

**Stage:** 2 (Perception Layer)
**Package:** `src.perception.feature_extraction`

## Purpose

This module runs YOLOv8 nano object detection on every image and combines the detections with a SegNet-predicted semantic mask and raw image statistics to produce a 22-dimensional feature vector per frame describing traffic, road, and visibility conditions. These per-frame features are the primary numeric interface between Stage 2 (Perception) and the downstream Stage 3 (Traffic Density), Stage 4 (Scenario Classification), and Stage 7 (Decision System) stages — everything from `traffic_density.py`'s vehicle-to-area ratio to `odd_classifier.py`'s RandomForest ODD classifier consumes this module's output CSV.

The 22 features are: the original 18-feature spec, plus `num_animals`/`num_autorickshaws` (added for the capstone proposal's India-specific-entity requirement), plus a heuristic pothole-candidate score/count (`pothole_heuristic_score`, `pothole_heuristic_count`).

## Paper / Formula Provenance

This module is not listed in the README's "Formula provenance" table — it is original engineering, not a formula transcribed from one of the 10 cited papers. It combines an external pretrained model (YOLOv8, via the `ultralytics` package) with hand-designed feature formulas (e.g. the `scene_complexity` weighted sum below) that are this project's own uncalibrated design choices, not cited to any paper.

**This module is also the origin of the "dataset-mapping correction" documented in the top-level README.** Its own module docstring explains: two of the original 18 features were renamed after discovering that IDD-Lite's semantic classes do not include a pothole or water/puddle class at all (verified directly against the pixel data) — `pothole_score`/`pothole_distance` are now `non_drivable_area_score`/`non_drivable_area_distance` (what class 1 actually is), and `water_level` is now `living_things_score` (what class 2 actually is: pedestrians/animals, not water). See "Key Design Decisions & Edge Cases" below for the full story.

## Public API

### Module Constants

```python
VEHICLES = ["car", "bus", "truck"]
TWO_WHEELERS = ["bicycle", "motorcycle"]
PEDESTRIANS = ["person"]
ANIMALS = ["cow", "horse", "dog", "sheep", "elephant", "bear"]

ROAD_CLASS_ID = 0
NON_DRIVABLE_CLASS_ID = 1
LIVING_THINGS_CLASS_ID = 2
VEHICLE_CLASS_ID = 3
SKY_CLASS_ID = 6
DISTANCE_K = 500
CONF_THRESHOLD = 0.3

AUTORICKSHAW_MIN_ASPECT = 0.55
AUTORICKSHAW_MAX_ASPECT = 1.05
AUTORICKSHAW_MAX_AREA_FRACTION = 0.04

POTHOLE_DARKNESS_STD_FACTOR = 1.5
POTHOLE_MIN_AREA = 15
POTHOLE_MAX_AREA_FRACTION = 0.05
POTHOLE_MIN_FILL_RATIO = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Prints `"Warning: No GPU detected. feature_extraction will run on CPU."` if no GPU is found.

### `run_detection`

```python
def run_detection(img: np.ndarray, model: YOLO) -> List[Dict]
```

Runs YOLOv8 object detection on a single BGR image (`(H, W, 3)`) using a loaded Ultralytics `YOLO` model. Returns a list of dicts, each with `'class'` (str), `'bbox'` (`[x, y, w, h]` ints), and `'confidence'` (float) keys. Internally calls `model(img, verbose=False)[0]` and converts each box's `xyxy` corners to `[x, y, w, h]`.

### `estimate_distance`

```python
def estimate_distance(bbox_height: float, k: float = DISTANCE_K) -> float
```

Estimates a rough object distance from bounding-box height as `k / (bbox_height + 1)`. Smaller/farther boxes (smaller `bbox_height`) yield larger estimated distances. `k` is a calibration constant (larger `k` → larger estimated distances).

### `detect_pothole_candidates`

```python
def detect_pothole_candidates(gray: np.ndarray, mask: np.ndarray) -> Tuple[float, int]
```

Heuristically flags dark, compact anomalies on the road as pothole candidates. See "Key Design Decisions & Edge Cases" for the full caveat.

- **Args:** `gray` — single-channel grayscale image `(H, W)`; `mask` — semantic segmentation mask `(H, W)`.
- **Returns:** `(score, count)` — `score` is the fraction of the road region flagged as pothole-candidate pixels, `count` is the number of surviving candidate blobs. Both are `0` if there's no road region.
- Algorithm: computes `road_binary = (mask == ROAD_CLASS_ID)`; if `road_area == 0`, returns `(0.0, 0)` immediately. Otherwise computes a darkness threshold as `road_pixels.mean() - POTHOLE_DARKNESS_STD_FACTOR * road_pixels.std()`, builds a `dark_mask` of pixels below that threshold and within the road region, applies `cv2.morphologyEx(..., cv2.MORPH_OPEN, np.ones((3,3)))` to remove speckle noise, then runs `cv2.connectedComponentsWithStats` and keeps only components whose area is within `[POTHOLE_MIN_AREA, POTHOLE_MAX_AREA_FRACTION * road_area]` and whose fill ratio (`area / (w*h)`) is `>= POTHOLE_MIN_FILL_RATIO` (rejects thin shadow streaks in favor of compact blobs).

### `is_autorickshaw_like`

```python
def is_autorickshaw_like(detection: Dict, image_shape: Tuple[int, int]) -> bool
```

Heuristically flags a `'car'`-class YOLO detection as auto-rickshaw-like, since COCO (and therefore YOLOv8) has no three-wheeler class.

- **Args:** `detection` — one detection dict as returned by `run_detection()`; `image_shape` — `(height, width)` of the source frame, for area normalization.
- **Returns:** `False` immediately if `detection["class"] != "car"` or if the box height is `0`. Otherwise computes `aspect_ratio = w / h` and `area_fraction = (w * h) / (image_shape[0] * image_shape[1])`, and returns `True` iff `AUTORICKSHAW_MIN_ASPECT <= aspect_ratio <= AUTORICKSHAW_MAX_ASPECT` **and** `area_fraction <= AUTORICKSHAW_MAX_AREA_FRACTION`.

### `compute_features`

```python
def compute_features(
    img: np.ndarray, mask: np.ndarray, detections: List[Dict]
) -> Dict[str, float]
```

Computes the (actually 22, see note below) ODD-relevant scene features for a single frame.

- **Args:** `img` — BGR image `(H, W, 3)`; `mask` — semantic segmentation mask `(H, W)`; `detections` — YOLO detections for this frame, as from `run_detection()`.
- **Returns:** a dict of 22 feature name → value pairs (see the docstring-accuracy note below — the function's own docstring text says "18").
- Filters detections to `confident_dets` (`confidence >= CONF_THRESHOLD`) before most counts.
- Computed keys, in return order: `vehicle_count`, `num_two_wheelers`, `num_pedestrians`, `num_animals`, `num_autorickshaws` (via `is_autorickshaw_like`), `object_presence` (`1` if any raw detection exists, regardless of confidence), `object_distance` (mean `estimate_distance` over **all** detections, `0.0` if none), `lead_vehicle_distance` (min `estimate_distance` over confident vehicle detections only, `0.0` if none), `traffic_density` (`len(confident_dets)`), `detection_confidence` (mean confidence over all detections, `0.0` if none), `drivable_area` (fraction of mask pixels equal to `ROAD_CLASS_ID`), `road_end_distance` (`mask.shape[0] - min(road_row_indices)`, `0.0` if no road rows), `non_drivable_area_score`/`non_drivable_area_distance` (analogous, for `NON_DRIVABLE_CLASS_ID`), `living_things_score` (fraction of mask pixels equal to `LIVING_THINGS_CLASS_ID`), `brightness` (mean pixel value of `img`), `visibility` (variance of the Laplacian of the grayscale image — a sharpness/blur proxy), `wetness` (`np.std(img)`), `road_quality` (variance of road-region pixel values, `0.0` if no road pixels), `pothole_heuristic_score`/`pothole_heuristic_count` (from `detect_pothole_candidates`), and `scene_complexity` (see formula below).
- `scene_complexity` formula (uncited, hand-tuned weights):
  ```python
  density_norm = min(traffic_density / 50, 1.0)
  tw_norm = min(num_two_wheelers / 20, 1.0)
  ped_norm = min(num_pedestrians / 20, 1.0)
  distance_risk = 1 / (lead_vehicle_distance + 1)
  confidence_risk = 1 - detection_confidence
  scene_complexity = min(
      0.30 * density_norm + 0.20 * tw_norm + 0.20 * ped_norm
      + 0.15 * distance_risk + 0.15 * confidence_risk,
      1.0,
  )
  ```

### `extract_all_features`

```python
def extract_all_features(
    images: np.ndarray,
    labels: np.ndarray,
    segnet_model: nn.Module,
    yolo_model: YOLO,
    save_path: str = "final_features.csv",
) -> pd.DataFrame
```

Extracts the feature vector for every image in the dataset.

- **Args:** `images` — `uint8` array `(M, H, W, 3)` (BGR); `labels` — `uint8` ground-truth array `(M, H, W)`, **accepted for signature completeness but not used** by the feature formulas (the docstring explains this mirrors a real deployment-time pipeline where no ground truth exists — features are computed from the SegNet's own predicted mask, not the label); `segnet_model` — a trained SegNet used to produce per-image predicted masks; `yolo_model` — a loaded Ultralytics YOLO model; `save_path` — where to save the resulting feature CSV.
- **Returns:** a `pd.DataFrame` with one row per image (docstring says "18 feature columns"; actually 22 — see note below).
- Runs `del labels` immediately (explicit no-op documented with the comment `# Not used: features are computed from the predicted mask.`) then, for every image, runs it through `segnet_model` in `no_grad()` mode to get a predicted mask (`argmax(dim=1)`), runs `run_detection`, calls `compute_features`, and appends the row. Saves the resulting DataFrame to `save_path` via `df.to_csv(save_path, index=False)` and prints a confirmation line.

## Key Design Decisions & Edge Cases

- **The class-ID mapping bugfix (important project history).** The module's own constants block (lines ~35–44) documents in detail that an early one-shot LLM scaffold which seeded this project's first draft claimed IDD-Lite's classes were `1=Pothole, 2=Water/Puddle, 3=Sidewalk, 6=Vehicle`. Direct inspection of the pixel data (spatial distribution statistics, per-class blob shape statistics, and rendered image/label overlays) showed this was wrong. The verified scheme actually used throughout this module is:
  ```
  0 = drivable-area   1 = non-drivable-area   2 = living-things (people/animals)
  3 = vehicles        4 = road-side-objects   5 = far-objects (buildings/vegetation)
  6 = sky             7 = unlabeled/void
  ```
  There is **no pothole or water/puddle class at all**. Consequently the originally-named features `pothole_score`/`pothole_distance` were renamed to `non_drivable_area_score`/`non_drivable_area_distance` (what class 1 actually measures), and `water_level` was renamed to `living_things_score` (what class 2 actually measures — a genuinely useful "Dynamic Entities" ODD signal on its own merits, not a repurposed water-detection feature). `tests/perception/test_feature_extraction.py::test_class_id_mapping_matches_verified_idd_lite_scheme` is a regression test guarding this fix.
- **`detect_pothole_candidates()` and `is_autorickshaw_like()` are both explicitly documented as uncalibrated heuristics, not trained/validated detectors.** IDD-Lite has no pothole ground truth (no pothole class exists) and no auto-rickshaw ground truth (COCO/YOLOv8 has no three-wheeler class), so neither can be tuned or scored against real labels. `detect_pothole_candidates`'s own docstring states it "is expected to also fire on shadows, oil stains, manhole covers, and other dark road-surface features, not exclusively real potholes; treat its output as a weak, best-effort signal." This matches the top-level README's "Known scope limitations" bullet: *"`detect_pothole_candidates()` is an uncalibrated heuristic, not a trained/validated pothole detector."*
- **Docstring/behavior inconsistencies found while reading this module (not corrected in code, documented here for accuracy):**
  1. `compute_features()`'s own docstring for the `mask` parameter still reads *"where class 0=road, 1=pothole, 2=water (per the IDD-Lite class scheme)"* — this is the **old, debunked** mapping that the rest of this very same file explicitly documents as wrong. The actual code correctly uses `NON_DRIVABLE_CLASS_ID = 1` and `LIVING_THINGS_CLASS_ID = 2` (not pothole/water), consistent with the corrected scheme and the module's own top-of-file explanation — only this one inline docstring sentence was not updated to match.
  2. Both `compute_features()`'s `Returns` docstring ("a dict mapping each of the 18 feature names...") and `extract_all_features()`'s docstring ("A DataFrame with one row per image and 18 feature columns") say **18**, but the actual returned dict/DataFrame has **22** columns (confirmed by `tests/perception/test_feature_extraction.py::test_compute_features_returns_22_columns` and `test_full_extraction_on_real_images`, and by the module's own top-of-file docstring which correctly says "22-dimensional feature vector"). These two inner docstrings are stale leftovers from before `num_animals`, `num_autorickshaws`, `pothole_heuristic_score`, and `pothole_heuristic_count` were added.
- **`object_presence`/`object_distance`/`detection_confidence` use *all* detections (not just confidence-filtered ones)**, while `vehicle_count`, `num_two_wheelers`, `num_pedestrians`, `num_animals`, `num_autorickshaws`, and `traffic_density` use only `confident_dets` (`confidence >= CONF_THRESHOLD`) — an intentional distinction between "is anything detected at all" signals and "how many things are confidently detected" signals.
- **Missing-data handling:** empty-detections and empty-road-region cases are handled explicitly throughout (`0.0` fallbacks for `object_distance`, `detection_confidence`, `lead_vehicle_distance`, `road_end_distance`, `road_quality`, and the pothole score/count), verified by `test_compute_features_no_detections_edge_case` and `test_compute_features_no_road_edge_case`.

## Dependencies

- **Standard library:** `os`, `typing.Dict`, `typing.List`, `typing.Tuple`
- **External:** `cv2`, `numpy`, `pandas`, `torch`, `torch.nn`, `ultralytics.YOLO`
- **Internal:** none at import time. The `__main__` block imports `src.common.paths` (`DATA_DIR`, `FEATURES_CSV`, `SEGNET_CHECKPOINT`, `YOLO_WEIGHTS`, `ensure_output_dirs`), `src.perception.data_pipeline.load_and_clean_dataset`, and `src.perception.segnet_model.load_segnet`.
- Downstream consumers (not imports of this module, but modules that import *from* it): `src.perception.detection_benchmark` imports `CONF_THRESHOLD`, `VEHICLES`, and `run_detection` from this module.

## Usage Example

```python
from ultralytics import YOLO
from src.perception.data_pipeline import load_and_clean_dataset
from src.perception.segnet_model import load_segnet
from src.perception.feature_extraction import extract_all_features

images, labels = load_and_clean_dataset("data/idd20k_lite")
segnet = load_segnet("models/refined_segnet.pth")
yolo = YOLO("models/yolov8n.pt")

df = extract_all_features(images, labels, segnet, yolo, save_path="outputs/final_features.csv")
print(df.shape)  # (M, 22)
```

## Running Standalone

```bash
python -m src.perception.feature_extraction
```

Ensures output directories exist, loads and cleans the dataset from `DATA_DIR`, loads the trained SegNet checkpoint (`SEGNET_CHECKPOINT`) and YOLOv8 weights (`YOLO_WEIGHTS`), and runs `extract_all_features` over the whole dataset, saving the resulting feature CSV to `FEATURES_CSV`.

## Tests

`tests/perception/test_feature_extraction.py` covers: a regression test for the class-ID mapping bugfix (`ROAD_CLASS_ID`/`NON_DRIVABLE_CLASS_ID`/`LIVING_THINGS_CLASS_ID`/`VEHICLE_CLASS_ID` equal 0/1/2/3); that `compute_features` returns exactly 22 columns; the no-detections and no-road edge cases; that `is_autorickshaw_like` only flags `'car'`-class detections; that `detect_pothole_candidates` returns `(0.0, 0)` on a no-road mask and on a perfectly flat/uniform road; and a real-data integration test (`test_full_extraction_on_real_images`, requiring the `segnet`/`yolo`/`small_images_labels` fixtures, skipped if models/dataset are absent) that runs `extract_all_features` end-to-end on 3 real images and sanity-checks the `brightness`/`drivable_area` value ranges.
