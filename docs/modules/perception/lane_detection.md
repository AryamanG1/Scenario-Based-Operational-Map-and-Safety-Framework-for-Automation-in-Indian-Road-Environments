# `lane_detection.py`

**Stage:** 2 (Perception Layer)
**Package:** `src.perception.lane_detection`

## Purpose

This module estimates the drivable corridor's left/right boundaries for a single frame using classical computer vision — Canny edge detection followed by a probabilistic Hough transform, restricted to the SegNet-predicted road mask. It fulfills the capstone proposal's Stage 2 requirement ("Lane detection via CV/deep learning") and produces scalar lane-quality features (`lane_confidence`, `num_lines_detected`, `lane_width_px`) consumed by downstream ODD-mapping and scenario-classification stages as evidence of how structured/well-marked a road segment is.

It uses classical CV rather than a trained/supervised lane detector because, as the module's own docstring states, **IDD-Lite has no lane-marking annotations**, so a supervised lane detector cannot be trained on this dataset.

## Paper / Formula Provenance

This module is not listed in the README's "Formula provenance" table and is original engineering: standard, well-known classical computer-vision techniques (Canny edge detection, probabilistic Hough line transform, least-squares line fitting) rather than a formula transcribed from one of the 10 cited papers. The `lane_confidence` scoring scheme (1.0/0.5/0.0 based on how many boundaries were fitted) is this project's own uncalibrated heuristic, documented as such in the code.

## Public API

### Module Constants

```python
ROAD_CLASS_ID = 0

CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LENGTH = 15
HOUGH_MAX_LINE_GAP = 10

MIN_ABS_SLOPE = 0.3

Line = Tuple[int, int, int, int]  # (x1, y1, x2, y2)
```

Thresholds/parameters are tuned for the 320×224 IDD-Lite frame size.

### `LaneDetectionResult`

```python
@dataclass
class LaneDetectionResult:
    left_lane: Optional[Tuple[float, float]]
    right_lane: Optional[Tuple[float, float]]
    left_lines: List[Line] = field(default_factory=list)
    right_lines: List[Line] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)
```

Result of one frame's lane-boundary estimation. `left_lane`/`right_lane` are `(slope, intercept)` of the fitted boundary or `None` if that side didn't survive filtering; `left_lines`/`right_lines` are the raw Hough segments classified as candidates for that side; `image_shape` is `(height, width)` of the source frame.

Properties:

```python
@property
def num_lines_detected(self) -> int
```
Total number of Hough line segments kept after slope filtering (`len(left_lines) + len(right_lines)`).

```python
@property
def lane_confidence(self) -> float
```
Heuristic confidence in `[0, 1]`: `1.0` if both boundaries fitted, `0.5` if only one side, `0.0` if neither (`sides_fitted / 2.0`). Per the docstring, downstream ODD stages treat a fully-fitted lane pair as evidence of a well-structured (higher-automation-readiness) road segment.

```python
@property
def lane_width_px(self) -> Optional[float]
```
Estimated lane width in pixels at the bottom row of the frame. Returns `None` if either `left_lane` or `right_lane` is `None`, or if either fitted slope is exactly `0`. Otherwise solves each line for `x` at `y = height - 1` and returns `abs(right_x - left_x)`.

### `detect_road_edges`

```python
def detect_road_edges(image: np.ndarray, road_mask: np.ndarray) -> np.ndarray
```

Extracts Canny edges restricted to the drivable road region. Converts to grayscale, applies a `5×5` Gaussian blur, runs `cv2.Canny(CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)`, builds a binary road mask from `road_mask == ROAD_CLASS_ID`, dilates it with a `5×5` kernel, and masks the edge map with `cv2.bitwise_and`. Returns a single-channel binary edge map `(H, W)`, zeroed outside the (dilated) road region.

### `hough_lane_lines`

```python
def hough_lane_lines(edges: np.ndarray) -> List[Line]
```

Runs `cv2.HoughLinesP` with the module's `HOUGH_*` constants. Returns `[]` if no lines are found (`lines is None`). Otherwise flattens each result entry — handling both the `(N, 1, 4)` and `(N, 4)` output shapes that different OpenCV versions produce — into a list of `(x1, y1, x2, y2)` tuples.

### `classify_lane_lines`

```python
def classify_lane_lines(lines: List[Line], image_width: int) -> Tuple[List[Line], List[Line]]
```

Splits Hough line segments into left- and right-boundary candidates.

- Discards vertical lines (`x2 == x1`, undefined slope).
- Discards near-horizontal lines: `abs(slope) < MIN_ABS_SLOPE`.
- Buckets survivors by slope sign and midpoint position relative to `center_x = image_width / 2.0`: negative slope + midpoint left of center → `left_lines`; positive slope + midpoint right of center → `right_lines`. (Lines with negative slope on the right, or positive slope on the left, are silently dropped by neither branch matching.)
- **Returns:** `(left_lines, right_lines)`.

### `fit_lane_boundary`

```python
def fit_lane_boundary(lines: List[Line]) -> Optional[Tuple[float, float]]
```

Fits a single representative line through a set of segments via `np.polyfit(xs, ys, deg=1)` over all segment endpoints (both endpoints of every line contribute to the fit). Returns `(slope, intercept)` of `y = slope*x + intercept`, or `None` if `lines` is empty.

### `detect_lanes`

```python
def detect_lanes(image: np.ndarray, road_mask: np.ndarray) -> LaneDetectionResult
```

Runs the full pipeline: `detect_road_edges` → `hough_lane_lines` → `classify_lane_lines` → `fit_lane_boundary` (once per side), and packages the result into a `LaneDetectionResult` (including `image_shape=(height, width)` from `image.shape[:2]`).

### `visualize_lanes`

```python
def visualize_lanes(image: np.ndarray, result: LaneDetectionResult) -> np.ndarray
```

Draws the fitted lane boundaries onto a copy of the input image: left boundary in blue `(255, 0, 0)`, right boundary in red `(0, 0, 255)` (both BGR), extrapolated from `y1 = height` to `y2 = int(height * 0.6)`. Skips a side if its lane is `None` or has slope `0`. Returns the annotated BGR image copy.

### `compute_lane_features`

```python
def compute_lane_features(result: LaneDetectionResult) -> dict
```

Summarizes a `LaneDetectionResult` into a dict: `{"lane_confidence": float, "num_lines_detected": int, "lane_width_px": float}` — `lane_width_px` is `0.0` (not `None`) when not estimable, for compatibility with downstream numeric feature pipelines.

## Key Design Decisions & Edge Cases

- **Road-mask dilation before edge masking.** `detect_road_edges` dilates the road binary mask by a `5×5` kernel before masking the Canny edges, with the comment: *"so edges right at the road/non-road boundary (where lane markings often sit) aren't clipped off by a one-pixel-tight mask."* Without this, lane markings sitting exactly at the SegNet-predicted road/non-road boundary could be silently discarded.
- **Slope-magnitude rejection filters road-texture noise.** `MIN_ABS_SLOPE = 0.3` exists because, per the comment, "lane boundaries are never near-horizontal in a forward-facing dashcam view, so this rejects road-surface texture/shadow edges."
- **Cross-OpenCV-version output-shape handling.** `hough_lane_lines` explicitly normalizes `cv2.HoughLinesP`'s output, whose shape differs `(N, 1, 4)` vs `(N, 4)` across OpenCV versions, via `tuple(np.asarray(line).flatten())`.
- **`lane_confidence` and `lane_width_px` are uncalibrated heuristics**, not validated against real lane-boundary ground truth (none exists in IDD-Lite). The confidence score is a simple count of how many sides fitted, not a geometric-quality or IoU-based confidence.
- **Missing-data handling:** `fit_lane_boundary([])` returns `None` rather than raising; `lane_width_px` returns `None` (not an exception) when a side is missing or has zero slope; `compute_lane_features` converts that `None` to `0.0` for downstream numeric consumers.
- Both `classify_lane_lines` and `fit_lane_boundary` operate on plain Python tuples/lists rather than numpy arrays for the line segments themselves, keeping the classification logic simple at the cost of some vectorization.

## Dependencies

- **Standard library:** `dataclasses.dataclass`, `dataclasses.field`, `typing.List`, `typing.Optional`, `typing.Tuple`
- **External:** `cv2`, `numpy`
- **Internal:** none at import time. The `__main__` block additionally imports `os`, `torch`, `src.common.paths` (`DATA_DIR`, `PLOTS_DIR`, `SEGNET_CHECKPOINT`), `src.perception.data_pipeline.load_and_clean_dataset`, and `src.perception.segnet_model.load_segnet`.

## Usage Example

```python
import torch
from src.perception.lane_detection import detect_lanes, compute_lane_features, visualize_lanes

img_t = torch.tensor(sample_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
with torch.no_grad():
    pred_mask = segnet(img_t).argmax(dim=1).squeeze(0).numpy().astype("uint8")

result = detect_lanes(sample_img, pred_mask)
features = compute_lane_features(result)
overlay = visualize_lanes(sample_img, result)
```

## Running Standalone

```bash
python -m src.perception.lane_detection
```

Loads and cleans the dataset from `DATA_DIR`, loads the trained SegNet from `SEGNET_CHECKPOINT`, predicts a mask for the first sample image, runs `detect_lanes`, prints `lane_confidence`, `num_lines_detected`, `lane_width_px`, and the `compute_lane_features` dict, then writes a visualization (`visualize_lanes`) to `outputs/plots/lane_detection_sample.png`.

## Tests

`tests/perception/test_lane_detection.py` covers: `LaneDetectionResult.lane_confidence` at all three fitted-side counts (0/1/2); `lane_width_px` returning `None` when a side is missing; `classify_lane_lines` correctly splitting a hand-constructed left/right line pair by slope and position, and rejecting a near-horizontal line; `fit_lane_boundary` returning `None` on empty input and fitting a correct slope/intercept on a simple two-segment case; `compute_lane_features` returning the expected key set; and an integration smoke test (`test_detect_lanes_runs_on_real_image_without_crashing`, using the `segnet`/`small_images_labels` fixtures) that runs the full `detect_lanes` pipeline on a real image/predicted mask and checks `lane_confidence` stays in `[0, 1]`.
