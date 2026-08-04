# `multi_stream_fusion.py`

**Stage:** 2 (Perception Layer)
**Package:** `src.perception.multi_stream_fusion`

## Purpose

This module implements dynamic, reliability-weighted fusion of the three "sensor streams" this project's Stage 2 produces from a single camera: the SegNet semantic mask, the YOLO detections, and raw image statistics. For each frame, it scores each stream's momentary trustworthiness (its "reliability"), combines that with a scene-wide context factor (how good the lighting/visibility conditions are), and turns those into normalized fusion weights and a single scalar `fused_confidence` — a summary number downstream stages (e.g. Stage 6 monitoring) can use directly as "how much should we trust this frame's perception output right now."

It exists because a single fixed weighting of SegNet vs. YOLO vs. image-stat features is brittle: a blurry frame should downweight image-stat and SegNet-derived features, a low-confidence YOLO run should downweight vehicle/pedestrian counts, and so on — this module makes that reweighting explicit, dynamic, and per-frame rather than relying on a single fixed set of feature weights.

## Paper / Formula Provenance

Per the README's "Formula provenance" table, this module implements the **dynamic multi-stream fusion weighting pattern** from **Sumalatha, Chaturvedi, R, Patil, Thethi, Hameed, "Autonomous Multi-sensor Fusion Techniques for Environmental Perception in Self-Driving Vehicles," IC3SE 2024**:

```
w_s = f(R_s, C_e)                                        (Eq. 1)
```

where `w_s` is the fusion weight for stream `s`, `R_s` is a reliability measure of that stream, and `C_e` is an environmental context signal.

**Important provenance caveat, stated explicitly in the module's own docstring:** the source paper states this weighting pattern only *conceptually* and never specifies `R_s` or `f`'s functional form ("derived from training the system using various environmental scenarios") — there is no equation to transcribe beyond the abstract `w_s = f(R_s, C_e)` relationship. This module therefore supplies a concrete, documented, **original** instantiation of that pattern rather than a formula copied from the paper:

- **Three streams** stand in for the paper's heterogeneous sensors (camera/LiDAR/radar), since this project has only a single camera: the SegNet semantic mask, the YOLO detections, and raw-image statistics.
- **`R_s`** — a per-stream reliability heuristic (see `compute_stream_reliabilities`), this project's own design.
- **`C_e`** — a scene-brightness/visibility-derived context factor in `[0, 1]` (see `compute_context_factor`), also original.
- **`f`** — a softmax over `(R_s * C_e)`, justified in the docstring as "a standard, monotonic way to turn relative reliability scores into normalized weights — the paper's own stated behavior ('as reliability falls, weight decreases; conversely weight increases') holds for any monotonic `f`, and softmax is the simplest such choice."

## Public API

### Module Constants

```python
SEGNET_STREAM = "segnet_mask"
YOLO_STREAM = "yolo_detections"
IMAGE_STREAM = "image_stats"
STREAMS = (SEGNET_STREAM, YOLO_STREAM, IMAGE_STREAM)

STREAM_COLUMNS: Dict[str, List[str]] = {
    SEGNET_STREAM: [
        "drivable_area", "road_quality", "non_drivable_area_score",
        "non_drivable_area_distance", "living_things_score", "road_end_distance",
    ],
    YOLO_STREAM: [
        "vehicle_count", "num_two_wheelers", "num_pedestrians", "num_animals",
        "num_autorickshaws", "object_presence", "object_distance",
        "lead_vehicle_distance", "traffic_density", "detection_confidence",
    ],
    IMAGE_STREAM: [
        "brightness", "visibility", "wetness", "scene_complexity",
        "pothole_heuristic_score", "pothole_heuristic_count",
    ],
}

SOFTMAX_TEMPERATURE = 4.0
```

`SOFTMAX_TEMPERATURE` is documented as "a documented hyperparameter choice, not a value from the paper" — higher values make weighting more decisive (winner-take-more) between streams of differing reliability.

### `StreamReferences`

```python
@dataclass
class StreamReferences:
    median_drivable_area: float
    median_visibility: float
    median_brightness: float
```

Calibration reference values used to normalize raw reliability signals.

### `calibrate_references`

```python
def calibrate_references(df: pd.DataFrame) -> StreamReferences
```

Calibrates `StreamReferences` from an observed features DataFrame by taking the median of its `drivable_area`, `visibility`, and `brightness` columns.

### `compute_stream_reliabilities`

```python
def compute_stream_reliabilities(row: pd.Series, refs: StreamReferences) -> Dict[str, float]
```

Computes each stream's reliability score `R_s` in `[0.0, 1.0]` for one scene:

- **SegNet:** `min(row["drivable_area"] / refs.median_drivable_area, 1.0)` (`0.0` if the reference median is `<= 0`) — rationale: "a mask that finds little to no road is a likely segmentation failure on a forward-facing dashcam frame."
- **YOLO:** `np.clip(row["detection_confidence"], 0.0, 1.0)` directly — rationale: "already in `[0, 1]`; naturally 0 when there are no detections, which is fine — there is little for that stream to contribute either way."
- **Image stats:** `min(row["visibility"] / refs.median_visibility, 1.0)` (`0.0` if the reference median is `<= 0`) — rationale: "a blurry frame makes brightness/wetness/scene_complexity less trustworthy signals."

All three are additionally clamped to `[0.0, 1.0]` via `np.clip` before returning. Returns a dict mapping each of `STREAMS` to its score.

### `compute_context_factor`

```python
def compute_context_factor(row: pd.Series, refs: StreamReferences) -> float
```

Computes the environmental context signal `C_e` in `[0.0, 1.0]`: a 50/50 composite of relative brightness and relative visibility (each computed the same way as the corresponding `compute_stream_reliabilities` ratio, `0.0` if the reference median is `<= 0`), clamped to `[0.0, 1.0]`:

```python
0.5 * brightness_factor + 0.5 * visibility_factor
```

Rationale from the docstring: "poor lighting/sharpness conditions (fog, night, motion blur) push `C_e` toward 0, uniformly discounting every stream's weight."

### `compute_fusion_weights`

```python
def compute_fusion_weights(
    reliabilities: Dict[str, float], context_factor: float, temperature: float = SOFTMAX_TEMPERATURE
) -> Dict[str, float]
```

Computes fusion weights `w_s = f(R_s, C_e)` via a softmax over `R_s * C_e`:

```python
scores = np.array([reliabilities[s] * context_factor for s in STREAMS])
exp_scores = np.exp(temperature * (scores - scores.max()))  # max-subtraction for stability
weights = exp_scores / exp_scores.sum()
```

Returns a dict mapping each stream to a normalized weight (sums to `1.0`).

### `compute_fused_confidence`

```python
def compute_fused_confidence(reliabilities: Dict[str, float], weights: Dict[str, float]) -> float
```

`fused_confidence = sum_s(w_s * R_s)` — the reliability-weighted average reliability, usable directly by later stages (e.g. Stage 6 monitoring) as one summary number in `[0.0, 1.0]`.

### `fuse_feature_row`

```python
def fuse_feature_row(row: pd.Series, weights: Dict[str, float]) -> pd.Series
```

Reweights a feature row's columns by their stream's fusion weight: every column listed in `STREAM_COLUMNS[stream]` (if present in `row.index`) is multiplied by `weights[stream]`. Produces "an alternative, fusion-weighted feature representation of the same width (usable as a drop-in alternative input to `odd_classifier`'s RandomForest)." Returns a new `pd.Series` (does not mutate `row` — `row.copy()` is used).

### `fuse_dataframe`

```python
def fuse_dataframe(df: pd.DataFrame, refs: StreamReferences = None) -> pd.DataFrame
```

Applies dynamic multi-stream fusion to every row of a features frame.

- **Args:** `df` — a features DataFrame; `refs` — pre-calibrated `StreamReferences` to reuse; if `None`, calibrated fresh from `df` via `calibrate_references`.
- **Returns:** a copy of `df` with added `reliability_<stream>`, `weight_<stream>` columns (one pair per entry in `STREAMS`) and a `fused_confidence` column.
- Iterates rows with `df.iterrows()`, computing reliabilities/context factor/weights/fused confidence per row and writing them back via `df_out.at[idx, ...]`. Note: this is a per-row Python loop (not vectorized), and does **not** call `fuse_feature_row` — it only adds the reliability/weight/confidence columns, not a reweighted copy of the original feature columns.

## Key Design Decisions & Edge Cases

- **The paper gives no functional form to reproduce — this module's `R_s`, `C_e`, and `f` are all original, documented design choices**, not values or formulas transcribed from Sumalatha et al. This is stated explicitly and prominently in the module's top-of-file docstring; treat this module's numeric behavior as this project's own reasonable instantiation of the paper's abstract pattern, not as ground truth from the source paper.
- **`SOFTMAX_TEMPERATURE = 4.0` is an explicitly documented hyperparameter, not a fitted or paper-derived value.**
- **Divide-by-zero guards** in `compute_stream_reliabilities` and `compute_context_factor`: every ratio against a calibrated median is wrapped in `if refs.median_x > 0 else 0.0`, so a degenerate all-zero calibration set doesn't raise.
- **YOLO reliability with zero detections is intentional, not a bug**: `detection_confidence` is `0.0` when no detections exist, which flows straight through as `R_yolo = 0.0` — the docstring explicitly calls this "fine," since there's genuinely nothing for that stream to contribute in that case.
- **Softmax max-subtraction** (`scores - scores.max()`) is the standard numerical-stability trick to prevent `np.exp` overflow, not paper-specific.
- **Zero context factor still yields valid (uniform) weights**, not a degenerate/`NaN` result: when `context_factor = 0`, every stream's `R_s * C_e` score becomes `0`, and softmax over equal scores yields equal (uniform) weights rather than zero or undefined weights — verified by `test_zero_context_factor_still_yields_valid_weights`.
- **`fuse_dataframe` is a row-at-a-time Python loop**, not vectorized across the DataFrame — a deliberate simplicity-over-performance tradeoff appropriate for this project's dataset scale (thousands of rows, not millions).

## Dependencies

- **Standard library:** `os`, `dataclasses.dataclass`, `typing.Dict`, `typing.List`
- **External:** `numpy`, `pandas`
- **Internal:** none at import time. The `__main__` block imports `src.common.paths.FEATURES_CSV`.

## Usage Example

```python
import pandas as pd
from src.perception.multi_stream_fusion import fuse_dataframe

features_df = pd.read_csv("outputs/final_features.csv")
fused_df = fuse_dataframe(features_df)

print(fused_df[["fused_confidence", "weight_segnet_mask", "weight_yolo_detections", "weight_image_stats"]].head())
```

## Running Standalone

```bash
python -m src.perception.multi_stream_fusion
```

Loads `outputs/final_features.csv` (via `FEATURES_CSV`), runs `fuse_dataframe` over it, and prints the mean per-stream reliability and fusion weight for each of the three streams, plus the mean `fused_confidence` across all rows.

## Tests

`tests/perception/test_multi_stream_fusion.py` covers: `calibrate_references` correctly computing medians from a 60-row synthetic features DataFrame fixture; that all reliability scores stay within `[0.0, 1.0]`; that YOLO reliability equals `detection_confidence` exactly; that the context factor stays within `[0.0, 1.0]`; that fusion weights sum to `1.0`; that a higher-reliability stream receives a higher weight than lower-reliability streams; that a zero context factor still produces valid, uniform weights summing to `1.0`; that `compute_fused_confidence` behaves as a correct weighted average in a trivial all-or-nothing case; and that `fuse_dataframe` adds the expected `reliability_*`/`weight_*`/`fused_confidence` columns without changing row count.
