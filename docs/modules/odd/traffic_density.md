# `traffic_density.py`

**Stage:** 3
**Package:** `src.odd.traffic_density`

## Purpose

This module implements Stage 3 of the pipeline, "Traffic Density Estimation," exactly as the capstone proposal specifies it: count detected objects per frame, compute a vehicle-to-area ratio, and classify that ratio into one of four density levels (`Low`, `Medium`, `High`, `Congested`). It is a small, deliberately simple module that turns Stage 2's raw detection counts (from `feature_extraction.py`) into a single categorical traffic-density label that downstream stages (Stage 4's composite scenario label, Stage 7's ODD classification) can consume.

Rather than hand-picking arbitrary cut points for the four density bands, the module calibrates them from the observed distribution of ratios in the actual dataset (quartiles), so each band is guaranteed to capture roughly a quarter of the data by construction. This calibration approach is reused by `fuzzy_odd.py` for its own breakpoints.

## Paper / Formula Provenance

This module does not implement a numbered equation from any of the 10 source papers — it is not listed in the README's "Formula provenance" table. Its logic comes directly from the capstone proposal's own Stage 3 description ("Count detected objects per frame; compute vehicle-to-area ratio; classify density into Low/Medium/High/Congested"), and its threshold-calibration method (quartiles of the observed distribution) is a documented, explicitly non-paper design choice made to avoid arbitrary constants.

The two formulas involved are simple and self-contained:

```
vehicle_area_ratio = (vehicle_count + num_two_wheelers + num_autorickshaws) / image_area * 1e4
```
(vehicles per 10,000 px², scaled for human-readable magnitude rather than the raw, very small vehicles-per-pixel ratio)

```
thresholds = {
    low_medium:     25th percentile of observed ratios (q1),
    medium_high:    50th percentile of observed ratios (q2, median),
    high_congested: 75th percentile of observed ratios (q3),
}

classify(ratio):
    ratio <= low_medium      -> Low
    ratio <= medium_high     -> Medium
    ratio <= high_congested  -> High
    otherwise                -> Congested
```

## Public API

### Constants

```python
IMAGE_AREA_PX = 320 * 224   # matches data_pipeline.IMAGE_SIZE (width * height)

LOW = "Low"
MEDIUM = "Medium"
HIGH = "High"
CONGESTED = "Congested"
DENSITY_LEVELS = (LOW, MEDIUM, HIGH, CONGESTED)
```

### `class TrafficDensityThresholds`

```python
@dataclass
class TrafficDensityThresholds:
    low_medium: float
    medium_high: float
    high_congested: float
```
Quartile-calibrated vehicle-to-area-ratio cut points. `low_medium` is the ratio at or below which a scene is `Low`; `medium_high` at or below which `Medium`; `high_congested` at or below which `High` (above it, `Congested`).

```python
def save(self, path: str) -> None
```
Serializes the thresholds to a JSON file (`json.dump(asdict(self), ...)`).

```python
@classmethod
def load(cls, path: str) -> "TrafficDensityThresholds"
```
Loads previously calibrated thresholds from a JSON file written by `save()`.

### `compute_vehicle_area_ratio`

```python
def compute_vehicle_area_ratio(
    vehicle_count: float,
    num_two_wheelers: float,
    num_autorickshaws: float,
    image_area: float = IMAGE_AREA_PX,
) -> float
```
Computes a vehicle-to-frame-area density ratio for one scene: `(vehicle_count + num_two_wheelers + num_autorickshaws) / image_area * 1e4`. `vehicle_count` is car/bus/truck detections, `num_two_wheelers` is bicycle/motorcycle detections, `num_autorickshaws` is auto-rickshaw-like detections. Returns vehicles per 10,000 px².

### `calibrate_thresholds`

```python
def calibrate_thresholds(ratios: List[float]) -> TrafficDensityThresholds
```
Calibrates Low/Medium/High/Congested cut points from observed data using the 25th/50th/75th percentiles (`np.percentile(ratios, [25, 50, 75])`) of the observed ratio distribution.

### `classify_density`

```python
def classify_density(ratio: float, thresholds: TrafficDensityThresholds) -> str
```
Classifies a single vehicle-area ratio into one of `"Low"`, `"Medium"`, `"High"`, `"Congested"`, using `<=` (inclusive) comparisons against the thresholds in order.

### `classify_dataframe`

```python
def classify_dataframe(
    df: pd.DataFrame, thresholds: Optional[TrafficDensityThresholds] = None
) -> Tuple[pd.DataFrame, TrafficDensityThresholds]
```
Adds `vehicle_area_ratio` and `traffic_density_level` columns to a features DataFrame that has `vehicle_count`, `num_two_wheelers`, and `num_autorickshaws` columns (as produced by `feature_extraction.extract_all_features`). If `thresholds` is `None`, thresholds are calibrated fresh from this DataFrame's own ratio distribution; otherwise the supplied thresholds are reused (e.g. thresholds calibrated during a prior training run, applied unchanged to new data). Returns `(df_with_density_columns, thresholds_used)`.

## Key Design Decisions & Edge Cases

- **Quartile calibration, not fixed constants.** `calibrate_thresholds` always produces four roughly-equal-sized bins by construction, at the cost of the bins' absolute meaning depending on whatever dataset it was calibrated on.
- **Boundary inclusivity is on the lower band.** `classify_density` uses `<=` at each cut point, so a ratio exactly equal to `low_medium` is classified `Low`, not `Medium` (verified by `test_classify_density_boundary_inclusive_on_lower_band`).
- **Scaling by `1e4`.** The raw vehicles-per-pixel ratio is tiny (image area is ~71,680 px²), so the ratio is expressed as "vehicles per 10,000 px²" purely for human readability; this does not change the classification outcome, only the displayed magnitude.
- **`IMAGE_AREA_PX` must match Stage 1/2's frame size.** It is hardcoded as `320 * 224` to match `data_pipeline.IMAGE_SIZE`; if that image size ever changes, this constant needs to be updated in lockstep (there is no runtime check tying the two together).
- **Threshold reuse vs. fresh calibration.** `classify_dataframe`'s optional `thresholds` argument lets a caller apply thresholds learned on a training split to a held-out/test split, rather than always recalibrating from scratch.

## Dependencies

- Standard library: `json`, `os`, `dataclasses`, `typing`
- External: `numpy`, `pandas`
- Internal (only inside the `__main__` block): `src.common.paths` (`FEATURES_CSV`, `TRAFFIC_DENSITY_THRESHOLDS_JSON`, `ensure_output_dirs`)

No other `src.odd` module is imported by `traffic_density.py`, but `fuzzy_odd.py` imports from it (`DENSITY_LEVELS`, `TrafficDensityThresholds`, `classify_dataframe`).

## Usage Example

```python
import pandas as pd
from src.odd.traffic_density import classify_dataframe, calibrate_thresholds, classify_density

df = pd.read_csv("outputs/final_features.csv")
labeled_df, thresholds = classify_dataframe(df)

print(thresholds)                                   # TrafficDensityThresholds(...)
print(labeled_df["traffic_density_level"].value_counts())

# Reusing thresholds calibrated on a training set for new data:
new_ratio = 12.4
level = classify_density(new_ratio, thresholds)
```

## Running Standalone

```bash
python -m src.odd.traffic_density
```
Loads `outputs/final_features.csv` (`FEATURES_CSV`), calibrates quartile thresholds and classifies every row, saves the calibrated thresholds to `outputs/traffic_density_thresholds.json` (`TRAFFIC_DENSITY_THRESHOLDS_JSON`), and prints the calibrated thresholds plus the resulting Low/Medium/High/Congested distribution.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/odd/test_traffic_density.py` covers: the ratio computation (zero when no vehicles, linear scaling with count), quartile-based threshold calibration against a known `1..100` sequence, classification into all four bands, lower-band boundary inclusivity, and a save/load JSON round-trip.
