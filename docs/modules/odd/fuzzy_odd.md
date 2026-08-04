# `fuzzy_odd.py`

**Stage:** 4
**Package:** `src.odd.fuzzy_odd`

## Purpose

This module implements Stage 4, "Scenario Classification," by partitioning each scene into a fuzzy micro-ODD (uODD) weather state — `NoRain`, `LowRain`, or `HeavyRain` — and then combining that state with Stage 3's traffic-density category into one composite scenario label (e.g. `"HeavyRain-Congested"`). This composite label is the primary hand-off artifact from perception/traffic-density into the ODD-mapping stages (Stage 5) and ultimately the Stage 7 decision system.

Because IDD-Lite is a static-image dataset with no precipitation-rate sensor, the module cannot observe true rainfall directly. Instead it approximates surface wetness/puddling from an image-derived `wetness` feature (pixel-intensity standard deviation, a reflective-surface proxy already computed in `feature_extraction.py`) and evaluates the fuzzy rules using only that proxy term — a documented, legitimate partial application of the source paper's OR-combined rules, not a different model.

## Paper / Formula Provenance

Salvi, Weiss, Trapp, Oboril, Buerkle, "Fuzzy Interpretation of Operational Design Domains in Autonomous Driving," IEEE IV 2022 — specifically:

- **Appendix I**: the triangular (`Tr`) and trapezoidal (`Tp`) membership function shapes.
- **Def. 3.2**: the Precipitation (P) and Precipitation-Deposits (PD) linguistic variables. Only PD is implemented here (P is not observable from IDD-Lite).
- **Table I**: the NoRain/LowRain/HeavyRain fuzzy rules, which are OR-combinations of P and PD (e.g. `"LowRain: P is Low OR PD is Wet"`). This module evaluates them using only the PD term.

Formulas, reproduced with the code's own notation:

```
Triangular membership Tr(x; a, b, c):
    left  = (x - a) / (b - a)     [= 1.0 if b == a]
    right = (c - x) / (c - b)     [= 1.0 if c == b]
    Tr(x; a, b, c) = max(min(left, right), 0.0)

Trapezoidal membership Tp(x; a, b, c, d):
    left  = (x - a) / (b - a)     [= 1.0 if b == a]
    right = (d - x) / (d - c)     [= 1.0 if d == c]
    Tp(x; a, b, c, d) = max(min(left, 1.0, right), 0.0)
```

Fuzzy uODD rules (Salvi et al. Table I, evaluated on the PD term only):

```
NoRain:    PD is Dry
LowRain:   PD is Wet
HeavyRain: PD is Puddles
```

with the PD membership functions instantiated over quartile-calibrated wetness breakpoints `(w_min, p25, p50, p75, w_max)`:

```
pd_dry_membership(w)     = Tp(w; w_min, w_min, p25, p50)
pd_wet_membership(w)     = Tr(w; p25, p50, p75)
pd_puddles_membership(w) = Tp(w; p50, p75, w_max, w_max)
```

The paper's own numeric membership breakpoints are given in physical precipitation-sensor units this project doesn't have access to, so — following the same calibration approach as `traffic_density.py` — the breakpoints are instead calibrated from the observed `wetness` feature's own quartiles.

## Public API

### Constants

```python
NO_RAIN = "NoRain"
LOW_RAIN = "LowRain"
HEAVY_RAIN = "HeavyRain"
MU_ODD_STATES = (NO_RAIN, LOW_RAIN, HEAVY_RAIN)
```

### `triangular_membership`

```python
def triangular_membership(x: float, a: float, b: float, c: float) -> float
```
Triangular membership function `Tr(x; a, b, c)` (Salvi et al., Appendix I). `a` = left foot (membership 0), `b` = peak (membership 1), `c` = right foot (membership 0). Returns a value in `[0.0, 1.0]`.

### `trapezoidal_membership`

```python
def trapezoidal_membership(x: float, a: float, b: float, c: float, d: float) -> float
```
Trapezoidal membership function `Tp(x; a, b, c, d)` (Salvi et al., Appendix I). `a` = left foot, `b` = left shoulder (membership reaches 1), `c` = right shoulder (membership starts leaving 1), `d` = right foot. Returns a value in `[0.0, 1.0]`.

### `class PDBreakpoints`

```python
@dataclass
class PDBreakpoints:
    w_min: float
    p25: float
    p50: float
    p75: float
    w_max: float
```
Quartile-calibrated breakpoints for the Precipitation-Deposits (PD) proxy: minimum observed wetness, 25th/50th/75th percentiles, and maximum observed wetness.

```python
def save(self, path: str) -> None
```
Serializes the breakpoints to a JSON file.

```python
@classmethod
def load(cls, path: str) -> "PDBreakpoints"
```
Loads previously calibrated breakpoints from a JSON file.

### `calibrate_pd_breakpoints`

```python
def calibrate_pd_breakpoints(wetness_values: List[float]) -> PDBreakpoints
```
Calibrates PD membership breakpoints from an observed wetness distribution via `np.percentile(wetness_values, [25, 50, 75])` plus `min`/`max`.

### `pd_dry_membership` / `pd_wet_membership` / `pd_puddles_membership`

```python
def pd_dry_membership(wetness: float, bp: PDBreakpoints) -> float
def pd_wet_membership(wetness: float, bp: PDBreakpoints) -> float
def pd_puddles_membership(wetness: float, bp: PDBreakpoints) -> float
```
Membership degrees of "PD is Dry" / "PD is Wet" / "PD is Puddles" (Salvi et al. Table I / Def. 3.2), computed as described above.

### `classify_mu_odd`

```python
def classify_mu_odd(wetness: float, bp: PDBreakpoints) -> Dict[str, float]
```
Computes fuzzy uODD membership degrees for one scene (Salvi et al. Table I), returning a dict mapping each of `MU_ODD_STATES` to its membership degree.

### `defuzzify_mu_odd`

```python
def defuzzify_mu_odd(memberships: Dict[str, float]) -> str
```
Picks the crisp uODD label with the highest membership degree (`max(memberships, key=memberships.get)`) — a max-membership defuzzifier, not a centroid/weighted-average one.

### `compose_scenario_label`

```python
def compose_scenario_label(mu_odd_label: str, traffic_level: str) -> str
```
Combines the weather micro-ODD and traffic-density category into one composite label, e.g. `f"{mu_odd_label}-{traffic_level}"` → `"HeavyRain-Congested"`.

### `classify_dataframe`

```python
def classify_dataframe(
    df: pd.DataFrame,
    pd_breakpoints: PDBreakpoints = None,
    density_thresholds: TrafficDensityThresholds = None,
)
```
Adds fuzzy uODD and composite scenario columns (`mu_odd_no_rain`, `mu_odd_low_rain`, `mu_odd_heavy_rain`, `mu_odd_label`, `scenario_label`) to a features DataFrame that has `wetness`, `vehicle_count`, `num_two_wheelers`, and `num_autorickshaws` columns. If `pd_breakpoints`/`density_thresholds` are `None`, they are calibrated fresh from this DataFrame. Internally calls `traffic_density.classify_dataframe` (imported as `classify_traffic_density`) to obtain the traffic-density columns first. Returns `(df_with_scenario_columns, pd_breakpoints_used, density_thresholds_used)`.

## Key Design Decisions & Edge Cases

- **PD-only rule evaluation is a documented partial application, not a different model.** Salvi et al.'s Table I rules OR-combine a Precipitation (P) term with a Precipitation-Deposits (PD) term; since P has no sensor here, only the PD disjunct is evaluated. The module docstring is explicit that this is legitimate because the rules are OR-combinations.
- **Quartile-calibrated breakpoints, not the paper's physical-unit breakpoints.** As with `traffic_density.py`, the paper's own numeric breakpoints are in units (precipitation rate, sensor-specific wetness units) this project cannot observe, so breakpoints are instead calibrated from the observed `wetness` feature's own distribution — a documented calibration choice, not a value taken from the paper.
- **Division-by-zero guards** in both membership functions (`b == a` or `d == c` short-circuits to `1.0` for that side) prevent `ZeroDivisionError` when breakpoints coincide (e.g. `pd_dry_membership` always passes `a=b=w_min`).
- **Max-membership defuzzification.** `defuzzify_mu_odd` simply returns the single highest-membership label; it does not perform a weighted/centroid defuzzification, and ties are broken by dict iteration order (Python 3.7+ insertion order — `NoRain` before `LowRain` before `HeavyRain`).
- **Depends on `traffic_density.py`**, so its output columns are a strict superset of that module's.

## Dependencies

- Standard library: `json`, `os`, `dataclasses`, `typing`
- External: `numpy`, `pandas`
- Internal: `src.odd.traffic_density` (`DENSITY_LEVELS`, `TrafficDensityThresholds`, `classify_dataframe` imported as `classify_traffic_density`); `src.common.paths` (only inside `__main__`, for `FEATURES_CSV`, `FUZZY_PD_BREAKPOINTS_JSON`, `ensure_output_dirs`)

## Usage Example

```python
import pandas as pd
from src.odd.fuzzy_odd import classify_dataframe

df = pd.read_csv("outputs/final_features.csv")
labeled_df, pd_breakpoints, density_thresholds = classify_dataframe(df)

print(labeled_df[["wetness", "mu_odd_label", "traffic_density_level", "scenario_label"]].head())
print(labeled_df["mu_odd_label"].value_counts())
```

## Running Standalone

```bash
python -m src.odd.fuzzy_odd
```
Loads `outputs/final_features.csv` (`FEATURES_CSV`), classifies every row into fuzzy uODD states and composite scenario labels, saves the calibrated PD breakpoints to `outputs/fuzzy_pd_breakpoints.json` (`FUZZY_PD_BREAKPOINTS_JSON`), and prints the PD breakpoints, the NoRain/LowRain/HeavyRain distribution, and the top-10 composite scenario labels by frequency.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/odd/test_fuzzy_odd.py` covers: triangular membership at its peak/feet and outside its support, trapezoidal membership's plateau and feet, quartile-breakpoint calibration against a known `1..100` sequence, that all three uODD memberships are bounded in `[0, 1]` for a range of wetness values, dry/wet/puddle membership sanity checks at their expected extremes, max-membership defuzzification, and the composite label's string format.
