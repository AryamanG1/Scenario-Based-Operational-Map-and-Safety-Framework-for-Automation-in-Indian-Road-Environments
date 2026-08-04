# `feasibility_map.py`

**Stage:** Cross-cutting (README lists it without a stage number; it is the pipeline-diagram's Stage 7 "Scenario-Based Mapping & Automation Feasibility" deliverable)
**Package:** `src.decision.feasibility_map`

## Purpose

This module is the project's final synthesis step: it ties every earlier stage's per-scene outputs together into one row per image, directly implementing the capstone proposal's explicit deliverable — "Classify road segments as automation-safe or unsafe and construct a feasibility map" (Work Plan item 9). For every scene it combines Stage 3's traffic-density category, Stage 4's fuzzy micro-ODD and composite scenario label, Stage 5's ODD-region status, and Stage 7's Normal/Degraded/Takeover decision into one `FeasibilityRecord`, plus a 0-100 `odd_score` and an L1/L2/L3 automation-feasibility band.

It exists to produce the artifact the dashboard consumes: `build_feasibility_map()` generates the full per-scene DataFrame, and `export_pipeline_stats_js()` writes an aggregate summary the dashboard's "Real Pipeline Results" panel reads directly.

## Paper / Formula Provenance

This module is a cross-stage combination layer rather than a direct implementation of a single cited paper formula — it calls into the already-cited Stage 3/4/5/7 modules (`traffic_density.py`, `fuzzy_odd.py`, `odd_boundary.py`, `odd_classifier.py`) and combines their outputs. Its own `compute_odd_score()` formula is explicitly **not** from a research paper: the module docstring states it "mirrors dashboard/app.js's `calculateODDScore()` formula, applied here to this project's real, MinMax-scaled pipeline features rather than the dashboard's synthetic demo data" — i.e. it reproduces this project's own dashboard heuristic on real pipeline output, not a formula sourced from one of the 10 cited papers.

The L1/L2/L3 automation-feasibility labels are also this project's own scenario-classification concept — the docstring is careful to distinguish them from the project's fixed SAE Level 2 system commitment (`sae_taxonomy.py`): "a segment can be labeled 'L3-feasible' (favorable conditions) while the deployed system itself always operates at Level 2 per the proposal."

## Public API

```python
@dataclass
class FeasibilityRecord:
    traffic_density_level: str
    scenario_label: str
    mu_odd_label: str
    odd_region: str
    final_mode: str
    odd_score: float
    automation_level: str
    automation_feasible: bool
```
One scene's complete cross-stage feasibility summary. `traffic_density_level` — Stage 3 output. `scenario_label` — Stage 4 composite scenario label. `mu_odd_label` — Stage 4 fuzzy micro-ODD label. `odd_region` — Stage 5 within/near/outside status. `final_mode` — Stage 7 combined Normal/Degraded/Takeover decision. `odd_score` — 0-100 automation-readiness score. `automation_level` — "L1"/"L2"/"L3" road-segment feasibility band. `automation_feasible` — `True` iff `final_mode != "Takeover"`.

```python
def compute_odd_score(scaled_row: pd.Series) -> float
```
Computes a 0-100 automation-readiness score from MinMax-scaled features. Starts at `100.0` and subtracts weighted penalties:
- `(1 - visibility) * 30`
- `(1 - brightness) * 10`
- `traffic_density * 25`
- `wetness * 15`
- `non_drivable_area_score * 10`
- `(1 - detection_confidence) * 10`

Requires `scaled_row` to include (at minimum) `visibility`, `brightness`, `traffic_density`, `wetness`, `non_drivable_area_score`, `detection_confidence`, all already MinMax-scaled to `[0, 1]`. Clamped to `[0.0, 100.0]`.

```python
def automation_level_from_score(score: float) -> str
```
Maps a `compute_odd_score()` output to a feasibility band: `"L3"` if `score >= 70`, `"L2"` if `40 <= score < 70`, else `"L1"`.

```python
def build_feasibility_map(
    features_df: pd.DataFrame,
    feature_scaler: FeatureScaler,
    odd_variables=DEFAULT_ODD_VARIABLES,
    default_monitoring_state: str = "Nominal",
) -> pd.DataFrame
```
Builds the full Scenario-Based Feasibility Map. Args: `features_df` — a features DataFrame (`final_features.csv` or `full_dataset_labeled.csv`-shaped); `feature_scaler` — a fitted `FeatureScaler` (`odd_classifier.py`), used to produce the scaled features `assign_mode()`/`compute_odd_score()` require; `odd_variables` — which columns to fit the Stage 5 ODD copula over (defaults to `odd_boundary.DEFAULT_ODD_VARIABLES`); `default_monitoring_state` — the Stage 6 status assumed for every row (see below).

Internally: classifies traffic density (`classify_traffic`) and scenario (`classify_scenario`) for every row, fits an ODD copula (`fit_odd_copula`) over `features_df`, MinMax-scales `features_df` via `feature_scaler.transform()`, then per-row computes `base_mode` (`assign_mode`), `odd_region` (`classify_odd_region`), `final_mode` (`combine_stage_outputs(base_mode, odd_region, default_monitoring_state)`), and `score` (`compute_odd_score`), assembling a `FeasibilityRecord` for each. Returns a copy of `features_df` with `traffic_density_level`, `scenario_label`, `mu_odd_label`, `odd_region`, `final_mode`, `odd_score`, `automation_level`, `automation_feasible` columns appended.

```python
def export_pipeline_stats_js(feasibility_df: pd.DataFrame, output_path: str) -> None
```
Exports aggregate feasibility-map stats as a **JavaScript** data file (not JSON) for the dashboard, writing:
```js
// Auto-generated by feasibility_map.py -- real IDD-Lite pipeline results.
const PIPELINE_STATS = { ... };
```
where the object contains `num_scenes`, `final_mode_counts`, `automation_level_counts`, `odd_region_counts`, `mu_odd_counts`, `traffic_density_counts` (each a value-count dict), plus `mean_odd_score` and `fraction_automation_feasible` (floats). Args: `feasibility_df` — output of `build_feasibility_map()`; `output_path` — destination `.js` file path (e.g. `dashboard/pipeline_stats.js`).

**Why `.js` and not `.json`:** the docstring is explicit: the dashboard (`dashboard/index.html`) is designed to be opened directly over `file://` with zero local server, and browsers block `fetch()`-ing a local JSON file under `file://` due to CORS restrictions, whereas a plain `<script src="...">` tag is not blocked. Emitting a `const PIPELINE_STATS = {...};` JS literal sidesteps the restriction entirely, mirroring how `dashboard/app.js` itself inlines its demo data.

## Key Design Decisions & Edge Cases

- **Stage 6 is deliberately NOT run per-row.** The module docstring explains that `perception_monitor.py`'s true monitoring requires drawing several perturbed pseudo-frames and running SegNet+YOLO on each, which is too expensive to precompute for an entire dataset in a batch map. Instead, `build_feasibility_map()` accepts a `default_monitoring_state` (default `"Nominal"`) applied uniformly to every row. This is called out explicitly in the README's "Known scope limitations": "Full-dataset feasibility maps assume a 'Nominal' Stage 6 status per row rather than running the (expensive) per-frame perturbation-sequence monitor at batch scale; call `perception_monitor.py` directly for a single scene's true monitoring state."
- **`.js` export instead of `.json`** is a deliberate CORS workaround for the file://-opened dashboard, not an accidental inconsistency — see the Public API section above.
- **L1/L2/L3 vs. SAE Level 2 are two distinct concepts that share similar-looking labels.** The docstring warns readers not to conflate them: L1/L2/L3 is a per-scene, per-road-segment automation-feasibility *ceiling* (how favorable are conditions right now), while this project's actual deployed system always operates at the fixed SAE Level 2 (`sae_taxonomy.py`) regardless of what a segment's feasibility band says.
- **`compute_odd_score` is a heuristic weighted-penalty formula**, not derived from a cited paper — it mirrors the dashboard's synthetic-demo scoring function, now applied to real scaled pipeline features. The six penalty weights (30/10/25/15/10/10, summing to 100) were chosen to weight visibility and traffic density most heavily.
- **`build_feasibility_map` requires a pre-fitted `FeatureScaler`** (produced by `odd_classifier.py`) — this module doesn't fit its own scaler, so `odd_classifier.py` must have been run at least once first to produce `feature_scaler.pkl`.

## Dependencies

**Internal:** `src.odd.fuzzy_odd.classify_dataframe` (aliased `classify_scenario`), `src.odd.odd_boundary` (`DEFAULT_ODD_VARIABLES`, `classify_odd_region`, `fit_odd_copula`, `odd_density`), `src.odd.odd_classifier` (`FeatureScaler`, `assign_mode`, `combine_stage_outputs`), `src.odd.traffic_density.classify_dataframe` (aliased `classify_traffic`). The `if __name__ == "__main__":` block additionally imports `src.common.paths` (`FEASIBILITY_MAP_CSV`, `FEATURES_CSV`, `FEATURE_SCALER_PATH`, `PIPELINE_STATS_JS`, `ensure_output_dirs`).

**External:** `json`, `os` (stdlib); `pandas`; `joblib` (in the standalone block, to load the saved `FeatureScaler`).

## Usage Example

```python
import joblib
import pandas as pd
from src.decision.feasibility_map import build_feasibility_map, export_pipeline_stats_js

features_df = pd.read_csv("outputs/final_features.csv")
scaler = joblib.load("models/feature_scaler.pkl")

feasibility_df = build_feasibility_map(features_df, scaler)
feasibility_df.to_csv("outputs/feasibility_map.csv", index=False)

export_pipeline_stats_js(feasibility_df, "dashboard/pipeline_stats.js")
```

## Running Standalone

```bash
python -m src.decision.feasibility_map
```

Loads `outputs/final_features.csv` and the saved `FeatureScaler`, builds the full feasibility map via `build_feasibility_map()`, saves it to `outputs/feasibility_map.csv`, exports `dashboard/pipeline_stats.js` via `export_pipeline_stats_js()`, then prints the automation-level distribution, final-mode distribution, and the overall fraction of scenes that are automation-feasible.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/decision/test_feasibility_map.py` covers: `compute_odd_score` returns near-100 for a best-case row and exactly `0.0` for a worst-case row; `compute_odd_score` stays within `[0, 100]` across 20 randomized rows; `automation_level_from_score` band boundaries (90/70 → L3, 55/40 → L2, 20 → L1); and an end-to-end `test_build_feasibility_map_end_to_end` (using a `real_features_df` fixture, skipped if `models/feature_scaler.pkl` doesn't exist) that builds a feasibility map for 15 real rows and checks the expected columns are present and `odd_score` stays within `[0, 100]`.
