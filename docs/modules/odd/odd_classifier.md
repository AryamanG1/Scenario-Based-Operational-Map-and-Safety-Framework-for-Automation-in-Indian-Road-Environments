# `odd_classifier.py`

**Stage:** 7
**Package:** `src.odd.odd_classifier`

## Purpose

This module implements the core of Stage 7, "Decision System": it turns per-frame perception features into the final ODD operating mode — `Normal`, `Degraded`, or `Takeover` — that the rest of the system (dashboard, `feasibility_map.py`, `main.py`) reports as the pipeline's ultimate output. It has four parts: (A) a rule-based ground-truth labeling function (`assign_mode`) that maps SAE J3016's Normal/Degraded/Takeover concepts onto this project's scaled features; (B) a feature-cleaning pipeline (`FeatureScaler`, `load_and_clean_features`) that normalizes, labels, and prunes the raw feature CSV for training; (C) a RandomForest classifier trained on that cleaned data plus its evaluation/plotting utilities; and (D) an end-to-end single-image inference function (`evaluate_road_scene`) that runs the whole perception-to-decision pipeline on one image. It also provides `combine_stage_outputs()`, the capstone proposal's Stage 7 "combine monitoring output + operational condition status" step, which merges this module's own rule-based mode with Stage 5's ODD-region status and Stage 6's real-time monitoring status into the final decision actually reported downstream.

This is the module every other stage's safety signal ultimately funnels into: Stage 5 (`odd_boundary.classify_odd_region`) and Stage 6 (`perception_monitor.run_perception_monitor`) both feed into `combine_stage_outputs()` here, alongside this module's own `assign_mode()` rule.

## Paper / Formula Provenance

`odd_classifier.py` is **not listed in the README's "Formula provenance" citation table** — none of its logic implements a specific numbered equation from the 10 source papers. Its threshold rule (`assign_mode`) and cross-stage combination rule (`combine_stage_outputs`) are project-specific engineering built directly from the capstone proposal's own Stage 7 description ("Combine monitoring output + operational condition status... select system mode"), operationalizing SAE J3016's conceptual definitions of Normal ADS operation, degraded/cautious operation, and Request-to-Intervene / driver takeover (see `sae_taxonomy.py`, which *is* cited to SAE J3016 Rev. APR2021 in the README table). The specific numeric cutoffs in `assign_mode` (e.g. `detection_confidence > 0.85`, `visibility > 0.45`) and the RandomForest hyperparameters in `train_odd_classifier` are calibration/engineering choices made for this project, not values taken from any paper. `combine_stage_outputs()`'s "most-severe-signal-wins" rule is a standard redundant-channel safety-engineering pattern, not a numbered formula.

Because this module has no equations to reproduce, the logic worth documenting precisely is procedural — see Public API and Key Design Decisions below.

## Public API

### Part A: Rule-based ground-truth labeling

```python
def assign_mode(row: pd.Series) -> str
```
Assigns an ODD mode label to one MinMax-scaled (0-1) feature row. Requires at least `object_presence`, `detection_confidence`, `visibility`, `traffic_density`, `non_drivable_area_score`, `living_things_score`. Returns one of `"Normal"`, `"Degraded"`, `"Takeover"`.

Decision logic (evaluated on scaled `[0,1]` features):
```
has_detections = object_presence > 0
det_conf_ok    = (detection_confidence > 0.85) if has_detections else True

NORMAL if:   det_conf_ok
         and visibility > 0.45
         and traffic_density < 0.4
         and non_drivable_area_score < 0.2
         and living_things_score < 0.2

DEGRADED elif:  ((detection_confidence > 0.5) if has_detections else True)
            and visibility > 0.25

TAKEOVER: otherwise
```

```python
def combine_stage_outputs(base_mode: str, odd_region: str, monitoring_state: str) -> str
```
Combines Stage 5/6 status with the Stage 7 rule-based mode into the final decision. `base_mode` is `assign_mode()`'s output (`"Normal"`/`"Degraded"`/`"Takeover"`); `odd_region` is `odd_boundary.classify_odd_region()`'s output (`"within"`/`"near"`/`"outside"`); `monitoring_state` is `perception_monitor.run_perception_monitor()`'s `.state` (`"Nominal"`/`"Warning"`/`"Critical"`). Returns the final combined mode, one of `"Normal"`, `"Degraded"`, `"Takeover"`. See Key Design Decisions for exactly how this works.

Internal severity-ranking dicts used by `combine_stage_outputs`:
```python
_MODE_SEVERITY = {"Normal": 0, "Degraded": 1, "Takeover": 2}
_ODD_REGION_SEVERITY = {"within": 0, "near": 1, "outside": 2}
_MONITORING_SEVERITY = {"Nominal": 0, "Warning": 1, "Critical": 2}
```

### Part B: Feature cleaning pipeline

```python
class FeatureScaler:
    def __init__(
        self,
        visibility_max: float,
        road_quality_max: float,
        minmax_scaler: MinMaxScaler,
        feature_columns: List[str],
    ) -> None
```
Bundles the brightness/visibility/road_quality normalization constants with a fitted `MinMaxScaler`, so this single artifact can reproduce the exact Part B transform on a single new image's features at inference time (a lone image has no batch to compute a max over).

```python
def transform(self, df: pd.DataFrame) -> pd.DataFrame
```
Applies the fitted normalize (`brightness /= 255.0`, `visibility /= visibility_max`, `road_quality /= road_quality_max`) + MinMax pipeline to new feature rows. `df` must contain at least `self.feature_columns`. Returns a DataFrame restricted to `self.feature_columns`.

```python
def load_and_clean_features(
    csv_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, FeatureScaler, LabelEncoder]
```
Loads, normalizes, labels, and prunes the raw feature CSV (`final_features.csv`, as produced by `feature_extraction.extract_all_features`). Returns `(X, df_cleaned, y, feature_scaler, label_encoder)`:
- `X`: pruned + scaled feature columns (RandomForest input).
- `df_cleaned`: `X` plus `'mode'`/`'mode_encoded'` columns — what gets saved to `final_features_cleaned.csv`.
- `y`: `mode_encoded` labels, aligned with `X`'s row order.
- `feature_scaler`: fitted `FeatureScaler` for single-image inference.
- `label_encoder`: fitted `LabelEncoder` (use `.classes_` for reporting, since its alphabetical fit order is **not** `Normal=0/Degraded=1/Takeover=2`).

### Part C: Train ODD classifier

```python
def train_odd_classifier(
    X: pd.DataFrame, y: pd.Series, model_path: str = "odd_classifier.pkl"
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, np.ndarray]
```
Trains and evaluates a `RandomForestClassifier(n_estimators=200, random_state=42)` on an 80/20 (`test_size=0.2`) train/test split. Warns (does not fail) if any encoded class has fewer than 2 samples. Prints accuracy, weighted F1-score, and a classification report; saves the model via `joblib.dump` to `model_path`. Returns `(model, X_test, y_test, y_pred)`.

```python
def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    save_dir: str = PLOTS_DIR,
) -> None
```
Plots and saves a confusion-matrix heatmap (`confusion_matrix.png` in `save_dir`), with axis labels correctly ordered via `label_encoder.classes_` / `label_encoder.transform(label_encoder.classes_)` rather than assuming a fixed class order.

```python
def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
    save_dir: str = PLOTS_DIR,
    top_n: int = 10,
) -> None
```
Plots and saves the top-`top_n` RandomForest feature importances as a horizontal bar chart (`feature_importance.png` in `save_dir`).

### Part D: End-to-end single-image inference

```python
def evaluate_road_scene(
    image_path: str,
    segnet_model: nn.Module,
    yolo_model: YOLO,
    odd_classifier: RandomForestClassifier,
    scaler: FeatureScaler,
    label_encoder: LabelEncoder,
    device: torch.device = DEVICE,
    save_path: Optional[str] = None,
) -> str
```
Runs the full perception-to-ODD pipeline on a single road image: reads and resizes the image (`IMAGE_SIZE`), runs SegNet inference to get a segmentation mask, runs YOLO detection, computes features (`compute_features`), scales them via `scaler.transform`, predicts a mode via `odd_classifier`, and decodes it via `label_encoder`. Optionally saves a 3-panel visualization figure (original image + YOLO bounding boxes, SegNet mask, feature summary + verdict text) to `save_path`. Returns the predicted mode string (`"Normal"`, `"Degraded"`, or `"Takeover"`). **Raises `FileNotFoundError`** if `image_path` cannot be read by `cv2.imread`.

Supporting constants:
```python
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_VERDICT_EMOJI = {"Normal": "🟢 NORMAL", "Degraded": "🟠 DEGRADED", "Takeover": "🔴 TAKEOVER"}
_BBOX_COLORS = {"vehicle": (0, 255, 0), "two_wheeler": (0, 0, 255), "pedestrian": (255, 0, 0)}
```

## Key Design Decisions & Edge Cases

### The pickling gotcha: module `__name__` vs `__main__`

Right after `FeatureScaler` is defined, the module does:
```python
FeatureScaler.__module__ = "src.odd.odd_classifier"
```
The problem this fixes: when a `.py` file is run directly (e.g. `python -m src.odd.odd_classifier`), Python sets that running module's `__name__` to `"__main__"`. Any class defined in that file would therefore, by default, get pickled with `__module__ == "__main__"`. A pickle saved that way is only loadable by a process whose *own* `"__main__"` module happens to define an identical `FeatureScaler` class — which is not true for `main.py` or `full_dataset_pipeline.py`, which instead do `import src.odd.odd_classifier` and expect `src.odd.odd_classifier.FeatureScaler`. Without the fix, `feature_scaler.pkl` saved by running this file directly would be unloadable from any other script's process. The line above forces the class's real, importable dotted path onto `__module__` unconditionally, regardless of how this file was invoked, so every `FeatureScaler` pickle is loadable the same way everywhere.

That fix is completed in the `__main__` block:
```python
sys.modules.setdefault("src.odd.odd_classifier", sys.modules[__name__])
```
This is necessary because pickle's own load-time consistency check requires `sys.modules[obj.__module__]` to actually *contain* the class being unpickled. Overriding `__module__` alone is not enough if `sys.modules` never has a `"src.odd.odd_classifier"` key pointing at the running module (which is exactly the situation when this file is executed as `__main__` — Python registers it in `sys.modules` under `"__main__"`, not under its real dotted path). This line aliases the running module into `sys.modules` under its real importable name too, so pickle's check passes. **Together**, the class-attribute override and the `sys.modules` alias make `FeatureScaler` pickle correctly and load correctly regardless of whether this file was run as a script or imported as a module.

### `combine_stage_outputs()`: cross-stage signal fusion

Three independent status vocabularies — Stage 7's own `base_mode` (`Normal`/`Degraded`/`Takeover`), Stage 5's `odd_region` (`within`/`near`/`outside`), and Stage 6's `monitoring_state` (`Nominal`/`Warning`/`Critical`) — are each mapped onto a shared severity scale of `{0, 1, 2}` via the three `_..._SEVERITY` dicts. The function takes the **maximum** severity across all three signals, then maps that severity value back to a mode name via `next(mode for mode, sev in _MODE_SEVERITY.items() if sev == final_severity)`. This is a "most-severe-signal-wins" redundant-channel pattern: any one of the three upstream signals can escalate the system toward a more conservative mode (e.g. Stage 6 detecting `Critical` monitoring conditions forces `Takeover` even if `assign_mode()` and the ODD region both say everything is fine), but no signal can override or downgrade another signal's legitimate escalation. This is exercised directly by `test_combine_stage_outputs_most_severe_wins`'s parametrized cases, including `("Degraded", "outside", "Critical") -> "Takeover"` and `("Takeover", "within", "Nominal") -> "Takeover"` (Stage 7's own rule alone is enough to force Takeover, regardless of the other two signals being nominal).

### `assign_mode()`'s "empty road" bug fix

The docstring calls out a specific, previously-buggy edge case: when `object_presence == 0` (no detections at all), `detection_confidence` is trivially `0.0` (there is nothing to be confident about). Naively checking `detection_confidence > 0.85` in that situation would incorrectly force `Takeover` on an empty, otherwise-clear road. The fix: `det_conf_ok` (used in the Normal branch) and the analogous confidence check in the Degraded branch are both set to `True` whenever `has_detections` is `False` — i.e., "no detections" is treated as trivially confidence-satisfying, not confidence-failing. `test_assign_mode_empty_road_is_normal_not_takeover` verifies this directly.

### `assign_mode()`'s Degraded branch does not re-check road-quality features

The Normal branch checks `non_drivable_area_score` and `living_things_score`, but the Degraded branch checks only confidence and visibility. Consequently a scene with high `non_drivable_area_score`/`living_things_score` but otherwise fine confidence and visibility lands in `Degraded`, not `Takeover` — the test suite's own comment confirms this is "the spec's actual rule, not a bug."

### `load_and_clean_features()`: label before prune

`assign_mode()` is deliberately run on the **scaled-but-unpruned** dataframe (`df_scaled`), before the variance/correlation pruning step, not after. The reason: a freshly computed feature might end up with variance `< 0.001` or be highly correlated (`> 0.9`) with another column and get dropped during pruning, but `assign_mode()` needs every column it references (`object_presence`, `detection_confidence`, `visibility`, `traffic_density`, `non_drivable_area_score`, `living_things_score`) to still exist at labeling time. Missing values in the raw CSV are handled with a blanket `df.fillna(0)` before any normalization.

### Pruning thresholds are engineering choices

Low-variance columns are dropped below a variance threshold of `0.001`; highly-correlated columns are dropped above an absolute correlation of `0.9` (upper-triangle check via `np.triu`). Both thresholds are project-specific calibration choices, not paper-derived.

### `LabelEncoder`'s class order is alphabetical, not severity order

`label_encoder.classes_` is sorted alphabetically (`Degraded`, `Normal`, `Takeover`), **not** `Normal=0/Degraded=1/Takeover=2`. Every consumer that needs to report or plot by class must use `label_encoder.classes_`/`label_encoder.transform(label_encoder.classes_)` rather than assuming an order — `plot_confusion_matrix` does this correctly.

### `train_odd_classifier` degenerate-data warning

If any encoded class has fewer than 2 samples, the function prints a warning but still proceeds — `train_test_split` or training may behave degenerately (e.g. a class entirely missing from the test split) in that case; this is a documented soft failure, not a hard error.

## Dependencies

- Standard library: `os`, `typing`
- External: `cv2` (OpenCV), `joblib`, `matplotlib.pyplot`, `numpy`, `pandas`, `seaborn`, `torch`, `torch.nn`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.metrics` (`accuracy_score`, `classification_report`, `confusion_matrix`, `f1_score`), `sklearn.model_selection.train_test_split`, `sklearn.preprocessing` (`LabelEncoder`, `MinMaxScaler`), `ultralytics.YOLO`
- Internal: `src.common.paths.PLOTS_DIR`; `src.perception.data_pipeline.IMAGE_SIZE`; `src.perception.feature_extraction` (`PEDESTRIANS`, `TWO_WHEELERS`, `VEHICLES`, `compute_features`, `run_detection`); `src.common.paths` (`FEATURES_CLEANED_CSV`, `FEATURES_CSV`, `FEATURE_SCALER_PATH`, `ODD_CLASSIFIER_PATH`, `ensure_output_dirs`, only inside `__main__`)

Note: this module does **not** import `odd_boundary.py` or `perception_monitor.py` directly — `combine_stage_outputs()` is a pure string-in/string-out function that expects their *outputs* as plain string arguments, so the caller (e.g. `main.py`, `feasibility_map.py`) is responsible for actually invoking `odd_boundary.classify_odd_region()` and `perception_monitor.run_perception_monitor()` and passing their results in.

## Usage Example

```python
# Lightweight usage: rule + cross-stage combination (no heavy ML deps needed)
import pandas as pd
from src.odd.odd_classifier import assign_mode, combine_stage_outputs

row = pd.Series({
    "object_presence": 1, "detection_confidence": 0.9, "visibility": 0.5,
    "traffic_density": 0.1, "non_drivable_area_score": 0.05, "living_things_score": 0.05,
})
base_mode = assign_mode(row)  # "Normal"

final_mode = combine_stage_outputs(
    base_mode=base_mode, odd_region="near", monitoring_state="Nominal"
)  # "Degraded" -- odd_region alone escalates past the Normal base_mode

# Full training pipeline
from src.odd.odd_classifier import load_and_clean_features, train_odd_classifier

X, df_cleaned, y, feature_scaler, label_encoder = load_and_clean_features("outputs/final_features.csv")
model, X_test, y_test, y_pred = train_odd_classifier(X, y, model_path="models/odd_classifier.pkl")
```

## Running Standalone

```bash
python -m src.odd.odd_classifier
```
Loads `outputs/final_features.csv` (`FEATURES_CSV`), cleans and labels it via `load_and_clean_features`, saves the cleaned+labeled CSV to `outputs/final_features_cleaned.csv` (`FEATURES_CLEANED_CSV`), prints the Normal/Degraded/Takeover mode distribution, trains the RandomForest classifier (printing accuracy/F1/classification report and saving it to `models/odd_classifier.pkl` via `ODD_CLASSIFIER_PATH`), saves the confusion matrix and feature importance plots, and saves the fitted `FeatureScaler` to `models/feature_scaler.pkl` (`FEATURE_SCALER_PATH`). Also performs the `sys.modules.setdefault(...)` aliasing described above so that the saved `feature_scaler.pkl` is loadable from other scripts.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/odd/test_odd_classifier.py` covers: `assign_mode()`'s Normal/Degraded/Takeover branch logic, including the empty-road-is-not-Takeover fix and the nuance that high `non_drivable_area_score`/`living_things_score` alone (with good confidence/visibility) lands in Degraded rather than Takeover; and `combine_stage_outputs()`'s "most-severe-signal-wins" behavior across a parametrized set of `(base_mode, odd_region, monitoring_state)` combinations, including cases where each of the three signals is individually the one forcing escalation.
