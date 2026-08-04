# `odd_boundary.py`

**Stage:** 5
**Package:** `src.odd.odd_boundary`

## Purpose

This module implements the core of Stage 5, "ODD Mapping": it formalizes a scene's location in a multi-dimensional "ODD space" and estimates how typical or extreme that location is. It does this in two parts. First, it populates Jiang et al.'s 3-category, 14-element ODD taxonomy (Scenery / Environmental Conditions / Dynamic Entities) with whatever this project's camera-only perception pipeline can actually supply, leaving unsupported fields honestly `None` rather than fabricating values. Second, it fits a small linear Structural Causal Model (SCM) over a hand-specified causal graph of ODD-relevant variables, and separately fits a Gaussian-copula model of the joint density of those variables across the whole dataset. The copula density is then used to classify any given scene as `within`, `near`, or `outside` the dense region of the ODD space the system was characterized on — the central Stage 5 output that Stage 7's `combine_stage_outputs()` consumes as one of its three severity signals.

## Paper / Formula Provenance

Jiang, Pan, Liu, Han, Pan, Li, Pan, "Enhancing Autonomous Vehicle Safety Based on Operational Design Domain Definition, Monitoring, and Functional Degradation," IEEE TIV, Oct 2024.

**Structural Causal Model (Eq. 1).** The paper's general (possibly nonlinear) form is `x_i = f_i(Pa_i, u_i)`. This module implements an explicitly simplified **linear special case**: each equation is fit as an ordinary linear regression of the child variable on its parent variables, and each equation's residual is treated as that variable's noise term `u_i`. Fitting a full nonlinear causal graph (e.g. via a general causal-discovery algorithm) is out of scope; the causal graph itself is also a small, hand-specified DAG (`CAUSAL_DAG`) rather than the output of a causal-discovery procedure — both are documented simplifications, not the paper's general formalism.

**Gaussian-copula joint ODD-space density (Eqs. 5-6):**

```
F(X)     = C(F(X1), ..., F(Xd))
phi_ODD  = c(F(X1), ..., F(Xd)) * f(X1) * ... * f(Xd)
```

Fit via the standard two-step "inference for margins" procedure:
1. A rank-based empirical marginal CDF `F(Xi)` per variable.
2. A KDE estimate of each marginal density `f(Xi)`.
3. A Gaussian copula correlation matrix fit on the normal-score-transformed marginals (i.e. each `F(Xi)` value passed through the standard normal quantile function `Phi^-1`, then correlated).

`odd_density()` evaluates `phi_ODD` at a point by computing the copula density `c(...)` as the ratio of the multivariate normal density at the normal-scores to the product of univariate normal densities at those same normal-scores, then multiplying by the product of the per-variable KDE marginal densities.

**3-category / 14-element ODD taxonomy (Table I).** `SceneryElements`, `EnvironmentalConditionElements`, and `DynamicEntityElements` mirror the paper's Scenery / Environmental Conditions / Dynamic Entities categories. Fields with no corresponding signal in this project's camera-only, single-frame perception pipeline (e.g. traffic-sign/signal recognition, wind intensity, lightning, vehicle speed) are left `None` and documented as such in each dataclass's docstring, rather than fabricated.

**within/near/outside classification** is *not* from the paper. The paper motivates the concept of "distance from the dense region of the ODD space" but does not specify exact cut points; this module's `classify_odd_region()` uses a documented percentile-based operationalization (15th and 50th percentiles of the training set's density distribution) as its own calibration choice.

## Public API

### ODD taxonomy dataclasses (Jiang et al. Table I)

```python
@dataclass
class SceneryElements:
    drivable_area: Optional[float] = None
    road_geometry_curvature: Optional[float] = None
    road_structure_quality: Optional[float] = None
    road_adhesion: Optional[float] = None
    lane_width: Optional[float] = None
    traffic_signs: Optional[float] = None
    traffic_signals: Optional[float] = None
```

```python
@dataclass
class EnvironmentalConditionElements:
    wind_intensity: Optional[float] = None
    lightning: Optional[float] = None
    rainfall_proxy: Optional[float] = None       # from 'wetness'
    fog_density_proxy: Optional[float] = None    # from inverse 'visibility'
    cloudy_proxy: Optional[float] = None          # from inverse 'brightness'
    wetness: Optional[float] = None
```

```python
@dataclass
class DynamicEntityElements:
    road_users_score: Optional[float] = None      # living_things_score + counts
    vehicle_speed: Optional[float] = None
    relative_distance: Optional[float] = None      # lead_vehicle_distance
```

```python
@dataclass
class ODDState:
    scenery: SceneryElements = field(default_factory=SceneryElements)
    environment: EnvironmentalConditionElements = field(default_factory=EnvironmentalConditionElements)
    dynamic_entities: DynamicEntityElements = field(default_factory=DynamicEntityElements)
```
One scene's full ODD taxonomy state across all 3 categories.

### `build_odd_state`

```python
def build_odd_state(row: pd.Series, lane_width_px: Optional[float] = None) -> ODDState
```
Populates an `ODDState` from one feature-extraction row (`final_features.csv` or `full_dataset_labeled.csv`). `lane_width_px` is an optional lane-width estimate from `lane_detection.py` for the same frame (not part of the core 22-feature set). Fields are populated via `row.get(...)`, so any missing column simply yields `None` for that field (e.g. `traffic_signs`, `vehicle_speed` are never populated by this function and stay `None`).

### Structural Causal Model (Eq. 1, linear special case)

```python
@dataclass
class StructuralEquation:
    child: str
    parents: List[str]
    model: LinearRegression
    residual_std: float
```
One fitted linear structural equation `x_i = f_i(Pa_i, u_i)`: `child` is the dependent variable, `parents` its causal parents, `model` the fitted `sklearn.linear_model.LinearRegression` (`child ~ parents`), and `residual_std` the standard deviation of the fitted residuals (the scale of the noise term `u_i`).

```python
CAUSAL_DAG: List[Tuple[str, List[str]]] = [
    ("visibility", ["brightness", "wetness"]),
    ("road_quality", ["drivable_area", "non_drivable_area_score"]),
    ("scene_complexity", ["traffic_density", "living_things_score", "detection_confidence"]),
]
```
A small, hand-specified causal DAG over ODD-relevant variables — a documented modeling choice standing in for a causal-discovery algorithm.

```python
def fit_structural_causal_model(df: pd.DataFrame) -> List[StructuralEquation]
```
Fits the project's linear SCM over `CAUSAL_DAG` from observed data (`df` must contain every variable named in `CAUSAL_DAG`). Returns one fitted `StructuralEquation` per DAG entry.

```python
def counterfactual_prediction(
    equation: StructuralEquation, parent_values: Dict[str, float]
) -> float
```
Predicts a child variable's value given hypothetical parent values. This implements only the "action + prediction" steps of Jiang et al.'s counterfactual reasoning (their Algorithm 1 uses abduction → action → prediction over an SCM); full abduction (solving for this specific observation's exact noise term `u_i`) is **not** performed — this returns `E[x_i | do(Pa_i)]`, the structural equation's point prediction, not a true counterfactual for a specific observed instance.

### Gaussian copula ODD-space density (Eqs. 5-6)

```python
DEFAULT_ODD_VARIABLES = [
    "visibility", "traffic_density", "road_quality",
    "non_drivable_area_score", "living_things_score",
]
```

```python
@dataclass
class ODDCopulaModel:
    variables: List[str]
    sorted_samples: Dict[str, np.ndarray]
    marginal_kdes: Dict[str, "stats.gaussian_kde"]
    correlation_matrix: np.ndarray
    density_percentiles: np.ndarray
```
A fitted Gaussian-copula model: `variables` are the modeled variable names in order; `sorted_samples` are per-variable sorted observed values used for the rank-based empirical CDF; `marginal_kdes` are per-variable fitted `scipy.stats.gaussian_kde` marginal density estimates; `correlation_matrix` is the Gaussian copula's correlation parameter fit on normal-score-transformed marginals; `density_percentiles` is a cached array of density values across the training set, used to calibrate the within/near/outside cut points.

```python
def fit_odd_copula(df: pd.DataFrame, variables: List[str] = None) -> ODDCopulaModel
```
Fits a Gaussian copula over the joint distribution of `variables` (defaults to `DEFAULT_ODD_VARIABLES`) from `df`. Also evaluates and caches `odd_density()` for every row of `df`, sorted, as `density_percentiles`.

```python
def odd_density(model: ODDCopulaModel, x: Dict[str, float]) -> float
```
Evaluates `phi_ODD(x) = c(F(X1),...,F(Xd)) * f(X1) * ... * f(Xd)` (Eqs. 5-6) at a point `x` (a dict mapping each of `model.variables` to a value). Returns a non-negative density value; returns `0.0` if the copula density term's denominator underflows to zero.

```python
def classify_odd_region(model: ODDCopulaModel, x: Dict[str, float]) -> str
```
Classifies a scene as `"within"` / `"near"` / `"outside"` the safe ODD-density region: `"outside"` if density is below the 15th percentile of the training set's density distribution, `"within"` if at or above the 50th percentile, `"near"` otherwise. Documented as an operationalization choice, not a value from the paper.

```python
def save_odd_copula(model: ODDCopulaModel, path: str) -> None
def load_odd_copula(path: str) -> ODDCopulaModel
```
Saves/loads a fitted `ODDCopulaModel` via `joblib`.

### Internal helper

```python
def _empirical_cdf(value: float, sorted_values: np.ndarray) -> float
```
Rank-based empirical CDF `F_hat(x)`: `u = rank / (n + 1)` (the `n+1` denominator keeps `u` strictly inside `(0, 1)`), then clipped to `[1e-6, 1 - 1e-6]` to avoid `±inf` normal quantiles downstream.

## Key Design Decisions & Edge Cases

- **Linear SCM is an explicit simplification.** Jiang et al.'s Eq. 1 allows arbitrary nonlinear `f_i`; this module fits ordinary linear regressions instead, and says so in both the module docstring and the `StructuralEquation`/`fit_structural_causal_model` docstrings. Do not read `fit_structural_causal_model` as a faithful implementation of a nonlinear SCM.
- **`CAUSAL_DAG` is hand-specified, not discovered.** The three causal edges (visibility ← brightness, wetness; road_quality ← drivable_area, non_drivable_area_score; scene_complexity ← traffic_density, living_things_score, detection_confidence) are a documented modeling choice, standing in for a causal-discovery algorithm that is out of scope for this project.
- **`counterfactual_prediction` skips abduction.** It returns the structural equation's point prediction `E[x_i | do(Pa_i)]`, not a full counterfactual for a specific observed instance (which would require solving for that instance's exact noise term first).
- **Honest `None` fields, not fabricated values.** Every ODD taxonomy dataclass documents in its own docstring which fields have no corresponding sensor/feature and are therefore always `None` (e.g. `traffic_signs`, `traffic_signals`, `wind_intensity`, `lightning`, `vehicle_speed`, `road_geometry_curvature`, `road_adhesion`). This mirrors the project's broader "no pothole/water-puddle class" honesty principle described in the top-level README.
- **within/near/outside thresholds (15th/50th percentile) are a calibration choice**, explicitly not derived from the paper, since the paper does not specify exact cut points for "distance from the dense region."
- **Empirical CDF clipping** (`1e-6` to `1 - 1e-6`) prevents `scipy.stats.norm.ppf` from returning `±inf` for values at or beyond the observed sample's extremes — important because `odd_density()` and `classify_odd_region()` are routinely called on out-of-sample (including deliberately extreme/outlier) inputs.
- **`odd_density` returns 0.0 on underflow** rather than raising or returning `NaN`, when the product of per-variable normal marginal densities collapses to zero (can happen for very extreme normal-score values).

## Dependencies

- Standard library: `json`, `os`, `dataclasses`, `typing`
- External: `joblib`, `numpy`, `pandas`, `scipy.stats`, `sklearn.linear_model.LinearRegression`
- Internal: none at import time; `src.common.paths` (`FEATURES_CSV`, `ODD_COPULA_PATH`, `ensure_output_dirs`) only inside `__main__`

`risk_estimation.py` and `odd_classifier.py`'s `combine_stage_outputs()` both depend on this module (`ODDCopulaModel`, `fit_odd_copula`, `odd_density`, `DEFAULT_ODD_VARIABLES`, and the `classify_odd_region()` output contract, respectively).

## Usage Example

```python
import pandas as pd
from src.odd.odd_boundary import (
    fit_structural_causal_model, fit_odd_copula, odd_density,
    classify_odd_region, build_odd_state, DEFAULT_ODD_VARIABLES,
)

df = pd.read_csv("outputs/final_features.csv")

equations = fit_structural_causal_model(df)
for eq in equations:
    print(eq.child, "~", eq.parents, "residual_std=", eq.residual_std)

copula = fit_odd_copula(df, DEFAULT_ODD_VARIABLES)
row = df.iloc[0]
x = {v: row[v] for v in DEFAULT_ODD_VARIABLES}
print(odd_density(copula, x), classify_odd_region(copula, x))

state = build_odd_state(row)
print(state.environment.wetness, state.scenery.traffic_signs)  # value, None
```

## Running Standalone

```bash
python -m src.odd.odd_boundary
```
Loads `outputs/final_features.csv` (`FEATURES_CSV`), fits and prints the linear SCM's equations (coefficients, intercept, residual std per variable), fits the Gaussian-copula ODD-density model and saves it via `joblib` to `models/odd_copula.pkl` (`ODD_COPULA_PATH`), prints the within/near/outside region distribution across the dataset, and prints one sample `ODDState` built from the first row.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/odd/test_odd_boundary.py` covers: that `fit_structural_causal_model` returns one equation per `CAUSAL_DAG` entry with non-negative residual std; that `odd_density` is non-negative; that a typical (median) scene has higher density than a synthetically extreme outlier; that `classify_odd_region` returns a valid label and correctly classifies an extreme outlier as `"outside"`; and that `build_odd_state` correctly populates known fields from a synthetic features row while leaving unsupported fields (`traffic_signs`, `vehicle_speed`) honestly `None`.
