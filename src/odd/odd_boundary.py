"""Stage 5: ODD Mapping -- Structural Causal Model + Gaussian copula boundary.

Implements Jiang et al. 2024's ("Enhancing Autonomous Vehicle Safety Based
on Operational Design Domain Definition, Monitoring, and Functional
Degradation", IEEE TIV) ODD-space formalism:

- The general Structural Causal Model form x_i = f_i(Pa_i, u_i) (Eq. 1),
  implemented here as a small set of fitted LINEAR structural equations --
  an explicitly simplified special case of the paper's general (possibly
  nonlinear) SCM, since fitting a full nonlinear causal graph is out of
  scope. Each equation's residual is that variable's noise term u_i.
- The Gaussian-copula joint ODD-space density (Eqs. 5-6):
      F(X) = C(F(X1), ..., F(Xd))
      phi_ODD = c(F(X1), ..., F(Xd)) * f(X1) * ... * f(Xd)
  fit via the standard two-step "inference for margins" procedure: rank-
  based empirical marginal CDFs F(Xi), a KDE estimate of each marginal
  density f(Xi), and a Gaussian copula correlation matrix fit on the
  normal-score-transformed marginals.
- Jiang et al.'s 3-category/14-element ODD taxonomy (their Table I:
  Scenery / Environmental Conditions / Dynamic Entities), populated with
  whichever elements this project's perception pipeline actually produces
  -- fields with no corresponding sensor/feature are left None rather than
  fabricated, and documented as such.

within/near/outside classification is a percentile-based operationalization
of "distance from the dense region of the fitted ODD space" -- the paper
motivates the concept but doesn't specify exact cut points, so these are
documented calibration choices, not values taken from the paper.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

# --------------------------------------------------------------------------
# Jiang et al. 2024, Table I: 3-category / 14-element ODD taxonomy
# --------------------------------------------------------------------------


@dataclass
class SceneryElements:
    """Scenery category (Jiang et al. Table I).

    Fields left None have no corresponding signal in this project's
    camera-only, single-frame perception pipeline (e.g. no traffic-sign/
    signal recognition model is implemented).
    """

    drivable_area: Optional[float] = None
    road_geometry_curvature: Optional[float] = None
    road_structure_quality: Optional[float] = None
    road_adhesion: Optional[float] = None
    lane_width: Optional[float] = None
    traffic_signs: Optional[float] = None
    traffic_signals: Optional[float] = None


@dataclass
class EnvironmentalConditionElements:
    """Environmental Conditions category (Jiang et al. Table I).

    Wind intensity and lightning have no camera-derived proxy and are left
    None; rainfall/fog/cloudiness are approximated from image statistics.
    """

    wind_intensity: Optional[float] = None
    lightning: Optional[float] = None
    rainfall_proxy: Optional[float] = None  # from 'wetness'
    fog_density_proxy: Optional[float] = None  # from inverse 'visibility'
    cloudy_proxy: Optional[float] = None  # from inverse 'brightness'
    wetness: Optional[float] = None


@dataclass
class DynamicEntityElements:
    """Dynamic Entities category (Jiang et al. Table I).

    Vehicle speed is not observable from a single static frame and is
    left None.
    """

    road_users_score: Optional[float] = None  # living_things_score + counts
    vehicle_speed: Optional[float] = None
    relative_distance: Optional[float] = None  # lead_vehicle_distance


@dataclass
class ODDState:
    """One scene's full ODD taxonomy state (Jiang et al. Table I, 3 categories)."""

    scenery: SceneryElements = field(default_factory=SceneryElements)
    environment: EnvironmentalConditionElements = field(
        default_factory=EnvironmentalConditionElements
    )
    dynamic_entities: DynamicEntityElements = field(default_factory=DynamicEntityElements)


def build_odd_state(row: pd.Series, lane_width_px: Optional[float] = None) -> ODDState:
    """Populates an ODDState from one feature-extraction row.

    Args:
        row: A row from final_features.csv (or full_dataset_labeled.csv).
        lane_width_px: Optional lane-width estimate from lane_detection.py,
            for this same frame (not part of the core 22-feature set).

    Returns:
        An ODDState with every field this pipeline can actually supply.
    """
    scenery = SceneryElements(
        drivable_area=row.get("drivable_area"),
        road_structure_quality=row.get("road_quality"),
        lane_width=lane_width_px,
    )
    environment = EnvironmentalConditionElements(
        rainfall_proxy=row.get("wetness"),
        fog_density_proxy=(1.0 / (row.get("visibility", 0.0) + 1.0)),
        cloudy_proxy=(255.0 - row.get("brightness", 0.0)) / 255.0,
        wetness=row.get("wetness"),
    )
    dynamic_entities = DynamicEntityElements(
        road_users_score=row.get("living_things_score"),
        relative_distance=row.get("lead_vehicle_distance"),
    )
    return ODDState(scenery=scenery, environment=environment, dynamic_entities=dynamic_entities)


# --------------------------------------------------------------------------
# Structural Causal Model (Jiang et al. Eq. 1, linear special case)
# --------------------------------------------------------------------------


@dataclass
class StructuralEquation:
    """One fitted linear structural equation x_i = f_i(Pa_i, u_i).

    Attributes:
        child: The dependent variable name.
        parents: The causal parent variable names.
        model: A fitted sklearn LinearRegression: child ~ parents.
        residual_std: Standard deviation of the fitted residuals, i.e. the
            scale of the noise term u_i.
    """

    child: str
    parents: List[str]
    model: LinearRegression
    residual_std: float


# A small, hand-specified causal DAG over ODD-relevant variables: each
# entry is (child, [parents]). This is a documented modeling choice (e.g.
# "visibility is influenced by brightness and wetness") standing in for a
# causal-discovery algorithm, which is out of scope here.
CAUSAL_DAG: List[Tuple[str, List[str]]] = [
    ("visibility", ["brightness", "wetness"]),
    ("road_quality", ["drivable_area", "non_drivable_area_score"]),
    ("scene_complexity", ["traffic_density", "living_things_score", "detection_confidence"]),
]


def fit_structural_causal_model(df: pd.DataFrame) -> List[StructuralEquation]:
    """Fits the project's linear SCM (Eq. 1) over CAUSAL_DAG from observed data.

    Args:
        df: A features DataFrame containing every variable named in CAUSAL_DAG.

    Returns:
        A list of fitted StructuralEquation instances, one per DAG entry.
    """
    equations = []
    for child, parents in CAUSAL_DAG:
        X = df[parents].to_numpy()
        y = df[child].to_numpy()
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        equations.append(
            StructuralEquation(
                child=child, parents=parents, model=model, residual_std=float(residuals.std())
            )
        )
    return equations


def counterfactual_prediction(
    equation: StructuralEquation, parent_values: Dict[str, float]
) -> float:
    """Predicts a child variable's value given parent values (abduction skipped).

    This implements the "action + prediction" steps of Jiang et al.'s
    counterfactual reasoning (their Algorithm 1 uses abduction -> action ->
    prediction over an SCM): given a hypothetical parent assignment, predict
    the expected child value under the fitted structural equation. Full
    abduction (solving for this specific observation's exact noise term u_i)
    is not performed here -- this returns E[x_i | do(Pa_i)], the structural
    equation's point prediction.

    Args:
        equation: A fitted StructuralEquation.
        parent_values: A dict mapping each of equation.parents to a value.

    Returns:
        The predicted child value.
    """
    x = np.array([[parent_values[p] for p in equation.parents]])
    return float(equation.model.predict(x)[0])


# --------------------------------------------------------------------------
# Gaussian copula ODD-space density (Jiang et al. Eqs. 5-6)
# --------------------------------------------------------------------------

DEFAULT_ODD_VARIABLES = [
    "visibility",
    "traffic_density",
    "road_quality",
    "non_drivable_area_score",
    "living_things_score",
]


@dataclass
class ODDCopulaModel:
    """A fitted Gaussian-copula model of the joint ODD-space density.

    Attributes:
        variables: The variable names this model was fit over, in order.
        sorted_samples: Per-variable sorted observed values, used for the
            rank-based empirical CDF F(Xi).
        marginal_kdes: Per-variable fitted gaussian_kde, the marginal
            density estimate f(Xi).
        correlation_matrix: The Gaussian copula's correlation parameter,
            fit on the normal-score-transformed marginals.
        density_percentiles: Cached density values across the training
            set, used to calibrate within/near/outside cut points.
    """

    variables: List[str]
    sorted_samples: Dict[str, np.ndarray]
    marginal_kdes: Dict[str, "stats.gaussian_kde"]
    correlation_matrix: np.ndarray
    density_percentiles: np.ndarray


def _empirical_cdf(value: float, sorted_values: np.ndarray) -> float:
    """Rank-based empirical CDF F_hat(x), clipped away from {0, 1}.

    Args:
        value: The value to evaluate.
        sorted_values: The training sample, pre-sorted ascending.

    Returns:
        F_hat(value) in (0, 1) (clipped to avoid +/-inf normal quantiles).
    """
    n = len(sorted_values)
    rank = np.searchsorted(sorted_values, value, side="right")
    u = rank / (n + 1)  # (n+1) denominator keeps u strictly inside (0, 1)
    return float(np.clip(u, 1e-6, 1 - 1e-6))


def fit_odd_copula(df: pd.DataFrame, variables: List[str] = None) -> ODDCopulaModel:
    """Fits a Gaussian copula over the joint distribution of ODD variables.

    Args:
        df: A features DataFrame containing every column in `variables`.
        variables: Which columns to model jointly; defaults to
            DEFAULT_ODD_VARIABLES.

    Returns:
        A fitted ODDCopulaModel.
    """
    variables = variables or DEFAULT_ODD_VARIABLES

    sorted_samples = {v: np.sort(df[v].to_numpy()) for v in variables}
    marginal_kdes = {v: stats.gaussian_kde(df[v].to_numpy()) for v in variables}

    normal_scores = np.column_stack(
        [
            stats.norm.ppf(
                [_empirical_cdf(x, sorted_samples[v]) for x in df[v].to_numpy()]
            )
            for v in variables
        ]
    )
    correlation_matrix = np.corrcoef(normal_scores, rowvar=False)

    model = ODDCopulaModel(
        variables=variables,
        sorted_samples=sorted_samples,
        marginal_kdes=marginal_kdes,
        correlation_matrix=correlation_matrix,
        density_percentiles=np.array([]),
    )

    densities = np.array([odd_density(model, dict(zip(variables, row))) for row in df[variables].to_numpy()])
    model.density_percentiles = np.sort(densities)
    return model


def odd_density(model: ODDCopulaModel, x: Dict[str, float]) -> float:
    """Evaluates phi_ODD(x) = c(F(X1),...,F(Xd)) * f(X1) * ... * f(Xd) (Eqs. 5-6).

    Args:
        model: A fitted ODDCopulaModel.
        x: A dict mapping each of model.variables to an observed value.

    Returns:
        The joint ODD-space density at x (non-negative; 0.0 if the copula
        density term underflows to zero).
    """
    z = np.array(
        [
            stats.norm.ppf(_empirical_cdf(x[v], model.sorted_samples[v]))
            for v in model.variables
        ]
    )

    copula_density = stats.multivariate_normal.pdf(
        z, mean=np.zeros(len(z)), cov=model.correlation_matrix
    )
    marginal_normal_density = np.prod(stats.norm.pdf(z))
    if marginal_normal_density <= 0:
        return 0.0
    copula_c = copula_density / marginal_normal_density

    marginal_f = np.prod([model.marginal_kdes[v].evaluate([x[v]])[0] for v in model.variables])
    return float(copula_c * marginal_f)


def classify_odd_region(model: ODDCopulaModel, x: Dict[str, float]) -> str:
    """Classifies a scene as within/near/outside the safe ODD-density region.

    Calibration: 'outside' if the scene's density falls below the 15th
    percentile of the training set's density distribution; 'within' if at
    or above the 50th percentile; 'near' otherwise. These cut points are a
    documented operationalization choice, not values taken from the paper.

    Args:
        model: A fitted ODDCopulaModel (with density_percentiles populated).
        x: A dict mapping each of model.variables to an observed value.

    Returns:
        One of "within", "near", "outside".
    """
    density = odd_density(model, x)
    p15 = np.percentile(model.density_percentiles, 15)
    p50 = np.percentile(model.density_percentiles, 50)

    if density < p15:
        return "outside"
    if density < p50:
        return "near"
    return "within"


def save_odd_copula(model: ODDCopulaModel, path: str) -> None:
    """Saves a fitted ODDCopulaModel via joblib.

    Args:
        model: The fitted model.
        path: Destination file path.
    """
    joblib.dump(model, path)


def load_odd_copula(path: str) -> ODDCopulaModel:
    """Loads a fitted ODDCopulaModel via joblib.

    Args:
        path: Source file path, as written by save_odd_copula().

    Returns:
        The loaded ODDCopulaModel.
    """
    return joblib.load(path)


if __name__ == "__main__":
    from src.common.paths import FEATURES_CSV, ODD_COPULA_PATH, ensure_output_dirs

    ensure_output_dirs()
    features_path = FEATURES_CSV
    copula_path = ODD_COPULA_PATH

    features_df = pd.read_csv(features_path)

    equations = fit_structural_causal_model(features_df)
    print("=== Structural Causal Model (linear special case) ===")
    for eq in equations:
        coefs = dict(zip(eq.parents, eq.model.coef_))
        print(f"{eq.child} ~ {coefs} + intercept={eq.model.intercept_:.4f} (residual_std={eq.residual_std:.4f})")

    copula = fit_odd_copula(features_df)
    save_odd_copula(copula, copula_path)
    print(f"\nSaved ODD copula model -> '{copula_path}'")

    regions = features_df[copula.variables].apply(
        lambda row: classify_odd_region(copula, dict(zip(copula.variables, row))), axis=1
    )
    print("\nODD region distribution:")
    print(regions.value_counts())

    sample_state = build_odd_state(features_df.iloc[0])
    print(f"\nSample ODDState (row 0): {sample_state}")
