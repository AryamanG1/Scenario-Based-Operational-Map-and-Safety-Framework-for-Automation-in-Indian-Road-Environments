"""Unit tests for odd_boundary.py's SCM and Gaussian-copula ODD-space model."""

import numpy as np
import pytest

from src.odd.odd_boundary import (
    DEFAULT_ODD_VARIABLES,
    build_odd_state,
    classify_odd_region,
    fit_odd_copula,
    fit_structural_causal_model,
    odd_density,
)


def test_fit_structural_causal_model_returns_one_equation_per_dag_entry(synthetic_features_df):
    equations = fit_structural_causal_model(synthetic_features_df)
    assert len(equations) == 3
    for eq in equations:
        assert eq.residual_std >= 0


def test_fit_odd_copula_and_density_is_nonnegative(synthetic_features_df):
    copula = fit_odd_copula(synthetic_features_df, DEFAULT_ODD_VARIABLES)
    row = synthetic_features_df.iloc[0]
    x = {v: row[v] for v in DEFAULT_ODD_VARIABLES}
    density = odd_density(copula, x)
    assert density >= 0.0


def test_typical_scene_has_higher_density_than_extreme_outlier(synthetic_features_df):
    copula = fit_odd_copula(synthetic_features_df, DEFAULT_ODD_VARIABLES)

    typical_row = synthetic_features_df.iloc[len(synthetic_features_df) // 2]
    typical_x = {v: typical_row[v] for v in DEFAULT_ODD_VARIABLES}

    outlier_x = {v: synthetic_features_df[v].max() * 100 + 1e6 for v in DEFAULT_ODD_VARIABLES}

    assert odd_density(copula, typical_x) > odd_density(copula, outlier_x)


def test_classify_odd_region_returns_valid_label(synthetic_features_df):
    copula = fit_odd_copula(synthetic_features_df, DEFAULT_ODD_VARIABLES)
    row = synthetic_features_df.iloc[0]
    x = {v: row[v] for v in DEFAULT_ODD_VARIABLES}
    region = classify_odd_region(copula, x)
    assert region in ("within", "near", "outside")


def test_classify_odd_region_extreme_outlier_is_outside(synthetic_features_df):
    copula = fit_odd_copula(synthetic_features_df, DEFAULT_ODD_VARIABLES)
    outlier_x = {v: synthetic_features_df[v].max() * 1000 + 1e9 for v in DEFAULT_ODD_VARIABLES}
    assert classify_odd_region(copula, outlier_x) == "outside"


def test_build_odd_state_populates_known_fields(synthetic_features_df):
    row = synthetic_features_df.iloc[0]
    state = build_odd_state(row)
    assert state.scenery.drivable_area == row["drivable_area"]
    assert state.environment.wetness == row["wetness"]
    assert state.dynamic_entities.relative_distance == row["lead_vehicle_distance"]
    # Fields with no available sensor/feature must stay honestly None.
    assert state.scenery.traffic_signs is None
    assert state.dynamic_entities.vehicle_speed is None
