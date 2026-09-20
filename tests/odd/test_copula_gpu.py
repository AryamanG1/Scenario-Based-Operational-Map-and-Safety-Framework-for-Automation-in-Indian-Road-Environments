"""Tests that the batched (GPU-capable) copula density reproduces odd_boundary.odd_density."""

import numpy as np
import pandas as pd

from src.odd.copula_gpu import classify_odd_region_batch, odd_density_batch
from src.odd.odd_boundary import (
    DEFAULT_ODD_VARIABLES,
    classify_odd_region,
    density_cutpoints,
    fit_odd_copula,
    odd_density,
)


def test_odd_density_batch_matches_scalar(synthetic_features_df):
    copula = fit_odd_copula(synthetic_features_df, DEFAULT_ODD_VARIABLES)
    X = synthetic_features_df[DEFAULT_ODD_VARIABLES]

    batched = odd_density_batch(copula, X)
    scalar = np.array([odd_density(copula, dict(zip(DEFAULT_ODD_VARIABLES, row))) for row in X.to_numpy()])

    assert batched.shape == (len(X),)
    np.testing.assert_allclose(batched, scalar, rtol=1e-10, atol=0)


def test_odd_density_batch_accepts_array_and_dicts(synthetic_features_df):
    copula = fit_odd_copula(synthetic_features_df, DEFAULT_ODD_VARIABLES)
    X = synthetic_features_df[DEFAULT_ODD_VARIABLES].head(5)

    from_df = odd_density_batch(copula, X)
    from_arr = odd_density_batch(copula, X.to_numpy())
    from_dicts = odd_density_batch(copula, X.to_dict("records"))
    np.testing.assert_array_equal(from_df, from_arr)
    np.testing.assert_array_equal(from_df, from_dicts)
    assert odd_density_batch(copula, X.head(0)).shape == (0,)


def test_density_percentiles_match_scalar_fit(synthetic_features_df):
    """fit_odd_copula now fills density_percentiles via the batch path."""
    copula = fit_odd_copula(synthetic_features_df, DEFAULT_ODD_VARIABLES)
    X = synthetic_features_df[DEFAULT_ODD_VARIABLES].to_numpy()
    scalar = np.sort([odd_density(copula, dict(zip(DEFAULT_ODD_VARIABLES, row))) for row in X])
    np.testing.assert_allclose(copula.density_percentiles, scalar, rtol=1e-10, atol=0)


def test_classify_odd_region_batch_matches_scalar(synthetic_features_df):
    copula = fit_odd_copula(synthetic_features_df, DEFAULT_ODD_VARIABLES)
    X = synthetic_features_df[DEFAULT_ODD_VARIABLES]

    batched = classify_odd_region_batch(copula, X)
    scalar = [classify_odd_region(copula, dict(zip(DEFAULT_ODD_VARIABLES, row))) for row in X.to_numpy()]
    assert list(batched) == scalar
    assert set(batched) <= {"within", "near", "outside"}


def test_density_cutpoints_cached_and_lazy(synthetic_features_df):
    copula = fit_odd_copula(synthetic_features_df, DEFAULT_ODD_VARIABLES)
    p15, p50 = density_cutpoints(copula)
    assert p15 <= p50
    assert copula.density_cutpoints == (p15, p50)

    # A model unpickled from before the field existed has it as None; the
    # accessor must recompute rather than fail.
    copula.density_cutpoints = None
    assert density_cutpoints(copula) == (p15, p50)
