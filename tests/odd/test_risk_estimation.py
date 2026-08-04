"""Unit tests for risk_estimation.py's staged failure-probability estimator."""

import numpy as np
import pytest

from src.odd.risk_estimation import density_to_severity, estimate_failure_probability


def test_estimate_failure_probability_recovers_known_uniform_quantile():
    # For a uniform population, P(X >= q) is exactly known analytically --
    # this is the key correctness check for the staged decomposition.
    rng = np.random.default_rng(42)
    population = rng.uniform(0, 1, size=20000)

    result = estimate_failure_probability(population, failure_threshold=0.95, conditional_prob_target=0.1)
    assert result.failure_probability == pytest.approx(0.05, abs=0.01)


def test_estimate_failure_probability_product_of_levels():
    rng = np.random.default_rng(0)
    population = rng.uniform(0, 1, size=20000)
    result = estimate_failure_probability(population, failure_threshold=0.99, conditional_prob_target=0.1)
    assert result.failure_probability == pytest.approx(np.prod(result.level_conditional_probabilities))


def test_estimate_failure_probability_threshold_above_population_max():
    population = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
    result = estimate_failure_probability(population, failure_threshold=100.0, conditional_prob_target=0.2)
    assert result.failure_probability == 0.0


def test_estimate_failure_probability_threshold_at_population_min():
    population = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 10)
    result = estimate_failure_probability(population, failure_threshold=0.0, conditional_prob_target=0.2)
    assert result.failure_probability > 0.9


def test_density_to_severity_extremes():
    percentiles = np.sort(np.arange(1, 101, dtype=float))
    assert density_to_severity(percentiles, 0.0) == pytest.approx(1.0)  # below everything
    assert density_to_severity(percentiles, 1000.0) == pytest.approx(0.0)  # above everything


def test_density_to_severity_is_monotonically_decreasing():
    percentiles = np.sort(np.arange(1, 101, dtype=float))
    sev_low = density_to_severity(percentiles, 10.0)
    sev_high = density_to_severity(percentiles, 90.0)
    assert sev_low > sev_high
