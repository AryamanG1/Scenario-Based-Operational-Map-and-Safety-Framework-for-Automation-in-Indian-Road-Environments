"""Unit tests for fuzzy_odd.py's membership functions and uODD classification."""

import pytest

from src.odd.fuzzy_odd import (
    MU_ODD_STATES,
    PDBreakpoints,
    calibrate_pd_breakpoints,
    classify_mu_odd,
    compose_scenario_label,
    defuzzify_mu_odd,
    pd_dry_membership,
    pd_puddles_membership,
    pd_wet_membership,
    trapezoidal_membership,
    triangular_membership,
)


def test_triangular_membership_peak_and_feet():
    assert triangular_membership(5, a=0, b=5, c=10) == pytest.approx(1.0)
    assert triangular_membership(0, a=0, b=5, c=10) == pytest.approx(0.0)
    assert triangular_membership(10, a=0, b=5, c=10) == pytest.approx(0.0)
    assert triangular_membership(2.5, a=0, b=5, c=10) == pytest.approx(0.5)


def test_triangular_membership_outside_support_is_zero():
    assert triangular_membership(-5, a=0, b=5, c=10) == 0.0
    assert triangular_membership(15, a=0, b=5, c=10) == 0.0


def test_trapezoidal_membership_plateau_and_feet():
    assert trapezoidal_membership(0, a=0, b=0, c=5, d=10) == pytest.approx(1.0)
    assert trapezoidal_membership(3, a=0, b=0, c=5, d=10) == pytest.approx(1.0)
    assert trapezoidal_membership(7.5, a=0, b=0, c=5, d=10) == pytest.approx(0.5)
    assert trapezoidal_membership(10, a=0, b=0, c=5, d=10) == pytest.approx(0.0)


def test_calibrate_pd_breakpoints_matches_percentiles():
    values = list(range(1, 101))  # 1..100
    bp = calibrate_pd_breakpoints(values)
    assert bp.w_min == 1
    assert bp.w_max == 100
    assert bp.p50 == pytest.approx(50.5, abs=1.0)


def test_mu_odd_memberships_sum_reasonably_and_are_bounded():
    bp = PDBreakpoints(w_min=0, p25=25, p50=50, p75=75, w_max=100)
    for w in (0, 25, 50, 75, 100, 37.5):
        memberships = classify_mu_odd(w, bp)
        assert set(memberships) == set(MU_ODD_STATES)
        for value in memberships.values():
            assert 0.0 <= value <= 1.0


def test_dry_membership_high_for_low_wetness():
    bp = PDBreakpoints(w_min=0, p25=25, p50=50, p75=75, w_max=100)
    assert pd_dry_membership(0, bp) == pytest.approx(1.0)
    assert pd_dry_membership(50, bp) == pytest.approx(0.0)


def test_puddles_membership_high_for_high_wetness():
    bp = PDBreakpoints(w_min=0, p25=25, p50=50, p75=75, w_max=100)
    assert pd_puddles_membership(100, bp) == pytest.approx(1.0)
    assert pd_puddles_membership(50, bp) == pytest.approx(0.0)


def test_wet_membership_peaks_at_median():
    bp = PDBreakpoints(w_min=0, p25=25, p50=50, p75=75, w_max=100)
    assert pd_wet_membership(50, bp) == pytest.approx(1.0)
    assert pd_wet_membership(0, bp) == pytest.approx(0.0)


def test_defuzzify_picks_max_membership():
    assert defuzzify_mu_odd({"NoRain": 0.2, "LowRain": 0.9, "HeavyRain": 0.1}) == "LowRain"


def test_compose_scenario_label_format():
    assert compose_scenario_label("HeavyRain", "Congested") == "HeavyRain-Congested"
