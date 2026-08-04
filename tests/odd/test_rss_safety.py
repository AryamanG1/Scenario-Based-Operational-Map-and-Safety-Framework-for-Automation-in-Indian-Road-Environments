"""Unit tests for rss_safety.py's RSS distance and sideslip stability formulas."""

import math

import pytest

from src.odd.rss_safety import (
    RSS_PARAMETERS,
    SideslipState,
    compute_stability_boundary,
    is_sideslip_stable,
    magic_tire_lateral_force,
    rss_minimum_safe_distance,
    sideslip_dynamics,
    stability_margin,
    unsafe_distance_membership,
)


def test_rss_distance_zero_when_ego_stationary_and_front_far_faster():
    # If ego is stationary and never moves, distance requirement collapses
    # to only the -v_front^2/(2*b_max) term, which is negative -> clamped to 0.
    params = RSS_PARAMETERS["NoRain"]
    d = rss_minimum_safe_distance(v_ego=0.0, v_front=30.0, params=params)
    assert d == 0.0


def test_rss_distance_matches_hand_computation():
    params = RSS_PARAMETERS["NoRain"]  # a_max=2.0, b_max=7.5, b_min=4.0, rho=0.2
    v_ego, v_front = 20.0, 15.0
    expected = (
        v_ego * params.rho
        + 0.5 * params.a_max * params.rho**2
        + (v_ego + params.rho * params.a_max) ** 2 / (2 * params.b_min)
        - v_front**2 / (2 * params.b_max)
    )
    assert rss_minimum_safe_distance(v_ego, v_front, params) == pytest.approx(expected)


def test_rss_distance_increases_as_weather_worsens():
    # Worse weather -> lower b_max/b_min -> larger required safe distance.
    v_ego, v_front = 20.0, 15.0
    d_no_rain = rss_minimum_safe_distance(v_ego, v_front, RSS_PARAMETERS["NoRain"])
    d_low_rain = rss_minimum_safe_distance(v_ego, v_front, RSS_PARAMETERS["LowRain"])
    d_heavy_rain = rss_minimum_safe_distance(v_ego, v_front, RSS_PARAMETERS["HeavyRain"])
    assert d_no_rain < d_low_rain < d_heavy_rain


def test_unsafe_distance_membership_boundaries():
    assert unsafe_distance_membership(d=2.0, d_unsafe=5.0, d_safe=20.0) == 1.0
    assert unsafe_distance_membership(d=25.0, d_unsafe=5.0, d_safe=20.0) == 0.0
    mid = unsafe_distance_membership(d=12.5, d_unsafe=5.0, d_safe=20.0)
    assert mid == pytest.approx(0.5)


def test_magic_tire_force_zero_at_zero_slip_angle():
    assert magic_tire_lateral_force(mu=0.8, f_z=5000.0, alpha=0.0) == pytest.approx(0.0)


def test_magic_tire_force_increases_with_slip_angle_then_saturates():
    f_small = magic_tire_lateral_force(mu=0.8, f_z=5000.0, alpha=0.05)
    f_medium = magic_tire_lateral_force(mu=0.8, f_z=5000.0, alpha=0.5)
    f_near_saturation = magic_tire_lateral_force(mu=0.8, f_z=5000.0, alpha=50.0)
    f_huge = magic_tire_lateral_force(mu=0.8, f_z=5000.0, alpha=5000.0)
    assert 0 < f_small < f_medium < f_near_saturation
    # sin(B*arctan(D*alpha)) saturates toward sin(B*pi/2) as alpha -> inf;
    # alpha=50 is already close to that asymptote, alpha=0.5 is not.
    assert f_near_saturation == pytest.approx(f_huge, rel=0.05)


def test_sideslip_dynamics_raises_on_zero_speed():
    state = SideslipState(
        beta=0.0, omega_r=0.0, v_x=0.0, mass=1500, i_z=2500, l_f=1.2, l_r=1.6,
        delta=0.0, mu=0.8, f_zf=7000, f_zr=6500, alpha_f=0.0, alpha_r=0.0,
    )
    with pytest.raises(ValueError):
        sideslip_dynamics(state)


def test_stability_boundary_is_symmetric_at_zero_steering():
    boundary = compute_stability_boundary(v_x=15.0, mu=0.8, delta=0.0)
    assert boundary.b2 == pytest.approx(-boundary.b3)


def test_is_sideslip_stable_true_at_zero_state():
    # beta=0, beta_dot=0 -> phase-plane value 0, which must lie within any
    # symmetric [b2, b3] boundary around 0 (as long as b2<=0<=b3).
    boundary = compute_stability_boundary(v_x=15.0, mu=0.8, delta=0.0)
    assert is_sideslip_stable(beta=0.0, beta_dot=0.0, boundary=boundary)
    assert stability_margin(0.0, 0.0, boundary) >= 0


def test_is_sideslip_stable_false_for_extreme_phase_plane_value():
    boundary = compute_stability_boundary(v_x=15.0, mu=0.8, delta=0.0)
    assert not is_sideslip_stable(beta=100.0, beta_dot=100.0, boundary=boundary)
