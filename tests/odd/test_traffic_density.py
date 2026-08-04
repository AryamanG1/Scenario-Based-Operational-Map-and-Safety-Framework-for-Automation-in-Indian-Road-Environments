"""Unit tests for traffic_density.py's ratio computation and quartile calibration."""

import pytest

from src.odd.traffic_density import (
    CONGESTED,
    HIGH,
    LOW,
    MEDIUM,
    TrafficDensityThresholds,
    calibrate_thresholds,
    classify_density,
    compute_vehicle_area_ratio,
)


def test_compute_vehicle_area_ratio_zero_when_no_vehicles():
    assert compute_vehicle_area_ratio(0, 0, 0, image_area=1000) == 0.0


def test_compute_vehicle_area_ratio_scales_with_count():
    r1 = compute_vehicle_area_ratio(1, 0, 0, image_area=1000)
    r2 = compute_vehicle_area_ratio(2, 0, 0, image_area=1000)
    assert r2 == pytest.approx(2 * r1)


def test_calibrate_thresholds_quartiles():
    ratios = list(range(1, 101))  # 1..100
    thresholds = calibrate_thresholds(ratios)
    assert thresholds.low_medium == pytest.approx(25.75, abs=1.0)
    assert thresholds.medium_high == pytest.approx(50.5, abs=1.0)
    assert thresholds.high_congested == pytest.approx(75.25, abs=1.0)


def test_classify_density_all_four_bands():
    thresholds = TrafficDensityThresholds(low_medium=10, medium_high=20, high_congested=30)
    assert classify_density(5, thresholds) == LOW
    assert classify_density(15, thresholds) == MEDIUM
    assert classify_density(25, thresholds) == HIGH
    assert classify_density(35, thresholds) == CONGESTED


def test_classify_density_boundary_inclusive_on_lower_band():
    thresholds = TrafficDensityThresholds(low_medium=10, medium_high=20, high_congested=30)
    assert classify_density(10, thresholds) == LOW


def test_thresholds_save_and_load_roundtrip(tmp_path):
    thresholds = TrafficDensityThresholds(low_medium=1.0, medium_high=2.0, high_congested=3.0)
    path = str(tmp_path / "thresholds.json")
    thresholds.save(path)
    loaded = TrafficDensityThresholds.load(path)
    assert loaded == thresholds
