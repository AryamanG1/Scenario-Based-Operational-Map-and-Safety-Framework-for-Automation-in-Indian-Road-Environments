"""Tests for feasibility_map.py's cross-stage scoring and feasibility map."""

import pytest

from src.decision.feasibility_map import automation_level_from_score, compute_odd_score


def _scaled_row(**overrides):
    import pandas as pd

    base = {
        "visibility": 0.9,
        "brightness": 0.9,
        "traffic_density": 0.1,
        "wetness": 0.1,
        "non_drivable_area_score": 0.05,
        "detection_confidence": 0.9,
    }
    base.update(overrides)
    return pd.Series(base)


def test_compute_odd_score_best_case_near_100():
    score = compute_odd_score(_scaled_row())
    assert score > 80


def test_compute_odd_score_worst_case_near_zero():
    row = _scaled_row(
        visibility=0.0, brightness=0.0, traffic_density=1.0, wetness=1.0,
        non_drivable_area_score=1.0, detection_confidence=0.0,
    )
    score = compute_odd_score(row)
    assert score == 0.0


def test_compute_odd_score_bounded():
    for _ in range(20):
        import random

        row = _scaled_row(
            visibility=random.random(), brightness=random.random(),
            traffic_density=random.random(), wetness=random.random(),
            non_drivable_area_score=random.random(), detection_confidence=random.random(),
        )
        score = compute_odd_score(row)
        assert 0.0 <= score <= 100.0


def test_automation_level_bands():
    assert automation_level_from_score(90) == "L3"
    assert automation_level_from_score(70) == "L3"
    assert automation_level_from_score(55) == "L2"
    assert automation_level_from_score(40) == "L2"
    assert automation_level_from_score(20) == "L1"


def test_build_feasibility_map_end_to_end(real_features_df):
    import os

    import joblib

    from src.common.paths import FEATURE_SCALER_PATH
    from src.decision.feasibility_map import build_feasibility_map

    if not os.path.isfile(FEATURE_SCALER_PATH):
        pytest.skip("models/feature_scaler.pkl not found -- run odd_classifier.py first.")

    scaler = joblib.load(FEATURE_SCALER_PATH)
    small_df = real_features_df.head(15)
    result = build_feasibility_map(small_df, scaler)

    assert len(result) == 15
    for col in ("traffic_density_level", "scenario_label", "odd_region", "final_mode", "odd_score"):
        assert col in result.columns
    assert result["odd_score"].between(0, 100).all()
