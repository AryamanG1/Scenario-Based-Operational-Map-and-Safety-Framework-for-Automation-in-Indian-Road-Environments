"""Unit tests for odd_classifier.py's assign_mode() and combine_stage_outputs()."""

import pandas as pd
import pytest

from src.odd.odd_classifier import assign_mode, combine_stage_outputs


def _row(**overrides):
    base = {
        "object_presence": 1,
        "detection_confidence": 0.9,
        "visibility": 0.5,
        "traffic_density": 0.1,
        "non_drivable_area_score": 0.05,
        "living_things_score": 0.05,
    }
    base.update(overrides)
    return pd.Series(base)


def test_assign_mode_normal_case():
    assert assign_mode(_row()) == "Normal"


def test_assign_mode_empty_road_is_normal_not_takeover():
    # Critical bug fix under test: object_presence=0 -> detection_confidence
    # trivially 0.0, which must NOT force Takeover on an otherwise-clear road.
    row = _row(object_presence=0, detection_confidence=0.0, visibility=0.6)
    assert assign_mode(row) == "Normal"


def test_assign_mode_low_confidence_with_detections_is_not_normal():
    row = _row(object_presence=1, detection_confidence=0.3, visibility=0.6)
    assert assign_mode(row) != "Normal"


def test_assign_mode_degraded_when_moderate_confidence():
    row = _row(object_presence=1, detection_confidence=0.6, visibility=0.3, traffic_density=0.6)
    assert assign_mode(row) == "Degraded"


def test_assign_mode_takeover_when_low_visibility():
    row = _row(visibility=0.1)
    assert assign_mode(row) == "Takeover"


def test_assign_mode_high_pothole_and_water_blocks_normal_but_not_degraded():
    # assign_mode()'s Degraded branch only checks confidence/visibility, not
    # non_drivable_area_score/living_things_score -- those two only gate the
    # Normal branch. A scene bad on those but fine on visibility/confidence
    # lands in Degraded, not Takeover (this is the spec's actual rule, not
    # a bug -- see the module docstring's threshold rationale).
    row = _row(non_drivable_area_score=0.5, living_things_score=0.5)
    assert assign_mode(row) == "Degraded"


def test_assign_mode_takeover_when_high_pothole_water_and_low_visibility():
    row = _row(non_drivable_area_score=0.5, living_things_score=0.5, visibility=0.1)
    assert assign_mode(row) == "Takeover"


@pytest.mark.parametrize(
    "base_mode,odd_region,monitoring_state,expected",
    [
        ("Normal", "within", "Nominal", "Normal"),
        ("Normal", "near", "Nominal", "Degraded"),
        ("Normal", "outside", "Nominal", "Takeover"),
        ("Normal", "within", "Warning", "Degraded"),
        ("Normal", "within", "Critical", "Takeover"),
        ("Takeover", "within", "Nominal", "Takeover"),
        ("Degraded", "outside", "Critical", "Takeover"),
    ],
)
def test_combine_stage_outputs_most_severe_wins(base_mode, odd_region, monitoring_state, expected):
    assert combine_stage_outputs(base_mode, odd_region, monitoring_state) == expected
