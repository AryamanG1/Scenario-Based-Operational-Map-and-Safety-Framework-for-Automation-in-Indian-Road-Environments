"""Tests for lane_detection.py's classical CV lane-boundary pipeline."""

import numpy as np
import pytest

from src.perception.lane_detection import (
    LaneDetectionResult,
    classify_lane_lines,
    compute_lane_features,
    detect_lanes,
    fit_lane_boundary,
)


def test_lane_detection_result_confidence_levels():
    result_none = LaneDetectionResult(left_lane=None, right_lane=None, image_shape=(224, 320))
    result_one = LaneDetectionResult(left_lane=(1.0, 0.0), right_lane=None, image_shape=(224, 320))
    result_both = LaneDetectionResult(left_lane=(1.0, 0.0), right_lane=(-1.0, 320.0), image_shape=(224, 320))
    assert result_none.lane_confidence == 0.0
    assert result_one.lane_confidence == 0.5
    assert result_both.lane_confidence == 1.0


def test_lane_width_none_when_missing_a_side():
    result = LaneDetectionResult(left_lane=(1.0, 0.0), right_lane=None, image_shape=(224, 320))
    assert result.lane_width_px is None


def test_classify_lane_lines_splits_by_slope_and_position():
    # Left lane boundary: bottom-left toward top-center -> x increases as y
    # decreases (negative slope), positioned left of center.
    # Right lane boundary: bottom-right toward top-center -> x decreases as
    # y decreases (positive slope), positioned right of center.
    left_line = (10, 200, 50, 100)
    right_line = (310, 200, 270, 100)
    left, right = classify_lane_lines([left_line, right_line], image_width=320)
    assert len(left) == 1
    assert len(right) == 1


def test_classify_lane_lines_rejects_near_horizontal():
    lines = [(0, 100, 320, 105)]  # nearly flat -> should be discarded
    left, right = classify_lane_lines(lines, image_width=320)
    assert left == [] and right == []


def test_fit_lane_boundary_empty_returns_none():
    assert fit_lane_boundary([]) is None


def test_fit_lane_boundary_fits_a_line():
    lines = [(0, 0, 10, 10), (0, 0, 20, 20)]
    fit = fit_lane_boundary(lines)
    assert fit is not None
    slope, intercept = fit
    assert slope == pytest.approx(1.0)
    assert intercept == pytest.approx(0.0, abs=1e-9)


def test_compute_lane_features_returns_expected_keys():
    result = LaneDetectionResult(left_lane=(1.0, 0.0), right_lane=(-1.0, 320.0), image_shape=(224, 320))
    features = compute_lane_features(result)
    assert set(features) == {"lane_confidence", "num_lines_detected", "lane_width_px"}


def test_detect_lanes_runs_on_real_image_without_crashing(small_images_labels, segnet):
    import torch

    images, _ = small_images_labels
    img = images[0]
    img_t = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        mask = segnet(img_t).argmax(dim=1).squeeze(0).numpy().astype(np.uint8)

    result = detect_lanes(img, mask)
    assert isinstance(result, LaneDetectionResult)
    assert 0.0 <= result.lane_confidence <= 1.0
