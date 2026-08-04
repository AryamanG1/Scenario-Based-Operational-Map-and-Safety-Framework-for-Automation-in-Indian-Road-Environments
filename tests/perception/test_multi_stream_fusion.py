"""Unit tests for multi_stream_fusion.py's dynamic reliability-weighted fusion."""

import pytest

from src.perception.multi_stream_fusion import (
    IMAGE_STREAM,
    SEGNET_STREAM,
    STREAMS,
    YOLO_STREAM,
    calibrate_references,
    compute_context_factor,
    compute_fused_confidence,
    compute_fusion_weights,
    compute_stream_reliabilities,
    fuse_dataframe,
)


def test_calibrate_references_uses_medians(synthetic_features_df):
    refs = calibrate_references(synthetic_features_df)
    assert refs.median_drivable_area == pytest.approx(synthetic_features_df["drivable_area"].median())


def test_stream_reliabilities_are_bounded(synthetic_features_df):
    refs = calibrate_references(synthetic_features_df)
    for _, row in synthetic_features_df.iterrows():
        reliabilities = compute_stream_reliabilities(row, refs)
        assert set(reliabilities) == set(STREAMS)
        for value in reliabilities.values():
            assert 0.0 <= value <= 1.0


def test_yolo_reliability_equals_detection_confidence(synthetic_features_df):
    refs = calibrate_references(synthetic_features_df)
    row = synthetic_features_df.iloc[0]
    reliabilities = compute_stream_reliabilities(row, refs)
    assert reliabilities[YOLO_STREAM] == pytest.approx(row["detection_confidence"])


def test_context_factor_bounded(synthetic_features_df):
    refs = calibrate_references(synthetic_features_df)
    for _, row in synthetic_features_df.iterrows():
        c = compute_context_factor(row, refs)
        assert 0.0 <= c <= 1.0


def test_fusion_weights_sum_to_one():
    reliabilities = {SEGNET_STREAM: 0.9, YOLO_STREAM: 0.3, IMAGE_STREAM: 0.6}
    weights = compute_fusion_weights(reliabilities, context_factor=1.0)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_higher_reliability_stream_gets_higher_weight():
    reliabilities = {SEGNET_STREAM: 0.95, YOLO_STREAM: 0.1, IMAGE_STREAM: 0.1}
    weights = compute_fusion_weights(reliabilities, context_factor=1.0)
    assert weights[SEGNET_STREAM] > weights[YOLO_STREAM]
    assert weights[SEGNET_STREAM] > weights[IMAGE_STREAM]


def test_zero_context_factor_still_yields_valid_weights():
    reliabilities = {SEGNET_STREAM: 0.9, YOLO_STREAM: 0.3, IMAGE_STREAM: 0.6}
    weights = compute_fusion_weights(reliabilities, context_factor=0.0)
    assert sum(weights.values()) == pytest.approx(1.0)
    # With context_factor=0, all streams score equally -> uniform weights.
    assert weights[SEGNET_STREAM] == pytest.approx(weights[YOLO_STREAM])


def test_fused_confidence_is_weighted_average():
    reliabilities = {SEGNET_STREAM: 1.0, YOLO_STREAM: 0.0, IMAGE_STREAM: 0.0}
    weights = {SEGNET_STREAM: 1.0, YOLO_STREAM: 0.0, IMAGE_STREAM: 0.0}
    assert compute_fused_confidence(reliabilities, weights) == pytest.approx(1.0)


def test_fuse_dataframe_adds_expected_columns(synthetic_features_df):
    fused = fuse_dataframe(synthetic_features_df)
    for stream in STREAMS:
        assert f"reliability_{stream}" in fused.columns
        assert f"weight_{stream}" in fused.columns
    assert "fused_confidence" in fused.columns
    assert len(fused) == len(synthetic_features_df)
