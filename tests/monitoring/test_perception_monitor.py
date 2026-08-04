"""Tests for perception_monitor.py's Stage 6 monitoring state machine logic."""

from src.monitoring.perception_monitor import (
    CONSECUTIVE_CRITICAL,
    CONSECUTIVE_WARNING,
    CRITICAL,
    NOMINAL,
    WARNING,
    FrameCheckResult,
)


def _classify(max_consecutive: int) -> str:
    if max_consecutive >= CONSECUTIVE_CRITICAL:
        return CRITICAL
    if max_consecutive >= CONSECUTIVE_WARNING:
        return WARNING
    return NOMINAL


def test_consecutive_bad_streak_classification_thresholds():
    assert _classify(0) == NOMINAL
    assert _classify(CONSECUTIVE_WARNING - 1) == NOMINAL
    assert _classify(CONSECUTIVE_WARNING) == WARNING
    assert _classify(CONSECUTIVE_CRITICAL - 1) == WARNING
    assert _classify(CONSECUTIVE_CRITICAL) == CRITICAL


def test_frame_check_result_fields():
    check = FrameCheckResult(mask_iou=0.8, confidence_drop=0.1, is_bad=False)
    assert check.mask_iou == 0.8
    assert check.is_bad is False


def test_run_perception_monitor_on_real_frame(small_images_labels, segnet, yolo):
    from src.monitoring.perception_monitor import run_perception_monitor

    images, _ = small_images_labels
    result = run_perception_monitor(images[0], segnet, yolo, num_checks=2)
    assert result.state in (NOMINAL, WARNING, CRITICAL)
    assert len(result.checks) == 2
    for check in result.checks:
        assert 0.0 <= check.mask_iou <= 1.0
        assert check.confidence_drop >= 0.0
