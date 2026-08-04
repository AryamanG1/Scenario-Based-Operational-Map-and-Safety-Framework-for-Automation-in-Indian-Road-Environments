"""Unit tests for detection_benchmark.py's IoU/AP/mAP/AOS formulas."""

import math

import numpy as np
import pytest

from src.perception.detection_benchmark import (
    auc_average_precision,
    compute_iou,
    compute_mean_ap,
    extract_pseudo_gt_boxes,
    interpolated_ap,
    match_detections_to_gt,
    orientation_similarity,
    precision_recall,
)


def test_iou_identical_boxes_is_one():
    assert compute_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert compute_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_iou_partial_overlap_hand_computed():
    # box A: [0,10]x[0,10] (area 100); box B: [5,15]x[5,15] (area 100)
    # intersection: [5,10]x[5,10] = 25; union = 100+100-25=175
    iou = compute_iou((0, 0, 10, 10), (5, 5, 10, 10))
    assert iou == pytest.approx(25 / 175)


def test_extract_pseudo_gt_boxes_finds_expected_blob_count():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 3  # 3x3 blob, area 9
    mask[10:16, 10:16] = 3  # 6x6 blob, area 36
    boxes = extract_pseudo_gt_boxes(mask, class_id=3, min_area=5)
    assert len(boxes) == 2


def test_extract_pseudo_gt_boxes_filters_small_blobs():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:2, 0:2] = 3  # area 4, below min_area
    boxes = extract_pseudo_gt_boxes(mask, class_id=3, min_area=10)
    assert boxes == []


def test_precision_recall_basic():
    p, r = precision_recall(n_tp=8, n_all=10, n_all_gt=10)
    assert p == 0.8 and r == 0.8


def test_precision_recall_zero_denominators():
    p, r = precision_recall(0, 0, 0)
    assert p == 0.0 and r == 0.0


def test_interpolated_ap_perfect_detector():
    assert interpolated_ap([True] * 10, num_gt=10, num_recall_points=11) == pytest.approx(1.0)


def test_interpolated_ap_worst_detector():
    assert interpolated_ap([False] * 10, num_gt=10, num_recall_points=11) == 0.0


def test_interpolated_ap_no_ground_truth_is_zero():
    assert interpolated_ap([True, False], num_gt=0) == 0.0


def test_auc_average_precision_perfect_detector_is_100():
    assert auc_average_precision([True] * 5, num_gt=5) == pytest.approx(100.0, abs=1.0)


def test_compute_mean_ap_averages_classes():
    assert compute_mean_ap({"car": 0.8, "bus": 0.6}) == pytest.approx(0.7)


def test_compute_mean_ap_empty_is_zero():
    assert compute_mean_ap({}) == 0.0


def test_orientation_similarity_perfect_alignment_all_tp():
    sim = orientation_similarity([0.0, 0.0], [True, True])
    assert sim == pytest.approx(1.0)


def test_orientation_similarity_opposite_alignment_is_zero():
    sim = orientation_similarity([math.pi, math.pi], [True, True])
    assert sim == pytest.approx(0.0, abs=1e-9)


def test_orientation_similarity_false_positive_contributes_zero():
    sim = orientation_similarity([0.0], [False])
    assert sim == pytest.approx(0.0)


def test_match_detections_to_gt_greedy_assignment():
    scored = [((0, 0, 10, 10), 0.9), ((100, 100, 10, 10), 0.5)]
    gt = [(0, 0, 10, 10)]
    flags = match_detections_to_gt(scored, gt, iou_threshold=0.5)
    assert flags == [True, False]


def test_match_detections_to_gt_each_gt_claimed_once():
    # Two detections both overlapping the same single GT box -- only the
    # higher-confidence one should match; the second is a false positive.
    scored = [((0, 0, 10, 10), 0.9), ((1, 1, 10, 10), 0.8)]
    gt = [(0, 0, 10, 10)]
    flags = match_detections_to_gt(scored, gt, iou_threshold=0.3)
    assert flags.count(True) == 1
