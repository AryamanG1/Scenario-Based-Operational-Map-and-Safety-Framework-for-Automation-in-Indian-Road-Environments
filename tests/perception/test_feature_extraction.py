"""Tests for feature_extraction.py, including the class-mapping bugfix and heuristics."""

import numpy as np

from src.perception.feature_extraction import (
    LIVING_THINGS_CLASS_ID,
    NON_DRIVABLE_CLASS_ID,
    ROAD_CLASS_ID,
    VEHICLE_CLASS_ID,
    compute_features,
    detect_pothole_candidates,
    is_autorickshaw_like,
)


def test_class_id_mapping_matches_verified_idd_lite_scheme():
    # Regression test for the class-mapping bug found during development:
    # the one-shot prompt this project started from claimed 1=Pothole,
    # 2=Water, 6=Vehicle, none of which matched the real pixel data
    # (verified via spatial statistics and direct visual rendering).
    assert ROAD_CLASS_ID == 0
    assert NON_DRIVABLE_CLASS_ID == 1
    assert LIVING_THINGS_CLASS_ID == 2
    assert VEHICLE_CLASS_ID == 3


def test_compute_features_returns_22_columns():
    img = np.full((224, 320, 3), 128, dtype=np.uint8)
    mask = np.zeros((224, 320), dtype=np.uint8)
    mask[150:, :] = ROAD_CLASS_ID
    features = compute_features(img, mask, detections=[])
    assert len(features) == 22


def test_compute_features_no_detections_edge_case():
    img = np.full((224, 320, 3), 100, dtype=np.uint8)
    mask = np.zeros((224, 320), dtype=np.uint8)
    features = compute_features(img, mask, detections=[])
    assert features["object_presence"] == 0
    assert features["object_distance"] == 0.0
    assert features["detection_confidence"] == 0.0
    assert features["lead_vehicle_distance"] == 0.0


def test_compute_features_no_road_edge_case():
    img = np.full((224, 320, 3), 100, dtype=np.uint8)
    mask = np.full((224, 320), 5, dtype=np.uint8)  # all sky, no road
    features = compute_features(img, mask, detections=[])
    assert features["drivable_area"] == 0.0
    assert features["road_end_distance"] == 0.0
    assert features["road_quality"] == 0.0


def test_is_autorickshaw_like_only_flags_car_class():
    det_car = {"class": "car", "bbox": [0, 0, 30, 40], "confidence": 0.9}
    det_bus = {"class": "bus", "bbox": [0, 0, 30, 40], "confidence": 0.9}
    assert is_autorickshaw_like(det_bus, (224, 320)) is False
    # car with a plausible auto-rickshaw-like aspect ratio/area
    result = is_autorickshaw_like(det_car, (224, 320))
    assert isinstance(result, bool)


def test_detect_pothole_candidates_no_road_returns_zero():
    gray = np.full((224, 320), 100, dtype=np.uint8)
    mask = np.full((224, 320), 5, dtype=np.uint8)  # no road pixels
    score, count = detect_pothole_candidates(gray, mask)
    assert score == 0.0 and count == 0


def test_detect_pothole_candidates_flat_road_finds_nothing():
    gray = np.full((224, 320), 128, dtype=np.uint8)  # perfectly uniform road
    mask = np.zeros((224, 320), dtype=np.uint8)
    mask[150:, :] = ROAD_CLASS_ID
    score, count = detect_pothole_candidates(gray, mask)
    assert score == 0.0 and count == 0


def test_full_extraction_on_real_images(small_images_labels, segnet, yolo):
    from src.perception.feature_extraction import extract_all_features

    images, labels = small_images_labels
    df = extract_all_features(images[:3], labels[:3], segnet, yolo, save_path="/tmp/test_features.csv")
    assert len(df) == 3
    assert len(df.columns) == 22
    assert (df["brightness"] >= 0).all() and (df["brightness"] <= 255).all()
    assert (df["drivable_area"] >= 0).all() and (df["drivable_area"] <= 1).all()
