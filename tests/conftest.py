"""Shared pytest fixtures for the capstone test suite.

Model/dataset-dependent fixtures are session-scoped so the (slow) SegNet/
YOLO loads and dataset read happen once per test run, not once per test.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.common.paths import DATA_DIR, FEATURES_CSV, SEGNET_CHECKPOINT, YOLO_WEIGHTS  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> str:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def real_features_df() -> pd.DataFrame:
    """Loads the real, already-extracted final_features.csv."""
    if not os.path.isfile(FEATURES_CSV):
        pytest.skip("outputs/final_features.csv not found -- run feature_extraction.py first.")
    return pd.read_csv(FEATURES_CSV)


@pytest.fixture(scope="session")
def small_images_labels():
    """Loads a small slice (10 images) of the real cleaned dataset."""
    from src.perception.data_pipeline import load_and_clean_dataset

    if not os.path.isdir(DATA_DIR):
        pytest.skip("dataset not found at data/idd20k_lite")
    images, labels = load_and_clean_dataset(DATA_DIR)
    return images[:10], labels[:10]


@pytest.fixture(scope="session")
def segnet():
    """Loads the trained SegNet checkpoint."""
    from src.perception.segnet_model import load_segnet

    if not os.path.isfile(SEGNET_CHECKPOINT):
        pytest.skip("models/refined_segnet.pth not found -- train SegNet first.")
    return load_segnet(SEGNET_CHECKPOINT)


@pytest.fixture(scope="session")
def yolo():
    """Loads the YOLOv8n model."""
    from ultralytics import YOLO

    if not os.path.isfile(YOLO_WEIGHTS):
        pytest.skip("models/yolov8n.pt not found.")
    return YOLO(YOLO_WEIGHTS)


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    """A small, fully synthetic features dataframe for fast, deterministic tests."""
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "vehicle_count": rng.integers(0, 6, n),
            "num_two_wheelers": rng.integers(0, 3, n),
            "num_pedestrians": rng.integers(0, 5, n),
            "num_animals": rng.integers(0, 2, n),
            "num_autorickshaws": rng.integers(0, 2, n),
            "object_presence": rng.integers(0, 2, n),
            "object_distance": rng.uniform(0, 50, n),
            "lead_vehicle_distance": rng.uniform(0, 40, n),
            "traffic_density": rng.integers(0, 15, n),
            "detection_confidence": rng.uniform(0, 1, n),
            "drivable_area": rng.uniform(0.1, 0.6, n),
            "road_end_distance": rng.uniform(50, 200, n),
            "non_drivable_area_score": rng.uniform(0, 0.05, n),
            "non_drivable_area_distance": rng.uniform(0, 100, n),
            "living_things_score": rng.uniform(0, 0.01, n),
            "brightness": rng.uniform(40, 160, n),
            "visibility": rng.uniform(50, 2000, n),
            "wetness": rng.uniform(20, 100, n),
            "road_quality": rng.uniform(100, 6000, n),
            "pothole_heuristic_score": rng.uniform(0, 0.1, n),
            "pothole_heuristic_count": rng.integers(0, 15, n),
            "scene_complexity": rng.uniform(0, 1, n),
        }
    )
