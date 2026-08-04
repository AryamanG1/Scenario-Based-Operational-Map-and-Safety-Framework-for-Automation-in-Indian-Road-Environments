"""Tests for data_pipeline.py against the real IDD-Lite dataset."""

import numpy as np

from src.perception.data_pipeline import IMAGE_SIZE, NUM_CLASSES, VOID_LABEL


def test_constants():
    assert IMAGE_SIZE == (320, 224)
    assert NUM_CLASSES == 8
    assert VOID_LABEL == 7


def test_load_and_clean_dataset_shapes_and_dtypes(small_images_labels):
    images, labels = small_images_labels
    assert images.dtype == np.uint8
    assert labels.dtype == np.uint8
    assert images.shape[1:] == (224, 320, 3)
    assert labels.shape[1:] == (224, 320)


def test_labels_within_valid_class_range(small_images_labels):
    _, labels = small_images_labels
    assert labels.min() >= 0
    assert labels.max() <= VOID_LABEL


def test_extract_dataset_roundtrip(tmp_path, project_root):
    import os

    from src.perception.data_pipeline import extract_dataset

    tar_path = os.path.join(project_root, "data", "idd-lite.tar.gz")
    if not os.path.isfile(tar_path):
        import pytest

        pytest.skip("data/idd-lite.tar.gz not found")

    extracted = extract_dataset(tar_path, str(tmp_path))
    assert os.path.isdir(extracted)
    assert os.path.isdir(os.path.join(extracted, "leftImg8bit"))
    assert os.path.isdir(os.path.join(extracted, "gtFine"))
