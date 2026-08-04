"""Module 1: IDD-Lite dataset extraction, cleaning, and loading.

Extracts the IDD-Lite (Indian Driving Dataset Lite) archive, pairs each road
scene image with its semantic segmentation mask, filters out corrupted or
poorly-labeled samples, resizes everything to a common shape, and returns
clean numpy arrays ready for SegNet training.
"""

import glob
import os
import tarfile
from typing import Tuple

import cv2
import numpy as np

IMAGE_SIZE = (320, 224)  # (width, height)
NUM_CLASSES = 8
VOID_LABEL = 7


def extract_dataset(tar_path: str, extract_to: str) -> str:
    """Extracts the IDD-Lite tar.gz archive.

    Args:
        tar_path: Path to the idd-lite.tar.gz archive.
        extract_to: Directory to extract the archive into.

    Returns:
        The path to the extracted idd20k_lite/ directory.

    Raises:
        FileNotFoundError: If tar_path does not exist.
    """
    if not os.path.isfile(tar_path):
        raise FileNotFoundError(f"Dataset archive not found: {tar_path}")

    os.makedirs(extract_to, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_to)

    dataset_dir = os.path.join(extract_to, "idd20k_lite")
    print(f"Extracted '{tar_path}' -> '{dataset_dir}'")
    return dataset_dir


def load_and_clean_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads, filters, and cleans the IDD-Lite train split.

    Pairs every leftImg8bit image with its gtFine semantic label, drops any
    sample that is unreadable, has a near-empty label (<=1 unique class), or
    has no road (class 0) pixels at all, then resizes survivors to
    IMAGE_SIZE and remaps void/out-of-range labels to VOID_LABEL.

    Args:
        dataset_dir: Path to the idd20k_lite/ directory (containing
            leftImg8bit/ and gtFine/ subdirectories).

    Returns:
        A tuple (images, labels):
            images: uint8 array of shape (M, 224, 320, 3).
            labels: uint8 array of shape (M, 224, 320).
    """
    image_paths = sorted(
        glob.glob(os.path.join(dataset_dir, "leftImg8bit", "train", "*", "*_image.jpg"))
    )

    clean_images = []
    clean_labels = []
    removed = 0

    for img_path in image_paths:
        lbl_path = img_path.replace("leftImg8bit", "gtFine").replace("_image.jpg", "_label.png")

        img = cv2.imread(img_path)
        lbl = cv2.imread(lbl_path, 0)

        if img is None or lbl is None:
            removed += 1
            continue
        if len(np.unique(lbl)) <= 1:
            removed += 1
            continue
        if 0 not in np.unique(lbl):
            removed += 1
            continue

        img_resized = cv2.resize(img, IMAGE_SIZE)
        lbl_resized = cv2.resize(lbl, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        lbl_resized = lbl_resized.copy()
        lbl_resized[lbl_resized == 255] = VOID_LABEL
        lbl_resized[lbl_resized > VOID_LABEL] = VOID_LABEL

        clean_images.append(img_resized)
        clean_labels.append(lbl_resized)

    print(f"Removed {removed} improper/corrupted files. Clean dataset size: {len(clean_images)}")

    images_array = np.array(clean_images, dtype=np.uint8)
    labels_array = np.array(clean_labels, dtype=np.uint8)
    return images_array, labels_array


if __name__ == "__main__":
    from src.common.paths import DATA_DIR

    images, labels = load_and_clean_dataset(DATA_DIR)
    print(f"images shape: {images.shape}, dtype: {images.dtype}")
    print(f"labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"label classes present: {sorted(np.unique(labels).tolist())}")
