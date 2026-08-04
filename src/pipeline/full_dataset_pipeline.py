"""Module 7: Full-dataset (IDD-20K-scale) batch feature extraction pipeline.

Streams images from disk in configurable batches, computes the same 18 ODD
features as feature_extraction.py plus a rule-based ODD mode, and
checkpoints progress every N images so a long run can be safely interrupted
and resumed without reprocessing already-completed images.
"""

import argparse
import glob
import os
from typing import List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.perception.data_pipeline import IMAGE_SIZE, VOID_LABEL
from src.perception.feature_extraction import compute_features, run_detection
from src.odd.odd_classifier import FeatureScaler, assign_mode  # noqa: F401 (FeatureScaler needed for joblib.load)
from src.monitoring.perturbation_engine import calculate_metrics
from src.perception.segnet_model import load_segnet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("Warning: No GPU detected. full_dataset_pipeline will run on CPU.")


def discover_images(data_dir: str) -> List[Tuple[str, Optional[str]]]:
    """Discovers every image in the dataset, paired with its label if one exists.

    Args:
        data_dir: Path to the idd20k_lite/ (or larger IDD-20K) directory.

    Returns:
        A sorted list of (image_path, label_path_or_None) tuples across the
        train, val, and test splits. Only train/val have ground-truth masks.
    """
    pairs: List[Tuple[str, Optional[str]]] = []
    for split in ("train", "val", "test"):
        image_paths = sorted(
            glob.glob(os.path.join(data_dir, "leftImg8bit", split, "*", "*_image.jpg"))
        )
        for img_path in image_paths:
            lbl_path = img_path.replace("leftImg8bit", "gtFine").replace(
                "_image.jpg", "_label.png"
            )
            pairs.append((img_path, lbl_path if os.path.isfile(lbl_path) else None))
    return pairs


def _load_checkpoint(checkpoint_path: str) -> pd.DataFrame:
    """Loads an in-progress checkpoint CSV, or an empty DataFrame if absent."""
    if os.path.isfile(checkpoint_path):
        return pd.read_csv(checkpoint_path)
    return pd.DataFrame()


def run_full_dataset_pipeline(
    data_dir: str,
    segnet_checkpoint: str,
    yolo_weights: str,
    feature_scaler_path: str,
    batch_size: int = 32,
    checkpoint_interval: int = 500,
    checkpoint_path: str = "full_dataset_checkpoint.csv",
    output_path: str = "full_dataset_labeled.csv",
) -> pd.DataFrame:
    """Extracts features + ODD labels for every image, with resumable checkpointing.

    assign_mode() operates on MinMax-scaled features (see odd_classifier.py),
    so each image's raw features are scaled through the same fitted
    FeatureScaler used to train the ODD classifier before labeling -- this
    module must therefore run after odd_classifier.py has produced
    feature_scaler_path at least once.

    Args:
        data_dir: Path to the idd20k_lite/ (or IDD-20K) directory.
        segnet_checkpoint: Path to a trained SegNet .pth file.
        yolo_weights: Path to YOLOv8 weights (e.g. yolov8n.pt).
        feature_scaler_path: Path to the FeatureScaler saved by
            odd_classifier.py (feature_scaler.pkl).
        batch_size: Number of images loaded/inferred per inner batch.
        checkpoint_interval: Flush progress to disk every N processed images.
        checkpoint_path: Where in-progress results are checkpointed.
        output_path: Final output CSV path once all images are processed.

    Returns:
        A DataFrame with one row per image: 18 raw features, 'mode',
        'image_path', 'has_mask', and 'miou' (NaN where no mask exists).
    """
    pairs = discover_images(data_dir)
    total = len(pairs)

    done_df = _load_checkpoint(checkpoint_path)
    done_paths = set(done_df["image_path"]) if not done_df.empty else set()
    remaining = [p for p in pairs if p[0] not in done_paths]

    print(
        f"Total images: {total}. Already processed: {len(done_paths)}. "
        f"Remaining: {len(remaining)}."
    )

    segnet = load_segnet(segnet_checkpoint, device=DEVICE)
    yolo = YOLO(yolo_weights)
    feature_scaler: FeatureScaler = joblib.load(feature_scaler_path)

    buffer_rows: List[dict] = []

    def _flush() -> None:
        if not buffer_rows:
            return
        new_df = pd.DataFrame(buffer_rows)
        combined = pd.concat([_load_checkpoint(checkpoint_path), new_df], ignore_index=True)
        combined.to_csv(checkpoint_path, index=False)
        buffer_rows.clear()

    for i in tqdm(range(0, len(remaining), batch_size), desc="Batches"):
        batch = remaining[i : i + batch_size]

        loaded_batch = []
        images = []
        for img_path, lbl_path in batch:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: could not read '{img_path}', skipping.")
                continue
            img = cv2.resize(img, IMAGE_SIZE)
            loaded_batch.append((img_path, lbl_path))
            images.append(img)

        if not images:
            continue

        images_t = torch.tensor(np.array(images)).permute(0, 3, 1, 2).float().to(DEVICE) / 255.0
        with torch.no_grad():
            outputs = segnet(images_t)
        pred_masks = outputs.argmax(dim=1).cpu().numpy().astype(np.uint8)

        for (img_path, lbl_path), img, mask in zip(loaded_batch, images, pred_masks):
            detections = run_detection(img, yolo)
            features = compute_features(img, mask, detections)

            scaled_row = feature_scaler.transform(pd.DataFrame([features])).iloc[0]
            mode = assign_mode(scaled_row)

            miou = np.nan
            if lbl_path is not None:
                lbl = cv2.imread(lbl_path, 0)
                lbl = cv2.resize(lbl, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
                lbl = lbl.copy()
                lbl[lbl == 255] = VOID_LABEL
                lbl[lbl > VOID_LABEL] = VOID_LABEL
                miou, _ = calculate_metrics(
                    torch.tensor(mask), torch.tensor(lbl), num_classes=8
                )

            row = dict(features)
            row["mode"] = mode
            row["image_path"] = img_path
            row["has_mask"] = lbl_path is not None
            row["miou"] = miou
            buffer_rows.append(row)

            if len(buffer_rows) >= checkpoint_interval:
                _flush()

    _flush()

    final_df = _load_checkpoint(checkpoint_path)
    final_df.to_csv(output_path, index=False)
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"Saved {len(final_df)} labeled rows -> '{output_path}'")

    masked = final_df[final_df["has_mask"]]
    print("=== Full Dataset Pipeline: Final Stats ===")
    print(f"Total images processed: {len(final_df)}")
    print("Class distribution:")
    print(final_df["mode"].value_counts())
    if len(masked) > 0:
        print(
            f"Mean mIoU (over {len(masked)}/{len(final_df)} images with "
            f"ground-truth masks): {masked['miou'].mean():.4f}"
        )
    else:
        print("No images had ground-truth masks; mIoU not computed.")

    takeover = final_df[final_df["mode"] == "Takeover"]
    if not takeover.empty:
        numeric_cols = takeover.select_dtypes(include=[np.number]).columns
        print("Top 5 mean feature values for Takeover-labeled images:")
        print(takeover[numeric_cols].mean().sort_values(ascending=False).head(5))

    return final_df


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the full-dataset pipeline."""
    from src.common.paths import DATA_DIR

    parser = argparse.ArgumentParser(
        description="Full-dataset (IDD-20K-scale) feature extraction pipeline."
    )
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_interval", type=int, default=500)
    return parser.parse_args()


if __name__ == "__main__":
    from src.common.paths import (
        FEATURE_SCALER_PATH,
        FULL_DATASET_CHECKPOINT_CSV,
        FULL_DATASET_LABELED_CSV,
        SEGNET_CHECKPOINT,
        YOLO_WEIGHTS,
        ensure_output_dirs,
    )

    ensure_output_dirs()
    args = _parse_args()
    run_full_dataset_pipeline(
        data_dir=args.data_dir,
        segnet_checkpoint=SEGNET_CHECKPOINT,
        yolo_weights=YOLO_WEIGHTS,
        feature_scaler_path=FEATURE_SCALER_PATH,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=FULL_DATASET_CHECKPOINT_CSV,
        output_path=FULL_DATASET_LABELED_CSV,
    )
