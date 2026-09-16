"""IDD-20K-II polygon-to-mask pipeline: the SegNet training-data upgrade.

IDD-Lite gives SegNet only 1,403 training images at 320x224. IDD-20K-II gives
7,034 at native 1920x1080 with full semantic ground truth -- but that ground
truth is stored as **polygon JSON, not pre-rasterized masks**, so
`data_pipeline.load_and_clean_dataset` (which reads `*_label.png`) cannot read
it. This module rasterizes those polygons into the same 8-class uint8 masks the
rest of the project already consumes, so `segnet_model.train_segnet` and every
mask-derived feature work unchanged.

Layout:

    idd20kII/
      leftImg8bit/{train,val}/<seq>/<frame>_leftImg8bit.jpg
      gtFine/{train,val}/<seq>/<frame>_gtFine_polygons.json

Polygon file structure:

    {"imgHeight": 1080, "imgWidth": 1920,
     "objects": [{"label": "road", "polygon": [[84.2, 1060.4], [790.4, 513.5], ...]}, ...]}

Objects are rasterized in file order, so later objects paint over earlier ones
-- the standard back-to-front convention these annotations are authored in.

**Class-mapping warning.** `docs/DATASET_NOTES.md` records that this project
already shipped one wrong class mapping and caught it only by inspecting the
pixel data, and it explicitly instructs re-verifying the scheme before trusting
any label-derived feature. IDD-20K-II's polygon labels are IDD's fine-grained
`level4Id` names, which must be grouped into the project's 8 `level3Id`
classes. `LABEL_TO_CLASS` below is therefore a **draft**: run

    python -m src.perception.idd20k_polygon_pipeline --scan

to enumerate the real label vocabulary first. Unmapped labels are reported
rather than silently folded into the void class.
"""

import argparse
import collections
import glob
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_SIZE = (320, 224)  # (width, height) -- matches data_pipeline.IMAGE_SIZE
NUM_CLASSES = 8
VOID_LABEL = 7

# Project 8-class level3Id scheme (verified for IDD-Lite, see DATASET_NOTES.md):
#   0 drivable-area  1 non-drivable-area  2 living-things  3 vehicles
#   4 road-side-objects  5 far-objects  6 sky  7 void
LABEL_TO_CLASS: Dict[str, int] = {
    # 0 -- drivable area
    "road": 0, "parking": 0, "drivable fallback": 0,
    # 1 -- non-drivable area
    "sidewalk": 1, "rail track": 1, "non-drivable fallback": 1, "curb": 1,
    # 2 -- living things
    "person": 2, "rider": 2, "animal": 2,
    # 3 -- vehicles
    "car": 3, "truck": 3, "bus": 3, "motorcycle": 3, "bicycle": 3,
    "autorickshaw": 3, "vehicle fallback": 3, "caravan": 3, "trailer": 3, "train": 3,
    # 4 -- road-side objects
    "building": 4, "wall": 4, "fence": 4, "guard rail": 4, "billboard": 4,
    "traffic sign": 4, "traffic light": 4, "pole": 4, "polegroup": 4,
    "obs-str-bar-fallback": 4, "bridge": 4, "tunnel": 4,
    # 5 -- far objects
    "vegetation": 5,
    # 6 -- sky
    "sky": 6,
    # 7 -- void / ignore
    "unlabeled": 7, "out of roi": 7, "ego vehicle": 7, "rectification border": 7,
    "license plate": 7, "ground": 7, "fallback background": 7,
}


def list_pairs(dataset_dir: str, split: str = "train") -> List[Tuple[str, str]]:
    """Pairs every image in a split with its polygon annotation.

    Args:
        dataset_dir: Path to the extracted `idd20kII/` directory.
        split: "train" or "val" ("test" has no released ground truth).

    Returns:
        A sorted list of (image_path, polygon_json_path) pairs. Images with no
        matching annotation are skipped.
    """
    image_paths = sorted(
        glob.glob(os.path.join(dataset_dir, "leftImg8bit", split, "*", "*_leftImg8bit.jpg"))
    )

    pairs: List[Tuple[str, str]] = []
    missing = 0
    for img_path in image_paths:
        ann_path = img_path.replace(
            os.sep + "leftImg8bit" + os.sep, os.sep + "gtFine" + os.sep
        ).replace("_leftImg8bit.jpg", "_gtFine_polygons.json")
        if not os.path.isfile(ann_path):
            missing += 1
            continue
        pairs.append((img_path, ann_path))

    if missing:
        print(f"Warning: {missing} image(s) in split '{split}' have no polygon file; skipped.")
    return pairs


def rasterize_polygons(
    ann_path: str, size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[np.ndarray, List[str]]:
    """Renders a polygon annotation file into a dense 8-class mask.

    Polygon vertices are scaled from native resolution to `size` with
    **independent** x and y factors: 1920x1080 (aspect 1.78) -> 320x224 (1.43)
    is aspect-distorting, and a single uniform factor would shear every mask
    relative to its image.

    Args:
        ann_path: Path to a `*_gtFine_polygons.json` file.
        size: Target (width, height).

    Returns:
        A tuple (mask, unmapped): `mask` is a uint8 array of shape
        (size[1], size[0]) holding class IDs, initialized to VOID_LABEL;
        `unmapped` lists label names absent from LABEL_TO_CLASS (painted as
        void, but reported rather than hidden).
    """
    with open(ann_path) as handle:
        data = json.load(handle)

    src_w = int(data.get("imgWidth") or 0)
    src_h = int(data.get("imgHeight") or 0)
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"'{ann_path}' has no usable imgWidth/imgHeight.")

    scale_x = size[0] / float(src_w)
    scale_y = size[1] / float(src_h)

    mask = np.full((size[1], size[0]), VOID_LABEL, dtype=np.uint8)
    unmapped: List[str] = []

    # File order is back-to-front: later objects paint over earlier ones.
    for obj in data.get("objects", []):
        label = str(obj.get("label", "")).strip().lower()
        polygon = obj.get("polygon") or []
        if len(polygon) < 3:
            continue

        if label in LABEL_TO_CLASS:
            class_id = LABEL_TO_CLASS[label]
        else:
            unmapped.append(label)
            class_id = VOID_LABEL

        points = np.array(
            [[round(px * scale_x), round(py * scale_y)] for px, py in polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [points], int(class_id))

    return mask, unmapped


def load_split(
    dataset_dir: str,
    split: str = "train",
    limit: Optional[int] = None,
    seed: int = 0,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a split as (images, labels) arrays ready for train_segnet.

    Applies the same cleaning rules as `data_pipeline.load_and_clean_dataset`:
    unreadable files, near-empty masks (<=1 distinct class) and masks with no
    drivable-area (class 0) pixels are dropped.

    Memory note: 7,034 train images at 320x224x3 is ~1.5 GB, so materializing
    the arrays is fine here -- unlike IDD117K, which is ~20.8 GB and must be
    streamed (see idd_detection_loader / extract_features_streaming).

    Args:
        dataset_dir: Path to the extracted `idd20kII/` directory.
        split: "train" or "val".
        limit: Optional cap on frames (deterministically sampled).
        seed: Sampling seed.
        size: Target (width, height).

    Returns:
        A tuple (images, labels): uint8 arrays of shape (M, H, W, 3) and
        (M, H, W).
    """
    pairs = list_pairs(dataset_dir, split)
    if limit is not None and limit < len(pairs):
        pairs = sorted(random.Random(seed).sample(pairs, limit))

    images: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    removed = 0
    unmapped_seen: "collections.Counter" = collections.Counter()

    for index, (img_path, ann_path) in enumerate(pairs, start=1):
        img = cv2.imread(img_path)
        if img is None:
            removed += 1
            continue

        mask, unmapped = rasterize_polygons(ann_path, size)
        unmapped_seen.update(unmapped)

        distinct = np.unique(mask)
        if len(distinct) <= 1 or 0 not in distinct:
            removed += 1
            continue

        images.append(cv2.resize(img, size))
        labels.append(mask)

        if index % 1000 == 0:
            print(f"  ... rasterized {index}/{len(pairs)} frames")

    print(f"Removed {removed} improper/corrupted frames. Clean split size: {len(images)}")
    if unmapped_seen:
        print(
            f"!! {len(unmapped_seen)} label(s) absent from LABEL_TO_CLASS were painted as void: "
            f"{', '.join(sorted(unmapped_seen))}"
        )

    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.uint8)


def scan_label_vocabulary(
    dataset_dir: str, splits: Sequence[str] = ("train", "val"), limit: Optional[int] = 2000
) -> "collections.Counter":
    """Counts every distinct polygon label name in the annotations.

    Run this before trusting LABEL_TO_CLASS -- see the module docstring.

    Args:
        dataset_dir: Path to the extracted `idd20kII/` directory.
        splits: Splits to scan.
        limit: Annotation files to sample per split (None scans all).

    Returns:
        A Counter mapping raw label name -> occurrence count.
    """
    counts: "collections.Counter" = collections.Counter()
    for split in splits:
        pairs = list_pairs(dataset_dir, split)
        if limit is not None and limit < len(pairs):
            pairs = sorted(random.Random(0).sample(pairs, limit))
        for _, ann_path in pairs:
            with open(ann_path) as handle:
                data = json.load(handle)
            for obj in data.get("objects", []):
                counts[str(obj.get("label", "")).strip().lower()] += 1
        print(f"Scanned {len(pairs)} polygon file(s) in split '{split}'.")
    return counts


def unmapped_labels(counts: "collections.Counter") -> List[str]:
    """Reports label names present in the data but absent from LABEL_TO_CLASS.

    Args:
        counts: Output of `scan_label_vocabulary`.

    Returns:
        A sorted list of unmapped names. Non-empty means LABEL_TO_CLASS needs
        updating before the rasterized masks can be trusted.
    """
    return sorted(name for name in counts if name not in LABEL_TO_CLASS)


def class_distribution(labels: np.ndarray) -> Dict[int, float]:
    """Computes each class's share of all labeled pixels.

    Useful both as a sanity check on the mapping (a class at 0.0 means nothing
    mapped to it) and as input to train_segnet's `class_weights`.

    Args:
        labels: uint8 mask array of shape (M, H, W).

    Returns:
        A dict mapping class ID -> pixel fraction.
    """
    counts = np.bincount(labels.reshape(-1), minlength=NUM_CLASSES)
    total = int(counts.sum()) or 1
    return {class_id: float(counts[class_id]) / total for class_id in range(NUM_CLASSES)}


def _main() -> None:
    """CLI: scan the label vocabulary, or rasterize and cache a split."""
    from src.common.paths import IDD20K_DIR, IDD20K_VOCAB_JSON, ensure_output_dirs

    parser = argparse.ArgumentParser(description="IDD-20K-II polygon -> mask pipeline.")
    parser.add_argument("--scan", action="store_true", help="Enumerate the polygon label vocabulary.")
    parser.add_argument("--check", action="store_true", help="Rasterize a small sample and report class balance.")
    parser.add_argument("--dataset_dir", default=IDD20K_DIR, help="Path to idd20kII/.")
    parser.add_argument("--split", default="train", help="Split to operate on.")
    parser.add_argument("--limit", type=int, default=2000, help="Frames to sample (0 = all).")
    args = parser.parse_args()

    if not args.scan and not args.check:
        parser.error("nothing to do: pass --scan and/or --check")

    ensure_output_dirs()

    if args.scan:
        counts = scan_label_vocabulary(args.dataset_dir, limit=args.limit or None)
        print(f"\n{'label':<28} {'count':>10}  class")
        print("-" * 54)
        for name, count in counts.most_common():
            print(f"{name:<28} {count:>10}  {LABEL_TO_CLASS.get(name, 'UNMAPPED')}")

        missing = unmapped_labels(counts)
        if missing:
            print(f"\n!! {len(missing)} label(s) NOT in LABEL_TO_CLASS -- update it before training:")
            for name in missing:
                print(f"     {name}  ({counts[name]} occurrences)")
        else:
            print("\nAll labels found are present in LABEL_TO_CLASS.")

        with open(IDD20K_VOCAB_JSON, "w") as handle:
            json.dump({"counts": dict(counts), "unmapped": missing}, handle, indent=2)
        print(f"\nWrote vocabulary -> '{IDD20K_VOCAB_JSON}'")

    if args.check:
        images, labels = load_split(args.dataset_dir, args.split, limit=args.limit or None)
        print(f"\nimages {images.shape} {images.dtype} | labels {labels.shape} {labels.dtype}")
        print("\nclass pixel distribution (a 0.0000 row means nothing mapped to that class):")
        names = {
            0: "drivable-area", 1: "non-drivable-area", 2: "living-things", 3: "vehicles",
            4: "road-side-objects", 5: "far-objects", 6: "sky", 7: "void",
        }
        for class_id, fraction in class_distribution(labels).items():
            print(f"  {class_id} {names[class_id]:<20} {fraction:.4f}")


if __name__ == "__main__":
    _main()
