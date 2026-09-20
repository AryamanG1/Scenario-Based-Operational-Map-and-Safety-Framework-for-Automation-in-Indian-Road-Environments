"""Stage 2 (Perception Layer): object-detection benchmarking toolkit.

Implements the standard detection-metrics formulas surveyed in Nawaz et al.
2023 ("Robust Cognitive Capability in Autonomous Driving Using Sensor
Fusion Techniques: A Survey", IEEE T-ITS), Section V -- mAP, precision,
recall, N-point interpolated AP, orientation similarity (AOS/APH), and the
AUC-based (Waymo-style) AP -- and applies the geometric ones (mAP/AP/
precision/recall) to benchmark YOLOv8 vehicle detections against IDD-Lite.

Two ground-truth sources are supported, scored by identical matching code
(`_score_matches`) so their numbers are directly comparable:

  * `evaluate_vehicle_detections` (IDD-Lite). `_inst_label.png` was verified
    byte-identical to `_label.png` (no real per-instance IDs exist in this
    "Lite" variant), so boxes are derived from the semantic mask's Vehicle
    class (id 3) via connected-component analysis -- each blob approximates
    one vehicle. This under-counts touching/overlapping vehicles and scores
    COCO four-wheelers against IDD's whole Vehicle class (which also holds
    motorcycles, bicycles and autorickshaws), so recall is structurally
    understated. Documented here as a real data limitation, not a hidden
    approximation.

  * `evaluate_vehicle_detections_from_json` (IDD117K-Detection). Real
    human-annotated per-object boxes, so touching vehicles are counted
    individually and the ground-truth class set is matched exactly to the
    detector's. This is the preferred benchmark where IDD117K is available;
    the mask-based one is kept so the two can be reported side by side.

AOS/APH require a per-detection orientation angle, which a 2D
bounding-box-only pipeline (ours) does not produce. Those two formulas are
still implemented exactly per the paper and unit-tested against synthetic
orientation deltas, but are not part of evaluate_vehicle_detections()'s
practical output.
"""

import math
import os
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from src.perception.feature_extraction import CONF_THRESHOLD, VEHICLES, load_yolo, run_detection

VEHICLE_CLASS_ID = 3  # IDD-Lite level3Id 'vehicles' class -- see feature_extraction.py
MIN_BLOB_AREA = 20
DEFAULT_IOU_THRESHOLD = 0.5

Box = Tuple[int, int, int, int]  # (x, y, w, h)


def extract_pseudo_gt_boxes(
    mask: np.ndarray, class_id: int = VEHICLE_CLASS_ID, min_area: int = MIN_BLOB_AREA
) -> List[Box]:
    """Derives pseudo-ground-truth object boxes from a semantic mask.

    Args:
        mask: Semantic segmentation mask of shape (H, W).
        class_id: The class whose connected components become boxes.
        min_area: Minimum blob area (px) to keep -- filters segmentation
            noise/single-pixel misclassifications.

    Returns:
        A list of (x, y, w, h) boxes, one per surviving connected component.
    """
    binary = (mask == class_id).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes: List[Box] = []
    for label in range(1, num_labels):  # label 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, w, h))
    return boxes


def compute_iou(box_a: Box, box_b: Box) -> float:
    """Computes intersection-over-union between two (x, y, w, h) boxes.

    Args:
        box_a: First box.
        box_b: Second box.

    Returns:
        IoU in [0.0, 1.0].
    """
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def match_detections_to_gt(
    scored_boxes: List[Tuple[Box, float]], gt_boxes: List[Box], iou_threshold: float
) -> List[bool]:
    """Greedily matches confidence-sorted detections to ground-truth boxes.

    Each ground-truth box can be claimed by at most one detection (its
    highest-IoU match among detections processed so far, in descending
    confidence order) -- the standard greedy matching rule used by
    KITTI/Pascal VOC-style AP evaluation.

    Args:
        scored_boxes: (box, confidence) pairs for one image, any order.
        gt_boxes: Ground-truth boxes for the same image.
        iou_threshold: Minimum IoU for a match to count as a true positive.

    Returns:
        A list of True/False (is-true-positive) flags, ordered by
        descending confidence (matching the input order after sorting).
    """
    ordered = sorted(scored_boxes, key=lambda item: item[1], reverse=True)
    claimed = [False] * len(gt_boxes)
    tp_flags: List[bool] = []

    for box, _ in ordered:
        best_iou = 0.0
        best_idx = -1
        for idx, gt_box in enumerate(gt_boxes):
            if claimed[idx]:
                continue
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            claimed[best_idx] = True
            tp_flags.append(True)
        else:
            tp_flags.append(False)

    return tp_flags


def precision_recall(n_tp: int, n_all: int, n_all_gt: int) -> Tuple[float, float]:
    """Computes precision and recall (Nawaz et al. Eqs. 2-3).

    Args:
        n_tp: Number of true positives.
        n_all: Number of all detections made.
        n_all_gt: Number of all ground-truth objects.

    Returns:
        A tuple (precision, recall). Both 0.0 if their denominator is 0.
    """
    precision = n_tp / n_all if n_all > 0 else 0.0
    recall = n_tp / n_all_gt if n_all_gt > 0 else 0.0
    return precision, recall


def interpolated_ap(sorted_tp_flags: List[bool], num_gt: int, num_recall_points: int = 11) -> float:
    """Computes N-point interpolated Average Precision (Nawaz et al. Eqs. 4-5).

    AP = (1/N) * sum_{r in S} P_interp(r),  P_interp(r) = max{p(r_hat) : r_hat >= r}

    where S is a set of N equally spaced recall levels. KITTI historically
    used S11 = [0, 0.1, ..., 1.0] (num_recall_points=11, the default here);
    the 2019+ KITTI convention uses S40 (num_recall_points=40).

    Args:
        sorted_tp_flags: True/False flags in descending-confidence order,
            across the whole evaluation set (not just one image).
        num_gt: Total number of ground-truth objects across the set.
        num_recall_points: N, the number of equally spaced recall levels.

    Returns:
        The interpolated AP in [0.0, 1.0]. 0.0 if num_gt is 0.
    """
    if num_gt == 0:
        return 0.0

    precisions: List[float] = []
    recalls: List[float] = []
    tp_cumulative = 0
    for rank, is_tp in enumerate(sorted_tp_flags, start=1):
        tp_cumulative += int(is_tp)
        precisions.append(tp_cumulative / rank)
        recalls.append(tp_cumulative / num_gt)

    recall_levels = np.linspace(0.0, 1.0, num_recall_points)
    interpolated_precisions = []
    for r in recall_levels:
        candidates = [p for p, rec in zip(precisions, recalls) if rec >= r]
        interpolated_precisions.append(max(candidates) if candidates else 0.0)

    return float(np.mean(interpolated_precisions))


def auc_average_precision(sorted_tp_flags: List[bool], num_gt: int) -> float:
    """Computes AUC-based Average Precision (Waymo convention, Nawaz Eq. 8).

    AP = 100 * integral_0^1 max{p(r) : r' >= r} dr, approximated here by
    trapezoidal integration over the empirical (recall, max-precision-at-
    or-above-recall) curve rather than fixed interpolation points.

    Args:
        sorted_tp_flags: True/False flags in descending-confidence order.
        num_gt: Total number of ground-truth objects.

    Returns:
        AP scaled to [0.0, 100.0]. 0.0 if num_gt is 0.
    """
    if num_gt == 0:
        return 0.0

    precisions: List[float] = []
    recalls: List[float] = []
    tp_cumulative = 0
    for rank, is_tp in enumerate(sorted_tp_flags, start=1):
        tp_cumulative += int(is_tp)
        precisions.append(tp_cumulative / rank)
        recalls.append(tp_cumulative / num_gt)

    # Monotonic envelope: precision at recall r is the max precision at any
    # recall >= r (standard AP smoothing to remove the sawtooth PR curve).
    envelope = np.maximum.accumulate(precisions[::-1])[::-1]

    # The PR curve implicitly starts at recall=0 (before any detection is
    # made); without prepending that origin point, trapezoidal integration
    # would silently drop the area between recall=0 and the first
    # detection's recall, undercounting AP for every detector.
    recalls_with_origin = np.concatenate([[0.0], recalls])
    envelope_with_origin = np.concatenate([[envelope[0]], envelope])

    return float(100.0 * np.trapezoid(envelope_with_origin, recalls_with_origin))


def compute_mean_ap(ap_per_class: Dict[str, float]) -> float:
    """Computes mAP as the mean AP across classes (Nawaz et al. Eq. 1).

    Args:
        ap_per_class: Mapping of class name -> AP.

    Returns:
        The unweighted mean of the per-class APs, or 0.0 if empty.
    """
    if not ap_per_class:
        return 0.0
    return float(np.mean(list(ap_per_class.values())))


def orientation_similarity(deltas: Sequence[float], is_true_positive: Sequence[bool]) -> float:
    """Computes Average Orientation Similarity at one recall level (Nawaz Eq. 7).

    s(r) = (1/|D(r)|) * sum_{i in D(r)} [(1 + cos(angle_delta_i)) / 2] * delta_i

    where delta_i = 1 if detection i is a true positive, else 0 (standard
    KITTI AOS convention -- orientation similarity is only accumulated over
    true positives).

    Args:
        deltas: Angular deviation (radians) between predicted and
            ground-truth heading, one per detection at this recall level.
        is_true_positive: Parallel True/False true-positive flags.

    Returns:
        The average orientation similarity in [0.0, 1.0]. 0.0 if empty.
    """
    if not deltas:
        return 0.0
    total = sum(
        ((1.0 + math.cos(angle)) / 2.0) * float(tp)
        for angle, tp in zip(deltas, is_true_positive)
    )
    return total / len(deltas)


def average_orientation_score(
    recall_levels: Sequence[float],
    similarity_by_recall: Sequence[float],
) -> float:
    """Computes AOS across recall levels (Nawaz et al. Eq. 6).

    AOS = (1/N) * sum_{r in S} max_{r_hat >= r} s(r_hat)

    Args:
        recall_levels: The N equally spaced recall levels S.
        similarity_by_recall: s(r) computed via orientation_similarity() at
            each of a finer set of sampled recall points (same length/order
            assumption as recall_levels for this simplified interface).

    Returns:
        The AOS in [0.0, 1.0].
    """
    if not recall_levels:
        return 0.0
    interpolated = [
        max(
            (s for r_hat, s in zip(recall_levels, similarity_by_recall) if r_hat >= r),
            default=0.0,
        )
        for r in recall_levels
    ]
    return float(np.mean(interpolated))


def _score_matches(
    all_scored: List[Tuple[Box, float, int]],
    gt_boxes_per_image: List[List[Box]],
    total_gt: int,
    iou_threshold: float,
) -> Dict[str, float]:
    """Greedily matches scored detections to ground truth and computes metrics.

    Shared by both ground-truth sources (mask-derived pseudo-boxes and
    IDD117K's real annotated boxes) so that the two are scored by identical
    code and their numbers are directly comparable.

    Detections are matched highest-confidence-first, each ground-truth box may
    be claimed at most once, and unmatched detections count as false positives.

    Args:
        all_scored: (box, confidence, image_idx) triples, any order.
        gt_boxes_per_image: Ground-truth boxes, indexed by image_idx.
        total_gt: Total number of ground-truth boxes.
        iou_threshold: IoU required for a detection to count as a match.

    Returns:
        A dict with 'ap' (11-point interpolated), 'auc_ap', 'mean_precision',
        'mean_recall', 'num_ground_truth', and 'num_detections'.
    """
    all_scored = sorted(all_scored, key=lambda item: item[1], reverse=True)

    claimed_per_image = [[False] * len(boxes) for boxes in gt_boxes_per_image]
    tp_flags: List[bool] = []
    for box, _, image_idx in all_scored:
        gt_boxes = gt_boxes_per_image[image_idx]
        claimed = claimed_per_image[image_idx]

        best_iou, best_idx = 0.0, -1
        for idx, gt_box in enumerate(gt_boxes):
            if claimed[idx]:
                continue
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_idx = iou, idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            claimed[best_idx] = True
            tp_flags.append(True)
        else:
            tp_flags.append(False)

    final_precision, final_recall = precision_recall(sum(tp_flags), len(tp_flags), total_gt)
    return {
        "ap": interpolated_ap(tp_flags, total_gt, num_recall_points=11),
        "auc_ap": auc_average_precision(tp_flags, total_gt),
        "mean_precision": final_precision,
        "mean_recall": final_recall,
        "num_ground_truth": total_gt,
        "num_detections": len(all_scored),
    }


def evaluate_vehicle_detections(
    images: np.ndarray,
    labels: np.ndarray,
    yolo_model,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict[str, float]:
    """Benchmarks YOLO vehicle detections against IDD-Lite ground truth.

    Ground truth comes from connected-component analysis of the Vehicle
    class in the *ground-truth* semantic mask (not the SegNet prediction),
    since this module measures YOLO's own accuracy independent of SegNet.

    Note this scores COCO four-wheelers (car/bus/truck) against IDD's whole
    Vehicle class, which also contains motorcycles, bicycles and
    autorickshaws -- those extra ground-truth boxes can never be matched, so
    recall is structurally understated. `evaluate_vehicle_detections_from_json`
    does not have this problem and is the preferred benchmark where
    IDD117K-Detection is available.

    Args:
        images: uint8 array of shape (M, H, W, 3).
        labels: uint8 ground-truth semantic mask array of shape (M, H, W).
        yolo_model: A loaded Ultralytics YOLO model.
        iou_threshold: IoU required for a detection to count as a match.

    Returns:
        A dict with 'ap' (11-point interpolated), 'auc_ap', 'mean_precision',
        'mean_recall', 'num_ground_truth', and 'num_detections'.
    """
    all_scored: List[Tuple[Box, float, int]] = []  # (box, confidence, image_idx)
    gt_boxes_per_image: List[List[Box]] = []
    total_gt = 0

    for i in range(len(images)):
        gt_boxes = extract_pseudo_gt_boxes(labels[i])
        gt_boxes_per_image.append(gt_boxes)
        total_gt += len(gt_boxes)

        detections = run_detection(images[i], yolo_model)
        for det in detections:
            if det["class"] not in VEHICLES or det["confidence"] < CONF_THRESHOLD:
                continue
            all_scored.append((tuple(det["bbox"]), det["confidence"], i))

    return _score_matches(all_scored, gt_boxes_per_image, total_gt, iou_threshold)


def evaluate_vehicle_detections_from_json(
    pairs: Sequence[Tuple[str, str]],
    yolo_model,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    gt_groups: Sequence[str] = None,
    detect_classes: Sequence[str] = None,
) -> Dict[str, float]:
    """Benchmarks YOLO detections against IDD117K-Detection's real boxes.

    This is the honest version of `evaluate_vehicle_detections`: ground truth
    is human-annotated per-object boxes rather than connected components of a
    semantic mask, so touching vehicles are counted individually instead of
    merging into one blob, and the ground-truth class set can be matched
    exactly to the detector's class set.

    Frames are streamed from disk one at a time, so this is safe to run over
    the full split.

    Args:
        pairs: (image_path, annotation_path) pairs from
            `idd_detection_loader.list_pairs` (optionally subsampled via
            `sample_pairs`).
        yolo_model: A loaded Ultralytics YOLO model.
        iou_threshold: IoU required for a detection to count as a match.
        gt_groups: Annotation groups to treat as ground truth. Defaults to
            `idd_detection_loader.DEFAULT_GT_GROUPS` (four-wheelers), which
            mirrors `detect_classes`.
        detect_classes: YOLO class names to score. Defaults to
            `feature_extraction.VEHICLES`.

    Returns:
        A dict with the same keys as `evaluate_vehicle_detections`.
    """
    from src.perception.idd_detection_loader import (
        DEFAULT_GT_GROUPS,
        filter_boxes_by_group,
        iter_frames,
    )

    gt_groups = DEFAULT_GT_GROUPS if gt_groups is None else gt_groups
    detect_classes = VEHICLES if detect_classes is None else detect_classes

    all_scored: List[Tuple[Box, float, int]] = []
    gt_boxes_per_image: List[List[Box]] = []
    total_gt = 0

    for i, (img, boxes) in enumerate(iter_frames(pairs)):
        gt_boxes = filter_boxes_by_group(boxes, gt_groups)
        gt_boxes_per_image.append(gt_boxes)
        total_gt += len(gt_boxes)

        for det in run_detection(img, yolo_model):
            if det["class"] not in detect_classes or det["confidence"] < CONF_THRESHOLD:
                continue
            all_scored.append((tuple(det["bbox"]), det["confidence"], i))

    return _score_matches(all_scored, gt_boxes_per_image, total_gt, iou_threshold)


if __name__ == "__main__":
    import argparse
    import json

    from ultralytics import YOLO

    from src.common.paths import (
        DATA_DIR,
        DETECTION_BENCHMARK_JSON,
        IDD117K_95K_DIR,
        YOLO_WEIGHTS,
        ensure_output_dirs,
    )

    parser = argparse.ArgumentParser(description="Benchmark YOLOv8 vehicle detection.")
    parser.add_argument(
        "--dataset",
        choices=("idd-lite", "idd117k"),
        default="idd117k",
        help="Ground-truth source: IDD-Lite mask pseudo-boxes, or IDD117K real boxes.",
    )
    parser.add_argument("--split", default="val", help="IDD117K split to benchmark on.")
    parser.add_argument("--limit", type=int, default=2000, help="Frames to evaluate (0 = all).")
    args = parser.parse_args()

    ensure_output_dirs()
    yolo = load_yolo(YOLO_WEIGHTS)

    if args.dataset == "idd-lite":
        from src.perception.data_pipeline import load_and_clean_dataset

        images, labels = load_and_clean_dataset(DATA_DIR)
        n = args.limit or len(images)
        results = evaluate_vehicle_detections(images[:n], labels[:n], yolo)
    else:
        from src.perception.idd_detection_loader import list_pairs, sample_pairs

        pairs = sample_pairs(list_pairs(IDD117K_95K_DIR, args.split), args.limit or None)
        print(f"Benchmarking on {len(pairs)} frame(s) from IDD117K '{args.split}'.")
        results = evaluate_vehicle_detections_from_json(pairs, yolo)

    print(f"=== YOLOv8 Vehicle Detection Benchmark ({args.dataset}) ===")
    for key, value in results.items():
        print(f"{key}: {value}")

    with open(DETECTION_BENCHMARK_JSON, "w") as handle:
        json.dump({"dataset": args.dataset, "split": args.split, **results}, handle, indent=2)
    print(f"Wrote -> '{DETECTION_BENCHMARK_JSON}'")
