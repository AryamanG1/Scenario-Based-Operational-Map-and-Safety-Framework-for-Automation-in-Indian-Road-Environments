"""Module 4: YOLOv8 object detection + SegNet mask + image-stat feature extraction.

Runs YOLOv8 nano on every image, combines the detections with a SegNet
predicted semantic mask and raw image statistics to produce a 22-dimensional
feature vector per frame describing traffic, road, and visibility conditions
(the original 18-feature spec, plus num_animals/num_autorickshaws for the
capstone proposal's India-specific-entity requirement, plus a heuristic
pothole-candidate score/count -- see the class-scheme note below).

Two of the original 18 features were renamed after discovering that
IDD-Lite's semantic classes do not include a pothole or water/puddle class
at all (verified directly against the pixel data): 'pothole_score' and
'pothole_distance' are now 'non_drivable_area_score'/'non_drivable_area_distance'
(what class 1 actually is), and 'water_level' is now 'living_things_score'
(what class 2 actually is -- pedestrians/animals, not water).
"""

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ultralytics import YOLO

VEHICLES = ["car", "bus", "truck"]
TWO_WHEELERS = ["bicycle", "motorcycle"]
PEDESTRIANS = ["person"]
# India-specific entities (capstone proposal, Stage 2): COCO has no
# auto-rickshaw class, so it is approximated via a bbox heuristic below.
# Animals are COCO classes already, just not previously counted.
ANIMALS = ["cow", "horse", "dog", "sheep", "elephant", "bear"]
# IDD-Lite level3Id semantic class scheme -- VERIFIED against actual pixel
# data (spatial distribution, blob shape statistics, and direct visual
# rendering of image/label pairs), since the one-shot prompt this project
# started from claimed an incorrect mapping (it asserted 1=Pothole,
# 2=Water/Puddle, 3=Sidewalk, 6=Vehicle -- none of which match the real
# data). The real scheme has no dedicated pothole or water/puddle class at
# all:
#   0 = drivable-area   1 = non-drivable-area   2 = living-things (people/animals)
#   3 = vehicles         4 = road-side-objects   5 = far-objects (buildings/vegetation)
#   6 = sky              7 = unlabeled/void
ROAD_CLASS_ID = 0
NON_DRIVABLE_CLASS_ID = 1
LIVING_THINGS_CLASS_ID = 2
VEHICLE_CLASS_ID = 3
SKY_CLASS_ID = 6
DISTANCE_K = 500
CONF_THRESHOLD = 0.3

# Heuristic auto-rickshaw bbox filter, applied only to 'car'-class detections
# (COCO has no three-wheeler class). Auto-rickshaws photograph as small,
# boxy vehicles -- narrower width/height ratio and smaller area than a
# typical car at similar distance. This is a coarse, uncalibrated proxy
# (no ground-truth auto-rickshaw labels exist in IDD-Lite to tune against)
# and is documented as such; it is not a substitute for a trained detector.
AUTORICKSHAW_MIN_ASPECT = 0.55
AUTORICKSHAW_MAX_ASPECT = 1.05
AUTORICKSHAW_MAX_AREA_FRACTION = 0.04

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("Warning: No GPU detected. feature_extraction will run on CPU.")


def run_detection(img: np.ndarray, model: YOLO) -> List[Dict]:
    """Runs YOLOv8 object detection on a single image.

    Args:
        img: BGR image array of shape (H, W, 3).
        model: A loaded Ultralytics YOLO model.

    Returns:
        A list of detection dicts, each with 'class' (str), 'bbox'
        ([x, y, w, h] ints), and 'confidence' (float) keys.
    """
    results = model(img, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = box.xyxy[0]
        dets.append(
            {
                "class": label,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": float(box.conf[0]),
            }
        )
    return dets


def estimate_distance(bbox_height: float, k: float = DISTANCE_K) -> float:
    """Estimates a rough object distance from its bounding-box height.

    Args:
        bbox_height: Bounding box height in pixels.
        k: Calibration constant (larger k -> larger estimated distances).

    Returns:
        Estimated distance; smaller/farther boxes yield larger distances.
    """
    return k / (bbox_height + 1)


# Heuristic pothole-candidate detector constants. IDD-Lite has no dedicated
# pothole class (see the class-scheme note above), so there is no ground
# truth to calibrate these against -- they are hand-picked, documented
# defaults, not fitted values.
POTHOLE_DARKNESS_STD_FACTOR = 1.5
POTHOLE_MIN_AREA = 15
POTHOLE_MAX_AREA_FRACTION = 0.05
POTHOLE_MIN_FILL_RATIO = 0.4


def detect_pothole_candidates(gray: np.ndarray, mask: np.ndarray) -> Tuple[float, int]:
    """Heuristically flags dark, compact anomalies on the road as pothole candidates.

    This is an UNCALIBRATED exploratory heuristic, not a trained or
    validated pothole detector -- IDD-Lite's semantic classes have no
    pothole label, so there's no ground truth to tune or score this
    against. It is expected to also fire on shadows, oil stains, manhole
    covers, and other dark road-surface features, not exclusively real
    potholes; treat its output as a weak, best-effort signal.

    Pixels within the road region darker than
    (road_mean - POTHOLE_DARKNESS_STD_FACTOR * road_std) are candidates;
    morphological opening removes speckle noise; surviving connected
    components are kept only if their area falls within a plausible
    pothole size range and their fill ratio (area / bounding-box area)
    indicates a compact blob rather than a thin shadow streak.

    Args:
        gray: Single-channel grayscale image of shape (H, W).
        mask: Semantic segmentation mask of shape (H, W).

    Returns:
        A tuple (score, count): score is the fraction of the road region
        flagged as pothole-candidate pixels; count is the number of
        surviving candidate blobs. Both are 0 if there's no road region.
    """
    road_binary = (mask == ROAD_CLASS_ID).astype(np.uint8)
    road_area = int(road_binary.sum())
    if road_area == 0:
        return 0.0, 0

    road_pixels = gray[road_binary.astype(bool)]
    threshold = float(road_pixels.mean()) - POTHOLE_DARKNESS_STD_FACTOR * float(road_pixels.std())

    dark_mask = ((gray.astype(np.float32) < threshold) & road_binary.astype(bool)).astype(np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

    max_area = POTHOLE_MAX_AREA_FRACTION * road_area
    candidate_area = 0
    count = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < POTHOLE_MIN_AREA or area > max_area:
            continue
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        fill_ratio = area / (w * h) if w * h > 0 else 0.0
        if fill_ratio < POTHOLE_MIN_FILL_RATIO:
            continue
        candidate_area += area
        count += 1

    return candidate_area / road_area, count


def is_autorickshaw_like(detection: Dict, image_shape: Tuple[int, int]) -> bool:
    """Heuristically flags a 'car'-class detection as auto-rickshaw-like.

    COCO (and therefore YOLOv8) has no three-wheeler class, so an
    auto-rickshaw is always detected as 'car'. This applies a bounding-box
    aspect-ratio and relative-area filter to flag the subset of car
    detections that look more like a compact three-wheeler -- a coarse,
    uncalibrated proxy documented as such (see AUTORICKSHAW_* constants).

    Args:
        detection: One YOLO detection dict (as returned by run_detection()).
        image_shape: (height, width) of the source frame, for area
            normalization.

    Returns:
        True if the detection's box geometry matches the heuristic filter.
    """
    if detection["class"] != "car":
        return False

    _, _, w, h = detection["bbox"]
    if h == 0:
        return False

    aspect_ratio = w / h
    area_fraction = (w * h) / (image_shape[0] * image_shape[1])

    return (
        AUTORICKSHAW_MIN_ASPECT <= aspect_ratio <= AUTORICKSHAW_MAX_ASPECT
        and area_fraction <= AUTORICKSHAW_MAX_AREA_FRACTION
    )


def compute_features(
    img: np.ndarray, mask: np.ndarray, detections: List[Dict]
) -> Dict[str, float]:
    """Computes the 22 ODD-relevant scene features for a single frame.

    Args:
        img: BGR image array of shape (H, W, 3).
        mask: Semantic segmentation mask of shape (H, W), using the verified
            IDD-Lite scheme: 0=drivable-area, 1=non-drivable-area,
            2=living-things, 3=vehicles, 4=road-side-objects, 5=far-objects,
            6=sky, 7=unlabeled (see docs/DATASET_NOTES.md).
        detections: YOLO detections for this frame, as returned by
            run_detection().

    Returns:
        A dict mapping each of the 22 feature names to its computed value.
    """
    confident_dets = [d for d in detections if d["confidence"] >= CONF_THRESHOLD]

    vehicle_count = sum(1 for d in confident_dets if d["class"] in VEHICLES)
    num_two_wheelers = sum(1 for d in confident_dets if d["class"] in TWO_WHEELERS)
    num_pedestrians = sum(1 for d in confident_dets if d["class"] in PEDESTRIANS)
    num_animals = sum(1 for d in confident_dets if d["class"] in ANIMALS)
    num_autorickshaws = sum(
        1 for d in confident_dets if is_autorickshaw_like(d, img.shape[:2])
    )
    object_presence = 1 if len(detections) > 0 else 0

    if detections:
        object_distance = float(np.mean([estimate_distance(d["bbox"][3]) for d in detections]))
        detection_confidence = float(np.mean([d["confidence"] for d in detections]))
    else:
        object_distance = 0.0
        detection_confidence = 0.0

    vehicle_dets = [d for d in confident_dets if d["class"] in VEHICLES]
    if vehicle_dets:
        lead_vehicle_distance = float(min(estimate_distance(d["bbox"][3]) for d in vehicle_dets))
    else:
        lead_vehicle_distance = 0.0

    traffic_density = len(confident_dets)

    drivable_area = float(np.sum(mask == ROAD_CLASS_ID) / mask.size)
    road_rows = np.where(mask == ROAD_CLASS_ID)[0]
    road_end_distance = float(mask.shape[0] - np.min(road_rows)) if road_rows.size else 0.0

    # Class 1 = non-drivable-area (sidewalk/curb/etc), NOT a pothole class --
    # there is no dedicated pothole class in IDD-Lite's 8-class scheme (see
    # the constants block above). Named honestly for what it measures.
    non_drivable_area_score = float(np.sum(mask == NON_DRIVABLE_CLASS_ID) / mask.size)
    non_drivable_rows = np.where(mask == NON_DRIVABLE_CLASS_ID)[0]
    non_drivable_area_distance = (
        float(mask.shape[0] - np.min(non_drivable_rows)) if non_drivable_rows.size else 0.0
    )

    # Class 2 = living-things (pedestrians/animals), NOT a water/puddle class.
    # This is a real, useful "Dynamic Entities" ODD signal on its own merits.
    living_things_score = float(np.sum(mask == LIVING_THINGS_CLASS_ID) / mask.size)

    brightness = float(np.mean(img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    visibility = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    wetness = float(np.std(img))

    road_pixels = img[mask == ROAD_CLASS_ID]
    road_quality = float(np.var(road_pixels)) if road_pixels.size else 0.0

    pothole_heuristic_score, pothole_heuristic_count = detect_pothole_candidates(gray, mask)

    density_norm = min(traffic_density / 50, 1.0)
    tw_norm = min(num_two_wheelers / 20, 1.0)
    ped_norm = min(num_pedestrians / 20, 1.0)
    distance_risk = 1 / (lead_vehicle_distance + 1)
    confidence_risk = 1 - detection_confidence
    scene_complexity = min(
        0.30 * density_norm
        + 0.20 * tw_norm
        + 0.20 * ped_norm
        + 0.15 * distance_risk
        + 0.15 * confidence_risk,
        1.0,
    )

    return {
        "vehicle_count": vehicle_count,
        "num_two_wheelers": num_two_wheelers,
        "num_pedestrians": num_pedestrians,
        "num_animals": num_animals,
        "num_autorickshaws": num_autorickshaws,
        "object_presence": object_presence,
        "object_distance": object_distance,
        "lead_vehicle_distance": lead_vehicle_distance,
        "traffic_density": traffic_density,
        "detection_confidence": detection_confidence,
        "drivable_area": drivable_area,
        "road_end_distance": road_end_distance,
        "non_drivable_area_score": non_drivable_area_score,
        "non_drivable_area_distance": non_drivable_area_distance,
        "living_things_score": living_things_score,
        "brightness": brightness,
        "visibility": visibility,
        "wetness": wetness,
        "road_quality": road_quality,
        "pothole_heuristic_score": pothole_heuristic_score,
        "pothole_heuristic_count": pothole_heuristic_count,
        "scene_complexity": scene_complexity,
    }


def extract_all_features(
    images: np.ndarray,
    labels: np.ndarray,
    segnet_model: nn.Module,
    yolo_model: YOLO,
    save_path: str = "final_features.csv",
) -> pd.DataFrame:
    """Extracts the 22-feature vector for every image in the dataset.

    Args:
        images: uint8 array of shape (M, H, W, 3) (BGR).
        labels: uint8 ground-truth array of shape (M, H, W). Accepted for
            signature completeness but not used by the feature formulas,
            which rely on the SegNet's own predicted mask -- matching a
            real deployment-time pipeline where no ground truth exists.
        segnet_model: A trained SegNet used to produce per-image predicted
            masks.
        yolo_model: A loaded Ultralytics YOLO model.
        save_path: Where to save the resulting feature CSV.

    Returns:
        A DataFrame with one row per image and 18 feature columns.
    """
    del labels  # Not used: features are computed from the predicted mask.
    segnet_model.eval()
    rows = []

    for i in range(len(images)):
        img = images[i]
        img_t = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        with torch.no_grad():
            output = segnet_model(img_t)
        mask = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        detections = run_detection(img, yolo_model)
        rows.append(compute_features(img, mask, detections))

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Extracted {len(df)} feature rows -> '{save_path}'")
    return df


if __name__ == "__main__":
    from src.common.paths import DATA_DIR, FEATURES_CSV, SEGNET_CHECKPOINT, YOLO_WEIGHTS, ensure_output_dirs
    from src.perception.data_pipeline import load_and_clean_dataset
    from src.perception.segnet_model import load_segnet

    ensure_output_dirs()
    dataset_dir = DATA_DIR
    checkpoint_path = SEGNET_CHECKPOINT
    yolo_weights_path = YOLO_WEIGHTS
    csv_path = FEATURES_CSV

    images, labels = load_and_clean_dataset(dataset_dir)
    segnet = load_segnet(checkpoint_path, device=DEVICE)
    yolo = YOLO(yolo_weights_path)

    extract_all_features(images, labels, segnet, yolo, save_path=csv_path)
