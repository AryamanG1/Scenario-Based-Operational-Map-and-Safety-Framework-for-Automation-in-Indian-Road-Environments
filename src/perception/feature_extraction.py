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
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Batched-extraction defaults. 64 frames at 320x224 is ~14 MB of uint8 input
# and well under 1 GB of activations for SegNet + YOLOv8n, so it is safe on
# any GPU; a 40GB+ card can take 256 without trouble. Worker count is capped
# so a small laptop is not oversubscribed; on a many-core server raise it
# toward `nproc` -- JPEG decode is the real limit there, not the GPU.
DEFAULT_FEATURE_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = min(8, os.cpu_count() or 1)

# cv2.Laplacian(ksize=1) kernel. Used with reflect padding, which is OpenCV's
# BORDER_REFLECT_101 default, so the GPU result matches cv2 bit-for-bit.
_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float64
).reshape(1, 1, 3, 3)


def load_yolo(weights: str, device: torch.device = DEVICE) -> YOLO:
    """Loads a YOLO model and pins it to `device`.

    `YOLO(weights)` alone leaves device selection to Ultralytics on every
    predict call. Being explicit makes the placement visible and avoids a
    re-selection per call.

    Args:
        weights: Path to the .pt weights.
        device: Device to run inference on.

    Returns:
        The loaded model.
    """
    model = YOLO(weights)
    model.to(device)
    return model


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
    return _detections_from_result(results, model.names)


def _detections_from_result(result, names: Dict[int, str]) -> List[Dict]:
    """Converts one Ultralytics Results object into detection dicts.

    Pulls the whole boxes tensor across in ONE device->host copy. The previous
    per-box `int(box.cls[0])` / `float(box.conf[0])` / `box.xyxy[0]` pattern
    forced ~5 separate GPU syncs per detection.

    Args:
        result: A single element of the list returned by `model(...)`.
        names: The model's class-id -> label mapping.

    Returns:
        Detection dicts in the same format as `run_detection`.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    # Columns: x1, y1, x2, y2, [track_id], conf, cls. conf/cls are always the
    # last two, so index from the end to tolerate the optional track column.
    data = boxes.data.detach().cpu().numpy()
    xyxy = data[:, :4]
    conf = data[:, -2]
    cls = data[:, -1]
    dets = []
    for i in range(data.shape[0]):
        x1, y1, x2, y2 = xyxy[i]
        dets.append(
            {
                "class": names[int(cls[i])],
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": float(conf[i]),
            }
        )
    return dets


def run_detection_batch(imgs: Sequence[np.ndarray], model: YOLO) -> List[List[Dict]]:
    """Runs YOLOv8 on a batch of same-sized images in one forward pass.

    Numerics: a batched convolution accumulates in a different order than a
    batch-1 one, so confidences differ from `run_detection` at the ~1e-6
    level (measured max 6.5e-6 over 1,964 boxes; zero box/class/count
    differences). This is the same order of drift as CPU-vs-GPU inference.
    A box sitting within ~1e-5 of CONF_THRESHOLD could in principle flip its
    counted/not-counted status; that is expected to affect a handful of rows
    per 100k frames at most.

    Args:
        imgs: BGR uint8 arrays, all of shape (H, W, 3).
        model: A loaded Ultralytics YOLO model.

    Returns:
        One detection list per input image, in input order.
    """
    if len(imgs) == 0:
        return []
    results = model(list(imgs), verbose=False)
    return [_detections_from_result(r, model.names) for r in results]


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


def compute_detection_features(
    detections: List[Dict], image_shape: Tuple[int, int]
) -> Dict[str, float]:
    """Computes the 10 detection-derived features (counts, distances, confidence).

    Pure Python over the detection dicts; shared by the per-frame and the
    batched feature paths so the two cannot drift apart.

    Args:
        detections: YOLO detections for one frame, as from `run_detection()`.
        image_shape: (height, width) of the frame, for the autorickshaw filter.

    Returns:
        A dict of the detection-derived feature values.
    """
    confident_dets = [d for d in detections if d["confidence"] >= CONF_THRESHOLD]

    vehicle_count = sum(1 for d in confident_dets if d["class"] in VEHICLES)
    num_two_wheelers = sum(1 for d in confident_dets if d["class"] in TWO_WHEELERS)
    num_pedestrians = sum(1 for d in confident_dets if d["class"] in PEDESTRIANS)
    num_animals = sum(1 for d in confident_dets if d["class"] in ANIMALS)
    num_autorickshaws = sum(
        1 for d in confident_dets if is_autorickshaw_like(d, image_shape)
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

    return {
        "vehicle_count": vehicle_count,
        "num_two_wheelers": num_two_wheelers,
        "num_pedestrians": num_pedestrians,
        "num_animals": num_animals,
        "num_autorickshaws": num_autorickshaws,
        "object_presence": object_presence,
        "object_distance": object_distance,
        "lead_vehicle_distance": lead_vehicle_distance,
        "traffic_density": len(confident_dets),
        "detection_confidence": detection_confidence,
    }


def _scene_complexity(
    traffic_density: float,
    num_two_wheelers: float,
    num_pedestrians: float,
    lead_vehicle_distance: float,
    detection_confidence: float,
) -> float:
    """Weighted, clipped scene-complexity score (shared by both feature paths)."""
    density_norm = min(traffic_density / 50, 1.0)
    tw_norm = min(num_two_wheelers / 20, 1.0)
    ped_norm = min(num_pedestrians / 20, 1.0)
    distance_risk = 1 / (lead_vehicle_distance + 1)
    confidence_risk = 1 - detection_confidence
    return min(
        0.30 * density_norm
        + 0.20 * tw_norm
        + 0.20 * ped_norm
        + 0.15 * distance_risk
        + 0.15 * confidence_risk,
        1.0,
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
    det = compute_detection_features(detections, img.shape[:2])
    vehicle_count = det["vehicle_count"]
    num_two_wheelers = det["num_two_wheelers"]
    num_pedestrians = det["num_pedestrians"]
    num_animals = det["num_animals"]
    num_autorickshaws = det["num_autorickshaws"]
    object_presence = det["object_presence"]
    object_distance = det["object_distance"]
    detection_confidence = det["detection_confidence"]
    lead_vehicle_distance = det["lead_vehicle_distance"]
    traffic_density = det["traffic_density"]

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

    scene_complexity = _scene_complexity(
        traffic_density, num_two_wheelers, num_pedestrians,
        lead_vehicle_distance, detection_confidence,
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


def predict_mask(img: np.ndarray, segnet_model: nn.Module) -> np.ndarray:
    """Runs SegNet on one image and returns its predicted semantic mask.

    Single-frame convenience wrapper; the batched paths use
    `predict_masks_batch` and keep the result on the GPU.

    Args:
        img: BGR uint8 image of shape (H, W, 3).
        segnet_model: A trained SegNet in eval mode.

    Returns:
        A uint8 mask of shape (H, W) holding per-pixel class IDs.
    """
    img_t = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        output = segnet_model(img_t)
    return output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


def images_to_tensor(imgs: Sequence[np.ndarray], device: torch.device = DEVICE) -> torch.Tensor:
    """Stacks BGR uint8 frames into a (B, 3, H, W) uint8 tensor on `device`.

    Kept as uint8 for the transfer (4x less PCIe traffic than float32); the
    caller converts to float on-device where needed.
    """
    stacked = np.stack(imgs, axis=0)  # (B, H, W, 3)
    return torch.from_numpy(stacked).to(device, non_blocking=True).permute(0, 3, 1, 2)


def predict_masks_batch(imgs_u8: torch.Tensor, segnet_model: nn.Module) -> torch.Tensor:
    """Runs SegNet on a batch already resident on the device.

    Args:
        imgs_u8: (B, 3, H, W) uint8 tensor on the model's device.
        segnet_model: A trained SegNet in eval mode.

    Returns:
        A (B, H, W) int64 tensor of class IDs, still on the device. Callers
        that need numpy should `.cpu()` once per batch, not per frame.
    """
    with torch.no_grad():
        output = segnet_model(imgs_u8.float() / 255.0)
    return output.argmax(dim=1)


def _masked_var(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Population variance of `values` where `mask`, per batch element.

    Two-pass (mean first, then squared deviations) to match numpy's `np.var`
    numerically rather than the cancellation-prone E[x^2] - E[x]^2 form.

    Args:
        values: (B, N) float64.
        mask: (B, N) bool.

    Returns:
        (B,) float64; 0.0 where the mask selects nothing.
    """
    m = mask.to(values.dtype)
    count = m.sum(dim=1)
    safe_count = count.clamp(min=1.0)
    mean = (values * m).sum(dim=1) / safe_count
    var = (((values - mean[:, None]) ** 2) * m).sum(dim=1) / safe_count
    return torch.where(count > 0, var, torch.zeros_like(var))


def _first_row_distance(class_mask: torch.Tensor) -> torch.Tensor:
    """H - (first row containing the class), or 0 if the class is absent.

    Reproduces `mask.shape[0] - np.min(np.where(mask == k)[0])` batched.

    Args:
        class_mask: (B, H, W) bool.

    Returns:
        (B,) float64.
    """
    b, h, _ = class_mask.shape
    row_any = class_mask.any(dim=2)  # (B, H)
    rows = torch.arange(h, device=class_mask.device).expand(b, h)
    first = torch.where(row_any, rows, torch.full_like(rows, h)).min(dim=1).values
    present = row_any.any(dim=1)
    return torch.where(present, (h - first).to(torch.float64), torch.zeros(b, dtype=torch.float64, device=class_mask.device))


def compute_mask_features_batch(
    imgs_u8: torch.Tensor, masks: torch.Tensor, grays_u8: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Computes the image/mask-derived features for a whole batch on the device.

    Every reduction here is the batched torch equivalent of the numpy/cv2 call
    in `compute_features`, in float64 so results agree to rounding:

      drivable_area / non_drivable_area_score / living_things_score
          np.sum(mask == k) / mask.size          -> (masks == k).mean()
      road_end_distance / non_drivable_area_distance
          H - np.min(np.where(mask == k)[0])     -> _first_row_distance
      brightness / wetness
          np.mean(img) / np.std(img)             -> .mean() / .std(correction=0)
      visibility
          cv2.Laplacian(gray, CV_64F).var()      -> conv2d(reflect-pad, kernel).var()
      road_quality
          np.var(img[mask == ROAD])              -> _masked_var

    `grays_u8` is computed by cv2.cvtColor on the host: OpenCV's BGR->gray
    uses a rounding path that a float formula does not reproduce exactly, and
    the conversion is ~20us/frame, so exactness wins.

    Args:
        imgs_u8: (B, 3, H, W) uint8 on device.
        masks: (B, H, W) integer class IDs on device.
        grays_u8: (B, H, W) uint8 on device.

    Returns:
        A dict of (B,) float64 tensors, one per feature.
    """
    b = imgs_u8.shape[0]
    img_d = imgs_u8.to(torch.float64)
    n_pix = masks.shape[1] * masks.shape[2]

    road = masks == ROAD_CLASS_ID
    non_drivable = masks == NON_DRIVABLE_CLASS_ID
    living = masks == LIVING_THINGS_CLASS_ID

    drivable_area = road.sum(dim=(1, 2)).to(torch.float64) / n_pix
    non_drivable_area_score = non_drivable.sum(dim=(1, 2)).to(torch.float64) / n_pix
    living_things_score = living.sum(dim=(1, 2)).to(torch.float64) / n_pix

    road_end_distance = _first_row_distance(road)
    non_drivable_area_distance = _first_row_distance(non_drivable)

    flat = img_d.reshape(b, -1)
    brightness = flat.mean(dim=1)
    wetness = flat.std(dim=1, correction=0)

    lap = F.conv2d(
        F.pad(grays_u8.to(torch.float64).unsqueeze(1), (1, 1, 1, 1), mode="reflect"),
        _LAPLACIAN_KERNEL.to(grays_u8.device),
    )
    visibility = lap.reshape(b, -1).var(dim=1, correction=0)

    # img[mask == ROAD] selects all 3 channels of every road pixel.
    road_3ch = road.unsqueeze(1).expand(-1, 3, -1, -1).reshape(b, -1)
    road_quality = _masked_var(flat, road_3ch)

    return {
        "drivable_area": drivable_area,
        "road_end_distance": road_end_distance,
        "non_drivable_area_score": non_drivable_area_score,
        "non_drivable_area_distance": non_drivable_area_distance,
        "living_things_score": living_things_score,
        "brightness": brightness,
        "visibility": visibility,
        "wetness": wetness,
        "road_quality": road_quality,
    }


def compute_features_batch(
    imgs: Sequence[np.ndarray],
    masks: torch.Tensor,
    detections_per_image: Sequence[List[Dict]],
    imgs_u8: Optional[torch.Tensor] = None,
) -> List[Dict[str, float]]:
    """Batched `compute_features`: same 22 columns, same order, same values.

    GPU work (all mask/image reductions, the Laplacian) runs once for the
    batch. What stays on the host: cv2.cvtColor (see
    `compute_mask_features_batch`), the pothole heuristic (its
    connected-components step has no torch equivalent), and the pure-Python
    detection counts.

    Args:
        imgs: BGR uint8 frames, all (H, W, 3).
        masks: (B, H, W) predicted class IDs on the device.
        detections_per_image: One detection list per frame.
        imgs_u8: Optional (B, 3, H, W) uint8 device tensor of `imgs`, if the
            caller already uploaded it for SegNet. Avoids a second transfer.

    Returns:
        One feature dict per frame, identical in layout to `compute_features`.
    """
    if len(imgs) == 0:
        return []
    device = masks.device
    if imgs_u8 is None:
        imgs_u8 = images_to_tensor(imgs, device)

    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    grays_u8 = torch.from_numpy(np.stack(grays, axis=0)).to(device, non_blocking=True)

    gpu = compute_mask_features_batch(imgs_u8, masks, grays_u8)
    # One device->host copy for all the GPU features, one for the masks.
    gpu_np = {k: v.cpu().numpy() for k, v in gpu.items()}
    masks_np = masks.cpu().numpy().astype(np.uint8)

    rows = []
    for i, (img, gray, dets) in enumerate(zip(imgs, grays, detections_per_image)):
        det = compute_detection_features(dets, img.shape[:2])
        pothole_score, pothole_count = detect_pothole_candidates(gray, masks_np[i])
        rows.append(
            {
                "vehicle_count": det["vehicle_count"],
                "num_two_wheelers": det["num_two_wheelers"],
                "num_pedestrians": det["num_pedestrians"],
                "num_animals": det["num_animals"],
                "num_autorickshaws": det["num_autorickshaws"],
                "object_presence": det["object_presence"],
                "object_distance": det["object_distance"],
                "lead_vehicle_distance": det["lead_vehicle_distance"],
                "traffic_density": det["traffic_density"],
                "detection_confidence": det["detection_confidence"],
                "drivable_area": float(gpu_np["drivable_area"][i]),
                "road_end_distance": float(gpu_np["road_end_distance"][i]),
                "non_drivable_area_score": float(gpu_np["non_drivable_area_score"][i]),
                "non_drivable_area_distance": float(gpu_np["non_drivable_area_distance"][i]),
                "living_things_score": float(gpu_np["living_things_score"][i]),
                "brightness": float(gpu_np["brightness"][i]),
                "visibility": float(gpu_np["visibility"][i]),
                "wetness": float(gpu_np["wetness"][i]),
                "road_quality": float(gpu_np["road_quality"][i]),
                "pothole_heuristic_score": pothole_score,
                "pothole_heuristic_count": pothole_count,
                "scene_complexity": _scene_complexity(
                    det["traffic_density"], det["num_two_wheelers"], det["num_pedestrians"],
                    det["lead_vehicle_distance"], det["detection_confidence"],
                ),
            }
        )
    return rows


def extract_features_for_batch(
    imgs: Sequence[np.ndarray], segnet_model: nn.Module, yolo_model: YOLO
) -> List[Dict[str, float]]:
    """SegNet + YOLO + features for one batch of frames, end to end.

    The single upload of `imgs` serves SegNet and the feature math; YOLO does
    its own letterbox/upload internally. Masks never leave the device until
    the one `.cpu()` inside `compute_features_batch`.

    Args:
        imgs: BGR uint8 frames, all (H, W, 3).
        segnet_model: A trained SegNet in eval mode, on DEVICE.
        yolo_model: A loaded YOLO model.

    Returns:
        One feature dict per frame.
    """
    if len(imgs) == 0:
        return []
    imgs_u8 = images_to_tensor(imgs, DEVICE)
    masks = predict_masks_batch(imgs_u8, segnet_model)
    detections = run_detection_batch(imgs, yolo_model)
    return compute_features_batch(imgs, masks, detections, imgs_u8=imgs_u8)


def extract_all_features(
    images: np.ndarray,
    labels: np.ndarray,
    segnet_model: nn.Module,
    yolo_model: YOLO,
    save_path: str = "final_features.csv",
    batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
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
        batch_size: Frames per SegNet/YOLO forward pass.

    Returns:
        A DataFrame with one row per image and 22 feature columns.
    """
    del labels  # Not used: features are computed from the predicted mask.
    segnet_model.eval()
    rows: List[Dict[str, float]] = []

    for start in range(0, len(images), batch_size):
        batch = [images[i] for i in range(start, min(start + batch_size, len(images)))]
        rows.extend(extract_features_for_batch(batch, segnet_model, yolo_model))

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Extracted {len(df)} feature rows -> '{save_path}'")
    return df


def extract_features_streaming(
    pairs,
    segnet_model: nn.Module,
    yolo_model: YOLO,
    save_path: str = "final_features_idd117k.csv",
    gt_save_path: str = None,
    flush_every: int = 500,
    batch_size: int = DEFAULT_FEATURE_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> pd.DataFrame:
    """Extracts features from IDD117K frames streamed from disk in batches.

    The array-based `extract_all_features` cannot be used on IDD117K: its
    96,897-image train split is ~20.8 GB as a single uint8 array at 320x224x3,
    before labels. This variant decodes frames in `num_workers` parallel
    processes (`idd_detection_loader.iter_frame_batches`), runs SegNet and
    YOLO on `batch_size` frames per forward pass, and flushes rows to CSV
    incrementally, so peak memory is independent of split size and a long run
    that dies partway still leaves usable output on disk.

    Feature columns are byte-identical to `extract_all_features`'s, so every
    downstream stage (traffic density, fuzzy ODD, copula, classifier,
    feasibility map) consumes this CSV unchanged.

    Ground-truth object counts derived from the annotations are written to a
    **separate** sidecar CSV, never into `save_path` (NaN for frames that
    have no annotation file, so "unknown" is never confused with "empty"):
    `odd_classifier.load_and_clean_features` treats every column of the
    feature CSV as a model input, so annotation-derived columns there would
    leak ground truth into the classifier. The sidecar shares the feature
    CSV's row order, so the two join positionally -- its purpose is to let
    the uncalibrated AUTORICKSHAW_* bbox heuristic above finally be scored
    against real 'autorickshaw' labels, which IDD-Lite never had.

    Args:
        pairs: (image_path, annotation_path) pairs from
            `idd_detection_loader.list_pairs` (optionally subsampled).
        segnet_model: A trained SegNet (IDD-Lite-trained; IDD117K has no masks).
        yolo_model: A loaded Ultralytics YOLO model.
        save_path: Where to write the feature CSV.
        gt_save_path: Where to write the ground-truth sidecar CSV. None skips it.
        flush_every: Rows to buffer before appending to disk.
        batch_size: Frames per SegNet/YOLO forward pass.
        num_workers: Parallel JPEG-decode worker processes (0 = serial).

    Returns:
        A DataFrame of the extracted feature rows.
    """
    from src.perception.idd_detection_loader import (
        GROUP_ANIMAL,
        GROUP_AUTORICKSHAW,
        GROUP_PEDESTRIAN,
        GROUP_TWO_WHEELER,
        GROUP_VEHICLE,
        group_of,
        iter_frame_batches,
    )

    segnet_model.eval()

    gt_groups = {
        "gt_num_vehicles": GROUP_VEHICLE,
        "gt_num_two_wheelers": GROUP_TWO_WHEELER,
        "gt_num_autorickshaws": GROUP_AUTORICKSHAW,
        "gt_num_pedestrians": GROUP_PEDESTRIAN,
        "gt_num_animals": GROUP_ANIMAL,
    }

    feature_rows, gt_rows = [], []
    all_rows, all_gt_rows = [], []
    written = 0

    def _flush(final: bool = False) -> None:
        """Appends buffered rows to the CSVs, writing headers only once."""
        nonlocal feature_rows, gt_rows, written
        if not feature_rows and not final:
            return
        if feature_rows:
            pd.DataFrame(feature_rows).to_csv(
                save_path, mode="a" if written else "w", header=not written, index=False
            )
            if gt_save_path:
                pd.DataFrame(gt_rows).to_csv(
                    gt_save_path, mode="a" if written else "w", header=not written, index=False
                )
            written += len(feature_rows)
            print(f"  ... {written} frames processed")
        feature_rows, gt_rows = [], []

    for imgs, boxes_per_image in iter_frame_batches(pairs, batch_size=batch_size, num_workers=num_workers):
        rows = extract_features_for_batch(imgs, segnet_model, yolo_model)
        feature_rows.extend(rows)
        all_rows.extend(rows)

        if gt_save_path:
            for boxes in boxes_per_image:
                if boxes is None:
                    # No annotation file for this frame: the counts are UNKNOWN,
                    # not zero. NaN keeps the row out of the classifier's label
                    # set instead of teaching it that the road was empty.
                    gt_row = {name: float("nan") for name in gt_groups}
                else:
                    groups = [group_of(box["name"]) for box in boxes]
                    gt_row = {name: groups.count(group) for name, group in gt_groups.items()}
                gt_rows.append(gt_row)
                all_gt_rows.append(gt_row)

        if len(feature_rows) >= flush_every:
            _flush()

    _flush(final=True)

    print(f"Extracted {written} feature rows -> '{save_path}'")
    if gt_save_path and written:
        print(f"Wrote ground-truth object counts -> '{gt_save_path}'")
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    import argparse

    from src.common.paths import (
        DATA_DIR,
        FEATURES_CSV,
        FEATURES_IDD117K_CSV,
        FEATURES_IDD117K_GT_CSV,
        IDD117K_95K_DIR,
        SEGNET_CHECKPOINT,
        YOLO_WEIGHTS,
        ensure_output_dirs,
    )
    from src.perception.segnet_model import load_segnet

    parser = argparse.ArgumentParser(description="Extract the per-frame ODD feature vector.")
    parser.add_argument(
        "--dataset",
        choices=("idd-lite", "idd117k"),
        default="idd-lite",
        help="Image corpus. SegNet is IDD-Lite-trained either way (IDD117K has no masks).",
    )
    parser.add_argument("--split", default="train", help="IDD117K split to extract from.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_FEATURE_BATCH_SIZE, help="Frames per SegNet/YOLO forward pass.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help="Parallel JPEG-decode workers (0 = serial).")
    parser.add_argument(
        "--limit",
        type=int,
        default=15000,
        help="Frames to extract for idd117k (0 = all). The downstream copula/classifier "
        "stages fit distributions, so a representative sample is sufficient.",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    segnet = load_segnet(SEGNET_CHECKPOINT, device=DEVICE)
    yolo = load_yolo(YOLO_WEIGHTS)

    if args.dataset == "idd-lite":
        from src.perception.data_pipeline import load_and_clean_dataset

        images, labels = load_and_clean_dataset(DATA_DIR)
        extract_all_features(images, labels, segnet, yolo, save_path=FEATURES_CSV, batch_size=args.batch_size)
    else:
        from src.perception.idd_detection_loader import list_pairs, sample_pairs

        pairs = sample_pairs(list_pairs(IDD117K_95K_DIR, args.split), args.limit or None)
        print(f"Extracting features from {len(pairs)} IDD117K '{args.split}' frame(s).")
        extract_features_streaming(
            pairs,
            segnet,
            yolo,
            save_path=FEATURES_IDD117K_CSV,
            gt_save_path=FEATURES_IDD117K_GT_CSV,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
