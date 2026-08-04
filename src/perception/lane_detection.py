"""Stage 2 (Perception Layer): classical computer-vision lane detection.

IDD-Lite has no lane-marking annotations, so a supervised lane detector
can't be trained here. Instead this module estimates the drivable
corridor boundaries with a classical Canny + probabilistic Hough-transform
pipeline restricted to the SegNet-predicted road mask, per the capstone
proposal's Stage 2 requirement ("Lane detection via CV/deep learning").
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROAD_CLASS_ID = 0

# Canny edge thresholds and Hough transform parameters, tuned for the
# 320x224 IDD-Lite frame size.
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 20
HOUGH_MIN_LINE_LENGTH = 15
HOUGH_MAX_LINE_GAP = 10

# A line is discarded if its slope magnitude is below this value -- lane
# boundaries are never near-horizontal in a forward-facing dashcam view,
# so this rejects road-surface texture/shadow edges.
MIN_ABS_SLOPE = 0.3

Line = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@dataclass
class LaneDetectionResult:
    """Result of one frame's lane-boundary estimation.

    Attributes:
        left_lane: (slope, intercept) of the fitted left boundary, or None
            if no left-side line survived filtering.
        right_lane: (slope, intercept) of the fitted right boundary, or
            None if no right-side line survived filtering.
        left_lines: Raw Hough segments classified as left-boundary candidates.
        right_lines: Raw Hough segments classified as right-boundary candidates.
        image_shape: (height, width) of the source frame.
    """

    left_lane: Optional[Tuple[float, float]]
    right_lane: Optional[Tuple[float, float]]
    left_lines: List[Line] = field(default_factory=list)
    right_lines: List[Line] = field(default_factory=list)
    image_shape: Tuple[int, int] = (0, 0)

    @property
    def num_lines_detected(self) -> int:
        """Total number of Hough line segments kept after slope filtering."""
        return len(self.left_lines) + len(self.right_lines)

    @property
    def lane_confidence(self) -> float:
        """Heuristic confidence in [0, 1]: 1.0 if both boundaries fitted.

        0.5 if only one side fitted, 0.0 if neither. Downstream ODD stages
        treat a fully-fitted lane pair as evidence of a well-structured
        (higher-automation-readiness) road segment.
        """
        sides_fitted = int(self.left_lane is not None) + int(self.right_lane is not None)
        return sides_fitted / 2.0

    @property
    def lane_width_px(self) -> Optional[float]:
        """Estimated lane width in pixels at the bottom of the frame.

        Returns:
            The horizontal gap between the fitted left/right boundaries at
            the image's bottom row, or None if either boundary is missing.
        """
        if self.left_lane is None or self.right_lane is None:
            return None
        height, _ = self.image_shape
        y = float(height - 1)
        left_slope, left_intercept = self.left_lane
        right_slope, right_intercept = self.right_lane
        if left_slope == 0 or right_slope == 0:
            return None
        left_x = (y - left_intercept) / left_slope
        right_x = (y - right_intercept) / right_slope
        return abs(right_x - left_x)


def detect_road_edges(image: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """Extracts Canny edges restricted to the drivable road region.

    Args:
        image: BGR image of shape (H, W, 3).
        road_mask: Semantic segmentation mask of shape (H, W), where
            ROAD_CLASS_ID marks drivable-road pixels.

    Returns:
        A single-channel binary edge map of shape (H, W), zeroed outside
        the road region.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)

    road_binary = (road_mask == ROAD_CLASS_ID).astype(np.uint8)
    # Dilate slightly so edges right at the road/non-road boundary (where
    # lane markings often sit) aren't clipped off by a one-pixel-tight mask.
    road_binary = cv2.dilate(road_binary, np.ones((5, 5), np.uint8), iterations=1)

    return cv2.bitwise_and(edges, edges, mask=road_binary)


def hough_lane_lines(edges: np.ndarray) -> List[Line]:
    """Runs a probabilistic Hough transform to find candidate line segments.

    Args:
        edges: Binary edge map of shape (H, W).

    Returns:
        A list of (x1, y1, x2, y2) line segments. Empty if none found.
    """
    lines = cv2.HoughLinesP(
        edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    if lines is None:
        return []
    # cv2.HoughLinesP's output is shaped (N, 1, 4) in some OpenCV versions
    # and (N, 4) in others; flatten each entry so both are handled.
    return [tuple(np.asarray(line).flatten()) for line in lines]


def classify_lane_lines(lines: List[Line], image_width: int) -> Tuple[List[Line], List[Line]]:
    """Splits Hough line segments into left- and right-boundary candidates.

    A line is discarded if it is nearly horizontal (slope magnitude below
    MIN_ABS_SLOPE). Surviving lines are bucketed by slope sign: negative
    slope (in image coordinates, y grows downward) and a midpoint left of
    center indicates a left boundary; positive slope and a midpoint right
    of center indicates a right boundary.

    Args:
        lines: Raw Hough line segments.
        image_width: Frame width in pixels, used to test line position
            relative to the horizontal center.

    Returns:
        A tuple (left_lines, right_lines).
    """
    left_lines: List[Line] = []
    right_lines: List[Line] = []
    center_x = image_width / 2.0

    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue  # Vertical line: undefined slope, skip.
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < MIN_ABS_SLOPE:
            continue

        midpoint_x = (x1 + x2) / 2.0
        if slope < 0 and midpoint_x < center_x:
            left_lines.append((x1, y1, x2, y2))
        elif slope > 0 and midpoint_x >= center_x:
            right_lines.append((x1, y1, x2, y2))

    return left_lines, right_lines


def fit_lane_boundary(lines: List[Line]) -> Optional[Tuple[float, float]]:
    """Fits a single representative line through a set of segments.

    Args:
        lines: Candidate line segments for one side (left or right).

    Returns:
        (slope, intercept) of a least-squares fit `y = slope*x + intercept`
        through all segment endpoints, or None if lines is empty.
    """
    if not lines:
        return None

    xs: List[int] = []
    ys: List[int] = []
    for x1, y1, x2, y2 in lines:
        xs.extend([x1, x2])
        ys.extend([y1, y2])

    slope, intercept = np.polyfit(xs, ys, deg=1)
    return float(slope), float(intercept)


def detect_lanes(image: np.ndarray, road_mask: np.ndarray) -> LaneDetectionResult:
    """Runs the full lane-boundary detection pipeline on one frame.

    Args:
        image: BGR image of shape (H, W, 3).
        road_mask: Semantic segmentation mask of shape (H, W).

    Returns:
        A LaneDetectionResult describing the fitted left/right boundaries.
    """
    height, width = image.shape[:2]
    edges = detect_road_edges(image, road_mask)
    raw_lines = hough_lane_lines(edges)
    left_lines, right_lines = classify_lane_lines(raw_lines, width)

    return LaneDetectionResult(
        left_lane=fit_lane_boundary(left_lines),
        right_lane=fit_lane_boundary(right_lines),
        left_lines=left_lines,
        right_lines=right_lines,
        image_shape=(height, width),
    )


def visualize_lanes(image: np.ndarray, result: LaneDetectionResult) -> np.ndarray:
    """Draws the fitted lane boundaries onto a copy of the input image.

    Args:
        image: BGR image of shape (H, W, 3).
        result: Output of detect_lanes() for this image.

    Returns:
        A BGR image copy with the left boundary drawn in blue and the
        right boundary drawn in red, extrapolated across the frame height.
    """
    overlay = image.copy()
    height = result.image_shape[0]
    y1, y2 = height, int(height * 0.6)

    for lane, color in ((result.left_lane, (255, 0, 0)), (result.right_lane, (0, 0, 255))):
        if lane is None:
            continue
        slope, intercept = lane
        if slope == 0:
            continue
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        cv2.line(overlay, (x1, y1), (x2, y2), color, 3)

    return overlay


def compute_lane_features(result: LaneDetectionResult) -> dict:
    """Summarizes a LaneDetectionResult into scalar features for the ODD pipeline.

    Args:
        result: Output of detect_lanes().

    Returns:
        A dict with 'lane_confidence' (float, 0-1), 'num_lines_detected'
        (int), and 'lane_width_px' (float, or 0.0 if not estimable).
    """
    lane_width = result.lane_width_px
    return {
        "lane_confidence": result.lane_confidence,
        "num_lines_detected": result.num_lines_detected,
        "lane_width_px": lane_width if lane_width is not None else 0.0,
    }


if __name__ == "__main__":
    import os

    import torch

    from src.common.paths import DATA_DIR, PLOTS_DIR, SEGNET_CHECKPOINT
    from src.perception.data_pipeline import load_and_clean_dataset
    from src.perception.segnet_model import load_segnet

    images, labels = load_and_clean_dataset(DATA_DIR)
    segnet = load_segnet(SEGNET_CHECKPOINT)

    sample_img = images[0]
    img_t = torch.tensor(sample_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        pred_mask = segnet(img_t).argmax(dim=1).squeeze(0).numpy().astype(np.uint8)

    lane_result = detect_lanes(sample_img, pred_mask)
    print(f"Lane confidence: {lane_result.lane_confidence}")
    print(f"Lines detected: {lane_result.num_lines_detected}")
    print(f"Lane width (px): {lane_result.lane_width_px}")
    print(f"Features: {compute_lane_features(lane_result)}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    vis = visualize_lanes(sample_img, lane_result)
    cv2.imwrite(os.path.join(PLOTS_DIR, "lane_detection_sample.png"), vis)
