"""Stage 2 (Perception Layer): dynamic multi-stream reliability-weighted fusion.

Implements Sumalatha et al. 2024's ("Autonomous Multi-sensor Fusion
Techniques for Environmental Perception in Self-Driving Vehicles", IC3SE)
dynamic-weighting fusion pattern:

    w_s = f(R_s, C_e)                                        (Eq. 1)

where w_s is the fusion weight for stream s, R_s is a reliability measure
of that stream, and C_e is an environmental context signal. The paper
states this conceptually but never specifies R_s or f's functional form
("derived from training the system using various environmental
scenarios") -- there is no equation to transcribe. This module supplies a
concrete, documented instantiation:

    - Three streams, standing in for the paper's heterogeneous sensors
      (camera/LiDAR/radar), since this project has only a single camera:
      the SegNet semantic mask, the YOLO detections, and raw-image
      statistics.
    - R_s: a per-stream reliability heuristic (see compute_stream_reliabilities).
    - C_e: a scene-brightness/visibility-derived context factor in [0, 1].
    - f: a softmax over (R_s * C_e), a standard, monotonic way to turn
      relative reliability scores into normalized weights -- the paper's
      own stated behavior ("as reliability falls, weight decreases;
      conversely weight increases") holds for any monotonic f, and softmax
      is the simplest such choice.
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

SEGNET_STREAM = "segnet_mask"
YOLO_STREAM = "yolo_detections"
IMAGE_STREAM = "image_stats"
STREAMS = (SEGNET_STREAM, YOLO_STREAM, IMAGE_STREAM)

STREAM_COLUMNS: Dict[str, List[str]] = {
    SEGNET_STREAM: [
        "drivable_area",
        "road_quality",
        "non_drivable_area_score",
        "non_drivable_area_distance",
        "living_things_score",
        "road_end_distance",
    ],
    YOLO_STREAM: [
        "vehicle_count",
        "num_two_wheelers",
        "num_pedestrians",
        "num_animals",
        "num_autorickshaws",
        "object_presence",
        "object_distance",
        "lead_vehicle_distance",
        "traffic_density",
        "detection_confidence",
    ],
    IMAGE_STREAM: [
        "brightness",
        "visibility",
        "wetness",
        "scene_complexity",
        "pothole_heuristic_score",
        "pothole_heuristic_count",
    ],
}

# Softmax temperature: higher values make the weighting more decisive
# (winner-take-more) between streams of differing reliability; a
# documented hyperparameter choice, not a value from the paper.
SOFTMAX_TEMPERATURE = 4.0


@dataclass
class StreamReferences:
    """Calibration reference values used to normalize raw reliability signals.

    Attributes:
        median_drivable_area: Reference drivable_area for SegNet reliability.
        median_visibility: Reference visibility (Laplacian variance) for
            image-stream reliability.
        median_brightness: Reference brightness for the context factor.
    """

    median_drivable_area: float
    median_visibility: float
    median_brightness: float


def calibrate_references(df: pd.DataFrame) -> StreamReferences:
    """Calibrates StreamReferences from an observed features dataframe.

    Args:
        df: A features DataFrame with 'drivable_area', 'visibility', and
            'brightness' columns.

    Returns:
        A StreamReferences instance.
    """
    return StreamReferences(
        median_drivable_area=float(df["drivable_area"].median()),
        median_visibility=float(df["visibility"].median()),
        median_brightness=float(df["brightness"].median()),
    )


def compute_stream_reliabilities(row: pd.Series, refs: StreamReferences) -> Dict[str, float]:
    """Computes each stream's reliability score R_s in [0.0, 1.0] for one scene.

    - SegNet: drivable_area relative to the calibrated median -- a mask
      that finds little to no road is a likely segmentation failure on a
      forward-facing dashcam frame.
    - YOLO: detection_confidence directly (already in [0, 1]; naturally 0
      when there are no detections, which is fine -- there is little for
      that stream to contribute either way).
    - Image stats: visibility (image sharpness) relative to the
      calibrated median -- a blurry frame makes brightness/wetness/
      scene_complexity less trustworthy signals.

    Args:
        row: A features row (as from final_features.csv).
        refs: Calibrated StreamReferences.

    Returns:
        A dict mapping each of STREAMS to its reliability score.
    """
    r_segnet = min(row["drivable_area"] / refs.median_drivable_area, 1.0) if refs.median_drivable_area > 0 else 0.0
    r_yolo = float(np.clip(row["detection_confidence"], 0.0, 1.0))
    r_image = min(row["visibility"] / refs.median_visibility, 1.0) if refs.median_visibility > 0 else 0.0

    return {
        SEGNET_STREAM: float(np.clip(r_segnet, 0.0, 1.0)),
        YOLO_STREAM: float(np.clip(r_yolo, 0.0, 1.0)),
        IMAGE_STREAM: float(np.clip(r_image, 0.0, 1.0)),
    }


def compute_context_factor(row: pd.Series, refs: StreamReferences) -> float:
    """Computes the environmental context signal C_e in [0.0, 1.0] for one scene.

    A composite of relative brightness and relative visibility: poor
    lighting/sharpness conditions (fog, night, motion blur) push C_e
    toward 0, uniformly discounting every stream's weight.

    Args:
        row: A features row.
        refs: Calibrated StreamReferences.

    Returns:
        The context factor in [0.0, 1.0].
    """
    brightness_factor = min(row["brightness"] / refs.median_brightness, 1.0) if refs.median_brightness > 0 else 0.0
    visibility_factor = min(row["visibility"] / refs.median_visibility, 1.0) if refs.median_visibility > 0 else 0.0
    return float(np.clip(0.5 * brightness_factor + 0.5 * visibility_factor, 0.0, 1.0))


def compute_fusion_weights(
    reliabilities: Dict[str, float], context_factor: float, temperature: float = SOFTMAX_TEMPERATURE
) -> Dict[str, float]:
    """Computes fusion weights w_s = f(R_s, C_e) via a softmax over R_s * C_e.

    Args:
        reliabilities: Per-stream reliability scores (compute_stream_reliabilities).
        context_factor: The scene's context factor (compute_context_factor).
        temperature: Softmax temperature; higher sharpens the distribution.

    Returns:
        A dict mapping each stream to a normalized weight (sums to 1.0).
    """
    scores = np.array([reliabilities[s] * context_factor for s in STREAMS])
    exp_scores = np.exp(temperature * (scores - scores.max()))  # max-subtraction for stability
    weights = exp_scores / exp_scores.sum()
    return dict(zip(STREAMS, weights.tolist()))


def compute_fused_confidence(reliabilities: Dict[str, float], weights: Dict[str, float]) -> float:
    """Computes a single scalar summary of overall perception trustworthiness.

    fused_confidence = sum_s(w_s * R_s) -- the reliability-weighted average
    reliability, usable directly by later stages (e.g. Stage 6 monitoring)
    as one summary number.

    Args:
        reliabilities: Per-stream reliability scores.
        weights: Per-stream fusion weights (compute_fusion_weights).

    Returns:
        The fused confidence in [0.0, 1.0].
    """
    return float(sum(weights[s] * reliabilities[s] for s in STREAMS))


def fuse_feature_row(row: pd.Series, weights: Dict[str, float]) -> pd.Series:
    """Reweights a feature row's columns by their stream's fusion weight.

    Each column in STREAM_COLUMNS[stream] is multiplied by weights[stream],
    producing an alternative, fusion-weighted feature representation of
    the same width (usable as a drop-in alternative input to
    odd_classifier's RandomForest).

    Args:
        row: A features row.
        weights: Per-stream fusion weights.

    Returns:
        A new pd.Series with the same feature columns, reweighted.
    """
    fused = row.copy()
    for stream, columns in STREAM_COLUMNS.items():
        for col in columns:
            if col in fused.index:
                fused[col] = row[col] * weights[stream]
    return fused


def fuse_dataframe(df: pd.DataFrame, refs: StreamReferences = None) -> pd.DataFrame:
    """Applies dynamic multi-stream fusion to every row of a features frame.

    Args:
        df: A features DataFrame.
        refs: Pre-calibrated StreamReferences to reuse; if None, calibrated
            fresh from this dataframe.

    Returns:
        A copy of df with added 'reliability_<stream>', 'weight_<stream>',
        and 'fused_confidence' columns.
    """
    if refs is None:
        refs = calibrate_references(df)

    df_out = df.copy()
    for stream in STREAMS:
        df_out[f"reliability_{stream}"] = 0.0
        df_out[f"weight_{stream}"] = 0.0
    df_out["fused_confidence"] = 0.0

    for idx, row in df.iterrows():
        reliabilities = compute_stream_reliabilities(row, refs)
        context_factor = compute_context_factor(row, refs)
        weights = compute_fusion_weights(reliabilities, context_factor)
        fused_confidence = compute_fused_confidence(reliabilities, weights)

        for stream in STREAMS:
            df_out.at[idx, f"reliability_{stream}"] = reliabilities[stream]
            df_out.at[idx, f"weight_{stream}"] = weights[stream]
        df_out.at[idx, "fused_confidence"] = fused_confidence

    return df_out


if __name__ == "__main__":
    from src.common.paths import FEATURES_CSV

    features_path = FEATURES_CSV

    features_df = pd.read_csv(features_path)
    fused_df = fuse_dataframe(features_df)

    print("Mean per-stream reliability and fusion weight:")
    for stream in STREAMS:
        print(
            f"  {stream}: reliability={fused_df[f'reliability_{stream}'].mean():.3f}, "
            f"weight={fused_df[f'weight_{stream}'].mean():.3f}"
        )
    print(f"Mean fused_confidence: {fused_df['fused_confidence'].mean():.3f}")
