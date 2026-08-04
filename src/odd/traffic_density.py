"""Stage 3: Traffic Density Estimation.

Per the capstone proposal's Stage 3 ("Count detected objects per frame;
compute vehicle-to-area ratio; classify density into Low/Medium/High/
Congested"): computes a vehicle-to-frame-area ratio from the Stage 2
detection counts and classifies it into four density levels, with cut
points calibrated from the actual observed distribution (quartiles) rather
than arbitrary constants.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

IMAGE_AREA_PX = 320 * 224  # matches data_pipeline.IMAGE_SIZE (width * height)

LOW = "Low"
MEDIUM = "Medium"
HIGH = "High"
CONGESTED = "Congested"
DENSITY_LEVELS = (LOW, MEDIUM, HIGH, CONGESTED)


@dataclass
class TrafficDensityThresholds:
    """Quartile-calibrated vehicle-to-area-ratio cut points.

    Attributes:
        low_medium: Ratio at or below which a scene is 'Low' density.
        medium_high: Ratio at or below which a scene is 'Medium' density.
        high_congested: Ratio at or below which a scene is 'High' density;
            above this it is 'Congested'.
    """

    low_medium: float
    medium_high: float
    high_congested: float

    def save(self, path: str) -> None:
        """Serializes the thresholds to a JSON file.

        Args:
            path: Destination file path.
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrafficDensityThresholds":
        """Loads previously calibrated thresholds from a JSON file.

        Args:
            path: Source file path, as written by save().

        Returns:
            A TrafficDensityThresholds instance.
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def compute_vehicle_area_ratio(
    vehicle_count: float,
    num_two_wheelers: float,
    num_autorickshaws: float,
    image_area: float = IMAGE_AREA_PX,
) -> float:
    """Computes a vehicle-to-frame-area density ratio for one scene.

    Args:
        vehicle_count: Count of car/bus/truck detections.
        num_two_wheelers: Count of bicycle/motorcycle detections.
        num_autorickshaws: Count of auto-rickshaw-like detections.
        image_area: Frame area in pixels.

    Returns:
        Vehicles per 10,000 px^2 -- scaled for a human-readable magnitude
        rather than the raw (very small) vehicles-per-pixel ratio.
    """
    total_vehicles = vehicle_count + num_two_wheelers + num_autorickshaws
    return (total_vehicles / image_area) * 1e4


def calibrate_thresholds(ratios: List[float]) -> TrafficDensityThresholds:
    """Calibrates Low/Medium/High/Congested cut points from observed data.

    Uses the 25th/50th/75th percentiles of the observed ratio distribution,
    so each of the four bins captures roughly a quarter of the dataset by
    construction, rather than relying on hand-picked constants.

    Args:
        ratios: Observed vehicle_area_ratio values across a dataset.

    Returns:
        A TrafficDensityThresholds instance.
    """
    q1, q2, q3 = np.percentile(ratios, [25, 50, 75])
    return TrafficDensityThresholds(
        low_medium=float(q1), medium_high=float(q2), high_congested=float(q3)
    )


def classify_density(ratio: float, thresholds: TrafficDensityThresholds) -> str:
    """Classifies a single vehicle-area ratio into a density level.

    Args:
        ratio: A vehicle_area_ratio value, as from compute_vehicle_area_ratio().
        thresholds: Calibrated cut points.

    Returns:
        One of "Low", "Medium", "High", "Congested".
    """
    if ratio <= thresholds.low_medium:
        return LOW
    if ratio <= thresholds.medium_high:
        return MEDIUM
    if ratio <= thresholds.high_congested:
        return HIGH
    return CONGESTED


def classify_dataframe(
    df: pd.DataFrame, thresholds: Optional[TrafficDensityThresholds] = None
) -> Tuple[pd.DataFrame, TrafficDensityThresholds]:
    """Adds vehicle_area_ratio and traffic_density_level columns to a features frame.

    Args:
        df: A features DataFrame with 'vehicle_count', 'num_two_wheelers',
            and 'num_autorickshaws' columns (as produced by
            feature_extraction.extract_all_features).
        thresholds: Pre-calibrated thresholds to reuse (e.g. from a prior
            training run); if None, thresholds are calibrated fresh from
            this dataframe's own ratio distribution.

    Returns:
        A tuple (df_with_density_columns, thresholds_used).
    """
    ratios = df.apply(
        lambda row: compute_vehicle_area_ratio(
            row["vehicle_count"], row["num_two_wheelers"], row["num_autorickshaws"]
        ),
        axis=1,
    )

    if thresholds is None:
        thresholds = calibrate_thresholds(ratios.tolist())

    levels = ratios.apply(lambda r: classify_density(r, thresholds))

    df_out = df.copy()
    df_out["vehicle_area_ratio"] = ratios
    df_out["traffic_density_level"] = levels
    return df_out, thresholds


if __name__ == "__main__":
    from src.common.paths import FEATURES_CSV, TRAFFIC_DENSITY_THRESHOLDS_JSON, ensure_output_dirs

    ensure_output_dirs()
    features_path = FEATURES_CSV
    thresholds_path = TRAFFIC_DENSITY_THRESHOLDS_JSON

    features_df = pd.read_csv(features_path)
    labeled_df, calibrated_thresholds = classify_dataframe(features_df)
    calibrated_thresholds.save(thresholds_path)

    print(f"Calibrated thresholds: {calibrated_thresholds}")
    print("Traffic density distribution:")
    print(labeled_df["traffic_density_level"].value_counts().reindex(DENSITY_LEVELS))
