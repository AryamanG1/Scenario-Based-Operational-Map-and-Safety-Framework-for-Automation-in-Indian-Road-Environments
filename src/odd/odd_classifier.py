"""Module 5 / Stage 7 (Decision System): ODD classification engine.

Provides rule-based ground-truth labeling (Normal/Degraded/Takeover per SAE
J3016), a feature-cleaning pipeline, a RandomForest ODD classifier, an
end-to-end single-image inference/visualization function, and
combine_stage_outputs() -- the Stage 7 "Combine monitoring output +
operational condition status" step that merges this module's own
feature-based rule with Stage 5's ODD-region status (odd_boundary.py) and
Stage 6's real-time monitoring status (perception_monitor.py) into the
final decision.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from ultralytics import YOLO

from src.common.paths import PLOTS_DIR
from src.perception.data_pipeline import IMAGE_SIZE
from src.perception.feature_extraction import PEDESTRIANS, TWO_WHEELERS, VEHICLES, compute_features, run_detection

RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("Warning: No GPU detected. odd_classifier will run on CPU.")


# --------------------------------------------------------------------------
# Part A: Rule-based ground-truth labeling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeThresholds:
    """The cut-offs assign_mode() compares MinMax-scaled features against.

    The legacy values (DEFAULT_MODE_THRESHOLDS) were hand-picked on 1,402
    IDD-Lite frames. Applied to a 104k-frame corpus whose MinMax range is
    set by a handful of extreme frames, nothing clears the "Normal" bar
    (0 Normal / 92% Takeover). calibrate_mode_thresholds() re-derives each
    cut-off as a percentile of the corpus it is actually applied to, so the
    rule's *structure* (what must be good for Normal, what is tolerable for
    Degraded) is unchanged but its numbers match the data.
    """

    det_conf_normal: float = 0.85     # detection_confidence must exceed this for Normal
    det_conf_degraded: float = 0.5    # ... and this for Degraded
    visibility_normal: float = 0.45   # visibility (sharpness) must exceed this for Normal
    visibility_degraded: float = 0.25 # ... and this for Degraded
    traffic_density_normal: float = 0.4     # traffic_density must be below this for Normal
    non_drivable_normal: float = 0.2        # non_drivable_area_score must be below this for Normal
    living_things_normal: float = 0.2       # living_things_score must be below this for Normal
    source: str = "legacy-constants"

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "ModeThresholds":
        with open(path) as handle:
            return cls(**json.load(handle))


ModeThresholds.__module__ = "src.odd.odd_classifier"
DEFAULT_MODE_THRESHOLDS = ModeThresholds()

# Percentile each cut-off is derived from in calibrate_mode_thresholds().
# These encode the same intent as the legacy constants ("Normal needs the
# good side of every axis; Degraded tolerates the middle") as ranks in the
# corpus rather than fixed positions on a data-dependent MinMax scale:
#   visibility: Normal must be sharper than the bottom 30%; Degraded than the bottom 10%.
#   detection_confidence (frames with detections): Normal above the bottom 40%; Degraded above the bottom 10%.
#   traffic_density: Normal below the 60th percentile.
#   non_drivable_area_score / living_things_score: Normal below the top 20%.
# Normal is a CONJUNCTION of five conditions, so its share is roughly the
# product of these pass rates (~10-15% of frames), not any single percentile.
# These are the documented operationalization knobs; on IDD-Lite they give
# ~11% Normal / 71% Degraded / 18% Takeover versus 1% / 31% / 68% for the
# legacy constants.
MODE_CALIBRATION_PERCENTILES = {
    "det_conf_normal": 40.0,
    "det_conf_degraded": 10.0,
    "visibility_normal": 30.0,
    "visibility_degraded": 10.0,
    "traffic_density_normal": 60.0,
    "non_drivable_normal": 80.0,
    "living_things_normal": 80.0,
}


def calibrate_mode_thresholds(
    df_scaled: pd.DataFrame, percentiles: Optional[dict] = None
) -> ModeThresholds:
    """Derives assign_mode() cut-offs as percentiles of a scaled feature table.

    Args:
        df_scaled: MinMax-scaled features for the corpus the rule will be
            applied to (FeatureScaler.transform() output).
        percentiles: Override of MODE_CALIBRATION_PERCENTILES.

    Returns:
        A ModeThresholds with source="percentiles".
    """
    pct = {**MODE_CALIBRATION_PERCENTILES, **(percentiles or {})}
    with_dets = df_scaled["detection_confidence"][df_scaled.get("object_presence", 1) > 0]
    if with_dets.empty:
        with_dets = df_scaled["detection_confidence"]
    return ModeThresholds(
        det_conf_normal=float(np.percentile(with_dets, pct["det_conf_normal"])),
        det_conf_degraded=float(np.percentile(with_dets, pct["det_conf_degraded"])),
        visibility_normal=float(np.percentile(df_scaled["visibility"], pct["visibility_normal"])),
        visibility_degraded=float(np.percentile(df_scaled["visibility"], pct["visibility_degraded"])),
        traffic_density_normal=float(np.percentile(df_scaled["traffic_density"], pct["traffic_density_normal"])),
        non_drivable_normal=float(np.percentile(df_scaled["non_drivable_area_score"], pct["non_drivable_normal"])),
        living_things_normal=float(np.percentile(df_scaled["living_things_score"], pct["living_things_normal"])),
        source=f"percentiles of {len(df_scaled)} scaled rows: {pct}",
    )


def load_mode_thresholds(path: str) -> ModeThresholds:
    """Loads persisted thresholds, or the legacy defaults (with a note) if absent."""
    if os.path.isfile(path):
        return ModeThresholds.load(path)
    print(f"Note: no calibrated mode thresholds at '{path}'; using legacy constants.")
    return DEFAULT_MODE_THRESHOLDS


def assign_mode(row: pd.Series, thresholds: ModeThresholds = DEFAULT_MODE_THRESHOLDS) -> str:
    """Assigns an ODD mode label to one MinMax-scaled feature row.

    SAE J3016 mapping (this capstone targets SAE Level 2 -- see
    sae_taxonomy.py): Normal -> ADS handles lateral+longitudinal control
    within its ODD while the driver supervises, Degraded -> cautious
    operation requiring closer driver attention, Takeover -> driver must
    resume full control (ODD exit / Request to Intervene).

    Critical bug fix: when there are no detected objects (object_presence ==
    0), detection_confidence is trivially 0.0. That alone must not force a
    Takeover -- an empty road under otherwise-clear conditions is safe. The
    confidence checks below are treated as satisfied whenever there are no
    detections.

    Args:
        row: A row of MinMax-scaled (0-1) features, including at least
            'object_presence', 'detection_confidence', 'visibility',
            'traffic_density', 'non_drivable_area_score', 'living_things_score'.
        thresholds: The cut-offs to compare against. Defaults to the legacy
            constants; main.py passes corpus-calibrated ones (see
            calibrate_mode_thresholds).

    Returns:
        One of "Normal", "Degraded", "Takeover".
    """
    t = thresholds
    has_detections = row.get("object_presence", 0) > 0
    det_conf_ok = (row["detection_confidence"] > t.det_conf_normal) if has_detections else True

    # NORMAL: full ADS operation within ODD (SAE L2, driver supervising)
    if (
        det_conf_ok
        and row["visibility"] > t.visibility_normal
        and row["traffic_density"] < t.traffic_density_normal
        and row["non_drivable_area_score"] < t.non_drivable_normal
        and row["living_things_score"] < t.living_things_normal
    ):
        return "Normal"

    # DEGRADED: cautious/supervised operation, closer driver attention needed
    elif (
        ((row["detection_confidence"] > t.det_conf_degraded) if has_detections else True)
        and row["visibility"] > t.visibility_degraded
    ):
        return "Degraded"

    # TAKEOVER: Driver must intervene (SAE RTI)
    else:
        return "Takeover"


_MODE_SEVERITY = {"Normal": 0, "Degraded": 1, "Takeover": 2}
_ODD_REGION_SEVERITY = {"within": 0, "near": 1, "outside": 2}
_MONITORING_SEVERITY = {"Nominal": 0, "Warning": 1, "Critical": 2}


def combine_stage_outputs(base_mode: str, odd_region: str, monitoring_state: str) -> str:
    """Combines Stage 5/6 status with the Stage 7 rule-based mode (final decision).

    Per the capstone proposal's Stage 7 ("Combine monitoring output +
    operational condition status... select system mode"): the final ODD
    mode is the MOST SEVERE of three independent signals --
    assign_mode()'s feature-based rule (Stage 7's own base classifier),
    odd_boundary.py's within/near/outside ODD-region status (Stage 5), and
    perception_monitor.py's Nominal/Warning/Critical status (Stage 6).
    This "most-severe-signal-wins" combination is a standard redundant-
    channel safety pattern: any one upstream signal can escalate the
    system toward a more conservative mode, but none can override another
    signal's legitimate escalation.

    Args:
        base_mode: Output of assign_mode(), one of "Normal"/"Degraded"/"Takeover".
        odd_region: Output of odd_boundary.classify_odd_region(), one of
            "within"/"near"/"outside".
        monitoring_state: Output of perception_monitor.run_perception_monitor()
            .state, one of "Nominal"/"Warning"/"Critical".

    Returns:
        The final combined mode, one of "Normal", "Degraded", "Takeover".
    """
    severities = [
        _MODE_SEVERITY[base_mode],
        _ODD_REGION_SEVERITY[odd_region],
        _MONITORING_SEVERITY[monitoring_state],
    ]
    final_severity = max(severities)
    return next(mode for mode, sev in _MODE_SEVERITY.items() if sev == final_severity)


# --------------------------------------------------------------------------
# Part B: Feature cleaning pipeline
# --------------------------------------------------------------------------


class FeatureScaler:
    """Bundles the brightness/visibility/road_quality normalization
    constants with a fitted MinMaxScaler, so this single artifact can
    reproduce the exact Part B transform on a single new image's features
    at inference time (a lone image has no batch to compute a max over).
    """

    def __init__(
        self,
        visibility_max: float,
        road_quality_max: float,
        minmax_scaler: MinMaxScaler,
        feature_columns: List[str],
    ) -> None:
        self.visibility_max = visibility_max
        self.road_quality_max = road_quality_max
        self.minmax_scaler = minmax_scaler
        self.feature_columns = feature_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the fitted normalize + MinMax pipeline to new feature rows.

        Args:
            df: A DataFrame containing at least self.feature_columns.

        Returns:
            A scaled DataFrame restricted to self.feature_columns.
        """
        df = df.copy()
        df["brightness"] = df["brightness"] / 255.0
        df["visibility"] = df["visibility"] / self.visibility_max
        df["road_quality"] = df["road_quality"] / self.road_quality_max
        scaled = self.minmax_scaler.transform(df[self.feature_columns])
        return pd.DataFrame(scaled, columns=self.feature_columns, index=df.index)


# Pickling gotcha: if this file is ever run directly (`python -m
# src.odd.odd_classifier`), Python sets this module's runtime __name__ to
# "__main__", so a class defined here would otherwise get pickled as
# "__main__.FeatureScaler" -- unloadable from any other script's process,
# including main.py and full_dataset_pipeline.py. Forcing the real,
# importable module name here makes every FeatureScaler pickle loadable
# regardless of how this file was invoked.
FeatureScaler.__module__ = "src.odd.odd_classifier"


# --------------------------------------------------------------------------
# Ground-truth (annotation-derived) labels
# --------------------------------------------------------------------------
#
# WHY THIS EXISTS. The original labelling ran assign_mode() -- a hand-written
# rule -- on the SAME scaled perception features the RandomForest is trained
# on. The label was therefore a deterministic function of the model's inputs,
# and any classifier scores ~100% on it (train-split and held-out alike),
# which measures nothing about generalization. That path is kept below as
# LABEL_SOURCE_RULE for backwards compatibility, but it prints a warning.
#
# LABEL_SOURCE_GT instead derives the label from the dataset's human box
# annotations (the *_gt.csv sidecars extract_features_streaming writes):
# scene complexity S = number of annotated road users in the frame. The
# classifier then has to predict an annotation-derived quantity from noisy
# perception outputs (YOLO recall on IDD117K is ~18%), which is a genuine
# learning problem and yields an honest accuracy. Rows whose source dataset
# has no box annotations (IDD-Lite, IDD-20K-II) are excluded from classifier
# training/evaluation; every other stage still uses them.

LABEL_SOURCE_GT = "gt"
LABEL_SOURCE_RULE = "rule"

GT_COUNT_COLUMNS = [
    "gt_num_vehicles",
    "gt_num_two_wheelers",
    "gt_num_autorickshaws",
    "gt_num_pedestrians",
    "gt_num_animals",
]


@dataclass
class GTLabelThresholds:
    """Cut points on annotation-derived scene complexity S.

    S <= low  -> "Normal"; low < S <= high -> "Degraded"; S > high -> "Takeover".
    Fitted once on the training pool (fit_gt_label_thresholds) and persisted
    so the held-out evaluation uses exactly the same label definition.
    """

    low: float
    high: float
    low_percentile: float
    high_percentile: float

    def save(self, path: str) -> None:
        with open(path, "w") as handle:
            json.dump(asdict(self), handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "GTLabelThresholds":
        with open(path) as handle:
            return cls(**json.load(handle))


GTLabelThresholds.__module__ = "src.odd.odd_classifier"


def gt_scene_complexity(gt_df: pd.DataFrame) -> pd.Series:
    """Annotation-derived scene complexity: total annotated road users per frame.

    Args:
        gt_df: A ground-truth sidecar DataFrame with GT_COUNT_COLUMNS.

    Returns:
        A float Series (NaN where any count column is NaN, i.e. no annotations).
    """
    return gt_df[GT_COUNT_COLUMNS].sum(axis=1, skipna=False).astype(float)


def fit_gt_label_thresholds(
    gt_df: pd.DataFrame, low_percentile: float = 100.0 / 3, high_percentile: float = 200.0 / 3
) -> GTLabelThresholds:
    """Fits tertile cut points of scene complexity on the training pool.

    Tertiles are used so the three classes are roughly balanced by
    construction (an integer-valued S with many ties makes them only
    approximately equal). The percentiles are stored alongside the values
    so the definition is reproducible.

    Args:
        gt_df: Ground-truth sidecar rows of the TRAINING pool only.
        low_percentile: Percentile of S at/below which a frame is "Normal".
        high_percentile: Percentile of S above which a frame is "Takeover".

    Returns:
        A GTLabelThresholds.
    """
    complexity = gt_scene_complexity(gt_df).dropna()
    if complexity.empty:
        raise ValueError("No annotated rows to fit ground-truth label thresholds on.")
    low = float(np.percentile(complexity, low_percentile))
    high = float(np.percentile(complexity, high_percentile))
    return GTLabelThresholds(low=low, high=high, low_percentile=low_percentile, high_percentile=high_percentile)


def assign_gt_mode(gt_df: pd.DataFrame, thresholds: GTLabelThresholds) -> pd.Series:
    """Maps annotation-derived scene complexity to Normal/Degraded/Takeover.

    Args:
        gt_df: Ground-truth sidecar DataFrame with GT_COUNT_COLUMNS.
        thresholds: Cut points from fit_gt_label_thresholds().

    Returns:
        A Series of mode strings, NaN where the row has no annotations.
    """
    complexity = gt_scene_complexity(gt_df)
    modes = pd.Series(
        np.where(
            complexity <= thresholds.low, "Normal",
            np.where(complexity <= thresholds.high, "Degraded", "Takeover"),
        ),
        index=gt_df.index,
        dtype=object,
    )
    modes[complexity.isna()] = np.nan
    return modes


def _majority_baseline(y: np.ndarray) -> float:
    """Accuracy of always predicting the most frequent class -- the floor any
    reported accuracy should be compared against."""
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def _read_gt_sidecar(gt_csv_path: str, expected_rows: int) -> pd.DataFrame:
    gt_df = pd.read_csv(gt_csv_path)
    if len(gt_df) != expected_rows:
        raise ValueError(
            f"Ground-truth sidecar '{gt_csv_path}' has {len(gt_df)} rows but the feature "
            f"table has {expected_rows}; they must be aligned row-for-row."
        )
    missing = [c for c in GT_COUNT_COLUMNS if c not in gt_df.columns]
    if missing:
        raise ValueError(f"Ground-truth sidecar '{gt_csv_path}' lacks columns {missing}.")
    return gt_df


def load_and_clean_features(
    csv_path: str,
    gt_csv_path: Optional[str] = None,
    gt_thresholds: Optional[GTLabelThresholds] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, FeatureScaler, LabelEncoder]:
    """Loads, normalizes, labels, and prunes the raw feature CSV.

    Label source:
      * `gt_csv_path` given  -> LABEL_SOURCE_GT. Labels come from the
        annotation sidecar via assign_gt_mode(); rows without annotations
        are dropped from X / y (the scaler is still fitted on every row, so
        it stays valid for the full corpus in later stages).
      * `gt_csv_path` None   -> LABEL_SOURCE_RULE (legacy). assign_mode() on
        the scaled inputs. Prints a warning: the resulting accuracy is not a
        generalization measure.

    assign_mode() is run on the scaled-but-unpruned dataframe (not the
    variance/correlation-pruned one): a freshly computed feature could end
    up with variance < 0.001 or be highly correlated with another column and
    get pruned below, and assign_mode() needs every column it references to
    still exist.

    Args:
        csv_path: Path to final_features.csv (as produced by
            feature_extraction.extract_all_features).
        gt_csv_path: Optional ground-truth sidecar aligned with csv_path.
        gt_thresholds: Cut points for the GT labels. If None (and
            gt_csv_path is given) they are fitted on this table's annotated
            rows -- pass the persisted training thresholds for any table
            that is NOT the training pool.

    Returns:
        A tuple (X, df_cleaned, y, feature_scaler, label_encoder):
            X: pruned + scaled feature columns (RandomForest input).
            df_cleaned: X plus 'mode'/'mode_encoded' columns -- what gets
                saved to final_features_cleaned.csv.
            y: mode_encoded labels, aligned with X's row order.
            feature_scaler: fitted FeatureScaler for single-image inference.
            label_encoder: fitted LabelEncoder (use .classes_ for reporting,
                since its alphabetical fit order is not Normal=0/Degraded=1/
                Takeover=2).
    """
    df = pd.read_csv(csv_path)
    df = df.fillna(0)

    visibility_max = df["visibility"].max()
    road_quality_max = df["road_quality"].max()

    df_norm = df.copy()
    df_norm["brightness"] = df_norm["brightness"] / 255.0
    df_norm["visibility"] = df_norm["visibility"] / visibility_max
    df_norm["road_quality"] = df_norm["road_quality"] / road_quality_max

    feature_columns = df_norm.columns.tolist()
    minmax_scaler = MinMaxScaler()
    scaled_values = minmax_scaler.fit_transform(df_norm[feature_columns])
    df_scaled = pd.DataFrame(scaled_values, columns=feature_columns, index=df_norm.index)

    if gt_csv_path:
        gt_df = _read_gt_sidecar(gt_csv_path, len(df))
        if gt_thresholds is None:
            gt_thresholds = fit_gt_label_thresholds(gt_df)
        modes = assign_gt_mode(gt_df, gt_thresholds)
        keep = modes.notna()
        print(
            f"Label source: ground-truth annotations ({int(keep.sum())} of {len(df)} rows have "
            f"box annotations; the rest are excluded from classifier training). "
            f"Scene-complexity cut points: Normal <= {gt_thresholds.low:g} < Degraded <= "
            f"{gt_thresholds.high:g} < Takeover."
        )
        df_scaled = df_scaled[keep]
        modes = modes[keep]
    else:
        print(
            "WARNING: label source is the assign_mode() rule applied to the classifier's own "
            "inputs. Any model scores ~100% on such labels; the accuracy below is NOT a "
            "generalization measure. Pass a ground-truth sidecar to use annotation-derived labels."
        )
        modes = df_scaled.apply(assign_mode, axis=1)

    label_encoder = LabelEncoder()
    mode_encoded = label_encoder.fit_transform(modes)
    print(
        "Label encoding:",
        dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))),
    )
    print("Label distribution:", modes.value_counts().to_dict())

    variances = df_scaled.var()
    low_variance_cols = variances[variances < 0.001].index.tolist()
    df_pruned = df_scaled.drop(columns=low_variance_cols)
    if low_variance_cols:
        print(f"Dropped {len(low_variance_cols)} low-variance columns: {low_variance_cols}")

    corr_matrix = df_pruned.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > 0.9).any()]
    df_pruned = df_pruned.drop(columns=to_drop)
    if to_drop:
        print(f"Dropped {len(to_drop)} highly-correlated columns: {to_drop}")

    feature_scaler = FeatureScaler(
        visibility_max=visibility_max,
        road_quality_max=road_quality_max,
        minmax_scaler=minmax_scaler,
        feature_columns=feature_columns,
    )

    df_cleaned = df_pruned.copy()
    df_cleaned["mode"] = modes.values
    df_cleaned["mode_encoded"] = mode_encoded

    y = pd.Series(mode_encoded, name="mode_encoded", index=df_pruned.index)
    return df_pruned, df_cleaned, y, feature_scaler, label_encoder


# --------------------------------------------------------------------------
# Part C: Train ODD classifier
# --------------------------------------------------------------------------


def train_odd_classifier(
    X: pd.DataFrame, y: pd.Series, model_path: str = "odd_classifier.pkl"
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series, np.ndarray]:
    """Trains and evaluates a RandomForest ODD classifier.

    Args:
        X: Pruned, scaled feature matrix.
        y: Encoded mode labels.
        model_path: Where to save the trained model via joblib.

    Returns:
        A tuple (model, X_test, y_test, y_pred).
    """
    class_counts = y.value_counts()
    if (class_counts < 2).any():
        print(
            f"Warning: at least one ODD class has fewer than 2 samples "
            f"({class_counts.to_dict()}); train/test split or training may "
            "behave degenerately."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # n_jobs=-1 parallelizes tree building across cores; with a fixed
    # random_state the resulting forest is identical to the single-threaded fit.
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Majority-class baseline accuracy: {_majority_baseline(np.asarray(y_test)):.4f}")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, model_path)
    print(f"Saved trained ODD classifier to '{model_path}'")

    return model, X_test, y_test, y_pred


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    save_dir: str = PLOTS_DIR,
    filename: str = "confusion_matrix.png",
    title: str = "ODD Classifier Confusion Matrix",
) -> None:
    """Plots and saves a confusion-matrix heatmap.

    Args:
        y_test: Ground-truth encoded labels.
        y_pred: Predicted encoded labels.
        label_encoder: Fitted LabelEncoder, used for correctly-ordered axis
            labels regardless of its actual (alphabetical) class order.
        save_dir: Directory to save the plot into.
        filename: Output filename within save_dir. Override this (e.g. to
            "confusion_matrix_holdout.png") when plotting a second
            confusion matrix so it doesn't overwrite the first.
        title: Plot title.
    """
    ordered_labels = label_encoder.transform(label_encoder.classes_)
    cm = confusion_matrix(y_test, y_pred, labels=ordered_labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: List[str],
    save_dir: str = PLOTS_DIR,
    top_n: int = 10,
) -> None:
    """Plots and saves the top-N RandomForest feature importances.

    Args:
        model: A trained RandomForestClassifier.
        feature_names: Column names aligned with the model's training X.
        save_dir: Directory to save the plot into.
        top_n: Number of top features to display.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    top_features.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "feature_importance.png"))
    plt.close()


def evaluate_on_holdout(
    model: RandomForestClassifier,
    feature_scaler: FeatureScaler,
    label_encoder: LabelEncoder,
    train_feature_columns: List[str],
    csv_path: str,
    gt_csv_path: Optional[str] = None,
    gt_thresholds: Optional[GTLabelThresholds] = None,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Evaluates an already-trained ODD classifier on genuinely held-out data.

    Unlike train_odd_classifier()'s internal train_test_split (a random
    slice of the SAME pool of images used to build the training feature
    table), this is meant to be called on a feature CSV built from a
    dataset's real val split -- images never touched by SegNet training,
    training-feature extraction, or classifier training. Applies the
    ALREADY-FITTED feature_scaler and label_encoder from training via
    their .transform() (never .fit_transform()), so nothing about the
    held-out data's own distribution leaks into preprocessing -- this is
    what makes the resulting accuracy a genuine generalization estimate.

    Args:
        model: Trained RandomForestClassifier (from train_odd_classifier).
        feature_scaler: The FeatureScaler fitted during training.
        label_encoder: The LabelEncoder fitted during training.
        train_feature_columns: X.columns.tolist() from training, so the
            held-out data is restricted to the exact same (variance/
            correlation-pruned) column set the model was actually fit on --
            feature_scaler.transform() alone returns the unpruned set.
        csv_path: Path to a raw feature CSV built from held-out images.
        gt_csv_path: Ground-truth sidecar aligned with csv_path. Required for
            annotation-derived labels; None falls back to the assign_mode()
            rule (with the same caveat as in load_and_clean_features).
        gt_thresholds: The TRAINING pool's persisted cut points. Must be
            given with gt_csv_path -- refitting them here would define the
            held-out label differently from the training label.

    Returns:
        A tuple (metrics, y_true, y_pred).
    """
    df = pd.read_csv(csv_path).fillna(0)
    df_scaled = feature_scaler.transform(df)

    if gt_csv_path:
        if gt_thresholds is None:
            raise ValueError("gt_thresholds (fitted on the training pool) are required with gt_csv_path.")
        gt_df = _read_gt_sidecar(gt_csv_path, len(df))
        modes = assign_gt_mode(gt_df, gt_thresholds)
        keep = modes.notna()
        print(f"Held-out label source: ground-truth annotations ({int(keep.sum())} of {len(df)} rows annotated).")
        df_scaled = df_scaled[keep]
        modes = modes[keep]
    else:
        print("WARNING: held-out labels come from the assign_mode() rule on the inputs (see load_and_clean_features).")
        modes = df_scaled.apply(assign_mode, axis=1)
    y_true = label_encoder.transform(modes)

    X_holdout = df_scaled[train_feature_columns]
    y_pred = model.predict(X_holdout)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    baseline = _majority_baseline(y_true)
    print(f"Held-out majority-class baseline accuracy: {baseline:.4f}")
    report = classification_report(
        y_true, y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    print(f"Held-out accuracy: {accuracy:.4f}")
    print(f"Held-out weighted F1: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    metrics = {
        "num_holdout_rows": int(len(y_true)),
        "num_rows_in_csv": int(len(df)),
        "label_source": LABEL_SOURCE_GT if gt_csv_path else LABEL_SOURCE_RULE,
        "accuracy": accuracy,
        "majority_class_baseline_accuracy": baseline,
        "weighted_f1": f1,
        "classification_report": report,
    }
    return metrics, y_true, y_pred


# --------------------------------------------------------------------------
# Part D: End-to-end single-image inference
# --------------------------------------------------------------------------


_VERDICT_EMOJI = {
    "Normal": "\U0001F7E2 NORMAL",
    "Degraded": "\U0001F7E0 DEGRADED",
    "Takeover": "\U0001F534 TAKEOVER",
}
_BBOX_COLORS = {"vehicle": (0, 255, 0), "two_wheeler": (0, 0, 255), "pedestrian": (255, 0, 0)}


def evaluate_road_scene(
    image_path: str,
    segnet_model: nn.Module,
    yolo_model: YOLO,
    odd_classifier: RandomForestClassifier,
    scaler: FeatureScaler,
    label_encoder: LabelEncoder,
    device: torch.device = DEVICE,
    save_path: Optional[str] = None,
) -> str:
    """Runs the full perception-to-ODD pipeline on a single road image.

    Args:
        image_path: Path to an image file.
        segnet_model: Trained SegNet.
        yolo_model: Loaded Ultralytics YOLO model.
        odd_classifier: Trained RandomForest ODD classifier.
        scaler: Fitted FeatureScaler matching the classifier's training
            transform.
        label_encoder: LabelEncoder used to decode the classifier's output.
        device: Torch device to run SegNet inference on.
        save_path: Optional path to save the 3-panel visualization figure.

    Returns:
        The predicted ODD mode string ("Normal", "Degraded", or "Takeover").

    Raises:
        FileNotFoundError: If image_path cannot be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_resized = cv2.resize(img, IMAGE_SIZE)

    img_t = torch.tensor(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    segnet_model.eval()
    with torch.no_grad():
        output = segnet_model(img_t)
    mask = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    detections = run_detection(img_resized, yolo_model)
    features = compute_features(img_resized, mask, detections)

    scaled_df = scaler.transform(pd.DataFrame([features]))
    pred_encoded = odd_classifier.predict(scaled_df)[0]
    mode = label_encoder.inverse_transform([pred_encoded])[0]
    verdict = _VERDICT_EMOJI[mode]

    vis_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        if det["class"] in VEHICLES:
            color = _BBOX_COLORS["vehicle"]
        elif det["class"] in TWO_WHEELERS:
            color = _BBOX_COLORS["two_wheeler"]
        elif det["class"] in PEDESTRIANS:
            color = _BBOX_COLORS["pedestrian"]
        else:
            continue
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, 2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(vis_img)
    axes[0].set_title("Original + YOLO Detections")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="tab20")
    axes[1].set_title("SegNet Mask")
    axes[1].axis("off")

    axes[2].axis("off")
    axes[2].set_title("Feature Summary & Verdict")
    summary_lines = [
        f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in features.items()
    ]
    summary_text = "\n".join(summary_lines) + f"\n\nVerdict: {verdict}"
    axes[2].text(0.02, 0.98, summary_text, va="top", ha="left", fontsize=9, family="monospace")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

    print(f"Predicted ODD mode: {verdict}")
    return mode


if __name__ == "__main__":
    import sys

    from src.common.paths import (
        FEATURES_CLEANED_CSV,
        FEATURES_CSV,
        FEATURE_SCALER_PATH,
        ODD_CLASSIFIER_PATH,
        ensure_output_dirs,
    )

    # Completes the __module__ override above: pickle's consistency check
    # requires sys.modules[obj.__module__] to actually contain the class
    # being pickled. Aliasing this running "__main__" module under its real
    # importable name satisfies that check, so feature_scaler.pkl loads
    # correctly from any other script that does `import src.odd.odd_classifier`.
    sys.modules.setdefault("src.odd.odd_classifier", sys.modules[__name__])

    ensure_output_dirs()
    csv_path = FEATURES_CSV
    cleaned_csv_path = FEATURES_CLEANED_CSV
    classifier_path = ODD_CLASSIFIER_PATH
    scaler_path = FEATURE_SCALER_PATH

    X, df_cleaned, y, feature_scaler, label_encoder = load_and_clean_features(csv_path)
    df_cleaned.to_csv(cleaned_csv_path, index=False)
    print(f"Saved cleaned features -> '{cleaned_csv_path}'")

    print("Mode distribution:")
    print(df_cleaned["mode"].value_counts())

    model, X_test, y_test, y_pred = train_odd_classifier(X, y, model_path=classifier_path)
    plot_confusion_matrix(y_test, y_pred, label_encoder)
    plot_feature_importance(model, X.columns.tolist())

    joblib.dump(feature_scaler, scaler_path)
    print(f"Saved feature scaler to '{scaler_path}'")
