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

import os
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


def assign_mode(row: pd.Series) -> str:
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

    Returns:
        One of "Normal", "Degraded", "Takeover".
    """
    has_detections = row.get("object_presence", 0) > 0
    det_conf_ok = (row["detection_confidence"] > 0.85) if has_detections else True

    # NORMAL: full ADS operation within ODD (SAE L2, driver supervising)
    if (
        det_conf_ok
        and row["visibility"] > 0.45
        and row["traffic_density"] < 0.4
        and row["non_drivable_area_score"] < 0.2
        and row["living_things_score"] < 0.2
    ):
        return "Normal"

    # DEGRADED: cautious/supervised operation, closer driver attention needed
    elif (
        ((row["detection_confidence"] > 0.5) if has_detections else True)
        and row["visibility"] > 0.25
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


def load_and_clean_features(
    csv_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, FeatureScaler, LabelEncoder]:
    """Loads, normalizes, labels, and prunes the raw feature CSV.

    assign_mode() is run on the scaled-but-unpruned dataframe (not the
    variance/correlation-pruned one): a freshly computed feature could end
    up with variance < 0.001 or be highly correlated with another column and
    get pruned below, and assign_mode() needs every column it references to
    still exist.

    Args:
        csv_path: Path to final_features.csv (as produced by
            feature_extraction.extract_all_features).

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

    modes = df_scaled.apply(assign_mode, axis=1)

    label_encoder = LabelEncoder()
    mode_encoded = label_encoder.fit_transform(modes)
    print(
        "Label encoding:",
        dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))),
    )

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

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

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
) -> None:
    """Plots and saves a confusion-matrix heatmap.

    Args:
        y_test: Ground-truth encoded labels.
        y_pred: Predicted encoded labels.
        label_encoder: Fitted LabelEncoder, used for correctly-ordered axis
            labels regardless of its actual (alphabetical) class order.
        save_dir: Directory to save the plot into.
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
    plt.title("ODD Classifier Confusion Matrix")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
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
