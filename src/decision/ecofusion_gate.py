"""Stage 7 (Decision System): EcoFusion energy/compute-aware branch selection.

Implements Malawade et al. 2022's ("EcoFusion: Energy-Aware Adaptive
Sensor Fusion for Efficient Autonomous Vehicle Perception", ACM/IEEE DAC)
adaptive fusion-configuration selection algorithm:

    Energy model:  E(phi, X) = P(phi, X) * t(phi, X)                    (Eq. 6)
    Candidates:    Phi* = {phi in Phi : Lf(phi) - Lf(phi') <= gamma}    (Eq. 7,
                   using the prose-consistent inequality -- the paper's
                   own printed equation duplicates a term and is treated
                   here as a transcription artifact, per direct reading)
    Joint loss:    L_joint(phi, lambda_E) = (1-lambda_E)*Lf(phi) + lambda_E*E(phi)  (Eq. 8)
    Selection:     phi* = argmin_{phi in Phi*} L_joint(phi, lambda_E)   (Eq. 9)
    Algorithm 1:   stem -> gate(Lf(Phi)) -> filter(Phi*) -> joint-loss argmin
                   -> run only the selected branches -> fuse

This project has one camera and three derived "sensor" streams (the
SegNet mask, YOLO detections, raw-image statistics -- see
multi_stream_fusion.py's STREAMS), standing in for the paper's physical
camera/LiDAR/radar sensors. "Energy" is replaced with measured wall-clock
latency per branch (this machine has no power-draw instrumentation);
Phi is the set of all 7 non-empty subsets of the 3 streams.

Two gating strategies are implemented, matching the paper's Knowledge
Gating and Deep Gating designs (the other two -- Attention Gating and the
ground-truth Loss-Based oracle -- require either a live simulator or
labeled ground truth this project doesn't have):

    - Knowledge Gating: Lf(phi) computed directly from
      multi_stream_fusion.py's per-stream reliability scores. This
      requires the branches to have already run to know their
      reliabilities, so it's useful for offline analysis/lambda_E tuning,
      not real deployment-time gating.
    - Deep Gating: a small regressor trained to predict those same
      reliability scores from CHEAP "stem" features only (brightness,
      Laplacian-variance sharpness, pixel-intensity std -- all computable
      directly from the raw image, without running SegNet or YOLO at
      all), so it CAN gate a real decision before paying for the
      expensive branches, matching the paper's actual deployment intent.
"""

import itertools
import os
import time
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

from src.perception.multi_stream_fusion import STREAMS, calibrate_references, compute_stream_reliabilities

RANDOM_STATE = 42
STEM_FEATURES = ("brightness", "visibility", "wetness")
DEFAULT_GAMMA = 0.5  # matches Malawade et al.'s own experimental choice
DEFAULT_LAMBDA_E = 0.1

# All non-empty subsets of the 3 streams (Phi, the fusion-configuration space).
CONFIGURATION_SPACE: List[FrozenSet[str]] = [
    frozenset(combo)
    for r in range(1, len(STREAMS) + 1)
    for combo in itertools.combinations(STREAMS, r)
]


def compute_stem_features(image: np.ndarray) -> Dict[str, float]:
    """Computes the cheap 'stem' features available before any expensive branch runs.

    Args:
        image: BGR image array of shape (H, W, 3).

    Returns:
        A dict with 'brightness', 'visibility', 'wetness' -- all computable
        via plain OpenCV/NumPy on the raw pixels, no SegNet or YOLO needed.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(np.mean(image)),
        "visibility": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "wetness": float(np.std(image)),
    }


def profile_branch_latencies(
    image: np.ndarray, segnet_model: torch.nn.Module, yolo_model, num_trials: int = 3
) -> Dict[str, float]:
    """Measures each branch's average wall-clock latency (stand-in for energy).

    Args:
        image: A representative BGR sample image of shape (H, W, 3).
        segnet_model: Trained SegNet.
        yolo_model: Loaded Ultralytics YOLO model.
        num_trials: Number of repeated timing runs to average over.

    Returns:
        A dict mapping each of multi_stream_fusion.STREAMS to its measured
        latency in seconds.
    """
    from src.perception.feature_extraction import run_detection

    img_tensor = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    segnet_model.eval()
    start = time.perf_counter()
    for _ in range(num_trials):
        with torch.no_grad():
            segnet_model(img_tensor)
    segnet_latency = (time.perf_counter() - start) / num_trials

    start = time.perf_counter()
    for _ in range(num_trials):
        run_detection(image, yolo_model)
    yolo_latency = (time.perf_counter() - start) / num_trials

    start = time.perf_counter()
    for _ in range(num_trials):
        compute_stem_features(image)
    image_latency = (time.perf_counter() - start) / num_trials

    return {
        "segnet_mask": segnet_latency,
        "yolo_detections": yolo_latency,
        "image_stats": max(image_latency, 1e-6),  # avoid a literal-zero cost branch
    }


def energy_cost(config: FrozenSet[str], branch_latencies: Dict[str, float]) -> float:
    """Computes E(phi, X) = sum of active branches' latency (Eq. 6, energy->latency).

    Args:
        config: A fusion configuration (subset of STREAMS).
        branch_latencies: Per-stream latency, as from profile_branch_latencies().

    Returns:
        Total latency (s) of running every branch in config.
    """
    return sum(branch_latencies[stream] for stream in config)


def knowledge_gate_loss(config: FrozenSet[str], reliabilities: Dict[str, float]) -> float:
    """Knowledge-Gating Lf(phi): fraction of total reliability the config OMITS.

    Lf(phi) = 1 - sum_{s in phi} R_s / sum_{s in STREAMS} R_s

    Args:
        config: A fusion configuration.
        reliabilities: Per-stream reliability scores (already computed --
            see multi_stream_fusion.compute_stream_reliabilities).

    Returns:
        A loss in [0.0, 1.0]; 0.0 for the full configuration.
    """
    total = sum(reliabilities.values())
    if total <= 0:
        return 0.0
    captured = sum(reliabilities[s] for s in config)
    return 1.0 - captured / total


def fit_deep_gate(fused_df: pd.DataFrame) -> RandomForestRegressor:
    """Trains a Deep-Gating regressor to predict per-stream reliability from stem features.

    Predicts multi_stream_fusion.py's reliability_<stream> columns from the
    cheap STEM_FEATURES alone, so the gate can be queried before running
    any expensive branch (matching the paper's deployment-time intent).

    Args:
        fused_df: Output of multi_stream_fusion.fuse_dataframe(), containing
            STEM_FEATURES columns and 'reliability_<stream>' columns.

    Returns:
        A fitted, multi-output RandomForestRegressor.
    """
    X = fused_df[list(STEM_FEATURES)].to_numpy()
    y = fused_df[[f"reliability_{s}" for s in STREAMS]].to_numpy()
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


def deep_gate_predict_reliabilities(
    model: RandomForestRegressor, stem_features: Dict[str, float]
) -> Dict[str, float]:
    """Predicts per-stream reliability from stem features via the Deep Gate.

    Args:
        model: A fitted Deep-Gating regressor (fit_deep_gate()).
        stem_features: Output of compute_stem_features().

    Returns:
        A dict mapping each stream to a predicted reliability in [0.0, 1.0].
    """
    x = np.array([[stem_features[f] for f in STEM_FEATURES]])
    predicted = model.predict(x)[0]
    return {stream: float(np.clip(value, 0.0, 1.0)) for stream, value in zip(STREAMS, predicted)}


def filter_pareto_candidates(
    losses: Dict[FrozenSet[str], float], gamma: float = DEFAULT_GAMMA
) -> List[FrozenSet[str]]:
    """Filters the configuration space to near-loss-optimal candidates (Eq. 7).

    Phi* = {phi in Phi : Lf(phi) - Lf(phi') <= gamma}, where phi' is the
    minimum-loss configuration (necessarily the full configuration, since
    Lf is monotonically non-increasing as streams are added).

    Args:
        losses: Lf(phi) for every phi in the configuration space.
        gamma: Maximum allowable loss gap from the optimum.

    Returns:
        The filtered candidate list Phi* (at least contains the argmin).
    """
    best_loss = min(losses.values())
    return [config for config, loss in losses.items() if loss - best_loss <= gamma]


def select_configuration(
    candidates: List[FrozenSet[str]],
    losses: Dict[FrozenSet[str], float],
    branch_latencies: Dict[str, float],
    lambda_e: float = DEFAULT_LAMBDA_E,
) -> Tuple[FrozenSet[str], float]:
    """Selects phi* = argmin joint loss over the Pareto-filtered candidates (Eqs. 8-9).

    Args:
        candidates: Phi*, as from filter_pareto_candidates().
        losses: Lf(phi) for every phi in candidates.
        branch_latencies: Per-stream latency (energy_cost() input).
        lambda_e: Trade-off weight in [0.0, 1.0]; 0 = pure accuracy,
            1 = pure speed.

    Returns:
        A tuple (phi*, L_joint(phi*, lambda_e)).
    """
    best_config, best_joint_loss = None, float("inf")
    for config in candidates:
        joint_loss = (1 - lambda_e) * losses[config] + lambda_e * energy_cost(config, branch_latencies)
        if joint_loss < best_joint_loss:
            best_config, best_joint_loss = config, joint_loss
    return best_config, best_joint_loss


@dataclass
class EcoFusionDecision:
    """The outcome of one EcoFusion gating decision.

    Attributes:
        selected_config: The chosen subset of STREAMS to actually run.
        joint_loss: The joint loss achieved by selected_config.
        candidates: Phi*, the Pareto-filtered candidate configurations.
        losses: Lf(phi) for every phi in the full configuration space.
    """

    selected_config: FrozenSet[str]
    joint_loss: float
    candidates: List[FrozenSet[str]]
    losses: Dict[FrozenSet[str], float]


def run_ecofusion_gate(
    stem_features: Dict[str, float],
    deep_gate: RandomForestRegressor,
    branch_latencies: Dict[str, float],
    gamma: float = DEFAULT_GAMMA,
    lambda_e: float = DEFAULT_LAMBDA_E,
) -> EcoFusionDecision:
    """Runs the full EcoFusion decision (Algorithm 1) using the Deep Gate.

    Args:
        stem_features: Output of compute_stem_features() for the current frame.
        deep_gate: A fitted Deep-Gating regressor.
        branch_latencies: Per-stream latency.
        gamma: Pareto-filtering tolerance.
        lambda_e: Accuracy/speed trade-off weight.

    Returns:
        An EcoFusionDecision.
    """
    predicted_reliabilities = deep_gate_predict_reliabilities(deep_gate, stem_features)
    losses = {config: knowledge_gate_loss(config, predicted_reliabilities) for config in CONFIGURATION_SPACE}
    candidates = filter_pareto_candidates(losses, gamma)
    selected_config, joint_loss = select_configuration(candidates, losses, branch_latencies, lambda_e)
    return EcoFusionDecision(
        selected_config=selected_config, joint_loss=joint_loss, candidates=candidates, losses=losses
    )


def run_selected_branches(
    image: np.ndarray,
    config: FrozenSet[str],
    segnet_model: Optional[torch.nn.Module],
    yolo_model,
) -> Dict[str, object]:
    """Executes only the branches named in config, skipping the rest.

    This is where EcoFusion's compute savings are actually realized: a
    config that omits 'segnet_mask' never pays for a SegNet forward pass.

    Args:
        image: BGR image array of shape (H, W, 3).
        config: The selected fusion configuration.
        segnet_model: Trained SegNet (only used if 'segnet_mask' in config).
        yolo_model: Loaded YOLO model (only used if 'yolo_detections' in config).

    Returns:
        A dict with whichever of 'mask', 'detections', 'stem_features' keys
        correspond to the branches that were actually run (others absent).
    """
    from src.perception.feature_extraction import run_detection

    results: Dict[str, object] = {}

    if "segnet_mask" in config:
        img_tensor = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        segnet_model.eval()
        with torch.no_grad():
            output = segnet_model(img_tensor)
        results["mask"] = output.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)

    if "yolo_detections" in config:
        results["detections"] = run_detection(image, yolo_model)

    if "image_stats" in config:
        results["stem_features"] = compute_stem_features(image)

    return results


if __name__ == "__main__":
    from ultralytics import YOLO

    from src.common.paths import (
        DATA_DIR,
        ECOFUSION_DEEP_GATE_PATH,
        FEATURES_CSV,
        SEGNET_CHECKPOINT,
        YOLO_WEIGHTS,
        ensure_output_dirs,
    )
    from src.perception.data_pipeline import load_and_clean_dataset
    from src.perception.segnet_model import load_segnet

    ensure_output_dirs()
    features_path = FEATURES_CSV
    dataset_dir = DATA_DIR

    features_df = pd.read_csv(features_path)
    refs = calibrate_references(features_df)

    reliabilities_df = features_df.apply(
        lambda row: pd.Series(compute_stream_reliabilities(row, refs)), axis=1
    )
    for stream in STREAMS:
        features_df[f"reliability_{stream}"] = reliabilities_df[stream]

    deep_gate = fit_deep_gate(features_df)
    joblib.dump(deep_gate, ECOFUSION_DEEP_GATE_PATH)
    print("Trained and saved EcoFusion Deep Gate.")

    images, _ = load_and_clean_dataset(dataset_dir)
    segnet = load_segnet(SEGNET_CHECKPOINT)
    yolo = YOLO(YOLO_WEIGHTS)
    branch_latencies = profile_branch_latencies(images[0], segnet, yolo)
    print(f"Profiled branch latencies (s): {branch_latencies}")

    for lambda_e in (0.0, 0.1, 0.5, 0.9):
        stem = compute_stem_features(images[0])
        decision = run_ecofusion_gate(stem, deep_gate, branch_latencies, lambda_e=lambda_e)
        print(
            f"lambda_E={lambda_e}: selected={sorted(decision.selected_config)}, "
            f"joint_loss={decision.joint_loss:.4f}, num_candidates={len(decision.candidates)}"
        )
