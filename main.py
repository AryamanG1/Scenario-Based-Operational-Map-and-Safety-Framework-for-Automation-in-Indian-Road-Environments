"""Orchestrates the full 7-stage ODD safety-framework pipeline end-to-end.

Stages, matching the capstone proposal's actual architecture (not the
simplified 5-stage framing of the one-shot prompt this project started
from):

    1. Input Data                    -- data_pipeline.py
    2. Perception Layer               -- segnet_model.py, feature_extraction.py,
                                          lane_detection.py, detection_benchmark.py,
                                          multi_stream_fusion.py
    3. Traffic Density Estimation      -- traffic_density.py
    4. Scenario Classification         -- fuzzy_odd.py
    5. ODD Mapping                     -- odd_boundary.py, rss_safety.py, risk_estimation.py
    6. Real-Time Performance Monitoring -- perception_monitor.py
    7. Decision System                 -- odd_classifier.py, ecofusion_gate.py, feasibility_map.py

Stages 2 and 4 (the two slowest artifact-producing steps: SegNet training
and feature extraction) are skipped and loaded from cache if their
artifacts already exist, unless --force is passed. Stage 6's per-frame
monitoring is demonstrated on a handful of sample frames rather than the
full dataset (see perception_monitor.py / feasibility_map.py docstrings
for why a full-dataset run is too expensive to precompute).
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from ultralytics import YOLO

from src.perception.data_pipeline import load_and_clean_dataset
from src.perception.detection_benchmark import evaluate_vehicle_detections
from src.decision.ecofusion_gate import (
    compute_stem_features,
    fit_deep_gate,
    profile_branch_latencies,
    run_ecofusion_gate,
)
from src.decision.feasibility_map import build_feasibility_map
from src.perception.feature_extraction import extract_all_features
from src.odd.fuzzy_odd import classify_dataframe as classify_scenario
from src.perception.lane_detection import compute_lane_features, detect_lanes
from src.perception.multi_stream_fusion import calibrate_references, compute_stream_reliabilities, fuse_dataframe
from src.odd.odd_boundary import DEFAULT_ODD_VARIABLES, classify_odd_region, fit_odd_copula
from src.odd.odd_classifier import (
    combine_stage_outputs,
    load_and_clean_features,
    plot_confusion_matrix,
    plot_feature_importance,
    train_odd_classifier,
)
from src.monitoring.perception_monitor import run_perception_monitor
from src.monitoring.perturbation_engine import evaluate_perturbation_robustness
from src.odd.risk_estimation import estimate_odd_failure_probability
from src.decision.sae_taxonomy import LEVEL_NAMES, THIS_PROJECT_DDT, classify_sae_level
from src.perception.segnet_model import load_segnet, plot_training_curves, train_segnet
from src.odd.traffic_density import classify_dataframe as classify_traffic
from src.simulation.carla_config import load_carla_config
from src.simulation.closed_loop_runner import run_closed_loop_simulation, write_carla_live_js
from src.common.paths import (
    CARLA_CONFIG_JSON,
    DASHBOARD_CARLA_LIVE_JS,
    DATA_DIR as DATASET_DIR,
    FEASIBILITY_MAP_CSV,
    FEATURES_CLEANED_CSV as CLEANED_CSV,
    FEATURES_CSV,
    FEATURE_SCALER_PATH as SCALER_PATH,
    ODD_CLASSIFIER_PATH as CLASSIFIER_PATH,
    SEGNET_CHECKPOINT,
    YOLO_WEIGHTS,
    ensure_output_dirs,
)

ensure_output_dirs()


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the orchestrator.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Run the full 7-stage ODD safety-framework pipeline.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run SegNet training and feature extraction even if cached artifacts exist.",
    )
    parser.add_argument(
        "--monitor_samples",
        type=int,
        default=3,
        help="Number of sample frames to demonstrate Stage 6 real-time monitoring on.",
    )
    parser.add_argument(
        "--carla",
        action="store_true",
        help="Also run Stage 8 (closed-loop simulation) after Stage 7. Uses a real CARLA "
        "server if configs/carla_config.json enables it and one is reachable; falls back "
        "to the local kinematic simulator otherwise. See src/simulation/closed_loop_runner.py "
        "for the primary, more configurable entry point: "
        "`python -m src.simulation.closed_loop_runner --num_ticks 50`.",
    )
    parser.add_argument("--carla_ticks", type=int, default=50, help="Number of Stage 8 ticks to run (only used with --carla).")
    parser.add_argument("--carla_config", type=str, default=CARLA_CONFIG_JSON, help="Path to a carla_config.json (only used with --carla).")
    return parser.parse_args()


def main() -> None:
    """Runs Stages 1-7 of the ODD safety-framework pipeline end-to-end."""
    args = _parse_args()

    print("=" * 60)
    print("STAGE 1: Input Data")
    print("=" * 60)
    images, labels = load_and_clean_dataset(DATASET_DIR)
    print(f"Clean dataset size: {len(images)} images")

    print("=" * 60)
    print("STAGE 2: Perception Layer")
    print("=" * 60)
    final_val_loss = None
    if os.path.isfile(SEGNET_CHECKPOINT) and not args.force:
        print(f"Found existing checkpoint '{SEGNET_CHECKPOINT}', loading instead of retraining.")
        segnet = load_segnet(SEGNET_CHECKPOINT)
    else:
        train_losses, val_losses = train_segnet(
            images, labels, epochs=30, batch_size=8, save_path=SEGNET_CHECKPOINT
        )
        plot_training_curves(train_losses, val_losses)
        segnet = load_segnet(SEGNET_CHECKPOINT)
        final_val_loss = val_losses[-1]

    yolo = YOLO(YOLO_WEIGHTS)

    if os.path.isfile(FEATURES_CSV) and not args.force:
        print(f"Found existing '{FEATURES_CSV}', skipping re-extraction.")
        features_df = pd.read_csv(FEATURES_CSV)
    else:
        features_df = extract_all_features(images, labels, segnet, yolo, save_path=FEATURES_CSV)

    lane_result = detect_lanes(images[0], labels[0])
    print(f"Lane detection sample: {compute_lane_features(lane_result)}")

    detection_bench = evaluate_vehicle_detections(images[:100], labels[:100], yolo)
    print(f"YOLO vehicle detection benchmark (100 images): {detection_bench}")

    refs = calibrate_references(features_df)
    fused_sample = fuse_dataframe(features_df.head(50), refs)
    print(f"Multi-stream fusion mean confidence (50 images): {fused_sample['fused_confidence'].mean():.3f}")

    print("=" * 60)
    print("STAGE 3: Traffic Density Estimation")
    print("=" * 60)
    traffic_df, density_thresholds = classify_traffic(features_df)
    print(traffic_df["traffic_density_level"].value_counts())

    print("=" * 60)
    print("STAGE 4: Scenario Classification")
    print("=" * 60)
    scenario_df, pd_breakpoints, _ = classify_scenario(traffic_df, density_thresholds=density_thresholds)
    print(scenario_df["mu_odd_label"].value_counts())

    print("=" * 60)
    print("STAGE 5: ODD Mapping")
    print("=" * 60)
    copula = fit_odd_copula(features_df, DEFAULT_ODD_VARIABLES)
    regions = features_df[DEFAULT_ODD_VARIABLES].apply(
        lambda row: classify_odd_region(copula, dict(zip(DEFAULT_ODD_VARIABLES, row))), axis=1
    )
    print("ODD region distribution:")
    print(regions.value_counts())

    risk_result = estimate_odd_failure_probability(copula, features_df, severity_threshold=0.95)
    print(f"P(scene in worst 5% of ODD space) ~= {risk_result.failure_probability:.4f}")

    print("=" * 60)
    print("STAGE 6: Real-Time Performance Monitoring")
    print("=" * 60)
    print(f"Demonstrating on {args.monitor_samples} sample frames (see module docstring for why " "a full-dataset run is not batch-precomputed):")
    for i in range(min(args.monitor_samples, len(images))):
        result = run_perception_monitor(images[i], segnet, yolo)
        print(f"  frame {i}: state={result.state}, max_consecutive_bad={result.max_consecutive_bad}")

    print("=" * 60)
    print("STAGE 7: Decision System")
    print("=" * 60)
    X, df_cleaned, y, feature_scaler, label_encoder = load_and_clean_features(FEATURES_CSV)
    df_cleaned.to_csv(CLEANED_CSV, index=False)
    model, X_test, y_test, y_pred = train_odd_classifier(X, y, model_path=CLASSIFIER_PATH)
    plot_confusion_matrix(y_test, y_pred, label_encoder)
    plot_feature_importance(model, X.columns.tolist())
    joblib.dump(feature_scaler, SCALER_PATH)
    odd_accuracy = accuracy_score(y_test, y_pred)

    sae_level = classify_sae_level(THIS_PROJECT_DDT)
    print(f"This system's SAE level: {sae_level} ({LEVEL_NAMES[sae_level]})")

    reliabilities = compute_stream_reliabilities(features_df.iloc[0], refs)
    deep_gate = fit_deep_gate(fuse_dataframe(features_df, refs))
    branch_latencies = profile_branch_latencies(images[0], segnet, yolo)
    stem = compute_stem_features(images[0])
    decision = run_ecofusion_gate(stem, deep_gate, branch_latencies, lambda_e=0.1)
    print(f"EcoFusion sample decision (lambda_E=0.1): selected={sorted(decision.selected_config)}")

    print("Building Scenario-Based Feasibility Map...")
    feasibility_df = build_feasibility_map(features_df, feature_scaler)
    feasibility_df.to_csv(FEASIBILITY_MAP_CSV, index=False)
    print(feasibility_df["final_mode"].value_counts())

    if args.carla:
        print("=" * 60)
        print("STAGE 8: Closed-Loop Simulation (optional)")
        print("=" * 60)
        carla_config = load_carla_config(args.carla_config)
        carla_result_df = run_closed_loop_simulation(
            num_ticks=args.carla_ticks,
            config=carla_config,
            segnet_model=segnet,
            yolo_model=yolo,
            feature_scaler=feature_scaler,
            copula_model=copula,
            pd_breakpoints=pd_breakpoints,
        )
        print(f"Used CARLA: {bool(carla_result_df['used_carla'].mean() > 0.5)}")
        print(carla_result_df["mode"].value_counts())
        write_carla_live_js(
            carla_result_df,
            road_name="Jan Marg",
            used_carla=bool(carla_result_df["used_carla"].mean() > 0.5),
            output_path=DASHBOARD_CARLA_LIVE_JS,
        )
        print(f"Wrote dashboard replay data -> '{DASHBOARD_CARLA_LIVE_JS}'")

    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Clean dataset size: {len(images)} images")
    if final_val_loss is not None:
        print(f"SegNet final validation loss: {final_val_loss:.4f}")
    print(f"YOLO vehicle detection AP (100 images): {detection_bench['ap']:.4f}")
    print(f"ODD classifier test accuracy: {odd_accuracy:.4f}")
    print(f"P(worst 5% ODD-space scene): {risk_result.failure_probability:.4f}")
    print(f"System SAE level: {sae_level} ({LEVEL_NAMES[sae_level]})")
    print("Final combined mode distribution:")
    print(feasibility_df["final_mode"].value_counts())
    print(f"Fraction automation-feasible: {feasibility_df['automation_feasible'].mean():.3f}")


if __name__ == "__main__":
    main()
