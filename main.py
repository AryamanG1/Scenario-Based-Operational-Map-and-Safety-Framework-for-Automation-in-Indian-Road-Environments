# === MARKER: COMBINED_DATASETS_PIPELINE_V2 ===
# If `grep -n "COMBINED_DATASETS_PIPELINE_V2" main.py` finds this line, this
# checkout includes the combined-dataset update: Stage 1/2 trains SegNet on
# IDD-20K-II by default (--segnet_dataset idd20k; IDD-Lite is available via
# --segnet_dataset idd-lite, and a concatenated corpus via --segnet_dataset
# combined, but neither is the default), Stage 2 feature extraction unions
# that corpus's features with ALL of IDD117K's train split by default
# (both IDD_95kDetection + IDD_Detection parts, --idd117k_feature_limit 0),
# and the Stage 2 detection benchmark defaults to IDD117K's val split, all
# frames (--detection_gt idd117k --detection_frames 0).
# V2 adds: Stage 7 now also evaluates the trained ODD classifier on a
# genuinely held-out feature table (IDD-20K-II's + IDD117K's real val
# splits, never touched by any training step) via evaluate_on_holdout(),
# on by default (--skip_holdout_eval to disable) -- distinct from
# train_odd_classifier()'s own internal random-split test accuracy, which
# only measures held-out rows from the SAME train-sourced pool.
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
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from ultralytics import YOLO

from src.perception.data_pipeline import load_and_clean_dataset
from src.perception.detection_benchmark import (
    evaluate_vehicle_detections,
    evaluate_vehicle_detections_from_json,
)
from src.perception.idd20k_polygon_pipeline import load_split as load_idd20k_split
from src.perception.idd_detection_loader import list_pairs, sample_pairs
from src.decision.ecofusion_gate import (
    compute_stem_features,
    fit_deep_gate,
    profile_branch_latencies,
    run_ecofusion_gate,
)
from src.decision.feasibility_map import build_feasibility_map, export_pipeline_stats_js
from src.perception.feature_extraction import extract_all_features, extract_features_streaming
from src.odd.fuzzy_odd import classify_dataframe as classify_scenario
from src.perception.lane_detection import compute_lane_features, detect_lanes
from src.perception.multi_stream_fusion import calibrate_references, compute_stream_reliabilities, fuse_dataframe
from src.odd.odd_boundary import (
    DEFAULT_ODD_VARIABLES,
    classify_odd_region,
    fit_odd_copula,
    save_odd_copula,
)
from src.odd.odd_classifier import (
    combine_stage_outputs,
    evaluate_on_holdout,
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
    DETECTION_BENCHMARK_JSON,
    ECOFUSION_DEEP_GATE_PATH,
    IDD117K_95K_DIR,
    IDD117K_DETECTION_DIR,
    IDD20K_DIR,
    FEASIBILITY_MAP_CSV,
    FEATURES_CLEANED_CSV as CLEANED_CSV,
    FEATURES_COMBINED_CSV,
    FEATURES_CSV,
    FEATURES_IDD117K_CSV,
    FEATURES_IDD117K_GT_CSV,
    FEATURES_IDD117K_VAL_CSV,
    FEATURES_IDD117K_VAL_GT_CSV,
    FEATURES_VAL_CSV,
    FEATURES_VAL_COMBINED_CSV,
    FEATURE_SCALER_PATH as SCALER_PATH,
    ODD_CLASSIFIER_HOLDOUT_EVAL_JSON,
    ODD_CLASSIFIER_PATH as CLASSIFIER_PATH,
    ODD_COPULA_PATH,
    PIPELINE_STATS_JS,
    SEGNET_CHECKPOINT,
    SEGNET_COMBINED_CHECKPOINT,
    SEGNET_IDD20K_CHECKPOINT,
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
    parser.add_argument(
        "--segnet_dataset",
        choices=("idd-lite", "idd20k", "combined"),
        default="idd20k",
        help="SegNet training corpus. 'idd20k' (default) trains on IDD-20K-II's 7,034 "
        "polygon-derived frames. 'idd-lite' uses the older 1,403-frame IDD-Lite set. "
        "'combined' concatenates both (8,437 frames) -- available but not the default.",
    )
    parser.add_argument(
        "--segnet_checkpoint",
        default=None,
        help="SegNet weights to use/train. Defaults to the checkpoint matching "
        "--segnet_dataset (segnet_idd20k.pth / refined_segnet.pth / "
        "segnet_combined_lite_20k.pth) if not given explicitly.",
    )
    parser.add_argument("--segnet_epochs", type=int, default=30, help="SegNet training epochs.")
    parser.add_argument(
        "--segnet_batch_size",
        type=int,
        default=8,
        help="SegNet mini-batch size. 8 suits CPU/small GPU; raise substantially "
        "(e.g. 128) on a large GPU (e.g. an 80GB H100) to keep it busy.",
    )
    parser.add_argument(
        "--segnet_limit",
        type=int,
        default=0,
        help="Cap on IDD-20K-II frames used when --segnet_dataset is 'idd20k' or "
        "'combined' (0 = all 7,034). Useful for a fast smoke test.",
    )
    parser.add_argument(
        "--detection_gt",
        choices=("idd-lite", "idd117k"),
        default="idd117k",
        help="Ground-truth source for the Stage 2 detection benchmark. Defaults to "
        "IDD117K-Detection's real annotated boxes (requires data/IDD117K_Detection/); "
        "'idd-lite' uses connected-component pseudo-boxes from the semantic mask.",
    )
    parser.add_argument(
        "--detection_frames",
        type=int,
        default=0,
        help="Frames to run the Stage 2 detection benchmark on (0 = all).",
    )
    parser.add_argument(
        "--idd117k_parts",
        choices=("95k", "both"),
        default="both",
        help="Which IDD117K-Detection part(s) to use for feature extraction and the "
        "detection benchmark. 'both' unions IDD_95kDetection + IDD_Detection to reach "
        "the full 96,897 train / 20,202 val totals; '95k' restricts to the "
        "archive-verified IDD_95kDetection part only.",
    )
    parser.add_argument(
        "--skip_idd117k_features",
        action="store_true",
        help="Skip unioning IDD117K images into the Stage 2 feature table (reverts to "
        "the legacy corpus-only final_features.csv used by Stage 3-7).",
    )
    parser.add_argument(
        "--idd117k_feature_split",
        default="train",
        help="IDD117K split to source Stage 2's additional feature-extraction images from.",
    )
    parser.add_argument(
        "--idd117k_feature_limit",
        type=int,
        default=None,
        help="Frames to extract IDD117K features from (0 = all ~96,897 train frames). "
        "Defaults to 0 if a GPU is detected; must be set explicitly on CPU.",
    )
    parser.add_argument(
        "--skip_holdout_eval",
        action="store_true",
        help="Skip Stage 7's held-out evaluation of the ODD classifier against the "
        "real val splits of IDD-20K-II + IDD117K (images never used in any training "
        "or the training feature table) -- reverts to only the internal random-split "
        "test accuracy train_odd_classifier() already reports.",
    )
    parser.add_argument(
        "--holdout_limit",
        type=int,
        default=0,
        help="Frames to sample from each dataset's real val split for the held-out "
        "evaluation (0 = all: 1,055 IDD-20K-II val + up to 9,977 IDD117K val).",
    )
    parser.add_argument("--carla_ticks", type=int, default=50, help="Number of Stage 8 ticks to run (only used with --carla).")
    parser.add_argument("--carla_config", type=str, default=CARLA_CONFIG_JSON, help="Path to a carla_config.json (only used with --carla).")
    args = parser.parse_args()

    if args.segnet_checkpoint is None:
        args.segnet_checkpoint = {
            "idd-lite": SEGNET_CHECKPOINT,
            "idd20k": SEGNET_IDD20K_CHECKPOINT,
            "combined": SEGNET_COMBINED_CHECKPOINT,
        }[args.segnet_dataset]

    if args.idd117k_feature_limit is None and not args.skip_idd117k_features:
        if torch.cuda.is_available():
            args.idd117k_feature_limit = 0
        else:
            parser.error(
                "--idd117k_feature_limit must be set explicitly when no GPU is "
                "detected (default is 'all ~96,897 images', impractically slow on "
                "CPU). Pass e.g. --idd117k_feature_limit 2000 for a CPU smoke test, "
                "or --skip_idd117k_features to skip IDD117K feature extraction entirely."
            )

    return args


def main() -> None:
    """Runs Stages 1-7 of the ODD safety-framework pipeline end-to-end."""
    args = _parse_args()
    print("MARKER: COMBINED_DATASETS_PIPELINE_V2")

    print("=" * 60)
    print("STAGE 1: Input Data")
    print("=" * 60)
    if args.segnet_dataset == "idd-lite":
        images, labels = load_and_clean_dataset(DATASET_DIR)
    elif args.segnet_dataset == "idd20k":
        images, labels = load_idd20k_split(IDD20K_DIR, "train", limit=args.segnet_limit or None)
    else:  # "combined"
        lite_images, lite_labels = load_and_clean_dataset(DATASET_DIR)
        idd20k_images, idd20k_labels = load_idd20k_split(
            IDD20K_DIR, "train", limit=args.segnet_limit or None
        )
        images = np.concatenate([lite_images, idd20k_images], axis=0)
        labels = np.concatenate([lite_labels, idd20k_labels], axis=0)
        print(
            f"Combined SegNet corpus: {len(lite_images)} IDD-Lite + "
            f"{len(idd20k_images)} IDD-20K-II = {len(images)} total frames"
        )
    print(f"Clean dataset size: {len(images)} images")

    print("=" * 60)
    print("STAGE 2: Perception Layer")
    print("=" * 60)
    final_val_loss = None
    if os.path.isfile(args.segnet_checkpoint) and not args.force:
        print(f"Found existing checkpoint '{args.segnet_checkpoint}', loading instead of retraining.")
        segnet = load_segnet(args.segnet_checkpoint)
    else:
        train_losses, val_losses = train_segnet(
            images, labels,
            epochs=args.segnet_epochs, batch_size=args.segnet_batch_size,
            save_path=args.segnet_checkpoint,
        )
        plot_training_curves(train_losses, val_losses)
        segnet = load_segnet(args.segnet_checkpoint)
        final_val_loss = val_losses[-1]

    yolo = YOLO(YOLO_WEIGHTS)

    if os.path.isfile(FEATURES_CSV) and not args.force:
        print(f"Found existing '{FEATURES_CSV}', skipping re-extraction.")
        features_df = pd.read_csv(FEATURES_CSV)
    else:
        features_df = extract_all_features(images, labels, segnet, yolo, save_path=FEATURES_CSV)

    if not args.skip_idd117k_features:
        idd117k_feature_pairs = list_pairs(IDD117K_95K_DIR, args.idd117k_feature_split)
        if args.idd117k_parts == "both":
            try:
                idd117k_feature_pairs += list_pairs(IDD117K_DETECTION_DIR, args.idd117k_feature_split)
            except FileNotFoundError as exc:
                print(f"Warning: skipping IDD_Detection part for feature extraction ({exc})")
        idd117k_feature_pairs = sample_pairs(idd117k_feature_pairs, args.idd117k_feature_limit or None)

        if os.path.isfile(FEATURES_IDD117K_CSV) and not args.force:
            print(f"Found existing '{FEATURES_IDD117K_CSV}', skipping re-extraction.")
            features_117k_df = pd.read_csv(FEATURES_IDD117K_CSV)
        else:
            print(f"Extracting features from {len(idd117k_feature_pairs)} IDD117K '{args.idd117k_feature_split}' frame(s).")
            features_117k_df = extract_features_streaming(
                idd117k_feature_pairs, segnet, yolo,
                save_path=FEATURES_IDD117K_CSV,
                gt_save_path=FEATURES_IDD117K_GT_CSV,
            )

        features_df = pd.concat([features_df, features_117k_df], ignore_index=True)
        features_df.to_csv(FEATURES_COMBINED_CSV, index=False)
        print(f"Combined feature corpus (base + IDD117K): {len(features_df)} rows -> '{FEATURES_COMBINED_CSV}'")

    lane_result = detect_lanes(images[0], labels[0])
    print(f"Lane detection sample: {compute_lane_features(lane_result)}")

    if args.detection_gt == "idd117k":
        # IDD117K ships real per-object boxes, so vehicles that touch are
        # counted individually instead of merging into one mask blob, and the
        # ground-truth class set matches YOLO's exactly.
        val_pairs = list_pairs(IDD117K_95K_DIR, "val")
        if args.idd117k_parts == "both":
            try:
                val_pairs += list_pairs(IDD117K_DETECTION_DIR, "val")
            except FileNotFoundError as exc:
                print(f"Warning: skipping IDD_Detection part for detection benchmark ({exc})")
        pairs = sample_pairs(val_pairs, args.detection_frames or None)
        detection_bench = evaluate_vehicle_detections_from_json(pairs, yolo)
        gt_source = f"IDD117K real boxes ({args.idd117k_parts}), {len(pairs)} images"
    else:
        n = args.detection_frames or len(images)
        detection_bench = evaluate_vehicle_detections(images[:n], labels[:n], yolo)
        gt_source = f"{args.segnet_dataset} mask pseudo-boxes, {n} images"
    print(f"YOLO vehicle detection benchmark ({gt_source}): {detection_bench}")

    with open(DETECTION_BENCHMARK_JSON, "w") as f:
        json.dump({"gt_source": gt_source, **detection_bench}, f, indent=2)
    print(f"Wrote detection benchmark -> '{DETECTION_BENCHMARK_JSON}'")

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
    save_odd_copula(copula, ODD_COPULA_PATH)
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
    stage7_features_csv = FEATURES_COMBINED_CSV if not args.skip_idd117k_features else FEATURES_CSV
    X, df_cleaned, y, feature_scaler, label_encoder = load_and_clean_features(stage7_features_csv)
    df_cleaned.to_csv(CLEANED_CSV, index=False)
    model, X_test, y_test, y_pred = train_odd_classifier(X, y, model_path=CLASSIFIER_PATH)
    plot_confusion_matrix(y_test, y_pred, label_encoder)
    plot_feature_importance(model, X.columns.tolist())
    joblib.dump(feature_scaler, SCALER_PATH)
    odd_accuracy = accuracy_score(y_test, y_pred)

    holdout_accuracy = None
    if not args.skip_holdout_eval:
        print("Building held-out validation feature table (real val splits, never used in any training step)...")
        val_images, val_labels = load_idd20k_split(IDD20K_DIR, "val", limit=args.holdout_limit or None)
        val_features_df = extract_all_features(val_images, val_labels, segnet, yolo, save_path=FEATURES_VAL_CSV)

        val_117k_pairs = sample_pairs(list_pairs(IDD117K_95K_DIR, "val"), args.holdout_limit or None)
        val_features_117k_df = extract_features_streaming(
            val_117k_pairs, segnet, yolo,
            save_path=FEATURES_IDD117K_VAL_CSV,
            gt_save_path=FEATURES_IDD117K_VAL_GT_CSV,
        )
        val_combined_df = pd.concat([val_features_df, val_features_117k_df], ignore_index=True)
        val_combined_df.to_csv(FEATURES_VAL_COMBINED_CSV, index=False)
        print(
            f"Held-out val corpus: {len(val_images)} IDD-20K-II + {len(val_117k_pairs)} "
            f"IDD117K = {len(val_combined_df)} rows -> '{FEATURES_VAL_COMBINED_CSV}'"
        )

        holdout_metrics, y_holdout, y_holdout_pred = evaluate_on_holdout(
            model, feature_scaler, label_encoder, X.columns.tolist(), FEATURES_VAL_COMBINED_CSV
        )
        plot_confusion_matrix(
            y_holdout, y_holdout_pred, label_encoder,
            filename="confusion_matrix_holdout.png",
            title="ODD Classifier Confusion Matrix (held-out val splits)",
        )
        with open(ODD_CLASSIFIER_HOLDOUT_EVAL_JSON, "w") as f:
            json.dump(holdout_metrics, f, indent=2)
        print(f"Wrote held-out evaluation -> '{ODD_CLASSIFIER_HOLDOUT_EVAL_JSON}'")
        holdout_accuracy = holdout_metrics["accuracy"]

    sae_level = classify_sae_level(THIS_PROJECT_DDT)
    print(f"This system's SAE level: {sae_level} ({LEVEL_NAMES[sae_level]})")

    reliabilities = compute_stream_reliabilities(features_df.iloc[0], refs)
    deep_gate = fit_deep_gate(fuse_dataframe(features_df, refs))
    joblib.dump(deep_gate, ECOFUSION_DEEP_GATE_PATH)
    branch_latencies = profile_branch_latencies(images[0], segnet, yolo)
    stem = compute_stem_features(images[0])
    decision = run_ecofusion_gate(stem, deep_gate, branch_latencies, lambda_e=0.1)
    print(f"EcoFusion sample decision (lambda_E=0.1): selected={sorted(decision.selected_config)}")

    print("Building Scenario-Based Feasibility Map...")
    feasibility_df = build_feasibility_map(features_df, feature_scaler)
    feasibility_df.to_csv(FEASIBILITY_MAP_CSV, index=False)
    export_pipeline_stats_js(feasibility_df, PIPELINE_STATS_JS)
    print(f"Wrote dashboard pipeline stats -> '{PIPELINE_STATS_JS}'")
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
    print(f"Total Stage 3-7 feature rows (combined corpus): {len(features_df)}")
    if final_val_loss is not None:
        print(f"SegNet final validation loss: {final_val_loss:.4f}")
    print(f"YOLO vehicle detection AP ({gt_source}): {detection_bench['ap']:.4f}")
    print(f"ODD classifier test accuracy (internal random split of train pool): {odd_accuracy:.4f}")
    if holdout_accuracy is not None:
        print(f"ODD classifier held-out accuracy (real val splits, never trained on): {holdout_accuracy:.4f}")
    print(f"P(worst 5% ODD-space scene): {risk_result.failure_probability:.4f}")
    print(f"System SAE level: {sae_level} ({LEVEL_NAMES[sae_level]})")
    print("Final combined mode distribution:")
    print(feasibility_df["final_mode"].value_counts())
    print(f"Fraction automation-feasible: {feasibility_df['automation_feasible'].mean():.3f}")


if __name__ == "__main__":
    main()
