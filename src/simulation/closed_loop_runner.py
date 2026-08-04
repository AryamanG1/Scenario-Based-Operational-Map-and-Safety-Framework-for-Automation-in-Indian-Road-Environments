"""Stage 8 (cross-cutting, optional): Closed-Loop Simulation orchestrator.

Runs this project's full Stage 2/4/5/7 per-tick decision flow against a
sequence of frames -- either real CARLA camera frames (CarlaTickSource) or
real IDD-Lite frames driven by a local kinematic model
(LocalKinematicTickSource) -- and logs the result. This is the concrete
answer to "the CARLA portion": a real closed-loop integration, with a
fully-functional, fully-tested fallback for when CARLA isn't attached.

Per-tick flow (see docs/modules/simulation/closed_loop_runner.md for the
full rationale of each step):

    tick_source.tick() -> frame, ego/lead kinematics
      -> SegNet mask + YOLO detections -> feature_extraction.compute_features()
      -> feature_scaler.transform() -> odd_classifier.assign_mode() [base_mode]
      -> odd_boundary.classify_odd_region() [odd_region]
      -> (every monitor_check_interval_ticks) perception_monitor.run_perception_monitor()
      -> odd_classifier.combine_stage_outputs() [final_mode]
      -> fuzzy_odd.classify_mu_odd()/defuzzify_mu_odd() [mu_odd_label]
      -> rss_safety.rss_parameters_for_mu_odd() -> rss_minimum_safe_distance()
      -> rss_safety sideslip stability (SideslipState -> stability_margin)
      -> sae_taxonomy.ADSStateMachine.step()
      -> tick_source.apply_mode(final_mode)

Every formula here is imported, not reimplemented, from the modules that
already own it (rss_safety.py, fuzzy_odd.py, odd_boundary.py,
odd_classifier.py, sae_taxonomy.py) -- this module is pure orchestration.
"""

import argparse
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.common.paths import (
    CARLA_CONFIG_JSON,
    DASHBOARD_CARLA_LIVE_JS,
    DATA_DIR,
    FEATURE_SCALER_PATH,
    FEATURES_CSV,
    SEGNET_CHECKPOINT,
    YOLO_WEIGHTS,
    ensure_output_dirs,
)
from src.decision.sae_taxonomy import ADSStateMachine
from src.odd.fuzzy_odd import PDBreakpoints, calibrate_pd_breakpoints, classify_mu_odd, defuzzify_mu_odd
from src.odd.odd_boundary import DEFAULT_ODD_VARIABLES, ODDCopulaModel, classify_odd_region, fit_odd_copula
from src.odd.odd_classifier import FeatureScaler, assign_mode, combine_stage_outputs
from src.odd.rss_safety import (
    SideslipState,
    compute_stability_boundary,
    is_sideslip_stable,
    rss_minimum_safe_distance,
    rss_parameters_for_mu_odd,
    sideslip_dynamics,
    stability_margin,
)
from src.perception.data_pipeline import load_and_clean_dataset
from src.perception.feature_extraction import DEVICE, compute_features, run_detection
from src.simulation.carla_bridge import CarlaTickSource, is_carla_available
from src.simulation.carla_config import CarlaConfig, load_carla_config
from src.simulation.local_kinematic_sim import LocalKinematicTickSource
from src.simulation.tick_source import TickSource

# Generic mid-size-sedan physical constants for the sideslip model, matching
# rss_safety.py's own __main__ demo values -- neither tick source measures
# these directly (CARLA doesn't expose yaw inertia/axle geometry through the
# basic Actor API used here, and IDD-Lite obviously can't), so they are
# fixed, documented assumptions, not per-tick fabricated precision.
_VEHICLE_MASS_KG = 1500.0
_VEHICLE_IZ_KGM2 = 2500.0
_VEHICLE_L_F_M = 1.2
_VEHICLE_L_R_M = 1.6
_ROAD_MU = 0.8
_TIRE_F_ZF_N = 7000.0
_TIRE_F_ZR_N = 6500.0
_MIN_VX_FOR_SIDESLIP = 0.1  # sideslip_dynamics() is undefined at v_x=0; a near-stationary


def _run_perception(frame: np.ndarray, segnet_model, yolo_model, device: torch.device = DEVICE):
    """Runs SegNet + YOLO on one frame, matching odd_classifier.evaluate_road_scene()'s pattern.

    Args:
        frame: BGR image array, IMAGE_SIZE-shaped.
        segnet_model: Trained SegNet (eval mode).
        yolo_model: Loaded Ultralytics YOLO model.
        device: Torch device to run SegNet inference on.

    Returns:
        A tuple (mask, detections).
    """
    img_t = torch.tensor(frame).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    segnet_model.eval()
    with torch.no_grad():
        output = segnet_model(img_t)
    mask = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    detections = run_detection(frame, yolo_model)
    return mask, detections


def run_closed_loop_simulation(
    num_ticks: int,
    config: CarlaConfig,
    segnet_model,
    yolo_model,
    feature_scaler: FeatureScaler,
    copula_model: Optional[ODDCopulaModel] = None,
    pd_breakpoints: Optional[PDBreakpoints] = None,
    force_fallback: bool = False,
) -> pd.DataFrame:
    """Runs `num_ticks` of the closed-loop simulation and returns a per-tick log.

    Args:
        num_ticks: How many simulation ticks to run.
        config: Full simulation configuration (see carla_config.py).
        segnet_model: Trained SegNet (as returned by segnet_model.load_segnet).
        yolo_model: Loaded Ultralytics YOLO model.
        feature_scaler: A fitted FeatureScaler (models/feature_scaler.pkl).
        copula_model: A fitted ODDCopulaModel for Stage 5's classify_odd_region().
            If None, fit once from outputs/final_features.csv.
        pd_breakpoints: Calibrated PDBreakpoints for Stage 4's fuzzy uODD
            classification. If None, calibrated once from
            outputs/final_features.csv's 'wetness' column.
        force_fallback: If True, always use LocalKinematicTickSource, even
            if config.use_carla is True and a CARLA server is reachable --
            the literal "run without CARLA even if it's attached" override.

    Returns:
        A DataFrame with one row per tick: tick, t, mode, base_mode,
        odd_region, monitoring_state, ads_state, mu_odd_label, v_ego,
        v_front, rss_distance, sideslip_stable, stability_margin,
        position_x, position_y, progress, used_carla.

    Raises:
        FileNotFoundError: If copula_model/pd_breakpoints are both None and
            outputs/final_features.csv does not exist (run
            feature_extraction.py or main.py at least once first).
    """
    if copula_model is None or pd_breakpoints is None:
        if not os.path.isfile(FEATURES_CSV):
            raise FileNotFoundError(
                f"'{FEATURES_CSV}' not found -- run feature_extraction.py (or main.py) at least "
                "once first, or pass copula_model/pd_breakpoints explicitly."
            )
        features_df = pd.read_csv(FEATURES_CSV)
        if pd_breakpoints is None:
            pd_breakpoints = calibrate_pd_breakpoints(features_df["wetness"].tolist())
        if copula_model is None:
            copula_model = fit_odd_copula(features_df, DEFAULT_ODD_VARIABLES)

    def _new_fallback_source() -> LocalKinematicTickSource:
        images, _ = load_and_clean_dataset(DATA_DIR)
        return LocalKinematicTickSource(config, images)

    used_carla = config.use_carla and not force_fallback and is_carla_available(config)
    tick_source: TickSource
    if used_carla:
        try:
            tick_source = CarlaTickSource(config)
        except Exception as exc:
            print(f"Warning: failed to start CarlaTickSource ({exc}); using the local kinematic simulator instead.")
            tick_source = _new_fallback_source()
            used_carla = False
    else:
        if config.use_carla and not force_fallback:
            print(
                "Warning: CARLA is not available here (package not installed, or no server "
                f"reachable at {config.host}:{config.port}); using the local kinematic simulator instead."
            )
        tick_source = _new_fallback_source()

    ads_machine = ADSStateMachine()
    records = []
    final_mode = "Normal"

    for _ in range(num_ticks):
        try:
            tick = tick_source.tick()
        except Exception as exc:
            if used_carla and config.fallback_on_error:
                print(f"Warning: CARLA tick failed ({exc}); switching to the local kinematic simulator for the rest of this run.")
                if hasattr(tick_source, "close"):
                    try:
                        tick_source.close()
                    except Exception:
                        pass
                tick_source = _new_fallback_source()
                tick_source.apply_mode(final_mode)
                used_carla = False
                tick = tick_source.tick()
            else:
                raise

        mask, detections = _run_perception(tick.frame, segnet_model, yolo_model)
        features = compute_features(tick.frame, mask, detections)

        scaled_row = feature_scaler.transform(pd.DataFrame([features])).iloc[0]
        base_mode = assign_mode(scaled_row)

        odd_row = {v: features[v] for v in DEFAULT_ODD_VARIABLES}
        odd_region = classify_odd_region(copula_model, odd_row)

        monitoring_state = "Nominal"
        if config.monitor_check_interval_ticks > 0 and tick.tick % config.monitor_check_interval_ticks == 0:
            from src.monitoring.perception_monitor import run_perception_monitor

            monitor_result = run_perception_monitor(tick.frame, segnet_model, yolo_model)
            monitoring_state = monitor_result.state

        final_mode = combine_stage_outputs(base_mode, odd_region, monitoring_state)

        memberships = classify_mu_odd(features["wetness"], pd_breakpoints)
        mu_odd_label = defuzzify_mu_odd(memberships)
        rss_params = rss_parameters_for_mu_odd(mu_odd_label)
        rss_distance = rss_minimum_safe_distance(tick.v_ego, tick.v_front, rss_params)

        # Per-axle tire slip angles aren't separately modeled by either tick
        # source (no steering-angle/yaw-geometry plumbing) -- both are
        # approximated as the vehicle's overall sideslip angle. See
        # docs/FUTURE_STEPS.md for a real per-axle model as an extension.
        sideslip_state = SideslipState(
            beta=tick.slip_angle,
            omega_r=tick.yaw_rate,
            v_x=max(tick.v_ego, _MIN_VX_FOR_SIDESLIP),
            mass=_VEHICLE_MASS_KG,
            i_z=_VEHICLE_IZ_KGM2,
            l_f=_VEHICLE_L_F_M,
            l_r=_VEHICLE_L_R_M,
            delta=0.0,
            mu=_ROAD_MU,
            f_zf=_TIRE_F_ZF_N,
            f_zr=_TIRE_F_ZR_N,
            alpha_f=tick.slip_angle,
            alpha_r=tick.slip_angle,
        )
        beta_dot, _omega_r_dot = sideslip_dynamics(sideslip_state)
        boundary = compute_stability_boundary(sideslip_state.v_x, sideslip_state.mu, sideslip_state.delta)
        sideslip_stable = is_sideslip_stable(sideslip_state.beta, beta_dot, boundary)
        margin = stability_margin(sideslip_state.beta, beta_dot, boundary)

        within_odd = final_mode != "Takeover"
        ads_state = ads_machine.step(within_odd=within_odd, driver_responsive=True)

        tick_source.apply_mode(final_mode)

        records.append(
            {
                "tick": tick.tick,
                "t": tick.tick * config.tick_dt,
                "base_mode": base_mode,
                "mode": final_mode,
                "odd_region": odd_region,
                "monitoring_state": monitoring_state,
                "mu_odd_label": mu_odd_label,
                "ads_state": ads_state.value,
                "v_ego": tick.v_ego,
                "v_front": tick.v_front,
                "rss_distance": rss_distance,
                "sideslip_stable": sideslip_stable,
                "stability_margin": margin,
                "position_x": tick.position[0],
                "position_y": tick.position[1],
                "progress": tick.progress,
                "used_carla": used_carla,
            }
        )

    if hasattr(tick_source, "close"):
        tick_source.close()

    return pd.DataFrame(records)


def write_carla_live_js(
    result_df: pd.DataFrame,
    road_name: str,
    used_carla: bool,
    output_path: str = DASHBOARD_CARLA_LIVE_JS,
) -> None:
    """Writes one full closed-loop run as a dashboard replay data file.

    Writes `const CARLA_LIVE = {...};` rather than a .json file, for the
    same file:// reason feasibility_map.export_pipeline_stats_js() does:
    the dashboard opens directly over file:// with no local server, and
    fetch()-ing a local file is blocked there, whereas a plain
    <script src="..."> tag is not.

    Args:
        result_df: Output of run_closed_loop_simulation().
        road_name: Which dashboard/app.js ROADS entry to animate the
            replay marker along.
        used_carla: Whether this run actually used a real CARLA server
            (result_df's own "used_carla" column may vary per-row if a
            mid-run fallback occurred; this top-level flag reflects
            whether CARLA was used for the majority of the run).
        output_path: Destination .js file path.
    """
    ticks = [
        {
            "tick": int(row.tick),
            "t": float(row.t),
            "mode": row.mode,
            "ads_state": row.ads_state,
            "rss_distance": float(row.rss_distance),
            "stability_margin": float(row.stability_margin),
            "v_ego": float(row.v_ego),
            "progress": float(row.progress),
        }
        for row in result_df.itertuples()
    ]
    data = {"used_carla": bool(used_carla), "road_name": road_name, "ticks": ticks}
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by closed_loop_runner.py -- one full closed-loop simulation run.\n")
        f.write(f"const CARLA_LIVE = {json.dumps(data, indent=2)};\n")


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the standalone closed-loop runner.

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run the closed-loop simulation (Stage 8): real CARLA if attached and reachable, else a local kinematic fallback."
    )
    parser.add_argument("--num_ticks", type=int, default=50, help="Number of simulation ticks to run.")
    parser.add_argument("--carla_config", type=str, default=CARLA_CONFIG_JSON, help="Path to a carla_config.json (defaults are used if absent).")
    parser.add_argument(
        "--force_fallback",
        action="store_true",
        help="Always use the local kinematic simulator, even if CARLA is attached and reachable.",
    )
    parser.add_argument(
        "--road_name",
        type=str,
        default="Jan Marg",
        help="Which dashboard/app.js ROADS entry to animate the replay marker along.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    import joblib
    from ultralytics import YOLO

    from src.perception.segnet_model import load_segnet

    ensure_output_dirs()
    args = _parse_args()
    config = load_carla_config(args.carla_config)

    segnet = load_segnet(SEGNET_CHECKPOINT)
    yolo = YOLO(YOLO_WEIGHTS)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)

    result_df = run_closed_loop_simulation(
        num_ticks=args.num_ticks,
        config=config,
        segnet_model=segnet,
        yolo_model=yolo,
        feature_scaler=feature_scaler,
        force_fallback=args.force_fallback,
    )

    print(result_df[["tick", "mode", "ads_state", "rss_distance", "stability_margin", "v_ego"]].to_string(index=False))
    print(f"\nUsed CARLA: {bool(result_df['used_carla'].iloc[0])}")
    print("\nFinal mode distribution:")
    print(result_df["mode"].value_counts())

    write_carla_live_js(
        result_df,
        road_name=args.road_name,
        used_carla=bool(result_df["used_carla"].mean() > 0.5),
        output_path=DASHBOARD_CARLA_LIVE_JS,
    )
    print(f"\nWrote dashboard replay data -> '{DASHBOARD_CARLA_LIVE_JS}'")
