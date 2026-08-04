"""Cross-cutting: Scenario-Based Feasibility Map.

Ties every stage's outputs together into one per-scene record, matching
the capstone proposal's explicit deliverable: "Classify road segments as
automation-safe or unsafe and construct a feasibility map" (Work Plan
item 9 / pipeline-diagram stage 7, "Scenario-Based Mapping & Automation
Feasibility").

Per scene, this combines: Stage 3's traffic-density category
(traffic_density.py), Stage 4's fuzzy micro-ODD + composite scenario label
(fuzzy_odd.py), Stage 5's ODD-region status (odd_boundary.py), and Stage
7's final Normal/Degraded/Takeover decision (odd_classifier.py) into a
single FeasibilityRecord, plus a 0-100 odd_score and an L1/L2/L3
automation-feasibility band mirroring dashboard/app.js's
calculateODDScore() -- but computed here from this project's real,
MinMax-scaled pipeline features rather than the dashboard's synthetic demo
data.

Note on the L1/L2/L3 labels: these describe a ROAD SEGMENT's current
automation-feasibility ceiling (a scenario-classification concept), which
is a distinct idea from this project's own fixed SAE Level 2 system
commitment (see sae_taxonomy.py) -- a segment can be labeled "L3-feasible"
(favorable conditions) while the deployed system itself always operates
at Level 2 per the proposal.

Stage 6 (perception_monitor.py) is deliberately NOT run per-row here: it
requires drawing several perturbed pseudo-frames and running SegNet+YOLO
on each, which is too expensive to precompute for an entire dataset in
this batch map. Batch records default to a "Nominal" monitoring
assumption; combine per-frame monitoring on demand via
perception_monitor.py + odd_classifier.combine_stage_outputs() for a
single scene's true final decision.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.odd.fuzzy_odd import classify_dataframe as classify_scenario
from src.odd.odd_boundary import DEFAULT_ODD_VARIABLES, classify_odd_region, fit_odd_copula, odd_density
from src.odd.odd_classifier import FeatureScaler, assign_mode, combine_stage_outputs
from src.odd.traffic_density import classify_dataframe as classify_traffic


@dataclass
class FeasibilityRecord:
    """One scene's complete cross-stage feasibility summary.

    Attributes:
        traffic_density_level: Stage 3 output.
        scenario_label: Stage 4 composite scenario label.
        mu_odd_label: Stage 4 fuzzy micro-ODD label.
        odd_region: Stage 5 within/near/outside status.
        final_mode: Stage 7 combined Normal/Degraded/Takeover decision.
        odd_score: A 0-100 automation-readiness score.
        automation_level: "L1"/"L2"/"L3" road-segment feasibility band.
        automation_feasible: True iff final_mode != "Takeover".
    """

    traffic_density_level: str
    scenario_label: str
    mu_odd_label: str
    odd_region: str
    final_mode: str
    odd_score: float
    automation_level: str
    automation_feasible: bool


def compute_odd_score(scaled_row: pd.Series) -> float:
    """Computes a 0-100 automation-readiness score from MinMax-scaled features.

    Mirrors dashboard/app.js's calculateODDScore() formula, applied here to
    this project's real pipeline features (post odd_classifier.FeatureScaler)
    rather than the dashboard's synthetic demo data.

    Args:
        scaled_row: A row of MinMax-scaled (0-1) features, including at
            least 'visibility', 'brightness', 'traffic_density', 'wetness',
            'non_drivable_area_score', 'detection_confidence'.

    Returns:
        A score in [0.0, 100.0].
    """
    score = 100.0
    score -= (1 - scaled_row["visibility"]) * 30
    score -= (1 - scaled_row["brightness"]) * 10
    score -= scaled_row["traffic_density"] * 25
    score -= scaled_row["wetness"] * 15
    score -= scaled_row["non_drivable_area_score"] * 10
    score -= (1 - scaled_row["detection_confidence"]) * 10
    return max(0.0, min(100.0, score))


def automation_level_from_score(score: float) -> str:
    """Maps a 0-100 odd_score to an L1/L2/L3 road-segment feasibility band.

    Args:
        score: Output of compute_odd_score().

    Returns:
        "L3" (score >= 70), "L2" (40-69), or "L1" (< 40).
    """
    if score >= 70:
        return "L3"
    if score >= 40:
        return "L2"
    return "L1"


def build_feasibility_map(
    features_df: pd.DataFrame,
    feature_scaler: FeatureScaler,
    odd_variables=DEFAULT_ODD_VARIABLES,
    default_monitoring_state: str = "Nominal",
) -> pd.DataFrame:
    """Builds the full Scenario-Based Feasibility Map for a features dataset.

    Args:
        features_df: A features DataFrame (final_features.csv or
            full_dataset_labeled.csv-shaped).
        feature_scaler: A fitted FeatureScaler (odd_classifier.py), used to
            produce the scaled features assign_mode()/compute_odd_score()
            require.
        odd_variables: Which columns to fit the Stage 5 ODD copula over.
        default_monitoring_state: The Stage 6 status assumed for every row
            (see module docstring for why Stage 6 isn't run per-row here).

    Returns:
        A copy of features_df with one FeasibilityRecord's fields appended
        as new columns.
    """
    traffic_df, _ = classify_traffic(features_df)
    scenario_df, _, _ = classify_scenario(traffic_df)

    copula = fit_odd_copula(features_df, odd_variables)

    scaled_df = feature_scaler.transform(features_df)

    records = []
    for i in range(len(features_df)):
        scaled_row = scaled_df.iloc[i]
        base_mode = assign_mode(scaled_row)

        odd_row = {v: features_df.iloc[i][v] for v in odd_variables}
        odd_region = classify_odd_region(copula, odd_row)

        final_mode = combine_stage_outputs(base_mode, odd_region, default_monitoring_state)
        score = compute_odd_score(scaled_row)

        records.append(
            FeasibilityRecord(
                traffic_density_level=scenario_df.iloc[i]["traffic_density_level"],
                scenario_label=scenario_df.iloc[i]["scenario_label"],
                mu_odd_label=scenario_df.iloc[i]["mu_odd_label"],
                odd_region=odd_region,
                final_mode=final_mode,
                odd_score=score,
                automation_level=automation_level_from_score(score),
                automation_feasible=(final_mode != "Takeover"),
            )
        )

    result_df = features_df.copy()
    result_df["traffic_density_level"] = [r.traffic_density_level for r in records]
    result_df["scenario_label"] = [r.scenario_label for r in records]
    result_df["mu_odd_label"] = [r.mu_odd_label for r in records]
    result_df["odd_region"] = [r.odd_region for r in records]
    result_df["final_mode"] = [r.final_mode for r in records]
    result_df["odd_score"] = [r.odd_score for r in records]
    result_df["automation_level"] = [r.automation_level for r in records]
    result_df["automation_feasible"] = [r.automation_feasible for r in records]
    return result_df


def export_pipeline_stats_js(feasibility_df: pd.DataFrame, output_path: str) -> None:
    """Exports aggregate feasibility-map stats as a JS data file for the dashboard.

    Writes `const PIPELINE_STATS = {...};` rather than a .json file: the
    dashboard (dashboard/index.html) is designed to open directly over
    file:// with zero local server (see dashboard/app.js's module
    docstring), and fetch()-ing a local JSON file is blocked by browsers
    under file://, whereas a plain <script src="..."> tag is not. This
    mirrors how dashboard/app.js itself inlines its demo data.

    Args:
        feasibility_df: Output of build_feasibility_map().
        output_path: Destination .js file path (e.g.
            dashboard/pipeline_stats.js).
    """
    stats = {
        "num_scenes": int(len(feasibility_df)),
        "final_mode_counts": feasibility_df["final_mode"].value_counts().to_dict(),
        "automation_level_counts": feasibility_df["automation_level"].value_counts().to_dict(),
        "odd_region_counts": feasibility_df["odd_region"].value_counts().to_dict(),
        "mu_odd_counts": feasibility_df["mu_odd_label"].value_counts().to_dict(),
        "traffic_density_counts": feasibility_df["traffic_density_level"].value_counts().to_dict(),
        "mean_odd_score": float(feasibility_df["odd_score"].mean()),
        "fraction_automation_feasible": float(feasibility_df["automation_feasible"].mean()),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"// Auto-generated by feasibility_map.py -- real IDD-Lite pipeline results.\n")
        f.write(f"const PIPELINE_STATS = {json.dumps(stats, indent=2)};\n")


if __name__ == "__main__":
    import joblib

    from src.common.paths import (
        FEASIBILITY_MAP_CSV,
        FEATURES_CSV,
        FEATURE_SCALER_PATH,
        PIPELINE_STATS_JS,
        ensure_output_dirs,
    )

    ensure_output_dirs()
    features_path = FEATURES_CSV
    scaler_path = FEATURE_SCALER_PATH
    output_path = FEASIBILITY_MAP_CSV
    stats_js_path = PIPELINE_STATS_JS

    features_df = pd.read_csv(features_path)
    scaler = joblib.load(scaler_path)

    feasibility_df = build_feasibility_map(features_df, scaler)
    feasibility_df.to_csv(output_path, index=False)
    print(f"Saved feasibility map -> '{output_path}'")

    export_pipeline_stats_js(feasibility_df, stats_js_path)
    print(f"Saved dashboard pipeline stats -> '{stats_js_path}'")

    print("\nAutomation level distribution:")
    print(feasibility_df["automation_level"].value_counts())
    print("\nFinal mode distribution:")
    print(feasibility_df["final_mode"].value_counts())
    print(f"\nFraction automation-feasible: {feasibility_df['automation_feasible'].mean():.3f}")
