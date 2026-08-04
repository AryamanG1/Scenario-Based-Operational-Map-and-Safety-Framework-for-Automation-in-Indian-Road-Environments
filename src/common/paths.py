"""Centralized, dependency-free filesystem path constants.

Every module in this project used to compute its own
`os.path.dirname(os.path.abspath(__file__))`-based project root and
re-join subpaths (data/, plots/, model checkpoints, ...) independently.
After moving the codebase into `src/<stage>/module.py` subpackages, that
per-file computation would silently break (each module now sits one
directory deeper than before). This module computes the project root
exactly once, robustly, regardless of which subpackage imports it, by
walking upward from this file's own location until it finds
`requirements.txt` (a file that only ever lives at the true project root).

Import from here instead of recomputing paths locally:
    from src.common.paths import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR
"""

import os


def _find_project_root(marker: str = "requirements.txt") -> str:
    """Walks upward from this file until a directory containing `marker` is found.

    Args:
        marker: A filename known to exist only at the project root.

    Returns:
        The absolute path to the project root.

    Raises:
        RuntimeError: If no ancestor directory contains `marker`.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.isfile(os.path.join(path, marker)):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            raise RuntimeError(
                f"Could not locate project root: no ancestor of {__file__} contains '{marker}'."
            )
        path = parent


PROJECT_ROOT = _find_project_root()

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "idd20k_lite")
DATA_ARCHIVE = os.path.join(PROJECT_ROOT, "data", "idd-lite.tar.gz")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SEGNET_CHECKPOINT = os.path.join(MODELS_DIR, "refined_segnet.pth")
YOLO_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n.pt")
ODD_CLASSIFIER_PATH = os.path.join(MODELS_DIR, "odd_classifier.pkl")
FEATURE_SCALER_PATH = os.path.join(MODELS_DIR, "feature_scaler.pkl")
ODD_COPULA_PATH = os.path.join(MODELS_DIR, "odd_copula.pkl")
ECOFUSION_DEEP_GATE_PATH = os.path.join(MODELS_DIR, "ecofusion_deep_gate.pkl")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
FEATURES_CSV = os.path.join(OUTPUTS_DIR, "final_features.csv")
FEATURES_CLEANED_CSV = os.path.join(OUTPUTS_DIR, "final_features_cleaned.csv")
FULL_DATASET_LABELED_CSV = os.path.join(OUTPUTS_DIR, "full_dataset_labeled.csv")
FULL_DATASET_CHECKPOINT_CSV = os.path.join(OUTPUTS_DIR, "full_dataset_checkpoint.csv")
FEASIBILITY_MAP_CSV = os.path.join(OUTPUTS_DIR, "feasibility_map.csv")
TRAFFIC_DENSITY_THRESHOLDS_JSON = os.path.join(OUTPUTS_DIR, "traffic_density_thresholds.json")
FUZZY_PD_BREAKPOINTS_JSON = os.path.join(OUTPUTS_DIR, "fuzzy_pd_breakpoints.json")

DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")
PIPELINE_STATS_JS = os.path.join(DASHBOARD_DIR, "pipeline_stats.js")
DASHBOARD_CARLA_LIVE_JS = os.path.join(DASHBOARD_DIR, "carla_live.js")

CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
CARLA_CONFIG_JSON = os.path.join(CONFIGS_DIR, "carla_config.json")


def ensure_output_dirs() -> None:
    """Creates MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, and CONFIGS_DIR if they don't already exist."""
    for directory in (MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, CONFIGS_DIR):
        os.makedirs(directory, exist_ok=True)
