"""End-to-end tests for closed_loop_runner.py, using the local kinematic fallback.

Runs with force_fallback=True throughout -- this exercises the exact same
path that is guaranteed to run in this GPU-less, CARLA-less sandbox (see
test_carla_bridge.py), against the REAL trained SegNet/YOLO/FeatureScaler
artifacts, matching this test suite's existing skip-on-missing-artifact
convention (tests/decision/test_feasibility_map.py::
test_build_feasibility_map_end_to_end is the direct template).
"""

import os

import joblib
import pytest

from src.common.paths import FEATURE_SCALER_PATH
from src.simulation.carla_config import CarlaConfig
from src.simulation.closed_loop_runner import run_closed_loop_simulation, write_carla_live_js

_EXPECTED_COLUMNS = {
    "tick",
    "t",
    "base_mode",
    "mode",
    "odd_region",
    "monitoring_state",
    "mu_odd_label",
    "ads_state",
    "v_ego",
    "v_front",
    "rss_distance",
    "sideslip_stable",
    "stability_margin",
    "position_x",
    "position_y",
    "progress",
    "used_carla",
}


@pytest.fixture(scope="module")
def feature_scaler():
    if not os.path.isfile(FEATURE_SCALER_PATH):
        pytest.skip("models/feature_scaler.pkl not found -- run odd_classifier.py first.")
    return joblib.load(FEATURE_SCALER_PATH)


@pytest.fixture(scope="module")
def small_run_df(segnet, yolo, feature_scaler):
    config = CarlaConfig(use_carla=False, tick_dt=0.1)
    return run_closed_loop_simulation(
        num_ticks=5,
        config=config,
        segnet_model=segnet,
        yolo_model=yolo,
        feature_scaler=feature_scaler,
        force_fallback=True,
    )


def test_returns_one_row_per_tick(small_run_df):
    assert len(small_run_df) == 5


def test_has_expected_columns(small_run_df):
    assert _EXPECTED_COLUMNS.issubset(set(small_run_df.columns))


def test_used_carla_is_false_under_force_fallback(small_run_df):
    assert (small_run_df["used_carla"] == False).all()  # noqa: E712


def test_modes_are_valid(small_run_df):
    assert small_run_df["mode"].isin(["Normal", "Degraded", "Takeover"]).all()
    assert small_run_df["base_mode"].isin(["Normal", "Degraded", "Takeover"]).all()


def test_odd_regions_are_valid(small_run_df):
    assert small_run_df["odd_region"].isin(["within", "near", "outside"]).all()


def test_ads_states_are_valid(small_run_df):
    from src.decision.sae_taxonomy import ADSState

    valid_state_values = {s.value for s in ADSState}
    assert small_run_df["ads_state"].isin(valid_state_values).all()


def test_rss_distances_are_non_negative(small_run_df):
    assert (small_run_df["rss_distance"] >= 0).all()


def test_tick_indices_are_sequential(small_run_df):
    assert list(small_run_df["tick"]) == [0, 1, 2, 3, 4]


def test_progress_is_bounded(small_run_df):
    assert small_run_df["progress"].between(0.0, 1.0).all()


def test_raises_without_features_csv_or_explicit_copula_and_breakpoints(segnet, yolo, feature_scaler, monkeypatch):
    import src.simulation.closed_loop_runner as runner_module

    monkeypatch.setattr(runner_module, "FEATURES_CSV", "/nonexistent/path/final_features.csv")
    with pytest.raises(FileNotFoundError):
        run_closed_loop_simulation(
            num_ticks=1,
            config=CarlaConfig(),
            segnet_model=segnet,
            yolo_model=yolo,
            feature_scaler=feature_scaler,
            force_fallback=True,
        )


def test_write_carla_live_js_produces_a_loadable_js_global(small_run_df, tmp_path):
    output_path = str(tmp_path / "carla_live.js")
    write_carla_live_js(small_run_df, road_name="Jan Marg", used_carla=False, output_path=output_path)

    assert os.path.isfile(output_path)
    content = open(output_path, encoding="utf-8").read()
    assert content.startswith("//")
    assert "const CARLA_LIVE = " in content
    assert content.strip().endswith(";")

    json_text = content.split("const CARLA_LIVE = ", 1)[1].rstrip(";\n")
    import json

    data = json.loads(json_text)
    assert data["road_name"] == "Jan Marg"
    assert data["used_carla"] is False
    assert len(data["ticks"]) == 5
    assert set(data["ticks"][0].keys()) == {
        "tick", "t", "mode", "ads_state", "rss_distance", "stability_margin", "v_ego", "progress",
    }
