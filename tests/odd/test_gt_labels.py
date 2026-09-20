"""Tests for the annotation-derived (ground-truth) ODD classifier labels."""

import os

import numpy as np
import pandas as pd
import pytest

from src.odd.odd_classifier import (
    GT_COUNT_COLUMNS,
    GTLabelThresholds,
    assign_gt_mode,
    evaluate_on_holdout,
    fit_gt_label_thresholds,
    gt_scene_complexity,
    load_and_clean_features,
    train_odd_classifier,
)


def _make_tables(tmp_path, n_unannotated=20, n_annotated=300, seed=0):
    """A feature CSV plus an aligned GT sidecar with NaN rows up front."""
    rng = np.random.default_rng(seed)
    n = n_unannotated + n_annotated
    gt = pd.DataFrame(np.nan, index=range(n), columns=GT_COUNT_COLUMNS)
    counts = rng.poisson(lam=[4, 3, 1, 2, 0.2], size=(n_annotated, 5))
    gt.iloc[n_unannotated:] = counts

    # Perception features loosely correlated with the GT counts, plus noise:
    # the classifier has something to learn but cannot be perfect.
    total = np.zeros(n)
    total[n_unannotated:] = counts.sum(axis=1)
    feats = pd.DataFrame(
        {
            "vehicle_count": np.clip(total * 0.3 + rng.normal(0, 1.5, n), 0, None).round(),
            "num_two_wheelers": np.clip(total * 0.2 + rng.normal(0, 1.5, n), 0, None).round(),
            "num_pedestrians": np.clip(total * 0.1 + rng.normal(0, 1.0, n), 0, None).round(),
            "num_animals": np.zeros(n),
            "num_autorickshaws": np.clip(rng.normal(0.3, 0.5, n), 0, None).round(),
            "object_presence": (total > 0).astype(int),
            "object_distance": rng.uniform(5, 50, n),
            "lead_vehicle_distance": rng.uniform(0, 30, n),
            "traffic_density": np.clip(total * 0.5 + rng.normal(0, 2, n), 0, None).round(),
            "detection_confidence": rng.uniform(0.3, 0.9, n),
            "drivable_area": rng.uniform(0.1, 0.6, n),
            "road_end_distance": rng.uniform(50, 200, n),
            "non_drivable_area_score": rng.uniform(0, 0.3, n),
            "non_drivable_area_distance": rng.uniform(0, 200, n),
            "living_things_score": rng.uniform(0, 0.05, n),
            "brightness": rng.uniform(60, 200, n),
            "visibility": rng.uniform(100, 3000, n),
            "wetness": rng.uniform(20, 80, n),
            "road_quality": rng.uniform(100, 3000, n),
            "pothole_heuristic_score": rng.uniform(0, 0.05, n),
            "pothole_heuristic_count": rng.integers(0, 5, n),
            "scene_complexity": rng.uniform(0, 1, n),
        }
    )
    f_path = os.path.join(tmp_path, "features.csv")
    g_path = os.path.join(tmp_path, "features_gt.csv")
    feats.to_csv(f_path, index=False)
    gt.to_csv(g_path, index=False)
    return f_path, g_path, gt


def test_thresholds_roundtrip_and_tertile_balance(tmp_path):
    _, _, gt = _make_tables(tmp_path)
    thr = fit_gt_label_thresholds(gt)
    assert thr.low <= thr.high
    path = os.path.join(tmp_path, "thr.json")
    thr.save(path)
    assert GTLabelThresholds.load(path) == thr

    modes = assign_gt_mode(gt, thr)
    assert modes.isna().sum() == 20  # unannotated rows carry no label
    counts = modes.value_counts()
    assert set(counts.index) == {"Normal", "Degraded", "Takeover"}
    # Tertiles on integer counts are only approximately balanced.
    assert counts.min() > 0.15 * counts.sum()


def test_gt_scene_complexity_is_nan_without_annotations():
    gt = pd.DataFrame({c: [1.0, np.nan] for c in GT_COUNT_COLUMNS})
    s = gt_scene_complexity(gt)
    assert s.iloc[0] == 5.0 and np.isnan(s.iloc[1])


def test_gt_labels_exclude_unannotated_rows_and_are_not_trivially_learnable(tmp_path):
    f_path, g_path, _ = _make_tables(tmp_path)
    X, df_cleaned, y, scaler, le = load_and_clean_features(f_path, gt_csv_path=g_path)

    assert len(X) == 300 and len(y) == 300  # the 20 unannotated rows are dropped
    assert set(df_cleaned["mode"]) <= {"Normal", "Degraded", "Takeover"}
    # The scaler was fitted on the FULL table so later stages can use it.
    assert scaler.transform(pd.read_csv(f_path)).shape[0] == 320

    model, X_test, y_test, y_pred = train_odd_classifier(X, y, model_path=os.path.join(tmp_path, "m.pkl"))
    acc = (y_pred == np.asarray(y_test)).mean()
    # Learnable (above chance) but not the ~100% the rule labels gave.
    assert 0.4 < acc < 0.999


def test_holdout_requires_training_thresholds_and_reports_baseline(tmp_path):
    f_path, g_path, gt = _make_tables(tmp_path)
    thr = fit_gt_label_thresholds(gt)
    X, _, y, scaler, le = load_and_clean_features(f_path, gt_csv_path=g_path, gt_thresholds=thr)
    model, *_ = train_odd_classifier(X, y, model_path=os.path.join(tmp_path, "m.pkl"))

    v_path, vg_path, _ = _make_tables(os.path.join(tmp_path), n_unannotated=5, n_annotated=100, seed=1)
    with pytest.raises(ValueError):
        evaluate_on_holdout(model, scaler, le, X.columns.tolist(), v_path, gt_csv_path=vg_path)

    metrics, y_true, y_pred = evaluate_on_holdout(
        model, scaler, le, X.columns.tolist(), v_path, gt_csv_path=vg_path, gt_thresholds=thr
    )
    assert metrics["label_source"] == "gt"
    assert metrics["num_holdout_rows"] == 100 and metrics["num_rows_in_csv"] == 105
    assert 0 < metrics["majority_class_baseline_accuracy"] < 1
    assert len(y_true) == len(y_pred) == 100


def test_misaligned_sidecar_is_rejected(tmp_path):
    f_path, g_path, gt = _make_tables(tmp_path)
    gt.iloc[:-1].to_csv(g_path, index=False)
    with pytest.raises(ValueError, match="aligned"):
        load_and_clean_features(f_path, gt_csv_path=g_path)


def test_calibrated_mode_thresholds_roundtrip_and_produce_all_modes(tmp_path):
    from src.odd.odd_classifier import (
        DEFAULT_MODE_THRESHOLDS,
        ModeThresholds,
        assign_mode,
        calibrate_mode_thresholds,
        load_mode_thresholds,
    )

    f_path, _, _ = _make_tables(tmp_path, n_unannotated=0, n_annotated=2000, seed=3)
    X, _, _, scaler, _ = load_and_clean_features(f_path)
    scaled = scaler.transform(pd.read_csv(f_path))

    thr = calibrate_mode_thresholds(scaled)
    assert thr.source.startswith("percentiles")
    assert thr.visibility_degraded <= thr.visibility_normal
    assert thr.det_conf_degraded <= thr.det_conf_normal

    path = os.path.join(tmp_path, "thr.json")
    thr.save(path)
    assert ModeThresholds.load(path) == thr
    assert load_mode_thresholds(path) == thr
    assert load_mode_thresholds(os.path.join(tmp_path, "missing.json")) == DEFAULT_MODE_THRESHOLDS

    modes = scaled.apply(lambda r: assign_mode(r, thr), axis=1).value_counts()
    assert set(modes.index) == {"Normal", "Degraded", "Takeover"}
    # Calibrated cut-offs must not collapse the corpus into a single mode.
    assert modes.max() < 0.9 * modes.sum()
    # And the default-argument path is unchanged.
    assert assign_mode(scaled.iloc[0]) == assign_mode(scaled.iloc[0], DEFAULT_MODE_THRESHOLDS)
