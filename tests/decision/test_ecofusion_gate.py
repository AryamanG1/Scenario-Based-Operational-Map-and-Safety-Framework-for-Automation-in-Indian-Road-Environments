"""Unit tests for ecofusion_gate.py's Pareto filtering and joint-loss selection."""

import pytest

from src.decision.ecofusion_gate import (
    CONFIGURATION_SPACE,
    energy_cost,
    filter_pareto_candidates,
    knowledge_gate_loss,
    select_configuration,
)


def test_configuration_space_has_seven_nonempty_subsets():
    assert len(CONFIGURATION_SPACE) == 7  # 2^3 - 1
    assert frozenset() not in CONFIGURATION_SPACE


def test_knowledge_gate_loss_zero_for_full_configuration():
    reliabilities = {"segnet_mask": 0.8, "yolo_detections": 0.5, "image_stats": 0.6}
    full_config = frozenset(reliabilities)
    assert knowledge_gate_loss(full_config, reliabilities) == pytest.approx(0.0)


def test_knowledge_gate_loss_higher_when_omitting_reliable_stream():
    reliabilities = {"segnet_mask": 0.9, "yolo_detections": 0.1, "image_stats": 0.1}
    omit_reliable = frozenset({"yolo_detections", "image_stats"})  # drops segnet_mask (0.9)
    omit_unreliable = frozenset({"segnet_mask", "image_stats"})  # drops yolo (0.1)
    loss_reliable_omitted = knowledge_gate_loss(omit_reliable, reliabilities)
    loss_unreliable_omitted = knowledge_gate_loss(omit_unreliable, reliabilities)
    assert loss_reliable_omitted > loss_unreliable_omitted


def test_energy_cost_sums_active_branch_latencies():
    latencies = {"segnet_mask": 0.03, "yolo_detections": 0.6, "image_stats": 0.001}
    config = frozenset({"segnet_mask", "image_stats"})
    assert energy_cost(config, latencies) == pytest.approx(0.031)


def test_filter_pareto_candidates_always_includes_the_optimum():
    losses = {frozenset({"a"}): 0.5, frozenset({"a", "b"}): 0.0, frozenset({"b"}): 0.9}
    candidates = filter_pareto_candidates(losses, gamma=0.1)
    assert frozenset({"a", "b"}) in candidates


def test_filter_pareto_candidates_excludes_far_worse_configs():
    losses = {frozenset({"a"}): 0.5, frozenset({"a", "b"}): 0.0, frozenset({"b"}): 0.9}
    candidates = filter_pareto_candidates(losses, gamma=0.1)
    assert frozenset({"b"}) not in candidates  # 0.9 - 0.0 = 0.9 > gamma


def test_select_configuration_prefers_accuracy_at_low_lambda():
    losses = {frozenset({"cheap"}): 0.5, frozenset({"expensive"}): 0.0}
    latencies = {"cheap": 0.01, "expensive": 1.0}
    candidates = list(losses.keys())
    selected, _ = select_configuration(candidates, losses, latencies, lambda_e=0.0)
    assert selected == frozenset({"expensive"})  # lowest Lf wins when lambda_e=0


def test_select_configuration_prefers_speed_at_high_lambda():
    losses = {frozenset({"cheap"}): 0.5, frozenset({"expensive"}): 0.0}
    latencies = {"cheap": 0.01, "expensive": 1.0}
    candidates = list(losses.keys())
    selected, _ = select_configuration(candidates, losses, latencies, lambda_e=1.0)
    assert selected == frozenset({"cheap"})  # lowest cost wins when lambda_e=1
