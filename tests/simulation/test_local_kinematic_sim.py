"""Tests for local_kinematic_sim.py's fallback stepper (no GPU, no CARLA, no artifacts needed)."""

import numpy as np
import pytest

from src.simulation.carla_config import CarlaConfig
from src.simulation.local_kinematic_sim import LocalKinematicTickSource


def _fake_images(n=5):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(n, 224, 320, 3), dtype=np.uint8)


def test_requires_at_least_one_image():
    with pytest.raises(ValueError):
        LocalKinematicTickSource(CarlaConfig(), np.zeros((0, 224, 320, 3), dtype=np.uint8))


def test_tick_returns_a_frame_from_the_provided_images():
    images = _fake_images(3)
    source = LocalKinematicTickSource(CarlaConfig(), images)
    result = source.tick()
    assert result.frame.shape == (224, 320, 3)
    assert np.array_equal(result.frame, images[0])


def test_frame_cursor_cycles_through_images():
    images = _fake_images(3)
    source = LocalKinematicTickSource(CarlaConfig(), images)
    seen = [source.tick().frame for _ in range(6)]
    assert np.array_equal(seen[0], seen[3])
    assert np.array_equal(seen[1], seen[4])
    assert np.array_equal(seen[2], seen[5])


def test_speed_converges_toward_mode_target_speed():
    config = CarlaConfig()
    source = LocalKinematicTickSource(config, _fake_images(), route_length_m=10_000.0)
    source.apply_mode("Normal")
    target = config.physics_profiles["Normal"].target_speed_mps

    speeds = [source.tick().v_ego for _ in range(200)]
    assert speeds[0] < speeds[-1]
    assert speeds[-1] == pytest.approx(target, rel=0.05)


def test_speed_never_overshoots_target_from_rest():
    config = CarlaConfig()
    source = LocalKinematicTickSource(config, _fake_images(), route_length_m=10_000.0)
    source.apply_mode("Normal")
    target = config.physics_profiles["Normal"].target_speed_mps

    for _ in range(200):
        result = source.tick()
        assert result.v_ego <= target + 1e-9


def test_position_advances_as_speed_increases():
    source = LocalKinematicTickSource(CarlaConfig(), _fake_images(), route_length_m=10_000.0)
    source.apply_mode("Normal")
    positions = [source.tick().position for _ in range(20)]
    assert positions[-1][0] > positions[0][0]


def test_progress_is_bounded_in_zero_one():
    source = LocalKinematicTickSource(CarlaConfig(), _fake_images(), route_length_m=5.0)
    source.apply_mode("Normal")
    for _ in range(500):
        result = source.tick()
        assert 0.0 <= result.progress <= 1.0


def test_slip_angle_disturbance_is_zero_in_normal_and_nonzero_in_degraded_and_takeover():
    source = LocalKinematicTickSource(CarlaConfig(), _fake_images())

    source.apply_mode("Normal")
    assert source.tick().slip_angle == 0.0

    source.apply_mode("Degraded")
    degraded_slip = source.tick().slip_angle
    assert degraded_slip > 0.0

    source.apply_mode("Takeover")
    takeover_slip = source.tick().slip_angle
    assert takeover_slip > degraded_slip


def test_v_front_defaults_to_matching_ego_speed_with_zero_closing_rate():
    config = CarlaConfig(fallback_closing_rate_mps=0.0)
    source = LocalKinematicTickSource(config, _fake_images())
    source.apply_mode("Normal")
    for _ in range(50):
        result = source.tick()
    assert result.v_front == pytest.approx(result.v_ego)


def test_v_front_is_reduced_by_a_positive_closing_rate():
    config = CarlaConfig(fallback_closing_rate_mps=3.0)
    source = LocalKinematicTickSource(config, _fake_images())
    source.apply_mode("Normal")
    for _ in range(50):
        result = source.tick()
    assert result.v_front == pytest.approx(max(0.0, result.v_ego - 3.0))
    assert result.v_front <= result.v_ego


def test_v_front_never_negative_even_when_closing_rate_exceeds_speed():
    config = CarlaConfig(fallback_closing_rate_mps=1000.0)
    source = LocalKinematicTickSource(config, _fake_images())
    source.apply_mode("Normal")
    result = source.tick()
    assert result.v_front == 0.0


def test_tick_index_increments():
    source = LocalKinematicTickSource(CarlaConfig(), _fake_images())
    ticks = [source.tick().tick for _ in range(5)]
    assert ticks == [0, 1, 2, 3, 4]
