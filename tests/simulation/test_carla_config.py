"""Tests for carla_config.py's tunable configuration load/save."""

import os

from src.simulation.carla_config import (
    CarlaConfig,
    PhysicsProfile,
    WeatherProfile,
    load_carla_config,
    save_default_carla_config,
)


def test_default_config_disables_carla_by_default():
    config = CarlaConfig()
    assert config.use_carla is False
    assert config.fallback_on_error is True


def test_default_physics_profiles_cover_all_three_modes():
    config = CarlaConfig()
    for mode in ("Normal", "Degraded", "Takeover"):
        assert mode in config.physics_profiles
        assert isinstance(config.physics_profiles[mode], PhysicsProfile)
        assert mode in config.weather_profiles
        assert isinstance(config.weather_profiles[mode], WeatherProfile)


def test_target_speed_decreases_as_mode_severity_increases():
    config = CarlaConfig()
    normal = config.physics_profiles["Normal"].target_speed_mps
    degraded = config.physics_profiles["Degraded"].target_speed_mps
    takeover = config.physics_profiles["Takeover"].target_speed_mps
    assert normal > degraded > takeover > 0


def test_load_carla_config_returns_defaults_when_file_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.json")
    config = load_carla_config(missing_path)
    assert config == CarlaConfig()


def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "carla_config.json")
    save_default_carla_config(path)
    assert os.path.isfile(path)

    loaded = load_carla_config(path)
    assert loaded.host == "localhost"
    assert loaded.port == 2000
    assert loaded.physics_profiles["Takeover"].target_speed_mps == 5.6
    assert loaded.weather_profiles["Degraded"].precipitation == 30.0


def test_round_trip_preserves_edited_values(tmp_path):
    path = str(tmp_path / "carla_config.json")
    config = CarlaConfig(use_carla=True, host="192.168.1.50", port=3000)
    config.physics_profiles["Normal"].target_speed_mps = 20.0

    import json
    from dataclasses import asdict

    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f)

    loaded = load_carla_config(path)
    assert loaded.use_carla is True
    assert loaded.host == "192.168.1.50"
    assert loaded.port == 3000
    assert loaded.physics_profiles["Normal"].target_speed_mps == 20.0
