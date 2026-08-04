"""Tunable configuration for the closed-loop simulation (Stage 8, cross-cutting).

Every knob a user needs to run this project's CARLA integration -- or to
deliberately avoid it and exercise the rest of the subsystem through the
local fallback simulator instead -- lives in one JSON-backed dataclass
here. This is the "tuning parameters" surface referenced throughout
docs/modules/simulation/ and docs/FUTURE_STEPS.md.

The per-mode `physics_profiles` and `weather_profiles` are this project's
own engineering choice, not values transcribed from a paper (contrast with
RSS_PARAMETERS in rss_safety.py, which *are* Salvi et al. 2022's exact
Table IV values and are referenced here by label, never duplicated). See
docs/FORMULA_PROVENANCE.md for that distinction.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict

from src.common.paths import CARLA_CONFIG_JSON


@dataclass
class PhysicsProfile:
    """Per-mode vehicle physics tuning, applied via CARLA's VehiclePhysicsControl.

    Attributes:
        target_speed_mps: Desired cruising speed for this mode (m/s). Used
            by both CarlaTickSource (as an autopilot/traffic-manager target)
            and LocalKinematicTickSource (as the stepper's target speed).
        tire_friction: CARLA wheel tire friction coefficient. Lower under
            Degraded/Takeover to reflect the wetter/lower-adhesion
            conditions those modes are correlated with in this project's
            own mode semantics (see odd_classifier.assign_mode()).
        damping_rate_full_throttle: CARLA engine damping rate at full
            throttle.
        max_rpm: CARLA engine max RPM.
    """

    target_speed_mps: float
    tire_friction: float = 3.5
    damping_rate_full_throttle: float = 0.15
    max_rpm: float = 6000.0


@dataclass
class WeatherProfile:
    """Per-mode CARLA weather parameters (carla.WeatherParameters fields).

    Attributes:
        cloudiness: 0-100.
        precipitation: 0-100 (rain intensity).
        fog_density: 0-100.
        wetness: 0-100 (road-surface wetness).
    """

    cloudiness: float = 0.0
    precipitation: float = 0.0
    fog_density: float = 0.0
    wetness: float = 0.0


def _default_physics_profiles() -> Dict[str, PhysicsProfile]:
    return {
        "Normal": PhysicsProfile(target_speed_mps=16.7, tire_friction=3.5),  # ~60 km/h
        "Degraded": PhysicsProfile(target_speed_mps=11.1, tire_friction=2.5),  # ~40 km/h
        "Takeover": PhysicsProfile(target_speed_mps=5.6, tire_friction=1.8),  # ~20 km/h, cautious
    }


def _default_weather_profiles() -> Dict[str, WeatherProfile]:
    return {
        "Normal": WeatherProfile(cloudiness=10.0, precipitation=0.0, fog_density=0.0, wetness=0.0),
        "Degraded": WeatherProfile(cloudiness=50.0, precipitation=30.0, fog_density=10.0, wetness=30.0),
        "Takeover": WeatherProfile(cloudiness=80.0, precipitation=80.0, fog_density=40.0, wetness=80.0),
    }


@dataclass
class CarlaConfig:
    """All tunable parameters for the closed-loop simulation.

    Attributes:
        use_carla: Master toggle. If True, closed_loop_runner.py tries to
            connect to a real CARLA server (see is_carla_available()) and
            automatically falls back to LocalKinematicTickSource with a
            printed warning if unavailable. If False, always uses the
            local fallback -- no attempt to import/connect to CARLA at
            all, so this also works on machines without the `carla`
            package installed.
        fallback_on_error: If True, a mid-run CarlaTickSource exception
            (server crash, actor destroyed, etc.) drops to the local
            fallback for the remaining ticks instead of crashing the run.
        host, port, timeout_s: CARLA server connection parameters.
        town: CARLA map/town name to load.
        vehicle_blueprint: Ego vehicle blueprint ID.
        lead_vehicle_blueprint: Lead-vehicle blueprint ID (CarlaTickSource
            only -- spawned on autopilot to provide a real v_front
            measurement for the RSS distance formula).
        lead_vehicle_spawn_gap_m: Initial gap ahead of the ego vehicle at
            which the lead vehicle is spawned (CarlaTickSource only).
        camera_x, camera_z: RGB camera mount offset from the vehicle origin
            (meters), matching this project's SegNet/YOLO input framing.
        camera_width, camera_height: Camera resolution; defaults match
            data_pipeline.IMAGE_SIZE so frames need no extra resizing.
        tick_dt: Simulation timestep (s) for both tick sources.
        monitor_check_interval_ticks: How often (in ticks) to run the real
            Stage 6 perturbation-based perception_monitor check instead of
            assuming "Nominal". 0 disables it (default -- matches
            feasibility_map.py's own documented "too expensive per-row"
            default), since it costs several extra SegNet+YOLO passes.
        fallback_closing_rate_mps: LocalKinematicTickSource has no real
            lead-vehicle measurement (IDD-Lite has no velocity/tracking
            signal at all), so v_front is a documented synthetic profile:
            v_front = clamp(v_ego - fallback_closing_rate_mps, 0, v_ego).
            Default 0 means "assume the lead vehicle matches ego speed"
            (the conservative baseline); set positive to deliberately
            exercise the RSS-braking scenario under the fallback.
        physics_profiles: Per-mode PhysicsProfile, keyed by
            "Normal"/"Degraded"/"Takeover".
        weather_profiles: Per-mode WeatherProfile (CarlaTickSource only;
            ignored, not an error, under the local fallback).
    """

    use_carla: bool = False
    fallback_on_error: bool = True

    host: str = "localhost"
    port: int = 2000
    timeout_s: float = 5.0
    town: str = "Town03"
    vehicle_blueprint: str = "vehicle.tesla.model3"
    lead_vehicle_blueprint: str = "vehicle.audi.a2"
    lead_vehicle_spawn_gap_m: float = 15.0

    camera_x: float = 1.5
    camera_z: float = 2.4
    camera_width: int = 320
    camera_height: int = 224

    tick_dt: float = 0.1
    monitor_check_interval_ticks: int = 0
    fallback_closing_rate_mps: float = 0.0

    physics_profiles: Dict[str, PhysicsProfile] = field(default_factory=_default_physics_profiles)
    weather_profiles: Dict[str, WeatherProfile] = field(default_factory=_default_weather_profiles)


def _config_to_dict(config: CarlaConfig) -> dict:
    d = asdict(config)
    return d


def _config_from_dict(d: dict) -> CarlaConfig:
    d = dict(d)
    physics_profiles = {mode: PhysicsProfile(**p) for mode, p in d.pop("physics_profiles", {}).items()}
    weather_profiles = {mode: WeatherProfile(**w) for mode, w in d.pop("weather_profiles", {}).items()}
    config = CarlaConfig(**d)
    if physics_profiles:
        config.physics_profiles = physics_profiles
    if weather_profiles:
        config.weather_profiles = weather_profiles
    return config


def save_default_carla_config(path: str = CARLA_CONFIG_JSON) -> None:
    """Writes a fresh CarlaConfig() with default values to `path` as JSON.

    Args:
        path: Destination JSON file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_config_to_dict(CarlaConfig()), f, indent=2)


def load_carla_config(path: str = CARLA_CONFIG_JSON) -> CarlaConfig:
    """Loads a CarlaConfig from `path`, or returns built-in defaults if absent.

    Args:
        path: Source JSON file path. If it doesn't exist, this function
            does NOT create it -- it simply returns CarlaConfig() defaults,
            so the closed-loop runner works out of the box with no setup.

    Returns:
        A CarlaConfig.
    """
    if not os.path.isfile(path):
        return CarlaConfig()
    with open(path, encoding="utf-8") as f:
        return _config_from_dict(json.load(f))


if __name__ == "__main__":
    from src.common.paths import ensure_output_dirs

    ensure_output_dirs()
    save_default_carla_config()
    print(f"Wrote default CARLA config -> '{CARLA_CONFIG_JSON}'")

    reloaded = load_carla_config()
    print(reloaded)
    assert reloaded.physics_profiles["Takeover"].target_speed_mps == 5.6
    print("Round-trip load OK.")
