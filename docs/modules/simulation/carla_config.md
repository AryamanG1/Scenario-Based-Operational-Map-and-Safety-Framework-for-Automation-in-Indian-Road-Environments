# `carla_config.py`

**Stage:** 8 (cross-cutting, optional)
**Package:** `src.simulation.carla_config`

## Purpose

Defines every tunable parameter for the closed-loop simulation in one JSON-backed dataclass, `CarlaConfig`. This is the literal "tuning parameters" deliverable: every knob needed to run this project's CARLA integration — or to deliberately avoid CARLA and exercise the rest of the subsystem through the local fallback simulator instead — lives here, editable in `configs/carla_config.json` without touching code.

## Paper / Formula Provenance

None — this module defines configuration structure and default values, not a formula. The per-mode `physics_profiles` and `weather_profiles` default values are this project's own engineering choice (target cruising speeds, tire friction, and weather intensities per Normal/Degraded/Takeover mode), not values transcribed from a paper. This is a deliberate, documented distinction from `RSS_PARAMETERS` in `rss_safety.py`, which *are* Salvi et al. 2022's exact Table IV values — this module imports/references those by label (via `rss_parameters_for_mu_odd()` in `closed_loop_runner.py`) and never duplicates them. See `docs/FORMULA_PROVENANCE.md`.

## Public API

```python
@dataclass
class PhysicsProfile:
    target_speed_mps: float
    tire_friction: float = 3.5
    damping_rate_full_throttle: float = 0.15
    max_rpm: float = 6000.0
```
Per-mode vehicle physics tuning, applied via CARLA's `VehiclePhysicsControl`. `target_speed_mps` is also used by `LocalKinematicTickSource` as its stepper's convergence target.

```python
@dataclass
class WeatherProfile:
    cloudiness: float = 0.0
    precipitation: float = 0.0
    fog_density: float = 0.0
    wetness: float = 0.0
```
Per-mode `carla.WeatherParameters` fields (0-100 each). Ignored, not an error, under the local fallback (which has no weather rendering).

```python
@dataclass
class CarlaConfig:
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
    physics_profiles: Dict[str, PhysicsProfile]
    weather_profiles: Dict[str, WeatherProfile]
```
Full field-by-field documentation is in the dataclass's own docstring. The three fields most directly answering "run without CARLA attached": `use_carla` (master toggle — `False` never even attempts an import/connection), `fallback_on_error` (mid-run resilience — a CARLA disconnect drops to the local simulator instead of crashing), and `monitor_check_interval_ticks=0` (disables the expensive real Stage 6 check by default, matching `feasibility_map.py`'s own documented "too expensive per-row" default).

```python
def save_default_carla_config(path: str = CARLA_CONFIG_JSON) -> None
def load_carla_config(path: str = CARLA_CONFIG_JSON) -> CarlaConfig
```
`load_carla_config` returns built-in `CarlaConfig()` defaults if `path` doesn't exist — it does **not** create the file — so `closed_loop_runner.py` works out of the box with zero setup. `save_default_carla_config` is how `configs/carla_config.json` (checked into the repo) was generated.

## Key Design Decisions & Edge Cases

- **Hand-rolled JSON mapping, not a generic dataclass-JSON helper.** No such helper exists anywhere in this codebase to reuse; this follows `odd_boundary.py`'s own hand-rolled `save_odd_copula`/`load_odd_copula` precedent instead of introducing a new dependency (e.g. `dacite`) for one config file.
- **`physics_profiles`/`weather_profiles` are nested dicts of dataclasses**, which `dataclasses.asdict()` flattens automatically on save but needs explicit reconstruction (`PhysicsProfile(**p)` / `WeatherProfile(**w)`) on load — plain `CarlaConfig(**d)` would leave them as plain dicts, breaking attribute access like `config.physics_profiles["Normal"].target_speed_mps`.
- **Loading a missing file returns defaults rather than raising or auto-creating the file** — this keeps `closed_loop_runner.py` (and any test) usable with zero setup, and avoids surprising a user by silently writing a file they didn't ask for.
- **`RSS_PARAMETERS`/`RSSParameters` from `rss_safety.py` are referenced by label (`"NoRain"`/`"LowRain"`/`"HeavyRain"`) inside `closed_loop_runner.py`, never duplicated here** — this config only owns Normal/Degraded/Takeover-mode knobs (a different axis: this project's own decision severity, not Salvi et al.'s weather micro-ODD).

## Dependencies

- Standard library: `json`, `os`, `dataclasses`
- External: none
- Internal: `src.common.paths` (for `CARLA_CONFIG_JSON`)

## Usage Example

```python
from src.simulation.carla_config import CarlaConfig, load_carla_config

config = load_carla_config()  # configs/carla_config.json, or defaults if absent
config.use_carla = True
config.host = "192.168.1.50"  # point at a remote GPU-capable CARLA server
config.physics_profiles["Takeover"].target_speed_mps = 3.0  # even more cautious
```

## Running Standalone

```bash
python -m src.simulation.carla_config
```
Writes a fresh default config to `configs/carla_config.json`, reloads it, prints the reloaded `CarlaConfig`, and asserts the round-trip preserved `Takeover`'s `target_speed_mps == 5.6`.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/simulation/test_carla_config.py` covers: `use_carla`/`fallback_on_error` defaults, all three modes present in both profile dicts, target speed strictly decreasing with mode severity (Normal > Degraded > Takeover > 0), `load_carla_config` returning exact defaults when the file is missing, a full save/load round trip via `save_default_carla_config`, and a round trip that preserves user-edited values (host/port/`use_carla`/a modified `target_speed_mps`).
