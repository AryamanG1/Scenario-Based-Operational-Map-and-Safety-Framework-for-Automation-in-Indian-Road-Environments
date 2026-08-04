# `carla_bridge.py`

**Stage:** 8 (cross-cutting, optional)
**Package:** `src.simulation.carla_bridge`

## Purpose

The real CARLA client integration: connects to a live CARLA 0.9.15 server, spawns an ego vehicle, an RGB camera, and an autopilot-driven lead vehicle, and exposes them as a `TickSource` (see `tick_source.py`) so `closed_loop_runner.py` can drive a genuine closed-loop simulation. Also owns `is_carla_available()`, the master fallback-trigger probe that lets the rest of the subsystem run without CARLA attached.

**Environment limitation, stated plainly:** this module is correct against the documented CARLA 0.9.15 Python API (verified by reading CARLA's own API reference), but it is **reviewed-by-reading only, not execution-verified**, in this project's development environment. This machine has no discrete GPU (CARLA's server needs one to render, even headless/off-screen), and the `carla` PyPI package has no wheel for this project's Python 3.12 venv (CARLA 0.9.13-0.9.15 wheels only support CPython 3.7-3.10) — confirmed directly: `import carla` raises `ModuleNotFoundError` here right now. This mirrors `rss_safety.py`'s own precedent of "implemented faithfully, verified in isolation" for formulas this sandbox categorically cannot run end-to-end. See `docs/FUTURE_STEPS.md` for exactly what a capable machine needs to actually exercise this path.

## Paper / Formula Provenance

None directly — this module contains no formulas of its own. It calls into `rss_safety.py`'s RSS-distance and sideslip-stability functions (via `closed_loop_runner.py`, not directly) with real CARLA-measured kinematics, and applies `carla_config.py`'s per-mode `PhysicsProfile`/`WeatherProfile` values (this project's own engineering choice, not paper-derived — see `carla_config.md`).

## Public API

```python
def is_carla_available(config: CarlaConfig) -> bool
```
Probes whether a real CARLA server is reachable, **never raises**. Tries `import carla` (catches `ImportError`), then a short-timeout `client.get_server_version()` call (catches any connection error). Returns `False` on any failure. This is the fully-exercised path in this sandbox — confirmed to return `False` here right now.

```python
class CarlaTickSource:
    def __init__(self, config: CarlaConfig) -> None
    def tick(self) -> TickResult
    def apply_mode(self, mode: str) -> None
    def close(self) -> None
```
Structurally satisfies `tick_source.TickSource`. `__init__` connects, loads `config.town`, enables synchronous mode at `config.tick_dt`, and spawns the ego vehicle (`config.vehicle_blueprint`), an RGB camera (`config.camera_width`×`config.camera_height`, mounted at `(config.camera_x, 0, config.camera_z)`), and a lead vehicle (`config.lead_vehicle_blueprint`) on autopilot `config.lead_vehicle_spawn_gap_m` ahead — the lead vehicle exists specifically to give the RSS formula a **real** `v_front` measurement (`lead_vehicle.get_velocity()`), not a fabricated one. `tick()` calls `world.tick()`, reads back the camera frame (BGRA→BGR) and ego/lead kinematics. `apply_mode()` sets `VehiclePhysicsControl`/`WeatherParameters` from the mode's profiles (only when the mode actually changes) and issues a simple proportional throttle/brake control toward the mode's `target_speed_mps`. `close()` destroys spawned actors and restores asynchronous world settings.

## Key Design Decisions & Edge Cases

- **A real lead-vehicle actor for `v_front`, not a formula.** RSS's `v_front` term needs a genuine leading-vehicle speed; rather than fabricate one, this module spawns an actual autopilot-driven vehicle and reads its real velocity — standard CARLA usage, and the honest way to get this signal (contrast with `LocalKinematicTickSource`, which has no such measurement available and uses an explicitly-labeled synthetic profile instead — see `local_kinematic_sim.md`).
- **Sideslip angle is approximated from velocity-heading vs. vehicle-heading**, since CARLA's basic `Actor` API doesn't directly expose a vehicle sideslip angle: `slip_angle = atan2(vy, vx) - yaw`, wrapped to `(-pi, pi]`. This is a standard approximation, not a CARLA-native measurement.
- **Physics/weather are only re-applied when the mode actually changes** (`if mode != self._current_mode`), not every tick — avoids redundant `apply_physics_control`/`set_weather` calls (both are relatively expensive CARLA operations) when the mode is stable across many ticks.
- **Speed control is a simple proportional controller** (`throttle = clamp(0.5 * speed_error, 0, 1)`, `brake` symmetric), not a full PID or CARLA's traffic-manager autopilot — sufficient to converge toward each mode's target speed, documented as approximate rather than presented as a tuned controller.
- **`carla` is imported lazily, inside `__init__`** (stored as `self._carla`), never at module level — so importing `carla_bridge.py` itself never fails even when the `carla` package isn't installed; only actually instantiating `CarlaTickSource` does, and only after `is_carla_available()` has already confirmed the import succeeds.
- **`close()` checks `getattr(self, "camera", None)` etc. rather than assuming all actors exist** — defensive against a partially-constructed instance (e.g. `__init__` raising partway through spawning) being cleaned up.

## Dependencies

- Standard library: `queue`
- External: `numpy`; `carla` (lazily imported inside `CarlaTickSource.__init__`/`is_carla_available`, NOT a hard dependency of this module or this project's main `requirements.txt` — see `requirements-carla.txt`)
- Internal: `src.simulation.carla_config` (`CarlaConfig`), `src.simulation.tick_source` (`TickResult`)

## Usage Example

```python
from src.simulation.carla_bridge import is_carla_available, CarlaTickSource
from src.simulation.carla_config import load_carla_config

config = load_carla_config()
config.use_carla = True

if is_carla_available(config):
    source = CarlaTickSource(config)
    try:
        for _ in range(50):
            result = source.tick()
            # ... run the Stage 2/4/5/7 decision flow on result.frame ...
            source.apply_mode("Normal")
    finally:
        source.close()
else:
    print("No CARLA server reachable -- see local_kinematic_sim.py for the fallback.")
```
In practice, use `closed_loop_runner.run_closed_loop_simulation()` instead of this directly — it performs exactly this selection/fallback logic plus the full per-tick decision flow.

## Running Standalone

No `__main__` block — this module requires a live CARLA server to do anything meaningful, which this sandbox cannot provide. Use `python -m src.simulation.closed_loop_runner --force_fallback` to exercise the rest of the pipeline via the local fallback instead, or run the same command **without** `--force_fallback` on a capable machine with `configs/carla_config.json`'s `use_carla` set to `true` and a CARLA server running.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/simulation/test_carla_bridge.py` covers: `is_carla_available()` returning `False` cleanly (never raising) with no reachable server, with an unreachable host (`203.0.113.1`, RFC 5737 TEST-NET-3), and with an invalid port — plus a documentation test that confirms `import carla` currently raises `ImportError` in this environment (fails loudly if that ever changes, as a signal that `CarlaTickSource` may become testable here). **`CarlaTickSource`'s live-server path has no automated test** — this is an intentional, permanent limitation of this development environment (see module docstring above), not a gap to chase; correctness there is by code review against the CARLA 0.9.15 API, not execution.
