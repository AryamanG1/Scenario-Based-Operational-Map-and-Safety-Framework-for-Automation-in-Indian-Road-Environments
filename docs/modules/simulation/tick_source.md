# `tick_source.py`

**Stage:** 8 (cross-cutting, optional)
**Package:** `src.simulation.tick_source`

## Purpose

Defines the single shared per-tick contract implemented by both `carla_bridge.CarlaTickSource` (real CARLA server) and `local_kinematic_sim.LocalKinematicTickSource` (no-CARLA fallback), so `closed_loop_runner.py` can drive either one identically without knowing which is active. This is the seam that makes "if carla is not attached I can run the rest of the subsystem" a structural guarantee rather than an if/else scattered across the orchestrator.

## Paper / Formula Provenance

None — this is a pure interface/data-shape module.

## Public API

```python
@dataclass
class TickResult:
    tick: int
    frame: np.ndarray       # BGR, IMAGE_SIZE-shaped
    v_ego: float             # m/s
    v_front: float           # m/s
    yaw_rate: float           # rad/s
    slip_angle: float         # rad
    position: Tuple[float, float]  # (x, y), meters, arbitrary local frame
    progress: float           # 0.0-1.0, for the dashboard replay marker
```
One simulation tick's complete observation, in exactly the shape `closed_loop_runner.py` needs to run Stage 2/4/5/7's per-tick decision flow. `position`/`progress` exist purely for `dashboard/app.js`'s replay animation, not for any decision logic.

```python
class TickSource(Protocol):
    def tick(self) -> TickResult: ...
    def apply_mode(self, mode: str) -> None: ...
```
Structural interface. `tick()` advances the simulation by one timestep and returns the observation; `apply_mode(mode)` feeds the current Stage 7 decision (`"Normal"`/`"Degraded"`/`"Takeover"`) back into the simulation (real physics/weather actuation for CARLA, a target-speed/disturbance adjustment for the fallback).

## Key Design Decisions & Edge Cases

- **`typing.Protocol`, not `abc.ABC` or a shared base class.** This codebase has zero uses of `abc.ABC` or class inheritance anywhere in `src/` (every "class" elsewhere is a `@dataclass`, a `torch.nn.Module` subclass, or a plain stateful utility like `odd_classifier.FeatureScaler`) — `Protocol` gives structural typing with zero runtime cost and no forced base class, matching that style exactly. `CarlaTickSource` and `LocalKinematicTickSource` each independently satisfy this shape with no import of each other or of this module at the class-definition level (only `TickResult` is actually imported by both).
- **No internal state or history here.** `TickResult` is a plain snapshot; each `TickSource` implementation owns all mutable simulation state (CARLA actor handles, or the kinematic stepper's position/speed) itself.

## Dependencies

- Standard library: `dataclasses`, `typing`
- External: `numpy` (for `TickResult.frame`'s type)
- Internal: none

## Usage Example

```python
from src.simulation.tick_source import TickResult, TickSource

def drive(source: TickSource, mode: str, n_ticks: int) -> list[TickResult]:
    results = []
    for _ in range(n_ticks):
        result = source.tick()
        results.append(result)
        source.apply_mode(mode)
    return results
```
This function works identically whether `source` is a `CarlaTickSource` or a `LocalKinematicTickSource` — that interchangeability is this module's entire purpose.

## Running Standalone

Not applicable — this module defines only a dataclass and a `Protocol`, with no `__main__` block (there is nothing to demonstrate in isolation; see `carla_config.py`, `carla_bridge.py`, or `local_kinematic_sim.py` for runnable demos of concrete `TickSource` implementations).

## Tests

No dedicated test file — `TickResult`/`TickSource` are exercised indirectly through every test in `tests/simulation/test_local_kinematic_sim.py` and `tests/simulation/test_closed_loop_runner.py`, which assert on `TickResult` field values returned by a concrete `TickSource`.
