# `local_kinematic_sim.py`

**Stage:** 8 (cross-cutting, optional)
**Package:** `src.simulation.local_kinematic_sim`

## Purpose

The no-CARLA fallback `TickSource`: real IDD-Lite frames plus a local kinematic stepper, requiring no GPU, no external server, and no `carla` package. Used whenever `CarlaConfig.use_carla` is `False`, `carla_bridge.is_carla_available()` returns `False`, or `force_fallback=True` is passed to `run_closed_loop_simulation()` — this module is the concrete answer to "if carla is not attached I can run the rest of the subsystem."

## Paper / Formula Provenance

None — this is an original, documented engineering choice (a first-order speed-convergence kinematic model), not a formula transcribed from any of the project's cited papers. It exists to drive real formulas elsewhere (`rss_safety.py`'s RSS distance and sideslip stability, called from `closed_loop_runner.py`) with plausible per-tick kinematics when no real vehicle/simulator is available.

## Public API

```python
class LocalKinematicTickSource:
    def __init__(self, config: CarlaConfig, images: np.ndarray, route_length_m: float = 200.0) -> None
    def tick(self) -> TickResult
    def apply_mode(self, mode: str) -> None
```
Structurally satisfies `tick_source.TickSource`. `images` must be a non-empty `(M, H, W, 3)` BGR `uint8` array, as returned by `data_pipeline.load_and_clean_dataset()` — raises `ValueError` if empty. `tick()` cycles through `images` (wrapping via modulo), steps the kinematic state by `config.tick_dt`, and returns a `TickResult`. `apply_mode(mode)` just records the mode; the *next* `tick()` call converges toward that mode's target speed.

## Key Design Decisions & Edge Cases

- **Real IDD-Lite frames, not fabricated camera images.** Directly follows `perception_monitor.py`'s existing, established precedent: "no live simulator — reuse real static images instead of inventing synthetic ones." Frames come from `data_pipeline.load_and_clean_dataset(DATA_DIR)`'s train-split (there is no validation-split loader anywhere in this codebase to reuse instead — see `docs/FUTURE_STEPS.md`).
- **First-order (single time-constant) speed convergence**, not a full vehicle dynamics model: `v += (target - v) * min(1, dt / 2.0)`. Simple, numerically stable at any reasonable `tick_dt`, and monotonic (never overshoots the target from rest) — sufficient to exercise the downstream RSS/sideslip formulas with plausible values without claiming vehicle-dynamics fidelity `CarlaTickSource` doesn't need to claim either (CARLA's own physics engine handles that side).
- **Straight-line motion only (`yaw_rate` is always `0.0`)** — there is no route/steering model in the fallback, unlike `CarlaTickSource`, which follows CARLA's actual road network. Documented as a simplification, not a hidden approximation.
- **Deliberate slip-angle disturbance under Degraded/Takeover** (`0.0` / `0.03` / `0.08` rad for Normal/Degraded/Takeover respectively), so `rss_safety.is_sideslip_stable()`/`stability_margin()` have something real to react to in the fallback path — a **chosen test scenario**, exactly parallel to how `perturbation_engine.py`'s perturbations are deliberately chosen, not random noise for its own sake.
- **`v_front` is an explicitly synthetic, labeled profile — not a fabricated real-world measurement.** IDD-Lite has no velocity or object-tracking signal of any kind (single static images), so there is no honest way to measure a real lead vehicle's speed here (contrast with `CarlaTickSource`, which reads an actual spawned lead-vehicle actor's velocity). Formula: `v_front = clamp(v_ego - config.fallback_closing_rate_mps, 0, v_ego)`. Default `fallback_closing_rate_mps=0.0` means "assume the lead vehicle matches ego speed" — the conservative baseline (RSS distance shrinks toward its response-phase-only term); a user can dial in a positive closing rate via `configs/carla_config.json` to deliberately exercise the RSS-braking scenario under the fallback.
- **`progress` is purely for dashboard visualization** (`distance_travelled / route_length_m`, clamped to `[0, 1]`), never consulted by any decision logic.

## Dependencies

- Standard library: `math`
- External: `numpy`
- Internal: `src.simulation.carla_config` (`CarlaConfig`), `src.simulation.tick_source` (`TickResult`)

## Usage Example

```python
from src.perception.data_pipeline import load_and_clean_dataset
from src.common.paths import DATA_DIR
from src.simulation.carla_config import CarlaConfig
from src.simulation.local_kinematic_sim import LocalKinematicTickSource

images, _ = load_and_clean_dataset(DATA_DIR)
source = LocalKinematicTickSource(CarlaConfig(), images)
source.apply_mode("Normal")
for _ in range(50):
    result = source.tick()
    print(result.tick, result.v_ego, result.progress)
```

## Running Standalone

No `__main__` block — this module has no formulas or artifacts worth demonstrating in isolation beyond what its unit tests already cover directly. Use `python -m src.simulation.closed_loop_runner --force_fallback` to see it driving the full pipeline end-to-end.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/simulation/test_local_kinematic_sim.py` covers: `ValueError` on an empty image array, frames matching and cycling through the provided images, speed converging toward (and never overshooting) the current mode's target speed from rest, position advancing as speed increases, `progress` staying bounded in `[0, 1]` even well past the notional route length, slip-angle disturbance being exactly `0.0` under Normal and strictly increasing under Degraded then Takeover, `v_front` matching ego speed at zero closing rate, `v_front` correctly reduced by a positive closing rate, `v_front` clamped at `0.0` (never negative) when the closing rate exceeds ego speed, and sequential tick indices. All tests run with zero external artifacts (no GPU, no trained models, no CARLA) — this is the fully-tested half of the CARLA-attached/not-attached split.
