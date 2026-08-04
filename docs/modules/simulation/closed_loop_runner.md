# `closed_loop_runner.py`

**Stage:** 8 (cross-cutting, optional)
**Package:** `src.simulation.closed_loop_runner`

## Purpose

The closed-loop simulation orchestrator: runs this project's full Stage 2/4/5/7 per-tick decision flow against a sequence of frames from either a real CARLA server (`carla_bridge.CarlaTickSource`) or the local kinematic fallback (`local_kinematic_sim.LocalKinematicTickSource`), and logs the result. This is the concrete deliverable for "the CARLA portion" — a real closed-loop integration with a fully-functional, fully-tested fallback for when CARLA isn't attached — and the primary entry point for running it (`python -m src.simulation.closed_loop_runner`).

## Paper / Formula Provenance

This module is pure orchestration — every formula it calls is imported, not reimplemented, from the module that already owns it:

| Step | Function | Source module |
|---|---|---|
| Stage 2 perception | `run_detection`, `compute_features` | `src.perception.feature_extraction` |
| Stage 4 fuzzy scenario | `classify_mu_odd`, `defuzzify_mu_odd` | `src.odd.fuzzy_odd` (Salvi et al. 2022) |
| Stage 5 ODD region | `classify_odd_region` | `src.odd.odd_boundary` (Jiang et al. 2024) |
| Stage 5 RSS distance | `rss_parameters_for_mu_odd`, `rss_minimum_safe_distance` | `src.odd.rss_safety` (Salvi et al. 2022, Eq. 1) |
| Stage 5 sideslip stability | `SideslipState`, `sideslip_dynamics`, `compute_stability_boundary`, `is_sideslip_stable`, `stability_margin` | `src.odd.rss_safety` (Jiang et al. 2024, Eqs. 7-9, 20) |
| Stage 6 monitoring (optional) | `run_perception_monitor` | `src.monitoring.perception_monitor` |
| Stage 7 base mode | `assign_mode` | `src.odd.odd_classifier` |
| Stage 7 combination | `combine_stage_outputs` | `src.odd.odd_classifier` |
| ADS state | `ADSStateMachine.step` | `src.decision.sae_taxonomy` (SAE J3016) |

The sideslip model's fixed vehicle constants (`_VEHICLE_MASS_KG=1500.0`, `_VEHICLE_IZ_KGM2=2500.0`, `_VEHICLE_L_F_M=1.2`, `_VEHICLE_L_R_M=1.6`, `_ROAD_MU=0.8`, `_TIRE_F_ZF_N=7000.0`, `_TIRE_F_ZR_N=6500.0`) are the same generic mid-size-sedan example values `rss_safety.py`'s own `__main__` demo uses — not a per-tick measurement, since neither tick source exposes real yaw inertia or axle geometry.

## Public API

```python
def run_closed_loop_simulation(
    num_ticks: int,
    config: CarlaConfig,
    segnet_model,
    yolo_model,
    feature_scaler: FeatureScaler,
    copula_model: Optional[ODDCopulaModel] = None,
    pd_breakpoints: Optional[PDBreakpoints] = None,
    force_fallback: bool = False,
) -> pd.DataFrame
```
Runs `num_ticks` of the closed-loop simulation. If `copula_model`/`pd_breakpoints` are `None`, both are fit once at startup from `outputs/final_features.csv` (raises `FileNotFoundError` if that doesn't exist and neither was passed in explicitly). Selects `CarlaTickSource` iff `config.use_carla and not force_fallback and is_carla_available(config)`, else `LocalKinematicTickSource`, printing a warning on fallback. A mid-run `CarlaTickSource` exception triggers a lazy swap to a fresh `LocalKinematicTickSource` for the remaining ticks if `config.fallback_on_error` (the literal "if carla drops out I can run the rest of the subsystem" behavior) — otherwise it re-raises. Returns a DataFrame with one row per tick: `tick`, `t`, `base_mode`, `mode`, `odd_region`, `monitoring_state`, `mu_odd_label`, `ads_state`, `v_ego`, `v_front`, `rss_distance`, `sideslip_stable`, `stability_margin`, `position_x`, `position_y`, `progress`, `used_carla`.

```python
def write_carla_live_js(
    result_df: pd.DataFrame, road_name: str, used_carla: bool,
    output_path: str = DASHBOARD_CARLA_LIVE_JS,
) -> None
```
Writes one full closed-loop run to `dashboard/carla_live.js` as `const CARLA_LIVE = {...};` (a compact per-tick array: `tick`, `t`, `mode`, `ads_state`, `rss_distance`, `stability_margin`, `v_ego`, `progress`) for `dashboard/app.js`'s client-side replay animation. See "Key Design Decisions" for why this is a `const` JS global, not a `.json` file.

## Key Design Decisions & Edge Cases

- **Uses `assign_mode`/`combine_stage_outputs` (the documented Stage-7 combination path), not `odd_classifier.evaluate_road_scene()`'s standalone trained-RandomForest `predict()` shortcut.** `combine_stage_outputs` is what actually consumes Stage 5 (`odd_region`) and Stage 6 (`monitoring_state`) signals per `docs/ARCHITECTURE.md`'s "most-severe-signal-wins" cross-stage design — this runner exercises the full documented architecture, not a single-classifier bypass.
- **`copula_model`/`pd_breakpoints` are fit once at startup, never per-tick.** Both need a training-set distribution (fitting a Gaussian copula or calibrating fuzzy-membership quartiles from one live frame is meaningless); fitting them from `outputs/final_features.csv` at the start of a run costs a few hundred milliseconds once, not per-tick — matching how `main.py`'s own Stage 4/5 already fit these fresh each pipeline run rather than caching them to disk (no `save_odd_copula`/`PDBreakpoints.save()` call exists in `main.py` either).
- **Real Stage 6 monitoring is opt-in and rate-limited (`config.monitor_check_interval_ticks`, default `0` = disabled)**, not run every tick. It costs several extra SegNet+YOLO passes per invocation (drawing a perturbed pseudo-sequence via `perturbation_engine.py`) — the same "too expensive to precompute per-row" tradeoff `feasibility_map.py` already documents for its own batch use, applied here to simulation-scale per-tick use instead. `monitoring_state` defaults to `"Nominal"` on ticks where it isn't run.
- **Per-axle tire slip angles aren't separately modeled** — neither tick source exposes steering-angle/yaw-geometry detail needed to compute true front/rear slip angles, so both `alpha_f` and `alpha_r` are approximated as the tick's overall vehicle sideslip angle. Documented explicitly in-code as a simplification, not silently fabricated precision (see `docs/FUTURE_STEPS.md` for a real per-axle model as a listed extension).
- **`v_x` is floored at `_MIN_VX_FOR_SIDESLIP=0.1`** before building `SideslipState`, since `rss_safety.sideslip_dynamics()` raises `ValueError` at `v_x=0` and a near-stationary vehicle has no meaningful sideslip risk anyway — a documented guard, not a silent bug-hider.
- **Dashboard output is a `const` JS global replay array, not a live-polling JSON file served over `http.server`.** This project's entire dashboard philosophy (`app.js`'s own docstring, `docs/SETUP.md`, `feasibility_map.export_pipeline_stats_js()`'s docstring) is deliberately `file://`-only, no local server, because `fetch()` of local files is blocked under `file://`. A closed-loop run is a finite, already-completed tick sequence anyway, so `write_carla_live_js()` writes the **entire** tick history at once (mirroring `export_pipeline_stats_js()`'s exact header-comment + `json.dumps` pattern), and `app.js` replays it client-side with `setInterval` — zero new infrastructure, works over plain `file://` like the rest of the dashboard. True live polling during an in-progress run is a documented `docs/FUTURE_STEPS.md` stretch item, not built here.
- **`FEATURE_SCALER_PATH`, `SEGNET_CHECKPOINT`, `YOLO_WEIGHTS` etc. are imported as module-level names from `src.common.paths`**, not hardcoded, so tests can `monkeypatch` them (e.g. `test_raises_without_features_csv_or_explicit_copula_and_breakpoints` overrides `FEATURES_CSV` to a nonexistent path to exercise the `FileNotFoundError` branch without touching the real file).
- **`hasattr(tick_source, "close")` guards the cleanup call** at the end of a run — `LocalKinematicTickSource` has no `close()` method (nothing to release), while `CarlaTickSource` does (actor destruction, restoring async world settings); this lets the same cleanup line work for either.

## Dependencies

- Standard library: `argparse`, `json`, `os`
- External: `numpy`, `pandas`, `torch`
- Internal: `src.common.paths`; `src.decision.sae_taxonomy`; `src.odd.fuzzy_odd`; `src.odd.odd_boundary`; `src.odd.odd_classifier`; `src.odd.rss_safety`; `src.perception.data_pipeline`; `src.perception.feature_extraction`; `src.simulation.carla_bridge`; `src.simulation.carla_config`; `src.simulation.local_kinematic_sim`; `src.simulation.tick_source`; `src.monitoring.perception_monitor` (lazily imported, only when `monitor_check_interval_ticks > 0` actually fires, to avoid the import cost on every run that doesn't use it)

## Usage Example

```python
import joblib
from ultralytics import YOLO

from src.common.paths import FEATURE_SCALER_PATH, SEGNET_CHECKPOINT, YOLO_WEIGHTS
from src.perception.segnet_model import load_segnet
from src.simulation.carla_config import CarlaConfig
from src.simulation.closed_loop_runner import run_closed_loop_simulation, write_carla_live_js

segnet = load_segnet(SEGNET_CHECKPOINT)
yolo = YOLO(YOLO_WEIGHTS)
feature_scaler = joblib.load(FEATURE_SCALER_PATH)

config = CarlaConfig(use_carla=False)  # always use the local fallback
result_df = run_closed_loop_simulation(
    num_ticks=50, config=config,
    segnet_model=segnet, yolo_model=yolo, feature_scaler=feature_scaler,
    force_fallback=True,
)
print(result_df["mode"].value_counts())
write_carla_live_js(result_df, road_name="Jan Marg", used_carla=False, output_path="dashboard/carla_live.js")
```

## Running Standalone

```bash
python -m src.simulation.closed_loop_runner --num_ticks 50 --carla_config configs/carla_config.json --force_fallback --road_name "Jan Marg"
```
Loads the trained SegNet/YOLO/`FeatureScaler` artifacts, runs the closed-loop simulation, prints a per-tick summary table plus the final mode distribution, and writes `dashboard/carla_live.js`. Drop `--force_fallback` to let it try a real CARLA server if `configs/carla_config.json`'s `use_carla` is `true` (auto-falls-back with a warning if none is reachable — the default state of this sandbox). `main.py --carla [--carla_ticks N] [--carla_config path]` is a secondary, lower-configuration entry point that runs this as an optional Stage 8 after the existing Stages 1-7, reusing already-loaded models.

## Tests

`/home/aryaman-gudwani/Desktop/Capstone_Car_Automation/capstone_project/tests/simulation/test_closed_loop_runner.py` covers, all with `force_fallback=True` against the real trained SegNet/YOLO/`FeatureScaler` artifacts (`pytest.skip()`s if `models/feature_scaler.pkl` is missing, matching `tests/decision/test_feasibility_map.py`'s established pattern): correct row count and column set, `used_carla` is `False` throughout under forced fallback, `mode`/`base_mode` are valid Normal/Degraded/Takeover values, `odd_region` is a valid within/near/outside value, `ads_state` is a valid `ADSState` value, RSS distances are non-negative, tick indices are sequential, `progress` stays bounded, `run_closed_loop_simulation` raises `FileNotFoundError` when `FEATURES_CSV` is unavailable and no explicit `copula_model`/`pd_breakpoints` are supplied, and `write_carla_live_js` produces a syntactically valid `const CARLA_LIVE = {...};` file whose embedded JSON round-trips correctly. The `CarlaTickSource` path this module can select is not separately exercised here beyond `is_carla_available()` returning `False` — see `carla_bridge.md`'s Tests section.
