# System Architecture

## Overview

This project implements a 7-stage pipeline that turns a raw road-scene image into an ODD (Operational Design Domain) automation decision: **Normal**, **Degraded**, or **Takeover**. The stage numbering and names follow the capstone proposal's actual pipeline diagram, not the flatter framing of the one-shot LLM scaffold this project's first implementation pass started from (see [`DATASET_NOTES.md`](DATASET_NOTES.md) and the top-level `README.md`'s "Development note" for that history).

```
Stage 1  Input Data                     src/perception/data_pipeline.py
Stage 2  Perception Layer               src/perception/{segnet_model,feature_extraction,
                                            lane_detection,detection_benchmark,multi_stream_fusion}.py
Stage 3  Traffic Density Estimation     src/odd/traffic_density.py
Stage 4  Scenario Classification        src/odd/fuzzy_odd.py
Stage 5  ODD Mapping                    src/odd/{odd_boundary,rss_safety,risk_estimation}.py
Stage 6  Real-Time Performance Monitor  src/monitoring/{perturbation_engine,perception_monitor}.py
Stage 7  Decision System                src/odd/odd_classifier.py, src/decision/{ecofusion_gate,feasibility_map}.py
Stage 8  Closed-Loop Simulation (opt.)  src/simulation/{carla_config,tick_source,carla_bridge,
                                            local_kinematic_sim,closed_loop_runner}.py
Cross-cutting                           src/decision/sae_taxonomy.py
Batch orchestration                     src/pipeline/full_dataset_pipeline.py, main.py
```

Stage 8 is a later, optional addition, not part of the original 7-stage proposal (see `docs/FORMULA_PROVENANCE.md`'s "Corrected/rejected citations" and the top-level README's "Known scope limitations" for why CARLA wasn't in scope initially, and `docs/FUTURE_STEPS.md` for what running it against a real CARLA server requires). `python main.py` runs Stages 1-7 unchanged whether or not Stage 8 exists; nothing in Stages 1-7 depends on `src/simulation/`.

## Package layout

```
capstone_project/
├── main.py                    # Stage 1-7 orchestrator (entry point)
├── requirements.txt
├── README.md
├── src/
│   ├── common/
│   │   └── paths.py           # centralized path constants, used by every module
│   ├── perception/             # Stage 2
│   ├── odd/                    # Stages 3, 4, 5, 7 (ODD formalization + classifier)
│   ├── monitoring/              # Stage 6
│   ├── decision/                # Stage 7 (decision system) + cross-cutting SAE taxonomy
│   ├── pipeline/                 # batch/full-dataset orchestration
│   └── simulation/                # Stage 8 (optional) -- CARLA + local-fallback closed-loop sim
├── tests/                      # mirrors src/ package-for-package
├── data/idd20k_lite/            # IDD-Lite dataset (images + semantic masks)
├── models/                     # trained checkpoints and fitted artifacts (.pth, .pkl)
├── outputs/                     # generated CSVs, calibration JSON, plots/
├── configs/carla_config.json     # Stage 8 tuning parameters
├── dashboard/                   # Leaflet.js map + real-pipeline-results panel + Stage 8 replay marker
└── docs/
    ├── ARCHITECTURE.md          # this file
    ├── FORMULA_PROVENANCE.md
    ├── DATASET_NOTES.md
    ├── SETUP.md
    └── modules/<package>/<module>.md   # one doc per source module
```

## Why this package split

The five `src/` subpackages mirror how the pipeline stages actually depend on each other — a clean DAG with no cycles:

```
perception   (no internal deps — the base layer: images/masks in, features out)
    ^   ^
    |   |
   odd  monitoring   (both depend only on perception)
    ^    ^
    |    |
decision-+    (depends on perception + odd)
    ^    ^
    |    |
pipeline simulation   (pipeline depends on perception+odd+monitoring;
                        simulation depends on perception+odd+decision+monitoring)
```

`simulation` sits at the top of the DAG alongside `pipeline` — nothing depends on it, so it carries zero cycle risk and Stages 1-7 are completely unaffected by its presence or absence. `src/common/paths.py` is deliberately dependency-free (only uses `os`) so every other module — regardless of position in the DAG — can import path constants without risking a circular import.

## Cross-stage data flow (Stage 7's `combine_stage_outputs`)

Stage 7's final decision is not a single classifier call — it merges three independent signals via a "most-severe-signal-wins" rule (`src/odd/odd_classifier.py::combine_stage_outputs`):

1. **`assign_mode()`** — a rule-based classifier over MinMax-scaled per-frame features (Stage 7's own base rule, with the critical "empty road ≠ Takeover" bug fix from the original notebook).
2. **`classify_odd_region()`** (Stage 5, `odd_boundary.py`) — within / near / outside the fitted Gaussian-copula ODD-density region.
3. **`run_perception_monitor().state`** (Stage 6, `perception_monitor.py`) — Nominal / Warning / Critical, from the perturbation-sequence consistency check.

Any one of the three can escalate the final mode toward Takeover; none can downgrade another's legitimate escalation. This mirrors a standard redundant-channel safety pattern and directly implements the proposal's Stage 7 description: *"Combine monitoring output + operational condition status... select system mode."*

## Running the pipeline

- **Full pipeline, one command:** `python main.py` (from the project root). Add `--force` to re-run SegNet training and feature extraction even if cached artifacts exist in `models/`/`outputs/`. Add `--carla [--carla_ticks N] [--carla_config path]` to also run optional Stage 8 afterward.
- **Batch/full-dataset, resumable:** `python -m src.pipeline.full_dataset_pipeline --batch_size 32 --checkpoint_interval 500`.
- **Stage 8 alone (its primary entry point):** `python -m src.simulation.closed_loop_runner --num_ticks 50 --force_fallback` — runs the local kinematic fallback with no CARLA/GPU required. Drop `--force_fallback` to let it try a real CARLA server per `configs/carla_config.json`.
- **Any individual module:** `python -m src.<package>.<module>` from the project root (e.g. `python -m src.odd.rss_safety`). This convention exists because the codebase is organized as real Python subpackages — see `src/common/paths.py`'s docstring for why direct file execution (`python src/odd/rss_safety.py`) is not supported after the reorganization (it would need per-file `sys.path` hacks that `-m` avoids entirely).
- **Tests:** `python -m pytest tests/ -q` from the project root (171 tests).

See [`SETUP.md`](SETUP.md) for environment setup (including optional CARLA setup) and [`FORMULA_PROVENANCE.md`](FORMULA_PROVENANCE.md) for the full paper-to-code citation map.
