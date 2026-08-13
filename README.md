# Scenario-Based Operational Map and Safety Framework for Automation in Indian Road Environments

**CPG No. 274 — Computer Science and Engineering Department, Thapar Institute of Engineering and Technology, Patiala**

## Abstract

Autonomous and semi-autonomous driving systems (ADAS) have been predominantly designed and validated for structured, well-regulated road environments. Indian roads, however, have chaotic mixed traffic, informal lanes, and unstructured scenarios compounded by adverse weather. This project builds a scenario-based Operational Design Domain (ODD) safety framework for Indian driving conditions: a full pipeline from dataset input through perception (semantic segmentation, YOLO-based object detection, lane detection, traffic-density estimation), reliability metric computation, environment complexity classification, and continuous runtime monitoring, culminating in an adaptive response system that transitions between **Normal**, **Degraded**, and **Driver Takeover** operation based on real-time conditions.

This system targets **SAE Level 2 (Partial Driving Automation)** per SAE J3016 — the driver remains engaged and responsible for object/event detection and response (OEDR) and fallback; the system sustains lateral and longitudinal control decisions within a limited ODD (see `src/decision/sae_taxonomy.py`).

## Documentation

| Doc | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Package layout, dependency DAG, cross-stage data flow, how to run everything |
| [`docs/FORMULA_PROVENANCE.md`](docs/FORMULA_PROVENANCE.md) | Full paper → equation → module citation table, including rejected/corrected citations |
| [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md) | The IDD-Lite class-mapping bug found and fixed during development |
| [`docs/DASHBOARD_ROAD_DATA.md`](docs/DASHBOARD_ROAD_DATA.md) | Where the dashboard's road geometry comes from (OpenStreetMap/Overpass) and how to regenerate it |
| [`docs/SETUP.md`](docs/SETUP.md) | Environment setup, first run, tests, dashboard, optional CARLA setup |
| [`docs/FUTURE_STEPS.md`](docs/FUTURE_STEPS.md) | What's next: real CARLA hardware setup, live dashboard polling, KITTI validation, and other stretch items |
| [`docs/modules/<package>/<module>.md`](docs/modules/) | One detailed doc per source module (purpose, formula provenance, full public API, design decisions, usage example) — 23 files, one per file under `src/` |

## Architecture — the 7-stage pipeline

```
                         ┌─────────────────────────────┐
                         │   STAGE 1: Input Data        │
                         │   data_pipeline.py            │
                         │   IDD-Lite (India Driving      │
                         │   Dataset), 2011 images        │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │   STAGE 2: Perception Layer   │
                         │   segnet_model.py (semantic seg)│
                         │   feature_extraction.py (YOLO) │
                         │   lane_detection.py             │
                         │   detection_benchmark.py        │
                         │   multi_stream_fusion.py        │
                         └──────────────┬──────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
  ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
  │ STAGE 3: Traffic       │ │ STAGE 4: Scenario      │ │ STAGE 5: ODD Mapping   │
  │ Density Estimation     │ │ Classification          │ │ odd_boundary.py        │
  │ traffic_density.py     │ │ fuzzy_odd.py             │ │ rss_safety.py          │
  │ Low/Med/High/Congested │ │ NoRain/LowRain/HeavyRain │ │ risk_estimation.py     │
  └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
              └─────────────────────────┴─────────────────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │  STAGE 6: Real-Time            │
                         │  Performance Monitoring         │
                         │  perception_monitor.py          │
                         │  Nominal / Warning / Critical    │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │  STAGE 7: Decision System        │
                         │  odd_classifier.py (rule + RF)   │
                         │  ecofusion_gate.py (adaptive      │
                         │  compute)                         │
                         │  feasibility_map.py                │
                         │  → Normal / Degraded / Takeover    │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │  STAGE 8 (optional): Closed-Loop  │
                         │  Simulation -- src/simulation/    │
                         │  real CARLA server, or a local     │
                         │  kinematic fallback if not attached│
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │  dashboard/ — Leaflet map +      │
                         │  real pipeline results panel +    │
                         │  Stage 8 replay marker             │
                         └─────────────────────────────┘
```

This mirrors the capstone proposal's actual pipeline (Input Data → Perception → Traffic Density → Scenario Classification → ODD Mapping → Real-Time Monitoring → Decision System), not the flatter, simplified framing of an early one-shot LLM scaffold this project started from (see [`docs/FORMULA_PROVENANCE.md`](docs/FORMULA_PROVENANCE.md) for that history). Stage 8 is a later, optional addition — the proposal itself never actually specifies a CARLA-simulation stage (see "Known scope limitations" below and `docs/FUTURE_STEPS.md`); it was added afterward to genuinely close that gap rather than leave the formulas unexercised.

## Project structure

```
capstone_project/
├── main.py                    # Stage 1-7 orchestrator (entry point: python main.py)
├── requirements.txt
├── README.md                  # this file
├── requirements-carla.txt      # separate, optional -- only for a Python 3.8-3.10 venv running real CARLA
├── src/
│   ├── common/paths.py         # centralized path constants used by every module
│   ├── perception/              # Stage 2 (6 modules)
│   ├── odd/                     # Stages 3, 4, 5, 7 -- ODD formalization + classifier (6 modules)
│   ├── monitoring/               # Stage 6 (2 modules)
│   ├── decision/                 # Stage 7 decision system + SAE taxonomy (3 modules)
│   ├── pipeline/                 # batch/full-dataset orchestration (1 module)
│   └── simulation/                # Stage 8 (optional) -- CARLA + local-fallback closed-loop sim (5 modules)
├── tests/                       # mirrors src/ package-for-package (171 tests)
├── data/idd20k_lite/             # IDD-Lite dataset (images + semantic masks)
├── models/                      # trained checkpoints + fitted artifacts (.pth, .pkl) -- generated
├── outputs/                      # generated CSVs, calibration JSON, plots/ -- generated
├── configs/carla_config.json     # Stage 8 tuning parameters (editable, checked in with defaults)
├── dashboard/                    # Leaflet.js map + real pipeline-results panel + Stage 8 replay marker
└── docs/                        # architecture docs + one doc per source module
```

`models/`, `outputs/`, `data/idd20k_lite/`, and `dashboard/carla_live.js` are populated by running the pipeline (or by copying the dataset in, for `data/`) — they hold generated artifacts, not source code.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU-only torch wheel (skip if you have a CUDA GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Copy or symlink the IDD-Lite dataset (`leftImg8bit/`, `gtFine/`) into `data/idd20k_lite/`. Full details, including first-run expectations and dashboard setup: [`docs/SETUP.md`](docs/SETUP.md).

## Usage

```bash
python main.py                                                              # runs all 7 stages end-to-end
python main.py --force                                                      # re-run SegNet training and feature extraction from scratch
python main.py --carla --carla_ticks 50                                     # also run optional Stage 8 (closed-loop sim)
python -m src.pipeline.full_dataset_pipeline --batch_size 32 --checkpoint_interval 500   # batch-scale, resumable
python -m src.simulation.closed_loop_runner --num_ticks 50 --force_fallback  # Stage 8 alone, no CARLA needed
python -m pytest tests/ -q                                                  # run the test suite (171 tests)
```

Every module is also independently runnable via `python -m src.<package>.<module>` from the project root (e.g. `python -m src.odd.rss_safety`, `python -m src.odd.fuzzy_odd`) — this `-m` convention is required (rather than `python src/odd/rss_safety.py`) because the codebase is organized as real importable subpackages; see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for why.

**Visualizing the road map / automation-readiness dashboard:** open `dashboard/index.html` directly in a browser (no server needed) — it works over plain `file://`. The road network drawn on the map is real OpenStreetMap geometry for central Chandigarh, not hand-drawn approximations ([`docs/DASHBOARD_ROAD_DATA.md`](docs/DASHBOARD_ROAD_DATA.md)); the per-road ODD-readiness *scores* painted onto it are synthetic demo values. Run `python -m src.decision.feasibility_map` first to populate its "Real Pipeline Results" panel with real IDD-Lite statistics, and/or `python -m src.simulation.closed_loop_runner --num_ticks 50 --force_fallback` to populate the "Stage 8: Closed-Loop Simulation Replay" panel and animate a moving marker (colored by Normal/Degraded/Takeover mode) along a road polyline. Re-open (or refresh) `index.html` after either command to see the updated data — both write plain `.js` files the dashboard reads at page load, no live connection needed.

## Modules

| Module | Stage | Purpose | Docs |
|---|---|---|---|
| `src/perception/data_pipeline.py` | 1 | IDD-Lite extraction, cleaning, loading | [doc](docs/modules/perception/data_pipeline.md) |
| `src/perception/segnet_model.py` | 2 | 3-stage BatchNorm SegNet (8-class semantic segmentation) | [doc](docs/modules/perception/segnet_model.md) |
| `src/perception/feature_extraction.py` | 2 | YOLOv8 detection + SegNet mask + image stats → 22 features per frame | [doc](docs/modules/perception/feature_extraction.md) |
| `src/perception/lane_detection.py` | 2 | Classical CV (Canny + Hough) lane/drivable-boundary estimation | [doc](docs/modules/perception/lane_detection.md) |
| `src/perception/detection_benchmark.py` | 2 | mAP/precision/recall/AP/AOS detection-evaluation toolkit | [doc](docs/modules/perception/detection_benchmark.md) |
| `src/perception/multi_stream_fusion.py` | 2 | Dynamic reliability-weighted fusion of the SegNet/YOLO/image-stat streams | [doc](docs/modules/perception/multi_stream_fusion.md) |
| `src/odd/traffic_density.py` | 3 | Vehicle-to-area ratio → Low/Medium/High/Congested | [doc](docs/modules/odd/traffic_density.md) |
| `src/odd/fuzzy_odd.py` | 4 | Fuzzy micro-ODD (NoRain/LowRain/HeavyRain) partitioning | [doc](docs/modules/odd/fuzzy_odd.md) |
| `src/odd/odd_boundary.py` | 5 | Structural Causal Model + Gaussian-copula ODD-space density, within/near/outside | [doc](docs/modules/odd/odd_boundary.md) |
| `src/odd/rss_safety.py` | 5 | RSS minimum safe distance + vehicle sideslip stability (standalone formulas) | [doc](docs/modules/odd/rss_safety.md) |
| `src/odd/risk_estimation.py` | 5 | Staged rare-event failure-probability estimation | [doc](docs/modules/odd/risk_estimation.md) |
| `src/monitoring/perturbation_engine.py` | 2/6 | GPU weather-perturbation engine (brightness/contrast, noise, fog) + regional profiles | [doc](docs/modules/monitoring/perturbation_engine.md) |
| `src/monitoring/perception_monitor.py` | 6 | Perturbation-sequence consistency monitoring → Nominal/Warning/Critical | [doc](docs/modules/monitoring/perception_monitor.md) |
| `src/odd/odd_classifier.py` | 7 | Rule-based labeling + RandomForest ODD classifier + cross-stage combination | [doc](docs/modules/odd/odd_classifier.md) |
| `src/decision/ecofusion_gate.py` | 7 | Energy/compute-aware adaptive branch selection | [doc](docs/modules/decision/ecofusion_gate.md) |
| `src/decision/feasibility_map.py` | 7 | Cross-stage Scenario-Based Feasibility Map | [doc](docs/modules/decision/feasibility_map.md) |
| `src/decision/sae_taxonomy.py` | — | SAE J3016 Level 0-5 classifier + ADS engagement state machine | [doc](docs/modules/decision/sae_taxonomy.md) |
| `src/pipeline/full_dataset_pipeline.py` | — | Batch, checkpointed/resumable version of Stages 2-7 | [doc](docs/modules/pipeline/full_dataset_pipeline.md) |
| `src/simulation/carla_config.py` | 8 | Tunable Stage 8 configuration (`configs/carla_config.json`) | [doc](docs/modules/simulation/carla_config.md) |
| `src/simulation/tick_source.py` | 8 | Shared per-tick interface for CARLA and the local fallback | [doc](docs/modules/simulation/tick_source.md) |
| `src/simulation/carla_bridge.py` | 8 | Real CARLA 0.9.15 client integration + CARLA-availability probe | [doc](docs/modules/simulation/carla_bridge.md) |
| `src/simulation/local_kinematic_sim.py` | 8 | No-CARLA fallback: real IDD-Lite frames + a kinematic stepper | [doc](docs/modules/simulation/local_kinematic_sim.md) |
| `src/simulation/closed_loop_runner.py` | 8 | Closed-loop simulation orchestrator (the primary Stage 8 entry point) | [doc](docs/modules/simulation/closed_loop_runner.md) |
| `main.py` | — | Orchestrates all 7 stages, plus optional Stage 8 via `--carla` | — |
| `dashboard/` | — | Leaflet.js map (real OpenStreetMap Chandigarh geometry, synthetic demo scoring) + real pipeline-results panel + Stage 8 replay marker | [doc](docs/DASHBOARD_ROAD_DATA.md) |

## A dataset-mapping correction found during development

An early one-shot LLM scaffold that seeded this project's first draft claimed IDD-Lite's 8 semantic classes were `0=Road, 1=Pothole, 2=Water/Puddle, 3=Sidewalk, 4=Vegetation, 5=Sky, 6=Vehicle, 7=Void`. Direct inspection of the pixel data (spatial distribution statistics, per-class blob shape statistics, and rendered image/label overlays) showed this is wrong. The verified scheme is:

```
0 = drivable-area   1 = non-drivable-area   2 = living-things (people/animals)
3 = vehicles        4 = road-side-objects   5 = far-objects (buildings/vegetation)
6 = sky             7 = unlabeled/void
```

There is **no pothole or water/puddle class at all** in this simplified "Lite" scheme. `feature_extraction.py`'s features were renamed accordingly (`pothole_score`→`non_drivable_area_score`, `water_level`→`living_things_score`) to describe what they actually measure, and a separate, explicitly heuristic (uncalibrated, no ground truth available) dark-blob pothole-candidate detector was added (`detect_pothole_candidates`) as a best-effort proxy. Full investigation writeup: [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md).

## Formula provenance

Every non-trivial formula in this codebase is cited to a specific paper and equation, verified by reading all 10 papers directly (not taken from the earlier one-shot scaffold's citations, two of which — a "6-Layer ODD Environment Model (Sun et al., IEEE ITS Magazine 2022)" and a "Risk = Severity × Exposure × Likelihood (Lee et al., IEEE IV 2020)" — could not be found in any of the 10 uploaded papers or their combined ~150 references, and are not used here). Full citation table with exact equation numbers: [`docs/FORMULA_PROVENANCE.md`](docs/FORMULA_PROVENANCE.md).

| Formula / method | Module | Source |
|---|---|---|
| SAE Level 0-5 taxonomy, DDT/ODD/MRC/RTI definitions | `sae_taxonomy.py` | SAE J3016 (Rev. APR2021) |
| Brightness/contrast & fog perturbation models, robust-learning framework | `perturbation_engine.py` | Sun, Cui, Ning, Lu, Cao, Khajepour, "Extending Operational Design Domain for Perception Systems Through Robust Learning," IEEE TIV, Oct 2024 |
| Fuzzy μODD (NoRain/LowRain/HeavyRain), membership functions, RSS parameter table | `fuzzy_odd.py`, `rss_safety.py` | Salvi, Weiss, Trapp, Oboril, Buerkle, "Fuzzy Interpretation of Operational Design Domains in Autonomous Driving," IEEE IV 2022 |
| RSS minimum safe distance | `rss_safety.py` | Salvi et al. 2022, Eq. 1 |
| Structural Causal Model, Gaussian-copula ODD density, sideslip stability + magic tire formula, Kriging-based Subset Simulation, 3-category ODD taxonomy | `odd_boundary.py`, `rss_safety.py`, `risk_estimation.py` | Jiang, Pan, Liu, Han, Pan, Li, Pan, "Enhancing Autonomous Vehicle Safety Based on Operational Design Domain Definition, Monitoring, and Functional Degradation," IEEE TIV, Oct 2024 |
| Dynamic multi-stream fusion weighting pattern | `multi_stream_fusion.py` | Sumalatha, Chaturvedi, R, Patil, Thethi, Hameed, "Autonomous Multi-sensor Fusion Techniques for Environmental Perception in Self-Driving Vehicles," IC3SE 2024 |
| mAP/precision/recall/AP/AOS/APH evaluation toolkit | `detection_benchmark.py` | Nawaz, Tang, Bibi, Xiao, Ho, Yuan, "Robust Cognitive Capability in Autonomous Driving Using Sensor Fusion Techniques: A Survey," IEEE T-ITS 2024 |
| Energy/compute-aware adaptive branch selection (early/late fusion, Pareto filtering, joint loss) | `ecofusion_gate.py` | Malawade, Mortlock, Al Faruque, "EcoFusion: Energy-Aware Adaptive Sensor Fusion for Efficient Autonomous Vehicle Perception," ACM/IEEE DAC 2022 |
| Fusion-level/calibration taxonomy (background/terminology only) | — | Yang, Li, Zeng, "A Review of Environmental Perception Technology Based on Multi-Sensor Information Fusion in Autonomous Driving," World Electric Vehicle Journal 2025 |
| ODD-constraining process, ODD-exit-monitor taxonomy (background/terminology only) | — | Hillen, Lorenz, Reich, Adler, Wolf, Zafar, Salvi, "Navigating the Landscape of Operational Design Domains: A Comprehensive Mapping Study," Next Research 2025 |

## Known scope limitations

- **CARLA closed-loop simulation is implemented, with an automatic no-CARLA fallback — not, itself, run against a real CARLA server in this environment.** `src/simulation/` (Stage 8, optional) drives `rss_safety.py`'s RSS-distance and sideslip-stability formulas from live per-tick vehicle kinematics, either from a real CARLA 0.9.15 server (`carla_bridge.py`, correct against CARLA's documented API but **not execution-verified here**) or, automatically when CARLA isn't attached, from a local kinematic stepper driving real IDD-Lite frames (`local_kinematic_sim.py`, fully tested). This machine has no discrete GPU (CARLA's server needs one even headless) and the `carla` PyPI package has no wheel for this project's Python 3.12 venv — both confirmed directly, not assumed. See [`docs/FUTURE_STEPS.md`](docs/FUTURE_STEPS.md) for exactly what a capable machine needs to exercise the CARLA-attached path, and `docs/modules/simulation/` for the full design.
  > Note on the proposal itself: `research-papers/CPG274_Proposal_updated.docx` does not actually specify CARLA as part of this project's own methodology — it only cites CARLA in literature-survey summaries of three *other* papers' validation setups (Jiang et al., Hasanujjaman et al., Salvi et al.); the proposal's own plan commits only to generic "simulation based evaluation." The detailed CARLA spec this Stage 8 implementation is grounded in — real API usage, per-mode physics/weather profiles, a capture-classify-actuate loop — was sketched (with unverified constants) in `Capstone_OneShot_Prompt.txt`, the non-authoritative scaffold this project's first pass started from; Stage 8 reimplements that idea using this project's own already-verified `rss_safety.py`/`sae_taxonomy.py` formulas instead.
- **IDD-Lite only, no KITTI.** The capstone proposal also lists KITTI as an input dataset; it's skipped here since it doesn't test this project's India-specific novelty claim.
- **Stage 6 monitoring uses a perturbation-sequence proxy**, not true video: IDD-Lite is a static-image dataset, so "consecutive frames" are simulated via repeated perturbed versions of the same image.
- **`detect_pothole_candidates()` is an uncalibrated heuristic**, not a trained/validated pothole detector — see [`docs/DATASET_NOTES.md`](docs/DATASET_NOTES.md).
- **AOS/APH** (orientation-similarity metrics in `detection_benchmark.py`) are implemented and unit-tested exactly per their source formulas but not used in `evaluate_vehicle_detections()`, since this project's 2D bounding boxes carry no heading/orientation ground truth.
- **Full-dataset feasibility maps assume a "Nominal" Stage 6 status** per row rather than running the (expensive) per-frame perturbation-sequence monitor at batch scale; call `perception_monitor.py` directly for a single scene's true monitoring state.

## Team

| Name | Role |
|---|---|
| Aryaman Gudwani | Project Lead & ODD Framework |
| Vipul Sati | Perception Model & Training |
| Mansehaj Preet Singh | Perturbation Model & Testing |
| Aishlee Joshi | Scenario Map & Perception Model |
| Shree Mishra | Design, Documentation & Deployment |

Mentors: Dr. V.P. Singh, Dr. Souvik Ganguli.
