# Setup

## Prerequisites

- Python 3.10+ (developed/tested on 3.12).
- ~64 GB free disk is comfortable but not required; the actual footprint (venv + dataset + checkpoints + outputs) is well under 1 GB.
- No GPU required. This project was built and verified entirely on a CPU-only machine (Intel iGPU, no NVIDIA GPU); every module auto-detects `torch.cuda.is_available()` and prints a warning + falls back to CPU where relevant (`segnet_model.py`, `perturbation_engine.py`, `feature_extraction.py`, `odd_classifier.py`, `full_dataset_pipeline.py`, `ecofusion_gate.py`).

## Environment

```bash
cd capstone_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU-only torch/torchvision wheels FIRST -- installing torch via the
# default PyPI index pulls a multi-GB chain of unused CUDA dependency
# packages on a machine with no NVIDIA GPU.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Everything else:
pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "
import torch, torchvision, cv2, numpy, pandas, sklearn, matplotlib, seaborn, joblib, tqdm, PIL
print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
from ultralytics import YOLO
m = YOLO('yolov8n.pt')   # triggers a first-run download into the cwd; move/copy it to models/ afterward
print('ultralytics OK, classes:', len(m.names))
"
```

`ultralytics` downloads `yolov8n.pt` into the current directory on first use if it isn't found; move it to `models/yolov8n.pt` (or run the command from inside `models/`) so `src/common/paths.py::YOLO_WEIGHTS` finds it.

## Dataset

Copy or symlink the IDD-Lite dataset so the layout looks like:

```
capstone_project/data/idd20k_lite/
├── leftImg8bit/{train,val,test}/*/*_image.jpg
└── gtFine/{train,val}/*/*_label.png   (+ *_inst_label.png, see docs/DATASET_NOTES.md)
```

`src/perception/data_pipeline.py::extract_dataset()` can also extract a `data/idd-lite.tar.gz` archive into this layout directly, if you have the dataset as a tarball rather than already-extracted.

## First run

```bash
python main.py
```

On a clean checkout (no cached artifacts in `models/`/`outputs/`) this will: clean the dataset, train SegNet for 30 epochs (~35 minutes on a modern CPU, based on this project's own build — the run prints per-epoch timing so you can gauge whether to interrupt and reduce `epochs` for iteration), download/run YOLOv8n over the dataset, fit all Stage 3-7 models, and write a Scenario-Based Feasibility Map plus a dashboard data file. Subsequent runs skip SegNet training and feature extraction automatically (cached in `models/refined_segnet.pth` and `outputs/final_features.csv`) unless you pass `--force`.

## Running tests

```bash
python -m pytest tests/ -q
```

171 tests total. Tests that need the trained SegNet checkpoint, YOLO weights, or the extracted feature CSV will `pytest.skip()` gracefully (with a clear reason) if those artifacts don't exist yet — run `python main.py` at least once first for full coverage.

## Viewing the dashboard

```bash
python -m src.decision.feasibility_map   # regenerates dashboard/pipeline_stats.js from current outputs/final_features.csv
python -m src.simulation.closed_loop_runner --num_ticks 50 --force_fallback   # regenerates dashboard/carla_live.js
```

Then open `dashboard/index.html` directly in a browser — no local server needed (see `dashboard/app.js`'s module docstring for why: all data is inlined as JS, not fetched, since `fetch()` of local files is blocked by browsers under `file://`). The sidebar's "Real Pipeline Results" panel reflects the first command's output; the "Stage 8: Closed-Loop Simulation Replay" panel and the moving colored marker on the map reflect the second's — re-run either command and refresh the page to see updated data.

## Optional: CARLA closed-loop simulation

`src/simulation/` (Stage 8) runs without CARLA out of the box — `python -m src.simulation.closed_loop_runner --force_fallback` needs nothing beyond what's already set up above. This section is only for actually exercising the real CARLA-attached path (`src/simulation/carla_bridge.py`).

**Hardware requirements:** a discrete GPU (NVIDIA recommended; CARLA's server renders even in headless/off-screen mode) and roughly 20-30 GB of free disk for the CARLA server download. This project's own development machine has neither (Intel iGPU only), so the CARLA-attached path is implemented and documented but not itself execution-verified here — see `docs/modules/simulation/carla_bridge.md` and `docs/FUTURE_STEPS.md`.

**Why a separate venv is required:** the official `carla` PyPI wheels (0.9.13-0.9.15) only support CPython 3.7-3.10 on Linux. This project's main `.venv` is Python 3.12 (see Prerequisites above) — there is no version overlap, so `carla` cannot be `pip install`-ed into the main venv at all.

```bash
# On a machine with a discrete GPU:
# 1. Download and run a CARLA 0.9.15 server (see carla.org/download) --
#    e.g. ./CarlaUE4.sh -quality-level=Low  (or -RenderOffScreen for headless)

# 2. In this project, set up a SECOND, separate venv just for the CARLA client:
python3.10 -m venv .venv-carla
source .venv-carla/bin/activate
pip install -r requirements-carla.txt   # carla==0.9.15
pip install -r requirements.txt         # this venv also needs the project's other deps
                                         # (torch, opencv, etc.) to call into
                                         # feature_extraction.py / odd_classifier.py / rss_safety.py

# 3. Point configs/carla_config.json at the server (defaults to localhost:2000;
#    edit "host"/"port" to target a remote GPU-capable machine instead) and set
#    "use_carla": true, then:
python -m src.simulation.closed_loop_runner --num_ticks 50
# Falls back to the local kinematic simulator automatically, with a printed
# warning, if the server isn't reachable -- this is the same code path
# already exercised in the main venv above.
```

See `docs/FUTURE_STEPS.md` for further CARLA-side extension ideas (multi-vehicle scenarios, RL-tuned physics profiles, etc.).
