# `full_dataset_pipeline.py`

**Stage:** Cross-cutting batch orchestration (README lists it without a stage number: "Batch, checkpointed/resumable version of Stages 2-7")
**Package:** `src.pipeline.full_dataset_pipeline`

## Purpose

This module is the batch-scale, resumable counterpart to running Stages 2-7 image-by-image. It streams every image in the IDD-Lite (or a larger IDD-20K-scale) dataset from disk in configurable batches, runs SegNet segmentation and YOLO detection on each, computes the same 18 raw ODD features `feature_extraction.py` computes per-frame, scales them and assigns a rule-based ODD mode, and writes one labeled row per image to an output CSV — while checkpointing progress to disk every N images so a long-running full-dataset pass can be safely interrupted (crash, manual stop, out-of-memory, etc.) and resumed later without reprocessing images that already finished.

It exists because the interactive, in-memory pipeline (`main.py` / `feature_extraction.py`) is designed for the curated IDD-Lite subset (2011 images) that fits comfortably in memory; this module is built for scaling that same feature-extraction + labeling process up to the much larger full IDD-20K dataset, where a single uninterrupted run may not be practical.

## Paper / Formula Provenance

This module does not introduce new formulas of its own — it is an orchestration/checkpointing wrapper around already-cited Stage 2 and Stage 7 logic: `feature_extraction.compute_features()` (Stage 2, the 18/22-feature extraction pipeline — see `feature_extraction.py`'s own provenance for its per-feature formulas), `feature_extraction.run_detection()` (YOLO detection), `perturbation_engine.calculate_metrics()` (mIoU/accuracy, itself following Sun et al. 2024's evaluation conventions — see `perturbation_engine.py`'s docs), and `odd_classifier.assign_mode()` (the Stage 7 rule-based Normal/Degraded/Takeover labeling rule). No paper equation is implemented directly in this file; its own logic (batching, checkpointing, resuming) is pure engineering, not a cited formula.

## Public API

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
def discover_images(data_dir: str) -> List[Tuple[str, Optional[str]]]
```
Discovers every image in the dataset, paired with its label path if one exists. Globs `leftImg8bit/{train,val,test}/*/*_image.jpg` under `data_dir`, and for each image derives the corresponding label path by replacing `leftImg8bit` → `gtFine` and `_image.jpg` → `_label.png`; the label is set to `None` if that file doesn't actually exist. Returns a sorted list of `(image_path, label_path_or_None)` tuples across all three splits. Only `train`/`val` splits have ground-truth masks in practice; `test` images will typically pair with `None`.

```python
def run_full_dataset_pipeline(
    data_dir: str,
    segnet_checkpoint: str,
    yolo_weights: str,
    feature_scaler_path: str,
    batch_size: int = 32,
    checkpoint_interval: int = 500,
    checkpoint_path: str = "full_dataset_checkpoint.csv",
    output_path: str = "full_dataset_labeled.csv",
) -> pd.DataFrame
```
The main entry point. Extracts features and an ODD mode label for every image in `data_dir`, with resumable checkpointing. Args:
- `data_dir` — path to the `idd20k_lite/` (or larger IDD-20K) directory.
- `segnet_checkpoint` — path to a trained SegNet `.pth` file.
- `yolo_weights` — path to YOLOv8 weights (e.g. `yolov8n.pt`).
- `feature_scaler_path` — path to the `FeatureScaler` saved by `odd_classifier.py` (`feature_scaler.pkl`). **This module must run after `odd_classifier.py` has produced this file at least once**, since `assign_mode()` operates on MinMax-scaled features and each image's raw features are scaled through this same fitted scaler before labeling.
- `batch_size` — number of images loaded/inferred per inner batch (default `32`).
- `checkpoint_interval` — flush progress to disk every N processed images (default `500`).
- `checkpoint_path` — where in-progress results are checkpointed (default `"full_dataset_checkpoint.csv"`).
- `output_path` — final output CSV path once all images are processed (default `"full_dataset_labeled.csv"`).

Returns a DataFrame with one row per image: the 18 raw features, `'mode'`, `'image_path'`, `'has_mask'`, and `'miou'` (`NaN` where no ground-truth mask exists).

**Private helper** (not public API, but central to the resume mechanism): `_load_checkpoint(checkpoint_path) -> pd.DataFrame` — loads an in-progress checkpoint CSV, or returns an empty DataFrame if the file doesn't exist yet.

```python
def _parse_args() -> argparse.Namespace
```
Parses CLI arguments for standalone execution: `--data_dir` (default `src.common.paths.DATA_DIR`), `--batch_size` (default `32`), `--checkpoint_interval` (default `500`).

## Key Design Decisions & Edge Cases

- **The resume-on-restart mechanism is the module's defining feature.** On every call, `run_full_dataset_pipeline()` first calls `discover_images()` to enumerate the *entire* dataset, then loads whatever checkpoint CSV already exists via `_load_checkpoint()` and builds `done_paths = set(done_df["image_path"])`. It then computes `remaining = [p for p in pairs if p[0] not in done_paths]` — i.e. the resume logic is a simple set-membership filter against the `image_path` column already written to the checkpoint CSV, not a byte-offset or index-based resume. This means: (a) a completed image is never reprocessed, no matter how many times the run is interrupted and restarted; (b) the checkpoint CSV's `image_path` column is the single source of truth for "what's already done"; (c) resuming works correctly even if `batch_size` changes between runs, since progress is tracked per-image, not per-batch.
- **Checkpointing is periodic, not per-image.** Rows accumulate in an in-memory `buffer_rows` list and are only flushed to `checkpoint_path` once `len(buffer_rows) >= checkpoint_interval` (via the local `_flush()` closure, which reads the existing checkpoint, concatenates the new rows, and rewrites the whole file). A larger `checkpoint_interval` means less disk I/O overhead per run but a larger amount of *unsaved* work lost if the process is killed between flushes — this is an explicit throughput/safety trade-off exposed directly as a CLI flag.
- **A final flush always happens** (`_flush()` is called unconditionally after the main loop, in addition to the periodic in-loop flushes), so partial batches smaller than `checkpoint_interval` are never silently dropped at the end of a run.
- **On successful completion, the checkpoint file is deleted.** After writing `output_path`, `os.remove(checkpoint_path)` runs if the checkpoint file exists — a completed run leaves only the final output CSV, not a stale checkpoint that could confuse a later invocation into thinking a *different* dataset/config was already partially processed.
- **Corrupted/unreadable images are skipped, not fatal.** Inside the batch-loading loop, `cv2.imread(img_path)` returning `None` triggers a printed warning and a `continue` — the image is simply excluded from that run's output (and, since it's never added to the checkpoint, it will be retried on the next run rather than permanently skipped).
- **Label maps are sanitized before mIoU computation.** Ground-truth labels are resized with `cv2.INTER_NEAREST` (avoiding interpolated/invalid class values), then `lbl[lbl == 255] = VOID_LABEL` and `lbl[lbl > VOID_LABEL] = VOID_LABEL` clamp any out-of-range or sentinel pixel values into the valid 8-class void bucket before computing mIoU via `perturbation_engine.calculate_metrics()`.
- **`miou` is `NaN`, not 0, for images without a ground-truth mask** (`lbl_path is None`) — correctly distinguishing "no ground truth available" from "zero overlap with ground truth" in downstream analysis (e.g. `.mean()` over the `miou` column naturally ignores `NaN` rows).
- **End-of-run diagnostics.** After saving, the function prints total images processed, the `mode` value-count distribution, mean mIoU restricted to rows that actually had a ground-truth mask (explicitly guarded against the `len(masked) == 0` case with a "No images had ground-truth masks" message instead of dividing by zero / crashing), and — if any `"Takeover"`-labeled rows exist — the top 5 highest-mean numeric feature values among them, as a quick sanity/debugging signal for what's driving Takeover labels at scale.

## Dependencies

**Internal:** `src.perception.data_pipeline` (`IMAGE_SIZE`, `VOID_LABEL`), `src.perception.feature_extraction` (`compute_features`, `run_detection`), `src.odd.odd_classifier` (`FeatureScaler`, `assign_mode` — the module comment notes `FeatureScaler` is imported specifically because `joblib.load()` needs the class definition in scope to unpickle a saved scaler, even though it isn't referenced by name elsewhere in this file, hence the `# noqa: F401`), `src.monitoring.perturbation_engine.calculate_metrics`, `src.perception.segnet_model.load_segnet`. The `if __name__ == "__main__":` block additionally imports `src.common.paths` (`FEATURE_SCALER_PATH`, `FULL_DATASET_CHECKPOINT_CSV`, `FULL_DATASET_LABELED_CSV`, `SEGNET_CHECKPOINT`, `YOLO_WEIGHTS`, `ensure_output_dirs`).

**External:** `argparse`, `glob`, `os` (stdlib); `cv2`, `joblib`, `numpy`, `pandas`, `torch`, `tqdm`, `ultralytics.YOLO`.

## Usage Example

```python
from src.pipeline.full_dataset_pipeline import run_full_dataset_pipeline

df = run_full_dataset_pipeline(
    data_dir="data/idd20k_lite",
    segnet_checkpoint="models/refined_segnet.pth",
    yolo_weights="models/yolov8n.pt",
    feature_scaler_path="models/feature_scaler.pkl",
    batch_size=32,
    checkpoint_interval=500,
    checkpoint_path="outputs/full_dataset_checkpoint.csv",
    output_path="outputs/full_dataset_labeled.csv",
)
print(df["mode"].value_counts())
```

If the process is interrupted partway through (e.g. `Ctrl+C`, crash, machine reboot), simply calling `run_full_dataset_pipeline()` again with the **same `checkpoint_path`** picks up exactly where it left off — already-completed image paths found in the checkpoint CSV are skipped automatically.

## Running Standalone

```bash
python -m src.pipeline.full_dataset_pipeline --batch_size 32 --checkpoint_interval 500
```

Also accepts `--data_dir` to point at a different dataset directory (defaults to `src.common.paths.DATA_DIR`). Running it: calls `ensure_output_dirs()`, then `run_full_dataset_pipeline()` using `src.common.paths` defaults for the SegNet checkpoint, YOLO weights, feature scaler, checkpoint CSV (`outputs/full_dataset_checkpoint.csv`), and output CSV (`outputs/full_dataset_labeled.csv`) — processing the full dataset in batches, printing a `tqdm` progress bar over batches, periodically checkpointing, and finally printing the class-distribution / mIoU / Takeover-diagnostics summary described above. Re-running the same command after an interruption resumes automatically rather than starting over.

## Tests

No dedicated test file exists for this module. There is no `tests/pipeline/test_full_dataset_pipeline.py` — the `tests/pipeline/` directory contains only an `__init__.py`. This module's constituent logic (feature extraction, mIoU computation, ODD mode assignment) is covered indirectly by the test suites of the modules it calls (`tests/perception/`, `tests/monitoring/test_perturbation_engine.py`, `tests/odd/`), but the batching/checkpointing/resume orchestration in this file itself has no direct unit-test coverage.
