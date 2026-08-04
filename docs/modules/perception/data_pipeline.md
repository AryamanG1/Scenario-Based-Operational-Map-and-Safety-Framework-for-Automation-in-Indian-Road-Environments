# `data_pipeline.py`

**Stage:** 2 (Perception Layer) — labeled Stage 1 ("Input Data") in the top-level README's pipeline diagram, but lives in the `perception` package
**Package:** `src.perception.data_pipeline`

## Purpose

This module is the entry point of the entire 7-stage pipeline: it extracts the IDD-Lite (Indian Driving Dataset Lite) archive, pairs every road-scene image with its semantic segmentation mask, filters out corrupted or poorly-labeled samples, resizes everything to a common shape, and returns clean numpy arrays ready for SegNet training and downstream perception processing.

It exists because IDD-Lite ships as a raw `.tar.gz` archive with per-city subfolders and occasional corrupted or degenerate label files; every other perception module (SegNet training, feature extraction, lane detection, detection benchmarking) assumes it receives already-cleaned, uniformly-sized `(images, labels)` numpy arrays, so this module is the single place that dataset-quality filtering happens.

## Paper / Formula Provenance

This module contains no formulas and is not listed in the README's "Formula provenance" table. It is original data-engineering code (archive extraction, quality filtering, resizing) with no citation — the README's provenance table only covers modules with non-trivial cited formulas, and this module has none.

## Public API

### Constants

```python
IMAGE_SIZE = (320, 224)  # (width, height)
NUM_CLASSES = 8
VOID_LABEL = 7
```

### `extract_dataset`

```python
def extract_dataset(tar_path: str, extract_to: str) -> str
```

Extracts the IDD-Lite `tar.gz` archive.

- **Args:** `tar_path` — path to the `idd-lite.tar.gz` archive; `extract_to` — directory to extract the archive into.
- **Returns:** the path to the extracted `idd20k_lite/` directory (i.e. `os.path.join(extract_to, "idd20k_lite")`).
- **Raises:** `FileNotFoundError` if `tar_path` does not exist.
- Creates `extract_to` if it doesn't exist (`os.makedirs(extract_to, exist_ok=True)`) and prints a confirmation line after extraction.

### `load_and_clean_dataset`

```python
def load_and_clean_dataset(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray]
```

Loads, filters, and cleans the IDD-Lite train split.

- **Args:** `dataset_dir` — path to the `idd20k_lite/` directory (containing `leftImg8bit/` and `gtFine/` subdirectories).
- **Returns:** `(images, labels)` where `images` is a `uint8` array of shape `(M, 224, 320, 3)` and `labels` is a `uint8` array of shape `(M, 224, 320)`. `M` is the number of samples that survived filtering.
- Globs `leftImg8bit/train/*/*_image.jpg`, and for each image derives its label path by string-replacing `"leftImg8bit"` → `"gtFine"` and `"_image.jpg"` → `"_label.png"`.
- Drops a sample (incrementing an internal `removed` counter) if any of the following hold:
  - `cv2.imread` returns `None` for either the image or the label (unreadable/corrupted file).
  - The label has `<= 1` unique class value (near-empty/degenerate label).
  - Class `0` (road) is not present anywhere in the label.
- Survivors are resized to `IMAGE_SIZE` with `cv2.resize` for the image (default bilinear-ish interpolation) and `cv2.INTER_NEAREST` for the label (nearest-neighbor, so label resizing never invents intermediate/blended class values).
- Label remapping: pixel value `255` and any value `> VOID_LABEL` (7) are remapped to `VOID_LABEL`.
- Prints `f"Removed {removed} improper/corrupted files. Clean dataset size: {len(clean_images)}"` before returning.

## Key Design Decisions & Edge Cases

- **Nearest-neighbor resizing for labels only.** Images use `cv2.resize`'s default interpolation; labels explicitly use `cv2.INTER_NEAREST` so class IDs are never blended into invalid fractional/interpolated values during resize.
- **Two-step void remapping.** `lbl_resized[lbl_resized == 255] = VOID_LABEL` catches the conventional "ignore" sentinel value used in many segmentation datasets, and `lbl_resized[lbl_resized > VOID_LABEL] = VOID_LABEL` is a defensive catch-all for any other out-of-range value, both collapsed into a single `VOID_LABEL = 7` class.
- **Quality filters are conservative but not exhaustive.** A sample only needs road pixels to be present and more than one unique class value to survive — this filters obviously broken files but does not validate label correctness beyond that (that deeper validation is what produced the dataset-mapping correction described in `feature_extraction.py` and the top-level README).
- **`.copy()` before in-place remapping.** `lbl_resized = lbl_resized.copy()` guards against mutating a view returned by `cv2.resize` before the remapping assignments.
- The function operates only on the `train` split (`leftImg8bit/train/*/*_image.jpg`); there is no separate val/test loading path in this module.

## Dependencies

- **Standard library:** `glob`, `os`, `tarfile`, `typing.Tuple`
- **External:** `cv2` (OpenCV), `numpy`
- **Internal:** none at import time; the `__main__` block imports `src.common.paths.DATA_DIR`.

## Usage Example

```python
from src.perception.data_pipeline import extract_dataset, load_and_clean_dataset

dataset_dir = extract_dataset("data/idd-lite.tar.gz", "data/")
images, labels = load_and_clean_dataset(dataset_dir)
print(images.shape, labels.shape)  # e.g. (1998, 224, 320, 3) (1998, 224, 320)
```

## Running Standalone

```bash
python -m src.perception.data_pipeline
```

Loads and cleans the dataset from `src.common.paths.DATA_DIR` (`data/idd20k_lite`) and prints the resulting `images`/`labels` array shapes, dtypes, and the sorted list of unique label classes present.

## Tests

`tests/perception/test_data_pipeline.py` covers: the three module constants (`IMAGE_SIZE`, `NUM_CLASSES`, `VOID_LABEL`); that the session-scoped `small_images_labels` fixture (a 10-image slice of the real cleaned dataset, skipped if `data/idd20k_lite` is absent) produces the correct shapes/dtypes; that all label values fall within `[0, VOID_LABEL]`; and an `extract_dataset` round-trip test (skipped if `data/idd-lite.tar.gz` is not present) that checks the extracted directory contains `leftImg8bit/` and `gtFine/` subdirectories.
