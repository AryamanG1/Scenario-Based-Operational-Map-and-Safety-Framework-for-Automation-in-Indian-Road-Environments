# Dataset Notes: IDD-Lite

## Source

IDD-Lite (India Driving Dataset — Lite variant), used via `data/idd20k_lite/`:
- `leftImg8bit/{train,val,test}/*/*_image.jpg` — RGB road-scene images (1403 train / 204 val / 404 test = 2011 total).
- `gtFine/{train,val}/*/*_label.png` — semantic segmentation masks (train+val only; test has no ground truth, as expected for a held-out split).
- `gtFine/{train,val}/*/*_inst_label.png` — **verified byte-identical to `_label.png`** (see below). Not a real instance map.

Images are resized to 320×224 (width × height) throughout the pipeline (`src/perception/data_pipeline.py::IMAGE_SIZE`).

## The class-mapping correction (important project history)

The project's first implementation pass was seeded from a one-shot LLM scaffold that asserted the following semantic class scheme:

```
0=Road  1=Pothole  2=Water/Puddle  3=Sidewalk  4=Vegetation  5=Sky  6=Vehicle  7=Void
```

This is **wrong**. It was caught during Stage 2 development (building `detection_benchmark.py`, which needed to know the true "vehicle" class ID) by direct inspection of the pixel data:

1. **Spatial distribution statistics** — for each class ID, the mean normalized row position (0=top, 1=bottom) and mean area fraction were computed across 100 sample masks. Class 6 came out with `mean_row_frac=0.120` — the topmost class in the entire scheme, even above the class that turned out to be Sky (`mean_row_frac=0.273`). A "Vehicle" class being located above the sky is physically impossible.
2. **Blob shape statistics** — connected-component analysis showed class 2 blobs have `mean_aspect_w/h=0.57` (taller than wide — consistent with human silhouettes), and class 3 blobs are numerous (~6/image) with near-square aspect ratios (consistent with vehicles).
3. **Direct visual confirmation** — rendering actual image/label pairs with a color-coded overlay showed unambiguously that class 3 pixels sit exactly on cars and motorcycles, and class 2 pixels sit exactly on the people riding them.

The verified real scheme (matching the AutoNUE/IDD `level3Id` semantics) is:

```
0 = drivable-area          1 = non-drivable-area       2 = living-things (people/animals)
3 = vehicles                4 = road-side-objects       5 = far-objects (buildings/vegetation)
6 = sky                     7 = unlabeled/void
```

**There is no dedicated pothole or water/puddle class anywhere in this 8-class scheme.** The original scaffold's claimed classes 1 and 2 do exist as classes, but measure something else entirely.

### Consequence for `feature_extraction.py`

Two of the original 18 spec'd features were computing correct pixel-fraction *math* against the correct class IDs — the class IDs `1` and `2` were coincidentally right — but were **named for the wrong physical quantity**:

| Old (wrong) name | New (honest) name | What it actually measures |
|---|---|---|
| `pothole_score` | `non_drivable_area_score` | Fraction of frame classified as non-drivable-area (curb/sidewalk-like), class 1 |
| `pothole_distance` | `non_drivable_area_distance` | Distance from bottom of frame to nearest non-drivable-area pixel |
| `water_level` | `living_things_score` | Fraction of frame classified as living-things (pedestrians/animals), class 2 |

Because the underlying pixel math was unchanged (only the class-ID *interpretation* was wrong, and class IDs 0/1/2/3 all happened to already be read correctly by `ROAD_CLASS_ID=0`), this rename did **not** change any numeric feature values or downstream ODD classification results — verified by re-running the full pipeline before and after the fix and confirming byte-identical mode distributions. What *did* need a real numeric fix was `detection_benchmark.py`'s `VEHICLE_CLASS_ID`, which the scaffold had wrong as `6` (sky) instead of `3` (vehicles) — this one was a genuine bug, not just a naming issue, and produced AP=0.0 until corrected.

A separate, explicitly heuristic pothole-candidate detector (`feature_extraction.py::detect_pothole_candidates`) was added afterward as a best-effort proxy: it flags dark, compact, irregular blobs within the drivable-area region as pothole *candidates*. It is **uncalibrated** (there is no ground-truth pothole label anywhere in IDD-Lite to validate against) and is expected to also fire on shadows, oil stains, and manhole covers. Treat its two output columns (`pothole_heuristic_score`, `pothole_heuristic_count`) as a weak, exploratory signal, not a validated detector.

### Consequence for `detection_benchmark.py`

Ground-truth vehicle bounding boxes for benchmarking YOLO's detection accuracy are derived from **connected-component analysis of the semantic mask's class-3 (vehicles) blobs** (`extract_pseudo_gt_boxes`), not from `_inst_label.png` (confirmed to carry no real instance information — see below). This under-counts touching/overlapping vehicles (they merge into one blob) and is documented as a real data limitation in the module's own docstring, not a hidden approximation.

## `_inst_label.png` carries no instance information

Direct byte comparison (`numpy.array_equal`) confirmed `_inst_label.png` is **identical** to `_label.png` for every sample checked — IDD-Lite's "Lite" variant does not ship real per-object instance IDs, only the same semantic-level classes duplicated under a different filename. This ruled out an initially-planned approach of extracting true per-vehicle instance boxes from instance IDs; the connected-component fallback above was used instead.

## Lessons this encodes for the rest of the codebase

Every class-ID-dependent constant in this codebase (`ROAD_CLASS_ID`, `NON_DRIVABLE_CLASS_ID`, `LIVING_THINGS_CLASS_ID`, `VEHICLE_CLASS_ID`, `SKY_CLASS_ID` in `src/perception/feature_extraction.py`) is defined against the **verified** scheme above, with a comment block explaining the correction. If you extend this project to a different IDD variant or a different dataset entirely, re-verify the class scheme the same way (spatial stats + visual rendering) before trusting any label-derived feature — do not assume a class list from a spec document or prompt is correct without checking it against the actual pixel data.
