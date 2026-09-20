"""IDD117K-Detection loader: image/annotation pairing, box parsing, sampling.

IDD117K-Detection ships **bounding boxes only** -- there are no semantic
segmentation masks anywhere in it. This module therefore does NOT replace
`data_pipeline.py` (which loads IDD-Lite's `*_label.png` masks and remains the
only source SegNet can train on). It sits alongside it and supplies:

  1. Real, human-annotated ground-truth boxes for `detection_benchmark.py`,
     replacing the connected-component pseudo-boxes derived from SegNet's
     mask (documented in that module as a real data limitation: it merges
     touching vehicles into one blob).
  2. A large image corpus for `feature_extraction.py`, streamed from disk
     rather than materialized as one array -- 96,897 train images at
     320x224x3 is ~20.8 GB before labels, which will not fit in RAM.

Directory layout (verified against the real archive for IDD_95kDetection;
the older IDD_Detection part is auto-detected, see `_detect_layout`):

    IDD117K_Detection/
      IDD_95kDetection/
        train/  leftImg8bit/<seq>/<frame>.jpg  Labels_json/<seq>/<frame>.json
        val/    leftImg8bit/<seq>/<frame>.jpg  Labels_json/<seq>/<frame>.json
        test/   leftImg8bit/<seq>/<frame>.jpg   (no labels -- held out)

Annotation format is a flat JSON list of absolute-pixel boxes on the native
1920x1080 image:

    [{"name": "autorickshaw", "bbox": {"xmin": 1, "ymin": 423,
                                       "xmax": 204, "ymax": 758}}, ...]

**Class-vocabulary warning.** `docs/DATASET_NOTES.md` records that this project
already shipped one wrong class mapping (an LLM-scaffolded guess that placed
"Vehicle" above "Sky" in the frame), caught only by direct inspection of the
data, and it ends with an explicit instruction to re-verify the scheme against
the real data before trusting any label-derived feature. The `NAME_TO_GROUP`
table below is therefore treated as a **draft**: run

    python -m src.perception.idd_detection_loader --scan

to enumerate the actual label vocabulary before relying on it. Any name the
scan finds that is not in the table is reported by `unmapped_labels()` instead
of being silently dropped.
"""

import argparse
import collections
import glob
import json
import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# (width, height) -- must match data_pipeline.IMAGE_SIZE so that boxes scaled
# here line up with the images every downstream module already works on.
IMAGE_SIZE = (320, 224)

Box = Tuple[int, int, int, int]  # (x, y, w, h) -- matches detection_benchmark.Box

# --- Label vocabulary (DRAFT -- verify with --scan before trusting) -----------
# Groups mirror the category buckets feature_extraction.py already uses, so the
# two modules speak the same language. 'autorickshaw' and 'animal' are REAL
# annotated classes here, unlike in IDD-Lite -- which is what finally makes the
# uncalibrated AUTORICKSHAW_* bbox heuristic in feature_extraction.py testable.
GROUP_VEHICLE = "vehicle"            # COCO-comparable four-wheelers
GROUP_TWO_WHEELER = "two_wheeler"
GROUP_AUTORICKSHAW = "autorickshaw"
GROUP_PEDESTRIAN = "pedestrian"
GROUP_ANIMAL = "animal"
GROUP_TRAFFIC_CONTROL = "traffic_control"
GROUP_STATIC = "static"
GROUP_IGNORE = "ignore"

NAME_TO_GROUP: Dict[str, str] = {
    "car": GROUP_VEHICLE,
    "bus": GROUP_VEHICLE,
    "truck": GROUP_VEHICLE,
    "caravan": GROUP_VEHICLE,
    "trailer": GROUP_VEHICLE,
    "train": GROUP_VEHICLE,
    "vehicle fallback": GROUP_VEHICLE,
    "motorcycle": GROUP_TWO_WHEELER,
    "bicycle": GROUP_TWO_WHEELER,
    "autorickshaw": GROUP_AUTORICKSHAW,
    "person": GROUP_PEDESTRIAN,
    "rider": GROUP_PEDESTRIAN,
    "animal": GROUP_ANIMAL,
    "traffic sign": GROUP_TRAFFIC_CONTROL,
    "traffic light": GROUP_TRAFFIC_CONTROL,
    "curb": GROUP_STATIC,
    "wall": GROUP_STATIC,
    "fence": GROUP_STATIC,
    "guard rail": GROUP_STATIC,
    "billboard": GROUP_STATIC,
    "bridge": GROUP_STATIC,
    "obs-str-bar-fallback": GROUP_STATIC,
    # The ego vehicle is the camera car's own hood: it is present in nearly
    # every frame and is not a detectable object. Dropped, like IDD-Lite's
    # void class.
    "ego vehicle": GROUP_IGNORE,
    "unlabeled": GROUP_IGNORE,
    "out of roi": GROUP_IGNORE,
    "rectification border": GROUP_IGNORE,
    "license plate": GROUP_IGNORE,
}

# The GT group compared against YOLO's VEHICLES = ["car", "bus", "truck"].
# Matching the COCO class set exactly avoids the apples-to-oranges comparison
# the mask-based benchmark was forced into (it scored COCO four-wheelers
# against IDD's *whole* vehicles class, which also contains motorcycles,
# bicycles and autorickshaws, inflating the false-negative count).
DEFAULT_GT_GROUPS = (GROUP_VEHICLE,)


def _detect_layout(split_dir: str) -> Tuple[str, str, str]:
    """Identifies the image dir, annotation dir, and annotation format.

    The 95k part uses `leftImg8bit/` + `Labels_json/`. The older 2018
    IDD_Detection part is distributed with Pascal VOC `Annotations/` +
    `JPEGImages/` instead, so both are auto-detected rather than assumed.

    Args:
        split_dir: A `<part>/<split>` directory.

    Returns:
        A tuple (image_dir, annotation_dir, extension). `annotation_dir` is
        "" for a test split with no released labels.

    Raises:
        FileNotFoundError: If no recognized image directory is present.
    """
    # The IDD117K archives are not consistent about directory names: the 95k
    # part ships 'Labels_json/' for val but 'labelsJSON/' for train. Match on
    # a normalized form (lower-case, no '_'/'-') so both resolve.
    def _find(candidates):
        wanted = {c.lower().replace("_", "").replace("-", "") for c in candidates}
        if not os.path.isdir(split_dir):
            return ""
        for entry in sorted(os.listdir(split_dir)):
            path = os.path.join(split_dir, entry)
            if os.path.isdir(path) and entry.lower().replace("_", "").replace("-", "") in wanted:
                return path
        return ""

    image_dir = _find(("leftImg8bit", "JPEGImages"))
    if not image_dir:
        raise FileNotFoundError(
            f"No 'leftImg8bit/' or 'JPEGImages/' directory under '{split_dir}'."
        )

    json_dir = _find(("Labels_json", "labelsJSON", "labels"))
    if json_dir:
        return image_dir, json_dir, ".json"
    xml_dir = _find(("Annotations",))
    if xml_dir:
        return image_dir, xml_dir, ".xml"
    return image_dir, "", ""


def _list_pairs_from_split_file(part_dir: str, split: str) -> List[Tuple[str, str]]:
    """Pairs images with annotations for the flat, list-file IDD_Detection layout.

    The original 2018 IDD_Detection release has no per-split directories:

        <part_dir>/JPEGImages/<camera>/<sequence>/<frame>.jpg
        <part_dir>/Annotations/<camera>/<sequence>/<frame>.xml
        <part_dir>/{train,val,test}.txt   # one '<camera>/<sequence>/<frame>' per line

    Args:
        part_dir: The IDD_Detection directory containing JPEGImages/ and the
            split list files.
        split: One of "train", "val", "test".

    Returns:
        A sorted list of (image_path, annotation_path) pairs, in the same
        format as `list_pairs`. Lines whose image is missing are skipped;
        lines whose annotation is missing are skipped for train/val and kept
        with annotation_path "" for test.

    Raises:
        FileNotFoundError: If `<part_dir>/<split>.txt` does not exist.
    """
    image_dir = os.path.join(part_dir, "JPEGImages")
    ann_dir = os.path.join(part_dir, "Annotations")
    list_path = os.path.join(part_dir, f"{split}.txt")
    if not os.path.isfile(list_path):
        raise FileNotFoundError(
            f"'{part_dir}' has a flat JPEGImages/ layout but no '{split}.txt' split "
            f"list next to it. The IDD_Detection release ships train.txt / val.txt / "
            f"test.txt at the top of the archive; copy them into '{part_dir}'."
        )

    with open(list_path) as handle:
        stems = [line.strip() for line in handle if line.strip()]

    pairs: List[Tuple[str, str]] = []
    missing_img = missing_ann = 0
    for stem in stems:
        stem = os.path.splitext(stem)[0] if stem.lower().endswith((".jpg", ".jpeg", ".xml")) else stem
        img_path = os.path.join(image_dir, stem + ".jpg")
        if not os.path.isfile(img_path):
            missing_img += 1
            continue
        ann_path = os.path.join(ann_dir, stem + ".xml")
        if not os.path.isfile(ann_path):
            if split == "test":
                pairs.append((img_path, ""))
            else:
                missing_ann += 1
            continue
        pairs.append((img_path, ann_path))

    if missing_img:
        print(f"Warning: {missing_img} line(s) in '{list_path}' have no image file; skipped.")
    if missing_ann:
        print(f"Warning: {missing_ann} image(s) listed in '{list_path}' have no annotation file; skipped.")
    return sorted(pairs)


def list_pairs(part_dir: str, split: str = "train") -> List[Tuple[str, str]]:
    """Pairs every image in a split with its annotation file.

    Two on-disk layouts are supported and auto-detected:

    1. Per-split directories (IDD_95kDetection, and IDD_Detection if it was
       re-organised): `<part_dir>/<split>/{leftImg8bit|JPEGImages}/<seq>/*.jpg`
       with matching `Labels_json/` or `Annotations/`.
    2. The original flat IDD_Detection release: `<part_dir>/JPEGImages/` and
       `<part_dir>/Annotations/` nested `<camera>/<sequence>/<frame>`, with
       the split given by `<part_dir>/<split>.txt`.

    Pairing is by identical relative path, differing only in the root
    directory and the extension.

    Args:
        part_dir: An IDD117K part directory (IDD_95kDetection or IDD_Detection).
        split: One of "train", "val", "test".

    Returns:
        A sorted list of (image_path, annotation_path) pairs. Images whose
        annotation file is missing are skipped. For "test" (no released
        labels) every annotation_path is "".
    """
    split_dir = os.path.join(part_dir, split)
    if not os.path.isdir(split_dir) and os.path.isdir(os.path.join(part_dir, "JPEGImages")):
        return _list_pairs_from_split_file(part_dir, split)

    image_dir, ann_dir, ext = _detect_layout(split_dir)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*", "*.jpg")))
    if not ann_dir:
        if split != "test":
            print(
                f"!! WARNING: no 'Labels_json/' or 'Annotations/' directory under '{split_dir}'. "
                f"All {len(image_paths)} '{split}' frames will be treated as UNLABELED: they "
                "contribute no ground-truth boxes to the detection benchmark and no labels to "
                "the ODD classifier. If this split is supposed to be annotated, the label "
                "archive was not extracted here."
            )
        return [(path, "") for path in image_paths]

    pairs: List[Tuple[str, str]] = []
    missing = 0
    for img_path in image_paths:
        rel = os.path.relpath(img_path, image_dir)
        ann_path = os.path.join(ann_dir, os.path.splitext(rel)[0] + ext)
        if not os.path.isfile(ann_path):
            missing += 1
            continue
        pairs.append((img_path, ann_path))

    if missing:
        print(f"Warning: {missing} image(s) in '{split_dir}' have no annotation file; skipped.")
    return pairs


def sample_pairs(
    pairs: Sequence[Tuple[str, str]], limit: Optional[int], seed: int = 0
) -> List[Tuple[str, str]]:
    """Deterministically subsamples a pair list.

    Sampling (rather than taking the first N) matters because the frames are
    ordered by capture sequence -- the first N would all come from a handful
    of consecutive drives and would not span the dataset's road/lighting
    conditions, which is exactly what the downstream ODD copula and classifier
    stages are fitting distributions over.

    Args:
        pairs: The full pair list.
        limit: Maximum pairs to keep. None or >= len(pairs) keeps all.
        seed: RNG seed, so a given limit always yields the same subset.

    Returns:
        A sorted list of at most `limit` pairs.
    """
    if limit is None or limit >= len(pairs):
        return list(pairs)
    return sorted(random.Random(seed).sample(list(pairs), limit))


def _parse_json_boxes(ann_path: str) -> List[Dict]:
    """Parses a `Labels_json` annotation file.

    Args:
        ann_path: Path to a `<frame>.json` file.

    Returns:
        A list of dicts with 'name' (str) and 'bbox' ((x, y, w, h) ints) keys,
        in native image pixel coordinates.
    """
    with open(ann_path) as handle:
        raw = json.load(handle)

    # Tolerate both the flat list form and an {"objects": [...]} wrapper.
    objects = raw.get("objects", []) if isinstance(raw, dict) else raw

    boxes: List[Dict] = []
    for obj in objects:
        bbox = obj.get("bbox") or {}
        xmin, ymin = int(bbox["xmin"]), int(bbox["ymin"])
        xmax, ymax = int(bbox["xmax"]), int(bbox["ymax"])
        boxes.append({"name": str(obj["name"]).strip().lower(),
                      "bbox": (xmin, ymin, xmax - xmin, ymax - ymin)})
    return boxes


def _parse_voc_boxes(ann_path: str) -> List[Dict]:
    """Parses a Pascal VOC XML annotation (the older IDD_Detection part).

    Args:
        ann_path: Path to a `<frame>.xml` file.

    Returns:
        A list of dicts in the same shape as `_parse_json_boxes`.
    """
    root = ET.parse(ann_path).getroot()
    boxes: List[Dict] = []
    for obj in root.findall("object"):
        name_node, bnd = obj.find("name"), obj.find("bndbox")
        if name_node is None or bnd is None:
            continue
        xmin, ymin = int(float(bnd.findtext("xmin", "0"))), int(float(bnd.findtext("ymin", "0")))
        xmax, ymax = int(float(bnd.findtext("xmax", "0"))), int(float(bnd.findtext("ymax", "0")))
        boxes.append({"name": (name_node.text or "").strip().lower(),
                      "bbox": (xmin, ymin, xmax - xmin, ymax - ymin)})
    return boxes


def load_boxes(ann_path: str) -> List[Dict]:
    """Loads annotation boxes, dispatching on file extension.

    Args:
        ann_path: A `.json` or `.xml` annotation path. "" yields [].

    Returns:
        A list of dicts with 'name' and 'bbox' ((x, y, w, h)) keys in native
        pixel coordinates.
    """
    if not ann_path:
        return []
    if ann_path.endswith(".json"):
        return _parse_json_boxes(ann_path)
    return _parse_voc_boxes(ann_path)


def scale_boxes(
    boxes: Sequence[Dict], src_wh: Tuple[int, int], dst_wh: Tuple[int, int] = IMAGE_SIZE
) -> List[Dict]:
    """Rescales boxes from native resolution to the pipeline's working size.

    IDD117K images are 1920x1080 (aspect 1.78) while the pipeline works at
    320x224 (aspect 1.43), and `cv2.resize` to a fixed size distorts the
    aspect ratio. The x and y scale factors are therefore applied
    **independently** -- using one uniform factor here silently shrinks IoU
    against YOLO's boxes and drives AP toward zero.

    Args:
        boxes: Boxes in native coordinates, as returned by `load_boxes`.
        src_wh: Native (width, height) of the source image.
        dst_wh: Target (width, height).

    Returns:
        A new list of box dicts in target coordinates. Boxes that round away
        to zero width or height are dropped.
    """
    scale_x = dst_wh[0] / float(src_wh[0])
    scale_y = dst_wh[1] / float(src_wh[1])

    scaled: List[Dict] = []
    for box in boxes:
        x, y, w, h = box["bbox"]
        new_w, new_h = int(round(w * scale_x)), int(round(h * scale_y))
        if new_w <= 0 or new_h <= 0:
            continue
        scaled.append({**box,
                       "bbox": (int(round(x * scale_x)), int(round(y * scale_y)), new_w, new_h)})
    return scaled


def group_of(name: str) -> str:
    """Maps a raw IDD label name to a project category group.

    Args:
        name: A raw annotation 'name' value.

    Returns:
        One of the GROUP_* constants. Unrecognized names map to GROUP_IGNORE
        (and are surfaced by `unmapped_labels`, not silently swallowed).
    """
    return NAME_TO_GROUP.get(name.strip().lower(), GROUP_IGNORE)


def filter_boxes_by_group(boxes: Sequence[Dict], groups: Sequence[str]) -> List[Box]:
    """Selects boxes whose label falls into any of the given groups.

    Args:
        boxes: Box dicts with a 'name' key.
        groups: GROUP_* values to keep.

    Returns:
        A list of bare (x, y, w, h) tuples, ready for detection_benchmark.
    """
    wanted = set(groups)
    return [box["bbox"] for box in boxes if group_of(box["name"]) in wanted]


def load_image_and_boxes(
    img_path: str, ann_path: str, size: Tuple[int, int] = IMAGE_SIZE
) -> Tuple[np.ndarray, List[Dict]]:
    """Loads one frame resized to `size`, with its boxes rescaled to match.

    Args:
        img_path: Path to the source .jpg.
        ann_path: Path to the matching annotation, or "" for none.
        size: Target (width, height).

    Returns:
        A tuple (image, boxes): a BGR uint8 array of shape (size[1], size[0], 3)
        and the box dicts in that image's coordinates. Returns (None, []) if
        the image is unreadable.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None, []

    native_wh = (img.shape[1], img.shape[0])
    boxes = scale_boxes(load_boxes(ann_path), native_wh, size)
    return cv2.resize(img, size), boxes


def iter_frames(
    pairs: Sequence[Tuple[str, str]], size: Tuple[int, int] = IMAGE_SIZE
) -> Iterator[Tuple[np.ndarray, List[Dict]]]:
    """Streams (image, boxes) one frame at a time.

    This is the memory-safe entry point: it never holds more than one decoded
    frame, so it scales to the full 96,897-image train split, which as a
    single array would need ~20.8 GB.

    Args:
        pairs: (image_path, annotation_path) pairs.
        size: Target (width, height).

    Yields:
        (image, boxes) tuples, skipping unreadable images.
    """
    for img_path, ann_path in pairs:
        img, boxes = load_image_and_boxes(img_path, ann_path, size)
        if img is None:
            continue
        yield img, boxes


class FramePairDataset:
    """A `torch.utils.data.Dataset` over (image_path, annotation_path) pairs.

    Exists so frame decoding can be pushed onto DataLoader worker processes.
    `iter_frames` above decodes one 1920x1080 JPEG at a time, inline on the
    caller's thread, which means every downstream GPU forward pass waits on a
    synchronous `cv2.imread`. That is the single biggest throughput limit on a
    GPU box: at ~104k frames the decode, not the model, sets the wall clock.

    The per-item body is `load_image_and_boxes()` verbatim, so frames and box
    coordinates are byte-identical to the serial path -- only *where* the work
    happens changes.

    Args:
        pairs: (image_path, annotation_path) pairs.
        size: Target (width, height).
    """

    def __init__(
        self, pairs: Sequence[Tuple[str, str]], size: Tuple[int, int] = IMAGE_SIZE
    ) -> None:
        self.pairs = list(pairs)
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        img_path, ann_path = self.pairs[index]
        img, boxes = load_image_and_boxes(img_path, ann_path, self.size)
        # Unreadable frames are dropped by the collate step, not here: a
        # Dataset must return something for every index it is asked for.
        # A frame with NO annotation file yields boxes=None (unknown), which
        # is different from [] (annotated, genuinely empty) -- consumers that
        # build ground-truth counts must not treat the two alike.
        return img, (boxes if ann_path else None)


def _collate_frames(batch):
    """Collates a batch of (image, boxes), dropping unreadable frames.

    Deliberately does NOT stack into a tensor: `compute_features` and the
    Ultralytics YOLO call both want a list of HxWx3 BGR uint8 arrays, and box
    lists are ragged. Stacking happens later, on the GPU side.

    Args:
        batch: A list of (image_or_None, boxes) tuples from FramePairDataset.

    Returns:
        A tuple (images, boxes_per_image) of equal length, with None images
        (and their boxes) removed -- matching `iter_frames`, which skips them.
        `boxes_per_image[i]` is None for a frame that had no annotation file.
    """
    images, boxes = [], []
    for img, box_list in batch:
        if img is None:
            continue
        images.append(img)
        boxes.append(box_list)
    return images, boxes


def _worker_init(_worker_id: int) -> None:
    """Disables OpenCV's internal thread pool inside DataLoader workers.

    Without this, each of N worker processes spawns its own OpenCV thread
    pool sized to the whole machine, and they oversubscribe the CPU and fight
    each other -- which can make a multi-worker loader *slower* than the
    serial one.
    """
    cv2.setNumThreads(0)


def iter_frame_batches(
    pairs: Sequence[Tuple[str, str]],
    batch_size: int = 32,
    num_workers: int = 8,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> Iterator[Tuple[List[np.ndarray], List[List[Dict]]]]:
    """Streams (images, boxes) in batches, decoding in parallel worker processes.

    The batched, parallel-decode counterpart to `iter_frames`. Yields the same
    frames in the same order, just grouped and decoded ahead of time. Peak
    memory stays bounded (batch_size * prefetch_factor * num_workers frames at
    320x224, a few hundred MB at most), so this is still safe on the full
    96,897-image train split.

    Falls back to the serial `iter_frames` path when `num_workers == 0` or when
    torch is unavailable, so nothing here becomes a hard torch dependency for
    callers that do not need it.

    Args:
        pairs: (image_path, annotation_path) pairs.
        batch_size: Frames per yielded batch.
        num_workers: DataLoader worker processes. 0 runs serially in-process.
        size: Target (width, height).

    Yields:
        (images, boxes_per_image) tuples. `images` is a list of BGR uint8
        arrays of shape (size[1], size[0], 3); `boxes_per_image[i]` holds the
        box dicts for `images[i]`, or None if that frame has no annotation
        file. Unreadable frames are skipped, so a batch may be shorter than
        `batch_size` (and may be empty).
    """
    pairs = list(pairs)

    if num_workers <= 0:
        batch_imgs, batch_boxes = [], []
        for (img_path, ann_path) in pairs:
            img, boxes = load_image_and_boxes(img_path, ann_path, size)
            if img is None:
                continue
            batch_imgs.append(img)
            batch_boxes.append(boxes if ann_path else None)
            if len(batch_imgs) >= batch_size:
                yield batch_imgs, batch_boxes
                batch_imgs, batch_boxes = [], []
        if batch_imgs:
            yield batch_imgs, batch_boxes
        return

    from torch.utils.data import DataLoader

    loader = DataLoader(
        FramePairDataset(pairs, size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_frames,
        worker_init_fn=_worker_init,
        pin_memory=False,  # These are numpy arrays, not tensors -- nothing to pin.
        persistent_workers=False,
        prefetch_factor=4,
    )
    for images, boxes in loader:
        yield images, boxes


def scan_label_vocabulary(
    part_dir: str, splits: Sequence[str] = ("train", "val"), limit: Optional[int] = 5000
) -> "collections.Counter":
    """Counts every distinct label name appearing in the annotations.

    Run this BEFORE trusting `NAME_TO_GROUP` -- see the module docstring and
    docs/DATASET_NOTES.md on why a mapping table is not assumed correct here.

    Args:
        part_dir: An IDD117K part directory.
        splits: Splits to scan.
        limit: Annotation files to sample per split (None scans all).

    Returns:
        A Counter mapping raw label name -> occurrence count.
    """
    counts: "collections.Counter" = collections.Counter()
    for split in splits:
        try:
            pairs = sample_pairs(list_pairs(part_dir, split), limit)
        except FileNotFoundError as exc:
            print(f"Skipping split '{split}': {exc}")
            continue
        for _, ann_path in pairs:
            for box in load_boxes(ann_path):
                counts[box["name"]] += 1
        print(f"Scanned {len(pairs)} annotation file(s) in split '{split}'.")
    return counts


def unmapped_labels(counts: "collections.Counter") -> List[str]:
    """Reports label names present in the data but absent from NAME_TO_GROUP.

    Args:
        counts: Output of `scan_label_vocabulary`.

    Returns:
        A sorted list of unmapped names. A non-empty result means
        NAME_TO_GROUP needs updating before the boxes can be trusted.
    """
    return sorted(name for name in counts if name not in NAME_TO_GROUP)


def _main() -> None:
    """CLI: enumerate and persist the real label vocabulary."""
    from src.common.paths import IDD117K_95K_DIR, IDD117K_VOCAB_JSON, ensure_output_dirs

    parser = argparse.ArgumentParser(description="Inspect the IDD117K-Detection label vocabulary.")
    parser.add_argument("--scan", action="store_true", help="Scan annotations and report label counts.")
    parser.add_argument("--part_dir", default=IDD117K_95K_DIR, help="IDD117K part directory to scan.")
    parser.add_argument("--limit", type=int, default=5000, help="Annotation files to sample per split (0 = all).")
    args = parser.parse_args()

    if not args.scan:
        parser.error("nothing to do: pass --scan")

    ensure_output_dirs()
    counts = scan_label_vocabulary(args.part_dir, limit=args.limit or None)

    print(f"\n{'label':<28} {'count':>10}  group")
    print("-" * 60)
    for name, count in counts.most_common():
        print(f"{name:<28} {count:>10}  {group_of(name)}")

    missing = unmapped_labels(counts)
    if missing:
        print(f"\n!! {len(missing)} label(s) NOT in NAME_TO_GROUP -- update it before trusting boxes:")
        for name in missing:
            print(f"     {name}  ({counts[name]} occurrences)")
    else:
        print("\nAll labels found are present in NAME_TO_GROUP.")

    with open(IDD117K_VOCAB_JSON, "w") as handle:
        json.dump({"counts": dict(counts), "unmapped": missing}, handle, indent=2)
    print(f"\nWrote vocabulary -> '{IDD117K_VOCAB_JSON}'")


if __name__ == "__main__":
    _main()
