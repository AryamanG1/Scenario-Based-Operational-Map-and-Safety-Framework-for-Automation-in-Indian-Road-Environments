"""Unit tests for idd20k_polygon_pipeline.py's polygon rasterization.

Runs against synthetic polygon files, so the suite needs neither the 5.55 GB
IDD-20K-II archive nor a GPU.
"""

import json

import numpy as np
import pytest

from src.perception.idd20k_polygon_pipeline import (
    LABEL_TO_CLASS,
    VOID_LABEL,
    class_distribution,
    list_pairs,
    rasterize_polygons,
    scan_label_vocabulary,
    unmapped_labels,
)

NATIVE = {"imgWidth": 1920, "imgHeight": 1080}


def _write(tmp_path, objects, split="train", seq="234", frame="frame1"):
    """Writes one synthetic image/polygon pair and returns the dataset root."""
    img_dir = tmp_path / "leftImg8bit" / split / seq
    ann_dir = tmp_path / "gtFine" / split / seq
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / f"{frame}_leftImg8bit.jpg").touch()
    path = ann_dir / f"{frame}_gtFine_polygons.json"
    path.write_text(json.dumps({**NATIVE, "objects": objects}))
    return tmp_path, str(path)


def _full_frame(label):
    return {"label": label, "polygon": [[0, 0], [1920, 0], [1920, 1080], [0, 1080]]}


def test_unannotated_image_is_skipped(tmp_path):
    root, _ = _write(tmp_path, [_full_frame("road")])
    (root / "leftImg8bit" / "train" / "234" / "frame99_leftImg8bit.jpg").touch()
    assert len(list_pairs(str(root), "train")) == 1


def test_mask_defaults_to_void_where_no_polygon_covers(tmp_path):
    _, ann = _write(tmp_path, [])
    mask, _ = rasterize_polygons(ann)
    assert mask.shape == (224, 320)
    assert np.all(mask == VOID_LABEL)


def test_full_frame_polygon_fills_the_whole_mask(tmp_path):
    _, ann = _write(tmp_path, [_full_frame("road")])
    mask, _ = rasterize_polygons(ann)
    assert np.all(mask == LABEL_TO_CLASS["road"])


def test_later_objects_paint_over_earlier_ones(tmp_path):
    # These annotations are authored back-to-front, so file order is z-order.
    _, ann = _write(tmp_path, [_full_frame("road"), _full_frame("car")])
    mask, _ = rasterize_polygons(ann)
    assert np.all(mask == LABEL_TO_CLASS["car"])


def test_polygon_vertices_scale_with_independent_x_and_y_factors(tmp_path):
    # Left half of a 1920x1080 frame must become the left half of a 320x224
    # mask: full height, half width. Under a single uniform factor the y
    # extent would be 1080 * 320/1920 = 180 rows, leaving the bottom 44 rows
    # of the mask unpainted -- so asserting the LAST row is painted is what
    # actually pins the independent-y behavior down.
    _, ann = _write(tmp_path, [{"label": "road",
                                "polygon": [[0, 0], [960, 0], [960, 1080], [0, 1080]]}])
    mask, _ = rasterize_polygons(ann)
    assert np.all(mask[:, :160] == LABEL_TO_CLASS["road"])
    assert np.all(mask[-1, :160] == LABEL_TO_CLASS["road"])
    # cv2.fillPoly is edge-inclusive, so the boundary vertex column x=160 is
    # painted too; everything strictly beyond it stays void.
    assert np.all(mask[:, 161:] == VOID_LABEL)


def test_degenerate_polygon_is_ignored(tmp_path):
    _, ann = _write(tmp_path, [{"label": "road", "polygon": [[0, 0], [10, 10]]}])
    mask, _ = rasterize_polygons(ann)
    assert np.all(mask == VOID_LABEL)


def test_unmapped_label_is_painted_void_and_reported(tmp_path):
    # docs/DATASET_NOTES.md requires the class scheme be verified against real
    # data; an unknown label must surface rather than silently become void.
    _, ann = _write(tmp_path, [_full_frame("hoverboard")])
    mask, unmapped = rasterize_polygons(ann)
    assert unmapped == ["hoverboard"]
    assert np.all(mask == VOID_LABEL)


def test_missing_image_dimensions_raise(tmp_path):
    ann_dir = tmp_path / "gtFine" / "train" / "234"
    ann_dir.mkdir(parents=True)
    path = ann_dir / "frame1_gtFine_polygons.json"
    path.write_text(json.dumps({"objects": []}))
    with pytest.raises(ValueError, match="imgWidth"):
        rasterize_polygons(str(path))


def test_scan_reports_labels_missing_from_the_mapping_table(tmp_path):
    root, _ = _write(tmp_path, [_full_frame("road"), _full_frame("hoverboard")])
    counts = scan_label_vocabulary(str(root), splits=("train",), limit=None)
    assert unmapped_labels(counts) == ["hoverboard"]
    assert counts["road"] == 1


def test_class_distribution_sums_to_one():
    labels = np.zeros((2, 224, 320), dtype=np.uint8)
    labels[1] = 3
    dist = class_distribution(labels)
    assert dist[0] == pytest.approx(0.5)
    assert dist[3] == pytest.approx(0.5)
    assert sum(dist.values()) == pytest.approx(1.0)
