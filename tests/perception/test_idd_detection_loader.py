"""Unit tests for idd_detection_loader.py's pairing, parsing, and scaling.

Everything here runs against a synthetic on-disk fixture rather than the real
117K-image archive, so the suite stays fast and does not require the dataset
to be present.
"""

import json
import os

import pytest

from src.perception.idd_detection_loader import (
    DEFAULT_GT_GROUPS,
    GROUP_AUTORICKSHAW,
    GROUP_IGNORE,
    filter_boxes_by_group,
    group_of,
    list_pairs,
    load_boxes,
    sample_pairs,
    scale_boxes,
    scan_label_vocabulary,
    unmapped_labels,
)

NATIVE_WH = (1920, 1080)
TARGET_WH = (320, 224)


@pytest.fixture
def idd_fixture(tmp_path):
    """Builds a miniature IDD117K-style tree with a Labels_json and a VOC part."""
    part = tmp_path / "IDD_95kDetection" / "val"
    img_dir, ann_dir = part / "leftImg8bit" / "00762", part / "Labels_json" / "00762"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    # Three images, two annotations: the third must be dropped by list_pairs.
    for name in ("00000", "00001", "00002"):
        (img_dir / f"{name}.jpg").touch()

    (ann_dir / "00000.json").write_text(json.dumps([
        {"name": "ego vehicle", "bbox": {"xmin": 955, "ymin": 1068, "xmax": 981, "ymax": 1080}},
        {"name": "autorickshaw", "bbox": {"xmin": 1, "ymin": 423, "xmax": 204, "ymax": 758}},
        {"name": "car", "bbox": {"xmin": 0, "ymin": 0, "xmax": 1920, "ymax": 1080}},
        {"name": "hoverboard", "bbox": {"xmin": 10, "ymin": 10, "xmax": 20, "ymax": 20}},
    ]))
    (ann_dir / "00001.json").write_text(json.dumps([
        {"name": "truck", "bbox": {"xmin": 100, "ymin": 200, "xmax": 300, "ymax": 400}},
    ]))

    voc = tmp_path / "IDD_Detection" / "val"
    (voc / "JPEGImages" / "0001").mkdir(parents=True)
    (voc / "Annotations" / "0001").mkdir(parents=True)
    (voc / "JPEGImages" / "0001" / "frame1.jpg").touch()
    (voc / "Annotations" / "0001" / "frame1.xml").write_text(
        "<annotation><object><name>bus</name><bndbox>"
        "<xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>140</ymax>"
        "</bndbox></object></annotation>"
    )
    return tmp_path


def test_list_pairs_skips_images_without_annotations(idd_fixture):
    pairs = list_pairs(str(idd_fixture / "IDD_95kDetection"), "val")
    assert len(pairs) == 2


def test_list_pairs_matches_image_to_annotation_by_relative_path(idd_fixture):
    for img_path, ann_path in list_pairs(str(idd_fixture / "IDD_95kDetection"), "val"):
        assert os.path.splitext(os.path.basename(img_path))[0] == os.path.splitext(
            os.path.basename(ann_path)
        )[0]


def test_load_boxes_converts_xyxy_to_xywh(idd_fixture):
    _, ann_path = list_pairs(str(idd_fixture / "IDD_95kDetection"), "val")[0]
    auto = [b for b in load_boxes(ann_path) if b["name"] == "autorickshaw"][0]
    assert auto["bbox"] == (1, 423, 203, 335)


def test_scale_boxes_maps_full_frame_to_full_target_frame():
    boxes = [{"name": "car", "bbox": (0, 0, 1920, 1080)}]
    assert scale_boxes(boxes, NATIVE_WH, TARGET_WH)[0]["bbox"] == (0, 0, 320, 224)


def test_scale_boxes_uses_independent_x_and_y_factors():
    # 1920x1080 (1.78) -> 320x224 (1.43) is aspect-distorting. A single uniform
    # factor would give h = round(335 * 320/1920) = 56; the correct height uses
    # 224/1080. Getting this wrong silently drives IoU -- and AP -- toward zero.
    boxes = [{"name": "autorickshaw", "bbox": (1, 423, 203, 335)}]
    scaled = scale_boxes(boxes, NATIVE_WH, TARGET_WH)[0]["bbox"]
    assert scaled[3] == round(335 * 224 / 1080)
    assert scaled[3] != round(335 * 320 / 1920)


def test_scale_boxes_drops_boxes_that_round_away_to_nothing():
    assert scale_boxes([{"name": "car", "bbox": (0, 0, 2, 2)}], NATIVE_WH, TARGET_WH) == []


def test_ego_vehicle_is_ignored():
    # The ego vehicle is the camera car's own hood -- present in nearly every
    # frame and not a detectable object.
    assert group_of("ego vehicle") == GROUP_IGNORE


def test_autorickshaw_is_its_own_group():
    # IDD-Lite had no autorickshaw label at all; having a real one here is what
    # makes feature_extraction's AUTORICKSHAW_* heuristic scoreable.
    assert group_of("autorickshaw") == GROUP_AUTORICKSHAW


def test_unknown_label_maps_to_ignore_rather_than_raising():
    assert group_of("hoverboard") == GROUP_IGNORE


def test_default_gt_groups_selects_only_four_wheelers(idd_fixture):
    _, ann_path = list_pairs(str(idd_fixture / "IDD_95kDetection"), "val")[0]
    scaled = scale_boxes(load_boxes(ann_path), NATIVE_WH, TARGET_WH)
    assert filter_boxes_by_group(scaled, DEFAULT_GT_GROUPS) == [(0, 0, 320, 224)]


def test_scan_reports_labels_missing_from_the_mapping_table(idd_fixture):
    # docs/DATASET_NOTES.md requires the class scheme be verified against real
    # data; an unmapped label must surface here, not vanish into GROUP_IGNORE.
    counts = scan_label_vocabulary(
        str(idd_fixture / "IDD_95kDetection"), splits=("val",), limit=None
    )
    assert unmapped_labels(counts) == ["hoverboard"]
    assert counts["truck"] == 1


def test_sample_pairs_is_deterministic_and_respects_limit(idd_fixture):
    pairs = list_pairs(str(idd_fixture / "IDD_95kDetection"), "val")
    assert sample_pairs(pairs, 1, seed=7) == sample_pairs(pairs, 1, seed=7)
    assert len(sample_pairs(pairs, 1)) == 1
    assert len(sample_pairs(pairs, None)) == len(pairs)


def test_voc_layout_is_autodetected_for_the_older_part(idd_fixture):
    pairs = list_pairs(str(idd_fixture / "IDD_Detection"), "val")
    assert len(pairs) == 1
    assert load_boxes(pairs[0][1])[0]["bbox"] == (10, 20, 100, 120)
