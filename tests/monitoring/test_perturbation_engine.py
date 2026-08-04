"""Unit tests for perturbation_engine.py's perturbation formulas and metrics."""

import torch

from src.monitoring.perturbation_engine import (
    HARYANA_DELTA,
    HIMACHAL_DELTA,
    PUNJAB_DELTA,
    add_fog_gpu,
    add_gaussian_noise_gpu,
    adjust_brightness_contrast_gpu,
    calculate_metrics,
    perturb_image_gpu,
    sample_delta,
)


def test_adjust_brightness_contrast_identity_at_defaults():
    img = torch.rand(3, 8, 8)
    out = adjust_brightness_contrast_gpu(img, delta_bright=0.0, delta_contrast=1.0)
    assert torch.allclose(out, img)


def test_adjust_brightness_contrast_clamped_to_unit_range():
    img = torch.ones(3, 8, 8)
    out = adjust_brightness_contrast_gpu(img, delta_bright=200.0, delta_contrast=2.0)
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_add_gaussian_noise_zero_std_is_identity():
    img = torch.rand(3, 8, 8)
    out = add_gaussian_noise_gpu(img, std=0.0)
    assert torch.allclose(out, img)


def test_add_fog_zero_strength_is_near_identity():
    img = torch.rand(3, 8, 8)
    out = add_fog_gpu(img, fog_strength=0.0)
    assert torch.allclose(out, img, atol=1e-5)


def test_add_fog_output_bounded():
    img = torch.rand(3, 16, 16)
    out = add_fog_gpu(img, fog_strength=2.0)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_perturb_image_applies_all_three_effects():
    img = torch.full((3, 8, 8), 0.5)
    delta = {"brightness": 50, "contrast": 1.0, "noise": 0.0, "fog": 0.0}
    out = perturb_image_gpu(img, delta)
    assert not torch.allclose(out, img)  # brightness shift should change it
    assert out.shape == img.shape


def test_sample_delta_within_regional_ranges():
    for region, ranges in (
        ("haryana", HARYANA_DELTA),
        ("punjab", PUNJAB_DELTA),
        ("himachal", HIMACHAL_DELTA),
    ):
        for _ in range(20):
            delta = sample_delta(region)
            assert ranges["brightness"][0] <= delta["brightness"] <= ranges["brightness"][1]
            assert ranges["contrast"][0] <= delta["contrast"] <= ranges["contrast"][1]
            assert ranges["noise"][0] <= delta["noise"] <= ranges["noise"][1]
            assert ranges["fog"][0] <= delta["fog"] <= ranges["fog"][1]


def test_sample_delta_unknown_region_raises():
    import pytest

    with pytest.raises(ValueError):
        sample_delta("atlantis")


def test_calculate_metrics_perfect_match():
    target = torch.randint(0, 8, (16, 16))
    miou, acc = calculate_metrics(target, target, num_classes=8)
    assert miou == 1.0 and acc == 1.0


def test_calculate_metrics_no_overlap():
    pred = torch.zeros(4, 4, dtype=torch.long)
    target = torch.ones(4, 4, dtype=torch.long)
    miou, acc = calculate_metrics(pred, target, num_classes=8)
    assert acc == 0.0
    assert miou == 0.0
