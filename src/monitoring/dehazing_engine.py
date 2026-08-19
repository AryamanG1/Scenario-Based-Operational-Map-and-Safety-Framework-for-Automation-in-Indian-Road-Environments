"""Module: Physics-Based Koschmieder Inversion & Atmospheric Dehazing Filter.

Implements real-time adaptive dehazing and contrast enhancement for Indian road scenes:
1. Koschmieder Transmission Map Inversion
2. CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB Color Space
3. Fast Bilateral / Guided Smoothing to eliminate sensor noise
"""

import cv2
import numpy as np
import torch


def koschmieder_dehaze_and_enhance(img_rgb: np.ndarray, atmospheric_light: float = 0.95, omega: float = 0.85) -> np.ndarray:
    """Restores foggy/smoggy road images using Dark Channel Prior & LAB-space CLAHE.
    
    Args:
        img_rgb: Input RGB image with pixel values in [0, 255] (uint8).
        atmospheric_light: Estimated atmospheric airlight (A).
        omega: Haze removal strength parameter (0.80 - 0.95).
        
    Returns:
        Dehazed, contrast-restored RGB image (uint8).
    """
    img_float = img_rgb.astype(np.float32) / 255.0

    # 1. Estimate Dark Channel Prior (DCP)
    min_channel = np.min(img_float, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(min_channel, kernel)

    # 2. Invert Koschmieder's Transmission Map: t(x) = 1 - omega * (dark / A)
    transmission = 1.0 - omega * (dark_channel / atmospheric_light)
    transmission = np.clip(transmission, 0.15, 1.0)  # Lower bound avoids noise blowout

    # Guided filtering for smooth edge-preserving transmission
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    transmission_refined = cv2.bilateralFilter(transmission.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=9)

    # 3. Recover Scene Radiance: J(x) = (I(x) - A) / max(t(x), t0) + A
    restored = np.zeros_like(img_float)
    for c in range(3):
        restored[:, :, c] = (img_float[:, :, c] - atmospheric_light) / transmission_refined + atmospheric_light

    restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)

    # 4. Adaptive Contrast Recovery via CLAHE in LAB space
    lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    final_dehazed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    return final_dehazed


def dehaze_tensor_gpu(img_tensor: torch.Tensor) -> torch.Tensor:
    """Applies dehazing filter to a PyTorch tensor (B, 3, H, W) on GPU/CPU."""
    device = img_tensor.device
    out_tensors = []
    
    # Process batch
    for i in range(img_tensor.shape[0]):
        img_np = (img_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        dehazed_np = koschmieder_dehaze_and_enhance(img_np)
        t_dehazed = torch.tensor(dehazed_np).permute(2, 0, 1).float() / 255.0
        out_tensors.append(t_dehazed)
        
    return torch.stack(out_tensors).to(device)


if __name__ == "__main__":
    print("[*] Koschmieder Inversion & Dehazing Engine module loaded successfully.")
