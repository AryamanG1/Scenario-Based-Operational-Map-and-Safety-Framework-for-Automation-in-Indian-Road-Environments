"""Refined Adaptive CLAHE & Bilateral Dehazing Filter tuned specifically for road perception.
Preserves natural asphalt/road color temperatures while suppressing airlight scattering.
"""

import cv2
import numpy as np


def adaptive_road_dehaze(img_rgb: np.ndarray) -> np.ndarray:
    """Fast, edge-preserving road restoration filter.
    
    1. LAB Color Space Separation
    2. Adaptive CLAHE on L-channel (recovers contrast hidden under fog)
    3. Bilateral Filter on chrominance (smooths sensor noise without blurring road edges)
    4. Unsharp Masking (sharpens road boundaries and vehicle contours)
    """
    # 1. Convert to LAB space
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # 2. Strong CLAHE for fog penetration on Luminance
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # 3. Bilateral smoothing to remove noise while keeping edges razor sharp
    l_filtered = cv2.bilateralFilter(l_enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    # Recombine LAB
    lab_enhanced = cv2.merge((l_filtered, a, b))
    restored = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # 4. Subtle unsharp mask to restore lane markers and obstacle outlines
    gaussian = cv2.GaussianBlur(restored, (0, 0), 2.0)
    sharpened = cv2.addWeighted(restored, 1.3, gaussian, -0.3, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)
