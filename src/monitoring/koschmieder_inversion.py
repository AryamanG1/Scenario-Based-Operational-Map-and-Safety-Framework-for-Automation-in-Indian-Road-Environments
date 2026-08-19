"""Exact Physics-Based Atmospheric Model Inversion & Denoising Filter.

Inverts Koschmieder's Law and Brightness/Contrast shifts mathematically:
Forward Equation (from Sun et al., IEEE TIV 2024 / repo):
  x_perturbed = T(d) * (delta_contrast * x + delta_bright/255) + A * (1 - T(d)) + noise
  where T(d) = exp(-beta * d(x))

Analytical Inversion Equation:
  1. Transmission Recovery: T_hat(x) = exp(-beta * d(x))
  2. Fog Inversion: x_unfog = (x_perturbed - A * (1 - T_hat)) / max(T_hat, epsilon)
  3. Linear De-bias: x_restored = (x_unfog - delta_bright/255) / delta_contrast
  4. Spatial Edge-Preserving Denoising to filter out the amplified noise
"""

import cv2
import numpy as np
import torch


def exact_koschmieder_inversion(
    perturbed_tensor: torch.Tensor,
    fog_strength: float = 0.50,
    atmospheric_light: float = 1.0,
    delta_contrast: float = 0.80,
    delta_bright: float = 0.0,
) -> torch.Tensor:
    """Mathematically inverts Koschmieder's scattering and contrast shift on GPU/CPU tensors."""
    h, w = perturbed_tensor.shape[-2], perturbed_tensor.shape[-1]
    x = torch.linspace(-1, 1, w, device=perturbed_tensor.device)
    y = torch.linspace(-1, 1, h, device=perturbed_tensor.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    distance = torch.sqrt(grid_x**2 + grid_y**2)

    # 1. Compute exact theoretical transmittance map T(x)
    T = torch.exp(-distance * fog_strength)
    T = torch.clamp(T, 0.15, 1.0)  # Epsilon clamping for stability

    # 2. Invert atmospheric airlight: I_clear = (I_fog - A*(1-T)) / T
    unfogged = (perturbed_tensor - atmospheric_light * (1.0 - T)) / T

    # 3. Invert contrast & brightness shifts: x = (x' - delta_bright) / delta_contrast
    db = delta_bright / 255.0
    restored = (unfogged - db) / max(delta_contrast, 0.1)
    restored = torch.clamp(restored, 0.0, 1.0)

    # 4. Post-filter residual sensor noise while preserving sharp boundaries
    restored_np = (restored.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    denoised_np = cv2.edgePreservingFilter(restored_np, flags=1, sigma_s=40, sigma_r=0.3)
    
    return torch.tensor(denoised_np).permute(2, 0, 1).float().unsqueeze(0).to(perturbed_tensor.device) / 255.0


if __name__ == "__main__":
    print("[*] Exact Koschmieder Inversion module ready.")
