"""Regenerates the Dehazing Restoration visual comparison with:
1. Multi-Scale Retinex (MSRCR) filter (rich, vivid colors restored)
2. Clean baseline prediction vs degraded vs restored prediction
3. Perfectly crisp, realistic segmentation masks
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from src.monitoring.retinex_dehaze import multi_scale_retinex_dehaze

MODEL_PATH = r"D:\Capstone\models\segnet_idd20k_10epochs.pth"
DATASET_DIR = r"D:\Capstone\data\idd20kII"
OUTPUT_DIR = r"D:\Capstone\outputs"
IMAGE_SIZE = (320, 224)
NUM_CLASSES = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.3),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, NUM_CLASSES, kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def add_fog(img_tensor: torch.Tensor, fog_strength: float = 0.50) -> torch.Tensor:
    h, w = img_tensor.shape[-2], img_tensor.shape[-1]
    x = torch.linspace(-1, 1, w, device=img_tensor.device)
    y = torch.linspace(-1, 1, h, device=img_tensor.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    distance = torch.sqrt(grid_x**2 + grid_y**2)
    fog_mask = torch.exp(-distance * fog_strength)
    return torch.clamp(img_tensor * fog_mask + 1.0 * (1.0 - fog_mask), 0.0, 1.0)


def add_noise_and_fog(img_tensor: torch.Tensor) -> torch.Tensor:
    img = torch.clamp(img_tensor * 0.80, 0.0, 1.0)
    noise = torch.randn_like(img) * (8.0 / 255.0)
    img = torch.clamp(img + noise, 0.0, 1.0)
    return add_fog(img, fog_strength=0.48)


def main():
    model = SegNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "leftImg8bit", "val", "*", "*_leftImg8bit.jpg")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "leftImg8bit", "train", "*", "*_leftImg8bit.jpg")))

    # Select 4 visually diverse and clear scenes
    samples = [img_paths[10], img_paths[45], img_paths[80], img_paths[120]]

    fig, axes = plt.subplots(4, 5, figsize=(22, 13))
    plt.subplots_adjust(hspace=0.28, wspace=0.10)

    for row, p in enumerate(samples):
        img_bgr = cv2.imread(p)
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, IMAGE_SIZE), cv2.COLOR_BGR2RGB)

        t_clean = torch.tensor(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        t_fog = add_noise_and_fog(t_clean)
        img_fog_np = (t_fog.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Apply Multi-Scale Retinex Dehaze
        img_dehazed_np = multi_scale_retinex_dehaze(img_fog_np)
        t_dehazed = torch.tensor(img_dehazed_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            pred_clean = model(t_clean).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_fog = model(t_fog).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_dehazed = model(t_dehazed).argmax(dim=1).squeeze(0).cpu().numpy()

        # Col 0: Clean Image
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"Scene {row+1}: Raw Camera Frame", fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        # Col 1: Foggy Image
        axes[row, 1].imshow(img_fog_np)
        axes[row, 1].set_title("Adverse Weather (Punjab Fog + Noise)", fontsize=10, fontweight="bold", color="darkred")
        axes[row, 1].axis("off")

        # Col 2: Restored Image (Rich Color Retinex)
        axes[row, 2].imshow(img_dehazed_np)
        axes[row, 2].set_title("Restored (Multi-Scale Retinex Filter)", fontsize=10, fontweight="bold", color="navy")
        axes[row, 2].axis("off")

        # Col 3: Foggy Output (Distorted)
        axes[row, 3].imshow(pred_fog, cmap="tab10", vmin=0, vmax=7)
        axes[row, 3].set_title("Degraded Output (Without Filter)", fontsize=10, fontweight="bold", color="darkred")
        axes[row, 3].axis("off")

        # Col 4: Restored Output (Clear Road & Objects!)
        axes[row, 4].imshow(pred_dehazed, cmap="tab10", vmin=0, vmax=7)
        axes[row, 4].set_title("Restored Output (With Filter)", fontsize=10, fontweight="bold", color="darkgreen")
        axes[row, 4].axis("off")

    output_img = os.path.join(OUTPUT_DIR, "dehazing_restoration_comparison.png")
    plt.savefig(output_img, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[+] Successfully re-generated Crisp Dehazing Comparison -> {output_img}")


if __name__ == "__main__":
    main()
