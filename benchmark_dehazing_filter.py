"""Benchmark & Visualizer for Atmospheric Inversion (Dehazing Filter):
Compares 6 stages:
1. Clean Camera Image
2. Perturbed Fog/Smoggy Image
3. Dehazed/Restored Image (Our Innovation Filter)
4. SegNet on Clean Image
5. SegNet on Foggy Image (Degraded)
6. SegNet on Dehazed Image (Restored Accuracy!)
"""

import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from src.monitoring.dehazing_engine import koschmieder_dehaze_and_enhance

MODEL_PATH = r"D:\Capstone\models\segnet_idd20k_10epochs.pth"
DATASET_DIR = r"D:\Capstone\data\idd20kII"
OUTPUT_DIR = r"D:\Capstone\outputs"
IMAGE_SIZE = (320, 224)
NUM_CLASSES = 8
VOID_LABEL = 7

LABEL_TO_ID = {
    "road": 0, "drivable": 0, "parking": 0,
    "sidewalk": 1, "curb": 1, "non-drivable": 1,
    "building": 2, "fence": 2, "guard rail": 2, "bridge": 2, "tunnel": 2, "pole": 2, "traffic light": 2, "traffic sign": 2,
    "vegetation": 3, "terrain": 3,
    "sky": 4,
    "person": 5, "rider": 5,
    "car": 6, "truck": 6, "bus": 6, "caravan": 6, "trailer": 6, "train": 6, "motorcycle": 6, "bicycle": 6, "autorickshaw": 6, "vehicle fallback": 6
}

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


def add_fog_gpu(img_tensor: torch.Tensor, fog_strength: float = 0.65) -> torch.Tensor:
    h, w = img_tensor.shape[-2], img_tensor.shape[-1]
    x = torch.linspace(-1, 1, w, device=img_tensor.device)
    y = torch.linspace(-1, 1, h, device=img_tensor.device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    distance = torch.sqrt(grid_x**2 + grid_y**2)
    fog_mask = torch.exp(-distance * fog_strength)
    return torch.clamp(img_tensor * fog_mask + 1.0 * (1.0 - fog_mask), 0.0, 1.0)


def add_noise_and_haze(img_tensor: torch.Tensor, fog=0.65, noise_std=18.0, contrast=0.65) -> torch.Tensor:
    img = torch.clamp(img_tensor * contrast, 0.0, 1.0)
    noise = torch.randn_like(img) * (noise_std / 255.0)
    img = torch.clamp(img + noise, 0.0, 1.0)
    return add_fog_gpu(img, fog_strength=fog)


def rasterize_polygons(json_path: str, orig_shape=(1080, 1920)) -> np.ndarray:
    mask = np.full(orig_shape, VOID_LABEL, dtype=np.uint8)
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        for obj in data.get("objects", []):
            label = obj.get("label", "").lower()
            poly = np.array(obj.get("polygon", []), dtype=np.int32)
            if len(poly) < 3:
                continue
            class_id = LABEL_TO_ID.get(label, VOID_LABEL)
            cv2.fillPoly(mask, [poly], class_id)
    except Exception:
        pass
    return mask


def main():
    model = SegNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load 4 test samples
    img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "leftImg8bit", "val", "*", "*_leftImg8bit.jpg")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "leftImg8bit", "train", "*", "*_leftImg8bit.jpg")))

    samples = img_paths[:: len(img_paths) // 4][:4]

    fig, axes = plt.subplots(4, 5, figsize=(22, 14))
    plt.subplots_adjust(hspace=0.28, wspace=0.12)

    total_fog_correct, total_dehazed_correct, total_clean_correct, total_pixels = 0, 0, 0, 0
    fog_road_inter, fog_road_union = 0, 0
    dehazed_road_inter, dehazed_road_union = 0, 0
    clean_road_inter, clean_road_union = 0, 0

    print("[*] Processing Dehazing Recovery Pipeline...")

    for row, p in enumerate(samples):
        dir_name = os.path.dirname(p).replace("leftImg8bit", "gtFine")
        base_name = os.path.basename(p).replace("_leftImg8bit.jpg", "_gtFine_polygons.json")
        json_p = os.path.join(dir_name, base_name)

        img_bgr = cv2.imread(p)
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        lbl_gt = cv2.resize(rasterize_polygons(json_p, (img_bgr.shape[0], img_bgr.shape[1])), IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        # 1. Clean Tensor
        t_clean = torch.tensor(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

        # 2. Perturbed Fog Tensor
        t_fog = add_noise_and_haze(t_clean, fog=0.70, noise_std=20.0, contrast=0.60)
        img_fog_np = (t_fog.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # 3. Restored Dehazed Image (Our Innovation)
        img_dehazed_np = koschmieder_dehaze_and_enhance(img_fog_np)
        t_dehazed = torch.tensor(img_dehazed_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

        # Predictions
        with torch.no_grad():
            pred_clean = model(t_clean).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_fog = model(t_fog).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_dehazed = model(t_dehazed).argmax(dim=1).squeeze(0).cpu().numpy()

        # Plot 5 columns:
        # Col 0: Clean Camera
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"Scene {row+1}: Clean Camera", fontsize=10, fontweight="bold")
        axes[row, 0].axis("off")

        # Col 1: Degraded (Fog/Smog)
        axes[row, 1].imshow(img_fog_np)
        axes[row, 1].set_title("Perturbed (Smog + Fog + Noise)", fontsize=10, fontweight="bold", color="darkred")
        axes[row, 1].axis("off")

        # Col 2: Restored via Dehazing Filter
        axes[row, 2].imshow(img_dehazed_np)
        axes[row, 2].set_title("Restored (Koschmieder Inversion)", fontsize=10, fontweight="bold", color="blue")
        axes[row, 2].axis("off")

        # Col 3: Foggy Output (Distorted)
        axes[row, 3].imshow(pred_fog, cmap="tab10", vmin=0, vmax=7)
        axes[row, 3].set_title("Degraded Output (Without Filter)", fontsize=10, fontweight="bold", color="darkred")
        axes[row, 3].axis("off")

        # Col 4: Restored Output (Accurate SegNet!)
        axes[row, 4].imshow(pred_dehazed, cmap="tab10", vmin=0, vmax=7)
        axes[row, 4].set_title("Restored Output (With Filter)", fontsize=10, fontweight="bold", color="darkgreen")
        axes[row, 4].axis("off")

    output_img = os.path.join(OUTPUT_DIR, "dehazing_restoration_comparison.png")
    plt.savefig(output_img, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved High-Resolution Comparison Image -> {output_img}")

    # Now evaluate quantitative recovery across 300 test frames
    print("\n[*] Measuring Quantitative Recovery across Test Split...")
    val_paths = img_paths[:300]
    for p in tqdm(val_paths, desc="Benchmarking Filter"):
        dir_name = os.path.dirname(p).replace("leftImg8bit", "gtFine")
        base_name = os.path.basename(p).replace("_leftImg8bit.jpg", "_gtFine_polygons.json")
        json_p = os.path.join(dir_name, base_name)
        img_bgr = cv2.imread(p)
        if img_bgr is None or not os.path.isfile(json_p):
            continue

        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
        lbl_gt = cv2.resize(rasterize_polygons(json_p, (img_bgr.shape[0], img_bgr.shape[1])), IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        t_clean = torch.tensor(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        t_fog = add_noise_and_haze(t_clean, fog=0.70, noise_std=20.0, contrast=0.60)
        img_fog_np = (t_fog.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        img_dehazed_np = koschmieder_dehaze_and_enhance(img_fog_np)
        t_dehazed = torch.tensor(img_dehazed_np).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0

        with torch.no_grad():
            pred_clean = model(t_clean).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_fog = model(t_fog).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_dehazed = model(t_dehazed).argmax(dim=1).squeeze(0).cpu().numpy()

        valid = lbl_gt != VOID_LABEL
        total_pixels += valid.sum()
        total_clean_correct += (pred_clean[valid] == lbl_gt[valid]).sum()
        total_fog_correct += (pred_fog[valid] == lbl_gt[valid]).sum()
        total_dehazed_correct += (pred_dehazed[valid] == lbl_gt[valid]).sum()

        # Road IoU
        gt_road = lbl_gt == 0
        clean_road_inter += ((pred_clean == 0) & gt_road).sum()
        clean_road_union += ((pred_clean == 0) | gt_road).sum()
        fog_road_inter += ((pred_fog == 0) & gt_road).sum()
        fog_road_union += ((pred_fog == 0) | gt_road).sum()
        dehazed_road_inter += ((pred_dehazed == 0) & gt_road).sum()
        dehazed_road_union += ((pred_dehazed == 0) | gt_road).sum()

    acc_clean = (total_clean_correct / max(total_pixels, 1)) * 100
    acc_fog = (total_fog_correct / max(total_pixels, 1)) * 100
    acc_dehazed = (total_dehazed_correct / max(total_pixels, 1)) * 100

    road_iou_clean = (clean_road_inter / max(clean_road_union, 1)) * 100
    road_iou_fog = (fog_road_inter / max(fog_road_union, 1)) * 100
    road_iou_dehazed = (dehazed_road_inter / max(dehazed_road_union, 1)) * 100

    print("\n" + "=" * 55)
    print("      DEHAZING RESTORATION BENCHMARK RESULTS")
    print("=" * 55)
    print(f"1. Clean Conditions:      Accuracy = {acc_clean:.2f}% | Road IoU = {road_iou_clean:.2f}%")
    print(f"2. Degraded (Smog/Fog):   Accuracy = {acc_fog:.2f}% | Road IoU = {road_iou_fog:.2f}%  (-{acc_clean - acc_fog:.2f}% Drop)")
    print(f"3. Restored with Filter:  Accuracy = {acc_dehazed:.2f}% | Road IoU = {road_iou_dehazed:.2f}%  (+{acc_dehazed - acc_fog:.2f}% Recovery!)")
    print("=" * 55)


if __name__ == "__main__":
    main()
