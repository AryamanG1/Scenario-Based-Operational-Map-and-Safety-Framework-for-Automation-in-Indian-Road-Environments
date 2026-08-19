"""Dedicated GPU training script for SegNet on the full IDD-20k Part II dataset.

Runs on local NVIDIA RTX 3050 GPU using PyTorch Automatic Mixed Precision (AMP).
Loads images from D:\\Capstone\\data\\idd20kII and saves checkpoints to D:\\Capstone\\models.
"""

import glob
import os
import time
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DATASET_DIR = r"D:\Capstone\data\idd20kII"
OUTPUT_DIR = r"D:\Capstone\outputs"
MODELS_DIR = r"D:\Capstone\models"
IMAGE_SIZE = (320, 224)  # (width, height)
NUM_CLASSES = 8
VOID_LABEL = 7
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Training Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
    vram = getattr(torch.cuda.get_device_properties(0), "total_memory", getattr(torch.cuda.get_device_properties(0), "total_mem", 4e9))
    print(f"[*] Total VRAM: {vram / 1e9:.2f} GB")


class SegNet(nn.Module):
    """3-stage Encoder-Decoder SegNet for 8-class Indian road semantic segmentation."""

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


LABEL_TO_ID = {
    "road": 0, "drivable": 0, "parking": 0,
    "sidewalk": 1, "curb": 1, "non-drivable": 1,
    "building": 2, "fence": 2, "guard rail": 2, "bridge": 2, "tunnel": 2, "pole": 2, "traffic light": 2, "traffic sign": 2,
    "vegetation": 3, "terrain": 3,
    "sky": 4,
    "person": 5, "rider": 5,
    "car": 6, "truck": 6, "bus": 6, "caravan": 6, "trailer": 6, "train": 6, "motorcycle": 6, "bicycle": 6, "autorickshaw": 6, "vehicle fallback": 6
}

import json

def rasterize_polygons(json_path: str, orig_shape: Tuple[int, int] = (1080, 1920)) -> np.ndarray:
    """Converts IDD polygon JSON annotations into an 8-class raster segmentation mask."""
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

def load_dataset(dataset_dir: str, max_samples: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
    """Loads and preprocesses images and polygon masks from IDD-20k Part II."""
    print("[*] Scanning dataset files...")
    search_pattern = os.path.join(dataset_dir, "leftImg8bit", "train", "*", "*_leftImg8bit.jpg")
    img_paths = sorted(glob.glob(search_pattern))

    print(f"[*] Found {len(img_paths)} raw images in training split.")
    img_paths = img_paths[:max_samples]

    clean_images, clean_labels = [], []
    for p in tqdm(img_paths, desc="Rasterizing & resizing"):
        dir_name = os.path.dirname(p).replace("leftImg8bit", "gtFine")
        base_name = os.path.basename(p).replace("_leftImg8bit.jpg", "_gtFine_polygons.json")
        json_p = os.path.join(dir_name, base_name)
        if not os.path.isfile(json_p):
            continue
        img = cv2.imread(p)
        if img is None:
            continue

        lbl = rasterize_polygons(json_p, (img.shape[0], img.shape[1]))
        if (lbl == 0).sum() == 0:  # Skip frames without road
            continue

        img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        lbl_resized = cv2.resize(lbl, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        clean_images.append(img_resized)
        clean_labels.append(lbl_resized)

    print(f"[+] Successfully paired and rasterized {len(clean_images)} samples!")
    return np.array(clean_images, dtype=np.uint8), np.array(clean_labels, dtype=np.uint8)


def train():
    images, labels = load_dataset(DATASET_DIR, max_samples=10000)
    if len(images) == 0:
        print("[!] No images loaded. Check dataset directory paths.")
        return

    images_t = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
    labels_t = torch.tensor(labels).long()

    train_idx, val_idx = train_test_split(np.arange(len(images_t)), test_size=0.2, random_state=RANDOM_STATE)
    train_loader = DataLoader(TensorDataset(images_t[train_idx], labels_t[train_idx]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(images_t[val_idx], labels_t[val_idx]), batch_size=BATCH_SIZE, shuffle=False)

    model = SegNet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))

    train_losses, val_losses = [], []
    save_path = os.path.join(MODELS_DIR, "segnet_idd20k_rtx3050.pth")

    print(f"\n[*] Starting SegNet GPU Training ({EPOCHS} Epochs, Batch Size: {BATCH_SIZE})...")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_train_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item() * imgs.size(0)
        train_loss = running_train_loss / len(train_idx)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                running_val_loss += loss.item() * imgs.size(0)
        val_loss = running_val_loss / len(val_idx)

        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"--> Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    total_time = time.time() - start_time
    print(f"\n[+] Training finished in {total_time/60:.2f} minutes!")
    torch.save(model.state_dict(), save_path)
    print(f"[+] Saved model checkpoint -> {save_path}")

    # Plot curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="royalblue", lw=2)
    plt.plot(val_losses, label="Val Loss", color="crimson", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("SegNet Training & Validation Curves (IDD-20k on RTX 3050)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(OUTPUT_DIR, "segnet_idd20k_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[+] Saved training plot -> {plot_path}")


if __name__ == "__main__":
    train()
