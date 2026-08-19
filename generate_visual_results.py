"""Generates visual segmentation comparison plots: Input Image vs Ground Truth vs SegNet Prediction.
Saves the comparison image to D:\\Capstone\\outputs\\segnet_visual_predictions.png.
"""

import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

MODEL_PATH = r"D:\Capstone\models\segnet_idd20k_rtx3050.pth"
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

    # Find 4 test/val images
    img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "leftImg8bit", "val", "*", "*_leftImg8bit.jpg")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "leftImg8bit", "train", "*", "*_leftImg8bit.jpg")))

    samples = img_paths[:: len(img_paths) // 4][:4]
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    for row, p in enumerate(samples):
        dir_name = os.path.dirname(p).replace("leftImg8bit", "gtFine")
        base_name = os.path.basename(p).replace("_leftImg8bit.jpg", "_gtFine_polygons.json")
        json_p = os.path.join(dir_name, base_name)

        img = cv2.imread(p)
        lbl = rasterize_polygons(json_p, (img.shape[0], img.shape[1])) if os.path.isfile(json_p) else np.zeros((img.shape[0], img.shape[1]))

        img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        lbl_resized = cv2.resize(lbl, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        # Predict
        tensor_in = torch.tensor(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        with torch.no_grad():
            pred = model(tensor_in).argmax(dim=1).squeeze(0).cpu().numpy()

        # Display RGB
        axes[row, 0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"Input Camera Frame {row+1}", fontsize=11, fontweight="bold")
        axes[row, 0].axis("off")

        # Ground Truth Mask
        axes[row, 1].imshow(lbl_resized, cmap="tab10", vmin=0, vmax=7)
        axes[row, 1].set_title("Ground Truth Mask", fontsize=11, fontweight="bold")
        axes[row, 1].axis("off")

        # Model Prediction
        axes[row, 2].imshow(pred, cmap="tab10", vmin=0, vmax=7)
        axes[row, 2].set_title("SegNet Model Output (RTX 3050)", fontsize=11, fontweight="bold", color="darkgreen")
        axes[row, 2].axis("off")

    output_file = os.path.join(OUTPUT_DIR, "segnet_visual_predictions.png")
    plt.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved visual prediction image to -> {output_file}")


if __name__ == "__main__":
    main()
