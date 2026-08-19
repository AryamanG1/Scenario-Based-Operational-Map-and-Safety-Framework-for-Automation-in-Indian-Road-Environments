"""Evaluates trained SegNet model on the separate Validation and Test splits of IDD-20k Part II.
Computes pixel accuracy and mIoU (Mean Intersection over Union) across classes.
"""

import os
import glob
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

MODEL_PATH = r"D:\Capstone\models\segnet_idd20k_rtx3050.pth"
DATASET_DIR = r"D:\Capstone\data\idd20kII"
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


def evaluate_split(model: nn.Module, split_name: str, max_samples: int = 1000):
    print(f"\n[*] Evaluating on '{split_name}' split...")
    img_paths = sorted(glob.glob(os.path.join(DATASET_DIR, "leftImg8bit", split_name, "*", "*_leftImg8bit.jpg")))
    img_paths = img_paths[:max_samples]

    total_correct = 0
    total_pixels = 0
    intersection = np.zeros(NUM_CLASSES)
    union = np.zeros(NUM_CLASSES)

    for p in tqdm(img_paths, desc=f"Evaluating {split_name}"):
        dir_name = os.path.dirname(p).replace("leftImg8bit", "gtFine")
        base_name = os.path.basename(p).replace("_leftImg8bit.jpg", "_gtFine_polygons.json")
        json_p = os.path.join(dir_name, base_name)
        if not os.path.isfile(json_p):
            continue

        img = cv2.imread(p)
        if img is None:
            continue

        lbl = rasterize_polygons(json_p, (img.shape[0], img.shape[1]))
        img_resized = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        lbl_resized = cv2.resize(lbl, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

        tensor_in = torch.tensor(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        with torch.no_grad():
            pred = model(tensor_in).argmax(dim=1).squeeze(0).cpu().numpy()

        valid_mask = lbl_resized != VOID_LABEL
        total_correct += (pred[valid_mask] == lbl_resized[valid_mask]).sum()
        total_pixels += valid_mask.sum()

        for c in range(NUM_CLASSES - 1):
            pred_c = pred == c
            lbl_c = lbl_resized == c
            intersection[c] += (pred_c & lbl_c).sum()
            union[c] += (pred_c | lbl_c).sum()

    pixel_acc = total_correct / max(total_pixels, 1)
    iou_per_class = [intersection[c] / max(union[c], 1) for c in range(NUM_CLASSES - 1)]
    mean_iou = np.mean(iou_per_class)

    print(f"[+] {split_name.upper()} Pixel Accuracy: {pixel_acc * 100:.2f}%")
    print(f"[+] {split_name.upper()} Mean IoU (mIoU): {mean_iou * 100:.2f}%")
    print(f"[+] Road Class IoU: {iou_per_class[0] * 100:.2f}%")


def main():
    model = SegNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    evaluate_split(model, "val", max_samples=1000)


if __name__ == "__main__":
    main()
