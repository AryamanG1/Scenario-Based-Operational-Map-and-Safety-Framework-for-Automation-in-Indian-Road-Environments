"""Module 2: SegNet semantic segmentation architecture, training, and evaluation.

Defines a 3-stage Encoder-Decoder SegNet for 8-class semantic segmentation of
Indian road scenes, using BatchNorm for stable convergence and Dropout for
regularization at the bottleneck.
"""

import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.common.paths import PLOTS_DIR

NUM_CLASSES = 8
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    print("Warning: No GPU detected. segnet_model will run on CPU.")


class SegNet(nn.Module):
    """3-stage Encoder-Decoder SegNet for 8-class semantic segmentation."""

    def __init__(self) -> None:
        """Initializes the encoder and decoder stacks."""
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
        """Runs a forward pass.

        Args:
            x: Input batch of shape (N, 3, H, W).

        Returns:
            Per-class logits of shape (N, NUM_CLASSES, H, W).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_segnet(checkpoint_path: str, device: torch.device = DEVICE) -> SegNet:
    """Loads a trained SegNet from a checkpoint file.

    Args:
        checkpoint_path: Path to a .pth state_dict saved by train_segnet.
        device: Device to load the model onto.

    Returns:
        A SegNet in eval() mode with the loaded weights.
    """
    model = SegNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def train_segnet(
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = 30,
    batch_size: int = 8,
    save_path: str = "refined_segnet.pth",
    class_weights: Optional[torch.Tensor] = None,
    initial_state_dict: Optional[dict] = None,
    lr: float = 0.001,
) -> Tuple[List[float], List[float]]:
    """Trains a SegNet model on cleaned IDD-Lite images/labels.

    Args:
        images: uint8 array of shape (M, 224, 320, 3).
        labels: uint8 array of shape (M, 224, 320).
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        save_path: Path to save the trained model's state_dict.
        class_weights: Optional per-class loss weights (length NUM_CLASSES),
            passed to nn.CrossEntropyLoss. If None, classes are weighted
            equally (the original behavior).
        initial_state_dict: Optional state_dict to initialize the model
            with before training starts, for fine-tuning an existing
            checkpoint instead of training from random initialization. If
            None, a freshly initialized SegNet is trained from scratch
            (the original behavior).
        lr: Adam learning rate.

    Returns:
        A tuple (train_losses, val_losses), one value per epoch.
    """
    images_t = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
    labels_t = torch.tensor(labels).long()

    train_idx, val_idx = train_test_split(
        np.arange(len(images_t)), test_size=0.2, random_state=RANDOM_STATE
    )
    train_ds = TensorDataset(images_t[train_idx], labels_t[train_idx])
    val_ds = TensorDataset(images_t[val_idx], labels_t[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SegNet().to(DEVICE)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE) if class_weights is not None else None
    )
    # GradScaler/autocast are CUDA-only accelerations; enabled=False on CPU
    # makes this a correct (if unaccelerated) no-op fallback per spec Rule #2.
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item() * imgs.size(0)
        train_loss = running_train_loss / len(train_ds)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                running_val_loss += loss.item() * imgs.size(0)
        val_loss = running_val_loss / len(val_ds)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    torch.save(model.state_dict(), save_path)
    print(f"Saved trained SegNet weights to '{save_path}'")

    return train_losses, val_losses


def plot_training_curves(train_losses: List[float], val_losses: List[float]) -> None:
    """Plots training vs. validation loss curves.

    Args:
        train_losses: Per-epoch training loss values.
        val_losses: Per-epoch validation loss values.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SegNet Training Curves")
    plt.legend()
    plt.grid(True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "segnet_training_curves.png"))
    plt.close()


def visualize_prediction(
    model: nn.Module, image_tensor: torch.Tensor, label_tensor: torch.Tensor
) -> None:
    """Visualizes a SegNet prediction against its ground truth.

    Args:
        model: Trained SegNet model.
        image_tensor: Single image tensor of shape (3, H, W), values in [0, 1].
        label_tensor: Ground-truth label tensor of shape (H, W).
    """
    model.eval()
    with torch.no_grad():
        input_batch = image_tensor.unsqueeze(0).to(DEVICE)
        output = model(input_batch)
        prediction = output.argmax(dim=1).squeeze(0).cpu().numpy()

    # Images are loaded via cv2 (BGR); flip channels for correct RGB display.
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()[..., ::-1]
    label_np = label_tensor.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(label_np, cmap="tab20")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(prediction, cmap="tab20")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "segnet_prediction_sample.png"))
    plt.close(fig)


if __name__ == "__main__":
    from src.common.paths import DATA_DIR, MODELS_DIR, SEGNET_CHECKPOINT
    from src.perception.data_pipeline import load_and_clean_dataset

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = SEGNET_CHECKPOINT

    images, labels = load_and_clean_dataset(DATA_DIR)
    train_losses, val_losses = train_segnet(images, labels, epochs=30, batch_size=8, save_path=save_path)
    plot_training_curves(train_losses, val_losses)

    model = load_segnet(save_path)
    sample_img = torch.tensor(images[0]).permute(2, 0, 1).float() / 255.0
    sample_lbl = torch.tensor(labels[0]).long()
    visualize_prediction(model, sample_img, sample_lbl)
