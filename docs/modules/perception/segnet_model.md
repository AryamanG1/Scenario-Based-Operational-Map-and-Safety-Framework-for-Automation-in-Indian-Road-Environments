# `segnet_model.py`

**Stage:** 2 (Perception Layer)
**Package:** `src.perception.segnet_model`

## Purpose

This module defines, trains, and evaluates the SegNet semantic-segmentation model that underlies the rest of Stage 2. It provides a 3-stage Encoder-Decoder convolutional network that classifies every pixel of an Indian road scene into one of 8 classes (drivable-area, non-drivable-area, living-things, vehicles, road-side-objects, far-objects, sky, void), using BatchNorm for stable convergence and Dropout for regularization at the bottleneck.

Its output (the predicted per-pixel class mask) feeds directly into `feature_extraction.py` (road/non-drivable-area/living-things scene features), `lane_detection.py` (the road mask that constrains edge detection), and `detection_benchmark.py`'s ground-truth extraction — it is effectively the perception backbone the rest of the pipeline is built on.

## Paper / Formula Provenance

This module is **not** listed in the README's "Formula provenance" table. SegNet's 3-stage encoder/decoder-with-BatchNorm architecture is a standard, well-known convolutional segmentation network design pattern (not a formula reproduced from one of this project's 10 cited papers); the training loop (Adam optimizer, StepLR schedule, cross-entropy loss, train/val split) is original, ordinary engineering rather than a formula transcribed from a paper. No equation in this file is cited to a specific paper/equation number.

## Public API

### Module Constants

```python
NUM_CLASSES = 8
RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

`torch.manual_seed(RANDOM_STATE)` is called at import time for reproducibility. If no GPU is detected, a warning is printed: `"Warning: No GPU detected. segnet_model will run on CPU."`

### `SegNet`

```python
class SegNet(nn.Module):
    def __init__(self) -> None
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

3-stage Encoder-Decoder SegNet for 8-class semantic segmentation.

- **Encoder:** three `Conv2d → BatchNorm2d → ReLU → MaxPool2d(2)` blocks with channel widths 3→32, 32→64, 64→128 (kernel size 3, padding 1), followed by `Dropout2d(p=0.3)` at the bottleneck.
- **Decoder:** three `ConvTranspose2d(kernel_size=2, stride=2) → BatchNorm2d → ReLU` blocks mirroring the encoder back down to `NUM_CLASSES` (128→64→32→8), with no activation/BatchNorm after the final transpose-conv (raw logits out).
- `forward(x)`: `x` is a batch of shape `(N, 3, H, W)`; returns per-class logits of shape `(N, NUM_CLASSES, H, W)`.

### `load_segnet`

```python
def load_segnet(checkpoint_path: str, device: torch.device = DEVICE) -> SegNet
```

Loads a trained SegNet from a checkpoint file. Constructs a `SegNet`, loads `torch.load(checkpoint_path, map_location=device)` into it via `load_state_dict`, calls `.eval()`, and returns the model.

### `train_segnet`

```python
def train_segnet(
    images: np.ndarray,
    labels: np.ndarray,
    epochs: int = 30,
    batch_size: int = 8,
    save_path: str = "refined_segnet.pth",
) -> Tuple[List[float], List[float]]
```

Trains a SegNet model on cleaned IDD-Lite images/labels.

- **Args:** `images` — `uint8` array `(M, 224, 320, 3)`; `labels` — `uint8` array `(M, 224, 320)`; `epochs`, `batch_size`; `save_path` — where the trained `state_dict` is saved.
- **Returns:** `(train_losses, val_losses)`, one float per epoch.
- Converts images to a float tensor via `.permute(0, 3, 1, 2).float() / 255.0` and labels via `.long()`.
- Splits into train/val with `sklearn.model_selection.train_test_split(test_size=0.2, random_state=RANDOM_STATE)`.
- Optimizer: `Adam(lr=0.001)`; scheduler: `StepLR(step_size=10, gamma=0.5)`; loss: `nn.CrossEntropyLoss()`.
- Uses `torch.amp.GradScaler(DEVICE.type, enabled=(DEVICE.type == "cuda"))` and `torch.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda"))` for both training and validation forward passes — disabled (no-op) on CPU.
- Prints one line per epoch: `f"Epoch {epoch}/{epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"`.
- Saves the trained `state_dict` to `save_path` at the end and prints a confirmation.

### `plot_training_curves`

```python
def plot_training_curves(train_losses: List[float], val_losses: List[float]) -> None
```

Plots train vs. validation loss curves (`matplotlib`) and saves them to `<PLOTS_DIR>/segnet_training_curves.png`, creating `PLOTS_DIR` if needed. Returns nothing.

### `visualize_prediction`

```python
def visualize_prediction(
    model: nn.Module, image_tensor: torch.Tensor, label_tensor: torch.Tensor
) -> None
```

Visualizes a single SegNet prediction against its ground truth.

- **Args:** `model` — trained model; `image_tensor` — shape `(3, H, W)`, values in `[0, 1]`; `label_tensor` — shape `(H, W)`.
- Runs the model in `eval()`/`no_grad()` mode on the unsqueezed batch, takes `argmax(dim=1)` for the predicted mask.
- Converts the image tensor back to a displayable RGB numpy array via `.permute(1, 2, 0).cpu().numpy()[..., ::-1]` (channel flip, since the source images were loaded via `cv2` and are therefore BGR).
- Saves a 3-panel figure (input image / ground truth / prediction, ground truth and prediction rendered with the `tab20` colormap) to `<PLOTS_DIR>/segnet_prediction_sample.png`.

## Key Design Decisions & Edge Cases

- **CPU fallback is a deliberate, documented no-op, not a missing feature.** The module's own comment on the `GradScaler` line states: *"GradScaler/autocast are CUDA-only accelerations; enabled=False on CPU makes this a correct (if unaccelerated) no-op fallback per spec Rule #2."* This means training still runs correctly on machines without a discrete GPU (as noted in the top-level README's "Known scope limitations" — this project's development machine has no discrete GPU), just without mixed-precision speedup.
- **Reproducibility:** a fixed `torch.manual_seed(RANDOM_STATE)` (42) at import time and the same `RANDOM_STATE` used for the `train_test_split`.
- **BGR→RGB channel flip in visualization** (`[..., ::-1]`) exists specifically because upstream images are loaded with `cv2.imread`, which returns BGR, not RGB — without the flip, `visualize_prediction`'s input-image panel would render with swapped red/blue channels.
- **Dropout only at the bottleneck** (`Dropout2d(p=0.3)` at the end of the encoder) — the decoder has no dropout, a common regularization placement pattern for encoder-decoder segmentation nets to control overfitting without disrupting the reconstruction path.
- No explicit handling for `epochs=0` or empty `images`/`labels` arrays — the module assumes valid, non-empty, pre-cleaned input as produced by `data_pipeline.load_and_clean_dataset`.

## Dependencies

- **Standard library:** `os`, `typing.List`, `typing.Tuple`
- **External:** `matplotlib.pyplot`, `numpy`, `torch`, `torch.nn`, `sklearn.model_selection.train_test_split`, `torch.optim.Adam`, `torch.optim.lr_scheduler.StepLR`, `torch.utils.data.DataLoader`, `torch.utils.data.TensorDataset`
- **Internal:** `src.common.paths.PLOTS_DIR` (module level, used by `plot_training_curves`/`visualize_prediction`). The `__main__` block additionally imports `src.common.paths.DATA_DIR`, `src.common.paths.MODELS_DIR`, `src.common.paths.SEGNET_CHECKPOINT`, and `src.perception.data_pipeline.load_and_clean_dataset`.

## Usage Example

```python
from src.perception.data_pipeline import load_and_clean_dataset
from src.perception.segnet_model import train_segnet, load_segnet, plot_training_curves

images, labels = load_and_clean_dataset("data/idd20k_lite")
train_losses, val_losses = train_segnet(images, labels, epochs=30, batch_size=8, save_path="models/refined_segnet.pth")
plot_training_curves(train_losses, val_losses)

model = load_segnet("models/refined_segnet.pth")
```

## Running Standalone

```bash
python -m src.perception.segnet_model
```

Loads and cleans the dataset from `DATA_DIR`, trains a SegNet for 30 epochs (`batch_size=8`), saves the checkpoint to `SEGNET_CHECKPOINT`, plots training curves, reloads the saved model, and runs `visualize_prediction` on the first sample image, writing both output images under `outputs/plots/`.

## Tests

No dedicated `tests/perception/test_segnet_model.py` file exists. The module is exercised **indirectly**: `tests/conftest.py` defines a session-scoped `segnet` fixture that calls `load_segnet(SEGNET_CHECKPOINT)` (skipped if no checkpoint is present), and this fixture is consumed by `test_feature_extraction.py` and `test_lane_detection.py` to run real-image integration tests. `SegNet.forward`, `train_segnet`, `plot_training_curves`, and `visualize_prediction` themselves have no direct unit tests in this codebase.
