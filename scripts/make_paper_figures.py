"""Generates publication-ready figures for every stage of the ODD pipeline.

Run from the project directory AFTER `python main.py` has produced its
artifacts (SegNet checkpoint, feature CSVs, copula, feasibility map,
held-out evaluation JSON):

    python scripts/make_paper_figures.py                      # everything
    python scripts/make_paper_figures.py --only seg,perturb   # a subset
    python scripts/make_paper_figures.py --num_frames 8 --dpi 300

Each figure is written to outputs/paper_figures/ as a PNG (and a PDF, which
LaTeX prefers). Every section is independent and is skipped with a message
if its inputs are missing, so the script is safe to run on a checkout that
only has IDD-Lite.

Sections (--only keys):
    seg       Qualitative segmentation grid: image | ground truth | SegNet | error map
    perturb   One frame under each Stage 6 perturbation, with the predicted mask and IoU
    robust    mIoU-vs-strength curves per perturbation type, and per-region boxplots
    detect    YOLO detections vs annotated boxes on IDD117K val frames (TP / FN visible)
    odd       Stage 5 copula density histogram with cut points + 2-D region scatter
    feas      Stage 7 feasibility map: mode distribution and ODD score by mode
    fusion    Stage 2 multi-stream fusion weight distributions
    clf       Stage 7 classifier: per-class precision/recall/F1 vs majority baseline
"""

import argparse
import json
import os
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.common.paths import (  # noqa: E402
    DATA_DIR,
    FEASIBILITY_MAP_CSV,
    FEATURES_COMBINED_CSV,
    FEATURES_CSV,
    IDD117K_95K_DIR,
    IDD117K_DETECTION_DIR,
    IDD20K_DIR,
    ODD_CLASSIFIER_HOLDOUT_EVAL_JSON,
    ODD_COPULA_PATH,
    OUTPUTS_DIR,
    SEGNET_CHECKPOINT,
    SEGNET_IDD20K_CHECKPOINT,
    YOLO_WEIGHTS,
)
from src.monitoring.perturbation_engine import (  # noqa: E402
    DEVICE,
    REGIONAL_PROFILES,
    add_fog_gpu,
    add_gaussian_noise_gpu,
    adjust_brightness_contrast_gpu,
    calculate_metrics,
    perturb_image_gpu,
    sample_delta,
)

# IDD-Lite level3Id scheme (see feature_extraction.py). Colours are a
# colour-blind-safe set with sky/void muted so the road classes dominate.
CLASS_NAMES = [
    "drivable area", "non-drivable area", "living things", "vehicles",
    "road-side objects", "far objects", "sky", "void",
]
CLASS_COLORS = np.array(
    [
        [128, 64, 128],   # drivable: purple-grey
        [244, 35, 232],   # non-drivable: magenta
        [220, 20, 60],    # living things: crimson
        [0, 0, 142],      # vehicles: navy
        [250, 170, 30],   # road-side objects: orange
        [107, 142, 35],   # far objects: olive
        [70, 130, 180],   # sky: steel blue
        [0, 0, 0],        # void
    ],
    dtype=np.uint8,
)
NUM_CLASSES = 8

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 100,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _save(fig: plt.Figure, out_dir: str, name: str, dpi: int) -> None:
    png = os.path.join(out_dir, f"{name}.png")
    fig.savefig(png, dpi=dpi)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    plt.close(fig)
    print(f"  wrote {png}")


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return img[..., ::-1]


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    return CLASS_COLORS[np.clip(mask, 0, NUM_CLASSES - 1)]


def _class_legend(ax, classes=range(NUM_CLASSES), ncol=4, loc="lower center"):
    handles = [Patch(facecolor=CLASS_COLORS[c] / 255.0, label=CLASS_NAMES[c]) for c in classes]
    ax.legend(handles=handles, ncol=ncol, loc=loc, frameon=False, bbox_to_anchor=(0.5, -0.02))


def _load_segnet():
    from src.perception.segnet_model import load_segnet

    ckpt = SEGNET_IDD20K_CHECKPOINT if os.path.isfile(SEGNET_IDD20K_CHECKPOINT) else SEGNET_CHECKPOINT
    if not os.path.isfile(ckpt):
        return None, ckpt
    model = load_segnet(ckpt, device=DEVICE)
    model.eval()
    return model, ckpt


def _load_frames(num_frames: int, seed: int):
    """Frames WITH ground-truth masks: IDD-20K-II val if present, else IDD-Lite."""
    if os.path.isdir(IDD20K_DIR):
        from src.perception.idd20k_polygon_pipeline import load_split

        images, labels = load_split(IDD20K_DIR, "val", limit=max(num_frames, 200), seed=seed)
        source = "IDD-20K-II val"
    elif os.path.isdir(DATA_DIR):
        from src.perception.data_pipeline import load_and_clean_dataset

        images, labels = load_and_clean_dataset(DATA_DIR)
        source = "IDD-Lite"
    else:
        return None, None, "no dataset found"
    return images, labels, source


def _to_tensor_batch(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(images)).to(DEVICE).permute(0, 3, 1, 2).float() / 255.0


@torch.no_grad()
def _predict(segnet, batch01: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [0,1] on DEVICE -> (B,H,W) class ids on DEVICE."""
    return segnet(batch01).argmax(dim=1)


def _miou_batch(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return np.array([calculate_metrics(p, t, NUM_CLASSES)[0] for p, t in zip(pred, target)])


def _pick_indices(n_available: int, n_wanted: int, seed: int):
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_available), min(n_wanted, n_available)))


# --------------------------------------------------------------------------
# Section: segmentation grid
# --------------------------------------------------------------------------


def fig_segmentation_grid(images, labels, source, segnet, out_dir, num_frames, dpi, seed):
    idx = _pick_indices(len(images), num_frames, seed)
    imgs = images[idx]
    gts = torch.from_numpy(labels[idx].astype(np.int64)).to(DEVICE)
    preds = _predict(segnet, _to_tensor_batch(imgs))
    mious = _miou_batch(preds, gts)
    preds_np = preds.cpu().numpy()
    gts_np = gts.cpu().numpy()

    n = len(idx)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.0 * n + 0.8), squeeze=False)
    col_titles = ["Input frame", "Ground truth", "SegNet prediction", "Error map (red = wrong)"]
    for r in range(n):
        err = np.zeros((*gts_np[r].shape, 3), dtype=np.uint8) + 235
        err[preds_np[r] != gts_np[r]] = [200, 30, 30]
        err[gts_np[r] == 7] = [120, 120, 120]  # void pixels are not scored
        panels = [_bgr_to_rgb(imgs[r]), _mask_to_rgb(gts_np[r]), _mask_to_rgb(preds_np[r]), err]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c])
        axes[r, 0].set_ylabel(f"#{idx[r]}\nmIoU {mious[r]:.2f}", rotation=0, labelpad=28, va="center")
    legend_ax = fig.add_axes([0.1, 0.0, 0.8, 0.04])
    legend_ax.set_axis_off()
    _class_legend(legend_ax, ncol=4)
    fig.suptitle(f"SegNet semantic segmentation on {source} (mean mIoU over shown frames {mious.mean():.3f})", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    _save(fig, out_dir, "fig_segmentation_grid", dpi)


# --------------------------------------------------------------------------
# Section: perturbation grid (one frame)
# --------------------------------------------------------------------------


def fig_perturbation_grid(images, labels, segnet, out_dir, dpi, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    i = rng.randrange(len(images))
    base = _to_tensor_batch(images[i : i + 1])[0]
    gt = torch.from_numpy(labels[i].astype(np.int64)).to(DEVICE)

    variants = [
        ("Original", base),
        ("Brightness -60", adjust_brightness_contrast_gpu(base, delta_bright=-60, delta_contrast=1.0)),
        ("Contrast x0.6", adjust_brightness_contrast_gpu(base, delta_bright=0, delta_contrast=0.6)),
        ("Gaussian noise σ=25", add_gaussian_noise_gpu(base, std=25)),
        ("Fog 0.7", add_fog_gpu(base, fog_strength=0.7)),
        ("Punjab profile (sampled)", perturb_image_gpu(base, sample_delta("punjab"))),
    ]
    batch = torch.stack([v for _, v in variants])
    preds = _predict(segnet, batch)
    base_pred = preds[0]
    iou_gt = _miou_batch(preds, gt.unsqueeze(0).expand(len(variants), -1, -1))
    iou_base = _miou_batch(preds, base_pred.unsqueeze(0).expand(len(variants), -1, -1))

    n = len(variants)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.0))
    for c, (name, img) in enumerate(variants):
        axes[0, c].imshow(np.clip(img.permute(1, 2, 0).cpu().numpy(), 0, 1)[..., ::-1])
        axes[0, c].set_title(name)
        axes[1, c].imshow(_mask_to_rgb(preds[c].cpu().numpy()))
        axes[1, c].set_xlabel(f"mIoU vs GT {iou_gt[c]:.2f}\nvs baseline {iou_base[c]:.2f}")
        for r in range(2):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("Perturbed input")
    axes[1, 0].set_ylabel("SegNet prediction")
    fig.suptitle("Stage 6 perturbation engine: input degradations and their effect on segmentation", y=1.0)
    fig.tight_layout()
    _save(fig, out_dir, "fig_perturbation_grid", dpi)


# --------------------------------------------------------------------------
# Section: robustness curves + regional boxplots
# --------------------------------------------------------------------------


def fig_robustness(images, labels, segnet, out_dir, dpi, seed, num_frames=100, batch_size=32):
    torch.manual_seed(seed)
    idx = _pick_indices(len(images), num_frames, seed)
    imgs = images[idx]
    gts_all = torch.from_numpy(labels[idx].astype(np.int64)).to(DEVICE)

    sweeps = {
        "Brightness shift": ("brightness", np.linspace(-100, 100, 11)),
        "Contrast factor": ("contrast", np.linspace(0.4, 1.6, 11)),
        "Gaussian noise σ": ("noise", np.linspace(0, 40, 11)),
        "Fog strength": ("fog", np.linspace(0, 1, 11)),
    }

    def apply(batch, kind, val):
        if kind == "brightness":
            return adjust_brightness_contrast_gpu(batch, delta_bright=float(val), delta_contrast=1.0)
        if kind == "contrast":
            return adjust_brightness_contrast_gpu(batch, delta_bright=0.0, delta_contrast=float(val))
        if kind == "noise":
            return add_gaussian_noise_gpu(batch, std=float(val)) if val > 0 else batch
        return add_fog_gpu(batch, fog_strength=float(val)) if val > 0 else batch

    results = {}
    for title, (kind, grid) in sweeps.items():
        means, stds = [], []
        for val in grid:
            per_frame = []
            for s in range(0, len(imgs), batch_size):
                b = _to_tensor_batch(imgs[s : s + batch_size])
                p = _predict(segnet, apply(b, kind, val))
                per_frame.append(_miou_batch(p, gts_all[s : s + batch_size]))
            per_frame = np.concatenate(per_frame)
            means.append(per_frame.mean()); stds.append(per_frame.std())
        results[title] = (grid, np.array(means), np.array(stds))

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), sharey=True)
    for ax, (title, (grid, m, sd)) in zip(axes, results.items()):
        ax.plot(grid, m, marker="o", ms=3, color="#1f4e79")
        ax.fill_between(grid, m - sd, m + sd, alpha=0.2, color="#1f4e79")
        ax.set_xlabel(title); ax.grid(alpha=0.3)
    axes[0].set_ylabel("mIoU vs ground truth")
    fig.suptitle(f"SegNet robustness to each Stage 6 perturbation ({len(imgs)} frames, mean ± 1 s.d.)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig_robustness_curves", dpi)

    # Regional profiles: K sampled deltas per region, distribution of mIoU.
    region_scores = {}
    for region in REGIONAL_PROFILES:
        scores = []
        for _ in range(5):
            delta = sample_delta(region)
            for s in range(0, len(imgs), batch_size):
                b = _to_tensor_batch(imgs[s : s + batch_size])
                pert = torch.stack([perturb_image_gpu(x, delta) for x in b])
                scores.append(_miou_batch(_predict(segnet, pert), gts_all[s : s + batch_size]))
        region_scores[region.capitalize()] = np.concatenate(scores)
    clean = []
    for s in range(0, len(imgs), batch_size):
        clean.append(_miou_batch(_predict(segnet, _to_tensor_batch(imgs[s : s + batch_size])), gts_all[s : s + batch_size]))
    region_scores = {"Clean": np.concatenate(clean), **region_scores}

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.boxplot(list(region_scores.values()), tick_labels=list(region_scores.keys()), showfliers=False)
    ax.set_ylabel("mIoU vs ground truth"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Segmentation quality under regional weather profiles")
    fig.tight_layout()
    _save(fig, out_dir, "fig_robustness_regions", dpi)


# --------------------------------------------------------------------------
# Section: detection overlays
# --------------------------------------------------------------------------


def fig_detection_overlays(out_dir, num_frames, dpi, seed):
    import cv2

    from src.perception.detection_benchmark import compute_iou
    from src.perception.feature_extraction import CONF_THRESHOLD, load_yolo, run_detection_batch
    from src.perception.idd_detection_loader import iter_frame_batches, list_pairs, sample_pairs

    pairs = []
    for part in (IDD117K_95K_DIR, IDD117K_DETECTION_DIR):
        try:
            pairs += list_pairs(part, "val")
        except FileNotFoundError:
            pass
    if not pairs:
        print("  skip detect: no IDD117K val split found")
        return
    pairs = sample_pairs(pairs, num_frames, seed=seed)
    yolo = load_yolo(YOLO_WEIGHTS)

    images, boxes_per_image = next(iter_frame_batches(pairs, batch_size=len(pairs), num_workers=0))
    dets_per_image = run_detection_batch(images, yolo)

    n = len(images)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 2.4 * rows), squeeze=False)
    for k, (img, gts, dets) in enumerate(zip(images, boxes_per_image, dets_per_image)):
        canvas = img.copy()
        dets = [d for d in dets if d["confidence"] >= CONF_THRESHOLD]
        matched = set()
        for g in gts:
            x, y, w, h = g["bbox"]
            hit = any(compute_iou([x, y, w, h], d["bbox"]) >= 0.5 for d in dets)
            color = (60, 180, 60) if hit else (30, 30, 220)  # BGR: green if detected, red if missed
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
        for d in dets:
            x, y, w, h = d["bbox"]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (230, 160, 20), 1)  # YOLO: blue-ish
        ax = axes[k // cols, k % cols]
        ax.imshow(_bgr_to_rgb(canvas)); ax.set_xticks([]); ax.set_yticks([])
        n_hit = sum(any(compute_iou(g["bbox"], d["bbox"]) >= 0.5 for d in dets) for g in gts)
        ax.set_title(f"GT {len(gts)} | detected {n_hit} | YOLO boxes {len(dets)}", fontsize=8)
    for k in range(n, rows * cols):
        axes[k // cols, k % cols].set_axis_off()
    handles = [Patch(facecolor="#3cb43c", label="GT box, detected (IoU ≥ 0.5)"),
               Patch(facecolor="#dc1e1e", label="GT box, missed"),
               Patch(facecolor="#14a0e6", label="YOLOv8 detection")]
    fig.legend(handles=handles, ncol=3, loc="lower center", frameon=False)
    fig.suptitle("YOLOv8n detections vs IDD117K annotations (320×224 input)", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    _save(fig, out_dir, "fig_detection_overlays", dpi)


# --------------------------------------------------------------------------
# Section: ODD copula density
# --------------------------------------------------------------------------


def fig_odd_density(out_dir, dpi, seed):
    import joblib

    from src.odd.copula_gpu import odd_density_batch
    from src.odd.odd_boundary import density_cutpoints

    features_csv = FEATURES_COMBINED_CSV if os.path.isfile(FEATURES_COMBINED_CSV) else FEATURES_CSV
    if not (os.path.isfile(ODD_COPULA_PATH) and os.path.isfile(features_csv)):
        print("  skip odd: copula or features CSV missing")
        return
    copula = joblib.load(ODD_COPULA_PATH)
    df = pd.read_csv(features_csv)
    dens = odd_density_batch(copula, df)
    p15, p50 = density_cutpoints(copula)
    region = np.where(dens < p15, "outside", np.where(dens < p50, "near", "within"))
    log_d = np.log10(np.clip(dens, 1e-300, None))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    ax.hist(log_d, bins=80, color="#8da0cb", edgecolor="none")
    for v, name, c in ((p15, "p15 (outside | near)", "#d62728"), (p50, "p50 (near | within)", "#ff7f0e")):
        ax.axvline(np.log10(max(v, 1e-300)), color=c, ls="--", label=name)
    ax.set_xlabel("log10 copula density φ_ODD(x)"); ax.set_ylabel("frames"); ax.legend(frameon=False)
    ax.set_title(f"ODD-space density over {len(df):,} frames")

    ax = axes[1]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(6000, len(df)), replace=False)
    colors = {"within": "#2ca02c", "near": "#ff7f0e", "outside": "#d62728"}
    xv, yv = copula.variables[0], copula.variables[1]
    for name in ("within", "near", "outside"):
        m = region[idx] == name
        ax.scatter(df[xv].to_numpy()[idx][m], df[yv].to_numpy()[idx][m], s=4, alpha=0.5, c=colors[name], label=f"{name} ({(region == name).mean():.0%})")
    ax.set_xlabel(xv); ax.set_ylabel(yv); ax.set_xscale("symlog"); ax.legend(frameon=False, markerscale=3)
    ax.set_title("ODD region by two of the copula variables")
    fig.tight_layout()
    _save(fig, out_dir, "fig_odd_density", dpi)


# --------------------------------------------------------------------------
# Section: feasibility map
# --------------------------------------------------------------------------


def fig_feasibility(out_dir, dpi):
    if not os.path.isfile(FEASIBILITY_MAP_CSV):
        print("  skip feas: feasibility_map.csv missing")
        return
    df = pd.read_csv(FEASIBILITY_MAP_CSV)
    order = ["Normal", "Degraded", "Takeover"]
    colors = {"Normal": "#2ca02c", "Degraded": "#ff7f0e", "Takeover": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    counts = df["final_mode"].value_counts().reindex(order).fillna(0)
    axes[0].bar(order, counts.values, color=[colors[m] for m in order])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v / len(df):.1%}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Final mode distribution"); axes[0].set_ylabel("frames")

    for m in order:
        sub = df.loc[df["final_mode"] == m, "odd_score"]
        if len(sub):
            axes[1].hist(sub, bins=40, alpha=0.6, color=colors[m], label=m)
    axes[1].set_xlabel("ODD score (0–100)"); axes[1].legend(frameon=False); axes[1].set_title("ODD score by final mode")

    ct = pd.crosstab(df["odd_region"], df["final_mode"]).reindex(columns=order).fillna(0)
    ct.plot(kind="bar", stacked=True, ax=axes[2], color=[colors[m] for m in order], width=0.7)
    axes[2].set_xlabel("Stage 5 ODD region"); axes[2].set_ylabel("frames"); axes[2].set_title("Region → mode composition")
    axes[2].legend(frameon=False); axes[2].tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, out_dir, "fig_feasibility_map", dpi)


# --------------------------------------------------------------------------
# Section: fusion weights
# --------------------------------------------------------------------------


def fig_fusion_weights(out_dir, dpi):
    from src.perception.multi_stream_fusion import STREAMS, calibrate_references, fuse_dataframe

    features_csv = FEATURES_COMBINED_CSV if os.path.isfile(FEATURES_COMBINED_CSV) else FEATURES_CSV
    if not os.path.isfile(features_csv):
        print("  skip fusion: features CSV missing")
        return
    df = pd.read_csv(features_csv)
    fused = fuse_dataframe(df, calibrate_references(df))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    axes[0].boxplot([fused[f"weight_{s}"] for s in STREAMS], tick_labels=[s.replace("_", "\n") for s in STREAMS], showfliers=False)
    axes[0].set_ylabel("fusion weight w_s"); axes[0].set_title("Per-stream fusion weights")
    axes[1].hist(fused["fused_confidence"], bins=50, color="#66c2a5")
    axes[1].set_xlabel("fused confidence"); axes[1].set_title(f"Fused perception confidence ({len(df):,} frames)")
    fig.tight_layout()
    _save(fig, out_dir, "fig_fusion_weights", dpi)


# --------------------------------------------------------------------------
# Section: classifier report
# --------------------------------------------------------------------------


def fig_classifier_report(out_dir, dpi):
    if not os.path.isfile(ODD_CLASSIFIER_HOLDOUT_EVAL_JSON):
        print("  skip clf: held-out evaluation JSON missing")
        return
    with open(ODD_CLASSIFIER_HOLDOUT_EVAL_JSON) as handle:
        m = json.load(handle)
    rep = m["classification_report"]
    classes = [c for c in ("Normal", "Degraded", "Takeover") if c in rep]
    metrics = ["precision", "recall", "f1-score"]
    vals = np.array([[rep[c][k] for k in metrics] for c in classes])

    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(classes)); w = 0.26
    for j, k in enumerate(metrics):
        ax.bar(x + (j - 1) * w, vals[:, j], w, label=k)
    ax.axhline(m["accuracy"], color="k", ls="-", lw=1, label=f"accuracy {m['accuracy']:.3f}")
    if "majority_class_baseline_accuracy" in m:
        ax.axhline(m["majority_class_baseline_accuracy"], color="grey", ls="--", lw=1, label=f"majority baseline {m['majority_class_baseline_accuracy']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(n={int(rep[c]['support'])})" for c in classes])
    ax.set_ylim(0, 1.05); ax.set_ylabel("score"); ax.legend(frameon=False, ncol=2, fontsize=7)
    src = m.get("label_source", "rule")
    ax.set_title(f"ODD classifier on held-out val ({m['num_holdout_rows']:,} rows; labels: {'annotations' if src == 'gt' else 'rule'})")
    fig.tight_layout()
    _save(fig, out_dir, "fig_classifier_holdout", dpi)


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default=os.path.join(OUTPUTS_DIR, "paper_figures"))
    parser.add_argument("--num_frames", type=int, default=6, help="Frames in the qualitative grids.")
    parser.add_argument("--robust_frames", type=int, default=100, help="Frames for the robustness sweeps.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", default="", help="Comma-separated subset of: seg,perturb,robust,detect,odd,feas,fusion,clf")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else {"seg", "perturb", "robust", "detect", "odd", "feas", "fusion", "clf"}
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"Device: {DEVICE}. Writing to {args.out_dir}")

    if wanted & {"seg", "perturb", "robust"}:
        segnet, ckpt = _load_segnet()
        images, labels, source = _load_frames(max(args.num_frames, args.robust_frames), args.seed)
        if segnet is None or images is None:
            print(f"  skip seg/perturb/robust: checkpoint '{ckpt}' or dataset missing ({source})")
        else:
            print(f"SegNet: {ckpt}; frames: {len(images)} from {source}")
            if "seg" in wanted:
                fig_segmentation_grid(images, labels, source, segnet, args.out_dir, args.num_frames, args.dpi, args.seed)
            if "perturb" in wanted:
                fig_perturbation_grid(images, labels, segnet, args.out_dir, args.dpi, args.seed)
            if "robust" in wanted:
                fig_robustness(images, labels, segnet, args.out_dir, args.dpi, args.seed, num_frames=args.robust_frames)
    if "detect" in wanted:
        fig_detection_overlays(args.out_dir, args.num_frames, args.dpi, args.seed)
    if "odd" in wanted:
        fig_odd_density(args.out_dir, args.dpi, args.seed)
    if "feas" in wanted:
        fig_feasibility(args.out_dir, args.dpi)
    if "fusion" in wanted:
        fig_fusion_weights(args.out_dir, args.dpi)
    if "clf" in wanted:
        fig_classifier_report(args.out_dir, args.dpi)
    print("Done.")


if __name__ == "__main__":
    main()
