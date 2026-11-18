"""
Benchmark script for evaluating models trained by:
  1. terratorch (SemanticSegmentationTask checkpoints)
  2. geoai/smp  (torch .pth model files)

on the test split of each tile category:
  BASE_DIR/<category>/test

Outputs a summary CSV + per-model metrics printed to stdout.

Paths
-----
BASE_DIR  tile root      (env, default: data/sentinal_datastore)
RUNS_DIR  run root       (env, default: BASE_DIR/runs)

The public helpers (TifSegmentationDataset, benchmark_model, get_batch_viz_data,
load_terratorch_model, load_geoai_smp_model, predict_terratorch, predict_smp)
are also used by src/evaluate-ms.ipynb and src/evaluate-ms-viz.ipynb.
"""

import argparse
import glob
import json
import multiprocessing as mp
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

BASE_DIR = Path(os.environ.get("BASE_DIR", "data/sentinal_datastore"))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", BASE_DIR / "runs"))

NUM_CLASSES = 2
CLASS_NAMES = ["Not Mine", "Mine"]
MINE_CLASS = 1

# ---------------------------------------------------------------------------
# Terratorch input definition (must match src/terratorch_segment.py).
# Bands are stored B02, B03, B04 for the 3-channel tiles.
# ---------------------------------------------------------------------------
TERRATORCH_SETUP = {
    3: {
        "means": [0.10808051888921406, 0.11986660345662084, 0.12866614283181937],
        "stds": [0.1934757183721109, 0.18732491116331573, 0.19596228303648386],
    },
    4: {
        "means": [
            0.12454803971715495,
            0.11744241986092058,
            0.10710630159373255,
            0.20427242687267635,
        ],
        "stds": [
            0.19299162071219028,
            0.18615814864941185,
            0.19368826952679166,
            0.20699387702969146,
        ],
    },
    6: {
        "means": [
            0.12461692144122513,
            0.11748835596513783,
            0.10716054885611168,
            0.20447855444772434,
            0.1717338935634562,
            0.12472681998494414,
        ],
        "stds": [
            0.19305906501863254,
            0.18622984697099554,
            0.19376097639648635,
            0.20699650249364526,
            0.12587038325968888,
            0.10388463481088178,
        ],
    },
}
TERRATORCH_SCALE = 1 / 10000.0

# geoai/smp training (geoai.train_segmentation_model) scales by 1/255 and does
# not apply any mean/std normalisation -> replicate exactly here.
GEOAI_SCALE = 1 / 255.0

# ---------------------------------------------------------------------------
# Which run directories hold models trained on which tile category.
# ---------------------------------------------------------------------------
CATEGORY_SPECS = [
    {
        "category": "rgb+ve_tiles_1024",
        "num_channels": 3,
        "geoai_runs": [
            "rgb+ve_1024_lr1e-4_epochs200",
            "nc3_bs8_lr1e-4_epochs200",
        ],
        "terratorch_ckpts": [
            "rgb+ve_1024/prithvi_frozen_unet_lr1e-4_ce_.25.75_v0/checkpoints/epoch=49-step=44700.ckpt",
            "rgb+ve_1024/prithvi_lr1e-5_unet_lr1e-4_ce_.25.75/checkpoints/epoch=99-step=89400.ckpt",
            "rgb+ve_1024/prithvi_lr1e-5unfreeze10_unet_lr1e-4_ce_.25.75/best-epoch=21-val_loss=0.0000.ckpt",
        ],
    },
    {
        "category": "rgb_nir+ve_tiles_1024",
        "num_channels": 4,
        "geoai_runs": [
            "rgb_nir+ve_tiles_1024_lr1e-4_epochs200",
            "nc4_bs4_lr1e-4_epochs200",
        ],
        "terratorch_ckpts": [
            "rgb_nir+ve_tiles_1024/prithvi_frozen_unet_lr1e-4_ce_.25.75/best-epoch=11-val_loss=0.0000.ckpt",
        ],
    },
    {
        "category": "rgb_irs+ve_tiles_1024",
        "num_channels": 6,
        "geoai_runs": [
            "rgb_irs+ve_tiles_1024_lr1e-4_epochs200",
            "nc6_bs4_lr1e-4_epochs200",
        ],
        "terratorch_ckpts": [
            "rgb_irs+ve_tiles_1024/prithvi_frozen_unet_lr1e-4_ce_.25.75/best-epoch=14-val_loss=0.0000.ckpt",
        ],
    },
    {
        "category": "all_bands+ve_tiles_1024",
        "num_channels": 15,
        "geoai_runs": ["nc15_bs2_lr1e-4_epochs200"],
        "terratorch_ckpts": [],
    },
]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_confusion(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = -1):
    """Return a (num_classes, num_classes) confusion matrix."""
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt_cls in range(num_classes):
        for pred_cls in range(num_classes):
            cm[gt_cls, pred_cls] = np.sum((target == gt_cls) & (pred == pred_cls))
    return cm


def _min_pixels_from_area(min_area_m2: float, pixel_area_m2: float) -> int:
    """Convert a min area in m2 to a min pixel count (ceil)."""
    if pixel_area_m2 <= 0:
        raise ValueError("pixel_area_m2 must be > 0")
    return int(np.ceil(min_area_m2 / pixel_area_m2))


def _label_connected_components(binary_mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Return labeled components and count for a binary mask."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    try:
        from scipy import ndimage
    except Exception as exc:
        raise ImportError("scipy is required for connected components") from exc

    if connectivity == 4:
        structure = np.array(
            [[0, 1, 0],
             [1, 1, 1],
             [0, 1, 0]],
            dtype=np.int8,
        )
    else:
        structure = np.ones((3, 3), dtype=np.int8)

    labeled, num = ndimage.label(binary_mask.astype(np.uint8), structure=structure)
    return labeled, int(num)


def _binary_mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert a binary mask to COCO RLE."""
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _append_coco_instances(
    binary_mask: np.ndarray,
    image_id: int,
    category_id: int,
    min_pixels: int,
    connectivity: int,
    ann_id_start: int,
    score_map: np.ndarray | None = None,
    is_gt: bool = True,
) -> tuple[list[dict], int]:
    """Create COCO-style annotations or detections from a binary mask."""
    from pycocotools import mask as mask_utils

    labeled, num = _label_connected_components(binary_mask, connectivity=connectivity)
    instances: list[dict] = []
    ann_id = ann_id_start

    for label_id in range(1, num + 1):
        comp = labeled == label_id
        if comp.sum() < min_pixels:
            continue

        rle = _binary_mask_to_rle(comp)
        area = float(mask_utils.area(rle))
        bbox = mask_utils.toBbox(rle).tolist()

        if is_gt:
            ann = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
            ann_id += 1
        else:
            score = 1.0
            if score_map is not None:
                score = float(score_map[comp].mean())
            ann = {
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "bbox": bbox,
                "score": score,
            }
        instances.append(ann)

    return instances, ann_id


def _compute_coco_map(
    images: list[dict],
    gt_anns: list[dict],
    pred_anns: list[dict],
    category_id: int,
) -> dict:
    """Compute COCO-style mAP for bbox and mask."""
    if not images or not gt_anns:
        return {
            "Bbox_mAP_50_95": 0.0,
            "Bbox_mAP_50": 0.0,
            "Mask_mAP_50_95": 0.0,
            "Mask_mAP_50": 0.0,
        }

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": gt_anns,
        "categories": [{"id": category_id, "name": "mine"}],
    }
    coco_gt.createIndex()

    if not pred_anns:
        return {
            "Bbox_mAP_50_95": 0.0,
            "Bbox_mAP_50": 0.0,
            "Mask_mAP_50_95": 0.0,
            "Mask_mAP_50": 0.0,
        }

    coco_dt = coco_gt.loadRes(pred_anns)

    bbox_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    bbox_eval.evaluate()
    bbox_eval.accumulate()
    bbox_eval.summarize()

    segm_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    segm_eval.evaluate()
    segm_eval.accumulate()
    segm_eval.summarize()

    return {
        "Bbox_mAP_50_95": float(bbox_eval.stats[0]),
        "Bbox_mAP_50": float(bbox_eval.stats[1]),
        "Mask_mAP_50_95": float(segm_eval.stats[0]),
        "Mask_mAP_50": float(segm_eval.stats[1]),
    }


def metrics_from_confusion(cm: np.ndarray, class_names: list[str] | None = None):
    """Derive per-class and aggregate metrics from confusion matrix."""
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    results = {}
    per_class_iou = []
    per_class_f1 = []
    per_class_precision = []
    per_class_recall = []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_iou.append(iou)
        per_class_f1.append(f1)
        per_class_precision.append(precision)
        per_class_recall.append(recall)

        results[f"IoU_{class_names[c]}"] = iou
        results[f"F1_{class_names[c]}"] = f1
        results[f"Precision_{class_names[c]}"] = precision
        results[f"Recall_{class_names[c]}"] = recall

    overall_acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
    results["Overall_Accuracy"] = overall_acc
    results["mIoU"] = np.mean(per_class_iou)
    results["mF1"] = np.mean(per_class_f1)
    results["mPrecision"] = np.mean(per_class_precision)
    results["mRecall"] = np.mean(per_class_recall)
    return results


# ---------------------------------------------------------------------------
# Test dataset (reads .tif pairs from images/ and masks/)
# ---------------------------------------------------------------------------

class TifSegmentationDataset(Dataset):
    """
    Reads .tif images and corresponding mask .tifs.
    Returns (image_tensor [C,H,W] float32, mask_tensor [H,W] int64, filename).
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        means: list[float] | None = None,
        stds: list[float] | None = None,
        constant_scale: float = 1.0,
        max_items: int | None = None,
    ):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.means = np.array(means, dtype=np.float32) if means else None
        self.stds = np.array(stds, dtype=np.float32) if stds else None
        self.constant_scale = constant_scale

        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.tif")))
        assert len(self.image_paths) > 0, f"No images found in {images_dir}"
        if max_items is not None:
            self.image_paths = self.image_paths[:max_items]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        fname = os.path.basename(img_path)
        mask_path = os.path.join(self.masks_dir, fname)

        with rasterio.open(img_path) as src:
            # (C, H, W) – all bands, in file order
            img = src.read().astype(np.float32)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        # Scale
        img *= self.constant_scale

        # Normalise
        if self.means is not None and self.stds is not None:
            for c in range(img.shape[0]):
                img[c] = (img[c] - self.means[c]) / self.stds[c]

        return torch.from_numpy(img), torch.from_numpy(mask), fname


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_terratorch_model(checkpoint_path: str, device: torch.device):
    """Load a terratorch SemanticSegmentationTask from a Lightning checkpoint."""
    import terratorch  # noqa: F401 – registers models
    from terratorch.tasks import SemanticSegmentationTask

    task = SemanticSegmentationTask.load_from_checkpoint(checkpoint_path, map_location=device)
    task.eval()
    task.to(device)
    return task


def _infer_in_channels(state_dict: dict) -> int | None:
    """Infer the number of input channels from a checkpoint's first conv weight."""
    for key in ("encoder.conv1.weight", "encoder.model.conv1.weight", "model.encoder.conv1.weight"):
        w = state_dict.get(key)
        if w is not None and hasattr(w, "shape") and len(w.shape) == 4:
            return int(w.shape[1])
    return None


def load_geoai_smp_model(model_dir: str, num_channel: int, device: torch.device):
    """
    Load a geoai / segmentation_models_pytorch model.
    geoai.train_segmentation_model saves:
      - best_model.pth  (state dict)
      - training_summary.txt / training_config.json (optional)
    The model is rebuilt from the config, or inferred from the directory name
    ("<architecture>_<encoder>", e.g. "segformer_mit_b1").
    """
    import segmentation_models_pytorch as smp

    # Try to find config
    config_path = None
    for candidate in ["training_config.json", "model_config.json", "config.json"]:
        p = os.path.join(model_dir, candidate)
        if os.path.isfile(p):
            config_path = p
            break

    if config_path is not None:
        with open(config_path) as f:
            cfg = json.load(f)
        architecture = cfg.get("architecture", "unet")
        encoder_name = cfg.get("encoder_name", "resnet50")
        num_classes = cfg.get("num_classes", NUM_CLASSES)
        num_channels = cfg.get("num_channels", num_channel)
        encoder_weights = None  # we load our own weights
    else:
        # Fallback: infer from directory name  e.g. "unet_resnet50"
        dirname = os.path.basename(model_dir)
        parts = dirname.split("_", 1)
        architecture = parts[0] if len(parts) >= 1 else "unet"
        encoder_name = parts[1] if len(parts) >= 2 else "resnet50"
        num_classes = NUM_CLASSES
        num_channels = num_channel
        encoder_weights = None

    # Build model
    arch_fn = {
        "unet": smp.Unet,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "segformer": getattr(smp, "Segformer", None),
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
    }

    builder = arch_fn.get(architecture.lower())
    if builder is None:
        # generic fallback via smp.create_model
        model = smp.create_model(
            architecture, encoder_name=encoder_name,
            in_channels=num_channels, classes=num_classes,
            encoder_weights=encoder_weights,
        )
    else:
        model = builder(
            encoder_name=encoder_name,
            in_channels=num_channels,
            classes=num_classes,
            encoder_weights=encoder_weights,
        )

    # Load weights
    weight_path = None
    for candidate in ["best_model.pth", "model.pth", "checkpoint.pth"]:
        p = os.path.join(model_dir, candidate)
        if os.path.isfile(p):
            weight_path = p
            break
    # Also check for .pt files
    if weight_path is None:
        for candidate in glob.glob(os.path.join(model_dir, "*.pth")) + glob.glob(os.path.join(model_dir, "*.pt")):
            weight_path = candidate
            break

    if weight_path is None:
        raise FileNotFoundError(f"No model weights found in {model_dir}")

    state = torch.load(weight_path, map_location=device, weights_only=False)
    # handle case where full checkpoint dict is saved
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # The checkpoint knows how many bands it was trained on - fail early and
    # clearly if it does not match the tiles we are about to feed it.
    ckpt_channels = _infer_in_channels(state)
    if ckpt_channels is not None and ckpt_channels != num_channels:
        raise ValueError(
            f"Channel mismatch: {os.path.basename(model_dir)} was trained on "
            f"{ckpt_channels} bands but the test tiles have {num_channels}. "
            "This model belongs to a different category."
        )

    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Inference wrappers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_terratorch(task, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Returns (B, H, W) int64 predicted class indices.
    Terratorch task.model returns a ModelOutput with .output
    """
    images = images.to(device)
    output = task.model(images)
    # ModelOutput → .output is logits (B, C, H, W)
    logits = output.output if hasattr(output, "output") else output
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    # Ensure spatial dims match input
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return logits.cpu()


@torch.no_grad()
def predict_smp(model, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Returns (B, H, W) int64 predicted class indices."""
    images = images.to(device)
    logits = model(images)  # (B, C, H, W)
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
    return logits.cpu()


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def benchmark_model(
    model_name: str,
    predict_logits_fn,
    model,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    class_names: list[str] | None = None,
) -> dict:
    """Run inference on the full test set and return metric dict."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images_batch, masks, fnames in tqdm(dataloader, desc=f"  {model_name}", leave=False):
        logits = predict_logits_fn(model, images_batch, device)  # (B, C, H, W)
        preds = logits.argmax(dim=1).numpy()
        masks_np = masks.numpy()

        for pred, mask in zip(preds, masks_np):
            cm += compute_confusion(pred, mask, num_classes=num_classes, ignore_index=-1)

    metrics = metrics_from_confusion(cm, class_names=class_names)
    metrics["model"] = model_name
    return metrics


def get_batch_viz_data(
    model_name: str,
    predict_logits_fn,
    model,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    class_names: list[str] | None = None,
    viz_idx: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, dict | None]:
    """Return (images_np, masks_np, preds_np), viz_metrics for a specific batch index."""
    for batch_index, (images_batch, masks, fnames) in enumerate(
        tqdm(dataloader, desc=f"  {model_name} (viz)", leave=False)
    ):
        if batch_index != viz_idx:
            continue

        logits = predict_logits_fn(model, images_batch, device)  # (B, C, H, W)
        preds = logits.argmax(dim=1).numpy()
        masks_np = masks.numpy()

        batch_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for pred, mask in zip(preds, masks_np):
            batch_cm += compute_confusion(pred, mask, num_classes=num_classes, ignore_index=-1)
        viz_metrics = metrics_from_confusion(batch_cm, class_names=class_names)

        images_np = images_batch.cpu().numpy()
        return (images_np, masks_np, preds), viz_metrics

    return None, None, None, None


def benchmark_model_with_coco(
    model_name: str,
    predict_logits_fn,
    model,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    class_names: list[str] | None = None,
    visualize: bool = False,
    mine_class: int = 1,
    cca_connectivity: int = 8,
    cca_min_area_m2: float = 0,
    pixel_area_m2: float = 100.0,
    prob_threshold: float = 0.5,
) -> dict:
    """Run inference on the full test set and return pixel + COCO metrics."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    best_batch_score = -1.0
    best_batch_index = -1
    best_batch_data = None
    images: list[dict] = []
    gt_anns: list[dict] = []
    pred_anns: list[dict] = []
    image_id_by_name: dict[str, int] = {}
    ann_id = 1
    min_pixels = _min_pixels_from_area(cca_min_area_m2, pixel_area_m2)
    print(f"Using min_pixels={min_pixels} for COCO instance filtering")
    print(f"Using prob_threshold={prob_threshold:.2f} for CCA binarization")
    total_gt_instances = 0
    total_pred_instances = 0
    total_scores: list[float] = []
    logged_images = 0

    for batch_index, (images_batch, masks, fnames) in enumerate(
        tqdm(dataloader, desc=f"  {model_name}", leave=False)
    ):
        logits = predict_logits_fn(model, images_batch, device)  # (B, C, H, W)
        preds = logits.argmax(dim=1).numpy()
        probs = torch.softmax(logits, dim=1)[:, mine_class].numpy()
        masks_np = masks.numpy()

        batch_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for i in range(len(preds)):
            gt_binary = masks_np[i] == mine_class
            pred_binary = probs[i] >= prob_threshold
            batch_cm += compute_confusion(pred_binary, gt_binary, num_classes=num_classes, ignore_index=-1)
        batch_metrics = metrics_from_confusion(batch_cm, class_names=class_names)
        batch_score = float(batch_metrics.get("mIoU", 0.0))
        if batch_score > best_batch_score:
            best_batch_score = batch_score
            best_batch_index = batch_index
            best_batch_data = (images_batch.cpu().numpy(), masks_np, preds)

        for i, (pred, mask) in enumerate(zip(preds, masks_np)):
            fname = fnames[i]
            if fname not in image_id_by_name:
                image_id = len(images) + 1
                image_id_by_name[fname] = image_id
                h, w = mask.shape
                images.append({
                    "id": image_id,
                    "width": int(w),
                    "height": int(h),
                    "file_name": fname,
                })
            image_id = image_id_by_name[fname]
            gt_binary = mask == mine_class
            pred_binary = probs[i] >= prob_threshold

            cm += compute_confusion(pred_binary, gt_binary, num_classes=num_classes, ignore_index=-1)
            gt_instances, ann_id = _append_coco_instances(
                gt_binary,
                image_id=image_id,
                category_id=mine_class,
                min_pixels=min_pixels,
                connectivity=cca_connectivity,
                ann_id_start=ann_id,
                score_map=None,
                is_gt=True,
            )
            pred_instances, _ = _append_coco_instances(
                pred_binary,
                image_id=image_id,
                category_id=mine_class,
                min_pixels=min_pixels,
                connectivity=cca_connectivity,
                ann_id_start=0,
                score_map=probs[i],
                is_gt=False,
            )
            total_gt_instances += len(gt_instances)
            total_pred_instances += len(pred_instances)
            pred_scores = [float(ann["score"]) for ann in pred_instances]
            total_scores.extend(pred_scores)
            gt_anns.extend(gt_instances)
            pred_anns.extend(pred_instances)

            if logged_images < 5:
                if pred_scores:
                    score_stats = (
                        f"score_mean={np.mean(pred_scores):.4f}, "
                        f"score_min={np.min(pred_scores):.4f}, "
                        f"score_max={np.max(pred_scores):.4f}"
                    )
                else:
                    score_stats = "score_mean=nan, score_min=nan, score_max=nan"
                print(
                    f"[COCO] {fname}: gt={len(gt_instances)} pred={len(pred_instances)} {score_stats}"
                )
                logged_images += 1

    metrics = metrics_from_confusion(cm, class_names=class_names)
    metrics.update(_compute_coco_map(images, gt_anns, pred_anns, category_id=mine_class))
    if visualize and best_batch_data is not None:
        images_np, masks_np, preds_np = best_batch_data
        metrics["best_batch_index"] = best_batch_index
        metrics["best_batch_images"] = images_np
        metrics["best_batch_masks"] = masks_np
        metrics["best_batch_preds"] = preds_np
    metrics["model"] = model_name
    if total_scores:
        score_summary = (
            f"score_mean={np.mean(total_scores):.4f}, "
            f"score_min={np.min(total_scores):.4f}, "
            f"score_max={np.max(total_scores):.4f}"
        )
    else:
        score_summary = "score_mean=nan, score_min=nan, score_max=nan"
    print(
        f"[COCO] Summary: gt_instances={total_gt_instances}, "
        f"pred_instances={total_pred_instances}, {score_summary}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def resolve_terratorch_ckpts(spec: dict, discover: bool = True) -> list[str]:
    """Resolve the terratorch checkpoints of one category (absolute paths)."""
    root = RUNS_DIR / "terratorch"
    resolved: list[str] = []

    for rel in spec.get("terratorch_ckpts", []):
        path = root / rel
        if path.exists():
            resolved.append(str(path))
        else:
            print(f"  ! missing checkpoint (skipped): {path}")

    if discover:
        known = set(resolved)
        for run in spec.get("terratorch_runs", []):
            run_dir = root / run
            if not run_dir.exists():
                continue
            found = sorted(glob.glob(str(run_dir / "**" / "best-*.ckpt"), recursive=True))
            for f in found:
                if f not in known:
                    resolved.append(f)
                    known.add(f)
    return resolved


def resolve_geoai_dirs(spec: dict, only_runs: list[str] | None = None) -> list[str]:
    """Resolve the geoai/smp model directories of one category (absolute paths)."""
    root = RUNS_DIR / "geoai_smp"
    dirs: list[str] = []
    for run in spec.get("geoai_runs", []):
        if only_runs and not any(f in run for f in only_runs):
            continue
        run_dir = root / run
        if not run_dir.exists():
            print(f"  ! missing run directory (skipped): {run_dir}")
            continue
        for dirpath, _dirnames, filenames in os.walk(run_dir):
            if any(f.endswith((".pth", ".pt")) for f in filenames):
                dirs.append(dirpath)
    return sorted(set(dirs))


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

def run_benchmarks(config) -> list[dict]:
    """Benchmark every model of every requested category."""
    device = config.device
    num_classes = NUM_CLASSES
    class_names = CLASS_NAMES
    mine_class = MINE_CLASS

    all_results = []

    for spec in CATEGORY_SPECS:
        category = spec["category"]
        if config.categories and category not in config.categories:
            continue

        test_dir = BASE_DIR / category / "test"
        images_dir = test_dir / "images"
        masks_dir = test_dir / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            print(f"\n!! test split not found for {category} ({test_dir}) -- skipping")
            continue

        num_channels = spec["num_channels"]
        print(f"\n{'=' * 80}\nCATEGORY {category}  ({num_channels} channels)\n{'=' * 80}")

        # ---- terratorch checkpoints
        terratorch_ckpts = resolve_terratorch_ckpts(spec, discover=config.discover)
        # ---- geoai model dirs
        geoai_dirs = resolve_geoai_dirs(spec, only_runs=config.geoai_runs)

        if not terratorch_ckpts and not geoai_dirs:
            print("  no models found for this category")
            continue

        # Datasets use the normalisation each framework was trained with.
        setup = TERRATORCH_SETUP.get(num_channels, TERRATORCH_SETUP[3])
        ds_terratorch = TifSegmentationDataset(
            str(images_dir), str(masks_dir),
            means=setup["means"], stds=setup["stds"],
            constant_scale=TERRATORCH_SCALE,
            max_items=config.max_items,
        )
        ds_geoai = TifSegmentationDataset(
            str(images_dir), str(masks_dir),
            means=None, stds=None,
            constant_scale=GEOAI_SCALE,
            max_items=config.max_items,
        )

        dl_terratorch = DataLoader(
            ds_terratorch, batch_size=config.batch_size,
            num_workers=config.num_workers, shuffle=False, pin_memory=True,
        )
        dl_geoai = DataLoader(
            ds_geoai, batch_size=config.batch_size,
            num_workers=config.num_workers, shuffle=False, pin_memory=True,
        )

        # ---- Benchmark terratorch models
        for ckpt_path in terratorch_ckpts:
            rel = Path(ckpt_path).relative_to(RUNS_DIR / "terratorch") \
                if str(ckpt_path).startswith(str(RUNS_DIR / "terratorch")) else Path(ckpt_path)
            model_name = f"terratorch::{category}::{rel.parent.name}"
            print(f"\nLoading terratorch model: {ckpt_path}")
            try:
                task = load_terratorch_model(ckpt_path, device)
                if config.compute_coco:
                    metrics = benchmark_model_with_coco(
                        model_name, predict_terratorch, task,
                        dl_terratorch, num_classes, device, class_names,
                        mine_class=mine_class,
                        cca_connectivity=config.cca_connectivity,
                        cca_min_area_m2=config.cca_min_area_m2,
                        pixel_area_m2=config.pixel_area_m2,
                        prob_threshold=config.prob_threshold,
                    )
                else:
                    metrics = benchmark_model(
                        model_name, predict_terratorch, task,
                        dl_terratorch, num_classes, device, class_names,
                    )
                metrics["checkpoint"] = ckpt_path
                metrics["category"] = category
                metrics["framework"] = "terratorch"
                all_results.append(metrics)
                del task
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ✗ Failed: {e}")

        # ---- Benchmark geoai / smp models
        for model_dir in geoai_dirs:
            run_dir = os.path.basename(os.path.dirname(model_dir))
            model_name = f"geoai::{category}::{run_dir}/{os.path.basename(model_dir)}"
            print(f"\nLoading geoai model: {model_dir}")
            try:
                model = load_geoai_smp_model(model_dir, num_channels, device)
                if config.compute_coco:
                    metrics = benchmark_model_with_coco(
                        model_name, predict_smp, model,
                        dl_geoai, num_classes, device, class_names,
                        mine_class=mine_class,
                        cca_connectivity=config.cca_connectivity,
                        cca_min_area_m2=config.cca_min_area_m2,
                        pixel_area_m2=config.pixel_area_m2,
                        prob_threshold=config.prob_threshold,
                    )
                else:
                    metrics = benchmark_model(
                        model_name, predict_smp, model,
                        dl_geoai, num_classes, device, class_names,
                    )
                metrics["checkpoint"] = model_dir
                metrics["category"] = category
                metrics["framework"] = "geoai"
                all_results.append(metrics)
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  ✗ Failed: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Benchmark geoai/smp and terratorch models")
    parser.add_argument("--categories", nargs="+", default=None, help="Limit to these tile categories")
    parser.add_argument("--geoai-runs", nargs="+", default=None,
                        help="Only evaluate geoai models from these run directories (substring match)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-items", type=int, default=None, help="Limit the number of test tiles")
    parser.add_argument("--no-coco", dest="compute_coco", action="store_false")
    parser.add_argument("--no-discover", dest="discover", action="store_false",
                        help="Only use the checkpoints listed in CATEGORY_SPECS")
    parser.add_argument("--cca-connectivity", type=int, default=8)
    parser.add_argument("--cca-min-area-m2", type=float, default=300.0)
    parser.add_argument("--pixel-area-m2", type=float, default=100.0)
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    output_csv = args.output_csv or str(RUNS_DIR / "rgb+ve_benchmark_results.csv")
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    results = run_benchmarks(args)

    if not results:
        print("\nNo models were successfully evaluated.")
        return

    df = pd.DataFrame(results)

    col_order = [
        "category", "framework", "model", "mIoU", "mF1", "Overall_Accuracy",
        "IoU_Not Mine", "IoU_Mine", "F1_Not Mine", "F1_Mine",
        "Precision_Not Mine", "Precision_Mine", "Recall_Not Mine", "Recall_Mine",
        "mPrecision", "mRecall", "Mask_mAP_50_95", "Mask_mAP_50",
        "Bbox_mAP_50_95", "Bbox_mAP_50", "checkpoint",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))
    print("=" * 100)

    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()


# ---------------------------------------------------------------------------
# Tiling wrapper dataset (full image -> tiles)
# ---------------------------------------------------------------------------

class TiledFromFullImageDataset(Dataset):
    """
    Wrap a base dataset that returns:
      (image [C,H,W] float32, mask [H,W] int64, fname)

    Exposes tiles of size tile_size x tile_size.
    Each returned item:
      (x_tile [C,t,t], y_tile [t,t], meta_dict)

    meta_dict contains:
      fname, row_off, col_off, tile_size, full_h, full_w
    """
    def __init__(self, base_ds: Dataset, tile_size: int):
        self.base_ds = base_ds
        self.tile_size = int(tile_size)

        # Infer full size from first sample (assumes all same size)
        x0, y0, f0 = self.base_ds[0]
        self.full_h = int(y0.shape[-2])
        self.full_w = int(y0.shape[-1])

        if self.full_h % self.tile_size != 0 or self.full_w % self.tile_size != 0:
            raise ValueError(
                f"Full size ({self.full_h},{self.full_w}) must be divisible by tile_size={self.tile_size}. "
                "If you need padding/overlap, we can add it."
            )

        self.tiles_y = self.full_h // self.tile_size
        self.tiles_x = self.full_w // self.tile_size
        self.tiles_per_image = self.tiles_y * self.tiles_x

    def __len__(self):
        return len(self.base_ds) * self.tiles_per_image

    def __getitem__(self, idx: int):
        base_index = idx // self.tiles_per_image
        tile_index = idx % self.tiles_per_image

        ty = tile_index // self.tiles_x
        tx = tile_index % self.tiles_x
        row_off = ty * self.tile_size
        col_off = tx * self.tile_size

        x_full, y_full, fname = self.base_ds[base_index]
        x_tile = x_full[:, row_off:row_off + self.tile_size, col_off:col_off + self.tile_size]
        y_tile = y_full[row_off:row_off + self.tile_size, col_off:col_off + self.tile_size]

        meta = {
            "fname": fname,
            "row_off": row_off,
            "col_off": col_off,
            "tile_size": self.tile_size,
            "full_h": self.full_h,
            "full_w": self.full_w,
            "tiles_per_image": self.tiles_per_image,
        }
        return x_tile, y_tile, meta


def _tile_meta_collate(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any]]]):
    """
    Custom collate so meta stays a list[dict] (default collate would try to stack dict values).
    """
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(metas)


# ---------------------------------------------------------------------------
# Per-2048-image benchmarking via tiling + stitching
# ---------------------------------------------------------------------------

def benchmark_model_per_image_from_tiles(
    model_name: str,
    predict_fn,
    model,
    tiled_dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    class_names: list[str] | None = None,
    eval_block_size: int = 64,
    visualize: bool = False,
) -> tuple[list[dict], dict]:
    """
    Evaluates tiles, stitches predictions per original fname, downsamples to one
    binary label per eval_block_size x eval_block_size block, and returns:
      - per_image_metrics: list of metric dicts, one per original 2048 fname
      - overall_metrics: metric dict over all images combined

    Assumes predict_fn returns (B,H,W) on CPU (as in predict_smp/predict_terratorch).
    """
    def _to_binary_blocks(arr: np.ndarray, block_size: int) -> np.ndarray:
        """
        Convert per-pixel 0/1 map to one label per block using majority logic:
        block is 1 when most pixels in the block are 1, else 0.
        """
        h, w = arr.shape
        if h % block_size != 0 or w % block_size != 0:
            raise ValueError(
                f"Image size ({h},{w}) must be divisible by eval_block_size={block_size}."
            )

        blocks = arr.reshape(h // block_size, block_size, w // block_size, block_size)
        ones_per_block = blocks.sum(axis=(1, 3))
        pixels_per_block = block_size * block_size
        return (ones_per_block > (pixels_per_block / 2)).astype(np.int64)

    if visualize:
        import matplotlib.pyplot as plt

    per_image_cm: dict[str, np.ndarray] = {}
    overall_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    # running buffers for stitching each image
    pred_buf: dict[str, np.ndarray] = {}
    gt_buf: dict[str, np.ndarray] = {}
    seen_tiles: dict[str, int] = defaultdict(int)
    expected_tiles: dict[str, int] = {}

    for images, masks, metas in tqdm(tiled_dataloader, desc=f"  {model_name} (tiled)", leave=False):
        preds = predict_fn(model, images, device)  # (B, t, t) on CPU torch
        preds_np = preds.numpy()
        masks_np = masks.numpy()

        for i, meta in enumerate(metas):
            fname = meta["fname"]
            row_off = int(meta["row_off"])
            col_off = int(meta["col_off"])
            tile_size = int(meta["tile_size"])
            full_h = int(meta["full_h"])
            full_w = int(meta["full_w"])
            tiles_per_image = int(meta["tiles_per_image"])

            if fname not in pred_buf:
                pred_buf[fname] = np.zeros((full_h, full_w), dtype=np.int64)
                gt_buf[fname] = np.zeros((full_h, full_w), dtype=np.int64)
                expected_tiles[fname] = tiles_per_image

            pred_buf[fname][row_off:row_off + tile_size, col_off:col_off + tile_size] = preds_np[i]
            gt_buf[fname][row_off:row_off + tile_size, col_off:col_off + tile_size] = masks_np[i]
            seen_tiles[fname] += 1

            # once complete, compute metrics for that 2048 image and drop buffers
            if seen_tiles[fname] == expected_tiles[fname]:
                pred_full = pred_buf[fname]
                gt_full = gt_buf[fname]
                pred_64 = _to_binary_blocks(pred_full, eval_block_size)
                gt_64 = _to_binary_blocks(gt_full, eval_block_size)
                cm = compute_confusion(pred_64, gt_64, num_classes=num_classes, ignore_index=-1)
                per_image_cm[fname] = cm
                overall_cm += cm

                if visualize:
                    vmax = max(1, num_classes - 1)
                    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
                    axs[0, 0].imshow(pred_full, cmap="gray", vmin=0, vmax=vmax)
                    axs[0, 0].set_title(f"{fname} - pred_buf")
                    axs[0, 1].imshow(gt_full, cmap="gray", vmin=0, vmax=vmax)
                    axs[0, 1].set_title(f"{fname} - gt_buf")
                    axs[1, 0].imshow(pred_64, cmap="gray", vmin=0, vmax=vmax)
                    axs[1, 0].set_title(f"{fname} - pred_{eval_block_size}")
                    axs[1, 1].imshow(gt_64, cmap="gray", vmin=0, vmax=vmax)
                    axs[1, 1].set_title(f"{fname} - gt_{eval_block_size}")
                    for ax in axs.flat:
                        ax.axis("off")
                    plt.tight_layout()
                    plt.show()

                del pred_buf[fname]
                del gt_buf[fname]
                del seen_tiles[fname]
                del expected_tiles[fname]

    per_image_metrics: list[dict] = []
    for fname, cm in sorted(per_image_cm.items()):
        m = metrics_from_confusion(cm, class_names=class_names)
        m["model"] = model_name
        m["fname"] = fname
        m["eval_block_size_px"] = eval_block_size
        per_image_metrics.append(m)

    overall_metrics = metrics_from_confusion(overall_cm, class_names=class_names)
    overall_metrics["model"] = model_name
    overall_metrics["eval_block_size_px"] = eval_block_size
    return per_image_metrics, overall_metrics
