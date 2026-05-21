"""
src/sg_metrics.py

Binary segmentation metrics — both training-time (logged each epoch) and
test-time (rich per-image metrics for NB05's cv_per_image.csv).

REGISTRY PATTERN:
    Add new training-time metric pairs by adding a branch to `get_metric_kind_pairs`.
    Add new per-image metrics by extending `compute_per_image_metrics_batch`.
    Do NOT modify existing branches.

Convention everywhere:
    pred / target shapes are (N, 1, H, W) or (N, H, W); both work.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import torch

import segmentation_models_pytorch as smp


# --------------------------------------------------------------------------- #
# Logits → binary mask
# --------------------------------------------------------------------------- #
def binarize_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """sigmoid + threshold → float {0.0, 1.0} tensor with same shape as logits."""
    return (torch.sigmoid(logits) > threshold).float()


def _flatten_batch(t: torch.Tensor) -> torch.Tensor:
    """(N, ...) -> (N, prod(...))"""
    return t.reshape(t.shape[0], -1)


# --------------------------------------------------------------------------- #
# Training-time scalars (per-image averaged, used by Lightning)
# --------------------------------------------------------------------------- #
def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    reduce: str = "mean",
) -> torch.Tensor:
    """Per-image Dice. Pass a binarized prediction (binarize_logits first)."""
    p = _flatten_batch(pred.float())
    t = _flatten_batch(target.float())
    inter = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return dice if reduce == "none" else dice.mean()


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    reduce: str = "mean",
) -> torch.Tensor:
    """Per-image IoU."""
    p = _flatten_batch(pred.float())
    t = _flatten_batch(target.float())
    inter = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1) - inter
    iou = (inter + smooth) / (union + smooth)
    return iou if reduce == "none" else iou.mean()


# --------------------------------------------------------------------------- #
# SMP-style stats accumulation (used by micro reductions during training)
# --------------------------------------------------------------------------- #
def get_smp_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-image (tp, fp, fn, tn) for a batch."""
    preds = (torch.sigmoid(logits) > threshold).long()
    return smp.metrics.get_stats(preds, target.long(), mode="binary")


def micro_dice_from_stats(tp, fp, fn, tn) -> torch.Tensor:
    """Globally pooled Dice (= F1)."""
    return smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")


def micro_iou_from_stats(tp, fp, fn, tn) -> torch.Tensor:
    """Globally pooled IoU."""
    return smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")


def get_metric_kind_pairs(kind: str) -> Dict[str, Callable]:
    """
    Returns a dict {logged_metric_name: reduction_fn} for a metric "kind".
    The Lightning module logs each entry every validation epoch.

    Supported kinds:
        "micro"        -> {"val_dice": micro_dice, "val_iou": micro_iou}
        "micro_macro"  -> same as "micro" (macro per-image averaging is done at
                          test time in sg_eval_utils; training always uses the
                          globally pooled micro reduction)
    """
    k = kind.lower()
    if k in ("micro", "micro_macro"):
        return {
            "val_dice": micro_dice_from_stats,
            "val_iou":  micro_iou_from_stats,
        }
    raise ValueError(f"unknown metric kind: {kind!r} (supported: 'micro', 'micro_macro')")


# --------------------------------------------------------------------------- #
# Test-time per-image metrics (Enhancement A)
# Used by sg_test_utils.evaluate_fold to populate cv_per_image.csv
# --------------------------------------------------------------------------- #
PER_IMAGE_METRIC_NAMES: Tuple[str, ...] = (
    "dice",
    "iou",
    "sensitivity",        # = recall = TP / (TP + FN)
    "precision",          # = TP / (TP + FP)
    "pred_mask_area_ratio",
    "gt_mask_area_ratio",
    "area_ratio_delta",   # signed: pred_area_ratio - gt_area_ratio
    "true_positive_pixels",
    "false_positive_pixels",
    "false_negative_pixels",
    "total_pixels",
)


def compute_per_image_metrics_batch(
    pred_binary: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Compute the full per-image metric suite for a batch.

    Inputs:
        pred_binary -- (N, 1, H, W) or (N, H, W), values in {0, 1}
        target      -- same shape as pred_binary
    Output:
        dict with keys = PER_IMAGE_METRIC_NAMES, each value a 1-D numpy array
        of length N. Use this in sg_test_utils to build the per-image CSV.

    Notes:
        - `smooth` is applied to dice/iou/sensitivity/precision to avoid
          division by zero on empty masks. With smooth=1 on 256x256 images
          (~65k pixels), the bias is negligible (~1e-5).
        - All counts (TP/FP/FN/total_pixels) are int64 numpy arrays.
        - Area ratios are float64 numpy arrays in [0, 1].
    """
    p = _flatten_batch(pred_binary.float())
    t = _flatten_batch(target.float())

    tp = (p * t).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    total = torch.full_like(tp, p.shape[1])

    dice        = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou         = (tp + smooth) / (tp + fp + fn + smooth)
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    precision   = (tp + smooth) / (tp + fp + smooth)

    pred_area = (tp + fp) / total
    gt_area   = (tp + fn) / total
    area_delta = pred_area - gt_area

    return {
        "dice":                  dice.detach().cpu().numpy().astype(np.float64),
        "iou":                   iou.detach().cpu().numpy().astype(np.float64),
        "sensitivity":           sensitivity.detach().cpu().numpy().astype(np.float64),
        "precision":             precision.detach().cpu().numpy().astype(np.float64),
        "pred_mask_area_ratio":  pred_area.detach().cpu().numpy().astype(np.float64),
        "gt_mask_area_ratio":    gt_area.detach().cpu().numpy().astype(np.float64),
        "area_ratio_delta":      area_delta.detach().cpu().numpy().astype(np.float64),
        "true_positive_pixels":  tp.detach().cpu().numpy().astype(np.int64),
        "false_positive_pixels": fp.detach().cpu().numpy().astype(np.int64),
        "false_negative_pixels": fn.detach().cpu().numpy().astype(np.int64),
        "total_pixels":          total.detach().cpu().numpy().astype(np.int64),
    }


def compute_per_image_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Convenience: binarize logits then compute the full metric suite."""
    pred_binary = binarize_logits(logits, threshold=threshold)
    return compute_per_image_metrics_batch(pred_binary, target, smooth=smooth)
