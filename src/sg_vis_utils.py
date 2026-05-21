"""
src/sg_vis_utils.py

Segmentation-specific visualization helpers. Three functions:

    show_triplet(image_path, mask_path, ...)
        2-panel: image | mask. Used for one-off sanity checks.

    show_overlay_triplet(image_path, mask_path, ...)
        3-panel: image | mask | overlay (red mask on image).

    show_image_gt_pred_overlay(image, gt_mask, pred_mask, ...)
        4-panel: image | GT | pred | TP/FN/FP overlay.
        Used by NB04 / NB05 to render best/worst/random predictions.
        Takes arrays directly (works on in-memory tensors at test time).

Shared loaders + show_class_examples live in src/vis_utils.py and are
imported here rather than duplicated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.vis_utils import load_grayscale_png, load_binary_mask_png
from src.file_utils import PathLike

# Color overlays — intentionally vivid, easy to read in the report.
_TP_COLOR = np.array([0.0, 1.0, 0.0])  # green: pred ∩ gt
_FN_COLOR = np.array([1.0, 0.0, 0.0])  # red:   gt only (missed)
_FP_COLOR = np.array([0.0, 0.4, 1.0])  # blue:  pred only (over-segmented)
_TUMOR_OVERLAY_COLOR = np.array([1.0, 0.0, 0.0])  # red, used by overlay-triplet


# --------------------------------------------------------------------------- #
# 2-panel
# --------------------------------------------------------------------------- #
def show_triplet(
    image_path: PathLike,
    mask_path: PathLike,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    """Show (image, mask) side by side."""
    image = load_grayscale_png(image_path)
    mask = load_binary_mask_png(mask_path)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("mask")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 3-panel: image | mask | red overlay
# --------------------------------------------------------------------------- #
def show_overlay_triplet(
    image_path: PathLike,
    mask_path: PathLike,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    show: bool = True,
    overlay_alpha: float = 0.4,
) -> None:
    """Show (image, mask, overlay). The overlay paints the tumor region red."""
    image = load_grayscale_png(image_path)
    mask = load_binary_mask_png(mask_path)

    rgb = np.stack([image, image, image], axis=-1).astype(np.float32) / 255.0
    overlay = rgb.copy()
    overlay[mask > 0] = (
        (1 - overlay_alpha) * overlay[mask > 0]
        + overlay_alpha * _TUMOR_OVERLAY_COLOR
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("mask")
    axes[2].imshow(np.clip(overlay, 0.0, 1.0))
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4-panel: image | GT | pred | TP/FN/FP overlay (NB04, NB05)
# --------------------------------------------------------------------------- #
def show_image_gt_pred_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: Optional[str] = None,
    dice: Optional[float] = None,
    iou: Optional[float] = None,
    save_path: Optional[PathLike] = None,
    show: bool = True,
    overlay_alpha: float = 0.45,
) -> None:
    """
    Four-panel figure: image | GT mask | predicted mask | TP/FN/FP overlay.

    Overlay convention:
        green = TP (pred ∩ gt)
        red   = FN (gt only — missed by the model)
        blue  = FP (pred only — over-segmented)

    `image`     can be (H, W) grayscale uint8 or (H, W, 3) RGB uint8.
                If float [0,1], it's scaled by 255 internally.
    `gt_mask`,
    `pred_mask` are binary (H, W). Values may be {0,1} or {0,255}; both work.
    """
    img = np.asarray(image)

    # Coerce to (H, W) uint8 grayscale for display.
    if img.dtype != np.uint8:
        if img.max() <= 1.0 + 1e-6:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    if img.ndim == 3:
        # Average across channels — works for grayscale-replicated 3-ch and
        # for true RGB without needing cv2.
        img = img.mean(axis=-1).astype(np.uint8)

    gt = (np.asarray(gt_mask)   > 0).astype(bool)
    pr = (np.asarray(pred_mask) > 0).astype(bool)

    # Build a colored overlay on a grayscale background.
    rgb = np.stack([img, img, img], axis=-1).astype(np.float32) / 255.0
    overlay = rgb.copy()

    tp = gt & pr
    fn = gt & ~pr
    fp = ~gt & pr

    overlay[tp] = (1 - overlay_alpha) * overlay[tp] + overlay_alpha * _TP_COLOR
    overlay[fn] = (1 - overlay_alpha) * overlay[fn] + overlay_alpha * _FN_COLOR
    overlay[fp] = (1 - overlay_alpha) * overlay[fp] + overlay_alpha * _FP_COLOR

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("image")
    axes[1].imshow(gt.astype(np.uint8), cmap="gray")
    axes[1].set_title("ground truth")
    axes[2].imshow(pr.astype(np.uint8), cmap="gray")
    axes[2].set_title("prediction")
    axes[3].imshow(np.clip(overlay, 0.0, 1.0))
    axes[3].set_title("overlay (green=TP, red=FN, blue=FP)")
    for ax in axes:
        ax.axis("off")

    suptitle_parts = []
    if title is not None:
        suptitle_parts.append(title)
    if dice is not None:
        suptitle_parts.append(f"Dice={dice:.4f}")
    if iou is not None:
        suptitle_parts.append(f"IoU={iou:.4f}")
    if suptitle_parts:
        fig.suptitle("  |  ".join(suptitle_parts), fontsize=12)

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)