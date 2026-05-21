"""
src/vis_utils.py

Shared visualization helpers — image loaders + the per-class examples grid
used in 01_data_preparation.ipynb. Loaders are exposed publicly so seg- and
cls-specific vis modules can reuse them without duplication.

Seg-specific helpers (triplet / overlay-triplet / 4-panel image+GT+pred)
live in src/sg_vis_utils.py. Cls-specific helpers (patch grids, confusion
matrices, F1 bars, gap charts) live in src/cls_vis_utils.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.file_utils import PathLike


# --------------------------------------------------------------------------- #
# Public image loaders (used by this module + sg_vis_utils.py)
# --------------------------------------------------------------------------- #
def load_grayscale_png(path: PathLike) -> np.ndarray:
    """Load a grayscale PNG as uint8 (H, W)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def load_binary_mask_png(path: PathLike) -> np.ndarray:
    """
    Load a binary mask PNG as uint8 (H, W) with values {0, 1}.
    Threshold at 127 in case the mask was saved as {0, 255}.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read mask: {path}")
    return (img > 127).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Class-by-class examples grid (used by 01_data_preparation.ipynb)
# --------------------------------------------------------------------------- #
def show_class_examples(
    metadata_df: pd.DataFrame,
    project_root: PathLike,
    classes: Optional[Sequence[str]] = None,
    n_per_class: int = 3,
    random_state: int = 42,
    overlay_alpha: float = 0.4,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> None:
    """
    Sample `n_per_class` examples for each tumor class and display them in a
    grid (rows = class, cols = example), with the tumor mask overlaid in red.

    Parameters
    ----------
    metadata_df:
        DataFrame produced by NB01. Must contain `tumor_class`, `image_id`,
        `image_path`, and `mask_path` columns. Path columns are interpreted
        as relative to `project_root` (matching the convention in
        preprocess_utils.convert_figshare_mat_to_png_record).
    project_root:
        Root used to resolve `image_path` and `mask_path` from metadata_df.
        Pass the project root in use right now (Drive on NB01, local SSD
        during training).
    classes:
        Tumor class labels to visualize. If None, takes every unique value
        from metadata_df['tumor_class'] sorted alphabetically.
    n_per_class:
        Number of example images per class (columns of the grid).
    random_state:
        RNG seed for reproducible sampling.
    overlay_alpha:
        Blend factor for the red mask overlay; 0 = invisible, 1 = solid.
    save_path:
        If given, also save the figure to this path.
    show:
        If True, call plt.show() after building the figure.
    """
    if classes is None:
        classes = sorted(metadata_df["tumor_class"].unique().tolist())

    root = Path(project_root)
    rng = np.random.RandomState(random_state)

    fig, axes = plt.subplots(
        len(classes),
        n_per_class,
        figsize=(3 * n_per_class, 3 * len(classes)),
    )
    # Normalize axes to a 2D array for uniform indexing.
    if len(classes) == 1:
        axes = np.array([axes])
    if n_per_class == 1:
        axes = axes.reshape(-1, 1)

    for r, cls in enumerate(classes):
        cls_df = metadata_df[metadata_df["tumor_class"] == cls]
        if len(cls_df) == 0:
            for c in range(n_per_class):
                axes[r, c].axis("off")
            continue

        n = min(n_per_class, len(cls_df))
        idx = rng.choice(len(cls_df), size=n, replace=False)
        sample = cls_df.iloc[idx]

        for c in range(n_per_class):
            ax = axes[r, c]
            if c >= n:
                ax.axis("off")
                continue

            row = sample.iloc[c]
            image = load_grayscale_png(root / row["image_path"])
            mask = load_binary_mask_png(root / row["mask_path"])

            rgb = np.stack([image, image, image], axis=-1).astype(np.float32) / 255.0
            rgb[mask > 0] = (
                (1 - overlay_alpha) * rgb[mask > 0]
                + overlay_alpha * np.array([1.0, 0.0, 0.0])
            )

            ax.imshow(np.clip(rgb, 0.0, 1.0))
            ax.set_title(f"{cls}\n{row['image_id']}", fontsize=9)
            ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)