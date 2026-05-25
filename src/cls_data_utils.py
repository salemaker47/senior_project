"""
src/cls_data_utils.py

Classification-specific data infrastructure:

    CLASS_TO_IDX / IDX_TO_CLASS  — canonical label maps for 3 figshare classes
    extract_patch                — crop tumor ROI from a grayscale MRI using a mask
    BrainTumorClsDataset         — PyTorch Dataset yielding (patch, label)
    build_train_transform_cls    — Albumentations pipeline with random aug
    build_eval_transform_cls     — Albumentations pipeline, no random aug
    build_dataloaders_cls        — train + val DataLoaders (always GT masks)
    build_test_loader_cls        — test DataLoader (GT or predicted mask source)

Mask sources:
    "gt"        read mask from metadata CSV's mask_path column
    "predicted" read mask from <predictions_dir>/<image_id>.png

Training always uses mask_source="gt". Only test-time Eval B uses "predicted".
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.file_utils import PathLike
from src.sg_data_utils import IMAGENET_MEAN, IMAGENET_STD  # single source of truth

# Canonical label mapping — order matches the FigShare reference notebook.
CLASS_TO_IDX: Dict[str, int] = {"meningioma": 0, "glioma": 1, "pituitary": 2}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}
# Ordered class name list derived from IDX_TO_CLASS; do not redefine elsewhere.
CLASS_NAMES: list = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS)]


# --------------------------------------------------------------------------- #
# Patch extraction
# --------------------------------------------------------------------------- #
def extract_patch(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: int = 224,
    padding_frac: float = 0.10,
) -> np.ndarray:
    """
    Crop the tumor ROI from a grayscale MRI image using its binary mask.

    Steps:
        1. Find the bounding rect of nonzero pixels in `mask`.
        2. Expand by `padding_frac * max(bbox_h, bbox_w)` on each side.
        3. Make the padded bbox square (expand the shorter axis to match the longer).
        4. Clamp to image bounds — note: clamping can make the crop non-square
           for tumors near an image edge; cv2.resize corrects this with slight
           aspect-ratio distortion.
        5. Crop → convert GRAY→RGB → resize to (target_size, target_size).

    Fallback: if `mask` has no nonzero pixels, use the whole image.

    Parameters
    ----------
    image       : H×W uint8 grayscale array
    mask        : H×W uint8 array, values 0 or 255
    target_size : output side length in pixels (default 224 for ImageNet models)
    padding_frac: fraction of the larger bbox dimension added as padding on each side

    Returns
    -------
    np.ndarray of shape (target_size, target_size, 3), dtype uint8, RGB
    """
    h, w = image.shape[:2]

    coords = cv2.findNonZero(mask)
    if coords is None:
        # Whole-image fallback: no tumor mask pixels found.
        x, y, bw, bh = 0, 0, w, h
    else:
        x, y, bw, bh = cv2.boundingRect(coords)

    # Padding
    pad = int(padding_frac * max(bw, bh))
    x1 = x - pad
    y1 = y - pad
    x2 = x + bw + pad
    y2 = y + bh + pad

    # Make square
    side = max(x2 - x1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = image  # last-resort fallback

    rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    return cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------- #
# Albumentations transforms
# --------------------------------------------------------------------------- #
def build_train_transform_cls() -> A.Compose:
    """
    Training augmentation for classification patches.
    Input: (target_size, target_size, 3) uint8 RGB (already sized by extract_patch;
    no A.Resize needed here).
    Mirrors the augmentation used in the FigShare reference notebook.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.5,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform_cls() -> A.Compose:
    """Validation/test transform: normalize only, no random aug.
    Patches are already sized by extract_patch; no A.Resize needed."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BrainTumorClsDataset(Dataset):
    """
    Reads tumor patch + class label from a fold CSV.

    Each item:
        patch  -> torch.float32 (3, target_size, target_size), ImageNet-normalized
        label  -> torch.long scalar, index into CLASS_TO_IDX

    If `return_meta=True`, __getitem__ also returns a dict with
    `image_id`, `patient_id`, `tumor_class`, `mask_source`.

    mask_source:
        "gt"        — read mask from metadata's `mask_path` column
        "predicted" — read mask from `<predictions_dir>/<image_id>.png`
                      `predictions_dir` must be provided when mask_source="predicted"
    """

    def __init__(
        self,
        df: pd.DataFrame,
        project_root: PathLike,
        transform: Optional[Callable] = None,
        mask_source: str = "gt",
        predictions_dir: Optional[PathLike] = None,
        target_size: int = 224,
        padding_frac: float = 0.10,
        return_meta: bool = False,
    ):
        if mask_source not in ("gt", "predicted"):
            raise ValueError(f"mask_source must be 'gt' or 'predicted', got {mask_source!r}")
        if mask_source == "predicted" and predictions_dir is None:
            raise ValueError("predictions_dir is required when mask_source='predicted'")

        self.df = df.reset_index(drop=True)
        self.project_root = Path(project_root)
        self.transform = transform
        self.mask_source = mask_source
        self.predictions_dir = Path(predictions_dir) if predictions_dir is not None else None
        self.target_size = target_size
        self.padding_frac = padding_frac
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.df)

    def _read_grayscale(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image: {path}")
        return img

    def _read_mask(self, path: Path) -> np.ndarray:
        """Returns {0, 255} uint8 — cv2.findNonZero only needs nonzero values;
        no loss function consumes this mask directly in the cls pipeline."""
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"could not read mask: {path}")
        return (m > 127).astype(np.uint8) * 255

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row.get("image_id", idx))

        image = self._read_grayscale(self.project_root / row["image_path"])

        if self.mask_source == "gt":
            mask = self._read_mask(self.project_root / row["mask_path"])
        else:
            pred_path = self.predictions_dir / f"{image_id}.png"
            mask = self._read_mask(pred_path)

        patch = extract_patch(
            image, mask,
            target_size=self.target_size,
            padding_frac=self.padding_frac,
        )

        if self.transform is not None:
            out = self.transform(image=patch)
            patch_tensor = out["image"]
        else:
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0

        tumor_class = str(row.get("tumor_class", ""))
        if tumor_class not in CLASS_TO_IDX:
            import warnings
            warnings.warn(
                f"Unknown tumor_class {tumor_class!r} for image_id={image_id!r}. "
                f"Expected one of {list(CLASS_TO_IDX)}. Defaulting to 0.",
                RuntimeWarning, stacklevel=2,
            )
        label = CLASS_TO_IDX.get(tumor_class, 0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_meta:
            meta = {
                "image_id":    image_id,
                "patient_id":  str(row.get("patient_id", "")),
                "tumor_class": tumor_class,
                "mask_source": self.mask_source,
            }
            return patch_tensor, label_tensor, meta
        return patch_tensor, label_tensor


# --------------------------------------------------------------------------- #
# DataLoader builders
# --------------------------------------------------------------------------- #
def build_dataloaders_cls(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    project_root: PathLike,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    padding_frac: float = 0.10,
    pin_memory: bool = True,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train + val DataLoaders for one classification fold.
    Training always uses mask_source='gt' (GT masks for ROI extraction).
    """
    train_tf = build_train_transform_cls()
    eval_tf  = build_eval_transform_cls()

    train_ds = BrainTumorClsDataset(
        train_df, project_root,
        transform=train_tf,
        mask_source="gt",
        target_size=image_size,
        padding_frac=padding_frac,
    )
    val_ds = BrainTumorClsDataset(
        val_df, project_root,
        transform=eval_tf,
        mask_source="gt",
        target_size=image_size,
        padding_frac=padding_frac,
    )

    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def build_test_loader_cls(
    test_df: pd.DataFrame,
    project_root: PathLike,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 224,
    padding_frac: float = 0.10,
    mask_source: str = "gt",
    predictions_dir: Optional[PathLike] = None,
    return_meta: bool = True,
) -> DataLoader:
    """
    Build the test DataLoader for classification.

    `mask_source="predicted"` + `predictions_dir` enables Eval B (predicted masks)
    without any changes to other code — the Dataset handles the lookup internally.

    Default `return_meta=True` so NB08 can key per-image results on `image_id`.
    """
    eval_tf = build_eval_transform_cls()
    ds = BrainTumorClsDataset(
        test_df, project_root,
        transform=eval_tf,
        mask_source=mask_source,
        predictions_dir=predictions_dir,
        target_size=image_size,
        padding_frac=padding_frac,
        return_meta=return_meta,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
