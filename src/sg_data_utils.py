"""
src/sg_data_utils.py

Segmentation-specific data infrastructure:

    BrainTumorDataset       — PyTorch Dataset reading (image, mask) pairs
    build_train_transform   — Albumentations pipeline with random aug
    build_eval_transform    — Albumentations pipeline, no random aug
    build_dataloaders       — train + val DataLoaders
    build_test_loader       — test DataLoader (returns metadata too)

Shared metadata / fold helpers live in src/data_utils.py.

Augmentation strengths (passed via EXPERIMENT["augmentation_strength"]):

    "none"       preprocessing + normalize, no random aug
    "light"      flip + small affine + brightness only
    "default"    our standard set (Affine, Elastic, Brightness, Blur, Noise)
    "reference"  matches the FigShare reference notebook exactly. Use only
                 for reference-comparison runs (the 7 reproduction experiments
                 in M6 use this).

Preprocessing options (passed via EXPERIMENT["preprocessing"]):

    "original" / "none" / ""  do nothing
    "clahe"                   CLAHE contrast normalization
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

PathLike = Union[str, Path]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Albumentations: preprocessing block (registry pattern, project §5)
# --------------------------------------------------------------------------- #
def _preprocessing_block(name: Optional[str]) -> list:
    """Returns the preprocessing portion of the augmentation pipeline."""
    name = (name or "original").lower()
    if name in ("original", "none", ""):
        return []
    if name == "clahe":
        return [A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)]
    raise ValueError(f"unknown preprocessing name: {name!r}")


# --------------------------------------------------------------------------- #
# Albumentations: transforms
# --------------------------------------------------------------------------- #
def build_train_transform(
    image_size: int = 256,
    preprocessing: str = "original",
    augmentation_strength: str = "default",
) -> A.Compose:
    """Training augmentation pipeline. See module docstring for strength options."""
    pre = _preprocessing_block(preprocessing)

    if augmentation_strength == "none":
        aug: list = []

    elif augmentation_strength == "light":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(p=0.3),
        ]

    elif augmentation_strength == "default":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.2),
        ]

    elif augmentation_strength == "reference":
        # Matches the FigShare reference notebook exactly — used by the 7
        # reproduction experiments in M6.
        aug = [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.95, 1.05),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.ElasticTransform(
                alpha=60, sigma=3.0,
                border_mode=cv2.BORDER_CONSTANT, p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            # std_range in albumentations >=2.0 is relative to image max (255 for uint8).
            # (0.012, 0.028) ≈ std 3–7 px, matching reference's var_limit=(10,50).
            A.GaussNoise(std_range=(0.012, 0.028), p=0.2),
        ]

    else:
        raise ValueError(f"unknown augmentation_strength: {augmentation_strength!r}")

    return A.Compose([
        A.Resize(image_size, image_size),
        *pre,
        *aug,
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(
    image_size: int = 256,
    preprocessing: str = "original",
) -> A.Compose:
    """Validation/test pipeline: preprocessing + normalize. No random aug."""
    pre = _preprocessing_block(preprocessing)
    return A.Compose([
        A.Resize(image_size, image_size),
        *pre,
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class BrainTumorDataset(Dataset):
    """
    Reads (image, mask) pairs from a fold CSV.

    Each item:
        image  -> torch.float32 (3, H, W), ImageNet-normalized
        mask   -> torch.float32 (1, H, W), values in {0.0, 1.0}

    If `return_meta=True`, __getitem__ also returns a dict with
    `image_id`, `patient_id`, `tumor_class` (used by NB05 to write per-image
    predictions and per-image metrics keyed on image_id).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        project_root: PathLike,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.project_root = Path(project_root)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.df)

    def _read_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"could not read image: {path}")
        # 3-channel grayscale-replicated for ImageNet-pretrained encoders.
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def _read_mask(self, path: Path) -> np.ndarray:
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"could not read mask: {path}")
        return (m > 127).astype(np.uint8)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self.project_root / row["image_path"]
        mask_path  = self.project_root / row["mask_path"]

        image = self._read_image(image_path)
        mask  = self._read_mask(mask_path)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask)

        # Albumentations + ToTensorV2 returns mask as (H, W) without channel.
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        if mask.max() > 1.5:                      # in case mask came in as {0, 255}
            mask = (mask > 0.5).float()

        if self.return_meta:
            meta = {
                "image_id":    str(row.get("image_id", idx)),
                "patient_id":  str(row.get("patient_id", "")),
                "tumor_class": str(row.get("tumor_class", "")),
            }
            return image, mask, meta
        return image, mask


# --------------------------------------------------------------------------- #
# DataLoader builders
# --------------------------------------------------------------------------- #
def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    project_root: PathLike,
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 256,
    preprocessing: str = "original",
    augmentation_strength: str = "default",
    pin_memory: bool = True,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders for one fold."""
    train_tf = build_train_transform(
        image_size=image_size,
        preprocessing=preprocessing,
        augmentation_strength=augmentation_strength,
    )
    eval_tf = build_eval_transform(image_size=image_size, preprocessing=preprocessing)

    train_ds = BrainTumorDataset(train_df, project_root, transform=train_tf)
    val_ds   = BrainTumorDataset(val_df,   project_root, transform=eval_tf)

    g = torch.Generator()
    if seed is not None:
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


def build_test_loader(
    test_df: pd.DataFrame,
    project_root: PathLike,
    batch_size: int = 16,
    num_workers: int = 2,
    image_size: int = 256,
    preprocessing: str = "original",
    return_meta: bool = True,
) -> DataLoader:
    """Build the test DataLoader. Default `return_meta=True` so NB05 can key
    per-image predictions and per-image metrics on `image_id`."""
    eval_tf = build_eval_transform(image_size=image_size, preprocessing=preprocessing)
    ds = BrainTumorDataset(test_df, project_root, transform=eval_tf, return_meta=return_meta)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )