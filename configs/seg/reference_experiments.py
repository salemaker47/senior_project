"""
configs/seg/reference_experiments.py

The 7 canonical reference experiments for the FigShare reproduction (§13 of
the project instruction). Replaces the old `00_env_setups.ipynb` cookbook —
now importable + version-controlled instead of copy-paste.

Usage in NB03 / NB05 cell 3:

    from configs.seg.reference_experiments import REFERENCE_EXPERIMENTS, get_experiment
    EXPERIMENT = get_experiment("01_dice_image_level")   # safe copy, mutate freely
    EXPERIMENT["fold"] = 3                               # per-run override

Pattern: a single `BASE_EXPERIMENT` dict holds every default that is shared
across all 7 experiments. Each experiment is then `BASE_EXPERIMENT` updated
with only the fields that make it unique (loss / model / preprocessing / batch).
This means a single edit to BASE_EXPERIMENT (e.g. new max_epochs default)
propagates to all 7 without 7 separate edits.

To add an 8th reference experiment: append one entry to
`_EXPERIMENT_OVERRIDES` below. To run a non-reference one-off (e.g. trying a
new optimizer just on fold 1), build the EXPERIMENT dict inline in the
notebook instead — registries are for canonical, reproducible recipes.
"""

from copy import deepcopy
from typing import Any, Dict


# --------------------------------------------------------------------------- #
# Shared defaults (every reference experiment starts from this)
# --------------------------------------------------------------------------- #
BASE_EXPERIMENT: Dict[str, Any] = {
    # filled per-experiment: name, loss_name, loss_kwargs,
    # plus rare overrides: model_name, preprocessing, batch_size

    "task":         "segmentation",
    "dataset":      "figshare",
    "split_scheme": "image_level",

    "fold":        1,
    "image_size":  256,
    "batch_size":  8,
    "num_workers": 2,

    "preprocessing":         "original",
    "augmentation_strength": "reference",

    "model_name":      "smp_unet_resnet34",
    "encoder_weights": "imagenet",

    "optimizer_name":   "adam",
    "optimizer_kwargs": {"lr": 1e-4},

    "scheduler_name":    "reduce_on_plateau",
    "scheduler_kwargs":  {"mode": "min", "factor": 0.1, "patience": 5, "min_lr": 1e-7},
    "scheduler_monitor": "val_loss",

    "metric_kind": "micro_macro",

    "max_epochs": 100,
    "patience":   15,
    "threshold":  0.5,
    "seed":       42,
}


# --------------------------------------------------------------------------- #
# Per-experiment overrides (only the fields each experiment changes)
# --------------------------------------------------------------------------- #
_EXPERIMENT_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "01_dice_image_level": {
        "name":        "01_dice_image_level",
        "loss_name":   "dice",
        "loss_kwargs": {},
    },
    "02_bce_image_level": {
        "name":        "02_bce_image_level",
        "loss_name":   "bce",
        "loss_kwargs": {},
    },
    "03_dicebce_image_level": {
        "name":        "03_dicebce_image_level",
        "loss_name":   "dice_bce",
        "loss_kwargs": {"bce_weight": 1.0, "dice_weight": 1.0},
    },
    "04_dicefocal_image_level": {
        "name":        "04_dicefocal_image_level",
        "loss_name":   "dice_focal",
        "loss_kwargs": {
            "dice_weight":  1.0,
            "focal_weight": 1.0,
            "alpha":        0.25,   # Focal class-balancing factor
            "gamma":        2.0,    # Focal modulation
        },
    },
    "05_lovasz_image_level": {
        "name":        "05_lovasz_image_level",
        "loss_name":   "lovasz",
        "loss_kwargs": {},
    },
    "06_clahe_dicebce_image_level": {
        "name":          "06_clahe_dicebce_image_level",
        "preprocessing": "clahe",        # ← only experiment with CLAHE
        "loss_name":     "dice_bce",
        "loss_kwargs":   {"bce_weight": 1.0, "dice_weight": 1.0},
    },
    "07_unetpp_effb4_dicebce_image_level": {
        "name":        "07_unetpp_effb4_dicebce_image_level",
        "model_name":  "smp_unetpp_efficientnetb4",   # ← only non-default model
        "batch_size":  6,                              # ← reduced for VRAM
        "loss_name":   "dice_bce",
        "loss_kwargs": {"bce_weight": 1.0, "dice_weight": 1.0},
    },
}


# --------------------------------------------------------------------------- #
# Public registry — fully materialized for easy iteration
# --------------------------------------------------------------------------- #
def _build(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge BASE_EXPERIMENT with per-experiment overrides into a complete dict."""
    cfg = deepcopy(BASE_EXPERIMENT)
    cfg.update(overrides)
    return cfg


REFERENCE_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    name: _build(ov) for name, ov in _EXPERIMENT_OVERRIDES.items()
}


def get_experiment(name: str) -> Dict[str, Any]:
    """
    Return a *fresh deep copy* of the named reference experiment so mutating
    `EXPERIMENT["fold"] = 3` in the notebook doesn't pollute the registry.
    """
    if name not in REFERENCE_EXPERIMENTS:
        raise KeyError(
            f"unknown reference experiment: {name!r}. "
            f"Available: {sorted(REFERENCE_EXPERIMENTS.keys())}"
        )
    return deepcopy(REFERENCE_EXPERIMENTS[name])