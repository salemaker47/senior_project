"""
configs/cls/reference_experiments.py

Canonical recipes for classification experiments on figshare.

Pattern mirrors configs/seg/reference_experiments.py exactly:
    BASE_EXPERIMENT  — defaults shared by every recipe
    _RECIPE_OVERRIDES — per-recipe deltas only
    get_experiment(recipe, *, dataset, split_scheme, fold, name) -> Dict

Usage:
    from configs.cls.reference_experiments import get_experiment, REFERENCE_RECIPES

    # Baseline ResNet-50 run, fold 1, image-level splits
    EXPERIMENT = get_experiment("cls01_resnet50", fold=1)

    # EfficientNet-B4, patient-level splits
    EXPERIMENT = get_experiment("cls03_effb4",
                                split_scheme="patient_level", fold=2)

    # Iterate over all recipes
    for recipe in REFERENCE_RECIPES:
        EXPERIMENT = get_experiment(recipe, fold=1)

Notes:
    - Classification experiments only run on figshare (3 classes).
      brats2020 is glioma-only — unsuitable for multiclass classification.
    - Training always uses GT masks (mask_source is irrelevant at train time).
      Eval A uses GT masks; Eval B uses predicted masks from a seg experiment.
    - No REFERENCE_EXPERIMENTS precomputed dict — run results are not embedded
      in config. Check outputs/tables/classification/<dataset>/<exp>/... for results.
"""

from copy import deepcopy
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Defaults shared across all classification recipes
# --------------------------------------------------------------------------- #
BASE_EXPERIMENT: Dict[str, Any] = {
    "task": "classification",
    # name, dataset, split_scheme, fold are passed per-call to get_experiment.

    "num_classes":  3,
    "patch_size":   224,
    "padding_frac": 0.10,
    "batch_size":   32,
    "num_workers":  2,

    "augmentation_strength": "light",

    "model_name": "resnet50",
    "pretrained": True,

    "loss_name":   "cross_entropy_smooth",
    "loss_kwargs": {"label_smoothing": 0.1, "class_weights": None},

    "optimizer_name":   "adamw",
    "optimizer_kwargs": {"lr": 1e-4, "weight_decay": 1e-4},

    "scheduler_name":   "cosine",
    "scheduler_kwargs": {"T_max": 50, "eta_min": 1e-6},

    "monitor":      "val_macro_f1",
    "monitor_mode": "max",

    "max_epochs": 50,
    "patience":   10,
    "seed":       42,
}


# --------------------------------------------------------------------------- #
# Recipes — each is JUST the delta from BASE_EXPERIMENT
# --------------------------------------------------------------------------- #
_RECIPE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "cls01_resnet50": {},                                       # all defaults (baseline)

    "cls02_effb0": {
        "model_name": "efficientnet_b0",
    },

    "cls03_effb4": {
        "model_name": "efficientnet_b4",
        "batch_size": 16,                                       # reduced for VRAM
    },

    "cls04_vit": {
        "model_name": "vit_small_patch16_224",
        "batch_size": 16,                                       # reduced for VRAM
    },
}

REFERENCE_RECIPES = sorted(_RECIPE_OVERRIDES.keys())


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_experiment(
    recipe: str,
    *,
    dataset: str = "figshare",
    split_scheme: str = "image_level",
    fold: int = 1,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compose a complete EXPERIMENT dict by merging BASE_EXPERIMENT + a recipe
    + per-call dataset/scheme/fold/name.

    Parameters
    ----------
    recipe       : one of REFERENCE_RECIPES (cls01_resnet50 ... cls04_vit).
    dataset      : "figshare" (only figshare supported for classification).
    split_scheme : "image_level" | "patient_level".
    fold         : 1..5.
    name         : experiment name used for all output directories and CSV row labels.
                   IMPORTANT: always pass this explicitly.
                   The auto-generated default f"{recipe}_{split_scheme}" (e.g.
                   "cls01_resnet50_image_level") diverges from the short-form names
                   used in notebooks (e.g. "cls01_resnet50"). If name= is not passed,
                   checkpoint and table directories will not match what the notebooks
                   expect.

    Returns
    -------
    dict — safe to mutate (each call is a fresh deep copy).
    """
    if recipe not in _RECIPE_OVERRIDES:
        raise KeyError(
            f"unknown cls recipe: {recipe!r}. Available: {REFERENCE_RECIPES}"
        )

    cfg = deepcopy(BASE_EXPERIMENT)
    cfg.update(_RECIPE_OVERRIDES[recipe])
    cfg["dataset"]      = dataset
    cfg["split_scheme"] = split_scheme
    cfg["fold"]         = int(fold)
    cfg["recipe"]       = recipe
    cfg["name"]         = name if name is not None else f"{recipe}_{split_scheme}"
    return cfg
