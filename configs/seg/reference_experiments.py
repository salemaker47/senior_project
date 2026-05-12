"""
configs/seg/reference_experiments.py

The 7 canonical recipes for segmentation experiments (§13 of the project
instruction). Replaces the old `00_env_setups.ipynb` cookbook.

Pattern: BASE_EXPERIMENT holds defaults shared by every run. _RECIPE_OVERRIDES
holds just the deltas per recipe (loss/model/preprocessing). dataset,
split_scheme, fold, and name are passed as explicit arguments to
`get_experiment(...)` — so the SAME 7 recipes drive runs on both figshare
and brats2020.

Usage:
    # FigShare §13 reference reproduction (image_level matches the paper)
    EXPERIMENT = get_experiment("01_dice", fold=1)
        # -> name="01_dice_image_level", dataset="figshare",
        #    split_scheme="image_level", fold=1

    # Same recipe on BraTS2020 with the methodologically correct scheme
    EXPERIMENT = get_experiment("01_dice", dataset="brats2020",
                                split_scheme="patient_level", fold=1)
        # -> name="01_dice_patient_level", dataset="brats2020",
        #    split_scheme="patient_level", fold=1

    # Iterate over all 7 recipes
    for recipe in REFERENCE_RECIPES:
        EXPERIMENT = get_experiment(recipe, dataset="brats2020",
                                    split_scheme="patient_level", fold=1)
"""

from copy import deepcopy
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Defaults shared across all reference recipes
# --------------------------------------------------------------------------- #
BASE_EXPERIMENT: Dict[str, Any] = {
    "task": "segmentation",
    # name, dataset, split_scheme, fold are passed per-call to get_experiment.

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

    "metric_kind": "micro",

    "max_epochs": 100,
    "patience":   15,
    "threshold":  0.5,
    "seed":       42,
}


# --------------------------------------------------------------------------- #
# The 7 §13 recipes — each is JUST the delta from BASE_EXPERIMENT
# --------------------------------------------------------------------------- #
_RECIPE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "01_dice": {
        "loss_name":   "dice",
        "loss_kwargs": {},
    },
    "02_bce": {
        "loss_name":   "bce",
        "loss_kwargs": {},
    },
    "03_dicebce": {
        "loss_name":   "dice_bce",
        "loss_kwargs": {"bce_weight": 1.0, "dice_weight": 1.0},
    },
    "04_dicefocal": {
        "loss_name":   "dice_focal",
        "loss_kwargs": {
            "dice_weight":  1.0,
            "focal_weight": 1.0,
            "alpha":        0.25,
            "gamma":        2.0,
        },
    },
    "05_lovasz": {
        "loss_name":   "lovasz",
        "loss_kwargs": {},
    },
    "06_clahe_dicebce": {
        "preprocessing": "clahe",                          # only recipe with CLAHE
        "loss_name":     "dice_bce",
        "loss_kwargs":   {"bce_weight": 1.0, "dice_weight": 1.0},
    },
    "07_unetpp_effb4_dicebce": {
        "model_name":  "smp_unetpp_efficientnetb4",        # different architecture
        "batch_size":  6,                                   # reduced for VRAM
        "loss_name":   "dice_bce",
        "loss_kwargs": {"bce_weight": 1.0, "dice_weight": 1.0},
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
    recipe : one of REFERENCE_RECIPES (`01_dice` ... `07_unetpp_effb4_dicebce`).
    dataset : "figshare" | "brats2020".
    split_scheme : "image_level" | "patient_level".
    fold : 1..5.
    name : if None, auto-composed as f"{recipe}_{split_scheme}" so output
           directories like `outputs/checkpoints/.../01_dice_image_level/` vs
           `.../01_dice_patient_level/` separate naturally.

    Returns
    -------
    dict — safe to mutate (each call is a fresh deep copy).
    """
    if recipe not in _RECIPE_OVERRIDES:
        raise KeyError(
            f"unknown recipe: {recipe!r}. Available: {REFERENCE_RECIPES}"
        )

    cfg = deepcopy(BASE_EXPERIMENT)
    cfg.update(_RECIPE_OVERRIDES[recipe])
    cfg["dataset"]      = dataset
    cfg["split_scheme"] = split_scheme
    cfg["fold"]         = int(fold)
    cfg["recipe"]       = recipe                          # track for analysis
    cfg["name"]         = name if name is not None else f"{recipe}_{split_scheme}"
    return cfg


# Backward-compat: precomputed registry keyed by the old combined name.
# Lets old code that does `REFERENCE_EXPERIMENTS["01_dice_image_level"]` keep
# working. New code should call get_experiment() directly.
REFERENCE_EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    f"{recipe}_image_level":   get_experiment(recipe, split_scheme="image_level")
    for recipe in REFERENCE_RECIPES
}
REFERENCE_EXPERIMENTS.update({
    f"{recipe}_patient_level": get_experiment(recipe, split_scheme="patient_level")
    for recipe in REFERENCE_RECIPES
})