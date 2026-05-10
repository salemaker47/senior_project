"""
src/optimizers.py

Optimizer + LR scheduler registries. Shared across seg and cls.

REGISTRY PATTERN (project instruction §5):
    Add new optimizers by appending a branch to `get_optimizer`.
    Add new schedulers by appending a branch to `get_scheduler`.
    Do NOT modify existing branches.

Notebooks reference both by string name in the EXPERIMENT dict, e.g.:
    "optimizer_name":   "adam",
    "optimizer_kwargs": {"lr": 1e-4},
    "scheduler_name":   "reduce_on_plateau",
    "scheduler_kwargs": {"mode": "min", "factor": 0.1, "patience": 5},
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import torch
from torch.optim.lr_scheduler import _LRScheduler  # for type hints only

LRScheduler = Optional[object]   # broad: ReduceLROnPlateau is not a _LRScheduler


# --------------------------------------------------------------------------- #
# Optimizers
# --------------------------------------------------------------------------- #
def get_optimizer(
    name: str,
    params: Iterable,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Build an optimizer by name.

    Supported names (case-insensitive):
        "adam"     torch.optim.Adam
        "adamw"    torch.optim.AdamW
        "sgd"      torch.optim.SGD     (momentum=0.9 default if not provided)
        "rmsprop"  torch.optim.RMSprop
    """
    n = name.lower()

    if n == "adam":
        return torch.optim.Adam(params, **kwargs)

    if n == "adamw":
        return torch.optim.AdamW(params, **kwargs)

    if n == "sgd":
        kwargs.setdefault("momentum", 0.9)
        return torch.optim.SGD(params, **kwargs)

    if n == "rmsprop":
        return torch.optim.RMSprop(params, **kwargs)

    raise ValueError(
        f"unknown optimizer name: {name!r}. "
        f"Add a new branch to get_optimizer in src/optimizers.py."
    )


# --------------------------------------------------------------------------- #
# LR schedulers
# --------------------------------------------------------------------------- #
def get_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    **kwargs: Any,
) -> LRScheduler:
    """
    Build a learning-rate scheduler by name.

    Supported names (case-insensitive):
        "reduce_on_plateau"     ReduceLROnPlateau (needs a monitored metric)
        "cosine"                CosineAnnealingLR
        "cosine_warm_restarts"  CosineAnnealingWarmRestarts
        "step"                  StepLR
        "multistep"             MultiStepLR
        "exponential"           ExponentialLR
        "none" / None / ""      returns None (no scheduler)

    The interval ('epoch' vs 'step') the Lightning trainer uses is decided by
    the caller in configure_optimizers — see sg_lightning_module.py.
    """
    if name is None:
        return None

    n = name.lower()
    if n in ("none", ""):
        return None

    if n == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)

    if n == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)

    if n == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)

    if n == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)

    if n == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)

    if n == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)

    raise ValueError(
        f"unknown scheduler name: {name!r}. "
        f"Add a new branch to get_scheduler in src/optimizers.py."
    )


# --------------------------------------------------------------------------- #
# Helper used by the Lightning module
# --------------------------------------------------------------------------- #
def scheduler_needs_metric(name: Optional[str]) -> bool:
    """
    True if the scheduler steps based on a monitored metric (used to set
    Lightning's `monitor` field correctly in configure_optimizers).
    ReduceLROnPlateau is the only one in this file that does.
    """
    if name is None:
        return False
    return name.lower() == "reduce_on_plateau"