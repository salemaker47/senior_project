"""
src/cls_losses.py

Classification loss registry.

Public API:
    get_cls_loss(name, **kwargs) -> nn.Module

Supported names (case-insensitive):
    "cross_entropy"         nn.CrossEntropyLoss, optional class_weights
    "cross_entropy_smooth"  nn.CrossEntropyLoss with label smoothing (default 0.1)
    "focal_ce"              Focal cross-entropy (down-weights easy examples)

To add a new loss: append a new `if n == "new_name":` branch.
Never modify existing branches.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Custom losses
# --------------------------------------------------------------------------- #
class FocalCELoss(nn.Module):
    """
    Focal cross-entropy: scales CE by (1 - p_t)^gamma so the model focuses
    on hard examples. gamma=0 recovers standard CE.

    Parameters
    ----------
    gamma       : focusing exponent (default 2.0, matches FigShare reference)
    weight      : optional per-class weight tensor
    reduction   : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CE per-sample (no reduction)
        log_p = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_p, targets, weight=self.weight, reduction="none")  # (N,)
        # p_t for the true class
        p_t = torch.exp(-ce)
        focal_loss = (1.0 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def get_cls_loss(name: str = "cross_entropy_smooth", **kwargs: Any) -> nn.Module:
    """
    Build a classification loss module by string name.

    Keyword arguments forwarded to each loss:
        cross_entropy        class_weights (Tensor | None)
        cross_entropy_smooth label_smoothing (float, default 0.1), class_weights
        focal_ce             gamma (float, default 2.0), class_weights
    """
    n = name.lower().strip()

    if n == "cross_entropy":
        weight = kwargs.get("class_weights")
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        return nn.CrossEntropyLoss(weight=weight)

    if n in ("cross_entropy_smooth", "ce_smooth"):
        weight = kwargs.get("class_weights")
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        smoothing = float(kwargs.get("label_smoothing", 0.1))
        return nn.CrossEntropyLoss(label_smoothing=smoothing, weight=weight)

    if n in ("focal_ce", "focal"):
        weight = kwargs.get("class_weights")
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        gamma = float(kwargs.get("gamma", 2.0))
        return FocalCELoss(gamma=gamma, weight=weight)

    raise ValueError(
        f"unknown cls loss name: {name!r}. "
        "Supported: 'cross_entropy', 'cross_entropy_smooth', 'focal_ce'"
    )
