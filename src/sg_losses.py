"""
src/sg_losses.py

Segmentation loss factory. Every loss is selected by NAME so the EXPERIMENT
dict can swap losses with a single config flag.

Built-in names (case-insensitive):
    "bce" / "bcewithlogits" / "bce_with_logits"   torch.nn.BCEWithLogitsLoss
    "dice"         SMP DiceLoss (mode='binary')
    "focal"        SMP FocalLoss (mode='binary')
    "lovasz"       SMP LovaszLoss (mode='binary')
    "dice_bce"     Dice + BCE  (sum, equal weight by default)
    "dice_focal"   Dice + Focal
    "combo"        generic weighted combination, see kwargs below

All losses expect raw LOGITS (not sigmoids), matching `activation=None`
on the model side.

REGISTRY PATTERN (project instruction §5):
    Add new losses by adding a new `if n == "...":` branch in `get_loss`.
    Do NOT modify existing branches. Old experiments must reproduce.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from segmentation_models_pytorch.losses import (
    DiceLoss,
    FocalLoss,
    LovaszLoss,
)


# --------------------------------------------------------------------------- #
# Combiner
# --------------------------------------------------------------------------- #
class CombinedLoss(nn.Module):
    """
    Weighted sum of per-component losses.

    Example:
        CombinedLoss(
            losses=[nn.BCEWithLogitsLoss(), DiceLoss(mode='binary')],
            weights=[1.0, 1.0],
            names=['bce', 'dice'],
        )

    Forward signature is `loss(logits, target)` so it's drop-in for any
    single-loss nn.Module.
    """

    def __init__(
        self,
        losses: Sequence[nn.Module],
        weights: Optional[Sequence[float]] = None,
        names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.losses = nn.ModuleList(list(losses))
        if weights is None:
            weights = [1.0] * len(self.losses)
        assert len(weights) == len(self.losses), "weights/losses length mismatch"
        self.register_buffer(
            "weights",
            torch.tensor(weights, dtype=torch.float32),
            persistent=False,
        )
        self.names = (
            list(names) if names is not None
            else [f"l{i}" for i in range(len(losses))]
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = logits.new_zeros(())   # scalar tensor on the same device/dtype as input
        for w, loss_fn in zip(self.weights, self.losses):
            total = total + w * loss_fn(logits, target)
        return total


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_loss(name: str = "bce", **kwargs) -> nn.Module:
    """
    Build a loss function by name.

    SMP losses accept target shape (N, 1, H, W) or (N, H, W); both work.
    All losses below operate on raw logits.

    Common kwargs:
        smooth        (dice components, default 1.0)
        alpha, gamma  (focal components)
        pos_weight    (BCE: scalar reweighting positive class)
        bce_weight, dice_weight, focal_weight  (combo weights)

    For `name="combo"`:
        components : list of names, e.g. ["bce", "dice", "focal"]
        weights    : list of floats, same length as components
    """
    n = name.lower()

    if n in ("bce", "bcewithlogits", "bce_with_logits"):
        pos_weight = kwargs.get("pos_weight", None)
        if pos_weight is not None:
            pos_weight = torch.tensor(float(pos_weight))
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if n == "dice":
        return DiceLoss(
            mode="binary",
            from_logits=True,
            smooth=kwargs.get("smooth", 1.0),
        )

    if n == "focal":
        return FocalLoss(
            mode="binary",
            alpha=kwargs.get("alpha", 0.25),
            gamma=kwargs.get("gamma", 2.0),
        )

    if n == "lovasz":
        return LovaszLoss(mode="binary", from_logits=True)

    if n in ("dice_bce", "bce_dice"):
        return CombinedLoss(
            losses=[
                nn.BCEWithLogitsLoss(),
                DiceLoss(mode="binary", from_logits=True,
                         smooth=kwargs.get("smooth", 1.0)),
            ],
            weights=[
                kwargs.get("bce_weight", 1.0),
                kwargs.get("dice_weight", 1.0),
            ],
            names=["bce", "dice"],
        )

    if n in ("dice_focal", "focal_dice"):
        return CombinedLoss(
            losses=[
                FocalLoss(
                    mode="binary",
                    alpha=kwargs.get("alpha", 0.25),
                    gamma=kwargs.get("gamma", 2.0),
                ),
                DiceLoss(mode="binary", from_logits=True,
                         smooth=kwargs.get("smooth", 1.0)),
            ],
            weights=[
                kwargs.get("focal_weight", 1.0),
                kwargs.get("dice_weight", 1.0),
            ],
            names=["focal", "dice"],
        )

    if n == "combo":
        components = kwargs.get("components")
        weights = kwargs.get("weights")
        if not components:
            raise ValueError("combo loss requires `components=[...]` kwarg")
        component_losses = [get_loss(c) for c in components]
        return CombinedLoss(
            losses=component_losses,
            weights=weights,
            names=components,
        )

    raise ValueError(
        f"unknown loss name: {name!r}. See src/sg_losses.py for the registry."
    )