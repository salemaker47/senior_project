"""
src/sg_models.py

Segmentation model factory. Each architecture is selected by NAME so the
EXPERIMENT dict can swap models with a single config flag.

Built-in names (case-insensitive):
    "smp_unet_resnet34"           U-Net, ResNet34 encoder    (used by exps 01-06)
    "smp_unet_resnet50"           U-Net, ResNet50 encoder
    "smp_unetpp_efficientnetb4"   U-Net++, EfficientNet-B4   (used by exp 07)
    "smp_unetpp_resnet34"         U-Net++, ResNet34
    "smp_resunet_resnet34"        Linknet (closest ResU-Net), ResNet34
    "smp_manet_resnet34"          MA-Net, ResNet34

All models output raw LOGITS (activation=None) so they pair correctly with
the loss functions in src/sg_losses.py.

REGISTRY PATTERN (project instruction §5):
    Add new architectures by adding a new branch to `build_model`.
    Do NOT modify existing branches. Old experiments must reproduce.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn
import segmentation_models_pytorch as smp


# Default kwargs for SMP architectures. Overrideable via build_model args.
_DEFAULTS: Dict[str, Any] = {
    "in_channels":     3,    # 3-ch grayscale-replicated for ImageNet pretraining
    "classes":         1,    # binary tumor segmentation
    "encoder_weights": "imagenet",
    "activation":      None, # raw logits
}


def build_model(
    name: str = "smp_unet_resnet34",
    in_channels: int = 3,
    classes: int = 1,
    encoder_weights: Optional[str] = "imagenet",
    activation: Optional[str] = None,
    **extra_kwargs: Any,
) -> nn.Module:
    """
    Build a segmentation model by name.

    Parameters mirror SMP's API. `extra_kwargs` are passed straight to the
    SMP constructor for things like decoder_channels overrides.
    """
    common = {
        "in_channels":     in_channels,
        "classes":         classes,
        "encoder_weights": encoder_weights,
        "activation":      activation,
    }
    common.update(extra_kwargs)

    n = name.lower()

    if n == "smp_unet_resnet34":
        return smp.Unet(encoder_name="resnet34", **common)

    if n == "smp_unet_resnet50":
        return smp.Unet(encoder_name="resnet50", **common)

    if n == "smp_unetpp_efficientnetb4":
        return smp.UnetPlusPlus(encoder_name="efficientnet-b4", **common)

    if n == "smp_unetpp_resnet34":
        return smp.UnetPlusPlus(encoder_name="resnet34", **common)

    if n == "smp_resunet_resnet34":
        # SMP doesn't ship a "ResUNet"; Linknet with a ResNet encoder is the
        # closest residual U-shape. If a strict ResUNet implementation is
        # needed later, this is where to drop it in.
        return smp.Linknet(encoder_name="resnet34", **common)

    if n == "smp_manet_resnet34":
        return smp.MAnet(encoder_name="resnet34", **common)

    raise ValueError(
        f"unknown model name: {name!r}. See src/sg_models.py for the registry."
    )


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count. Used by Enhancement E (training summary)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)