"""
src/cls_models.py

Classification model registry — timm-based.

Public API:
    build_cls_model(name, num_classes, pretrained, **extra_kwargs) -> nn.Module
    count_parameters(model) -> int

Supported names (case-insensitive):
    "resnet50"               ResNet-50 (default baseline)
    "efficientnet_b0"        EfficientNet-B0 (lightweight)
    "efficientnet_b4"        EfficientNet-B4 (matches FigShare helper)
    "vit_small_patch16_224"  ViT-Small/16 (224×224 input)

To add a new model: append a new `if n == "new_name":` branch.
Never modify existing branches — old experiment configs must stay reproducible.
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn


def build_cls_model(
    name: str = "resnet50",
    num_classes: int = 3,
    pretrained: bool = True,
    **extra_kwargs: Any,
) -> nn.Module:
    """
    Build a timm classification model by string name.

    Parameters
    ----------
    name        : registry key (case-insensitive)
    num_classes : number of output classes (3 for figshare meningioma/glioma/pituitary)
    pretrained  : use ImageNet pretrained weights
    **extra_kwargs : forwarded to timm.create_model

    Returns
    -------
    nn.Module with a fully connected head of size `num_classes`
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for classification models. "
            "Install it with: pip install timm"
        ) from exc

    n = name.lower().strip()

    if n == "resnet50":
        return timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=num_classes, **extra_kwargs
        )

    if n == "efficientnet_b0":
        return timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=num_classes, **extra_kwargs
        )

    if n == "efficientnet_b4":
        return timm.create_model(
            "efficientnet_b4", pretrained=pretrained, num_classes=num_classes, **extra_kwargs
        )

    if n == "vit_small_patch16_224":
        return timm.create_model(
            "vit_small_patch16_224", pretrained=pretrained,
            num_classes=num_classes, **extra_kwargs
        )

    raise ValueError(
        f"unknown cls model name: {name!r}. "
        "Supported: 'resnet50', 'efficientnet_b0', 'efficientnet_b4', "
        "'vit_small_patch16_224'"
    )


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
