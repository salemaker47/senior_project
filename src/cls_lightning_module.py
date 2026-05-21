"""
src/cls_lightning_module.py

PyTorch Lightning module for 3-class brain tumor classification: BrainTumorClsModule.

REGISTRY-DRIVEN: optimizer and scheduler are selected by string name.
New variants are added by extending src/optimizers.py — never by editing this file.

Defaults:
    optimizer    = adamw, lr=1e-4, weight_decay=1e-4
    scheduler    = cosine, T_max=50, eta_min=1e-6
    monitor      = val_macro_f1   (EarlyStopping + ModelCheckpoint maximize this)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.cls_metrics import macro_f1_from_preds, accuracy_from_preds
from src.optimizers import get_optimizer, get_scheduler, scheduler_needs_metric


class BrainTumorClsModule(pl.LightningModule):
    """
    Parameters
    ----------
    model
        nn.Module producing raw logits of shape (N, num_classes).
    loss_fn
        nn.Module operating on (logits, labels). See src/cls_losses.py.
    num_classes
        Number of output classes (default 3).

    optimizer_name, optimizer_kwargs
        Forwarded to src/optimizers.get_optimizer. Must include 'lr'.
    scheduler_name, scheduler_kwargs
        Forwarded to src/optimizers.get_scheduler. Pass `None` to disable.
    scheduler_monitor
        Metric name the scheduler watches (only relevant for metric-driven
        schedulers like ReduceLROnPlateau).
    scheduler_interval
        'epoch' or 'step'. Default 'epoch'.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        num_classes: int = 3,
        # ---- optimizer ----
        optimizer_name: str = "adamw",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # ---- scheduler ----
        scheduler_name: Optional[str] = "cosine",
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_monitor: str = "val_macro_f1",
        scheduler_interval: str = "epoch",
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.num_classes = num_classes

        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = dict(optimizer_kwargs or {"lr": 1e-4, "weight_decay": 1e-4})
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = dict(scheduler_kwargs or {"T_max": 50, "eta_min": 1e-6})
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_interval = scheduler_interval

        # Per-epoch accumulation buffers; flushed in _flush_cls_buffers.
        self._val_logits: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log(
            "train_loss", loss,
            prog_bar=True, on_step=False, on_epoch=True,
            batch_size=x.size(0),
        )
        return loss

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(y.detach().cpu())

        self.log(
            "val_loss", loss,
            prog_bar=False, on_step=False, on_epoch=True,
            batch_size=x.size(0),
        )
        return loss

    def _flush_cls_buffers(
        self,
        logits_buf: List[torch.Tensor],
        labels_buf: List[torch.Tensor],
        prefix: str,
    ) -> None:
        all_logits = torch.cat(logits_buf, dim=0)    # (N, C)
        all_labels = torch.cat(labels_buf, dim=0)    # (N,)
        preds  = all_logits.argmax(dim=1).numpy()
        labels = all_labels.numpy()

        macro_f1 = macro_f1_from_preds(preds, labels, num_classes=self.num_classes)
        acc      = accuracy_from_preds(preds, labels)

        on_bar = (prefix == "val_")
        self.log(f"{prefix}macro_f1", macro_f1, prog_bar=on_bar)
        self.log(f"{prefix}accuracy", acc,      prog_bar=False)

        logits_buf.clear()
        labels_buf.clear()

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        self._flush_cls_buffers(self._val_logits, self._val_labels, "val_")

    # ------------------------------------------------------------------ #
    # Optimizer + scheduler — both registry-driven
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.optimizer_name,
            self.parameters(),
            **self.optimizer_kwargs,
        )
        scheduler = get_scheduler(
            self.scheduler_name,
            optimizer,
            **self.scheduler_kwargs,
        )
        if scheduler is None:
            return optimizer

        sched_cfg: Dict[str, Any] = {
            "scheduler": scheduler,
            "interval": self.scheduler_interval,
            "frequency": 1,
        }
        if scheduler_needs_metric(self.scheduler_name):
            sched_cfg["monitor"] = self.scheduler_monitor

        return {"optimizer": optimizer, "lr_scheduler": sched_cfg}
