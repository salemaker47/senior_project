"""
src/sg_lightning_module.py

PyTorch Lightning module for binary tumor segmentation: BrainTumorSegModule.

REGISTRY-DRIVEN: optimizer, scheduler, and metric-kind are all selected by
string name. New variants are added by extending the registries in
src/sg_metrics.py and src/optimizers.py — never by editing this file.

Defaults match the FigShare reference notebook exactly:
    optimizer    = adam, lr=1e-4
    scheduler    = reduce_on_plateau, factor=0.1, patience=5, monitor=val_loss
    metric_kind  = "micro"  (globally pooled Dice/IoU; EarlyStopping monitors val_dice)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.sg_metrics import get_smp_stats, get_metric_kind_pairs
from src.optimizers import get_optimizer, get_scheduler, scheduler_needs_metric, build_scheduler_cfg


class BrainTumorSegModule(pl.LightningModule):
    """
    Parameters
    ----------
    model
        nn.Module producing raw logits of shape (N, 1, H, W).
    loss_fn
        nn.Module operating on (logits, target). See src/sg_losses.py.
    threshold
        Binarization threshold for prediction-time metrics. Default 0.5.

    optimizer_name, optimizer_kwargs
        Forwarded to src/optimizers.get_optimizer. Must include 'lr'.
    scheduler_name, scheduler_kwargs
        Forwarded to src/optimizers.get_scheduler. Pass `None` to disable.
    scheduler_monitor
        Metric name the scheduler watches (only used by metric-driven
        schedulers like ReduceLROnPlateau).
    scheduler_interval
        'epoch' or 'step'. Default 'epoch'.

    metric_kind
        "micro". Globally pooled Dice/IoU logged as val_dice / val_iou.
        Only "micro" is supported.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        threshold: float = 0.5,
        # ---- optimizer ----
        optimizer_name: str = "adam",
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # ---- scheduler ----
        scheduler_name: Optional[str] = "reduce_on_plateau",
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_monitor: str = "val_loss",
        scheduler_interval: str = "epoch",
        # ---- metrics ----
        metric_kind: str = "micro",
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.threshold = threshold

        # Optimizer / scheduler config (built lazily in configure_optimizers).
        self.optimizer_name = optimizer_name
        self.optimizer_kwargs = dict(optimizer_kwargs or {"lr": 1e-4})
        self.scheduler_name = scheduler_name
        self.scheduler_kwargs = dict(scheduler_kwargs or {
            "mode": "min", "factor": 0.1, "patience": 5, "min_lr": 1e-7,
        })
        self.scheduler_monitor = scheduler_monitor
        self.scheduler_interval = scheduler_interval

        # Metric pairs (dict: logged_name -> reduction_fn(tp,fp,fn,tn)).
        self.metric_kind = metric_kind
        self._metric_pairs = get_metric_kind_pairs(metric_kind)

        # Per-epoch stat buffers — flushed in _flush_epoch_buffers.
        self._val_tp: list = []
        self._val_fp: list = []
        self._val_fn: list = []
        self._val_tn: list = []

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

        tp, fp, fn, tn = get_smp_stats(logits, y, threshold=self.threshold)
        self._val_tp.append(tp); self._val_fp.append(fp)
        self._val_fn.append(fn); self._val_tn.append(tn)

        self.log(
            "val_loss", loss,
            prog_bar=True, on_step=False, on_epoch=True,
            batch_size=x.size(0),
        )
        return loss

    def _flush_val_buffers(self) -> None:
        tp = torch.cat(self._val_tp); fp = torch.cat(self._val_fp)
        fn = torch.cat(self._val_fn); tn = torch.cat(self._val_tn)
        for val_name, fn_ in self._metric_pairs.items():
            on_bar = val_name in ("val_dice", "val_iou")
            self.log(val_name, fn_(tp, fp, fn, tn), prog_bar=on_bar)
        self._val_tp.clear(); self._val_fp.clear()
        self._val_fn.clear(); self._val_tn.clear()

    def on_validation_epoch_end(self):
        if not self._val_tp:
            return
        self._flush_val_buffers()

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
        monitor = self.scheduler_monitor if scheduler_needs_metric(self.scheduler_name) else None
        sched_cfg = build_scheduler_cfg(scheduler, self.scheduler_interval, monitor)
        if sched_cfg is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": sched_cfg}