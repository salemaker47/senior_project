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
from src.optimizers import get_optimizer, get_scheduler, scheduler_needs_metric


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

        # Per-epoch buffers (concatenated in epoch_end hooks).
        self._val_tp:  list = []
        self._val_fp:  list = []
        self._val_fn:  list = []
        self._val_tn:  list = []
        self._test_tp: list = []
        self._test_fp: list = []
        self._test_fn: list = []
        self._test_tn: list = []

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

    def on_validation_epoch_end(self):
        if not self._val_tp:
            return
        tp = torch.cat(self._val_tp); fp = torch.cat(self._val_fp)
        fn = torch.cat(self._val_fn); tn = torch.cat(self._val_tn)

        seen = set()
        for log_name, fn_ in self._metric_pairs.items():
            if log_name in seen:
                continue
            seen.add(log_name)
            value = fn_(tp, fp, fn, tn)
            on_bar = log_name in ("val_dice", "val_iou")
            self.log(log_name, value, prog_bar=on_bar)

        self._val_tp.clear(); self._val_fp.clear()
        self._val_fn.clear(); self._val_tn.clear()

    # ------------------------------------------------------------------ #
    # Test
    # ------------------------------------------------------------------ #
    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        tp, fp, fn, tn = get_smp_stats(logits, y, threshold=self.threshold)
        self._test_tp.append(tp); self._test_fp.append(fp)
        self._test_fn.append(fn); self._test_tn.append(tn)
        return loss

    def on_test_epoch_end(self):
        if not self._test_tp:
            return
        tp = torch.cat(self._test_tp); fp = torch.cat(self._test_fp)
        fn = torch.cat(self._test_fn); tn = torch.cat(self._test_tn)

        seen = set()
        for log_name, fn_ in self._metric_pairs.items():
            test_name = log_name.replace("val_", "test_")
            if test_name in seen:
                continue
            seen.add(test_name)
            value = fn_(tp, fp, fn, tn)
            self.log(test_name, value)

        self._test_tp.clear(); self._test_fp.clear()
        self._test_fn.clear(); self._test_tn.clear()

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