"""
src/train_utils.py

Training-side helpers for PyTorch Lightning. Shared across seg and cls.

    set_global_seed              seed RNGs reproducibly
    gather_repro_metadata        Enhancement F: git/lib/GPU stamp for *_config.json
    TrainingTimingCallback       Enhancement E: wall-clock + peak GPU memory
    build_callbacks              ModelCheckpoint + EarlyStopping + LearningRateMonitor + TimingCallback
    build_trainer                Lightning Trainer + CSVLogger
    export_plain_state_dict      writes best_model.pt for non-Lightning reload
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger

PathLike = Union[str, Path]

DEFAULT_REPO_ROOT = "/content/senior_project"   # the cloned-repo path on Colab


# --------------------------------------------------------------------------- #
# Reproducibility — RNG seeding
# --------------------------------------------------------------------------- #
def set_global_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed all standard sources of randomness. `deterministic=True` forces CuDNN
    into a deterministic mode (slower; only enable for byte-for-byte repro).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# Reproducibility — Enhancement F: metadata stamp
# --------------------------------------------------------------------------- #
def _safe_run(cmd: List[str], cwd: Optional[PathLike] = None) -> Optional[str]:
    """Run a subprocess, return stdout stripped, or None on any failure."""
    try:
        result = subprocess.run(
            cmd, cwd=str(cwd) if cwd else None,
            check=True, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _safe_version(pkg: str) -> Optional[str]:
    """Return importlib version of a package, or None if not installed."""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:                                                    # pragma: no cover
        return None
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def gather_repro_metadata(repo_root: Optional[PathLike] = None) -> Dict[str, Any]:
    """
    Collect reproducibility metadata for stamping into experiment_config.json
    and test_eval_config.json.

    Returns a dict with:
        timestamp_utc            ISO-8601
        python_version           "3.X.Y"
        platform                 Linux-X.Y-Z or similar
        lib_versions             {torch, pytorch_lightning, segmentation_models_pytorch,
                                  albumentations, numpy, pandas, opencv-python-headless}
        git_commit_sha           HEAD sha (or "unknown")
        git_branch               branch name (or "unknown")
        git_working_tree_clean   bool (False if uncommitted changes)
        gpu                      {available, name, count, compute_capability} or {available: False}
        cuda                     {version, cudnn_version} or None

    `repo_root` defaults to /content/senior_project (the Colab clone). Pass
    a different path when running from a different working tree.
    """
    from datetime import datetime, timezone

    repo_root = Path(repo_root) if repo_root is not None else Path(DEFAULT_REPO_ROOT)

    # Git info — gracefully handles non-repo / git-not-installed.
    git_sha    = _safe_run(["git", "rev-parse", "HEAD"], cwd=repo_root) or "unknown"
    git_branch = _safe_run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root) or "unknown"
    status_out = _safe_run(["git", "status", "--porcelain"], cwd=repo_root)
    git_clean  = (status_out is not None) and (status_out == "")

    # GPU + CUDA
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        gpu_info = {
            "available":          True,
            "name":               torch.cuda.get_device_name(0),
            "count":              int(torch.cuda.device_count()),
            "compute_capability": f"{cap[0]}.{cap[1]}",
        }
        cuda_info = {
            "version":        getattr(torch.version, "cuda", None),
            "cudnn_version":  torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        }
    else:
        gpu_info = {"available": False}
        cuda_info = None

    return {
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "python_version":  platform.python_version(),
        "platform":        platform.platform(),
        "lib_versions": {
            "torch":                         _safe_version("torch"),
            "pytorch_lightning":             _safe_version("pytorch-lightning"),
            "segmentation_models_pytorch":   _safe_version("segmentation-models-pytorch"),
            "albumentations":                _safe_version("albumentations"),
            "numpy":                         _safe_version("numpy"),
            "pandas":                        _safe_version("pandas"),
            "opencv-python-headless":        _safe_version("opencv-python-headless"),
            "timm":                          _safe_version("timm"),
        },
        "git_commit_sha":         git_sha,
        "git_branch":             git_branch,
        "git_working_tree_clean": git_clean,
        "gpu":                    gpu_info,
        "cuda":                   cuda_info,
    }


# --------------------------------------------------------------------------- #
# Enhancement E: training-time + peak GPU memory tracker
# --------------------------------------------------------------------------- #
class TrainingTimingCallback(pl.Callback):
    """
    Tracks wall-clock training time and peak GPU memory per fold. After
    `trainer.fit(...)` returns, read:
        cb.train_seconds       float (None if training never started)
        cb.peak_gpu_mem_mb     float (None if running on CPU)

    Used by Enhancement E (`cv_training_summary.csv` aggregator in M6).
    """

    def __init__(self) -> None:
        super().__init__()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.peak_gpu_mem_mb: Optional[float] = None

    def on_train_start(self, trainer, pl_module) -> None:
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_end(self, trainer, pl_module) -> None:
        self.end_time = time.time()
        if torch.cuda.is_available():
            self.peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated()) / (1024 ** 2)

    @property
    def train_seconds(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return float(self.end_time - self.start_time)


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
class EpochSummaryPrinter(pl.Callback):
    """
    Print one persistent line per validation epoch — alongside the default
    progress bar — so per-epoch metrics survive notebook reopens (the
    progress bar's line gets overwritten on reload; print() lines do not).

    Output format:
        [HH:MM:SS] E  12/99 | train_loss=0.2410 | val_loss=0.3120 | val_dice=0.7820 | val_iou=0.6510
    """

    def __init__(
        self,
        metric_keys=("train_loss", "val_loss", "val_dice", "val_iou"),
    ):
        super().__init__()
        self.metric_keys = tuple(metric_keys)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return                                     # skip the pre-train sanity check
        m = trainer.callback_metrics
        parts = [f"E{trainer.current_epoch:>3}/{trainer.max_epochs - 1}"]
        for key in self.metric_keys:
            v = m.get(key)
            if v is None:
                continue
            try:
                parts.append(f"{key}={float(v):.4f}")
            except (TypeError, ValueError):
                pass
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        # flush=True forces Colab to commit the line immediately, so it
        # appears in the output stream right after each validation epoch
        # rather than being held in a buffer.
        print(f"[{ts}] " + " | ".join(parts), flush=True)


def build_callbacks(
    ckpt_dir,
    monitor: str = "val_dice",
    mode: str = "max",
    patience: int = 15,
    epoch_summary_keys=("train_loss", "val_loss", "val_dice", "val_iou"),
):
    """
    Build the standard callback list for a seg training run:
        - ModelCheckpoint        (save best.ckpt on `monitor`)
        - EarlyStopping          (stop when `monitor` plateaus for `patience` epochs)
        - LearningRateMonitor    (log LR per epoch — useful for ReduceLROnPlateau)
        - TrainingTimingCallback (Enhancement E — captures train_seconds, peak_gpu_mem)
        - EpochSummaryPrinter    (persistent one-line-per-epoch in notebook output)
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        ),
        EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        TrainingTimingCallback(),
        EpochSummaryPrinter(metric_keys=epoch_summary_keys),
    ]
    return callbacks


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
def build_trainer(
    callbacks: List[pl.Callback],
    log_dir: PathLike,
    max_epochs: int = 50,
    accelerator: str = "auto",
    devices: Any = "auto",
    precision: Any = "auto",
    deterministic: bool = False,
    log_every_n_steps: int = 10,
    enable_progress_bar: bool = True,
) -> pl.Trainer:
    """
    Build a Lightning Trainer with CSVLogger so we get a clean
    metrics.csv we can replot or aggregate later.

    `precision="auto"` selects "16-mixed" on GPU and "32-true" on CPU.
    Logs land at log_dir/version_0/metrics.csv (no lightning_logs subfolder).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(log_dir), name="", version=0)

    if precision == "auto":
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    return pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        deterministic=deterministic,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=enable_progress_bar,
        num_sanity_val_steps=0,
    )


# --------------------------------------------------------------------------- #
# State-dict helpers
# --------------------------------------------------------------------------- #
def strip_model_prefix(sd: dict) -> dict:
    """Remove the 'model.' prefix that LightningModule.model adds to all keys."""
    return {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}


# --------------------------------------------------------------------------- #
# Plain-PyTorch checkpoint export
# --------------------------------------------------------------------------- #
def export_plain_state_dict(
    lightning_ckpt_path: PathLike,
    out_pt_path: PathLike,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Read a Lightning .ckpt and write a plain torch checkpoint with the
    "model." prefix stripped, so non-Lightning code can do:

        ckpt = torch.load("best_model.pt", map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    Output schema:
        {
            "state_dict":  <model.state_dict() w/ "model." prefix stripped>,
            "epoch":       <int>,
            "best_score":  <float | None>,
            "extra_meta":  <whatever the caller supplied>,
        }
    """
    src = Path(lightning_ckpt_path)
    dst = Path(out_pt_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    blob = torch.load(src, map_location="cpu", weights_only=False)
    stripped = strip_model_prefix(blob.get("state_dict", blob))
    best_score_raw = blob.get("best_model_score")
    out = {
        "state_dict": stripped,
        "epoch":      int(blob.get("epoch", -1)),
        "best_score": float(best_score_raw) if best_score_raw is not None else None,
        "extra_meta": extra_meta or {},
    }
    torch.save(out, dst)
    return dst