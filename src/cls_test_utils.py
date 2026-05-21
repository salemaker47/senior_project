"""
src/cls_test_utils.py

Classification inference + per-fold evaluation:

    load_cls_model_from_pt   reload a saved best_model.pt
    evaluate_fold_cls        batched test-set evaluation:
                               * runs inference on test_loader
                               * computes per-image metrics (predicted class, correct,
                                 per-class softmax probabilities)
                               * computes fold-level macro F1, accuracy, per-class P/R/F1,
                                 confusion matrix
                               * writes a per-fold manifest.json for traceability

Both mask sources are supported:
    mask_source="gt"        Eval A — GT masks from metadata CSV
    mask_source="predicted" Eval B — predicted masks from seg experiment predictions

The notebook (NB08) calls evaluate_fold_cls once per fold per eval variant, then passes
all fold summaries to cls_eval_utils.aggregate_cv_results_cls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from src.cls_data_utils import build_test_loader_cls, IDX_TO_CLASS
from src.cls_metrics import (
    compute_per_image_metrics_cls,
    macro_f1_from_preds,
    accuracy_from_preds,
    per_class_metrics,
    confusion_matrix_from_preds,
    PER_IMAGE_METRIC_NAMES,
)
from src.cls_models import build_cls_model
from src.train_utils import strip_model_prefix
from src.file_utils import sha256_of_file, save_json, PathLike


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def load_cls_model_from_pt(
    pt_path: PathLike,
    model_name: str,
    num_classes: int = 3,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Load a plain PyTorch checkpoint saved by train_utils.export_plain_state_dict.

    Parameters
    ----------
    pt_path    : path to best_model.pt
    model_name : model registry key (must match what was used during training)
    num_classes: number of output classes (default 3)
    device     : 'cuda' or 'cpu'

    Returns
    -------
    nn.Module in eval mode on `device`
    """
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Support both the full blob format (with state_dict key) and a raw state dict.
    state_dict = blob.get("state_dict", blob)

    # The Lightning checkpoint wraps model weights under "model." prefix.
    state_dict = strip_model_prefix(state_dict)

    model = build_cls_model(name=model_name, num_classes=num_classes, pretrained=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[load_cls_model_from_pt] missing={len(missing)} unexpected={len(unexpected)}"
        )

    model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# Batched fold evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_fold_cls(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    project_root: PathLike,
    eval_dir: PathLike,
    fold: int,
    experiment_name: str,
    dataset: str,
    split_scheme: str,
    checkpoint_path: PathLike,
    test_csv_path: PathLike,
    model_name: str,
    mask_source: str = "gt",
    seg_experiment_name: Optional[str] = None,
    predictions_dir: Optional[PathLike] = None,
    image_size: int = 224,
    padding_frac: float = 0.10,
    batch_size: int = 32,
    num_workers: int = 2,
    num_classes: int = 3,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run inference on the entire fold's test set and compute classification metrics.

    Parameters
    ----------
    model             : model in eval mode (from load_cls_model_from_pt)
    test_df           : fold test-set DataFrame (from fold_split_csv_paths)
    project_root      : project root (LOCAL_ROOT in notebooks)
    eval_dir          : where to write fold manifest; from cls_eval_paths()
    fold              : fold number (1–5)
    experiment_name   : classification experiment name, e.g. "cls01_resnet50"
    dataset           : e.g. "figshare"
    split_scheme      : e.g. "image_level"
    checkpoint_path   : path to best_model.pt (for manifest hash)
    test_csv_path     : path to fold test CSV (for manifest hash)
    model_name        : timm model name used during training
    mask_source       : "gt" (Eval A) or "predicted" (Eval B)
    seg_experiment_name: seg experiment whose predictions to use (Eval B only)
    predictions_dir   : per-fold seg predictions dir (Eval B only)
    image_size        : patch size (default 224)
    padding_frac      : fraction of bbox side added as padding (default 0.10)
    batch_size        : inference batch size
    num_workers       : DataLoader workers
    device            : "cuda" or "cpu"

    Returns
    -------
    {
        "per_image_df":    DataFrame (one row per test image)
        "fold_metrics":    {"macro_f1", "accuracy", per-class metrics}
        "confusion_matrix": np.ndarray (num_classes, num_classes)
        "manifest_path":   Path to written manifest.json
        "manifest":        manifest dict
    }
    """
    if mask_source == "predicted" and predictions_dir is None:
        raise ValueError(
            "predictions_dir is required for mask_source='predicted'. "
            "Call seg_predictions_dir() to get the fold-level dir."
        )

    project_root = Path(project_root)
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_df = test_df.reset_index(drop=True)

    loader = build_test_loader_cls(
        test_df, project_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        padding_frac=padding_frac,
        mask_source=mask_source,
        predictions_dir=predictions_dir,
        return_meta=True,
    )

    rows: List[Dict[str, Any]] = []

    for patches, labels, metas in loader:
        patches = patches.to(device, non_blocking=True)
        logits  = model(patches)

        m = compute_per_image_metrics_cls(logits, labels, num_classes=num_classes)

        n_b = patches.size(0)
        for i in range(n_b):
            record: Dict[str, Any] = {
                "image_id":    metas["image_id"][i],
                "patient_id":  metas["patient_id"][i],
                "tumor_class": metas["tumor_class"][i],
                "mask_source": mask_source,
                "dataset":     dataset,
                "fold":        int(fold),
                "experiment":  experiment_name,
            }
            for k in PER_IMAGE_METRIC_NAMES:
                record[k] = int(m[k][i]) if k in ("predicted_class", "true_class", "correct") \
                             else float(m[k][i])
            # Add human-readable predicted class name for convenience
            record["predicted_class_name"] = IDX_TO_CLASS.get(
                int(m["predicted_class"][i]), str(int(m["predicted_class"][i]))
            )
            rows.append(record)

    per_image_df = pd.DataFrame(rows)

    # ---- Fold-level aggregated metrics ----
    all_preds  = per_image_df["predicted_class"].to_numpy(dtype=np.int64)
    all_true   = per_image_df["true_class"].to_numpy(dtype=np.int64)

    macro_f1 = macro_f1_from_preds(all_preds, all_true, num_classes=num_classes)
    accuracy  = accuracy_from_preds(all_preds, all_true)
    pcm       = per_class_metrics(all_preds, all_true, num_classes=num_classes)
    cm        = confusion_matrix_from_preds(all_preds, all_true, num_classes=num_classes)

    fold_metrics: Dict[str, Any] = {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }
    for cls_name, cls_m in pcm.items():
        fold_metrics[f"f1_{cls_name}"]        = cls_m["f1"]
        fold_metrics[f"precision_{cls_name}"] = cls_m["precision"]
        fold_metrics[f"recall_{cls_name}"]    = cls_m["recall"]
        fold_metrics[f"support_{cls_name}"]   = cls_m["support"]

    # ---- Per-fold manifest ----
    manifest = {
        "cls_experiment_name":  experiment_name,
        "task":                 "classification",
        "dataset":              dataset,
        "split_scheme":         split_scheme,
        "fold":                 int(fold),
        "eval_mask_source":     mask_source,
        "seg_experiment_name":  seg_experiment_name,
        "checkpoint_path":      str(Path(checkpoint_path)),
        "checkpoint_sha256":    sha256_of_file(checkpoint_path),
        "test_csv_path":        str(Path(test_csv_path)),
        "test_csv_sha256":      sha256_of_file(test_csv_path),
        "n_predictions":        int(len(per_image_df)),
        "image_size":           int(image_size),
        "model_name":           model_name,
        "macro_f1":             float(macro_f1),
        "accuracy":             float(accuracy),
        "generated_at":         datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = eval_dir / f"fold_{fold}_manifest.json"
    save_json(manifest, manifest_path)

    return {
        "per_image_df":     per_image_df,
        "fold_metrics":     fold_metrics,
        "confusion_matrix": cm,
        "manifest_path":    manifest_path,
        "manifest":         manifest,
    }
