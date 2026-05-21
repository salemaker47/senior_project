"""
src/cls_metrics.py

Classification metrics — no torchmetrics dependency.
All functions work on numpy arrays (int predicted/true class indices).

Public API:
    PER_IMAGE_METRIC_NAMES  — column names for per-image CSV

    compute_per_image_metrics_cls(logits, labels, num_classes) -> Dict[str, np.ndarray]
        Per-batch: predicted class, correct flag, per-class softmax probabilities.

    macro_f1_from_preds(preds, labels, num_classes) -> float
        Unweighted average of per-class F1 (handles zero-division gracefully).

    accuracy_from_preds(preds, labels) -> float

    per_class_metrics(preds, labels, num_classes) -> Dict[str, Dict[str, float]]
        {"meningioma": {"precision", "recall", "f1", "support"}, ...}

    confusion_matrix_from_preds(preds, labels, num_classes) -> np.ndarray  # (C, C)
        rows = true class, cols = predicted class
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from src.cls_data_utils import IDX_TO_CLASS

PER_IMAGE_METRIC_NAMES: Tuple[str, ...] = (
    "predicted_class",   # int class index
    "true_class",        # int class index
    "correct",           # bool (1 if predicted == true)
    "prob_meningioma",
    "prob_glioma",
    "prob_pituitary",
)


# --------------------------------------------------------------------------- #
# Batch-level (called inside DataLoader loop)
# --------------------------------------------------------------------------- #
def compute_per_image_metrics_cls(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Compute per-image classification metrics for a batch.

    Parameters
    ----------
    logits : (N, num_classes) float tensor (raw model output, pre-softmax)
    labels : (N,) long tensor, ground-truth class indices

    Returns
    -------
    dict with keys = PER_IMAGE_METRIC_NAMES, each value a 1-D numpy array of length N
    """
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()      # (N, C)
    preds = probs.argmax(axis=1).astype(np.int64)                    # (N,)
    true  = labels.detach().cpu().numpy().astype(np.int64)           # (N,)

    result: Dict[str, np.ndarray] = {
        "predicted_class": preds,
        "true_class":      true,
        "correct":         (preds == true).astype(np.int64),
    }

    class_names = [IDX_TO_CLASS.get(i, str(i)) for i in range(num_classes)]
    for i, cls_name in enumerate(class_names):
        result[f"prob_{cls_name}"] = probs[:, i].astype(np.float64)

    return result


# --------------------------------------------------------------------------- #
# Aggregate metrics (called after collecting all per-image results)
# --------------------------------------------------------------------------- #
def confusion_matrix_from_preds(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 3,
) -> np.ndarray:
    """
    Confusion matrix C where C[i, j] = number of samples with true class i
    predicted as class j.

    Returns np.ndarray of shape (num_classes, num_classes), dtype int64.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
    np.add.at(cm, (labels[valid], preds[valid]), 1)
    return cm


def per_class_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Per-class precision, recall, F1, and support.

    Returns
    -------
    {class_name: {"precision": float, "recall": float, "f1": float, "support": int}}
    """
    cm = confusion_matrix_from_preds(preds, labels, num_classes)
    result: Dict[str, Dict[str, float]] = {}
    for i in range(num_classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        cls_name = IDX_TO_CLASS.get(i, str(i))
        result[cls_name] = {
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "support":   support,
        }
    return result


def macro_f1_from_preds(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 3,
) -> float:
    """Unweighted mean of per-class F1 scores (= sklearn's macro F1)."""
    pcm = per_class_metrics(preds, labels, num_classes)
    f1s = [v["f1"] for v in pcm.values()]
    return float(np.mean(f1s))


def accuracy_from_preds(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of correct predictions."""
    if len(labels) == 0:
        return 0.0
    return float((preds == labels).mean())
