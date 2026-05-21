"""
src/cls_eval_utils.py

Classification-specific evaluation aggregators. Used by:
    - NB08 final aggregation cells (cross-fold tables)

Public API:
    aggregate_cv_results_cls(fold_summaries, experiment_name, dataset,
                             split_scheme, mask_source, seg_experiment_name,
                             repro_metadata)
        -> {"cv_results", "cv_summary", "cv_summary_enriched",
            "cv_confusion", "cv_by_class", "cv_per_image"}


Column convention
-----------------
cv_results columns (one row per fold):
    fold, macro_f1, accuracy, f1_meningioma, f1_glioma, f1_pituitary,
    precision_*, recall_*, support_*, n_images

cv_summary columns (one row):
    For each metric `m` in _HEADLINE_METRICS: `m_mean`, `m_std`
    Plus: experiment_name, dataset, split_scheme, mask_source,
          seg_experiment_name, n_folds, n_test_images_total,
          report_macro_f1 (formatted string),
          repro_metadata_json (if supplied)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.eval_utils import enriched_aggregate, add_mean_std
from src.cls_metrics import IDX_TO_CLASS

# All labels in canonical order
_CLASS_NAMES = [IDX_TO_CLASS[i] for i in sorted(IDX_TO_CLASS)]

_HEADLINE_METRICS: List[str] = [
    "macro_f1",
    "accuracy",
    "f1_meningioma",
    "f1_glioma",
    "f1_pituitary",
]


# --------------------------------------------------------------------------- #
# Cross-fold aggregation
# --------------------------------------------------------------------------- #
def aggregate_cv_results_cls(
    fold_summaries: Sequence[Dict[str, Any]],
    experiment_name: str,
    dataset: str = "figshare",
    split_scheme: str = "image_level",
    mask_source: str = "gt",
    seg_experiment_name: Optional[str] = None,
    repro_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Combine per-fold summaries (from cls_test_utils.evaluate_fold_cls) into
    cross-fold tables.

    Each element of `fold_summaries` is the dict returned by evaluate_fold_cls:
        {per_image_df, fold_metrics, confusion_matrix, manifest_path, manifest}

    Returns
    -------
    cv_results          1 row per fold — headline metrics + per-class metrics
    cv_summary          1 row — mean ± std across folds, report_macro_f1 string
    cv_summary_enriched 1 row per headline metric: mean/std/median/IQR/95%CI
    cv_confusion        labelled confusion matrix DataFrame (sum across folds)
    cv_by_class         1 row per (fold, class) — per-class F1/P/R/support
    cv_per_image        concat of all per_image_df DataFrames (~3,064 rows for figshare)
    """
    if not fold_summaries:
        raise ValueError("aggregate_cv_results_cls requires at least one fold summary")

    # ---- cv_per_image ----
    cv_per_image = pd.concat(
        [s["per_image_df"] for s in fold_summaries], ignore_index=True
    )

    # ---- cv_results (one row per fold) ----
    result_rows: List[Dict[str, Any]] = []
    by_class_rows: List[Dict[str, Any]] = []

    for s in fold_summaries:
        m = s["fold_metrics"]
        mfst = s["manifest"]
        fold = int(mfst["fold"])
        n_images = int(len(s["per_image_df"]))

        row: Dict[str, Any] = {
            "experiment": experiment_name,
            "fold":       fold,
            "mask_source": mask_source,
            "n_images":   n_images,
        }
        for k in _HEADLINE_METRICS:
            row[k] = float(m.get(k, 0.0))
        # Include all per-class precision/recall/support too
        for cls in _CLASS_NAMES:
            for stat in ("precision", "recall", "support"):
                key = f"{stat}_{cls}"
                row[key] = m.get(key, 0.0)
        result_rows.append(row)

        # per-class rows (for cv_by_class)
        for cls in _CLASS_NAMES:
            by_class_rows.append({
                "experiment":  experiment_name,
                "fold":        fold,
                "mask_source": mask_source,
                "tumor_class": cls,
                "n_images":    int(
                    s["per_image_df"]["tumor_class"]
                    .eq(cls).sum()
                ),
                "f1":          float(m.get(f"f1_{cls}", 0.0)),
                "precision":   float(m.get(f"precision_{cls}", 0.0)),
                "recall":      float(m.get(f"recall_{cls}", 0.0)),
                "support":     int(m.get(f"support_{cls}", 0)),
            })

    cv_results  = pd.DataFrame(result_rows).sort_values("fold").reset_index(drop=True)
    cv_by_class = pd.DataFrame(by_class_rows).sort_values(["fold", "tumor_class"]).reset_index(drop=True)

    # ---- cv_summary ----
    summary: Dict[str, Any] = {
        "experiment_name":      experiment_name,
        "dataset":              dataset,
        "split_scheme":         split_scheme,
        "mask_source":          mask_source,
        "seg_experiment_name":  seg_experiment_name,
        "n_folds":              int(cv_results["fold"].nunique()),
        "n_test_images_total":  int(cv_results["n_images"].sum()),
    }
    add_mean_std(summary, cv_results, _HEADLINE_METRICS)
    summary["report_macro_f1"] = (
        f"{summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}"
    )
    if repro_metadata is not None:
        summary["repro_metadata_json"] = json.dumps(repro_metadata, default=str)

    cv_summary = pd.DataFrame([summary])

    # ---- cv_summary_enriched ----
    cv_summary_enriched = enriched_aggregate(cv_results, metric_columns=_HEADLINE_METRICS)
    cv_summary_enriched.insert(0, "experiment_name", experiment_name)
    cv_summary_enriched.insert(1, "dataset",         dataset)
    cv_summary_enriched.insert(2, "split_scheme",    split_scheme)
    cv_summary_enriched.insert(3, "mask_source",     mask_source)

    # ---- cv_confusion (summed across all folds) ----
    cms = [s["confusion_matrix"] for s in fold_summaries]
    cm_sum = sum(cms)  # element-wise, all are np.ndarray (3,3)
    cv_confusion = aggregate_cv_confusion_from_matrix(cm_sum)

    return {
        "cv_results":          cv_results,
        "cv_summary":          cv_summary,
        "cv_summary_enriched": cv_summary_enriched,
        "cv_confusion":        cv_confusion,
        "cv_by_class":         cv_by_class,
        "cv_per_image":        cv_per_image,
    }


# --------------------------------------------------------------------------- #
# Confusion matrix helpers
# --------------------------------------------------------------------------- #
def aggregate_cv_confusion_from_matrix(cm: np.ndarray) -> pd.DataFrame:
    """
    Convert a raw (C, C) confusion matrix into a labelled DataFrame.
    rows = true class (index), cols = predicted class (columns).
    """
    return pd.DataFrame(
        cm,
        index   =[f"true_{c}"      for c in _CLASS_NAMES],
        columns =[f"pred_{c}"      for c in _CLASS_NAMES],
    )
