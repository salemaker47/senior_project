"""
src/sg_eval_utils.py

Segmentation-specific evaluation aggregators. Used by:
    - sg_test_utils.evaluate_fold (per-fold summary tables)
    - NB05 final cells           (cross-fold aggregation -> cv_*.csv)

Public API:
    # Pooled-stats helpers
    micro_dice_from_counts(tp, fp, fn)
    micro_iou_from_counts(tp, fp, fn)

    # Single-fold summary
    summarize_fold_results(per_image_df, fold, experiment_name, micro_counts)
        -> {"fold_overall", "fold_by_class", "per_image"}

    # Cross-fold aggregation
    aggregate_cv_results(fold_summaries, experiment_name, dataset, split_scheme,
                        repro_metadata=None)
        -> {"cv_results", "cv_summary", "cv_summary_enriched",
            "cv_by_class", "cv_class_summary", "cv_per_image"}

    # Enhancement C — per-patient
    aggregate_cv_per_patient(cv_per_image_df) -> DataFrame

    # Enhancement E — training-time summary
    aggregate_cv_training_summary(per_fold_training_meta) -> DataFrame

Column convention
-----------------
fold_overall columns (one value per fold):
    dice_micro, iou_micro, sensitivity_micro, precision_micro
    total_tp, total_fp, total_fn, total_tn, total_pixels
    n_images

cv_summary columns (one row, across folds):
    For each metric `m` above:  `m_mean`, `m_std`   (cross-fold)
    Plus: experiment_name, dataset, split_scheme, n_folds, n_test_images_total,
          report_dice_micro, report_iou_micro,
          repro_metadata_json  (if supplied).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.eval_utils import enriched_aggregate


# --------------------------------------------------------------------------- #
# Pooled-stats helpers
# --------------------------------------------------------------------------- #
def micro_dice_from_counts(tp: int, fp: int, fn: int, smooth: float = 1.0) -> float:
    """Pooled Dice (equivalent to F1 with reduction='micro')."""
    return float((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth))


def micro_iou_from_counts(tp: int, fp: int, fn: int, smooth: float = 1.0) -> float:
    """Pooled IoU (Jaccard) from globally pooled tp/fp/fn."""
    return float((tp + smooth) / (tp + fp + fn + smooth))


# --------------------------------------------------------------------------- #
# Internal: row builders
# --------------------------------------------------------------------------- #
_PER_IMAGE_METRIC_KEYS = (
    "dice", "iou", "sensitivity", "precision",
    "pred_mask_area_ratio", "gt_mask_area_ratio", "area_ratio_delta",
)


def _fold_overall_row(
    sub: pd.DataFrame,
    micro_counts: Dict[str, int],
    n_total_pixels: Optional[int] = None,
) -> Dict[str, float]:
    """
    Build one fold_overall row (or one fold_by_class row) from a per-image
    DataFrame `sub`. `micro_counts` provides the pooled (tp, fp, fn, tn) for
    *this group of images* — for fold_overall it's the whole fold; for
    fold_by_class it's just the per-class subset.
    """
    tp = int(micro_counts["tp"])
    fp = int(micro_counts["fp"])
    fn = int(micro_counts["fn"])
    tn = int(micro_counts["tn"])

    n = int(len(sub))
    if n_total_pixels is None:
        n_total_pixels = int(sub["total_pixels"].sum()) if "total_pixels" in sub.columns else 0

    micro_sens = (tp + 1.0) / (tp + fn + 1.0)
    micro_prec = (tp + 1.0) / (tp + fp + 1.0)

    return {
        "n_images":           n,
        "dice_micro":         micro_dice_from_counts(tp, fp, fn),
        "iou_micro":          micro_iou_from_counts(tp, fp, fn),
        "sensitivity_micro":  float(micro_sens),
        "precision_micro":    float(micro_prec),
        "total_tp":           tp,
        "total_fp":           fp,
        "total_fn":           fn,
        "total_tn":           tn,
        "total_pixels":       int(n_total_pixels),
    }


# --------------------------------------------------------------------------- #
# Single-fold summary
# --------------------------------------------------------------------------- #
def summarize_fold_results(
    per_image_df: pd.DataFrame,
    fold: int,
    experiment_name: str,
    micro_counts: Dict[str, int],
) -> Dict[str, pd.DataFrame]:
    """
    Build per-fold summary tables from a fold's per-image metric DataFrame.

    `per_image_df` columns (produced by sg_test_utils.evaluate_fold):
        image_id, patient_id, tumor_class, dataset,
        dice, iou, sensitivity, precision,
        pred_mask_area_ratio, gt_mask_area_ratio, area_ratio_delta,
        true_positive_pixels, false_positive_pixels,
        false_negative_pixels, total_pixels.

    `micro_counts` = {"tp", "fp", "fn", "tn"} summed across the whole fold.

    Returns dict with three DataFrames:
        "fold_overall"  -- 1 row, fold-level macro + micro metrics
        "fold_by_class" -- 1 row per tumor_class
        "per_image"     -- per_image_df with experiment/fold prepended
    """
    # Fold-level row
    fold_row = {"experiment": experiment_name, "fold": int(fold)}
    fold_row.update(_fold_overall_row(per_image_df, micro_counts))
    fold_overall = pd.DataFrame([fold_row])

    # Per-class rows
    by_class_rows: List[Dict[str, Any]] = []
    for cls, sub in per_image_df.groupby("tumor_class"):
        class_counts = {
            "tp": int(sub["true_positive_pixels"].sum()),
            "fp": int(sub["false_positive_pixels"].sum()),
            "fn": int(sub["false_negative_pixels"].sum()),
            "tn": int(
                sub["total_pixels"].sum()
                - sub["true_positive_pixels"].sum()
                - sub["false_positive_pixels"].sum()
                - sub["false_negative_pixels"].sum()
            ),
        }
        row = {"experiment": experiment_name, "fold": int(fold), "tumor_class": cls}
        row.update(_fold_overall_row(sub, class_counts))
        by_class_rows.append(row)
    fold_by_class = pd.DataFrame(by_class_rows).sort_values("tumor_class").reset_index(drop=True)

    # Per-image (passthrough with experiment+fold tags)
    per_image_out = per_image_df.copy()
    if "fold" not in per_image_out.columns:
        per_image_out.insert(0, "fold", int(fold))
    if "experiment" not in per_image_out.columns:
        per_image_out.insert(0, "experiment", experiment_name)

    return {
        "fold_overall":  fold_overall,
        "fold_by_class": fold_by_class,
        "per_image":     per_image_out,
    }


# --------------------------------------------------------------------------- #
# Cross-fold aggregation (with Enhancement B)
# --------------------------------------------------------------------------- #
_HEADLINE_METRICS: List[str] = [
    "dice_micro",
    "iou_micro",
    "sensitivity_micro",
    "precision_micro",
]


def aggregate_cv_results(
    fold_summaries: Sequence[Dict[str, pd.DataFrame]],
    experiment_name: str,
    dataset: str = "figshare",
    split_scheme: str = "image_level",
    repro_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Combine 5 per-fold summaries (from summarize_fold_results) into the
    cross-fold tables expected under outputs/tables/segmentation/<dataset>/<exp>/:

    Returns:
        cv_results          -- 1 row per fold (concat of fold_overall)
        cv_summary          -- 1 row, headline mean ± std per metric, plus
                               report_* pretty strings, repro_metadata_json
        cv_summary_enriched -- 1 row per headline metric: mean/std/median/IQR/CI
        cv_by_class         -- 1 row per (fold, class) (concat of fold_by_class)
        cv_class_summary    -- 1 row per tumor_class, mean ± std across folds
        cv_per_image        -- concat of all per_image tables (~3,064 rows)
    """
    if not fold_summaries:
        raise ValueError("aggregate_cv_results requires at least one fold summary")

    cv_results  = pd.concat([s["fold_overall"]  for s in fold_summaries], ignore_index=True)
    cv_by_class = pd.concat([s["fold_by_class"] for s in fold_summaries], ignore_index=True)
    cv_per_image = pd.concat([s["per_image"]    for s in fold_summaries], ignore_index=True)

    # ---- cv_summary (one row, headline mean ± std across folds) ----
    summary: Dict[str, Any] = {
        "experiment_name":      experiment_name,
        "dataset":              dataset,
        "split_scheme":         split_scheme,
        "n_folds":              int(cv_results["fold"].nunique()),
        "n_test_images_total":  int(cv_results["n_images"].sum()),
    }
    for m in _HEADLINE_METRICS:
        vals = cv_results[m].astype(float)
        summary[f"{m}_mean"] = float(vals.mean())
        summary[f"{m}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    # Pretty report strings for the two headline numbers people quote
    for m in ("dice_micro", "iou_micro"):
        summary[f"report_{m}"] = (
            f"{summary[f'{m}_mean']:.4f} ± {summary[f'{m}_std']:.4f}"
        )

    # Enhancement F: stamp the repro metadata (JSON-serialised so it lands as
    # a single CSV cell — fine for grep / report copy/paste)
    if repro_metadata is not None:
        summary["repro_metadata_json"] = json.dumps(repro_metadata, default=str)

    cv_summary = pd.DataFrame([summary])

    # ---- cv_summary_enriched (Enhancement B: long format) ----
    cv_summary_enriched = enriched_aggregate(cv_results, metric_columns=_HEADLINE_METRICS)
    cv_summary_enriched.insert(0, "experiment_name", experiment_name)
    cv_summary_enriched.insert(1, "dataset",         dataset)
    cv_summary_enriched.insert(2, "split_scheme",    split_scheme)

    # ---- cv_class_summary (per-class across folds) ----
    class_rows: List[Dict[str, Any]] = []
    for cls, sub in cv_by_class.groupby("tumor_class"):
        row: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "dataset":         dataset,
            "tumor_class":     cls,
            "n_folds":         int(sub["fold"].nunique()),
            "n_test_images_total": int(sub["n_images"].sum()),
        }
        for m in _HEADLINE_METRICS:
            vals = sub[m].astype(float)
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        # The headline report string people will paste into the report:
        row["report_dice_micro"] = (
            f"{row['dice_micro_mean']:.4f} ± {row['dice_micro_std']:.4f}"
        )
        class_rows.append(row)
    cv_class_summary = (
        pd.DataFrame(class_rows).sort_values("tumor_class").reset_index(drop=True)
    )

    return {
        "cv_results":          cv_results,
        "cv_summary":          cv_summary,
        "cv_summary_enriched": cv_summary_enriched,
        "cv_by_class":         cv_by_class,
        "cv_class_summary":    cv_class_summary,
        "cv_per_image":        cv_per_image,
    }


# --------------------------------------------------------------------------- #
# Enhancement C — per-patient aggregation
# --------------------------------------------------------------------------- #
def aggregate_cv_per_patient(cv_per_image_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per patient. Aggregates across all of a patient's test-set slices
    (which under `patient_level` scheme appear in exactly one fold; under
    `image_level` scheme can appear across multiple folds).

    Columns:
        patient_id, n_slices, n_folds_appearing_in, dominant_tumor_class,
        dice_mean, dice_median, dice_min, dice_max,
        iou_mean, iou_median,
        sensitivity_mean, precision_mean,
        gt_area_ratio_mean, pred_area_ratio_mean, area_delta_mean
    """
    if cv_per_image_df.empty:
        return pd.DataFrame()

    rows = []
    grouped = cv_per_image_df.groupby("patient_id", sort=True)
    for pid, sub in grouped:
        # Dominant class = most-common tumor_class for this patient
        dom_class = (
            sub["tumor_class"].mode().iloc[0]
            if not sub["tumor_class"].mode().empty else ""
        )
        rows.append({
            "patient_id":              pid,
            "n_slices":                int(len(sub)),
            "n_folds_appearing_in":    int(sub["fold"].nunique()),
            "dominant_tumor_class":    dom_class,

            "dice_mean":     float(sub["dice"].mean()),
            "dice_median":   float(sub["dice"].median()),
            "dice_min":      float(sub["dice"].min()),
            "dice_max":      float(sub["dice"].max()),

            "iou_mean":      float(sub["iou"].mean()),
            "iou_median":    float(sub["iou"].median()),

            "sensitivity_mean":    float(sub["sensitivity"].mean()),
            "precision_mean":      float(sub["precision"].mean()),

            "gt_area_ratio_mean":   float(sub["gt_mask_area_ratio"].mean()),
            "pred_area_ratio_mean": float(sub["pred_mask_area_ratio"].mean()),
            "area_delta_mean":      float(sub["area_ratio_delta"].mean()),
        })
    return pd.DataFrame(rows).sort_values("patient_id").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Enhancement E — training-time summary across folds
# --------------------------------------------------------------------------- #
_TRAINING_META_KEYS = (
    "fold", "best_epoch", "best_val_dice",
    "total_epochs_trained", "train_seconds", "train_minutes",
    "params_count", "peak_gpu_mem_mb",
)


def aggregate_cv_training_summary(
    per_fold_training_meta: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    """
    One row per fold. Each input dict typically comes from NB03's training
    cell (extracted from the trainer / TrainingTimingCallback / ModelCheckpoint).

    Expected dict keys (any missing key -> NaN):
        fold, best_epoch, best_val_dice, total_epochs_trained,
        train_seconds, params_count, peak_gpu_mem_mb

    Adds a derived `train_minutes` column for convenience.
    """
    if not per_fold_training_meta:
        return pd.DataFrame(columns=list(_TRAINING_META_KEYS))

    rows = []
    for meta in per_fold_training_meta:
        train_s = meta.get("train_seconds")
        rows.append({
            "fold":                 meta.get("fold"),
            "best_epoch":           meta.get("best_epoch"),
            "best_val_dice":        meta.get("best_val_dice"),
            "total_epochs_trained": meta.get("total_epochs_trained"),
            "train_seconds":        train_s,
            "train_minutes":        (train_s / 60.0) if train_s is not None else None,
            "params_count":         meta.get("params_count"),
            "peak_gpu_mem_mb":      meta.get("peak_gpu_mem_mb"),
        })

    df = pd.DataFrame(rows)
    if "fold" in df.columns:
        df = df.sort_values("fold").reset_index(drop=True)
    return df