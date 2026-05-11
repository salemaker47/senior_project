"""
src/eval_utils.py

Shared evaluation helpers — used by both NB02 and the task-specific
aggregators (sg_eval_utils.py for Phase 1, cls_eval_utils.py for Phase 2).

Public API:
    build_fold_summary    per-(fold, split) counts table; matches the
                          n_class_<cls> column convention NB02 already writes.

    enriched_aggregate    Enhancement B: cross-fold summary table with
                          mean, std, median, IQR, min/max, 95% CI per metric.
                          One row per metric (wide-formatted for printing).

Task-specific aggregators (per-image seg metrics, per-class confusion
matrices, per-patient roll-ups, ...) live in src/sg_eval_utils.py and
src/cls_eval_utils.py.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


# --------------------------------------------------------------------------- #
# Fold-split summary table (used by NB02)
# --------------------------------------------------------------------------- #
def build_fold_summary(
    fold_dfs: Dict[int, Dict[str, pd.DataFrame]],
    group_col: str = "patient_id",
    stratify_col: str = "tumor_class",
) -> pd.DataFrame:
    """
    Build a per-(fold, split) counts table:
        columns = fold, split, n_images, n_patients, n_class_<cls> (one per class)

    `fold_dfs`:
        {fold_num: {"train": train_df, "val": val_df, "test": test_df}, ...}

    Column names match the convention used by NB02's inline summary so
    `cv_fold_summary.csv` files are interchangeable whether NB02 computed
    them inline or via this helper.
    """
    all_classes = set()
    for parts in fold_dfs.values():
        for df in parts.values():
            all_classes.update(df[stratify_col].astype(str).unique().tolist())
    all_classes = sorted(all_classes)

    rows = []
    for fold_num in sorted(fold_dfs.keys()):
        for split_name in ("train", "val", "test"):
            df = fold_dfs[fold_num][split_name]
            n = len(df)
            row = {
                "fold":       fold_num,
                "split":      split_name,
                "n_images":   int(n),
                "n_patients": int(df[group_col].astype(str).nunique()),
            }
            counts = df[stratify_col].astype(str).value_counts().to_dict()
            for cls in all_classes:
                row[f"n_class_{cls}"] = int(counts.get(cls, 0))
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Enhancement B: enriched cross-fold aggregation
# --------------------------------------------------------------------------- #
def enriched_aggregate(
    df: pd.DataFrame,
    metric_columns: Sequence[str],
    ci_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Build a cross-fold summary table. For each column in `metric_columns`,
    computes:
        n, mean, std, median, q25, q75, min, max, ci95_lower, ci95_upper

    All variance/CI computation uses the SAMPLE std (ddof=1), the statistically
    appropriate estimator for finite-sample CIs. The 95% confidence interval
    of the mean uses the t-distribution with df = n - 1.

    Returns one row per metric (wide format — easy to print and to merge
    horizontally with single-value columns like `experiment_name`).

    Missing values are dropped per-column (a metric with NaNs in some folds
    is still summarized over its non-NaN folds, with n reflecting that).
    """
    rows = []
    for col in metric_columns:
        if col not in df.columns:
            continue
        values = df[col].dropna().to_numpy(dtype=float)
        n = int(len(values))
        if n == 0:
            continue

        mean = float(values.mean())
        std  = float(values.std(ddof=1)) if n > 1 else 0.0

        if n > 1:
            t_val = float(student_t.ppf(1.0 - ci_alpha / 2.0, df=n - 1))
            se = std / math.sqrt(n)
            ci_lo = mean - t_val * se
            ci_hi = mean + t_val * se
        else:
            ci_lo = ci_hi = mean

        q25, median, q75 = np.percentile(values, [25, 50, 75])

        rows.append({
            "metric":     col,
            "n":          n,
            "mean":       mean,
            "std":        std,
            "median":     float(median),
            "q25":        float(q25),
            "q75":        float(q75),
            "min":        float(values.min()),
            "max":        float(values.max()),
            "ci95_lower": float(ci_lo),
            "ci95_upper": float(ci_hi),
        })
    return pd.DataFrame(rows)