"""
src/data_utils.py

Shared metadata + cross-validation helpers. Used by NB02 (split generation)
and by training/testing notebooks (load fold CSVs, validate them).

Public API:
    REQUIRED_METADATA_COLUMNS

    load_metadata(csv_path) -> DataFrame
    validate_metadata(df, ..., check_files_exist=False) -> None
    metadata_summary(df) -> DataFrame

    create_patient_folds(df, n_splits=5, ...) -> (splits, splitter_name)
    make_train_val_from_pool(train_pool_df, val_ratio=0.1111, ...) -> (train, val)
    verify_no_patient_leakage(train, val, test, ...) -> dict of counts

    create_image_level_folds(df, n_splits=5, ...) -> (splits, splitter_name)
    make_train_val_image_level(train_pool_df, val_ratio=0.15, ...) -> (train, val)

    save_fold_csvs(fold_dfs, splits_folds_dir) -> dict of saved paths

Seg-specific data infrastructure (BrainTumorDataset, transforms, build_dataloaders)
lives in src/sg_data_utils.py (M5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Metadata loading & validation
# --------------------------------------------------------------------------- #
REQUIRED_METADATA_COLUMNS: Tuple[str, ...] = (
    "image_id",
    "patient_id",
    "image_path",
    "mask_path",
    "tumor_class",
    "dataset",
)


def load_metadata(csv_path: PathLike) -> pd.DataFrame:
    """
    Load metadata.csv (or any fold CSV) with id columns coerced to str so
    leading-zero patient IDs aren't lost to int parsing.
    """
    return pd.read_csv(
        csv_path,
        dtype={
            "image_id":    str,
            "patient_id":  str,
            "tumor_class": str,
            "dataset":     str,
        },
    )


def validate_metadata(
    df: pd.DataFrame,
    required_columns: Optional[Iterable[str]] = None,
    project_root: Optional[PathLike] = None,
    check_files_exist: bool = False,
) -> None:
    """
    Hard-assert basic invariants on a metadata-shaped DataFrame:
      - Required columns present.
      - image_id is unique.
      - No nulls in core columns.
      - (Optional) every image_path / mask_path resolves under project_root.
    """
    cols = (
        tuple(required_columns)
        if required_columns is not None
        else REQUIRED_METADATA_COLUMNS
    )

    missing_cols = [c for c in cols if c not in df.columns]
    assert not missing_cols, f"metadata is missing required columns: {missing_cols}"

    assert df["image_id"].is_unique, "metadata image_id is not unique"
    for c in ("patient_id", "image_path", "mask_path", "tumor_class"):
        assert df[c].notna().all(), f"metadata column {c} has missing values"

    if check_files_exist:
        assert project_root is not None, (
            "project_root is required when check_files_exist=True"
        )
        root = Path(project_root)
        missing_img = [p for p in df["image_path"] if not (root / p).exists()]
        missing_msk = [p for p in df["mask_path"]  if not (root / p).exists()]
        assert not missing_img, (
            f"{len(missing_img)} image files missing, e.g. {missing_img[:3]}"
        )
        assert not missing_msk, (
            f"{len(missing_msk)} mask files missing, e.g. {missing_msk[:3]}"
        )


def metadata_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact long-form summary table: number of images, patients, per-class
    counts, plus mask-area-ratio mean/median if present in df.
    """
    rows = [
        {"metric": "num_images",   "value": int(len(df))},
        {"metric": "num_patients", "value": int(df["patient_id"].nunique())},
    ]
    for cls, count in df["tumor_class"].value_counts().items():
        rows.append({"metric": f"class_{cls}", "value": int(count)})
    if "mask_area_ratio" in df.columns:
        rows.append({"metric": "mask_area_ratio_mean",
                     "value": float(df["mask_area_ratio"].mean())})
        rows.append({"metric": "mask_area_ratio_median",
                     "value": float(df["mask_area_ratio"].median())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Patient-level fold splitting (group + stratify by patient)
# --------------------------------------------------------------------------- #
try:
    from sklearn.model_selection import StratifiedGroupKFold  # sklearn >= 1.0
    _HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    StratifiedGroupKFold = None  # type: ignore[assignment]
    _HAS_STRATIFIED_GROUP_KFOLD = False

from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)


def create_patient_folds(
    metadata_df: pd.DataFrame,
    n_splits: int = 5,
    group_col: str = "patient_id",
    stratify_col: str = "tumor_class",
    random_state: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    """
    Build K patient-disjoint (train_pool, test) splits.

    Strategy:
        - Multi-class datasets (>=2 unique `stratify_col` values) ->
          StratifiedGroupKFold (preserves class balance per fold while keeping
          patients disjoint).
        - Single-class datasets (e.g. brats2020 where every row is 'glioma')
          cannot be stratified — n_classes < n_splits violates the algorithm.
          Patients are pre-shuffled with `random_state` then split by plain
          GroupKFold (sklearn's GroupKFold is otherwise deterministic on
          input order).
        - If StratifiedGroupKFold raises unexpectedly on multi-class data,
          we fall back to the same shuffled-GroupKFold path.

    Returns
    -------
    (splits, method_name)
        splits: list of n_splits tuples (train_pool_indices, test_indices)
                indexing positions in `metadata_df`.
        method_name: "StratifiedGroupKFold" | "GroupKFold_singleclass"
                   | "GroupKFold_fallback"
    """
    groups = metadata_df[group_col].astype(str).to_numpy()
    y      = metadata_df[stratify_col].astype(str).to_numpy()
    n_classes = len(np.unique(y))

    # ---- Multi-class -> StratifiedGroupKFold ----
    if _HAS_STRATIFIED_GROUP_KFOLD and n_classes >= 2:
        try:
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state,
            )
            splits = list(sgkf.split(np.zeros(len(metadata_df)), y, groups))
            print(
                f"[create_patient_folds] StratifiedGroupKFold | "
                f"classes={n_classes}, patients={len(np.unique(groups))}, "
                f"n_splits={n_splits}, random_state={random_state}"
            )
            return splits, "StratifiedGroupKFold"
        except Exception as e:
            print(
                f"[create_patient_folds] StratifiedGroupKFold failed ({e}); "
                f"falling back to GroupKFold."
            )

    # ---- Single-class (or fallback) -> shuffled GroupKFold ----
    rng = np.random.default_rng(random_state)
    unique_patients = sorted(set(groups.tolist()))
    permutation = rng.permutation(len(unique_patients))
    new_order = {unique_patients[i]: rank for rank, i in enumerate(permutation)}

    permuted = metadata_df.assign(
        _p_order=metadata_df[group_col].astype(str).map(new_order)
    )
    sort_keys = ["_p_order"]
    if "image_id" in metadata_df.columns:
        sort_keys.append("image_id")
    permuted = permuted.sort_values(sort_keys).drop(columns=["_p_order"])

    permuted_to_original = permuted.index.to_numpy()
    perm_groups = permuted[group_col].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for tv_idx_p, te_idx_p in gkf.split(np.zeros(len(permuted)), groups=perm_groups):
        splits.append(
            (permuted_to_original[tv_idx_p], permuted_to_original[te_idx_p])
        )

    method = "GroupKFold_singleclass" if n_classes < 2 else "GroupKFold_fallback"
    print(
        f"[create_patient_folds] {method} | "
        f"classes={n_classes}, patients={len(unique_patients)}, "
        f"n_splits={n_splits}, random_state={random_state} "
        f"(patients pre-shuffled with random_state)"
    )
    return splits, method


def make_train_val_from_pool(
    train_pool_df: pd.DataFrame,
    val_ratio: float = 0.1111,
    group_col: str = "patient_id",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve a patient-disjoint validation set off a fold's train pool.
    val_ratio default 0.1111 ≈ 1/9 → with 4/5 of patients in the pool, this
    gives an overall ~80/9/11 train/val/test split.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    groups = train_pool_df[group_col].astype(str).values
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss.split(np.zeros(len(train_pool_df)), groups=groups))

    train_df = train_pool_df.iloc[train_idx].reset_index(drop=True)
    val_df   = train_pool_df.iloc[val_idx].reset_index(drop=True)
    return train_df, val_df


def verify_no_patient_leakage(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    group_col: str = "patient_id",
) -> Dict[str, int]:
    """
    Hard-assert no patient appears in more than one of train/val/test.
    Returns counts on success; raises AssertionError listing offenders on failure.
    """
    s_train = set(train_df[group_col].astype(str))
    s_val   = set(val_df  [group_col].astype(str))
    s_test  = set(test_df [group_col].astype(str))

    overlaps = {
        "train_val":  s_train & s_val,
        "train_test": s_train & s_test,
        "val_test":   s_val   & s_test,
    }
    bad = {k: sorted(v) for k, v in overlaps.items() if v}
    assert not bad, f"PATIENT LEAKAGE DETECTED: {bad}"

    return {
        "n_train_patients": len(s_train),
        "n_val_patients":   len(s_val),
        "n_test_patients":  len(s_test),
        "n_train_images":   len(train_df),
        "n_val_images":     len(val_df),
        "n_test_images":    len(test_df),
    }


# --------------------------------------------------------------------------- #
# Image-level fold splitting (the leaky reference setup)
# --------------------------------------------------------------------------- #
def create_image_level_folds(
    metadata_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], str]:
    """
    Plain KFold on rows (i.e. images) — same as the FigShare reference
    notebook's `KFold(shuffle=True, random_state=42)`. NOT patient-disjoint.

    Use only for reference comparison. For the project's final results, use
    create_patient_folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(kf.split(np.zeros(len(metadata_df))))
    return splits, "KFold (image-level, leaky — reference comparison only)"


def make_train_val_image_level(
    train_pool_df: pd.DataFrame,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Image-level (non-grouped) train/val split. Mirrors the reference
    notebook's `train_test_split(test_size=0.15, random_state=42)`.
    """
    train_df, val_df = train_test_split(
        train_pool_df,
        test_size=val_ratio,
        random_state=random_state,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_fold_csvs(
    fold_dfs: Dict[int, Dict[str, pd.DataFrame]],
    splits_folds_dir: PathLike,
) -> Dict[int, Dict[str, Path]]:
    """
    Write fold CSVs to disk in the canonical folds/ layout:
        <splits_folds_dir>/fold_<k>_train.csv
        <splits_folds_dir>/fold_<k>_val.csv
        <splits_folds_dir>/fold_<k>_test.csv

    Parameters
    ----------
    fold_dfs:
        {fold_num: {"train": train_df, "val": val_df, "test": test_df}, ...}
    splits_folds_dir:
        The folds/ directory (use file_utils.split_scheme_dir(...) / "folds").

    Returns
    -------
    {fold_num: {"train": path, "val": path, "test": path}, ...}
    """
    out_paths: Dict[int, Dict[str, Path]] = {}
    out_dir = Path(splits_folds_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fold_num, parts in fold_dfs.items():
        out_paths[fold_num] = {}
        for split_name, df in parts.items():
            csv_path = out_dir / f"fold_{fold_num}_{split_name}.csv"
            df.to_csv(csv_path, index=False)
            out_paths[fold_num][split_name] = csv_path

    return out_paths