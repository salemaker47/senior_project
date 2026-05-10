"""
src/file_utils.py

Path handling, JSON I/O, and prediction-manifest helpers for Senior_Project.

Three orthogonal axes define every artifact:
    - task:            "segmentation" or "classification"
    - dataset:         "figshare", "brats2024", ...
    - experiment_name: e.g. "01_dice_image_level", "cls01_resnet50"

Layout (mirrors §3 of the project instruction):
    <root>/
        data/<dataset>/
            raw/
            processed/
                images/<image_id>.png
                masks/<image_id>.png
                metadata.csv
                metadata_summary.csv
                preprocessing_config.json
            splits/<scheme>/
                cv_split_config.json
                cv_fold_summary.csv
                folds/fold_X_{train,val,test}.csv
        outputs/
            checkpoints/<task>/<dataset>/<exp>/fold_X/{best.ckpt, best_model.pt, experiment_config.json}
            logs/<task>/<dataset>/<exp>/fold_X/lightning_logs/version_0/metrics.csv
            figures/<task>/<dataset>/<exp>/fold_X/...
            figures/data_preparation/<dataset>/...
            tables/<task>/<dataset>/<exp>/...
            predictions/segmentation/<dataset>/<seg_exp>/fold_X/{<image_id>.png, manifest.json}
            predictions/segmentation/<dataset>/<seg_exp>/prediction_manifest.json
            reports/

Note:
    `cls_eval_paths` (the classification eval-variant helper) is deferred to
    Phase 2. Phase 1 covers segmentation only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

PathLike = Union[str, Path]

VALID_TASKS = ("segmentation", "classification")


# --------------------------------------------------------------------------- #
# Folder + JSON helpers
# --------------------------------------------------------------------------- #
def ensure_dir(path: PathLike) -> Path:
    """Create directory (and parents) if missing. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: PathLike, indent: int = 2) -> Path:
    """Save a Python object to JSON, creating parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=str)
    return p


def load_json(path: PathLike) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# File hashing (used by prediction manifests)
# --------------------------------------------------------------------------- #
def sha256_of_file(path: PathLike, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, chunked for large files."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _validate_task(task: str) -> None:
    if task not in VALID_TASKS:
        raise ValueError(
            f"task must be one of {VALID_TASKS}, got {task!r}"
        )


# --------------------------------------------------------------------------- #
# Top-level project layout
# --------------------------------------------------------------------------- #
def project_dirs(root: PathLike) -> Dict[str, Path]:
    """
    Top-level project directories under `root` (no per-dataset substructure).
    Creates everything it returns. Pass either a Drive root or a local-SSD
    root; the function is agnostic about which.
    """
    root = Path(root)
    dirs: Dict[str, Path] = {
        "project_root":         root,
        "data":                 root / "data",
        "outputs":              root / "outputs",
        "outputs_checkpoints":  root / "outputs" / "checkpoints",
        "outputs_logs":         root / "outputs" / "logs",
        "outputs_predictions":  root / "outputs" / "predictions",
        "outputs_figures":      root / "outputs" / "figures",
        "outputs_tables":       root / "outputs" / "tables",
        "outputs_reports":      root / "outputs" / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


# --------------------------------------------------------------------------- #
# Per-dataset paths
# --------------------------------------------------------------------------- #
def dataset_paths(root: PathLike, dataset: str) -> Dict[str, Path]:
    """
    Canonical paths under data/<dataset>/, plus the dataset's data-preparation
    figures dir under outputs/figures/data_preparation/<dataset>/.

    Creates folders. File paths (metadata.csv etc.) are returned but not
    created — those are produced by NB01.
    """
    root = Path(root)
    base = root / "data" / dataset

    paths: Dict[str, Path] = {
        "dataset_root":              base,
        "raw":                       base / "raw",
        "processed":                 base / "processed",
        "images":                    base / "processed" / "images",
        "masks":                     base / "processed" / "masks",
        "metadata_csv":              base / "processed" / "metadata.csv",
        "metadata_summary_csv":      base / "processed" / "metadata_summary.csv",
        "preprocessing_config_json": base / "processed" / "preprocessing_config.json",
        "splits":                    base / "splits",
        "figures_dataprep":          root / "outputs" / "figures" / "data_preparation" / dataset,
    }

    # Create directories only (not file paths).
    for key in (
        "dataset_root", "raw", "processed", "images", "masks",
        "splits", "figures_dataprep",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)

    return paths


# --------------------------------------------------------------------------- #
# Split-scheme paths
# --------------------------------------------------------------------------- #
def split_scheme_dir(root: PathLike, dataset: str, scheme: str) -> Path:
    """
    Returns data/<dataset>/splits/<scheme>/ and ensures the folds/ subdir exists.
    Also returned paths assume cv_split_config.json + cv_fold_summary.csv
    live directly inside this scheme dir.
    """
    root = Path(root)
    p = root / "data" / dataset / "splits" / scheme
    (p / "folds").mkdir(parents=True, exist_ok=True)
    return p


def fold_split_csv_paths(
    root: PathLike,
    dataset: str,
    scheme: str,
    fold: int,
) -> Dict[str, Path]:
    """
    train/val/test CSV paths for fold k under
    data/<dataset>/splits/<scheme>/folds/.
    """
    folds_dir = split_scheme_dir(root, dataset, scheme) / "folds"
    return {
        "train": folds_dir / f"fold_{fold}_train.csv",
        "val":   folds_dir / f"fold_{fold}_val.csv",
        "test":  folds_dir / f"fold_{fold}_test.csv",
    }


# --------------------------------------------------------------------------- #
# Per-experiment, per-fold paths
# --------------------------------------------------------------------------- #
def experiment_paths(
    root: PathLike,
    task: str,
    dataset: str,
    experiment_name: str,
    fold: int,
) -> Dict[str, Path]:
    """
    Per-(task, dataset, experiment, fold) output paths for checkpoints, logs,
    figures, tables. Predictions are intentionally NOT included here:
      - seg uses seg_predictions_dir() (manifest contract)
      - cls Eval B will use cls_eval_paths() in Phase 2 (extra eval_variant axis)
    """
    _validate_task(task)
    root = Path(root)
    suffix = Path(task) / dataset / experiment_name / f"fold_{fold}"

    ckpt_dir  = root / "outputs" / "checkpoints" / suffix
    log_dir   = root / "outputs" / "logs"        / suffix
    fig_dir   = root / "outputs" / "figures"     / suffix
    table_dir = root / "outputs" / "tables"      / suffix

    for d in (ckpt_dir, log_dir, fig_dir, table_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "checkpoints":            ckpt_dir,
        "best_ckpt":              ckpt_dir / "best.ckpt",
        "best_model":             ckpt_dir / "best_model.pt",
        "experiment_config_json": ckpt_dir / "experiment_config.json",
        "logs":                   log_dir,
        "metrics_csv":            log_dir / "lightning_logs" / "version_0" / "metrics.csv",
        "figures":                fig_dir,
        "tables":                 table_dir,
    }


def experiment_root_paths(
    root: PathLike,
    task: str,
    dataset: str,
    experiment_name: str,
) -> Dict[str, Path]:
    """
    Per-(task, dataset, experiment) paths *above* the fold level. Used by
    cross-fold aggregation: cv_results.csv, cv_summary.csv, etc.
    """
    _validate_task(task)
    root = Path(root)
    suffix = Path(task) / dataset / experiment_name

    out = {
        "checkpoints": root / "outputs" / "checkpoints" / suffix,
        "logs":        root / "outputs" / "logs"        / suffix,
        "figures":     root / "outputs" / "figures"     / suffix,
        "tables":      root / "outputs" / "tables"      / suffix,
    }
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Segmentation prediction tree + manifests
# --------------------------------------------------------------------------- #
def seg_predictions_dir(
    root: PathLike,
    dataset: str,
    seg_experiment_name: str,
    fold: Optional[int] = None,
) -> Path:
    """
    Path to a seg experiment's prediction tree.
        fold=None -> outputs/predictions/segmentation/<dataset>/<seg_exp>/
        fold=k    -> outputs/predictions/segmentation/<dataset>/<seg_exp>/fold_k/
    """
    root = Path(root)
    base = root / "outputs" / "predictions" / "segmentation" / dataset / seg_experiment_name
    out = base if fold is None else base / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_seg_prediction_manifest(
    root: PathLike,
    dataset: str,
    seg_experiment_name: str,
    fold: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Read a seg prediction manifest written by sg_test_utils.evaluate_fold.
        fold=None -> top-level prediction_manifest.json (experiment-wide)
        fold=k    -> per-fold manifest.json
    Raises FileNotFoundError if the manifest doesn't exist.
    """
    if fold is None:
        manifest_path = seg_predictions_dir(root, dataset, seg_experiment_name) \
                        / "prediction_manifest.json"
    else:
        manifest_path = seg_predictions_dir(root, dataset, seg_experiment_name, fold) \
                        / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No prediction manifest at {manifest_path}.\n"
            f"Run NB05 (segmentation/05_test) for {seg_experiment_name}"
            + (f" fold {fold}" if fold is not None else "")
            + " first."
        )
    return load_json(manifest_path)


def verify_seg_predictions_match(
    root: PathLike,
    dataset: str,
    seg_experiment_name: str,
    fold: int,
    checkpoint_path: PathLike,
    test_csv_path: PathLike,
) -> Dict[str, Any]:
    """
    Compare the per-fold manifest's recorded checkpoint+test-CSV SHA-256
    hashes against the files at the given paths.

    Used by cls Eval B (Phase 2) before reading any predicted masks, and by
    sg_test_utils itself as a self-check after writing.

    Manifests synthesized by NB00 for legacy data carry `legacy=true` and
    skip hash verification — the manifest is still returned but the caller
    should treat the predictions as unverified.

    Raises ValueError on hash mismatch. Returns the manifest dict on success.
    """
    manifest = load_seg_prediction_manifest(
        root, dataset, seg_experiment_name, fold
    )

    if manifest.get("legacy", False):
        return manifest

    actual_ckpt_hash = sha256_of_file(checkpoint_path)
    actual_csv_hash  = sha256_of_file(test_csv_path)
    expected_ckpt_hash = manifest.get("checkpoint_sha256")
    expected_csv_hash  = manifest.get("test_csv_sha256")

    mismatches = []
    if expected_ckpt_hash != actual_ckpt_hash:
        mismatches.append(
            f"checkpoint hash: manifest={expected_ckpt_hash} "
            f"vs actual={actual_ckpt_hash} for {checkpoint_path}"
        )
    if expected_csv_hash != actual_csv_hash:
        mismatches.append(
            f"test_csv hash: manifest={expected_csv_hash} "
            f"vs actual={actual_csv_hash} for {test_csv_path}"
        )

    if mismatches:
        raise ValueError(
            f"Seg prediction manifest for "
            f"{seg_experiment_name} fold {fold} is stale. "
            f"Re-run NB05 to regenerate predictions. Mismatches:\n  - "
            + "\n  - ".join(mismatches)
        )

    return manifest