"""
src/sg_test_utils.py

Segmentation inference + per-fold evaluation:

    load_model_from_pt    reload a saved best_model.pt
    load_model_from_ckpt  fallback: reload from a Lightning .ckpt
    predict_mask          single-image inference (used by NB04 figures)
    evaluate_fold         batched test-set evaluation:
                            * runs inference on test_loader
                            * saves predicted PNGs to predictions_dir
                            * computes the full per-image metric suite (Enhancement A)
                            * writes the per-fold manifest.json per §8
                            * self-validates manifest hashes
    write_experiment_manifest  finalizer that walks all folds and writes the
                               experiment-level prediction_manifest.json per §8
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch

from src.sg_data_utils import build_eval_transform, build_test_loader
from src.sg_metrics    import (
    binarize_logits,
    compute_per_image_metrics_from_logits,
    PER_IMAGE_METRIC_NAMES,
)
from src.sg_models     import build_model
from src.train_utils   import strip_model_prefix

from src.file_utils import (
    sha256_of_file,
    save_json,
    seg_predictions_dir,
    load_seg_prediction_manifest,
    verify_seg_predictions_match,
    PathLike,
)


# --------------------------------------------------------------------------- #
# Checkpoint loading
# --------------------------------------------------------------------------- #
def load_model_from_pt(
    pt_path: PathLike,
    model_name: Optional[str] = None,
    encoder_weights: Optional[str] = None,
    device: str = "cuda",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load a plain PyTorch checkpoint saved by train_utils.export_plain_state_dict.
    `model_name` is inferred from extra_meta when not provided.

    Returns (model_in_eval_mode, extra_meta_dict).
    """
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    state_dict = blob["state_dict"]
    extra = blob.get("extra_meta", {}) or {}

    if model_name is None:
        model_name = (
            extra.get("model_name")
            or (extra.get("experiment") or {}).get("model_name")
        )
        if model_name is None:
            raise ValueError(
                "Could not infer model_name from checkpoint; pass it explicitly."
            )

    # Avoid re-downloading ImageNet weights at test time — the trained state_dict
    # overrides anything we'd download anyway.
    model = build_model(
        name=model_name,
        encoder_weights=encoder_weights,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[load_model_from_pt] missing={len(missing)} unexpected={len(unexpected)}"
        )

    model.to(device).eval()
    return model, extra


def load_model_from_ckpt(
    ckpt_path: PathLike,
    model_name: str,
    encoder_weights: Optional[str] = None,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Fallback: load directly from a Lightning .ckpt. Strips the 'model.' prefix
    that the LightningModule wrap adds.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = strip_model_prefix(blob.get("state_dict", blob))

    model = build_model(
        name=model_name,
        encoder_weights=encoder_weights,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"[load_model_from_ckpt] missing={len(missing)} unexpected={len(unexpected)}"
        )

    model.to(device).eval()
    return model


# --------------------------------------------------------------------------- #
# Single-image prediction (used by NB04)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image_path: PathLike,
    transform,
    device: str = "cuda",
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (probability_map, binary_mask) both as 2-D numpy arrays at the
    model's working resolution.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    augmented = transform(image=img_rgb, mask=np.zeros_like(img))
    x = augmented["image"].unsqueeze(0).to(device)

    logits = model(x)
    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
    binary = (probs > threshold).astype(np.uint8)
    return probs, binary


# --------------------------------------------------------------------------- #
# Batched evaluation of a single fold (with manifests + Enhancement A)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_fold(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    project_root: PathLike,
    predictions_dir: PathLike,
    fold: int,
    experiment_name: str,
    dataset: str,
    split_scheme: str,
    checkpoint_path: PathLike,
    test_csv_path: PathLike,
    model_name: str,
    encoder_weights: Optional[str] = "imagenet",
    image_size: int = 256,
    preprocessing: str = "original",
    batch_size: int = 32,
    num_workers: int = 2,
    device: str = "cuda",
    threshold: float = 0.5,
    save_pngs: bool = True,
) -> Dict[str, Any]:
    """
    Run inference on the entire fold's test set, save predicted PNGs,
    compute the Enhancement A per-image metric suite, and write the per-fold
    manifest.json per §8.

    Returned dict (consumed by NB05's per-fold and cross-fold cells):
        per_image_df:   DataFrame with columns
                        ['image_id','patient_id','tumor_class','dataset',
                         'image_path','mask_path','pred_path',
                         <all PER_IMAGE_METRIC_NAMES>]
        micro_counts:   {"tp", "fp", "fn", "tn"} totals across the fold
        manifest_path:  Path to the written manifest.json
        manifest:       the manifest dict (also written to disk)
    """
    project_root = Path(project_root)
    predictions_dir = Path(predictions_dir)
    if save_pngs:
        predictions_dir.mkdir(parents=True, exist_ok=True)

    test_df = test_df.reset_index(drop=True)

    loader = build_test_loader(
        test_df, project_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        preprocessing=preprocessing,
        return_meta=True,
    )

    # Build image_id lookup so row access is robust to DataLoader ordering.
    _id_to_row = {str(r["image_id"]): r for _, r in test_df.iterrows()}

    rows: List[Dict[str, Any]] = []
    micro = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for x, y, metas in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        # Enhancement A: full per-image metric suite
        m = compute_per_image_metrics_from_logits(logits, y, threshold=threshold)

        # TN per image — not in the per-image suite (it's huge for sparse masks),
        # but we still need it for accumulating fold-level micro_counts["tn"].
        # Compute on the binary tensors.
        pred_b = binarize_logits(logits, threshold=threshold)
        pb = (pred_b > 0.5).bool()
        yb = (y      > 0.5).bool()
        tn_b = (~pb & ~yb).sum(dim=(1, 2, 3)).cpu().numpy()

        # Save predicted PNGs + assemble rows
        pred_np = pred_b.cpu().numpy().astype(np.uint8)[:, 0]
        n_b = x.size(0)

        for i in range(n_b):
            img_id = str(metas["image_id"][i])
            row = _id_to_row[img_id]

            pred_path_rel = ""
            if save_pngs:
                out_png = predictions_dir / f"{img_id}.png"
                cv2.imwrite(str(out_png), pred_np[i] * 255)
                try:
                    pred_path_rel = str(out_png.relative_to(project_root))
                except ValueError:
                    pred_path_rel = str(out_png)

            record = {
                "image_id":    img_id,
                "patient_id":  str(row.get("patient_id", "")),
                "tumor_class": str(row.get("tumor_class", "")),
                "dataset":     dataset,
                "image_path":  row.get("image_path", ""),
                "mask_path":   row.get("mask_path", ""),
                "pred_path":   pred_path_rel,
            }
            # All Enhancement-A metric columns
            for k in PER_IMAGE_METRIC_NAMES:
                v = m[k][i]
                record[k] = int(v) if k.endswith("_pixels") else float(v)
            rows.append(record)

        # Accumulate fold-level micro counts
        micro["tp"] += int(m["true_positive_pixels"].sum())
        micro["fp"] += int(m["false_positive_pixels"].sum())
        micro["fn"] += int(m["false_negative_pixels"].sum())
        micro["tn"] += int(tn_b.sum())

    per_image_df = pd.DataFrame(rows)

    # ---- §8 per-fold manifest ----
    manifest = {
        "seg_experiment_name": experiment_name,
        "task":                "segmentation",
        "dataset":             dataset,
        "split_scheme":        split_scheme,
        "fold":                int(fold),
        "checkpoint_path":     str(Path(checkpoint_path)),
        "checkpoint_sha256":   sha256_of_file(checkpoint_path),
        "test_csv_path":       str(Path(test_csv_path)),
        "test_csv_sha256":     sha256_of_file(test_csv_path),
        "n_predictions":       int(len(per_image_df)),
        "image_size":          int(image_size),
        "threshold":           float(threshold),
        "mask_format":         "binary_uint8_0_255",
        "model_name":          model_name,
        "encoder_weights":     encoder_weights,
        "generated_at":        datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = predictions_dir / "manifest.json"
    save_json(manifest, manifest_path)

    # Self-validate: verify the manifest matches the files on disk right now.
    # If this raises, we wrote something inconsistent — surface it loudly.
    verify_seg_predictions_match(
        root=project_root,
        dataset=dataset,
        seg_experiment_name=experiment_name,
        fold=int(fold),
        checkpoint_path=checkpoint_path,
        test_csv_path=test_csv_path,
    )

    return {
        "per_image_df":  per_image_df,
        "micro_counts":  micro,
        "manifest_path": manifest_path,
        "manifest":      manifest,
    }


# --------------------------------------------------------------------------- #
# Experiment-level manifest finalizer (NB05 calls this once, after all folds)
# --------------------------------------------------------------------------- #
def write_experiment_manifest(
    project_root: PathLike,
    dataset: str,
    experiment_name: str,
    expected_n_total: Optional[int] = None,
    expected_folds: Tuple[int, ...] = (1, 2, 3, 4, 5),
) -> Tuple[Path, Dict[str, Any]]:
    """
    Walk every fold's manifest under
        outputs/predictions/segmentation/<dataset>/<experiment_name>/fold_X/manifest.json
    and write the experiment-level manifest:
        outputs/predictions/segmentation/<dataset>/<experiment_name>/prediction_manifest.json

    Returns (prediction_manifest_path, manifest_dict).

    `expected_n_total` is the count of all test images across folds (e.g.
    3,064 for figshare). If supplied, the manifest's
    `all_test_images_covered` is True only when every expected fold is
    present AND sum(n_predictions) == expected_n_total.
    """
    project_root = Path(project_root)
    folds_present: List[int] = []
    total_predictions = 0

    for fold in expected_folds:
        try:
            m = load_seg_prediction_manifest(project_root, dataset, experiment_name, fold=fold)
        except FileNotFoundError:
            continue
        folds_present.append(int(m["fold"]))
        total_predictions += int(m.get("n_predictions", 0))

    folds_present.sort()
    all_present = (set(folds_present) == set(expected_folds))

    if expected_n_total is None:
        all_covered = all_present
    else:
        all_covered = all_present and (total_predictions == int(expected_n_total))

    top = {
        "seg_experiment_name":     experiment_name,
        "dataset":                 dataset,
        "folds_present":           folds_present,
        "expected_folds":          list(expected_folds),
        "total_predictions":       int(total_predictions),
        "expected_n_total":        int(expected_n_total) if expected_n_total is not None else None,
        "all_test_images_covered": bool(all_covered),
        "generated_at":            datetime.now(timezone.utc).isoformat(),
    }

    out_dir = seg_predictions_dir(project_root, dataset, experiment_name)
    out_path = out_dir / "prediction_manifest.json"
    save_json(top, out_path)
    return out_path, top