"""
src/cls_vis_utils.py

Publication-ready visualization helpers for classification results.
Mirrors sg_vis_utils.py in structure; reads from aggregated DataFrames
produced by cls_eval_utils.aggregate_cv_results_cls.

Public API
----------
plot_confusion_matrix(cv_confusion, title, normalize, cmap,
                      save_path, show, ax) -> plt.Axes
    Single labeled confusion matrix (row-normalised colour + raw count text).

plot_confusion_pair(cv_confusion_a, cv_confusion_b,
                    title_a, title_b, suptitle,
                    save_path, show) -> plt.Figure
    Two confusion matrices side-by-side (Eval A / Eval B).

plot_per_class_f1(cv_results_a, cv_results_b, class_names, title,
                  save_path, show) -> plt.Figure
    Grouped bar chart: per-class F1 + macro, with fold std error bars.
    Renders 1-2 bar sets depending on whether cv_results_b is supplied.

plot_eval_gap(cv_results_a, cv_results_b, title, seg_experiment_name,
              save_path, show) -> plt.Figure
    Two-panel gap chart:
    Left  — per-fold macro F1 for Eval A (blue) and Eval B (orange)
            connected by dashed gap lines.
    Right — horizontal gap bars (A-B per fold) + mean ± std annotation.

plot_sample_patches(cv_per_image, project_root, mask_source,
                    predictions_dir, n_per_class, kind,
                    target_size, padding_frac, random_state,
                    title, save_path, show) -> plt.Figure
    Grid of extracted tumor patches (rows=true class, cols=examples).
    Green border = correct prediction; red border = incorrect.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.file_utils import PathLike
from src.cls_data_utils import CLASS_NAMES as _DEFAULT_CLASS_NAMES, extract_patch

# Colour palette for Eval A / Eval B bars
_COLOR_A = "#4878CF"   # blue
_COLOR_B = "#E87A41"   # orange


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _class_names_from_confusion(cv_confusion: pd.DataFrame) -> List[str]:
    """
    Derive clean class names from the labeled confusion DataFrame.
    Columns are expected to be like 'pred_meningioma'; strip the prefix.
    Falls back to _DEFAULT_CLASS_NAMES if parsing fails.
    """
    try:
        return [c.replace("pred_", "") for c in cv_confusion.columns.tolist()]
    except Exception:
        return list(_DEFAULT_CLASS_NAMES)


def _tight_save(fig: plt.Figure, save_path: Optional[PathLike]) -> None:
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")


def _render_cm_on_ax(
    ax: plt.Axes,
    cv_confusion: pd.DataFrame,
    title: str = "",
    normalize: bool = True,
    cmap: str = "Blues",
) -> "matplotlib.image.AxesImage":
    """
    Render a confusion matrix onto an existing Axes and return the AxesImage.
    Colour encodes row-normalised recall; text shows raw integer counts.
    Called by both plot_confusion_matrix and plot_confusion_pair.
    """
    class_names = _class_names_from_confusion(cv_confusion)
    cm = cv_confusion.to_numpy().astype(float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_display = cm / row_sums
    else:
        cm_display = cm.copy()

    im = ax.imshow(cm_display, vmin=0, vmax=1 if normalize else None, cmap=cmap)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    if title:
        ax.set_title(title, fontsize=10, pad=6)

    for r in range(len(class_names)):
        for c in range(len(class_names)):
            text_color = "white" if cm_display[r, c] > 0.55 else "black"
            ax.text(
                c, r, str(int(cm[r, c])),
                ha="center", va="center",
                color=text_color, fontsize=10, fontweight="bold",
            )

    return im


# --------------------------------------------------------------------------- #
# Single confusion matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    cv_confusion: pd.DataFrame,
    title: str = "",
    normalize: bool = True,
    cmap: str = "Blues",
    save_path: Optional[PathLike] = None,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Render a single labeled confusion matrix.

    Parameters
    ----------
    cv_confusion : labeled DataFrame from cls_eval_utils.aggregate_cv_confusion_from_matrix.
                   Rows = true class (index "true_<cls>"), cols = predicted class.
    normalize    : if True, row-normalise the colour map (recall per row);
                   raw integer counts are always shown as text.
    cmap         : matplotlib colour map name.
    save_path    : if given, save the figure to this path.
                   Ignored when ax= is provided (caller owns the figure).
    show         : if True, call plt.show().
                   Ignored when ax= is provided.
    ax           : if given, render into this Axes (no new figure created);
                   save_path, show, and the colorbar are all skipped.

    Returns
    -------
    The Axes object (useful when embedding in a wider figure).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4.5, 4))

    im = _render_cm_on_ax(ax, cv_confusion, title=title, normalize=normalize, cmap=cmap)

    if standalone:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        _tight_save(fig, save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)

    return ax


# --------------------------------------------------------------------------- #
# Side-by-side confusion pair
# --------------------------------------------------------------------------- #
def plot_confusion_pair(
    cv_confusion_a: pd.DataFrame,
    cv_confusion_b: pd.DataFrame,
    title_a: str = "Eval A — GT masks",
    title_b: str = "Eval B — predicted masks",
    suptitle: str = "",
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Two confusion matrices side-by-side with a shared colour scale (0–1).

    Parameters
    ----------
    cv_confusion_a/b : labeled DataFrames from cls_eval_utils.
    title_a/b        : subplot titles.
    suptitle         : figure-level super-title (e.g. experiment name).
    save_path        : optional output path.
    show             : if True, call plt.show().
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for ax, cv_conf, title in zip(axes, [cv_confusion_a, cv_confusion_b], [title_a, title_b]):
        im = _render_cm_on_ax(ax, cv_conf, title=title)

    # Shared colour-bar anchored to the right panel
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=1.02)

    fig.tight_layout()
    _tight_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Per-class F1 grouped bar chart
# --------------------------------------------------------------------------- #
def plot_per_class_f1(
    cv_results_a: pd.DataFrame,
    cv_results_b: Optional[pd.DataFrame] = None,
    class_names: Optional[List[str]] = None,
    title: str = "Per-class F1 — 5-fold cross-validation",
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Grouped bar chart of per-class F1 + macro F1 with fold std error bars.

    Parameters
    ----------
    cv_results_a : DataFrame from aggregate_cv_results_cls (1 row per fold).
                   Expected columns: f1_meningioma, f1_glioma, f1_pituitary, macro_f1.
    cv_results_b : optional second variant (Eval B). If None, single-variant chart.
    class_names  : class order (default ["meningioma","glioma","pituitary"]).
    title        : figure title.
    """
    class_names = class_names or list(_DEFAULT_CLASS_NAMES)
    metrics = [f"f1_{c}" for c in class_names] + ["macro_f1"]
    labels  = class_names + ["macro"]
    n_groups = len(labels)

    def _stats(df: pd.DataFrame, col: str):
        vals = df[col].astype(float)
        ddof = 1 if len(vals) > 1 else 0
        return float(vals.mean()), float(vals.std(ddof=ddof))

    fig, ax = plt.subplots(figsize=(max(7, n_groups * 1.6), 4.5))

    x = np.arange(n_groups)
    has_b = cv_results_b is not None
    width = 0.38 if has_b else 0.6

    offsets_a = -width / 2 if has_b else 0.0
    means_a = [_stats(cv_results_a, m)[0] for m in metrics]
    stds_a  = [_stats(cv_results_a, m)[1] for m in metrics]

    bars_a = ax.bar(
        x + offsets_a, means_a, width,
        yerr=stds_a, capsize=4,
        label="Eval A (GT masks)",
        color=_COLOR_A, alpha=0.85,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )

    if has_b:
        means_b = [_stats(cv_results_b, m)[0] for m in metrics]
        stds_b  = [_stats(cv_results_b, m)[1] for m in metrics]
        ax.bar(
            x + width / 2, means_b, width,
            yerr=stds_b, capsize=4,
            label="Eval B (pred masks)",
            color=_COLOR_B, alpha=0.85,
            error_kw={"elinewidth": 1.2, "ecolor": "black"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("F1 score", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels above each bar
    for bar in bars_a:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, h + 0.012,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5, color=_COLOR_A,
        )

    fig.tight_layout()
    _tight_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Eval A vs Eval B gap chart
# --------------------------------------------------------------------------- #
def plot_eval_gap(
    cv_results_a: pd.DataFrame,
    cv_results_b: pd.DataFrame,
    title: str = "Eval A vs Eval B — macro F1 gap",
    seg_experiment_name: str = "",
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Two-panel figure visualising the gap between GT-mask and predicted-mask eval.

    Left panel  — per-fold macro F1 dots for Eval A (blue) and Eval B (orange),
                  connected by dashed vertical gap lines; horizontal mean lines.
    Right panel — horizontal bar per fold showing gap = A − B;
                  dashed vertical line at mean gap; 'mean ± std' annotation.

    Parameters
    ----------
    cv_results_a/b       : DataFrames from aggregate_cv_results_cls.
                           Required columns: fold, macro_f1.
    seg_experiment_name  : appended to the subtitle for traceability.
    """
    df_a = cv_results_a[["fold", "macro_f1"]].copy().rename(
        columns={"macro_f1": "f1_a"}
    )
    df_b = cv_results_b[["fold", "macro_f1"]].copy().rename(
        columns={"macro_f1": "f1_b"}
    )
    gap_df = df_a.merge(df_b, on="fold", how="inner").sort_values("fold")
    gap_df["gap"] = gap_df["f1_a"] - gap_df["f1_b"]

    folds = gap_df["fold"].tolist()
    y_pos = np.arange(len(folds))
    n_folds = len(folds)
    ddof = 1 if n_folds > 1 else 0

    mean_a = float(gap_df["f1_a"].mean())
    mean_b = float(gap_df["f1_b"].mean())
    mean_g = float(gap_df["gap"].mean())
    std_g  = float(gap_df["gap"].std(ddof=ddof))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, max(3.5, n_folds * 0.75 + 1.5)))

    # ---- Left panel: per-fold F1 ----
    ax_l.scatter(gap_df["f1_a"], y_pos, color=_COLOR_A, s=70, zorder=4, label="Eval A (GT)")
    ax_l.scatter(gap_df["f1_b"], y_pos, color=_COLOR_B, s=70, zorder=4, label="Eval B (pred)")

    for i, row in gap_df.iterrows():
        yy = y_pos[folds.index(int(row["fold"]))]
        ax_l.plot(
            [row["f1_b"], row["f1_a"]], [yy, yy],
            color="gray", lw=1.2, ls="--", alpha=0.6, zorder=3,
        )

    ax_l.axvline(mean_a, color=_COLOR_A, lw=1.2, ls=":", alpha=0.7)
    ax_l.axvline(mean_b, color=_COLOR_B, lw=1.2, ls=":", alpha=0.7)

    ax_l.set_yticks(y_pos)
    ax_l.set_yticklabels([f"fold {f}" for f in folds], fontsize=9)
    ax_l.set_xlabel("Macro F1", fontsize=10)
    ax_l.set_title("Per-fold macro F1", fontsize=10)
    ax_l.legend(fontsize=8, loc="lower right")
    ax_l.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax_l.spines[["top", "right"]].set_visible(False)

    # ---- Right panel: gap bars ----
    gap_colors = [
        "#4CAF50" if g >= 0 else "#E53935"
        for g in gap_df["gap"].tolist()
    ]
    ax_r.barh(y_pos, gap_df["gap"], color=gap_colors, alpha=0.8, height=0.55)
    ax_r.axvline(0, color="black", lw=0.8)
    ax_r.axvline(mean_g, color="gray", lw=1.5, ls="--")

    ax_r.annotate(
        f"mean gap\n{mean_g:+.4f} ± {std_g:.4f}",
        xy=(mean_g, n_folds - 0.5),
        xytext=(mean_g + (0.005 if mean_g >= 0 else -0.005), n_folds - 0.2),
        fontsize=8, ha="left" if mean_g >= 0 else "right",
        color="dimgray",
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.8),
    )

    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels([f"fold {f}" for f in folds], fontsize=9)
    ax_r.set_xlabel("Gap = Eval A − Eval B", fontsize=10)
    ax_r.set_title("Macro F1 gap  (A − B)", fontsize=10)
    ax_r.grid(axis="x", alpha=0.3, linewidth=0.6)
    ax_r.spines[["top", "right"]].set_visible(False)

    subtitle = f"seg: {seg_experiment_name}" if seg_experiment_name else ""
    fig.suptitle(
        title + (f"\n{subtitle}" if subtitle else ""),
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    _tight_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Sample patch grid
# --------------------------------------------------------------------------- #
def plot_sample_patches(
    cv_per_image: pd.DataFrame,
    project_root: PathLike,
    mask_source: str = "gt",
    predictions_dir: Optional[PathLike] = None,
    n_per_class: int = 3,
    kind: str = "correct",
    target_size: int = 224,
    padding_frac: float = 0.10,
    random_state: int = 42,
    title: str = "",
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Grid of extracted tumor patches (rows = true class, cols = examples).

    Each panel shows the ROI patch with a subtitle "true=<cls>  pred=<cls>".
    Correct predictions: green border; incorrect: red border.

    Parameters
    ----------
    cv_per_image    : per-image result DataFrame from aggregate_cv_results_cls
                      (contains image_id, tumor_class, predicted_class_name, correct, …).
    project_root    : local project root used to locate processed images + masks.
    mask_source     : "gt" or "predicted"; selects which mask to extract the patch from.
    predictions_dir : path to fold-level seg prediction PNGs (required if mask_source="predicted").
    n_per_class     : number of example columns.
    kind            : "correct"    — sample only correctly classified images
                      "incorrect"  — sample only misclassified images
                      "random"     — sample without filtering
    target_size     : patch resize target (default 224).
    padding_frac    : padding around the mask bbox (default 0.10).
    random_state    : RNG seed for reproducible sampling.
    title           : figure super-title.
    save_path       : optional output path.
    show            : if True, call plt.show().
    """
    project_root = Path(project_root)
    rng = np.random.RandomState(random_state)

    class_names = list(_DEFAULT_CLASS_NAMES)

    # Resolve dataset from DataFrame if available
    dataset = cv_per_image["dataset"].iloc[0] if "dataset" in cv_per_image.columns else "figshare"
    images_dir = project_root / "data" / dataset / "processed" / "images"
    masks_dir  = project_root / "data" / dataset / "processed" / "masks"

    # Filter by kind
    if kind == "correct":
        df = cv_per_image[cv_per_image["correct"] == 1]
    elif kind == "incorrect":
        df = cv_per_image[cv_per_image["correct"] == 0]
    else:
        df = cv_per_image.copy()

    fig, axes = plt.subplots(
        len(class_names), n_per_class,
        figsize=(n_per_class * 2.5, len(class_names) * 2.8),
    )
    if len(class_names) == 1:
        axes = axes[np.newaxis, :]
    if n_per_class == 1:
        axes = axes.reshape(-1, 1)

    for r, cls_name in enumerate(class_names):
        cls_df = df[df["tumor_class"] == cls_name]
        n = min(n_per_class, len(cls_df))
        if n > 0:
            sample = cls_df.sample(n=n, random_state=rng)
        else:
            sample = cls_df  # empty

        for c in range(n_per_class):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])

            if c >= len(sample):
                ax.axis("off")
                continue

            row = sample.iloc[c]
            image_id = str(row["image_id"])

            img_path  = images_dir / f"{image_id}.png"
            if mask_source == "predicted" and predictions_dir is not None:
                mask_path = Path(predictions_dir) / f"{image_id}.png"
            else:
                mask_path = masks_dir / f"{image_id}.png"

            # Load
            img = cv2.imread(str(img_path),  cv2.IMREAD_GRAYSCALE) if img_path.exists()  else None
            msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None

            if img is None:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                continue

            # Fall back to a zeros mask (whole-image crop) when the mask file is missing.
            msk_bin = (msk > 127).astype(np.uint8) * 255 if msk is not None else np.zeros_like(img)
            patch = extract_patch(img, msk_bin, target_size=target_size, padding_frac=padding_frac)
            ax.imshow(patch)

            # Coloured border
            is_correct = bool(row["correct"])
            border_color = "#4CAF50" if is_correct else "#E53935"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

            pred_name = row.get("predicted_class_name", str(row.get("predicted_class", "?")))
            ax.set_xlabel(
                f"true: {cls_name}\npred: {pred_name}",
                fontsize=7.5,
                labelpad=2,
            )

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

    fig.tight_layout()
    _tight_save(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
