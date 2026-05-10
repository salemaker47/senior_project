"""
src/preprocess_utils.py

FigShare brain-tumor dataset converter. Reads each .mat sample and writes:

    <image_out_dir>/<image_id>.png   grayscale uint8 256x256
    <mask_out_dir>/<image_id>.png    binary    uint8 256x256, values 0/255

plus returns one metadata record per image (dict). NB01 collects these into
data/<dataset>/processed/metadata.csv.

Output layout (under each dataset's processed/ folder, see file_utils.dataset_paths):

    data/figshare/processed/
        images/<image_id>.png
        masks/<image_id>.png
        metadata.csv

Dataset reminder
----------------
Each numbered .mat file has a top-level group `cjdata` containing:
    cjdata/image       - MRI image
    cjdata/tumorMask   - ground-truth tumor mask
    cjdata/label       - 1 = meningioma, 2 = glioma, 3 = pituitary
    cjdata/PID         - patient ID (uint16 array of ASCII codes)
    cjdata/tumorBorder - tumor border points (not used here)

`cvind.mat` is a CV index file shipped alongside the samples and must NOT be
treated as an image. It is filtered out by `discover_mat_files`.

Adding a new dataset (registry pattern, project instruction §5)
---------------------------------------------------------------
Add a new `convert_<dataset>_to_png_record(...)` function below. Do NOT
modify existing converters. NB01 reads the `DATASET` knob and dispatches
to the matching converter.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np

PathLike = Union[str, Path]

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DATASET_NAME = "figshare"

# Numbered sample filenames look like 1.mat, 17.mat, 1261.mat, ...
_NUMBERED_MAT_RE = re.compile(r"^\d+\.mat$", re.IGNORECASE)

# Files that exist in the dataset but are NOT image samples.
_NON_SAMPLE_FILENAMES = {"cvind.mat"}

TUMOR_CLASS_NAMES: Dict[int, str] = {
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_mat_files(dataset_root: PathLike) -> List[Path]:
    """
    Recursively find all numbered FigShare sample .mat files under
    `dataset_root`, sorted by their numeric stem.

    `cvind.mat` and any other non-numeric .mat files are filtered out.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    candidates = [p for p in root.rglob("*.mat") if p.is_file()]
    samples = [
        p for p in candidates
        if _NUMBERED_MAT_RE.match(p.name)
        and p.name.lower() not in _NON_SAMPLE_FILENAMES
    ]
    samples.sort(key=lambda p: int(p.stem))
    return samples


# --------------------------------------------------------------------------- #
# Low-level .mat reader
# --------------------------------------------------------------------------- #
def _decode_pid(pid_array: np.ndarray) -> str:
    """
    Decode the cjdata/PID field. PID is stored as an array of small uint16
    values that are ASCII character codes. We decode and strip whitespace.
    """
    arr = np.asarray(pid_array).astype(np.int64).reshape(-1)
    chars = []
    for v in arr:
        if 0 < int(v) < 128:
            chars.append(chr(int(v)))
    pid = "".join(chars).strip()
    return pid if pid else "unknown"


def read_figshare_mat(mat_path: PathLike) -> Dict[str, object]:
    """
    Read one FigShare .mat file and return a dict with raw arrays.
    The image and mask are transposed to undo MATLAB's column-major ordering.
    """
    p = Path(mat_path)
    with h5py.File(p, "r") as f:
        cjdata = f["cjdata"]
        image = np.array(cjdata["image"]).T               # (H, W)
        mask = np.array(cjdata["tumorMask"]).T            # (H, W)
        label_id = int(np.array(cjdata["label"]).reshape(-1)[0])
        patient_id = _decode_pid(np.array(cjdata["PID"]))

    return {
        "image": image,
        "mask": mask,
        "label_id": label_id,
        "patient_id": patient_id,
    }


# --------------------------------------------------------------------------- #
# Image / mask normalization helpers
# --------------------------------------------------------------------------- #
def _normalize_image_to_uint8(
    image: np.ndarray,
    method: str = "percentile",
) -> np.ndarray:
    """
    Convert a raw int16 / float MRI slice to uint8 [0, 255].

    method = "percentile" (default):
        Robust contrast: clip to [p1, p99] then linearly map to [0, 255].
    method = "minmax":
        Standard cv2.NORM_MINMAX scaling.
    """
    img = image.astype(np.float32)

    if method == "percentile":
        lo, hi = np.percentile(img, [1.0, 99.0])
        if hi <= lo:
            hi = lo + 1.0
        img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)

    if method == "minmax":
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    raise ValueError(f"unknown normalization method: {method!r}")


def _binarize_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    """
    FigShare masks come as 0/1 uint8 (or sometimes float). Convert to a
    proper binary uint8 PNG: 0 = background, 255 = tumor.
    """
    m = np.asarray(mask)
    return (m > 0).astype(np.uint8) * 255


def _resize_image_uint8(
    image: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Resize a uint8 image with bilinear interpolation."""
    th, tw = target_size
    return cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)


def _resize_mask_uint8(
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a binary mask. We use NEAREST so we don't get ghost grey pixels,
    then re-threshold to be safe.
    """
    th, tw = target_size
    out = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    return ((out > 127).astype(np.uint8)) * 255


# --------------------------------------------------------------------------- #
# Public conversion entry point (figshare)
# --------------------------------------------------------------------------- #
def convert_figshare_mat_to_png_record(
    mat_path: PathLike,
    image_out_dir: PathLike,
    mask_out_dir: PathLike,
    target_size: Tuple[int, int] = (256, 256),
    normalization: str = "percentile",
    path_style: str = "relative",
    project_root: Optional[PathLike] = None,
) -> Dict[str, object]:
    """
    Convert one FigShare .mat file to:

        <image_out_dir>/<image_id>.png   (grayscale uint8 256x256)
        <mask_out_dir>/<image_id>.png    (binary    uint8 256x256, values 0/255)

    Returns one metadata record (dict) ready to be appended to a list and
    turned into metadata.csv by NB01.

    Path style
    ----------
    `path_style="relative"` with `project_root` set causes the metadata
    record's image_path / mask_path to be stored relative to project_root
    (e.g. `data/figshare/processed/images/1.png`). This makes metadata.csv
    portable across Drive <-> local SSD; data_utils resolves at load time.

    `path_style="absolute"` stores the absolute paths. Use only if you
    don't intend to share the metadata across environments.
    """
    mat_path = Path(mat_path)
    image_out_dir = Path(image_out_dir)
    mask_out_dir = Path(mask_out_dir)
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    image_id = mat_path.stem  # e.g. "1261"

    # 1. Read .mat.
    raw = read_figshare_mat(mat_path)
    image = raw["image"]
    mask = raw["mask"]
    label_id = int(raw["label_id"])
    patient_id = str(raw["patient_id"])

    # 2. Normalize + resize.
    image_u8 = _normalize_image_to_uint8(image, method=normalization)
    image_u8 = _resize_image_uint8(image_u8, target_size)

    mask_u8 = _binarize_mask_to_uint8(mask)
    mask_u8 = _resize_mask_uint8(mask_u8, target_size)

    # 3. Write PNGs (grayscale).
    image_out_path = image_out_dir / f"{image_id}.png"
    mask_out_path = mask_out_dir / f"{image_id}.png"
    cv2.imwrite(str(image_out_path), image_u8)
    cv2.imwrite(str(mask_out_path), mask_u8)

    # 4. Stats.
    h, w = image_u8.shape[:2]
    positive_pixels = int((mask_u8 > 0).sum())
    total_pixels = int(mask_u8.size)
    mask_area_ratio = (
        float(positive_pixels) / float(total_pixels) if total_pixels else 0.0
    )

    # 5. Path formatting for metadata.
    if path_style == "relative" and project_root is not None:
        root = Path(project_root)
        image_path_str = str(image_out_path.relative_to(root))
        mask_path_str = str(mask_out_path.relative_to(root))
        source_mat_str = (
            str(mat_path.relative_to(root))
            if str(mat_path).startswith(str(root))
            else str(mat_path)
        )
    else:
        image_path_str = str(image_out_path)
        mask_path_str = str(mask_out_path)
        source_mat_str = str(mat_path)

    return {
        "image_id":             image_id,
        "patient_id":           patient_id,
        "image_path":           image_path_str,
        "mask_path":            mask_path_str,
        "tumor_class":          TUMOR_CLASS_NAMES.get(label_id, "unknown"),
        "tumor_class_id":       label_id,
        "dataset":              DATASET_NAME,
        "source_mat_path":      source_mat_str,
        "height":               int(h),
        "width":                int(w),
        "mask_positive_pixels": positive_pixels,
        "mask_area_ratio":      mask_area_ratio,
    }