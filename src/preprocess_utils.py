"""
src/preprocess_utils.py

Dataset converters: read raw dataset files and write the figshare-shaped
output that every downstream notebook expects:

    data/<dataset>/processed/
        images/<image_id>.png      grayscale uint8 256x256
        masks/<image_id>.png       binary    uint8 256x256, values 0/255
        metadata.csv               one row per image (image_id, patient_id, ...)

Supported datasets
------------------
    figshare   233 patients, 3,064 single-slice MRI in .mat (single modality)
               classes: meningioma, glioma, pituitary
               KaggleHub: ashkhagan/figshare-brain-tumor-dataset
               -> discover_mat_files + convert_figshare_mat_to_png_record

    brats2020  369 glioma patients, ~57,000 pre-extracted 2D slices in .h5
               (4 modalities + 3 mask sub-regions per slice). We extract the
               FLAIR modality + a binary "whole tumor" mask (NCR ∪ ED ∪ ET),
               and keep only slices that contain tumor (to match figshare's
               100%-tumor character).
               KaggleHub: awsaf49/brats2020-training-data
               -> discover_brats2020_h5_files + convert_brats2020_h5_to_png_record

REGISTRY PATTERN (project instruction §5)
-----------------------------------------
Adding a new dataset means:
    1. Add a new constant DATASET_NAME_<X>
    2. Add a `discover_<x>_files(root)` and a
       `convert_<x>_<format>_to_png_record(...)` pair
    3. Add a branch to `get_dataset_converter(name)` returning the pair
       plus the KaggleHub dataset id
    4. Do NOT modify existing branches. Old NB01 runs must reproduce.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

from src.file_utils import PathLike

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DATASET_NAME            = "figshare"     # legacy alias (kept for old NB01 imports)
DATASET_NAME_FIGSHARE   = "figshare"
DATASET_NAME_BRATS2020  = "brats2020"

# Tumor-class label space — matches the original figshare encoding so that
# rows from different datasets concatenate cleanly via tumor_class_id.
TUMOR_CLASS_NAMES: Dict[int, str] = {
    0: "no_tumor",
    1: "meningioma",
    2: "glioma",
    3: "pituitary",
}
TUMOR_CLASS_IDS: Dict[str, int] = {v: k for k, v in TUMOR_CLASS_NAMES.items()}

# Figshare numbered .mat filenames: 1.mat, 17.mat, 1261.mat, ...
_FIGSHARE_FILENAME_RE = re.compile(r"^\d+\.mat$", re.IGNORECASE)
_FIGSHARE_NON_SAMPLE  = {"cvind.mat"}

# BraTS2020 .h5 filenames: volume_<n>_slice_<m>.h5
_BRATS_FILENAME_RE = re.compile(r"^volume_(\d+)_slice_(\d+)\.h5$", re.IGNORECASE)

# Channel index in the BraTS .h5 'image' dataset (shape: H x W x 4).
BRATS_MODALITY_INDEX: Dict[str, int] = {
    "t1":    0,
    "t1ce":  1,
    "t2":    2,
    "flair": 3,
}


# --------------------------------------------------------------------------- #
# Shared image/mask normalization helpers (used by both converters)
# --------------------------------------------------------------------------- #
def _normalize_image_to_uint8(
    image: np.ndarray,
    method: str = "percentile",
) -> np.ndarray:
    """Convert a float/int array to uint8 [0, 255] via percentile or minmax."""
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
    """Any positive pixel -> 255, else 0."""
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _resize_to(
    arr: np.ndarray,
    target_size: Tuple[int, int],
    interp: int = cv2.INTER_LINEAR,
    is_mask: bool = False,
) -> np.ndarray:
    th, tw = target_size
    out = cv2.resize(arr, (tw, th), interpolation=interp)
    if is_mask:
        return ((out > 127).astype(np.uint8)) * 255
    return out


def _resize_image_uint8(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return _resize_to(image, target_size, cv2.INTER_LINEAR, is_mask=False)


def _resize_mask_uint8(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return _resize_to(mask, target_size, cv2.INTER_NEAREST, is_mask=True)


def _format_metadata_paths(
    image_out_path: Path,
    mask_out_path: Path,
    source_path: Path,
    path_style: str,
    project_root: Optional[PathLike],
) -> Tuple[str, str, str]:
    """Resolve absolute/relative paths consistently across converters."""
    if path_style == "relative" and project_root is not None:
        root = Path(project_root)
        image_path_str = str(image_out_path.relative_to(root))
        mask_path_str  = str(mask_out_path.relative_to(root))
        source_str = (
            str(source_path.relative_to(root))
            if str(source_path).startswith(str(root)) else str(source_path)
        )
    else:
        image_path_str = str(image_out_path)
        mask_path_str  = str(mask_out_path)
        source_str     = str(source_path)
    return image_path_str, mask_path_str, source_str


# --------------------------------------------------------------------------- #
# FIGSHARE — discovery + converter (unchanged registry entry)
# --------------------------------------------------------------------------- #
def discover_mat_files(dataset_root: PathLike) -> List[Path]:
    """Find all numbered figshare .mat files (filter out cvind.mat and similar)."""
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    candidates = [p for p in root.rglob("*.mat") if p.is_file()]
    samples = [
        p for p in candidates
        if _FIGSHARE_FILENAME_RE.match(p.name)
        and p.name.lower() not in _FIGSHARE_NON_SAMPLE
    ]
    samples.sort(key=lambda p: int(p.stem))
    return samples


def _decode_figshare_pid(pid_array: np.ndarray) -> str:
    arr = np.asarray(pid_array).astype(np.int64).reshape(-1)
    chars = [chr(int(v)) for v in arr if 0 < int(v) < 128]
    pid = "".join(chars).strip()
    return pid if pid else "unknown"


def read_figshare_mat(mat_path: PathLike) -> Dict[str, object]:
    """Read one figshare .mat into a dict with image, mask, label_id, patient_id."""
    with h5py.File(Path(mat_path), "r") as f:
        cjdata = f["cjdata"]
        image = np.array(cjdata["image"]).T
        mask = np.array(cjdata["tumorMask"]).T
        label_id = int(np.array(cjdata["label"]).reshape(-1)[0])
        patient_id = _decode_figshare_pid(np.array(cjdata["PID"]))
    return {
        "image": image, "mask": mask,
        "label_id": label_id, "patient_id": patient_id,
    }


def convert_figshare_mat_to_png_record(
    mat_path: PathLike,
    image_out_dir: PathLike,
    mask_out_dir: PathLike,
    target_size: Tuple[int, int] = (256, 256),
    normalization: str = "percentile",
    path_style: str = "relative",
    project_root: Optional[PathLike] = None,
) -> Optional[Dict[str, object]]:
    """
    Convert one figshare .mat sample. Always returns a metadata record (figshare
    samples always contain a tumor — never None).
    """
    mat_path = Path(mat_path)
    image_out_dir = Path(image_out_dir); image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir  = Path(mask_out_dir);  mask_out_dir.mkdir(parents=True, exist_ok=True)

    image_id = mat_path.stem
    raw = read_figshare_mat(mat_path)

    image_u8 = _normalize_image_to_uint8(raw["image"], method=normalization)
    image_u8 = _resize_image_uint8(image_u8, target_size)
    mask_u8  = _binarize_mask_to_uint8(raw["mask"])
    mask_u8  = _resize_mask_uint8(mask_u8, target_size)

    image_out_path = image_out_dir / f"{image_id}.png"
    mask_out_path  = mask_out_dir  / f"{image_id}.png"
    cv2.imwrite(str(image_out_path), image_u8)
    cv2.imwrite(str(mask_out_path),  mask_u8)

    h, w = image_u8.shape[:2]
    pos = int((mask_u8 > 0).sum())
    total = int(mask_u8.size)
    area_ratio = float(pos) / float(total) if total else 0.0

    img_p, msk_p, src_p = _format_metadata_paths(
        image_out_path, mask_out_path, mat_path, path_style, project_root,
    )

    label_id = int(raw["label_id"])
    return {
        "image_id":             image_id,
        "patient_id":           str(raw["patient_id"]),
        "image_path":           img_p,
        "mask_path":            msk_p,
        "tumor_class":          TUMOR_CLASS_NAMES.get(label_id, "unknown"),
        "tumor_class_id":       label_id,
        "dataset":              DATASET_NAME_FIGSHARE,
        "source_path":          src_p,
        "modality":             "t1",      # FigShare is single-modality T1-weighted
        "height":               int(h),
        "width":                int(w),
        "mask_positive_pixels": pos,
        "mask_area_ratio":      area_ratio,
    }


# --------------------------------------------------------------------------- #
# BRATS2020 — discovery + converter (new registry entry)
# --------------------------------------------------------------------------- #
def discover_brats2020_h5_files(dataset_root: PathLike) -> List[Path]:
    """
    Find all `volume_<n>_slice_<m>.h5` files under dataset_root, recursively,
    sorted by (volume, slice). Non-matching .h5 files (e.g. any metadata.h5)
    are filtered out.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    matched: List[Tuple[int, int, Path]] = []
    for p in root.rglob("*.h5"):
        if not p.is_file():
            continue
        m = _BRATS_FILENAME_RE.match(p.name)
        if m:
            matched.append((int(m.group(1)), int(m.group(2)), p))
    matched.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in matched]


def read_brats2020_h5(h5_path: PathLike) -> Dict[str, np.ndarray]:
    """
    Read one BraTS .h5 slice. Returns:
        image: float32 (H, W, 4)    4 modalities (T1/T1ce/T2/FLAIR)
        mask:  uint8   (H, W, 3)    3 sub-regions (NCR, ED, ET) — usually 0/1
    Handles either ('image','mask') or ('x','y') key conventions, and either
    channel-last (H,W,C) or channel-first (C,H,W) ordering.
    """
    with h5py.File(Path(h5_path), "r") as f:
        keys = list(f.keys())
        img_key  = "image" if "image" in f else ("x" if "x" in f else keys[0])
        mask_key = "mask"  if "mask"  in f else ("y" if "y" in f else keys[1])
        image = np.array(f[img_key])
        mask  = np.array(f[mask_key])

    # Normalize to channel-last (H, W, C) if it arrived channel-first.
    if image.ndim == 3 and image.shape[0] in (3, 4) and image.shape[-1] not in (3, 4):
        image = np.transpose(image, (1, 2, 0))
    if mask.ndim == 3 and mask.shape[0] in (1, 3, 4) and mask.shape[-1] not in (1, 3, 4):
        mask = np.transpose(mask, (1, 2, 0))

    return {"image": image.astype(np.float32), "mask": mask}


def convert_brats2020_h5_to_png_record(
    h5_path: PathLike,
    image_out_dir: PathLike,
    mask_out_dir: PathLike,
    target_size: Tuple[int, int] = (256, 256),
    normalization: str = "percentile",
    modality: str = "flair",
    keep_only_with_tumor: bool = True,
    min_tumor_pixels: int = 1,
    path_style: str = "relative",
    project_root: Optional[PathLike] = None,
) -> Optional[Dict[str, object]]:
    """
    Convert one BraTS2020 .h5 slice. Returns the metadata record dict, or
    `None` if the slice should be skipped (empty / near-empty tumor mask
    and `keep_only_with_tumor=True`).

    Strategy:
        - extract one MRI modality (default: FLAIR)
        - merge the 3 tumor sub-region channels into a single binary
          "whole tumor" mask  (NCR ∪ ED ∪ ET)
        - resize + normalize, write PNGs identical in shape to figshare's

    The figshare and brats2020 outputs are then interchangeable to every
    downstream module (BrainTumorDataset, training, evaluation).
    """
    h5_path = Path(h5_path)
    image_out_dir = Path(image_out_dir); image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir  = Path(mask_out_dir);  mask_out_dir.mkdir(parents=True, exist_ok=True)

    # Parse volume + slice indices from filename
    m = _BRATS_FILENAME_RE.match(h5_path.name)
    if not m:
        raise ValueError(
            f"file does not match 'volume_<n>_slice_<m>.h5' pattern: {h5_path.name}"
        )
    volume_idx = int(m.group(1))
    slice_idx  = int(m.group(2))

    # Read
    mod = modality.lower()
    if mod not in BRATS_MODALITY_INDEX:
        raise ValueError(
            f"unknown modality: {modality!r}. Available: {list(BRATS_MODALITY_INDEX)}"
        )
    mod_idx = BRATS_MODALITY_INDEX[mod]

    raw = read_brats2020_h5(h5_path)
    image_4ch = raw["image"]    # (H, W, 4)
    mask_3ch  = raw["mask"]     # (H, W, 3) or (H, W, 1)

    # Merge sub-regions into binary whole-tumor mask
    if mask_3ch.ndim == 3:
        mask_binary = (mask_3ch > 0).any(axis=-1).astype(np.uint8)
    else:
        mask_binary = (mask_3ch > 0).astype(np.uint8)

    # Early exit on empty tumor (after merging) — saves PNG writes
    tumor_pixels_native = int(mask_binary.sum())
    if keep_only_with_tumor and tumor_pixels_native < min_tumor_pixels:
        return None

    # Extract chosen modality and normalize
    image_2d = image_4ch[..., mod_idx]
    image_u8 = _normalize_image_to_uint8(image_2d, method=normalization)
    image_u8 = _resize_image_uint8(image_u8, target_size)

    mask_u8 = (mask_binary * 255).astype(np.uint8)
    mask_u8 = _resize_mask_uint8(mask_u8, target_size)

    # IDs — patient_id matches the official BraTS naming convention so
    # patient_level CV splits group correctly.
    image_id   = f"brats20_v{volume_idx:03d}_s{slice_idx:03d}"
    patient_id = f"BraTS20_Training_{volume_idx:03d}"

    image_out_path = image_out_dir / f"{image_id}.png"
    mask_out_path  = mask_out_dir  / f"{image_id}.png"
    cv2.imwrite(str(image_out_path), image_u8)
    cv2.imwrite(str(mask_out_path),  mask_u8)

    h, w = image_u8.shape[:2]
    pos = int((mask_u8 > 0).sum())
    total = int(mask_u8.size)
    area_ratio = float(pos) / float(total) if total else 0.0

    img_p, msk_p, src_p = _format_metadata_paths(
        image_out_path, mask_out_path, h5_path, path_style, project_root,
    )

    # Label by slice content: slices with no visible tumor are "no_tumor".
    # BraTS2020 patients are all glioma, so tumor-bearing slices get "glioma".
    slice_class = "glioma" if tumor_pixels_native > 0 else "no_tumor"

    return {
        "image_id":             image_id,
        "patient_id":           patient_id,
        "image_path":           img_p,
        "mask_path":            msk_p,
        "tumor_class":          slice_class,
        "tumor_class_id":       TUMOR_CLASS_IDS[slice_class],
        "has_tumor":            tumor_pixels_native > 0,
        "dataset":              DATASET_NAME_BRATS2020,
        "source_path":          src_p,
        "modality":             mod,
        "volume_idx":           volume_idx,
        "slice_idx":            slice_idx,
        "height":               int(h),
        "width":                int(w),
        "mask_positive_pixels": pos,
        "mask_area_ratio":      area_ratio,
    }


# --------------------------------------------------------------------------- #
# Dataset dispatch (used by NB01)
# --------------------------------------------------------------------------- #
DatasetConverter = Tuple[Callable[..., List[Path]], Callable[..., Optional[Dict]], str]


def get_dataset_converter(dataset_name: str) -> DatasetConverter:
    """
    Return (discover_fn, convert_fn, kagglehub_dataset_id) for `dataset_name`.

    Used by NB01 to dispatch on the DATASET knob without hardcoding any
    dataset-specific paths or filenames in the notebook.
    """
    n = dataset_name.lower()

    if n == DATASET_NAME_FIGSHARE:
        return (
            discover_mat_files,
            convert_figshare_mat_to_png_record,
            "ashkhagan/figshare-brain-tumor-dataset",
        )

    if n == DATASET_NAME_BRATS2020:
        return (
            discover_brats2020_h5_files,
            convert_brats2020_h5_to_png_record,
            "awsaf49/brats2020-training-data",
        )

    raise ValueError(
        f"unknown dataset: {dataset_name!r}. "
        f"Available: {[DATASET_NAME_FIGSHARE, DATASET_NAME_BRATS2020]}. "
        f"Add a new branch to get_dataset_converter in src/preprocess_utils.py."
    )