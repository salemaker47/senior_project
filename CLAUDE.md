# Senior_Project — Updated Project Instruction

A multi-dataset, multi-task medical imaging project for brain tumor analysis. Successor to the BraTS_figshare prototype, restructured to support multiple datasets and both segmentation and classification tasks under one consistent layout.

---

## 1. Project goal

Build complete pipelines for two tasks on brain MRI data:

- **Segmentation** — input: brain MRI (grayscale 256×256). Output: binary tumor mask (0 = background, 1 = tumor).
- **Classification** — input: tumor patch cropped from the MRI using a tumor mask. Output: tumor class (meningioma, glioma, pituitary). Two evaluation variants: GT masks (Eval A) and predicted masks from segmentation (Eval B). The gap between Eval A and Eval B quantifies the cost of using imperfect ROI in deployment.

Methodology: 5-fold cross-validation. Two split schemes:
- `patient_level` — `StratifiedGroupKFold` on `patient_id`. Used for the project's final results.
- `image_level` — plain `KFold` on filenames. Used to reproduce the FigShare reference benchmark's leaky-split numbers.

---

## 2. Datasets

| dataset | status | size | classes | source |
|---|---|---|---|---|
| `figshare` | active | 3,064 images / 233 patients | meningioma, glioma, pituitary | KaggleHub `ashkhagan/figshare-brain-tumor-dataset` |
| `brats2020` | active | ~57k slices / 369 patients | glioma (segmentation only) | KaggleHub `awsaf49/brats2020-training-data` |

Adding a new dataset means: dropping its raw files into `data/<dataset>/raw/`, running NB01 with `DATASET = "<dataset>"`, running NB02 to produce fold CSVs, and changing one knob in any training/testing notebook. No source-code edits required.

---

## 3. Folder structure

```
Senior_Project/
├── _backups/
├── configs/
│   ├── seg/reference_experiments.py    get_experiment() for seg recipes 01–07
│   └── cls/reference_experiments.py    get_experiment() for cls recipes cls01–cls04
├── data/
│   ├── figshare/
│   │   ├── raw/
│   │   ├── processed/
│   │   │   ├── images/<image_id>.png
│   │   │   ├── masks/<image_id>.png
│   │   │   ├── metadata.csv
│   │   │   ├── metadata_summary.csv
│   │   │   └── preprocessing_config.json
│   │   └── splits/
│   │       ├── patient_level/
│   │       │   ├── cv_split_config.json
│   │       │   ├── cv_fold_summary.csv
│   │       │   └── folds/fold_X_{train,val,test}.csv
│   │       └── image_level/  (same layout)
│   └── brats2020/  (same layout)
├── outputs/
│   ├── checkpoints/<task>/<dataset>/<exp>/fold_X/{best.ckpt, best_model.pt, experiment_config.json}
│   ├── logs/<task>/<dataset>/<exp>/fold_X/lightning_logs/version_0/metrics.csv
│   ├── predictions/
│   │   └── segmentation/<dataset>/<seg_exp>/
│   │       ├── prediction_manifest.json
│   │       └── fold_X/
│   │           ├── manifest.json
│   │           └── <image_id>.png
│   ├── tables/<task>/<dataset>/<exp>/
│   │   ├── (segmentation)  cv_results.csv, cv_summary.csv, cv_summary_enriched.csv,
│   │   │                   cv_by_class.csv, cv_class_summary.csv, cv_per_image.csv,
│   │   │                   fold_X_*.csv
│   │   └── (classification)
│   │       ├── eval_gt/  {cv_results, cv_summary, cv_summary_enriched,
│   │       │             cv_confusion, cv_by_class, cv_per_image, fold_X_*}.csv
│   │       └── eval_pred__<seg_exp>/  (same structure)
│   └── figures/
│       ├── segmentation/<dataset>/<seg_exp>/fold_X/sample_predictions/*.png
│       └── classification/<dataset>/<cls_exp>/
│           ├── <eval_variant>/fold_X/{confusion_matrix.png, sample_patches/*.png}
│           └── {confusion_pair, per_class_f1, eval_gap, sample_patches}.png
├── notebooks/
│   ├── setups/
│   │   ├── 01_data_preparation_figshare.ipynb
│   │   ├── 01_data_preparation_brats2020.ipynb
│   │   └── 02_split.ipynb
│   ├── segmentation/
│   │   ├── 03_train.ipynb
│   │   ├── 04_data_vis.ipynb
│   │   ├── 05_test.ipynb
│   │   └── 06_seg_compare.ipynb
│   └── classification/
│       ├── 06_classification_data.ipynb
│       ├── 07_train_cls.ipynb
│       ├── 08_test_cls.ipynb
│       └── 09_cls_results.ipynb
└── src/  (see §4)
```

This is the canonical layout as it lives in Google Drive. See §11 for how it maps to the Colab runtime and the GitHub repo.

### Path conventions

Three orthogonal axes define every artifact:

- **task** — `segmentation` or `classification`. First segment under `outputs/<category>/`.
- **dataset** — `figshare`, `brats2020`, etc. Second segment.
- **experiment_name** — config-only identifier, e.g. `01_dice_image_level` or `cls01_resnet50`. No dataset suffix.

Glob queries become natural:

- All seg experiments on figshare → `outputs/checkpoints/segmentation/figshare/*/`
- Same experiment across all datasets → `outputs/*/*/figshare/01_dice_image_level/`

---

## 4. src/ — module layout and naming convention

**Naming rule:** files used by both tasks have no prefix. Files used only by one task are prefixed `sg_` (segmentation) or `cls_` (classification).

`PathLike = Union[str, Path]` is defined once in `file_utils.py` and imported everywhere else — never redefined locally.

`IMAGENET_MEAN/STD` is defined once in `sg_data_utils.py` and imported by `cls_data_utils.py`.

| order | file | scope | responsibility |
|---|---|---|---|
| 1 | `file_utils.py` | shared | `PathLike`, `project_dirs`, `dataset_paths`, `experiment_paths`, `experiment_root_paths`, `split_scheme_dir`, `fold_split_csv_paths`, `seg_predictions_dir`, `load_seg_prediction_manifest`, `verify_seg_predictions_match`, `cls_eval_paths`, JSON helpers, SHA-256 hashing |
| 2 | `notebook_setup.py` | shared | `setup_environment` (mounts Drive + git pull), `copy_to_local`, `sync_outputs_to_drive` |
| 3 | `preprocess_utils.py` | shared | figshare (.mat) and brats2020 (.h5) converters; `get_dataset_converter` dispatch |
| 4a | `vis_utils.py` | shared | `load_grayscale_png`, `load_binary_mask_png` (cv2-based), `show_class_examples` |
| 4b | `sg_vis_utils.py` | seg | `show_triplet`, `show_overlay_triplet`, `show_image_gt_pred_overlay` (4-panel) |
| 5a | `data_utils.py` | shared | `load_metadata`, `validate_metadata`, `metadata_summary`, `create_patient_folds`, `create_image_level_folds`, `make_train_val_from_pool`, `make_train_val_image_level`, `verify_no_patient_leakage`, `save_fold_csvs` |
| 5b | `sg_data_utils.py` | seg | `IMAGENET_MEAN`, `IMAGENET_STD`, `BrainTumorDataset`, `build_train_transform`, `build_eval_transform`, `build_dataloaders`, `build_test_loader` |
| 6a | `eval_utils.py` | shared | `enriched_aggregate`, `build_fold_summary`, `add_mean_std` |
| 6b | `sg_eval_utils.py` | seg | `micro_dice_from_counts`, `micro_iou_from_counts`, `micro_sensitivity_from_counts`, `micro_precision_from_counts`, `summarize_fold_results`, `aggregate_cv_results`, `aggregate_cv_per_patient`, `aggregate_cv_training_summary` |
| 7 | `sg_metrics.py` | seg | `binarize_logits`, `dice_score`, `iou_score`, `get_smp_stats`, `micro_dice_from_stats`, `micro_iou_from_stats`, `get_metric_kind_pairs`, `compute_per_image_metrics_from_logits` |
| 8 | `sg_losses.py` | seg | bce, dice, focal, lovasz, dice_bce, dice_focal, combo |
| 9 | `sg_models.py` | seg | SMP-based registry: `smp_unet_resnet34`, `smp_unetpp_efficientnetb4`, etc. |
| 10 | `optimizers.py` | shared | adam, adamw, sgd, rmsprop / reduce_on_plateau, cosine, cosine_warm_restarts, step, multistep, exponential, none |
| 11 | `train_utils.py` | shared | `set_global_seed` (delegates to `pl.seed_everything`), `gather_repro_metadata`, `TrainingTimingCallback`, `EpochSummaryPrinter`, `build_callbacks`, `build_trainer`, `export_plain_state_dict`, `strip_model_prefix` |
| 12 | `sg_lightning_module.py` | seg | `BrainTumorSegModule` — train/val steps, micro metric accumulation, registry-driven optimizer + scheduler |
| 13 | `sg_test_utils.py` | seg | `load_model_from_pt`, `load_model_from_ckpt`, `predict_mask`, `evaluate_fold` (image_id-keyed row lookup, writes prediction manifests), `write_experiment_manifest` |
| 14 | `cls_data_utils.py` | cls | `extract_patch`, `BrainTumorClsDataset`, cls transforms, `build_dataloaders_cls`, `build_test_loader_cls` |
| 15 | `cls_models.py` | cls | timm-based registry: resnet50, efficientnet_b0/b4, vit_small_patch16_224 |
| 16 | `cls_losses.py` | cls | cross_entropy, cross_entropy_smooth, focal_ce |
| 17 | `cls_metrics.py` | cls | `macro_f1_from_preds`, `accuracy_from_preds`, `per_class_metrics`, `confusion_matrix_from_preds` (vectorised via `np.add.at`), `compute_per_image_metrics_cls` |
| 18 | `cls_lightning_module.py` | cls | `BrainTumorClsModule` — train/val steps, macro F1 + accuracy accumulation, registry-driven optimizer + scheduler |
| 19 | `cls_test_utils.py` | cls | `load_cls_model_from_pt`, `evaluate_fold_cls` (handles both mask sources, writes per-fold manifest and tables) |
| 20 | `cls_eval_utils.py` | cls | `aggregate_cv_results_cls`, `aggregate_cv_confusion_from_matrix` |
| 21 | `cls_vis_utils.py` | cls | `plot_confusion_matrix`, `plot_confusion_pair`, `plot_per_class_f1`, `plot_eval_gap`, `plot_sample_patches` |

All 21 files are implemented and smoke-tested.

---

## 5. Architecture: registry pattern

All components are selected by string name in the `EXPERIMENT` dict. To add a new model/loss/metric/optimizer/scheduler/aug variant:

1. Add a new `if n == "new_name":` branch in the appropriate registry (`sg_models.py`, `cls_models.py`, `sg_losses.py`, `cls_losses.py`, `optimizers.py`, etc.).
2. Do **not** modify existing branches.
3. Reference the new name in an `EXPERIMENT` dict.
4. Old experiments are unaffected.

Canonical recipes live in `configs/seg/reference_experiments.py` and `configs/cls/reference_experiments.py`. Notebooks call `get_experiment(recipe, ...)` rather than constructing the dict by hand.

---

## 6. EXPERIMENT dict — the single source of truth per run

### Segmentation

```python
EXPERIMENT = get_experiment("03_dicebce_image_level", fold=1)
# or manually:
EXPERIMENT = {
    "name":         "01_dice_image_level",
    "task":         "segmentation",
    "dataset":      "figshare",
    "split_scheme": "image_level",

    "fold": 1,
    "image_size": 256,
    "batch_size": 8,
    "num_workers": 0,

    "preprocessing":         "original",
    "augmentation_strength": "reference",

    "model_name":      "smp_unet_resnet34",
    "encoder_weights": "imagenet",

    "loss_name":   "dice",
    "loss_kwargs": {},

    "optimizer_name":   "adam",
    "optimizer_kwargs": {"lr": 1e-4},
    "scheduler_name":   "reduce_on_plateau",
    "scheduler_kwargs": {"mode": "min", "factor": 0.1, "patience": 5, "min_lr": 1e-7},
    "scheduler_monitor":"val_loss",

    "metric_kind": "micro_macro",
    "max_epochs": 100,
    "patience":   15,
    "threshold":  0.5,
    "seed":       42,
}
```

### Classification

```python
# Always pass name= explicitly — the auto-generated default diverges from
# short-form names expected by notebooks.
EXPERIMENT = get_experiment(
    "cls01_resnet50",
    dataset="figshare",
    split_scheme="image_level",
    fold=1,
    name="cls01_resnet50",
)
# or manually:
EXPERIMENT = {
    "name":         "cls01_resnet50",
    "task":         "classification",
    "dataset":      "figshare",
    "split_scheme": "image_level",

    "fold": 1,
    "patch_size":   224,
    "padding_frac": 0.10,
    "batch_size":   32,
    "num_workers":  0,

    "augmentation_strength": "light",

    "model_name":  "resnet50",
    "pretrained":  True,
    "num_classes": 3,

    "loss_name":   "cross_entropy_smooth",
    "loss_kwargs": {"label_smoothing": 0.1, "class_weights": None},

    "optimizer_name":   "adamw",
    "optimizer_kwargs": {"lr": 1e-4, "weight_decay": 1e-4},
    "scheduler_name":   "cosine",
    "scheduler_kwargs": {"T_max": 50, "eta_min": 1e-6},

    "monitor":      "val_macro_f1",
    "monitor_mode": "max",

    "max_epochs": 50,
    "patience":   10,
    "seed":       42,
}
```

### Cell 3 sanity check (every training notebook)

```python
assert EXPERIMENT["task"] == "segmentation"      # or "classification" in NB07
assert EXPERIMENT["dataset"] in ("figshare", "brats2020")
```

---

## 7. Classification testing — the two evaluation variants

A trained cls experiment is tested twice against the same test set:

| variant | `mask_source` | mask source | output dir |
|---|---|---|---|
| **Eval A** | `"gt"` | ground-truth masks from `data/<dataset>/processed/masks/` | `outputs/tables/classification/<dataset>/<cls_exp>/eval_gt/` |
| **Eval B** | `"predicted"` | predicted masks from a chosen seg experiment, fold-aligned | `outputs/tables/classification/<dataset>/<cls_exp>/eval_pred__<seg_exp>/` |

Both evals share the same trained checkpoint, the same fold CSVs, the same labels, and the same patch extraction logic. The only thing that differs is the mask source.

The headline metric is the **gap (Eval A macro F1 − Eval B macro F1)**, mean ± std across the 5 folds.

### Fold alignment guarantee

Because cls and seg use the same fold CSVs (same `split_scheme`, same seed, same generation code), cls fold k's test set equals seg fold k's test set. So for cls fold k's test images, predicted masks live at `outputs/predictions/segmentation/<dataset>/<seg_exp>/fold_k/<image_id>.png` — no out-of-fold lookup needed.

---

## 8. Prediction manifests — the contract between seg and cls

Every seg experiment writes two manifest levels alongside its prediction PNGs.

### `outputs/predictions/segmentation/<dataset>/<seg_exp>/fold_X/manifest.json`

```json
{
  "seg_experiment_name":  "07_unetpp_effb4_dicebce_image_level",
  "task":                 "segmentation",
  "dataset":              "figshare",
  "split_scheme":         "image_level",
  "fold":                 3,
  "checkpoint_path":      "outputs/checkpoints/segmentation/figshare/07_.../fold_3/best_model.pt",
  "checkpoint_sha256":    "abc123…",
  "test_csv_path":        "data/figshare/splits/image_level/folds/fold_3_test.csv",
  "test_csv_sha256":      "def456…",
  "n_predictions":        612,
  "image_size":           256,
  "threshold":            0.5,
  "mask_format":          "binary_uint8_0_255",
  "model_name":           "smp_unetpp_efficientnetb4",
  "encoder_weights":      "imagenet",
  "generated_at":         "2026-05-08T10:00:00Z"
}
```

### `prediction_manifest.json` (top-level for the seg experiment)

```json
{
  "seg_experiment_name":     "07_unetpp_effb4_dicebce_image_level",
  "dataset":                 "figshare",
  "folds_present":           [1, 2, 3, 4, 5],
  "total_predictions":       3064,
  "all_test_images_covered": true
}
```

### Why hashes matter

When cls Eval B starts, it calls `verify_seg_predictions_match()` for the requested fold. The function recomputes the checkpoint and test-CSV hashes and compares them to the manifest. If they don't match, Eval B aborts with a clear error.

This catches the failure mode where seg gets retrained after Eval B has already been run, and the predictions on disk silently go stale. Without manifests, you'd evaluate against the old masks for weeks before noticing.

---

## 9. Data pipeline — training-time flow (segmentation)

```
data/figshare/splits/image_level/folds/fold_X_train.csv
                       (image_path, mask_path, patient_id, tumor_class, ...)
↓
BrainTumorDataset.__getitem__:
  cv2.imread(image_path, GRAYSCALE)
  cv2.cvtColor(GRAY2RGB)              → 3-channel for ImageNet-pretrained encoder
  mask = (mask > 127).astype(np.uint8)
↓
Albumentations Compose:
  [optional CLAHE on full-res]
  Resize(256, 256)
  [random aug, train only]
  Normalize(IMAGENET_MEAN, IMAGENET_STD)
  ToTensorV2()
↓
Returns (image: float32 [3,256,256], mask: float32 [1,256,256])
↓
smp.Unet(encoder=resnet34, weights=imagenet, classes=1, activation=None)
↓
raw logits [N,1,256,256]
↓
Loss: dice_bce / dice / bce / dice_focal / lovasz / etc.
↓
Adam, lr=1e-4 + ReduceLROnPlateau(factor=0.1, patience=5, monitor=val_loss)
↓
EarlyStopping on val_dice (micro), patience=15
ModelCheckpoint best on val_dice
  → outputs/checkpoints/segmentation/figshare/<exp>/fold_X/best.ckpt
```

---

## 10. Data pipeline — training-time flow (classification)

```
data/figshare/splits/image_level/folds/fold_X_train.csv
↓
BrainTumorClsDataset.__getitem__:
  cv2.imread(image_path, GRAYSCALE)  →  cv2.cvtColor(GRAY2RGB)
  mask = read GT mask (training always uses GT)
  patch = extract_patch(image, mask,
                        target_size=224,
                        padding_frac=0.10)
  → bbox of tumor + padding, made square, clamped, resized
  Note: clamping can produce a slightly non-square crop near image edges;
        cv2.resize corrects this with minor aspect-ratio distortion.
↓
Albumentations Compose (light aug):
  HorizontalFlip / VerticalFlip / RandomBrightnessContrast
  Normalize(IMAGENET_MEAN, IMAGENET_STD)
  ToTensorV2()
↓
Returns (patch: float32 [3,224,224], label: long [3-class index])
↓
timm model (resnet50, etc.), num_classes=3
↓
logits [N, 3]
↓
Loss: cross_entropy_smooth (label_smoothing=0.1) / focal_ce / etc.
↓
AdamW + Cosine schedule
↓
EarlyStopping on val_macro_f1, patience=10
ModelCheckpoint best on val_macro_f1
  → outputs/checkpoints/classification/figshare/<exp>/fold_X/best.ckpt
```

---

## 11. Dev/Run workflow (GitHub + Drive + Colab + VS Code)

The project lives in three places, each holding a different kind of artifact:

| location | contents | role |
|---|---|---|
| **GitHub** (`<user>/senior_project`) | `src/`, `configs/`, `notebooks/`, `README.md`, `.gitignore`, `requirements.txt` | single source of truth for code |
| **Google Drive** (`MyDrive/Senior_Project/`) | `data/<dataset>/`, `outputs/`, `_backups/` | canonical data + final outputs (large, regenerable, not version-controlled) |
| **Colab runtime** (`/content/`) | clone of repo + working copy of data + in-progress outputs | ephemeral, fast SSD, wiped on disconnect |

VS Code on the desktop is the editor and Git client; it never runs notebooks. Colab is the only place notebooks execute.

### Storage roles during a run

```
/content/drive/MyDrive/Senior_Project/    ← Drive: canonical, persists
    data/<dataset>/...
    outputs/<cat>/<task>/<dataset>/<exp>/...

/content/senior_project/                  ← GitHub clone: code only
    src/, configs/, notebooks/

/content/Senior_Project_local/            ← local SSD scratch: training writes here
    data/<dataset>/                       (copied from Drive at start)
    outputs/<cat>/<task>/<dataset>/<exp>/ (synced back to Drive at end)
```

`PROJECT_ROOT` during a training run is `/content/Senior_Project_local/`. `src/` is on `sys.path` from `/content/senior_project/` (the cloned repo), not from the local working dir.

### Why writes never go to Drive directly

Long training runs over Drive FUSE hang. `notebook_setup.py` consolidates the workaround:

- **`setup_environment(repo_url, project_folder_name="Senior_Project")`** — mounts Drive, clones the repo to `/content/senior_project/` (or `git pull --ff-only` if already present), adds it to `sys.path`. Returns `(DRIVE_ROOT, REPO_ROOT)`.
- **`copy_to_local(drive_root, datasets=["figshare"])`** — copies `data/<dataset>/` from Drive to `/content/Senior_Project_local/data/`, creates an empty local `outputs/`, chdirs there. Returns `LOCAL_ROOT`. The zip-extraction path tries `unzip` first (faster) and falls back to Python `zipfile`.
- **`sync_outputs_to_drive(drive_root, local_root, task, dataset, experiment_name, categories)`** — at the end of the run, copies `outputs/<cat>/<task>/<dataset>/<exp>/` from local SSD back to Drive in one batched copytree per category.

Every training/testing notebook starts with the first two calls and ends with the sync. NB01 (data prep) and NB02 (split generation) are the exception: they write small artifacts directly to Drive since they're not in the heavy-write training hot path.

### Where each artifact actually lives during a run

| during run | after sync |
|---|---|
| `/content/Senior_Project_local/outputs/checkpoints/<task>/<ds>/<exp>/fold_k/best.ckpt` | `Drive/.../outputs/checkpoints/.../best.ckpt` |
| `/content/Senior_Project_local/outputs/logs/.../metrics.csv` | `Drive/.../outputs/logs/.../metrics.csv` |
| `/content/Senior_Project_local/outputs/predictions/...` | `Drive/.../outputs/predictions/...` |
| `/content/Senior_Project_local/outputs/tables/...` | `Drive/.../outputs/tables/...` |

When the runtime disconnects, the local copy evaporates; the Drive copy persists.

### Daily loop

| edit | where |
|---|---|
| `src/*.py`, `configs/*.py` | desktop in VS Code → commit → push. Re-run cell 2 in Colab to `git pull`. **Never** edit `.py` inside Colab; those edits die with the runtime. |
| Notebook structural edits (cells, code logic, markdown) | desktop in VS Code → commit → push. Reload the Colab tab. |
| Notebook runtime tweaks (e.g. `EXPERIMENT["fold"]`, `EXPERIMENT["loss_name"]`) | inside the open Colab tab. **Do not** push these back to GitHub — the canonical notebook stays clean; the variant is captured by `experiment_config.json` on Drive. |

### Opening a notebook on Colab

Direct GitHub→Colab URL pattern (bookmark or put in README badges):

```
https://colab.research.google.com/github/<user>/senior_project/blob/main/notebooks/<folder>/<notebook>.ipynb
```

Run procedure: open the URL → Connect to GPU → run cell 1 (`pip install`) → run cell 2 (bootstrap: mount Drive, clone/pull repo, copy data to local) → review/edit cell 3 (`EXPERIMENT` dict) → run the rest → confirm the final sync cell ran. Close the tab.

### `.gitignore` essentials

```
data/
outputs/
_backups/
__pycache__/
.ipynb_checkpoints/
.venv/
*.ckpt
*.pt
```

Recommended: `pip install nbstripout && nbstripout --install` in the local repo so notebook outputs don't bloat Git history.

---

## 12. Notebooks — responsibilities

| folder | notebook | role |
|---|---|---|
| `setups/` | `01_data_preparation_figshare.ipynb` | figshare: `.mat` → PNG, build `metadata.csv`, `preprocessing_config.json` |
| `setups/` | `01_data_preparation_brats2020.ipynb` | brats2020: `.h5` → PNG (FLAIR + whole-tumor mask), same output layout |
| `setups/` | `02_split.ipynb` | per-dataset, per-scheme: build fold CSVs. Knobs: `DATASET`, `SPLIT_SCHEME` |
| `segmentation/` | `03_train.ipynb` | train one fold or all 5, per `EXPERIMENT` dict |
| `segmentation/` | `04_data_vis.ipynb` | qualitative figures: best/worst/random predictions, per-class Dice bar chart, training curves |
| `segmentation/` | `05_test.ipynb` | quantitative test eval, writes prediction PNGs + per-fold manifests + all 6 CV tables |
| `segmentation/` | `06_seg_compare.ipynb` | cross-experiment comparison: overlay `cv_summary.csv` tables, rank by Dice |
| `classification/` | `06_classification_data.ipynb` | QA: visualize patches with both mask sources, sanity-check before training |
| `classification/` | `07_train_cls.ipynb` | train one fold or all 5, GT masks only |
| `classification/` | `08_test_cls.ipynb` | run Eval A (`mask_source="gt"`) and Eval B (`mask_source="predicted"`) separately, compute and print gap |
| `classification/` | `09_cls_results.ipynb` | load saved CSVs from Drive, produce publication-ready figures: confusion matrix pair, per-class F1 bars, Eval A vs B gap chart, sample-patch grid |

Switching experiments only requires editing the `EXPERIMENT` dict (or `get_experiment()` call) in cell 3 of the relevant training notebook. `src/` is never edited between experiments — only when adding a new model/loss/etc. variant.

---

## 13. Reference experiments

### Segmentation — 7-experiment FigShare reference reproduction

| exp name | loss / arch | reference Dice |
|---|---|---|
| `01_dice_image_level` | Dice | 0.8426 |
| `02_bce_image_level` | BCE | 0.8485 |
| `03_dicebce_image_level` | Dice + BCE | 0.8541 |
| `04_dicefocal_image_level` | Dice + Focal | 0.8509 |
| `05_lovasz_image_level` | Lovasz | 0.8418 |
| `06_clahe_dicebce_image_level` | CLAHE + Dice + BCE | 0.8501 |
| `07_unetpp_effb4_dicebce_image_level` | UNet++ + EfficientNet-B4 + Dice + BCE | 0.8608 (best) |

All use: 5-fold image-level KFold, batch 8 (6 for exp 07), max_epochs=100, patience=15, Adam lr=1e-4, ReduceLROnPlateau factor=0.1 patience=5, "reference" augmentation, ImageNet normalization, threshold=0.5, seed=42.

### Classification — figshare recipes

| recipe | model | notes |
|---|---|---|
| `cls01_resnet50` | ResNet-50 | baseline; Eval B uses `07_unetpp_effb4_dicebce_image_level` |
| `cls02_effb0` | EfficientNet-B0 | |
| `cls03_effb4` | EfficientNet-B4 | batch_size=16 |
| `cls04_vit` | ViT-Small/16 | batch_size=16 |

All use: AdamW lr=1e-4 wd=1e-4, cosine schedule T_max=50, label-smoothing CE (0.1), max_epochs=50, patience=10, seed=42.

---

## 14. Reproducibility

A run is fully specified by:

1. `data/<dataset>/splits/<scheme>/cv_split_config.json` — which images are in which fold
2. `outputs/checkpoints/<task>/<dataset>/<exp>/fold_X/experiment_config.json` — full EXPERIMENT dict for that run
3. (For seg → cls Eval B only) `outputs/predictions/segmentation/<dataset>/<seg_exp>/fold_X/manifest.json` — which checkpoint produced which masks

Headline tables:
- **seg**: `outputs/tables/segmentation/<dataset>/<exp>/cv_summary.csv` — macro/micro Dice mean ± std across folds
- **cls Eval A**: `outputs/tables/classification/<dataset>/<exp>/eval_gt/cv_summary.csv` — macro F1 mean ± std
- **cls Eval B**: `outputs/tables/classification/<dataset>/<exp>/eval_pred__<seg_exp>/cv_summary.csv` — same metrics, predicted masks
- **gap report**: written by NB08 final cell, comparing Eval A vs Eval B

---

## 15. Working agreement (maintenance)

Both phases are complete. All 21 `src/` files exist and are smoke-tested. Active work is finalising the report and running experiments.

Rules that carry forward:

1. Add new model/loss/aug variants as new registry branches — never modify existing ones.
2. When editing an existing `src/` file: syntax-check locally (`python3 -m py_compile src/<file>.py`), push to GitHub, smoke-test in Colab before using in a run.
3. `PathLike` is imported from `file_utils` everywhere — never redefine it locally in a new file.
4. `IMAGENET_MEAN/STD` is imported from `sg_data_utils` in any file that needs it — never redefine locally.
5. `add_mean_std` is imported from `eval_utils` — never add a local `_add_mean_std` copy.
6. Notebook structural changes go through VS Code → GitHub → `git pull` in Colab. Runtime-only tweaks (EXPERIMENT dict knobs) stay in the Colab tab only.
7. Update this file whenever a dataset is added, a module changes its public API, or a path convention changes.

---

This is the living instruction for the project. Update it whenever a structural decision changes. Keep this file in the GitHub repo so future-you doesn't have to reconstruct the design from notebook cells.
