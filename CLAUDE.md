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
| `brats2024` | planned | TBD | TBD | TBD |

Adding a new dataset means: dropping its raw files into `data/<dataset>/raw/`, running NB01 with `DATASET = "<dataset>"`, running NB02 to produce fold CSVs, and changing one knob in any training/testing notebook. No source-code edits required.

---

## 3. Folder structure

```
Senior_Project/
├── _backups/
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
│   └── brats2024/  (same layout when added)
├── outputs/
│   ├── checkpoints/<task>/<dataset>/<exp>/fold_X/{best.ckpt, best_model.pt, experiment_config.json}
│   ├── logs/<task>/<dataset>/<exp>/fold_X/lightning_logs/version_0/metrics.csv
│   ├── predictions/
│   │   ├── segmentation/<dataset>/<seg_exp>/
│   │   │   ├── prediction_manifest.json
│   │   │   └── fold_X/
│   │   │       ├── manifest.json
│   │   │       └── <image_id>.png
│   │   └── classification/<dataset>/<cls_exp>/<eval_variant>/fold_X/<image_id>.json
│   ├── tables/<task>/<dataset>/<exp>/
│   │   ├── (segmentation)  cv_results.csv, cv_summary.csv, cv_by_class.csv,
│   │   │                   cv_class_summary.csv, cv_per_image.csv,
│   │   │                   fold_X_*.csv, test_eval_config.json
│   │   └── (classification)
│   │       ├── eval_gt/  {cv_results, cv_summary, cv_confusion, fold_X_*}.csv
│   │       └── eval_pred__<seg_exp>/  {cv_results, cv_summary, cv_confusion, fold_X_*}.csv
│   ├── figures/
│   │   ├── data_preparation/<dataset>/
│   │   ├── segmentation/<dataset>/<seg_exp>/fold_X/sample_predictions/*.png
│   │   └── classification/<dataset>/<cls_exp>/<eval_variant>/fold_X/{confusion_matrix.png, sample_patches/*.png}
│   └── reports/
├── notebooks/
│   ├── setups/
│   │   ├── 00_migration.ipynb
│   │   ├── 01_data_preparation.ipynb
│   │   └── 02_split.ipynb
│   ├── segmentation/
│   │   ├── 03_train.ipynb
│   │   ├── 04_results.ipynb
│   │   └── 05_test.ipynb
│   └── classification/
│       ├── 06_classification_data.ipynb
│       ├── 07_train_cls.ipynb
│       └── 08_test_cls.ipynb
└── src/  (see §4)
```

This is the canonical layout as it lives in Google Drive. See §11 for how it maps to the Colab runtime and the GitHub repo.

### Path conventions

Three orthogonal axes define every artifact:

- **task** — `segmentation` or `classification`. First segment under `outputs/<category>/`.
- **dataset** — `figshare`, `brats2024`, etc. Second segment.
- **experiment_name** — config-only identifier, e.g. `01_dice_image_level` or `cls01_resnet50`. No dataset suffix.

Glob queries become natural:

- All seg experiments on figshare → `outputs/checkpoints/segmentation/figshare/*/`
- Same experiment across all datasets → `outputs/*/*/figshare/01_dice_image_level/`

---

## 4. src/ — module layout and naming convention

**Naming rule:** files used by both tasks have no prefix. Files used only by one task are prefixed `sg_` (segmentation) or `cls_` (classification).

| order | file | scope | responsibility |
|---|---|---|---|
| 1 | `file_utils.py` | shared | `project_dirs`, `dataset_paths`, `experiment_paths` (task+dataset aware), `split_scheme_dir`, `fold_split_csv_paths`, `seg_predictions_dir`, `load_seg_prediction_manifest`, `verify_seg_predictions_match`, `cls_eval_paths`, JSON helpers |
| 2 | `notebook_setup.py` | shared | `setup_environment` (mounts Drive + `git clone`/`git pull` repo from GitHub), `copy_to_local`, `sync_outputs_to_drive` — replaces the duplicated env/copy/sync cells |
| 3 | `preprocess_utils.py` | shared | `discover_mat_files`, `convert_figshare_mat_to_png_record` — dataset prep produces both images and masks |
| 4a | `vis_utils.py` | shared | `show_class_examples` and generic image-display helpers used by NB01 |
| 4b | `sg_vis_utils.py` | seg | `show_triplet`, `show_overlay_triplet`, `show_image_gt_pred_overlay` (4-panel) |
| 5a | `data_utils.py` | shared | `load_metadata`, `validate_metadata`, `metadata_summary`, `create_patient_folds`, `create_image_level_folds`, `make_train_val_from_pool`, `make_train_val_image_level`, `verify_no_patient_leakage`, `save_fold_csvs` |
| 5b | `sg_data_utils.py` | seg | `BrainTumorDataset`, `build_train_transform`, `build_eval_transform`, `build_dataloaders` |
| 6a | `eval_utils.py` | shared | `build_fold_summary` |
| 6b | `sg_eval_utils.py` | seg | `summarize_fold_results`, `aggregate_cv_results`, `micro_dice_from_counts` |
| 7 | `sg_metrics.py` | seg | `dice_score`, `iou_score`, `get_smp_stats`, `get_metric_kind_pairs` |
| 8 | `sg_losses.py` | seg | bce, dice, focal, lovasz, dice_bce, dice_focal, combo |
| 9 | `sg_models.py` | seg | SMP-based registry: `smp_unet_resnet34`, `smp_unetpp_efficientnetb4`, etc. |
| 10 | `optimizers.py` | shared | adam, adamw, sgd, rmsprop / reduce_on_plateau, cosine, cosine_warm_restarts, step, multistep, exponential, none |
| 11 | `train_utils.py` | shared | `set_global_seed`, `build_callbacks`, `build_trainer`, `export_plain_state_dict` |
| 12 | `sg_lightning_module.py` | seg | `BrainTumorSegModule` |
| 13 | `sg_test_utils.py` | seg | `load_model_from_pt`, `predict_mask`, `evaluate_fold` (writes prediction manifests) |
| 14 | `cls_data_utils.py` | cls | `extract_patch`, `BrainTumorClsDataset`, cls transforms, `build_dataloaders_cls` |
| 15 | `cls_models.py` | cls | timm-based registry (resnet50, efficientnet_b0/b4, vit_small_patch16_224) |
| 16 | `cls_losses.py` | cls | cross_entropy, cross_entropy_smooth, focal_ce |
| 17 | `cls_metrics.py` | cls | macro F1, accuracy, per-class P/R/F1, confusion matrix |
| 18 | `cls_lightning_module.py` | cls | `BrainTumorClsModule` |
| 19 | `cls_test_utils.py` | cls | `load_cls_model_from_pt`, `predict_class`, `evaluate_fold_cls` (handles both `mask_source` modes) |
| 20 | `cls_eval_utils.py` | cls | macro F1 cross-fold aggregation, `aggregate_cv_confusion` |

23 files total: 7 shared, 9 seg-prefixed, 7 cls-prefixed.

---

## 5. Architecture: registry pattern

All components are selected by string name in the `EXPERIMENT` dict. To add a new model/loss/metric/optimizer/scheduler/aug variant:

1. Add a new `if n == "new_name":` branch in the appropriate registry (`sg_models.py`, `cls_models.py`, `sg_losses.py`, `cls_losses.py`, `optimizers.py`, etc.).
2. Do **not** modify existing branches.
3. Reference the new name in an `EXPERIMENT` dict.
4. Old experiments are unaffected.

---

## 6. EXPERIMENT dict — the single source of truth per run

### Segmentation

```python
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
assert EXPERIMENT["dataset"] in ("figshare", "brats2024")
```

---

## 7. Classification testing — the two evaluation variants

A trained cls experiment is tested twice against the same test set:

| variant | `EVAL_MASK_SOURCE` | mask source | output dir |
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
↓
Albumentations Compose (light aug):
  Resize(224, 224)
  Flip / brightness / rotation (small)
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
| **GitHub** (`<user>/senior_project`) | `src/`, `notebooks/`, `README.md`, `.gitignore`, `requirements.txt` | single source of truth for code |
| **Google Drive** (`MyDrive/Senior_Project/`) | `data/<dataset>/`, `outputs/`, `_backups/` | canonical data + final outputs (large, regenerable, not version-controlled) |
| **Colab runtime** (`/content/`) | clone of repo + working copy of data + in-progress outputs | ephemeral, fast SSD, wiped on disconnect |

VS Code on the desktop is the editor and Git client; it never runs notebooks. Colab is the only place notebooks execute.

### Storage roles during a run

```
/content/drive/MyDrive/Senior_Project/    ← Drive: canonical, persists
    data/<dataset>/...
    outputs/<cat>/<task>/<dataset>/<exp>/...

/content/senior_project/                  ← GitHub clone: code only
    src/, notebooks/

/content/Senior_Project_local/            ← local SSD scratch: training writes here
    data/<dataset>/                       (copied from Drive at start)
    outputs/<cat>/<task>/<dataset>/<exp>/ (synced back to Drive at end)
```

`PROJECT_ROOT` during a training run is `/content/Senior_Project_local/`. `src/` is on `sys.path` from `/content/senior_project/` (the cloned repo), not from the local working dir.

### Why writes never go to Drive directly

Long training runs over Drive FUSE hang. `notebook_setup.py` consolidates the workaround:

- **`setup_environment(repo_url, project_folder_name="Senior_Project")`** — mounts Drive, clones the repo to `/content/senior_project/` (or `git pull --ff-only` if already present), adds it to `sys.path`. Returns `(DRIVE_ROOT, REPO_ROOT)`.
- **`copy_to_local(drive_root, datasets=["figshare"])`** — copies `data/<dataset>/` from Drive to `/content/Senior_Project_local/data/`, creates an empty local `outputs/`, chdirs there. Returns `LOCAL_ROOT`.
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
| `src/*.py` | desktop in VS Code → commit → push. Re-run cell 2 in Colab to `git pull`. **Never** edit `.py` inside Colab; those edits die with the runtime. |
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
| `setups/` | `00_migration.ipynb` | one-time restructure tasks; manifest synthesis for legacy data |
| `setups/` | `01_data_preparation.ipynb` | per-dataset: `.mat` → PNG, build `metadata.csv`, `preprocessing_config.json`. Knob: `DATASET` |
| `setups/` | `02_split.ipynb` | per-dataset, per-scheme: build fold CSVs. Knobs: `DATASET`, `SPLIT_SCHEME` |
| `segmentation/` | `03_train.ipynb` | train one fold or all 5, per `EXPERIMENT` dict |
| `segmentation/` | `04_results.ipynb` | qualitative figures (best/worst/random predictions) |
| `segmentation/` | `05_test.ipynb` | quantitative test eval, writes `predictions/.../manifest.json` |
| `classification/` | `06_classification_data.ipynb` | QA: visualize patches with both mask sources, sanity-check before training |
| `classification/` | `07_train_cls.ipynb` | train one fold or all 5, GT masks only |
| `classification/` | `08_test_cls.ipynb` | run Eval A (`EVAL_MASK_SOURCE="gt"`) and Eval B (`EVAL_MASK_SOURCE="predicted"`) separately, compute gap |

Switching experiments only requires editing the `EXPERIMENT` dict in cell 3 of the relevant training notebook, and `EXPERIMENT_NAME` + `DATASET` in cell 3 of testing/results notebooks. `src/` is never edited between experiments — only when adding a new model/loss/etc. variant.

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

### Classification

`cls01_resnet50` is the first cls experiment. Trained once on figshare with GT masks, evaluated twice (Eval A, Eval B). The Eval-B `seg_experiment` is `07_unetpp_effb4_dicebce_image_level` (the best seg model).

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

## 15. Working agreement (for the rebuild)

The rewrite is happening one file at a time, in chat. Rules:

1. Files are rewritten in dependency order (see §4 ordering column).
2. Every rewrite is a **complete file**, not a diff. You paste the whole thing into the new empty target.
3. Notebooks are rewritten cell-by-cell, every cell included even if textually unchanged from the original.
4. A notebook isn't touched until every src file it imports has been rewritten.
5. After each src file is rewritten, push it to GitHub and smoke-test it in Colab (`from src.<module> import *`) before requesting the next.

The phased rebuild plan from earlier is the macro-level schedule; this working agreement governs the per-file mechanics.

---

This is the living instruction for the project. Update it whenever a structural decision changes (e.g., when `brats2024` is added, when a new task is introduced, when prediction manifests gain new fields). Keep it as `README.md` in the GitHub repo so future-you doesn't have to reconstruct the design from notebook cells.